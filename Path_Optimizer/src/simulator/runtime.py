"""Scenario execution, pacing, and logging for the flight-only simulator."""

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from core.logging import Logger
from core.modes import ControlMode, FlightMode, ModeSelection, SystemMode, validate_modes
from simulator.scenario import Scenario, modes_from_event
from simulator.telemetry import TelemetrySimulator


class SimulatorRuntime:
    def __init__(
        self,
        scenario: Scenario,
        *,
        telemetry_hz: float = 100.0,
        physics_hz: float = 100.0,
        heartbeat_hz: float = 2.0,
        realtime: bool = True,
        mavlink_server: Any = None,
    ):
        self.scenario = scenario
        self.realtime = realtime
        self.simulator = TelemetrySimulator(
            telemetry_hz=telemetry_hz,
            physics_hz=physics_hz,
            heartbeat_hz=heartbeat_hz,
            initial_state=scenario.initial_state,
            noise_seed=scenario.seed,
            sensor_noise=scenario.telemetry.get("sensor_noise"),
            telemetry_latency_s=float(scenario.telemetry.get("latency_s", 0.0)),
            telemetry_jitter_s=float(scenario.telemetry.get("jitter_s", 0.0)),
            packet_drop_rate=float(scenario.telemetry.get("packet_drop_rate", 0.0)),
        )
        self.simulator.model.cd *= self._air_density() / 1.225
        self.modes = scenario.initial_modes
        self.reference: dict[str, Any] = {}
        self.force_std_n = np.zeros(3)
        self.moment_std_nm = np.zeros(3)
        self.rng = np.random.default_rng(scenario.seed)
        self.logger = Logger(metadata=self._metadata())
        self.event_index = 0
        self.collisions: list[dict[str, Any]] = []
        self.mavlink_server = mavlink_server
        self.outage_started_s: float | None = None
        self.outage_until_s: float | None = None
        self.command_timeout_s = 0.5
        self.sim_time_s = 0.0

    def run(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path) if output_path else self.timestamped_run_path()
        period_s = 1.0 / self.simulator.telemetry_hz
        cycles = int(math.floor(self.scenario.duration_s * self.simulator.telemetry_hz)) + 1
        started = time.perf_counter()
        self._log_event("initialized", 0)
        for cycle in range(cycles):
            sim_time_s = cycle * period_s
            self.sim_time_s = sim_time_s
            if self.realtime:
                remaining = started + sim_time_s - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
            self._apply_events(sim_time_s)
            if self.mavlink_server:
                self.mavlink_server.poll()
            disturbance = self._disturbance()
            self.simulator.set_disturbance(*disturbance)
            self._apply_reference()
            sample = self.simulator.step()
            if self.mavlink_server and self.simulator.should_emit_telemetry():
                heartbeat_interval = max(1, round(self.simulator.telemetry_hz / self.simulator.heartbeat_hz))
                self.mavlink_server.emit(sample, heartbeat=cycle % heartbeat_interval == 0)
            self.logger.log_cycle(
                cycle=cycle,
                sim_time_ns=sample.sim_time_ns,
                wall_elapsed_ms=(time.perf_counter() - started) * 1000.0,
                modes=self._mode_snapshot(),
                telemetry=sample,
                reference=self.reference,
                simulator_truth=self._truth(),
                disturbance={"force_i_n": disturbance[0], "moment_b_nm": disturbance[1]},
                collisions=self.collisions,
            )
        self._log_event("shutdown", round(self.scenario.duration_s * 1e9))
        self.logger.save_run(output)
        return output

    def _apply_events(self, sim_time_s: float) -> None:
        while self.event_index < len(self.scenario.events):
            event = self.scenario.events[self.event_index]
            if float(event["at_s"]) > sim_time_s + 1e-12:
                return
            self.event_index += 1
            self._apply_event(event, round(sim_time_s * 1e9))

    def _apply_event(self, event: dict[str, Any], sim_time_ns: int) -> None:
        event_type = event["type"]
        if event_type == "set_modes":
            try:
                modes = modes_from_event(event)
                validate_modes(modes, flight_only=True)
            except ValueError as error:
                self.modes = ModeSelection(SystemMode.FAULT)
                self.simulator.disarm()
                self._log_event("fault", sim_time_ns, reason=str(error))
                return
            self.modes = modes
            if modes.system in (SystemMode.ARMED, SystemMode.RACING):
                self.simulator.arm()
            else:
                self.simulator.disarm()
            self._log_event("modes_changed", sim_time_ns)
        elif event_type == "reference":
            self.reference = {key: value for key, value in event.items() if key not in ("at_s", "type")}
            self._log_event("reference_changed", sim_time_ns)
        elif event_type == "turbulence":
            self.force_std_n = np.asarray(event.get("force_std_n", [0.0, 0.0, 0.0]), dtype=float)
            self.moment_std_nm = np.asarray(event.get("moment_std_nm", [0.0, 0.0, 0.0]), dtype=float)
            self._log_event("turbulence_changed", sim_time_ns)
        elif event_type == "reset":
            self.simulator.reset()
            self._log_event("reset", sim_time_ns)
        elif event_type == "collision":
            collision = {"collision_id": int(event.get("collision_id", 1002)), "impact_kg_mps": float(event.get("impact_kg_mps", 0.0))}
            self.collisions.append(collision)
            if self.mavlink_server:
                self.mavlink_server.emit_collision(**collision)
            self._log_event("collision", sim_time_ns)
        elif event_type == "command_outage":
            self.outage_started_s = float(event["at_s"])
            self.outage_until_s = self.outage_started_s + float(event.get("duration_s", self.command_timeout_s))
            self._log_event("command_outage", sim_time_ns)
        else:
            raise ValueError(f"unsupported scenario event type: {event_type}")

    def _apply_reference(self) -> None:
        if self.modes.system not in (SystemMode.ARMED, SystemMode.RACING):
            self.simulator.disarm()
            return
        if self.outage_until_s is not None and self.sim_time_s < self.outage_until_s:
            if self.outage_started_s is not None and self.sim_time_s - self.outage_started_s >= self.command_timeout_s:
                self.modes = ModeSelection(SystemMode.FAULT)
                self.simulator.disarm()
                self._log_event("fault", round(self.sim_time_s * 1e9), reason="command timeout")
            return
        control = self.modes.control
        if control in (ControlMode.POSITION_HOLD, ControlMode.VELOCITY, ControlMode.WAYPOINT_FOLLOW, ControlMode.TRAJECTORY_TRACK, ControlMode.MPCC_TRACKER):
            self.simulator.apply_position_target(self._position_target())
        elif control in (ControlMode.ATTITUDE, ControlMode.RATE_DIRECT):
            self.simulator.apply_attitude_target({
                "quaternion": self.reference.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
                "body_rates_rps": self.reference.get("body_rates_rps", [0.0, 0.0, 0.0]),
                "thrust": float(self.reference.get("thrust", 0.5)),
            })

    def _position_target(self) -> dict[str, Any]:
        target = {
            "position_local_ned_m": self.reference.get("position_ned_m"),
            "velocity_local_ned_mps": self.reference.get("velocity_ned_mps", [0.0, 0.0, 0.0]),
            "yaw_rad": self.reference.get("yaw_rad"),
        }
        trajectory = self.reference.get("trajectory")
        if trajectory:
            times = np.asarray([point["t_s"] for point in trajectory], dtype=float)
            target["position_local_ned_m"] = [
                float(np.interp(self.sim_time_s, times, [point["position_ned_m"][axis] for point in trajectory]))
                for axis in range(3)
            ]
            target["velocity_local_ned_mps"] = [
                float(np.interp(self.sim_time_s, times, [point.get("velocity_ned_mps", [0.0, 0.0, 0.0])[axis] for point in trajectory]))
                for axis in range(3)
            ]
        waypoints = self.reference.get("waypoints")
        if waypoints:
            period = float(self.reference.get("waypoint_period_s", 1.0))
            index = min(int(self.sim_time_s / period), len(waypoints) - 1)
            target["position_local_ned_m"] = waypoints[index]
        return target

    def _disturbance(self) -> tuple[np.ndarray, np.ndarray]:
        return self.rng.normal(0.0, self.force_std_n), self.rng.normal(0.0, self.moment_std_nm)

    def _truth(self) -> dict[str, Any]:
        state = self.simulator._harness.state
        return {
            "state_vector": state,
            "position_local_ned_m": state[0:3],
            "velocity_local_ned_mps": state[3:6],
            "attitude_quaternion": state[6:10],
            "body_rates_rps": state[10:13],
            "motor_commands": self.simulator._motor_command,
        }

    def _metadata(self) -> dict[str, Any]:
        temperature_c = self.scenario.environment.get("temperature_c", 15.0)
        pressure_pa = self.scenario.environment.get("pressure_pa", 101325.0)
        return {
            "scenario": self.scenario.name,
            "duration_s": self.scenario.duration_s,
            "seed": self.scenario.seed,
            "environment": {"temperature_c": temperature_c, "pressure_pa": pressure_pa, "air_density_kg_m3": self._air_density()},
            "telemetry_boundary": self.scenario.telemetry,
        }

    def _air_density(self) -> float:
        temperature_c = self.scenario.environment.get("temperature_c", 15.0)
        pressure_pa = self.scenario.environment.get("pressure_pa", 101325.0)
        return pressure_pa / (287.05 * (temperature_c + 273.15))

    def _mode_snapshot(self) -> dict[str, str | None]:
        return {key: value.value if value else None for key, value in asdict(self.modes).items()}

    def _log_event(self, event: str, sim_time_ns: int, **data: Any) -> None:
        self.logger.log_event(event, sim_time_ns=sim_time_ns, modes=self._mode_snapshot(), **data)

    @staticmethod
    def timestamped_run_path(root: str | Path | None = None) -> Path:
        logs_dir = Path(root) if root else Path(__file__).resolve().parents[2] / "logs" / "sim"
        return Logger.timestamped_dir(logs_dir) / "run.json"
