from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.control.forward_velocity import ForwardVelocityAltitudeController
from core.control.hover import HoverPIDController
from sensing.telemetry import MavlinkBridge

from .config import Q1RuntimeConfig


@dataclass(frozen=True)
class AttitudeFeedbackProbeConfig:
    endpoint: str = Q1RuntimeConfig.mavlink_endpoint
    control_hz: float = 30.0
    run_s: float = 3.0
    start_delay_s: float = 0.5
    target_velocity_local_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_altitude_offset_ned_m: float = 0.0
    neutral_thrust: float = 0.495
    use_hover_trim: bool = True
    max_tilt_rad: float = 0.08
    max_tilt_rate_rps: float = 0.15
    max_horizontal_accel_mps2: float = 0.5
    max_vertical_accel_mps2: float = 4.0
    arm_on_start: bool = True
    disarm_on_exit: bool = True
    reset_on_start: bool = False
    reset_wait_s: float = 2.0
    reset_ready_timeout_s: float = 20.0
    reset_stable_s: float = 2.0
    reset_stable_max_speed_mps: float = 0.05
    post_reset_delay_s: float = 2.0
    precase_clearance: bool = False
    precase_clearance_motor_value: float = 0.34
    precase_clearance_s: float = 0.16
    max_speed_mps: float = 20.0
    min_z_ned_m: float = -5000.0
    max_z_ned_m: float = 5000.0
    safety_enabled: bool = True
    progress: bool = False
    output_path: Path | None = Path("logs/q1runtime/attitude-feedback-probe.json")


class AttitudeFeedbackProbe:
    """Closed-loop SET_ATTITUDE_TARGET probe using the existing feedback controller."""

    def __init__(self, bridge: Any, config: AttitudeFeedbackProbeConfig | None = None):
        self.bridge = bridge
        self.config = config or AttitudeFeedbackProbeConfig()
        self.hover = HoverPIDController(dt_s=1.0 / max(1e-6, self.config.control_hz), neutral_thrust=self.config.neutral_thrust)
        self.controller = ForwardVelocityAltitudeController(
            dt_s=1.0 / max(1e-6, self.config.control_hz),
            neutral_thrust=self.config.neutral_thrust,
            max_tilt_rad=self.config.max_tilt_rad,
            max_tilt_rate_rps=self.config.max_tilt_rate_rps,
            max_horizontal_accel_mps2=self.config.max_horizontal_accel_mps2,
            max_vertical_accel_mps2=self.config.max_vertical_accel_mps2,
        )

    def run_live(self) -> dict[str, Any]:
        self.bridge.connect()
        self.bridge.start_heartbeat()
        self.bridge.subscribe_telemetry()
        self._wait_for_telemetry()
        if self.config.reset_on_start:
            self._reset_simulator()
        if self.config.arm_on_start:
            self.bridge.arm()
            time.sleep(0.2)
        if self.config.precase_clearance:
            self._run_precase_clearance()
        if self.config.start_delay_s > 0.0:
            self._progress(f"waiting start_delay_s={self.config.start_delay_s:g} before feedback control")
            time.sleep(self.config.start_delay_s)

        start_telemetry = self._require_telemetry()
        target_altitude = float(start_telemetry.position_local_ned_m[2]) + float(self.config.target_altitude_offset_ned_m)
        start_reset_count = getattr(start_telemetry, "reset_count", None)
        target_velocity = np.asarray(self.config.target_velocity_local_ned_mps, dtype=float)
        self.controller.reset()

        cycles: list[dict[str, Any]] = []
        aborted = False
        abort_reason: str | None = None
        try:
            deadline = time.monotonic() + max(0.0, self.config.run_s)
            next_tick = time.monotonic()
            cycle = 0
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += 1.0 / max(1e-6, self.config.control_hz)
                telemetry = self._require_telemetry()
                safety_reason = self._safety_reason(telemetry, start_reset_count)
                if safety_reason is not None:
                    aborted = True
                    abort_reason = safety_reason
                    self._progress(f"safety abort cycle={cycle}: {safety_reason}")
                    break
                hover_quaternion, hover_thrust = self.hover.update(
                    telemetry.acceleration_local_ned_mps2 or (0.0, 0.0, 0.0),
                    telemetry.attitude,
                )
                command = self.controller.update(
                    position_local_ned_m=telemetry.position_local_ned_m,
                    velocity_local_ned_mps=telemetry.velocity_local_ned_mps,
                    attitude_quaternion=telemetry.attitude,
                    target_altitude_ned_m=target_altitude,
                    target_velocity_local_ned_mps=target_velocity,
                    trim_thrust=hover_thrust if self.config.use_hover_trim else self.config.neutral_thrust,
                )
                self.bridge.send_attitude_target(command)
                cycle_payload = {
                    "cycle": cycle,
                    "sim_time_ns": int(telemetry.sim_time_ns),
                    "position_local_ned_m": [float(value) for value in telemetry.position_local_ned_m],
                    "velocity_local_ned_mps": [float(value) for value in telemetry.velocity_local_ned_mps],
                    "attitude": [float(value) for value in telemetry.attitude],
                    "acceleration_local_ned_mps2": None
                    if telemetry.acceleration_local_ned_mps2 is None
                    else [float(value) for value in telemetry.acceleration_local_ned_mps2],
                    "command": command,
                    "hover_reference": {
                        "quaternion": [float(value) for value in hover_quaternion],
                        "thrust": float(hover_thrust),
                        "used_for_trim": bool(self.config.use_hover_trim),
                    },
                    "actuator_output": self._actuator_output(),
                    "collision_count": len(getattr(self.bridge, "collisions", []) or []),
                }
                cycles.append(cycle_payload)
                if self.config.progress and cycle % max(1, round(self.config.control_hz / 5.0)) == 0:
                    self._progress(
                        f"cycle={cycle} pos={np.round(cycle_payload['position_local_ned_m'], 3).tolist()} "
                        f"vel={np.round(cycle_payload['velocity_local_ned_mps'], 3).tolist()} "
                        f"thrust={float(command['thrust']):.3f} "
                        f"rpy={np.round(command['roll_pitch_yaw_rad'], 3).tolist()}"
                    )
                cycle += 1
        finally:
            self._send_zero_thrust(0.2)
            if self.config.disarm_on_exit:
                try:
                    self.bridge.disarm()
                except Exception:
                    pass
            self.bridge.shutdown()

        payload = self._payload(cycles, target_altitude, target_velocity, aborted, abort_reason)
        if self.config.output_path is not None:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        cycles = payload["cycles"]
        first = cycles[0] if cycles else None
        last = cycles[-1] if cycles else None
        displacement = None
        if first and last:
            displacement = (
                np.asarray(last["position_local_ned_m"], dtype=float)
                - np.asarray(first["position_local_ned_m"], dtype=float)
            ).tolist()
        speeds = [float(np.linalg.norm(np.asarray(cycle["velocity_local_ned_mps"], dtype=float))) for cycle in cycles]
        thrusts = [float(cycle["command"]["thrust"]) for cycle in cycles]
        return {
            "created_utc": payload["created_utc"],
            "output_path": None if self.config.output_path is None else str(self.config.output_path),
            "cycle_count": len(cycles),
            "aborted": payload["aborted"],
            "abort_reason": payload["abort_reason"],
            "target_altitude_ned_m": payload["target_altitude_ned_m"],
            "target_velocity_local_ned_mps": payload["target_velocity_local_ned_mps"],
            "start_position_local_ned_m": None if first is None else first["position_local_ned_m"],
            "end_position_local_ned_m": None if last is None else last["position_local_ned_m"],
            "displacement_local_ned_m": displacement,
            "max_speed_mps": None if not speeds else max(speeds),
            "min_thrust": None if not thrusts else min(thrusts),
            "max_thrust": None if not thrusts else max(thrusts),
            "collision_count": len(payload.get("bridge_snapshot", {}).get("collisions", []) or [])
            if payload.get("bridge_snapshot")
            else None,
        }

    def _payload(
        self,
        cycles: list[dict[str, Any]],
        target_altitude: float,
        target_velocity: np.ndarray,
        aborted: bool,
        abort_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "target_altitude_ned_m": float(target_altitude),
            "target_velocity_local_ned_mps": [float(value) for value in target_velocity],
            "aborted": aborted,
            "abort_reason": abort_reason,
            "cycles": cycles,
            "bridge_snapshot": self._bridge_snapshot(),
        }

    def _send_zero_thrust(self, duration_s: float) -> None:
        telemetry = self.bridge.get_latest_telemetry()
        quaternion = (1.0, 0.0, 0.0, 0.0) if telemetry is None else tuple(float(value) for value in telemetry.attitude)
        payload = {"quaternion": list(quaternion), "thrust": 0.0, "attitude_type_mask": 7, "source": "q1runtime_feedback_stop"}
        deadline = time.monotonic() + max(0.0, duration_s)
        interval = 1.0 / max(1e-6, self.config.control_hz)
        while time.monotonic() < deadline:
            self.bridge.send_attitude_target(payload)
            time.sleep(interval)

    def _run_precase_clearance(self) -> None:
        command = [float(self.config.precase_clearance_motor_value)] * 4
        self._progress(
            f"precase motor clearance motors={command} duration_s={self.config.precase_clearance_s:g}"
        )
        deadline = time.monotonic() + max(0.0, float(self.config.precase_clearance_s))
        interval = 1.0 / max(1e-6, self.config.control_hz)
        while time.monotonic() < deadline:
            self.bridge.send_motor_target(command)
            time.sleep(interval)

    def _reset_simulator(self) -> None:
        self._progress("sending simulator reset before attitude feedback probe")
        previous = self.bridge.get_latest_telemetry()
        previous_reset_count = None if previous is None else getattr(previous, "reset_count", None)
        try:
            deadline = time.monotonic() + 0.1
            while time.monotonic() < deadline:
                self.bridge.send_motor_target([0.0, 0.0, 0.0, 0.0])
                time.sleep(1.0 / max(1e-6, self.config.control_hz))
        except Exception:
            pass
        self.bridge.send_sim_reset_command()
        time.sleep(self.config.reset_wait_s)
        self._wait_for_telemetry()
        self._wait_for_reset_ready(previous_reset_count)

    def _wait_for_reset_ready(self, previous_reset_count: Any | None) -> None:
        deadline = time.monotonic() + max(0.01, float(self.config.reset_ready_timeout_s))
        stable_start: float | None = None
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is None or getattr(telemetry, "position_local_ned_m", None) is None:
                stable_start = None
                time.sleep(0.05)
                continue
            reset_count = getattr(telemetry, "reset_count", None)
            if previous_reset_count is not None and reset_count == previous_reset_count:
                stable_start = None
                time.sleep(0.05)
                continue
            velocity = getattr(telemetry, "velocity_local_ned_mps", None)
            speed = 0.0 if velocity is None else float(np.linalg.norm(np.asarray(velocity, dtype=float)))
            if speed <= float(self.config.reset_stable_max_speed_mps):
                if stable_start is None:
                    stable_start = time.monotonic()
                if time.monotonic() - stable_start >= float(self.config.reset_stable_s):
                    self._progress(
                        f"reset ready reset_count={reset_count} speed={speed:.4f} "
                        f"stable_s={self.config.reset_stable_s:g}; post_delay={self.config.post_reset_delay_s:g}"
                    )
                    if self.config.post_reset_delay_s > 0.0:
                        time.sleep(self.config.post_reset_delay_s)
                    return
            else:
                stable_start = None
            time.sleep(0.05)
        raise TimeoutError("Simulator did not reach stable post-reset telemetry before timeout.")

    def _wait_for_telemetry(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None and getattr(telemetry, "position_local_ned_m", None) is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("No local-NED telemetry received for attitude feedback probe.")

    def _require_telemetry(self) -> Any:
        telemetry = self.bridge.get_latest_telemetry()
        if telemetry is None or telemetry.position_local_ned_m is None:
            raise RuntimeError("Telemetry with position_local_ned_m is required.")
        return telemetry

    def _safety_reason(self, telemetry: Any, start_reset_count: int | None) -> str | None:
        if not self.config.safety_enabled:
            return None
        if start_reset_count is not None and telemetry.reset_count is not None and int(telemetry.reset_count) != int(start_reset_count):
            return f"reset_count_changed:{start_reset_count}->{telemetry.reset_count}"
        z = float(telemetry.position_local_ned_m[2])
        if z < self.config.min_z_ned_m or z > self.config.max_z_ned_m:
            return f"z_bounds_exceeded:{z}"
        speed = float(np.linalg.norm(np.asarray(telemetry.velocity_local_ned_mps, dtype=float)))
        if speed > self.config.max_speed_mps or not math.isfinite(speed):
            return f"velocity_limit_exceeded:{speed}"
        return None

    def _actuator_output(self) -> list[float] | None:
        latest = getattr(self.bridge, "latest_actuator_output", None)
        if not latest:
            return None
        commands = latest.get("motor_commands") if isinstance(latest, dict) else None
        return None if commands is None else [float(value) for value in commands]

    def _bridge_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.bridge, "snapshot", None)
        if not callable(snapshot):
            return None
        return snapshot()

    def _progress(self, message: str) -> None:
        if self.config.progress:
            print(message, flush=True)


def _float_tuple3_from_csv(raw: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated values.")
    return values  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop SET_ATTITUDE_TARGET feedback probe.")
    parser.add_argument("--endpoint", default=AttitudeFeedbackProbeConfig.endpoint)
    parser.add_argument("--control-hz", type=float, default=AttitudeFeedbackProbeConfig.control_hz)
    parser.add_argument("--run-s", type=float, default=AttitudeFeedbackProbeConfig.run_s)
    parser.add_argument("--start-delay-s", type=float, default=AttitudeFeedbackProbeConfig.start_delay_s)
    parser.add_argument("--target-velocity", type=_float_tuple3_from_csv, default=AttitudeFeedbackProbeConfig.target_velocity_local_ned_mps)
    parser.add_argument("--target-altitude-offset-ned-m", type=float, default=AttitudeFeedbackProbeConfig.target_altitude_offset_ned_m)
    parser.add_argument("--neutral-thrust", type=float, default=AttitudeFeedbackProbeConfig.neutral_thrust)
    parser.add_argument("--fixed-trim", action="store_true")
    parser.add_argument("--max-tilt-rad", type=float, default=AttitudeFeedbackProbeConfig.max_tilt_rad)
    parser.add_argument("--max-tilt-rate-rps", type=float, default=AttitudeFeedbackProbeConfig.max_tilt_rate_rps)
    parser.add_argument("--max-horizontal-accel-mps2", type=float, default=AttitudeFeedbackProbeConfig.max_horizontal_accel_mps2)
    parser.add_argument("--max-vertical-accel-mps2", type=float, default=AttitudeFeedbackProbeConfig.max_vertical_accel_mps2)
    parser.add_argument("--reset-on-start", action="store_true")
    parser.add_argument("--reset-wait-s", type=float, default=AttitudeFeedbackProbeConfig.reset_wait_s)
    parser.add_argument("--reset-ready-timeout-s", type=float, default=AttitudeFeedbackProbeConfig.reset_ready_timeout_s)
    parser.add_argument("--reset-stable-s", type=float, default=AttitudeFeedbackProbeConfig.reset_stable_s)
    parser.add_argument("--reset-stable-max-speed-mps", type=float, default=AttitudeFeedbackProbeConfig.reset_stable_max_speed_mps)
    parser.add_argument("--post-reset-delay-s", type=float, default=AttitudeFeedbackProbeConfig.post_reset_delay_s)
    parser.add_argument("--precase-clearance", action="store_true")
    parser.add_argument("--precase-clearance-motor-value", type=float, default=AttitudeFeedbackProbeConfig.precase_clearance_motor_value)
    parser.add_argument("--precase-clearance-s", type=float, default=AttitudeFeedbackProbeConfig.precase_clearance_s)
    parser.add_argument("--max-speed-mps", type=float, default=AttitudeFeedbackProbeConfig.max_speed_mps)
    parser.add_argument("--min-z-ned-m", type=float, default=AttitudeFeedbackProbeConfig.min_z_ned_m)
    parser.add_argument("--max-z-ned-m", type=float, default=AttitudeFeedbackProbeConfig.max_z_ned_m)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=AttitudeFeedbackProbeConfig.output_path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = AttitudeFeedbackProbeConfig(
        endpoint=args.endpoint,
        control_hz=args.control_hz,
        run_s=args.run_s,
        start_delay_s=args.start_delay_s,
        target_velocity_local_ned_mps=tuple(float(value) for value in args.target_velocity),
        target_altitude_offset_ned_m=args.target_altitude_offset_ned_m,
        neutral_thrust=args.neutral_thrust,
        use_hover_trim=not args.fixed_trim,
        max_tilt_rad=args.max_tilt_rad,
        max_tilt_rate_rps=args.max_tilt_rate_rps,
        max_horizontal_accel_mps2=args.max_horizontal_accel_mps2,
        max_vertical_accel_mps2=args.max_vertical_accel_mps2,
        arm_on_start=not args.no_arm,
        disarm_on_exit=not args.no_disarm_on_exit,
        reset_on_start=bool(args.reset_on_start),
        reset_wait_s=args.reset_wait_s,
        reset_ready_timeout_s=args.reset_ready_timeout_s,
        reset_stable_s=args.reset_stable_s,
        reset_stable_max_speed_mps=args.reset_stable_max_speed_mps,
        post_reset_delay_s=args.post_reset_delay_s,
        precase_clearance=bool(args.precase_clearance),
        precase_clearance_motor_value=args.precase_clearance_motor_value,
        precase_clearance_s=args.precase_clearance_s,
        max_speed_mps=args.max_speed_mps,
        min_z_ned_m=args.min_z_ned_m,
        max_z_ned_m=args.max_z_ned_m,
        safety_enabled=not args.no_safety,
        progress=bool(args.progress),
        output_path=args.output,
    )
    probe = AttitudeFeedbackProbe(MavlinkBridge(config.endpoint), config)
    payload = probe.run_live()
    print(json.dumps(probe.summary(payload), indent=2, default=str))


if __name__ == "__main__":
    main()
