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

from sensing.telemetry import MavlinkBridge

from .body_rate_position_guidance import BodyRatePositionGuidance, BodyRatePositionGuidanceConfig
from .config import Q1RuntimeConfig


@dataclass(frozen=True)
class BodyRateFeedbackConfig:
    endpoint: str = Q1RuntimeConfig.mavlink_endpoint
    control_hz: float = 30.0
    run_s: float = 2.0
    start_delay_s: float = 0.2
    thrust: float = 0.22
    altitude_hold: bool = False
    target_z_offset_ned_m: float = 0.0
    vertical_kp: float = 0.08
    vertical_kd: float = 0.12
    min_thrust: float = 0.18
    max_thrust: float = 0.30
    prelevel_s: float = 0.0
    prelevel_thrust: float = 0.18
    prelevel_target_roll_deg: float = 0.0
    prelevel_target_pitch_deg: float = 0.0
    velocity_source: str = "local_position"
    position_guidance: bool = False
    target_position_offset_local_ned_m: tuple[float, float, float] | None = None
    position_kp_deg_per_m: float = 2.0
    velocity_kd_deg_per_mps: float = 1.5
    max_guidance_tilt_deg: float = 4.0
    position_deadband_m: float = 0.05
    target_roll_deg: float = 0.0
    target_pitch_deg: float = 0.0
    roll_kp: float = 1.2
    pitch_kp: float = 1.2
    max_roll_rate_rps: float = 0.25
    max_pitch_rate_rps: float = 0.25
    yaw_rate_rps: float = 0.0
    reset_on_start: bool = True
    reset_wait_s: float = 2.0
    reset_ready_timeout_s: float = 20.0
    reset_stable_s: float = 2.0
    reset_stable_max_speed_mps: float = 0.05
    post_reset_delay_s: float = 2.0
    arm_on_start: bool = True
    disarm_on_exit: bool = True
    max_speed_mps: float = 3.0
    min_z_ned_m: float = -3.0
    max_z_ned_m: float = 3.0
    progress: bool = False
    output_path: Path | None = Path("logs/q1runtime/body-rate-feedback-probe.json")


class BodyRateFeedbackProbe:
    """Closed-loop body-rate-only SET_ATTITUDE_TARGET probe."""

    def __init__(self, bridge: Any, config: BodyRateFeedbackConfig | None = None):
        self.bridge = bridge
        self.config = config or BodyRateFeedbackConfig()
        self._target_z_ned_m: float | None = None
        self._target_position_local_ned_m: list[float] | None = None
        self._position_guidance = BodyRatePositionGuidance(
            BodyRatePositionGuidanceConfig(
                position_kp_deg_per_m=self.config.position_kp_deg_per_m,
                velocity_kd_deg_per_mps=self.config.velocity_kd_deg_per_mps,
                max_tilt_deg=self.config.max_guidance_tilt_deg,
                position_deadband_m=self.config.position_deadband_m,
            )
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
        if self.config.start_delay_s > 0.0:
            self._progress(f"waiting start_delay_s={self.config.start_delay_s:g} before body-rate feedback")
            time.sleep(self.config.start_delay_s)

        start = self.snapshot()
        start_reset_count = start["reset_count"]
        cycles: list[dict[str, Any]] = []
        aborted = False
        abort_reason: str | None = None
        end = start
        try:
            if self.config.prelevel_s > 0.0:
                self._progress(
                    f"preleveling for {self.config.prelevel_s:g}s at thrust={self.config.prelevel_thrust:.3f}"
                )
                prelevel_end, prelevel_aborted, prelevel_abort_reason = self._run_control_phase(
                    phase="prelevel",
                    duration_s=float(self.config.prelevel_s),
                    start_reset_count=start_reset_count,
                    cycles=cycles,
                    override_thrust=float(self.config.prelevel_thrust),
                    target_roll_deg=float(self.config.prelevel_target_roll_deg),
                    target_pitch_deg=float(self.config.prelevel_target_pitch_deg),
                )
                end = prelevel_end
                aborted = prelevel_aborted
                abort_reason = prelevel_abort_reason

            if not aborted:
                target_origin = self.snapshot()
                if (
                    self.config.position_guidance
                    and self.config.target_position_offset_local_ned_m is not None
                    and target_origin["position_local_ned_m"] is not None
                ):
                    origin = np.asarray(target_origin["position_local_ned_m"], dtype=float)
                    offset = np.asarray(self.config.target_position_offset_local_ned_m, dtype=float)
                    self._target_position_local_ned_m = (origin + offset).astype(float).tolist()
                    self._target_z_ned_m = float(self._target_position_local_ned_m[2])
                    self._progress(f"position guidance target_local_ned_m={np.round(self._target_position_local_ned_m, 3).tolist()}")
                if self.config.altitude_hold and target_origin["position_local_ned_m"] is not None:
                    if self._target_z_ned_m is None:
                        self._target_z_ned_m = (
                            float(target_origin["position_local_ned_m"][2]) + float(self.config.target_z_offset_ned_m)
                        )
                    self._progress(f"altitude hold target_z_ned_m={self._target_z_ned_m:.3f}")

            if not aborted:
                end, aborted, abort_reason = self._run_control_phase(
                    phase="main",
                    duration_s=max(0.0, self.config.run_s),
                    start_reset_count=start_reset_count,
                    cycles=cycles,
                )
        finally:
            if cycles and end is start:
                end = cycles[-1]["sample"]
            self._send_stop(0.2)
            if self.config.disarm_on_exit:
                try:
                    self.bridge.disarm()
                except Exception:
                    pass
            self.bridge.shutdown()

        payload = self._payload(start, end, cycles, aborted, abort_reason)
        if self.config.output_path is not None:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def _run_control_phase(
        self,
        *,
        phase: str,
        duration_s: float,
        start_reset_count: int | None,
        cycles: list[dict[str, Any]],
        override_thrust: float | None = None,
        target_roll_deg: float | None = None,
        target_pitch_deg: float | None = None,
    ) -> tuple[dict[str, Any], bool, str | None]:
        end = self.snapshot()
        aborted = False
        abort_reason: str | None = None
        if duration_s <= 0.0:
            return end, aborted, abort_reason
        deadline = time.monotonic() + duration_s
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += 1.0 / max(1e-6, self.config.control_hz)
            sample = self.snapshot()
            safety_reason = self._safety_reason(sample, start_reset_count)
            if safety_reason is not None:
                aborted = True
                abort_reason = safety_reason
                self._progress(f"safety abort phase={phase} cycle={len(cycles)}: {safety_reason}")
                end = sample
                break
            guidance_result = self._guidance_result(sample) if phase == "main" else None
            command_target_roll_deg = target_roll_deg
            command_target_pitch_deg = target_pitch_deg
            if guidance_result is not None:
                command_target_roll_deg = float(guidance_result["target_roll_deg"])
                command_target_pitch_deg = float(guidance_result["target_pitch_deg"])
            payload = self.build_payload(
                sample,
                override_thrust=override_thrust,
                phase=phase,
                target_roll_deg=command_target_roll_deg,
                target_pitch_deg=command_target_pitch_deg,
            )
            if guidance_result is not None:
                payload["position_guidance"] = guidance_result
            self.bridge.send_attitude_target(payload)
            cycle_payload = {
                "cycle": len(cycles),
                "phase": phase,
                "sample": sample,
                "command": payload,
                "actuator_output": self._actuator_output(),
            }
            cycles.append(cycle_payload)
            if self.config.progress and len(cycles) % max(1, round(self.config.control_hz / 5.0)) == 1:
                self._progress(
                    f"phase={phase} cycle={len(cycles) - 1} pos={np.round(sample['position_local_ned_m'], 3).tolist()} "
                    f"vel={np.round(sample['velocity_local_ned_mps'], 3).tolist()} "
                    f"euler={np.round(sample['euler_deg'], 2).tolist()} "
                    f"rates={np.round(payload['body_rates_rps'], 3).tolist()} thrust={payload['thrust']:.3f}"
                )
            end = sample
        return end, aborted, abort_reason

    def _guidance_result(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        if not self.config.position_guidance or self._target_position_local_ned_m is None:
            return None
        if sample["position_local_ned_m"] is None or sample["velocity_local_ned_mps"] is None:
            return None
        return self._position_guidance.compute(
            current_position_local_ned_m=sample["position_local_ned_m"],
            current_velocity_local_ned_mps=sample["velocity_local_ned_mps"],
            target_position_local_ned_m=self._target_position_local_ned_m,
        )

    def build_payload(
        self,
        sample: dict[str, Any],
        *,
        override_thrust: float | None = None,
        phase: str = "main",
        target_roll_deg: float | None = None,
        target_pitch_deg: float | None = None,
    ) -> dict[str, Any]:
        roll_deg, pitch_deg, _yaw_deg = sample["euler_deg"]
        command_target_roll_deg = self.config.target_roll_deg if target_roll_deg is None else target_roll_deg
        command_target_pitch_deg = self.config.target_pitch_deg if target_pitch_deg is None else target_pitch_deg
        target_roll = math.radians(float(command_target_roll_deg))
        target_pitch = math.radians(float(command_target_pitch_deg))
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        # In this simulator's body-rate-only surface, positive roll-rate decreases Euler roll;
        # positive pitch-rate increases Euler pitch.
        roll_rate = -self.config.roll_kp * (target_roll - roll)
        pitch_rate = self.config.pitch_kp * (target_pitch - pitch)
        roll_rate = float(np.clip(roll_rate, -self.config.max_roll_rate_rps, self.config.max_roll_rate_rps))
        pitch_rate = float(np.clip(pitch_rate, -self.config.max_pitch_rate_rps, self.config.max_pitch_rate_rps))
        if override_thrust is None:
            thrust, thrust_debug = self._compute_thrust(sample)
        else:
            thrust = float(np.clip(override_thrust, 0.0, 1.0))
            thrust_debug = {
                "mode": "override_fixed",
                "base_thrust": float(override_thrust),
                "ned_z_note": "positive_z_is_down",
            }
        return {
            "quaternion": sample["attitude"],
            "thrust": thrust,
            "attitude_type_mask": 128,
            "body_rates_rps": [roll_rate, pitch_rate, float(self.config.yaw_rate_rps)],
            "source": "q1runtime_body_rate_feedback_probe",
            "phase": phase,
            "target_roll_pitch_deg": [float(command_target_roll_deg), float(command_target_pitch_deg)],
            "thrust_control": thrust_debug,
        }

    def _compute_thrust(self, sample: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        base = float(self.config.thrust)
        target_z = self._target_z_ned_m
        if not self.config.altitude_hold or target_z is None or sample.get("position_local_ned_m") is None:
            return float(np.clip(base, 0.0, 1.0)), {
                "mode": "fixed",
                "base_thrust": base,
                "ned_z_note": "positive_z_is_down",
            }

        z = float(sample["position_local_ned_m"][2])
        velocity = sample.get("velocity_local_ned_mps") or [0.0, 0.0, 0.0]
        vz = float(velocity[2])
        z_error = z - float(target_z)
        thrust_raw = base + float(self.config.vertical_kp) * z_error + float(self.config.vertical_kd) * vz
        thrust = float(np.clip(thrust_raw, float(self.config.min_thrust), float(self.config.max_thrust)))
        return thrust, {
            "mode": "altitude_hold",
            "base_thrust": base,
            "target_z_ned_m": float(target_z),
            "z_error_ned_m": z_error,
            "vz_ned_mps": vz,
            "vertical_kp": float(self.config.vertical_kp),
            "vertical_kd": float(self.config.vertical_kd),
            "raw_thrust": float(thrust_raw),
            "min_thrust": float(self.config.min_thrust),
            "max_thrust": float(self.config.max_thrust),
            "ned_z_note": "positive_z_is_down; falling/downward velocity increases thrust",
        }

    def snapshot(self) -> dict[str, Any]:
        telemetry = self.bridge.get_latest_telemetry()
        if telemetry is None:
            return {
                "monotonic_s": time.monotonic(),
                "sim_time_ns": None,
                "position_local_ned_m": None,
                "velocity_local_ned_mps": None,
                "attitude": [1.0, 0.0, 0.0, 0.0],
                "euler_deg": [0.0, 0.0, 0.0],
                "reset_count": None,
            }
        attitude = [float(value) for value in telemetry.attitude]
        velocity, velocity_source, velocity_debug = self._velocity_local_ned(telemetry)
        return {
            "monotonic_s": time.monotonic(),
            "sim_time_ns": int(telemetry.sim_time_ns),
            "position_local_ned_m": [float(value) for value in telemetry.position_local_ned_m],
            "velocity_local_ned_mps": velocity,
            "velocity_source": velocity_source,
            "velocity_debug": velocity_debug,
            "attitude": attitude,
            "euler_deg": self._euler_deg(attitude),
            "reset_count": telemetry.reset_count,
            "collision_count": len(getattr(self.bridge, "collisions", []) or []),
        }

    def _velocity_local_ned(self, telemetry: Any) -> tuple[list[float], str, dict[str, Any]]:
        odometry_velocity = [float(value) for value in telemetry.velocity_local_ned_mps]
        latest_local_position = getattr(self.bridge, "latest_local_position", None)
        local_position_velocity = None
        if isinstance(latest_local_position, dict):
            velocity = latest_local_position.get("velocity_local_ned_mps")
            if velocity is not None:
                local_position_velocity = [float(value) for value in velocity]

        if self.config.velocity_source == "local_position" and local_position_velocity is not None:
            return local_position_velocity, "local_position_ned", {
                "odometry_velocity_mps": odometry_velocity,
                "local_position_velocity_mps": local_position_velocity,
            }
        return odometry_velocity, "odometry", {
            "odometry_velocity_mps": odometry_velocity,
            "local_position_velocity_mps": local_position_velocity,
        }

    def _payload(
        self,
        start: dict[str, Any],
        end: dict[str, Any],
        cycles: list[dict[str, Any]],
        aborted: bool,
        abort_reason: str | None,
    ) -> dict[str, Any]:
        displacement = None
        if start["position_local_ned_m"] is not None and end["position_local_ned_m"] is not None:
            displacement = (
                np.asarray(end["position_local_ned_m"], dtype=float)
                - np.asarray(start["position_local_ned_m"], dtype=float)
            ).tolist()
        speeds = [
            float(np.linalg.norm(np.asarray(cycle["sample"]["velocity_local_ned_mps"], dtype=float)))
            for cycle in cycles
            if cycle["sample"]["velocity_local_ned_mps"] is not None
        ]
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "aborted": aborted,
            "abort_reason": abort_reason,
            "cycle_count": len(cycles),
            "start": start,
            "end": end,
            "displacement_local_ned_m": displacement,
            "max_speed_mps": None if not speeds else max(speeds),
            "cycles": cycles,
            "bridge_snapshot": self._bridge_snapshot(),
        }

    def summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "created_utc": payload["created_utc"],
            "output_path": None if self.config.output_path is None else str(self.config.output_path),
            "cycle_count": payload["cycle_count"],
            "aborted": payload["aborted"],
            "abort_reason": payload["abort_reason"],
            "start_position_local_ned_m": payload["start"]["position_local_ned_m"],
            "end_position_local_ned_m": payload["end"]["position_local_ned_m"],
            "displacement_local_ned_m": payload["displacement_local_ned_m"],
            "start_euler_deg": payload["start"]["euler_deg"],
            "end_euler_deg": payload["end"]["euler_deg"],
            "max_speed_mps": payload["max_speed_mps"],
        }

    def _send_stop(self, duration_s: float) -> None:
        sample = self.snapshot()
        payload = {
            "quaternion": sample["attitude"],
            "thrust": 0.0,
            "attitude_type_mask": 128,
            "body_rates_rps": [0.0, 0.0, 0.0],
            "source": "q1runtime_body_rate_feedback_stop",
        }
        deadline = time.monotonic() + max(0.0, duration_s)
        interval = 1.0 / max(1e-6, self.config.control_hz)
        while time.monotonic() < deadline:
            self.bridge.send_attitude_target(payload)
            time.sleep(interval)

    def _reset_simulator(self) -> None:
        self._progress("sending simulator reset before body-rate feedback probe")
        previous = self.bridge.get_latest_telemetry()
        previous_reset_count = None if previous is None else getattr(previous, "reset_count", None)
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
            speed = float(np.linalg.norm(np.asarray(telemetry.velocity_local_ned_mps, dtype=float)))
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
        raise TimeoutError("No local-NED telemetry received for body-rate feedback probe.")

    def _safety_reason(self, sample: dict[str, Any], start_reset_count: int | None) -> str | None:
        if start_reset_count is not None and sample["reset_count"] is not None and int(sample["reset_count"]) != int(start_reset_count):
            return f"reset_count_changed:{start_reset_count}->{sample['reset_count']}"
        z = float(sample["position_local_ned_m"][2])
        if z < self.config.min_z_ned_m or z > self.config.max_z_ned_m:
            return f"z_bounds_exceeded:{z}"
        speed = float(np.linalg.norm(np.asarray(sample["velocity_local_ned_mps"], dtype=float)))
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
        return None if not callable(snapshot) else snapshot()

    def _progress(self, message: str) -> None:
        if self.config.progress:
            print(message, flush=True)

    @staticmethod
    def _euler_deg(quaternion: list[float]) -> list[float]:
        w, x, y, z = [float(value) for value in quaternion]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return [math.degrees(value) for value in (roll, pitch, yaw)]


def _float_tuple3_from_csv(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats, for example 1.0,0.0,-0.4")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop body-rate-only SET_ATTITUDE_TARGET probe.")
    parser.add_argument("--endpoint", default=BodyRateFeedbackConfig.endpoint)
    parser.add_argument("--control-hz", type=float, default=BodyRateFeedbackConfig.control_hz)
    parser.add_argument("--run-s", type=float, default=BodyRateFeedbackConfig.run_s)
    parser.add_argument("--start-delay-s", type=float, default=BodyRateFeedbackConfig.start_delay_s)
    parser.add_argument("--thrust", type=float, default=BodyRateFeedbackConfig.thrust)
    parser.add_argument("--altitude-hold", action="store_true")
    parser.add_argument("--target-z-offset-ned-m", type=float, default=BodyRateFeedbackConfig.target_z_offset_ned_m)
    parser.add_argument("--vertical-kp", type=float, default=BodyRateFeedbackConfig.vertical_kp)
    parser.add_argument("--vertical-kd", type=float, default=BodyRateFeedbackConfig.vertical_kd)
    parser.add_argument("--min-thrust", type=float, default=BodyRateFeedbackConfig.min_thrust)
    parser.add_argument("--max-thrust", type=float, default=BodyRateFeedbackConfig.max_thrust)
    parser.add_argument("--prelevel-s", type=float, default=BodyRateFeedbackConfig.prelevel_s)
    parser.add_argument("--prelevel-thrust", type=float, default=BodyRateFeedbackConfig.prelevel_thrust)
    parser.add_argument("--prelevel-target-roll-deg", type=float, default=BodyRateFeedbackConfig.prelevel_target_roll_deg)
    parser.add_argument("--prelevel-target-pitch-deg", type=float, default=BodyRateFeedbackConfig.prelevel_target_pitch_deg)
    parser.add_argument(
        "--velocity-source",
        choices=("local_position", "odometry"),
        default=BodyRateFeedbackConfig.velocity_source,
    )
    parser.add_argument("--position-guidance", action="store_true")
    parser.add_argument(
        "--target-position-offset-local-ned-m",
        type=_float_tuple3_from_csv,
        default=BodyRateFeedbackConfig.target_position_offset_local_ned_m,
    )
    parser.add_argument("--position-kp-deg-per-m", type=float, default=BodyRateFeedbackConfig.position_kp_deg_per_m)
    parser.add_argument("--velocity-kd-deg-per-mps", type=float, default=BodyRateFeedbackConfig.velocity_kd_deg_per_mps)
    parser.add_argument("--max-guidance-tilt-deg", type=float, default=BodyRateFeedbackConfig.max_guidance_tilt_deg)
    parser.add_argument("--position-deadband-m", type=float, default=BodyRateFeedbackConfig.position_deadband_m)
    parser.add_argument("--target-roll-deg", type=float, default=BodyRateFeedbackConfig.target_roll_deg)
    parser.add_argument("--target-pitch-deg", type=float, default=BodyRateFeedbackConfig.target_pitch_deg)
    parser.add_argument("--roll-kp", type=float, default=BodyRateFeedbackConfig.roll_kp)
    parser.add_argument("--pitch-kp", type=float, default=BodyRateFeedbackConfig.pitch_kp)
    parser.add_argument("--max-roll-rate-rps", type=float, default=BodyRateFeedbackConfig.max_roll_rate_rps)
    parser.add_argument("--max-pitch-rate-rps", type=float, default=BodyRateFeedbackConfig.max_pitch_rate_rps)
    parser.add_argument("--yaw-rate-rps", type=float, default=BodyRateFeedbackConfig.yaw_rate_rps)
    parser.add_argument("--no-reset-on-start", action="store_true")
    parser.add_argument("--reset-wait-s", type=float, default=BodyRateFeedbackConfig.reset_wait_s)
    parser.add_argument("--reset-ready-timeout-s", type=float, default=BodyRateFeedbackConfig.reset_ready_timeout_s)
    parser.add_argument("--reset-stable-s", type=float, default=BodyRateFeedbackConfig.reset_stable_s)
    parser.add_argument("--reset-stable-max-speed-mps", type=float, default=BodyRateFeedbackConfig.reset_stable_max_speed_mps)
    parser.add_argument("--post-reset-delay-s", type=float, default=BodyRateFeedbackConfig.post_reset_delay_s)
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--max-speed-mps", type=float, default=BodyRateFeedbackConfig.max_speed_mps)
    parser.add_argument("--min-z-ned-m", type=float, default=BodyRateFeedbackConfig.min_z_ned_m)
    parser.add_argument("--max-z-ned-m", type=float, default=BodyRateFeedbackConfig.max_z_ned_m)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=BodyRateFeedbackConfig.output_path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = BodyRateFeedbackConfig(
        endpoint=args.endpoint,
        control_hz=args.control_hz,
        run_s=args.run_s,
        start_delay_s=args.start_delay_s,
        thrust=args.thrust,
        altitude_hold=bool(args.altitude_hold),
        target_z_offset_ned_m=args.target_z_offset_ned_m,
        vertical_kp=args.vertical_kp,
        vertical_kd=args.vertical_kd,
        min_thrust=args.min_thrust,
        max_thrust=args.max_thrust,
        prelevel_s=args.prelevel_s,
        prelevel_thrust=args.prelevel_thrust,
        prelevel_target_roll_deg=args.prelevel_target_roll_deg,
        prelevel_target_pitch_deg=args.prelevel_target_pitch_deg,
        velocity_source=args.velocity_source,
        position_guidance=bool(args.position_guidance),
        target_position_offset_local_ned_m=args.target_position_offset_local_ned_m,
        position_kp_deg_per_m=args.position_kp_deg_per_m,
        velocity_kd_deg_per_mps=args.velocity_kd_deg_per_mps,
        max_guidance_tilt_deg=args.max_guidance_tilt_deg,
        position_deadband_m=args.position_deadband_m,
        target_roll_deg=args.target_roll_deg,
        target_pitch_deg=args.target_pitch_deg,
        roll_kp=args.roll_kp,
        pitch_kp=args.pitch_kp,
        max_roll_rate_rps=args.max_roll_rate_rps,
        max_pitch_rate_rps=args.max_pitch_rate_rps,
        yaw_rate_rps=args.yaw_rate_rps,
        reset_on_start=not args.no_reset_on_start,
        reset_wait_s=args.reset_wait_s,
        reset_ready_timeout_s=args.reset_ready_timeout_s,
        reset_stable_s=args.reset_stable_s,
        reset_stable_max_speed_mps=args.reset_stable_max_speed_mps,
        post_reset_delay_s=args.post_reset_delay_s,
        arm_on_start=not args.no_arm,
        disarm_on_exit=not args.no_disarm_on_exit,
        max_speed_mps=args.max_speed_mps,
        min_z_ned_m=args.min_z_ned_m,
        max_z_ned_m=args.max_z_ned_m,
        progress=bool(args.progress),
        output_path=args.output,
    )
    probe = BodyRateFeedbackProbe(MavlinkBridge(config.endpoint), config)
    payload = probe.run_live()
    print(json.dumps(probe.summary(payload), indent=2, default=str))


if __name__ == "__main__":
    main()
