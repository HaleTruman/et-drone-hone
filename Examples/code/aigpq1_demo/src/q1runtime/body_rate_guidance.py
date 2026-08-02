from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .body_rate_position_guidance import BodyRatePositionGuidance, BodyRatePositionGuidanceConfig
from .command_transform import CommandFrameTransform
from .target_mapper import LocalVisionTarget


@dataclass(frozen=True)
class BodyRateGuidanceControlConfig:
    velocity_source: str = "local_position"
    base_thrust: float = 0.24
    vertical_kp: float = 0.14
    vertical_kd: float = 0.12
    min_thrust: float = 0.20
    max_thrust: float = 0.34
    roll_kp: float = 1.2
    pitch_kp: float = 2.0
    max_roll_rate_rps: float = 0.25
    max_pitch_rate_rps: float = 0.5
    yaw_rate_rps: float = 0.0
    position_kp_deg_per_m: float = 2.0
    velocity_kd_deg_per_mps: float = 4.0
    max_guidance_tilt_deg: float = 2.0
    position_deadband_m: float = 0.05
    max_horizontal_target_m: float = 1.0
    max_up_target_m: float = 0.40
    max_down_target_m: float = 0.10
    dry_run: bool = False


class BodyRateGuidanceController:
    """Emit the discovered body-rate-only control surface for local-NED targets."""

    def __init__(
        self,
        bridge: Any,
        config: BodyRateGuidanceControlConfig | None = None,
        *,
        command_transform: CommandFrameTransform | None = None,
    ):
        self.bridge = bridge
        self.config = config or BodyRateGuidanceControlConfig()
        self.command_transform = command_transform or CommandFrameTransform.identity()
        self.position_guidance = BodyRatePositionGuidance(
            BodyRatePositionGuidanceConfig(
                position_kp_deg_per_m=self.config.position_kp_deg_per_m,
                velocity_kd_deg_per_mps=self.config.velocity_kd_deg_per_mps,
                max_tilt_deg=self.config.max_guidance_tilt_deg,
                position_deadband_m=self.config.position_deadband_m,
            )
        )
        self.last_payload: dict[str, Any] | None = None

    def emit_guidance(
        self,
        *,
        telemetry: Any,
        target: LocalVisionTarget | None,
        phase: str = "main",
        source: str = "q1runtime_body_rate_guidance",
    ) -> dict[str, Any]:
        sample = self.snapshot(telemetry)
        target_payload = (
            self._target_payload(sample, target)
            if target is not None
            else self._hold_target_payload(sample, reason="no_active_target_hold_position")
        )
        guidance = self.position_guidance.compute(
            current_position_local_ned_m=sample["position_local_ned_m"],
            current_velocity_local_ned_mps=sample["velocity_local_ned_mps"],
            target_position_local_ned_m=target_payload["target_position_local_ned_m"],
        )
        payload = self.build_payload(
            sample=sample,
            target_roll_deg=float(guidance["target_roll_deg"]),
            target_pitch_deg=float(guidance["target_pitch_deg"]),
            target_z_ned_m=float(target_payload["target_position_local_ned_m"][2]),
            phase=phase,
            source=source,
            target_payload=target_payload,
            guidance=guidance,
            target=target,
        )
        return self.emit_payload(payload)

    def emit_prelevel(
        self,
        *,
        telemetry: Any,
        target_roll_deg: float = 0.0,
        target_pitch_deg: float = 0.0,
        thrust: float = 0.20,
    ) -> dict[str, Any]:
        sample = self.snapshot(telemetry)
        payload = self.build_payload(
            sample=sample,
            target_roll_deg=target_roll_deg,
            target_pitch_deg=target_pitch_deg,
            target_z_ned_m=float(sample["position_local_ned_m"][2]),
            phase="prelevel",
            source="q1runtime_body_rate_prelevel",
            override_thrust=float(thrust),
            target_payload=self._hold_target_payload(sample, reason="prelevel"),
            guidance=None,
            target=None,
        )
        return self.emit_payload(payload)

    def emit_stop(self, telemetry: Any | None = None, *, source: str = "q1runtime_body_rate_stop") -> dict[str, Any]:
        attitude = [1.0, 0.0, 0.0, 0.0]
        if telemetry is not None and getattr(telemetry, "attitude", None) is not None:
            attitude = [float(value) for value in telemetry.attitude]
        payload = {
            "quaternion": attitude,
            "thrust": 0.0,
            "attitude_type_mask": 128,
            "body_rates_rps": [0.0, 0.0, 0.0],
            "source": source,
            "phase": "stop",
        }
        return self.emit_payload(payload)

    def emit_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.last_payload = payload
        if not self.config.dry_run:
            self.bridge.send_attitude_target(payload)
        return payload

    def build_payload(
        self,
        *,
        sample: dict[str, Any],
        target_roll_deg: float,
        target_pitch_deg: float,
        target_z_ned_m: float,
        phase: str,
        source: str,
        target_payload: dict[str, Any],
        guidance: dict[str, Any] | None,
        target: LocalVisionTarget | None,
        override_thrust: float | None = None,
    ) -> dict[str, Any]:
        roll_deg, pitch_deg, _yaw_deg = sample["euler_deg"]
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        target_roll = math.radians(float(target_roll_deg))
        target_pitch = math.radians(float(target_pitch_deg))
        # Empirical simulator convention: positive roll-rate decreases Euler roll;
        # positive pitch-rate increases Euler pitch.
        roll_rate = -float(self.config.roll_kp) * (target_roll - roll)
        pitch_rate = float(self.config.pitch_kp) * (target_pitch - pitch)
        roll_rate = float(np.clip(roll_rate, -self.config.max_roll_rate_rps, self.config.max_roll_rate_rps))
        pitch_rate = float(np.clip(pitch_rate, -self.config.max_pitch_rate_rps, self.config.max_pitch_rate_rps))

        if override_thrust is None:
            thrust, thrust_control = self._compute_thrust(sample, target_z_ned_m)
        else:
            thrust = float(np.clip(override_thrust, 0.0, 1.0))
            thrust_control = {
                "mode": "override_fixed",
                "base_thrust": float(override_thrust),
                "ned_z_note": "positive_z_is_down",
            }

        payload: dict[str, Any] = {
            "quaternion": sample["attitude"],
            "thrust": thrust,
            "attitude_type_mask": 128,
            "body_rates_rps": [roll_rate, pitch_rate, float(self.config.yaw_rate_rps)],
            "source": source,
            "phase": phase,
            "target_roll_pitch_deg": [float(target_roll_deg), float(target_pitch_deg)],
            "thrust_control": thrust_control,
            "velocity_source": sample["velocity_source"],
            "velocity_debug": sample["velocity_debug"],
            "target_control": target_payload,
            "position_guidance": guidance,
        }
        if target is not None:
            payload.update(
                {
                    "vision_frame_id": int(target.frame_id),
                    "vision_sim_time_ns": int(target.sim_time_ns),
                    "gate_id": target.gate_id,
                    "position_confidence": float(target.position_confidence),
                    "target_position_camera_m": [float(value) for value in target.position_camera_m],
                    "raw_target_position_local_ned_m": [float(value) for value in target.position_local_ned_m],
                }
            )
        return payload

    def snapshot(self, telemetry: Any) -> dict[str, Any]:
        position = getattr(telemetry, "position_local_ned_m", None)
        attitude = getattr(telemetry, "attitude", None)
        if position is None:
            raise ValueError("Telemetry must include position_local_ned_m for body-rate guidance.")
        if attitude is None:
            raise ValueError("Telemetry must include attitude for body-rate guidance.")
        velocity, velocity_source, velocity_debug = self._velocity_local_ned(telemetry)
        attitude_list = [float(value) for value in attitude]
        return {
            "sim_time_ns": int(getattr(telemetry, "sim_time_ns", 0)),
            "position_local_ned_m": [float(value) for value in position],
            "velocity_local_ned_mps": velocity,
            "velocity_source": velocity_source,
            "velocity_debug": velocity_debug,
            "attitude": attitude_list,
            "euler_deg": self.euler_deg(attitude_list),
            "reset_count": getattr(telemetry, "reset_count", None),
        }

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "command_transform": self.command_transform.to_payload(),
            "last_payload": self.last_payload,
        }

    def _target_payload(self, sample: dict[str, Any], target: LocalVisionTarget) -> dict[str, Any]:
        current = np.asarray(sample["position_local_ned_m"], dtype=float)
        raw_target = np.asarray(target.position_local_ned_m, dtype=float)
        raw_delta = raw_target - current
        clipped_delta = self._clip_delta(raw_delta)
        transformed_delta = self.command_transform.apply(clipped_delta)
        command_target = current + transformed_delta
        return {
            "mode": "vision_top1_clipped",
            "target_position_local_ned_m": command_target.astype(float).tolist(),
            "raw_target_position_local_ned_m": raw_target.astype(float).tolist(),
            "raw_delta_local_ned_m": raw_delta.astype(float).tolist(),
            "clipped_delta_local_ned_m": clipped_delta.astype(float).tolist(),
            "transformed_delta_local_ned_m": transformed_delta.astype(float).tolist(),
            "command_transform": self.command_transform.to_payload(),
            "limits": {
                "max_horizontal_target_m": float(self.config.max_horizontal_target_m),
                "max_up_target_m": float(self.config.max_up_target_m),
                "max_down_target_m": float(self.config.max_down_target_m),
            },
        }

    def _hold_target_payload(self, sample: dict[str, Any], *, reason: str) -> dict[str, Any]:
        position = [float(value) for value in sample["position_local_ned_m"]]
        return {
            "mode": "hold_current_position",
            "reason": reason,
            "target_position_local_ned_m": position,
            "raw_target_position_local_ned_m": position,
            "raw_delta_local_ned_m": [0.0, 0.0, 0.0],
            "clipped_delta_local_ned_m": [0.0, 0.0, 0.0],
            "transformed_delta_local_ned_m": [0.0, 0.0, 0.0],
            "command_transform": self.command_transform.to_payload(),
        }

    def _clip_delta(self, delta: np.ndarray) -> np.ndarray:
        clipped = np.asarray(delta, dtype=float).copy()
        horizontal_norm = float(np.linalg.norm(clipped[:2]))
        max_horizontal = max(0.0, float(self.config.max_horizontal_target_m))
        if horizontal_norm > max_horizontal > 0.0:
            clipped[:2] = clipped[:2] / horizontal_norm * max_horizontal
        elif max_horizontal <= 0.0:
            clipped[:2] = 0.0
        clipped[2] = float(
            np.clip(
                clipped[2],
                -max(0.0, float(self.config.max_up_target_m)),
                max(0.0, float(self.config.max_down_target_m)),
            )
        )
        return clipped

    def _compute_thrust(self, sample: dict[str, Any], target_z_ned_m: float) -> tuple[float, dict[str, Any]]:
        z = float(sample["position_local_ned_m"][2])
        vz = float(sample["velocity_local_ned_mps"][2])
        z_error = z - float(target_z_ned_m)
        raw = float(self.config.base_thrust) + float(self.config.vertical_kp) * z_error + float(self.config.vertical_kd) * vz
        thrust = float(np.clip(raw, float(self.config.min_thrust), float(self.config.max_thrust)))
        return thrust, {
            "mode": "altitude_hold",
            "base_thrust": float(self.config.base_thrust),
            "target_z_ned_m": float(target_z_ned_m),
            "z_error_ned_m": z_error,
            "vz_ned_mps": vz,
            "vertical_kp": float(self.config.vertical_kp),
            "vertical_kd": float(self.config.vertical_kd),
            "raw_thrust": raw,
            "min_thrust": float(self.config.min_thrust),
            "max_thrust": float(self.config.max_thrust),
            "ned_z_note": "positive_z_is_down; falling/downward velocity increases thrust",
        }

    def _velocity_local_ned(self, telemetry: Any) -> tuple[list[float], str, dict[str, Any]]:
        odometry_velocity = [float(value) for value in getattr(telemetry, "velocity_local_ned_mps")]
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

    @staticmethod
    def euler_deg(quaternion: list[float] | tuple[float, float, float, float]) -> list[float]:
        w, x, y, z = [float(value) for value in quaternion]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return [math.degrees(value) for value in (roll, pitch, yaw)]
