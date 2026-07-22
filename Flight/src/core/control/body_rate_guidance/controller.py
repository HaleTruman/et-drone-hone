import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from core.schemas import VehicleState


@dataclass(frozen=True)
class BodyRatePositionGuidanceConfig:
    position_kp_deg_per_m: float = 16.0
    velocity_kd_deg_per_mps: float = 4.0
    max_tilt_deg: float = 16.0
    position_deadband_m: float = 0.05


@dataclass(frozen=True)
class BodyRateGuidanceConfig:
    base_thrust: float = 0.255
    vertical_kp: float = 0.14
    vertical_kd: float = 0.18
    min_thrust: float = 0.20
    max_thrust: float = 0.38
    roll_kp: float = 1.2
    pitch_kp: float = 2.0
    max_roll_rate_rps: float = 0.25
    max_pitch_rate_rps: float = 0.5
    yaw_rate_rps: float = 0.0
    max_horizontal_target_m: float = 1.0
    max_up_target_m: float = 0.40
    max_down_target_m: float = 2.0
    position_guidance: BodyRatePositionGuidanceConfig = field(default_factory=BodyRatePositionGuidanceConfig)


class BodyRateGuidanceController:
    """Map local-NED gate targets to body-rate-only SET_ATTITUDE_TARGET payloads."""

    def __init__(self, config: BodyRateGuidanceConfig | None = None):
        self.config = config or BodyRateGuidanceConfig()
        self.last_payload: dict[str, Any] | None = None

    def build_guidance_command(
        self,
        *,
        vehicle_state: VehicleState,
        target: Any | None,
        source: str = "main_body_rate_guidance",
        phase: str = "main",
    ) -> dict[str, Any]:
        sample = self._sample(vehicle_state)
        target_payload = (
            self._target_payload(sample, target)
            if target is not None
            else self._hold_target_payload(sample, reason="no_active_target_hold_position")
        )
        guidance = self._position_guidance(
            current_position_local_ned_m=sample["position_local_ned_m"],
            current_velocity_local_ned_mps=sample["velocity_local_ned_mps"],
            target_position_local_ned_m=target_payload["target_position_local_ned_m"],
        )
        payload = self._build_payload(
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
        self.last_payload = payload
        return payload

    def build_prelevel_command(
        self,
        *,
        vehicle_state: VehicleState,
        thrust: float = 0.20,
        target_roll_deg: float = 0.0,
        target_pitch_deg: float = 0.0,
    ) -> dict[str, Any]:
        payload = self.build_rate_command(
            body_rates_rps=(0.0, 0.0, 0.0),
            thrust=thrust,
            source="main_body_rate_prelevel",
            phase="prelevel",
            metadata={
                "target_roll_pitch_deg": [float(target_roll_deg), float(target_pitch_deg)],
                "vehicle_state_sim_time_ns": int(vehicle_state.sim_time_ns),
            },
        )
        self.last_payload = payload
        return payload

    def build_rate_command(
        self,
        *,
        body_rates_rps: tuple[float, float, float],
        thrust: float,
        source: str,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "thrust": float(np.clip(thrust, 0.0, 1.0)),
            "attitude_type_mask": 128,
            "body_rates_rps": [float(value) for value in body_rates_rps],
            "source": source,
            "phase": phase,
        }
        if metadata:
            payload.update(metadata)
        self.last_payload = payload
        return payload

    def build_attitude_command(
        self,
        *,
        quaternion: tuple[float, float, float, float],
        thrust: float,
        source: str,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "quaternion": [float(value) for value in quaternion],
            "thrust": float(np.clip(thrust, 0.0, 1.0)),
            "attitude_type_mask": 7,
            "source": source,
            "phase": phase,
        }
        if metadata:
            payload.update(metadata)
        self.last_payload = payload
        return payload

    def build_stop_command(
        self,
        vehicle_state: VehicleState | None = None,
        *,
        source: str = "main_body_rate_stop",
    ) -> dict[str, Any]:
        attitude = [1.0, 0.0, 0.0, 0.0]
        if vehicle_state is not None:
            attitude = [float(value) for value in vehicle_state.attitude_quaternion]
        payload = {
            "quaternion": attitude,
            "thrust": 0.0,
            "attitude_type_mask": 128,
            "body_rates_rps": [0.0, 0.0, 0.0],
            "source": source,
            "phase": "stop",
        }
        self.last_payload = payload
        return payload

    def run_prelevel(
        self,
        *,
        telemetry_client: Any,
        vehicle_state_estimator: Any,
        send_attitude_target: Any,
        duration_s: float,
        thrust: float,
        hz: float,
        log_event: Any | None = None,
    ) -> int:
        if duration_s <= 0.0:
            return 0
        if log_event is not None:
            log_event("prelevel_started", duration_s=duration_s, thrust=thrust)
        interval_s = 1.0 / max(1e-6, float(hz))
        deadline_s = time.perf_counter() + float(duration_s)
        next_tick_s = time.perf_counter()
        cycles = 0
        while time.perf_counter() < deadline_s:
            now_s = time.perf_counter()
            if now_s < next_tick_s:
                time.sleep(min(0.002, next_tick_s - now_s))
                continue
            raw_telemetry = telemetry_client.get_telemetry()
            telemetry = vehicle_state_estimator.update_telemetry(raw_telemetry)
            if telemetry is not None:
                send_attitude_target(
                    self.build_prelevel_command(
                        vehicle_state=vehicle_state_estimator.state,
                        thrust=thrust,
                    )
                )
                cycles += 1
            next_tick_s += interval_s
        if log_event is not None:
            log_event("prelevel_finished", cycles=cycles)
        return cycles

    def snapshot(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "last_payload": self.last_payload}

    def _build_payload(
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
        target: Any | None,
        override_thrust: float | None = None,
    ) -> dict[str, Any]:
        roll_deg, pitch_deg, _yaw_deg = sample["euler_deg"]
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        target_roll = math.radians(float(target_roll_deg))
        target_pitch = math.radians(float(target_pitch_deg))
        roll_rate = float(self.config.roll_kp) * (target_roll - roll)
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
            "target_control": target_payload,
            "position_guidance": guidance,
        }
        if target is not None:
            payload.update(
                {
                    "vision_frame_id": getattr(target, "frame_id", None),
                    "gate_id": getattr(target, "gate_id", None),
                    "position_confidence": float(getattr(target, "confidence", 0.0)),
                    "raw_target_position_local_ned_m": [
                        float(value) for value in getattr(target, "position_local_ned_m")
                    ],
                }
            )
        return payload

    def _target_payload(self, sample: dict[str, Any], target: Any) -> dict[str, Any]:
        current = np.asarray(sample["position_local_ned_m"], dtype=float)
        raw_target = np.asarray(getattr(target, "position_local_ned_m"), dtype=float)
        relative_position = getattr(target, "position_relative_ned_m", None)
        raw_delta = (
            np.asarray(relative_position, dtype=float)
            if relative_position is not None
            else raw_target - current
        )
        clipped_delta = self._clip_delta(raw_delta)
        command_target = current + clipped_delta
        return {
            "mode": "vision_gate_clipped",
            "target_position_local_ned_m": command_target.astype(float).tolist(),
            "raw_target_position_local_ned_m": raw_target.astype(float).tolist(),
            "raw_target_position_relative_ned_m": raw_delta.astype(float).tolist(),
            "raw_delta_local_ned_m": raw_delta.astype(float).tolist(),
            "clipped_delta_local_ned_m": clipped_delta.astype(float).tolist(),
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

    def _position_guidance(
        self,
        *,
        current_position_local_ned_m: list[float],
        current_velocity_local_ned_mps: list[float],
        target_position_local_ned_m: list[float],
    ) -> dict[str, Any]:
        config = self.config.position_guidance
        current = np.asarray(current_position_local_ned_m, dtype=float)
        velocity = np.asarray(current_velocity_local_ned_mps, dtype=float)
        target = np.asarray(target_position_local_ned_m, dtype=float)
        error = target - current
        xy_error = error[:2].copy()
        xy_error[np.abs(xy_error) < float(config.position_deadband_m)] = 0.0

        pitch_raw = -float(config.position_kp_deg_per_m) * float(xy_error[0])
        pitch_raw += float(config.velocity_kd_deg_per_mps) * float(velocity[0])
        roll_raw = float(config.position_kp_deg_per_m) * float(xy_error[1])
        roll_raw -= float(config.velocity_kd_deg_per_mps) * float(velocity[1])

        max_tilt = abs(float(config.max_tilt_deg))
        target_pitch = float(np.clip(pitch_raw, -max_tilt, max_tilt))
        target_roll = float(np.clip(roll_raw, -max_tilt, max_tilt))
        return {
            "target_roll_deg": target_roll,
            "target_pitch_deg": target_pitch,
            "raw_target_roll_deg": roll_raw,
            "raw_target_pitch_deg": pitch_raw,
            "position_error_local_ned_m": error.tolist(),
            "velocity_local_ned_mps": velocity.tolist(),
            "target_position_local_ned_m": target.tolist(),
            "config": asdict(config),
            "mapping_note": "positive local X -> negative pitch; positive local Y -> positive roll",
        }

    @staticmethod
    def _sample(vehicle_state: VehicleState) -> dict[str, Any]:
        position = vehicle_state.position_local_ned_m
        velocity = vehicle_state.velocity_local_ned_mps
        attitude = vehicle_state.attitude_quaternion
        if position is None:
            raise ValueError("VehicleState must include position_local_ned_m for body-rate guidance.")
        if velocity is None:
            raise ValueError("VehicleState must include velocity_local_ned_mps for body-rate guidance.")
        if attitude is None:
            raise ValueError("VehicleState must include attitude for body-rate guidance.")
        attitude_list = [float(value) for value in attitude]
        return {
            "sim_time_ns": int(vehicle_state.sim_time_ns),
            "position_local_ned_m": [float(value) for value in position],
            "velocity_local_ned_mps": [float(value) for value in velocity],
            "attitude": attitude_list,
            "euler_deg": BodyRateGuidanceController.euler_deg(attitude_list),
            "reset_count": None,
        }

    @staticmethod
    def euler_deg(quaternion: list[float] | tuple[float, float, float, float]) -> list[float]:
        w, x, y, z = [float(value) for value in quaternion]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return [math.degrees(value) for value in (roll, pitch, yaw)]
