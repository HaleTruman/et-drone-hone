from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .target_mapper import LocalVisionTarget


@dataclass(frozen=True)
class EmittedCommand:
    target: LocalVisionTarget | None
    payload: dict[str, Any]


class CommandEmitter:
    """Builds and emits MAVLink local-NED position target payloads."""

    def __init__(
        self,
        bridge: Any,
        *,
        command_mode: str = "velocity",
        max_approach_speed_mps: float = 2.0,
        max_vertical_speed_mps: float = 0.8,
        approach_gain_hz: float = 0.8,
        arrival_radius_m: float = 0.35,
        dry_run: bool = False,
    ):
        self.bridge = bridge
        self.command_mode = str(command_mode)
        self.max_approach_speed_mps = float(max_approach_speed_mps)
        self.max_vertical_speed_mps = float(max_vertical_speed_mps)
        self.approach_gain_hz = float(approach_gain_hz)
        self.arrival_radius_m = float(arrival_radius_m)
        self.dry_run = bool(dry_run)

    def build_position_target(self, target: LocalVisionTarget, telemetry: Any) -> dict[str, Any]:
        current_position = getattr(telemetry, "position_local_ned_m", None)
        if current_position is None:
            raise ValueError("Telemetry must include position_local_ned_m to build a position target.")
        velocity = self.approach_velocity(
            current_position_local_ned_m=current_position,
            target_position_local_ned_m=target.position_local_ned_m,
        )
        payload = {
            "velocity_local_ned_mps": [float(value) for value in velocity],
            "yaw_rad": None,
            "source": "q1runtime_top1_regressor_passthrough",
            "vision_frame_id": int(target.frame_id),
            "vision_sim_time_ns": int(target.sim_time_ns),
            "gate_id": target.gate_id,
            "position_confidence": float(target.position_confidence),
            "target_position_local_ned_m": [float(value) for value in target.position_local_ned_m],
            "target_position_camera_m": [float(value) for value in target.position_camera_m],
        }
        if self.command_mode == "position_velocity":
            payload["position_local_ned_m"] = [float(value) for value in target.position_local_ned_m]
            payload["position_axes"] = [True, True, True]
            payload["yaw_rad"] = float(target.yaw_rad)
        elif self.command_mode != "velocity":
            raise ValueError("command_mode must be 'velocity' or 'position_velocity'")
        return payload

    def build_velocity_target(
        self,
        velocity_local_ned_mps: np.ndarray | tuple[float, float, float],
        *,
        target: LocalVisionTarget | None = None,
        source: str = "q1runtime_velocity",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "velocity_local_ned_mps": [float(value) for value in np.asarray(velocity_local_ned_mps, dtype=float)],
            "yaw_rad": None,
            "source": source,
        }
        if target is not None:
            payload.update(
                {
                    "vision_frame_id": int(target.frame_id),
                    "vision_sim_time_ns": int(target.sim_time_ns),
                    "gate_id": target.gate_id,
                    "position_confidence": float(target.position_confidence),
                    "target_position_local_ned_m": [float(value) for value in target.position_local_ned_m],
                    "target_position_camera_m": [float(value) for value in target.position_camera_m],
                }
            )
        if extra:
            payload.update(extra)
        return payload

    def build_stop_target(self) -> dict[str, Any]:
        return {
            "velocity_local_ned_mps": [0.0, 0.0, 0.0],
            "yaw_rad": None,
            "source": "q1runtime_stop_no_active_target",
        }

    def emit(self, target: LocalVisionTarget, telemetry: Any) -> EmittedCommand:
        payload = self.build_position_target(target, telemetry)
        if not self.dry_run:
            self.bridge.send_position_target(payload)
        return EmittedCommand(target=target, payload=payload)

    def emit_velocity(
        self,
        velocity_local_ned_mps: np.ndarray | tuple[float, float, float],
        *,
        target: LocalVisionTarget | None = None,
        source: str = "q1runtime_velocity",
        extra: dict[str, Any] | None = None,
    ) -> EmittedCommand:
        payload = self.build_velocity_target(velocity_local_ned_mps, target=target, source=source, extra=extra)
        if not self.dry_run:
            self.bridge.send_position_target(payload)
        return EmittedCommand(target=target, payload=payload)

    def emit_stop(self) -> EmittedCommand:
        payload = self.build_stop_target()
        if not self.dry_run:
            self.bridge.send_position_target(payload)
        return EmittedCommand(target=None, payload=payload)

    def approach_velocity(
        self,
        *,
        current_position_local_ned_m: np.ndarray | tuple[float, float, float],
        target_position_local_ned_m: np.ndarray | tuple[float, float, float],
    ) -> np.ndarray:
        current = np.asarray(current_position_local_ned_m, dtype=float)
        target = np.asarray(target_position_local_ned_m, dtype=float)
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance <= max(self.arrival_radius_m, 1e-9):
            return np.zeros(3, dtype=float)
        direction = delta / distance
        proportional_speed = max(0.0, distance - self.arrival_radius_m) * self.approach_gain_hz
        speed = min(self.max_approach_speed_mps, proportional_speed)
        velocity = direction * speed
        velocity[2] = float(np.clip(velocity[2], -self.max_vertical_speed_mps, self.max_vertical_speed_mps))
        return velocity

    def snapshot(self) -> dict[str, Any]:
        return {
            "command_mode": self.command_mode,
            "max_approach_speed_mps": self.max_approach_speed_mps,
            "max_vertical_speed_mps": self.max_vertical_speed_mps,
            "approach_gain_hz": self.approach_gain_hz,
            "arrival_radius_m": self.arrival_radius_m,
            "dry_run": self.dry_run,
        }
