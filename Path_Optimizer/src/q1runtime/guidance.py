from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .target_mapper import LocalVisionTarget


@dataclass(frozen=True)
class VelocityPlannerConfig:
    max_speed_mps: float = 0.5
    max_vertical_speed_mps: float = 0.0
    approach_gain_hz: float = 0.8
    arrival_radius_m: float = 0.35
    vertical_mode: str = "hold"
    max_accel_mps2: float = 1.0


class VelocityPlanner:
    """Converts a tracked local-NED target into a bounded desired velocity."""

    def __init__(self, config: VelocityPlannerConfig | None = None):
        self.config = config or VelocityPlannerConfig()
        self._last_velocity = np.zeros(3, dtype=float)
        self._last_time_s: float | None = None

    def reset(self) -> None:
        self._last_velocity[:] = 0.0
        self._last_time_s = None

    def desired_velocity(self, target: LocalVisionTarget, telemetry: Any, *, now_s: float | None = None) -> np.ndarray:
        current_position = getattr(telemetry, "position_local_ned_m", None)
        if current_position is None:
            raise ValueError("Telemetry must include position_local_ned_m for velocity planning.")
        current = np.asarray(current_position, dtype=float)
        target_position = np.asarray(target.position_local_ned_m, dtype=float)
        delta = target_position - current
        if self.config.vertical_mode == "hold":
            delta[2] = 0.0
        elif self.config.vertical_mode == "target":
            pass
        else:
            raise ValueError("vertical_mode must be 'hold' or 'target'")

        distance = float(np.linalg.norm(delta))
        if distance <= max(self.config.arrival_radius_m, 1e-9):
            desired = np.zeros(3, dtype=float)
        else:
            direction = delta / distance
            proportional_speed = max(0.0, distance - self.config.arrival_radius_m) * self.config.approach_gain_hz
            speed = min(float(self.config.max_speed_mps), proportional_speed)
            desired = direction * speed
        desired[2] = float(
            np.clip(desired[2], -float(self.config.max_vertical_speed_mps), float(self.config.max_vertical_speed_mps))
        )
        return self._limit_acceleration(desired, now_s=now_s)

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                "max_speed_mps": self.config.max_speed_mps,
                "max_vertical_speed_mps": self.config.max_vertical_speed_mps,
                "approach_gain_hz": self.config.approach_gain_hz,
                "arrival_radius_m": self.config.arrival_radius_m,
                "vertical_mode": self.config.vertical_mode,
                "max_accel_mps2": self.config.max_accel_mps2,
            },
            "last_velocity_local_ned_mps": self._last_velocity.tolist(),
            "last_time_s": self._last_time_s,
        }

    def _limit_acceleration(self, desired: np.ndarray, *, now_s: float | None) -> np.ndarray:
        if self.config.max_accel_mps2 <= 0.0:
            self._last_velocity = desired.copy()
            self._last_time_s = now_s
            return desired
        if now_s is None or self._last_time_s is None:
            self._last_velocity = desired.copy()
            self._last_time_s = now_s
            return desired
        dt = max(0.0, float(now_s) - float(self._last_time_s))
        max_delta = float(self.config.max_accel_mps2) * dt
        delta = desired - self._last_velocity
        norm = float(np.linalg.norm(delta))
        if norm > max_delta > 0.0:
            desired = self._last_velocity + (delta / norm) * max_delta
        self._last_velocity = desired.copy()
        self._last_time_s = now_s
        return desired
