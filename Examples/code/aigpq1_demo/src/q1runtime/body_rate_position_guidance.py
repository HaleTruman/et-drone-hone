from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BodyRatePositionGuidanceConfig:
    position_kp_deg_per_m: float = 2.0
    velocity_kd_deg_per_mps: float = 1.5
    max_tilt_deg: float = 4.0
    position_deadband_m: float = 0.05


class BodyRatePositionGuidance:
    """Map local-NED target error onto the discovered body-rate attitude surface."""

    def __init__(self, config: BodyRatePositionGuidanceConfig | None = None):
        self.config = config or BodyRatePositionGuidanceConfig()

    def compute(
        self,
        *,
        current_position_local_ned_m: list[float] | tuple[float, float, float],
        current_velocity_local_ned_mps: list[float] | tuple[float, float, float],
        target_position_local_ned_m: list[float] | tuple[float, float, float],
    ) -> dict[str, Any]:
        current = np.asarray(current_position_local_ned_m, dtype=float)
        velocity = np.asarray(current_velocity_local_ned_mps, dtype=float)
        target = np.asarray(target_position_local_ned_m, dtype=float)
        error = target - current
        xy_error = error[:2].copy()
        xy_error[np.abs(xy_error) < float(self.config.position_deadband_m)] = 0.0

        pitch_raw = float(self.config.position_kp_deg_per_m) * float(xy_error[0])
        pitch_raw -= float(self.config.velocity_kd_deg_per_mps) * float(velocity[0])
        roll_raw = float(self.config.position_kp_deg_per_m) * float(xy_error[1])
        roll_raw -= float(self.config.velocity_kd_deg_per_mps) * float(velocity[1])

        max_tilt = abs(float(self.config.max_tilt_deg))
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
            "config": asdict(self.config),
            "mapping_note": "positive local X -> positive pitch; positive local Y -> positive roll",
        }
