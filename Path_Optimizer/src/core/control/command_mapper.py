from typing import Any

import numpy as np


class CommandMapper:
    def to_velocity_target(
        self,
        velocity_local_ned_mps: np.ndarray,
        yaw_rad: float | None = None,
    ) -> dict[str, Any]:
        return {
            "velocity_local_ned_mps": np.asarray(velocity_local_ned_mps, dtype=float).tolist(),
            "yaw_rad": None if yaw_rad is None else float(yaw_rad),
        }

    def to_position_target(
        self,
        position_local_ned_m: np.ndarray,
        velocity_local_ned_mps: np.ndarray,
        yaw_rad: float,
    ) -> dict[str, Any]:
        return {
            "position_local_ned_m": np.asarray(position_local_ned_m, dtype=float).tolist(),
            "velocity_local_ned_mps": np.asarray(velocity_local_ned_mps, dtype=float).tolist(),
            "yaw_rad": float(yaw_rad),
        }

    def to_attitude_target(
        self,
        quaternion: np.ndarray,
        thrust: float,
        body_rates_rps: np.ndarray | None = None,
    ) -> dict[str, Any]:
        payload = {
            "quaternion": np.asarray(quaternion, dtype=float).tolist(),
            "thrust": self.scale_thrust(thrust),
        }
        if body_rates_rps is not None:
            payload["body_rates_rps"] = np.asarray(body_rates_rps, dtype=float).tolist()
        return payload

    def scale_thrust(self, thrust: float) -> float:
        return float(np.clip(thrust, 0.0, 1.0))
