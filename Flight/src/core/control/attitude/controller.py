from collections.abc import Iterable
from typing import Any

import numpy as np

from core.coordinates import euler_from_quaternion, normalize_quaternion
from core.schemas import VehicleState


class AttitudeController:
    """Convert a desired local-NED attitude quaternion into FRD body rates."""

    def __init__(
        self,
        *,
        roll_gain: float = 2.0,
        pitch_gain: float = 2.0,
        yaw_gain: float = 0.2,
        max_body_rate_rps: float | Iterable[float] | None = None,
    ) -> None:
        self.gains = np.array((float(roll_gain), float(pitch_gain), float(yaw_gain)), dtype=float)
        self.max_body_rate_rps = self._rate_limits(max_body_rate_rps)
        self.last_payload: dict[str, Any] | None = None

    def compute_control(
        self,
        vehicle_state: VehicleState,
        desired_attitude_quaternion: Iterable[float],
        *,
        thrust: float | None = None,
    ) -> dict[str, Any]:
        current = normalize_quaternion(vehicle_state.attitude_quaternion)
        desired = normalize_quaternion(desired_attitude_quaternion)

        attitude_error = self._quaternion_multiply(desired, self._quaternion_conjugate(current))
        if attitude_error[0] < 0.0:
            attitude_error = -attitude_error

        body_rotation_error = np.asarray(euler_from_quaternion(attitude_error), dtype=float)
        body_rates = -self.gains * body_rotation_error
        if self.max_body_rate_rps is not None:
            body_rates = np.clip(body_rates, -self.max_body_rate_rps, self.max_body_rate_rps)

        payload: dict[str, Any] = {
            "body_rates_rps": tuple(float(value) for value in body_rates),
        }
        if thrust is not None:
            payload["thrust"] = float(thrust)

        payload["body_angle_error"] = body_rotation_error
        self.last_payload = payload
        return payload

    @staticmethod
    def _rate_limits(max_body_rate_rps: float | Iterable[float] | None) -> np.ndarray | None:
        if max_body_rate_rps is None:
            return None
        if isinstance(max_body_rate_rps, (int, float)):
            value = abs(float(max_body_rate_rps))
            return np.array((value, value, value), dtype=float)

        limits = np.asarray(tuple(max_body_rate_rps), dtype=float)
        if limits.shape != (3,):
            raise ValueError("max_body_rate_rps must be a scalar or three values")
        return np.abs(limits)

    @staticmethod
    def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
        return np.array((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]), dtype=float)

    @staticmethod
    def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dtype=float,
        )
