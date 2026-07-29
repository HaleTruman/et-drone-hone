from collections.abc import Iterable
from typing import Any

import numpy as np

from core.coordinates import euler_from_quaternion, normalize_quaternion
from core.schema import VehicleState

class AttitudeController:
    """Convert a desired local-NED attitude quaternion into FRD body rates."""

    def __init__(
        self,
        *,
        roll_gain: float = 1.8,      # slightly higher now that we have damping
        pitch_gain: float = 1.8,
        yaw_gain: float = 0.9,
        damping: float = 0.15,       # new: rate damping gain (tune this!)
        max_body_rate_rps: float | Iterable[float] | None = None,
    ) -> None:
        self.gains = np.array((float(roll_gain), float(pitch_gain), float(yaw_gain)), dtype=float)
        self.damping = float(damping)                     # scalar for simplicity (can be per-axis)
        self.max_body_rate_rps = self._rate_limits(max_body_rate_rps)
        self.last_payload: dict[str, Any] | None = None

    def compute_control(
        self,
        vehicle_state: VehicleState,
        desired_attitude_quaternion: Iterable[float],
        *,
        thrust: float | None = None,
        dt: float | None = None,          # keep for future integral term
    ) -> dict[str, Any]:
        """
        Compute body rates command from desired attitude (Quaternion + Damping).
        """
        current = normalize_quaternion(vehicle_state.attitude_quaternion)
        desired = normalize_quaternion(desired_attitude_quaternion)

        # Error quaternion
        q_err = self._quaternion_multiply(desired, self._quaternion_conjugate(current))
        if q_err[0] < 0.0:
            q_err = -q_err

        # Vector part (robust attitude error)
        error_vec = np.asarray(q_err[1:4], dtype=float)
        attitude_error = 2.0 * error_vec

        # Current body rates (assume available in VehicleState)
        current_rates = np.asarray(vehicle_state.body_rates_frd_rps, dtype=float)

        # PD control: Proportional + Damping
        body_rates = self.gains * attitude_error - self.damping * current_rates

        # Rate limiting
        if self.max_body_rate_rps is not None:
            body_rates = np.clip(body_rates, -self.max_body_rate_rps, self.max_body_rate_rps)

        # Payload
        payload: dict[str, Any] = {
            "body_rates_rps": tuple(float(v) for v in body_rates),
        }
        if thrust is not None:
            payload["thrust"] = float(thrust)

        payload["body_angle_error"] = tuple(attitude_error.tolist())
        payload["error_quaternion"] = tuple(q_err.tolist())

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
