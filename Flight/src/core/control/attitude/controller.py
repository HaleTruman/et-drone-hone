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
        rate_filter_alpha: float = 0.35,
        max_body_rate_rps: float | Iterable[float] | None = None,
        error_quaternion_roll_scale: float = 1.0,
        error_quaternion_pitch_scale: float = 1.0,
        error_quaternion_yaw_scale: float = 1.0,
    ) -> None:
        self.gains = np.array((float(roll_gain), float(pitch_gain), float(yaw_gain)), dtype=float)
        self.damping = float(damping)                     # scalar for simplicity (can be per-axis)
        self.rate_filter_alpha = float(np.clip(float(rate_filter_alpha), 0.0, 1.0))
        self._filtered_rates_frd_rps: np.ndarray | None = None
        self.max_body_rate_rps = self._rate_limits(max_body_rate_rps)
        self.error_quaternion_scales = np.array(
            (
                float(error_quaternion_roll_scale),
                float(error_quaternion_pitch_scale),
                float(error_quaternion_yaw_scale),
            ),
            dtype=float,
        )
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
        filtered_rates = self._filtered_body_rates(current_rates)

        # PD control: Proportional + Damping
        body_rates = self.gains * attitude_error - self.damping * filtered_rates

        # Rate limiting
        if self.max_body_rate_rps is not None:
            body_rates = np.clip(body_rates, -self.max_body_rate_rps, self.max_body_rate_rps)

        # Payload
        payload: dict[str, Any] = {
            "quaternion": tuple(float(v) for v in desired),
            "body_rates_rps": tuple(float(v) for v in body_rates),
        }
        if thrust is not None:
            payload["thrust"] = float(thrust)

        # Convert to the simulator's observed quaternion-error convention.
        q_err_tgt = q_err.tolist()
        q_err_tgt_converted = (q_err_tgt[0], -q_err_tgt[1], q_err_tgt[2], -q_err_tgt[3])
        q_err_tgt_scaled = self._scaled_error_quaternion(q_err_tgt_converted)

        payload["computed_body_rates_rps"] = tuple(float(v) for v in body_rates)
        payload["body_angle_error"] = tuple(attitude_error.tolist())
        payload["error_quaternion"] = tuple(q_err.tolist())
        payload["error_quaternion_target_converted"] = tuple(q_err_tgt_converted)
        payload["error_quaternion_target_scaled"] = q_err_tgt_scaled
        payload["error_quaternion_vector_scales"] = tuple(
            float(value) for value in self.error_quaternion_scales
        )
        payload["filtered_body_rates_frd_rps"] = tuple(filtered_rates.tolist())

        self.last_payload = payload
        return payload

    def _scaled_error_quaternion(
        self, quaternion: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        q = np.asarray(quaternion, dtype=float).copy()
        q[1:4] *= self.error_quaternion_scales
        return tuple(float(value) for value in normalize_quaternion(q))

    def _filtered_body_rates(self, current_rates: np.ndarray) -> np.ndarray:
        alpha = self.rate_filter_alpha
        if self._filtered_rates_frd_rps is None or alpha >= 1.0:
            self._filtered_rates_frd_rps = current_rates.astype(float)
        elif alpha > 0.0:
            self._filtered_rates_frd_rps = (
                (1.0 - alpha) * self._filtered_rates_frd_rps + alpha * current_rates
            )
        return self._filtered_rates_frd_rps.copy()

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
