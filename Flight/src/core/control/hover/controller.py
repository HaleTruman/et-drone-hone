import math
from typing import Any

import numpy as np

from core.control.command_mapper import CommandMapper
from core.coordinates import euler_from_quaternion, quaternion_from_rotation_matrix
from core.schema import VehicleState


class HoverController:
    """VehicleState-facing hover controller that emits attitude and normalized thrust."""

    def __init__(
        self,
        *,
        initial_thrust: float | None = None,
        min_thrust: float = 0.0,
        max_thrust: float = 1.0,
        gravity_mps2: float = 9.80665,
        lateral_velocity_gain: float = 1.0,
        vertical_velocity_gain: float = 0.08,
        vertical_acceleration_gain: float = 0.015,
        hover_thrust_adaptation_gain: float = 0.03,
        nominal_dt_s: float = 0.02,
        max_adaptation_dt_s: float = 0.2,
        command_mapper: CommandMapper | None = None,
    ):
        self.min_thrust = float(min_thrust)
        self.max_thrust = float(max_thrust)
        thrust_midpoint = 0.265 * (self.min_thrust + self.max_thrust)
        self.hover_thrust_estimate = float(
            np.clip(
                thrust_midpoint if initial_thrust is None else float(initial_thrust),
                self.min_thrust,
                self.max_thrust,
            )
        )
        self.gravity_mps2 = float(gravity_mps2)
        self.lateral_velocity_gain = float(lateral_velocity_gain)
        self.vertical_velocity_gain = float(vertical_velocity_gain)
        self.vertical_acceleration_gain = float(vertical_acceleration_gain)
        self.hover_thrust_adaptation_gain = float(hover_thrust_adaptation_gain)
        self.nominal_dt_s = float(nominal_dt_s)
        self.max_adaptation_dt_s = float(max_adaptation_dt_s)
        self.command_mapper = command_mapper or CommandMapper()
        self.last_payload: dict[str, Any] | None = None
        self._last_sim_time_ns: int | None = None

        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if self.min_thrust > self.max_thrust:
            raise ValueError("min_thrust cannot be greater than max_thrust")
        if self.nominal_dt_s <= 0.0:
            raise ValueError("nominal_dt_s must be positive")
        if self.max_adaptation_dt_s <= 0.0:
            raise ValueError("max_adaptation_dt_s must be positive")

    def compute_control(
        self,
        vehicle_state: VehicleState,
        *,
        yaw_rad: float | None = None,
    ) -> dict[str, Any]:
        velocity_ned = self._vec3(
            vehicle_state.velocity_local_ned_mps,
            "vehicle_state.velocity_local_ned_mps",
        )
        acceleration_ned = self._vec3(
            vehicle_state.acceleration_local_ned_mps2,
            "vehicle_state.acceleration_local_ned_mps2",
        )
        yaw = (
            euler_from_quaternion(vehicle_state.attitude_quaternion)[2]
            if yaw_rad is None
            else float(yaw_rad)
        )

        desired_acceleration_ned = np.array(
            (
                -self.lateral_velocity_gain * float(velocity_ned[0]),
                -self.lateral_velocity_gain * float(velocity_ned[1]),
                0.0,
            ),
            dtype=float,
        )

        gravity_ned = np.array((0.0, 0.0, self.gravity_mps2), dtype=float)
        thrust_acceleration_ned = desired_acceleration_ned - gravity_ned
        thrust_acceleration_norm = float(np.linalg.norm(thrust_acceleration_ned))
        if thrust_acceleration_norm <= 1e-9:
            raise ValueError("thrust acceleration vector cannot be zero")

        dt_s = self._sample_dt_s(vehicle_state.sim_time_ns)
        vertical_feedback = (
            self.vertical_velocity_gain * float(velocity_ned[2])
            + self.vertical_acceleration_gain * float(acceleration_ned[2])
        )
        self._adapt_hover_thrust_estimate(vertical_feedback, dt_s)
        tilt_compensation = thrust_acceleration_norm / self.gravity_mps2

        quaternion = np.asarray(
            self._attitude_for_thrust_acceleration(thrust_acceleration_ned, yaw),
            dtype=float,
        )
        thrust = float(
            np.clip(
                self.hover_thrust_estimate * tilt_compensation + vertical_feedback,
                self.min_thrust,
                self.max_thrust,
            )
        )
        payload = self.command_mapper.to_attitude_target(quaternion, thrust)
        payload["source"] = "hover"
        payload["velocity_error_local_ned_mps"] = [
            -float(velocity_ned[0]),
            -float(velocity_ned[1]),
            0.0,
        ]
        payload["acceleration_correction_local_ned_mps2"] = [
            float(value) for value in desired_acceleration_ned
        ]
        payload["thrust_control"] = {
            "hover_thrust_estimate": float(self.hover_thrust_estimate),
            "vertical_feedback": float(vertical_feedback),
            "tilt_compensation": float(tilt_compensation),
            "dt_s": float(dt_s),
        }
        self.last_payload = payload
        return payload

    def _sample_dt_s(self, sim_time_ns: int) -> float:
        sim_time = int(sim_time_ns)
        if self._last_sim_time_ns is None:
            self._last_sim_time_ns = sim_time
            return self.nominal_dt_s

        dt_s = (sim_time - self._last_sim_time_ns) * 1e-9
        self._last_sim_time_ns = sim_time
        if dt_s <= 0.0:
            return self.nominal_dt_s
        return float(min(dt_s, self.max_adaptation_dt_s))

    def _adapt_hover_thrust_estimate(self, vertical_feedback: float, dt_s: float) -> None:
        self.hover_thrust_estimate = float(
            np.clip(
                self.hover_thrust_estimate
                + self.hover_thrust_adaptation_gain * float(vertical_feedback) * float(dt_s),
                self.min_thrust,
                self.max_thrust,
            )
        )

    @staticmethod
    def _attitude_for_thrust_acceleration(
        thrust_acceleration_ned: np.ndarray,
        yaw_rad: float,
    ) -> tuple[float, float, float, float]:
        body_z_down_ned = -thrust_acceleration_ned / np.linalg.norm(thrust_acceleration_ned)
        yaw_forward_ned = np.array((math.cos(yaw_rad), math.sin(yaw_rad), 0.0), dtype=float)
        body_x_forward_ned = yaw_forward_ned - body_z_down_ned * float(
            np.dot(yaw_forward_ned, body_z_down_ned)
        )
        x_norm = float(np.linalg.norm(body_x_forward_ned))
        if x_norm <= 1e-9:
            body_x_forward_ned = HoverController._orthogonal_horizontal(body_z_down_ned)
        else:
            body_x_forward_ned /= x_norm

        body_y_right_ned = np.cross(body_z_down_ned, body_x_forward_ned)
        body_y_right_ned /= np.linalg.norm(body_y_right_ned)
        body_x_forward_ned = np.cross(body_y_right_ned, body_z_down_ned)
        rotation = np.column_stack((body_x_forward_ned, body_y_right_ned, body_z_down_ned))
        return quaternion_from_rotation_matrix(rotation)

    @staticmethod
    def _orthogonal_horizontal(vector: np.ndarray) -> np.ndarray:
        candidate = np.array((1.0, 0.0, 0.0), dtype=float)
        candidate = candidate - vector * float(np.dot(candidate, vector))
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-9:
            candidate = np.array((0.0, 1.0, 0.0), dtype=float)
            candidate = candidate - vector * float(np.dot(candidate, vector))
            norm = float(np.linalg.norm(candidate))
        return candidate / norm

    @staticmethod
    def _vec3(value: tuple[float, float, float], name: str) -> np.ndarray:
        array = np.asarray(tuple(value), dtype=float)
        if array.shape != (3,):
            raise ValueError(f"{name} must contain exactly three values")
        return array
