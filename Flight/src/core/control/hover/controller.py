import math
from typing import Iterable

import numpy as np

from core.control.command_mapper import CommandMapper
from core.coordinates import euler_from_quaternion, quaternion_from_rotation_matrix


class HoverPIDController:
    def __init__(
        self,
        dt_s: float = 1.0 / 30.0,
        position_gain: Iterable[float] = (0.8, 0.8, 1.2),
        velocity_gain: Iterable[float] = (1.1, 1.1, 1.6),
        mass_kg: float = 1.2,
        thrust_coefficient_n: float = 12.0,
        gravity_mps2: float = 9.81,
        max_tilt_deg: float = 10.0,
        max_vertical_acceleration_mps2: float = 2.0,
        min_thrust: float = 0.35,
        max_thrust: float = 0.65,
    ):
        self.dt_s = float(dt_s)
        self.position_gain = np.asarray(tuple(position_gain), dtype=float)
        self.velocity_gain = np.asarray(tuple(velocity_gain), dtype=float)
        self.mass_kg = float(mass_kg)
        self.thrust_coefficient_n = float(thrust_coefficient_n)
        self.gravity_mps2 = float(gravity_mps2)
        self.max_tilt_rad = math.radians(float(max_tilt_deg))
        self.max_vertical_acceleration_mps2 = float(max_vertical_acceleration_mps2)
        self.min_thrust = float(min_thrust)
        self.max_thrust = float(max_thrust)

    def update(
        self,
        position_local_ned_m,
        velocity_local_ned_mps,
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_position_local_ned_m=(0.0, 0.0, 0.0),
        target_velocity_local_ned_mps=(0.0, 0.0, 0.0),
        yaw_rad=None,
    ):
        position = _vec3(position_local_ned_m, "position_local_ned_m")
        velocity = _vec3(velocity_local_ned_mps, "velocity_local_ned_mps")
        target_position = _vec3(target_position_local_ned_m, "target_position_local_ned_m")
        target_velocity = _vec3(target_velocity_local_ned_mps, "target_velocity_local_ned_mps")

        position_error = target_position - position
        velocity_error = target_velocity - velocity
        desired_acceleration_ned = self.position_gain * position_error + self.velocity_gain * velocity_error
        desired_acceleration_ned = self._limit_desired_acceleration(desired_acceleration_ned)

        gravity_ned = np.array([0.0, 0.0, self.gravity_mps2], dtype=float)
        required_force_ned = self.mass_kg * (desired_acceleration_ned - gravity_ned)
        total_thrust_n = float(np.linalg.norm(required_force_ned))
        if total_thrust_n <= 1e-9:
            raise ValueError("required thrust vector cannot be zero")

        command_yaw = euler_from_quaternion(attitude_quaternion)[2] if yaw_rad is None else float(yaw_rad)
        quaternion = _attitude_for_force(required_force_ned, command_yaw)
        normalized_thrust = math.sqrt(total_thrust_n / (4.0 * self.thrust_coefficient_n))
        normalized_thrust = float(np.clip(normalized_thrust, self.min_thrust, self.max_thrust))
        return np.asarray(quaternion, dtype=float), normalized_thrust

    def _limit_desired_acceleration(self, desired_acceleration_ned: np.ndarray) -> np.ndarray:
        limited = np.asarray(desired_acceleration_ned, dtype=float).copy()
        max_horizontal_acceleration = self.gravity_mps2 * math.tan(self.max_tilt_rad)
        horizontal_norm = float(np.linalg.norm(limited[:2]))
        if horizontal_norm > max_horizontal_acceleration:
            limited[:2] *= max_horizontal_acceleration / horizontal_norm
        limited[2] = float(
            np.clip(
                limited[2],
                -self.max_vertical_acceleration_mps2,
                self.max_vertical_acceleration_mps2,
            )
        )
        return limited


class HoverController:
    """State-facing hover controller that emits MAVLink-ready attitude targets."""

    def __init__(
        self,
        *,
        dt_s: float = 1.0 / 30.0,
        target_position_local_ned_m=None,
        target_velocity_local_ned_mps=(0.0, 0.0, 0.0),
        yaw_rad: float | None = None,
        pid_controller: HoverPIDController | None = None,
        command_mapper: CommandMapper | None = None,
    ):
        self.pid_controller = pid_controller or HoverPIDController(dt_s=dt_s)
        self.command_mapper = command_mapper or CommandMapper()
        self.target_position_local_ned_m = (
            None
            if target_position_local_ned_m is None
            else tuple(_vec3(target_position_local_ned_m, "target_position_local_ned_m"))
        )
        self.target_velocity_local_ned_mps = tuple(
            _vec3(target_velocity_local_ned_mps, "target_velocity_local_ned_mps")
        )
        self.yaw_rad = yaw_rad

    def compute_control(self, state):
        if self.target_position_local_ned_m is None:
            self.target_position_local_ned_m = tuple(
                _vec3(state.position_local_ned_m, "state.position_local_ned_m")
            )

        quaternion, thrust = self.pid_controller.update(
            position_local_ned_m=state.position_local_ned_m,
            velocity_local_ned_mps=state.velocity_local_ned_mps,
            attitude_quaternion=state.attitude_quaternion,
            target_position_local_ned_m=self.target_position_local_ned_m,
            target_velocity_local_ned_mps=self.target_velocity_local_ned_mps,
            yaw_rad=self.yaw_rad,
        )
        return self.command_mapper.to_attitude_target(quaternion, thrust)


def _attitude_for_force(required_force_ned: np.ndarray, yaw_rad: float) -> tuple[float, float, float, float]:
    body_z_down_ned = -required_force_ned / np.linalg.norm(required_force_ned)
    yaw_forward_ned = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=float)

    body_x_forward_ned = yaw_forward_ned - body_z_down_ned * float(np.dot(yaw_forward_ned, body_z_down_ned))
    x_norm = float(np.linalg.norm(body_x_forward_ned))
    if x_norm <= 1e-9:
        raise ValueError("yaw heading is parallel to desired body z axis")
    body_x_forward_ned = body_x_forward_ned / x_norm

    body_y_right_ned = np.cross(body_z_down_ned, body_x_forward_ned)
    body_y_right_ned = body_y_right_ned / np.linalg.norm(body_y_right_ned)
    body_x_forward_ned = np.cross(body_y_right_ned, body_z_down_ned)

    body_to_ned = np.column_stack((body_x_forward_ned, body_y_right_ned, body_z_down_ned))
    return quaternion_from_rotation_matrix(body_to_ned)


def _vec3(value, name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array
