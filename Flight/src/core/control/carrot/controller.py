import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from autonomy.planning.path_manager import PathManager
from core.control.command_mapper import CommandMapper
from core.coordinates import euler_from_quaternion, quaternion_from_rotation_matrix
from core.schemas import QuatWxyz, Vec3, VehicleState


@dataclass(frozen=True)
class CarrotChaserConfig:
    desired_speed_mps: float = 8.0
    lookahead_time_s: float = 0.45
    min_lookahead_m: float = 2.0
    path_tolerance_m: float = 0.75
    lateral_position_gain: float = 1.4
    velocity_gain: float = 1.8
    max_along_track_acceleration_mps2: float = 4.0
    max_lateral_acceleration_mps2: float = 6.0
    max_vertical_acceleration_mps2: float = 4.0
    mass_kg: float = 1.2
    thrust_coefficient_n: float = 12.0
    gravity_mps2: float = 9.81
    min_thrust: float = 0.20
    max_thrust: float = 0.90


class CarrotChaserController:
    """3D carrot-chasing path follower that emits attitude/thrust targets."""

    def __init__(
        self,
        flight_path: Iterable[Vec3] | PathManager | None = None,
        *,
        config: CarrotChaserConfig | None = None,
        command_mapper: CommandMapper | None = None,
    ):
        self.config = config or CarrotChaserConfig()
        self.command_mapper = command_mapper or CommandMapper()
        self.path_manager = flight_path if isinstance(flight_path, PathManager) else PathManager()
        if flight_path is not None and not isinstance(flight_path, PathManager):
            self.path_manager.set_waypoints(flight_path)

    def set_flight_path(self, flight_path: Iterable[Vec3] | PathManager) -> None:
        if isinstance(flight_path, PathManager):
            self.path_manager = flight_path
            return
        self.path_manager.set_waypoints(flight_path)

    def compute_control(
        self,
        vehicle_state: VehicleState,
        flight_path: Iterable[Vec3] | PathManager | None = None,
    ) -> dict[str, Any]:
        if flight_path is not None:
            self.set_flight_path(flight_path)

        position = _vec3(vehicle_state.position_local_ned_m, "vehicle_state.position_local_ned_m")
        velocity = _vec3(vehicle_state.velocity_local_ned_mps, "vehicle_state.velocity_local_ned_mps")
        lookahead_m = max(
            float(self.config.min_lookahead_m),
            float(self.config.desired_speed_mps) * float(self.config.lookahead_time_s),
        )
        carrot = self.path_manager.carrot_point(tuple(position), lookahead_m)
        carrot_position = _vec3(carrot["position_local_ned_m"], "carrot.position_local_ned_m")
        tangent = _unit(carrot["tangent_local_ned"], "carrot.tangent_local_ned")

        desired_velocity = tangent * float(self.config.desired_speed_mps)
        lateral_error = _lateral_error(position, carrot_position, tangent, float(self.config.path_tolerance_m))
        desired_acceleration = (
            float(self.config.lateral_position_gain) * lateral_error
            + float(self.config.velocity_gain) * (desired_velocity - velocity)
        )
        desired_acceleration = self._limit_acceleration(desired_acceleration, tangent)

        gravity_ned = np.array((0.0, 0.0, float(self.config.gravity_mps2)), dtype=float)
        required_force_ned = float(self.config.mass_kg) * (desired_acceleration - gravity_ned)
        total_thrust_n = float(np.linalg.norm(required_force_ned))
        if total_thrust_n <= 1e-9:
            raise ValueError("required thrust vector cannot be zero")

        yaw_rad = math.atan2(float(tangent[1]), float(tangent[0]))
        if not math.isfinite(yaw_rad):
            yaw_rad = euler_from_quaternion(vehicle_state.attitude_quaternion)[2]
        quaternion = _attitude_for_force(required_force_ned, yaw_rad)
        normalized_thrust = math.sqrt(total_thrust_n / (4.0 * float(self.config.thrust_coefficient_n)))
        normalized_thrust = float(
            np.clip(normalized_thrust, float(self.config.min_thrust), float(self.config.max_thrust))
        )
        command = self.command_mapper.to_attitude_target(np.asarray(quaternion, dtype=float), normalized_thrust)
        command["source"] = "carrot_chaser"
        command["carrot"] = {
            "position_local_ned_m": [float(value) for value in carrot_position],
            "tangent_local_ned": [float(value) for value in tangent],
            "cross_track_error_m": float(carrot["cross_track_error_m"]),
            "path_tolerance_m": float(self.config.path_tolerance_m),
            "lookahead_m": lookahead_m,
            "desired_speed_mps": float(self.config.desired_speed_mps),
        }
        return command

    def _limit_acceleration(self, acceleration_ned: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        along_value = float(
            np.clip(
                float(np.dot(acceleration_ned, tangent)),
                -float(self.config.max_along_track_acceleration_mps2),
                float(self.config.max_along_track_acceleration_mps2),
            )
        )
        along = tangent * along_value
        lateral = acceleration_ned - along
        lateral_norm = float(np.linalg.norm(lateral))
        max_lateral = float(self.config.max_lateral_acceleration_mps2)
        if lateral_norm > max_lateral:
            lateral *= max_lateral / lateral_norm
        limited = along + lateral
        limited[2] = float(
            np.clip(
                limited[2],
                -float(self.config.max_vertical_acceleration_mps2),
                float(self.config.max_vertical_acceleration_mps2),
            )
        )
        return limited


def _lateral_error(position: np.ndarray, target: np.ndarray, tangent: np.ndarray, tolerance_m: float) -> np.ndarray:
    error = target - position
    lateral = error - tangent * float(np.dot(error, tangent))
    norm = float(np.linalg.norm(lateral))
    tolerance = max(0.0, float(tolerance_m))
    if norm <= tolerance:
        return np.zeros(3, dtype=float)
    return lateral * ((norm - tolerance) / norm)


def _attitude_for_force(required_force_ned: np.ndarray, yaw_rad: float) -> QuatWxyz:
    body_z_down_ned = -required_force_ned / np.linalg.norm(required_force_ned)
    yaw_forward_ned = np.array((math.cos(yaw_rad), math.sin(yaw_rad), 0.0), dtype=float)
    body_x_forward_ned = yaw_forward_ned - body_z_down_ned * float(np.dot(yaw_forward_ned, body_z_down_ned))
    x_norm = float(np.linalg.norm(body_x_forward_ned))
    if x_norm <= 1e-9:
        body_x_forward_ned = _orthogonal_horizontal(body_z_down_ned)
    else:
        body_x_forward_ned /= x_norm

    body_y_right_ned = np.cross(body_z_down_ned, body_x_forward_ned)
    body_y_right_ned /= np.linalg.norm(body_y_right_ned)
    body_x_forward_ned = np.cross(body_y_right_ned, body_z_down_ned)
    return quaternion_from_rotation_matrix(np.column_stack((body_x_forward_ned, body_y_right_ned, body_z_down_ned)))


def _orthogonal_horizontal(vector: np.ndarray) -> np.ndarray:
    fallback = np.array((1.0, 0.0, 0.0), dtype=float)
    candidate = fallback - vector * float(np.dot(fallback, vector))
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-9:
        candidate = np.array((0.0, 1.0, 0.0), dtype=float)
        candidate = candidate - vector * float(np.dot(candidate, vector))
        norm = float(np.linalg.norm(candidate))
    return candidate / norm


def _unit(value: Iterable[float], name: str) -> np.ndarray:
    array = _vec3(value, name)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{name} cannot be zero")
    return array / norm


def _vec3(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array
