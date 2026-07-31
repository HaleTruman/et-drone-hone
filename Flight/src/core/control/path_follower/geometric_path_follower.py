"""Geometric path follower for local-NED quadrotor racing paths.

The controller consumes the runtime :class:`core.schema.VehicleState` and an
:class:`autonomy.pathing.PathManager`.
It emits the repository's standard attitude-target payload:

    {"quaternion": [w, x, y, z], "thrust": normalized_thrust, ...}

Coordinate conventions:
- Inertial: local NED, +z down, gravity = [0, 0, +g]
- Body: FRD, +z down, positive collective thrust accelerates along -body_z
- Quaternion: scalar-first [w, x, y, z], body-to-local-NED rotation
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import math

import numpy as np

from autonomy.pathing import PathManager
from core.control.command_mapper import CommandMapper
from core.coordinates import quaternion_from_rotation_matrix, GRAVITY_MPS2
from core.schema import QuatWxyz, VehicleState


class GeometricPathFollower:
    """High-level path follower that produces attitude and normalized thrust."""

    def __init__(
        self,
        path_manager: PathManager,
        *,
        kp_cross: float = 1.2,
        kd_cross: float = 1.4,
        kp_speed: float = 0.45,
        acceleration_filter_alpha: float = 0.25,
        lookahead_m: float = 3.0,
        speed_lookahead_m: float = 10.0,
        v_max: float = 12.0,
        a_lat_max: float = 8.0,
        v_min: float = 1.5,
        curvature_speed_deadband: float = 0.04,
        curvature_speed_ramp: float = 0.08,
        curvature_feedforward_gain: float = 1.0,
        curvature_feedforward_max_acceleration_mps2: float = 8.0,
        hover_thrust: float = 0.265,
        max_commanded_acceleration_mps2: float = 6.0,
        max_upward_acceleration_mps2: float = 8.0,
        max_downward_acceleration_mps2: float = 3.0,
        max_tilt_deg: float = 60.0,
        min_normalized_thrust: float = 0.05,
        max_normalized_thrust: float = 1.0,
        drag_coeff: float = 0.0,
        gravity_mps2: float = GRAVITY_MPS2,
        command_mapper: CommandMapper | None = None,
    ) -> None:
        self.path_manager = path_manager
        self.kp_cross = float(kp_cross)
        self.kd_cross = float(kd_cross)
        self.kp_speed = float(kp_speed)
        self.acceleration_filter_alpha = float(acceleration_filter_alpha)
        self.lookahead_m = float(lookahead_m)
        self.speed_lookahead_m = float(speed_lookahead_m)
        self.v_max = float(v_max)
        self.a_lat_max = float(a_lat_max)
        self.v_min = float(v_min)
        self.curvature_speed_deadband = float(curvature_speed_deadband)
        self.curvature_speed_ramp = float(curvature_speed_ramp)
        self.curvature_feedforward_gain = float(curvature_feedforward_gain)
        self.curvature_feedforward_max_acceleration_mps2 = float(
            curvature_feedforward_max_acceleration_mps2
        )
        self.hover_thrust = float(hover_thrust)
        self.max_commanded_acceleration_mps2 = float(max_commanded_acceleration_mps2)
        self.max_upward_acceleration_mps2 = float(max_upward_acceleration_mps2)
        self.max_downward_acceleration_mps2 = float(max_downward_acceleration_mps2)
        self.max_tilt_rad = math.radians(float(max_tilt_deg))
        self.min_normalized_thrust = float(min_normalized_thrust)
        self.max_normalized_thrust = float(max_normalized_thrust)
        self.drag_coeff = float(drag_coeff)
        self.gravity_ned = np.array((0.0, 0.0, float(gravity_mps2)), dtype=float)
        self.command_mapper = command_mapper or CommandMapper()
        self.last_payload: dict[str, Any] | None = None
        self._filtered_acceleration_ned: np.ndarray | None = None

        if gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        self._validate_gains()

    def set_path_manager(self, path_manager: PathManager) -> None:
        self.path_manager = path_manager

    def compute_control(self, vehicle_state: VehicleState) -> dict[str, Any]:
        position = np.asarray(vehicle_state.position_local_ned_m, dtype=float)
        velocity = np.asarray(vehicle_state.velocity_local_ned_mps, dtype=float)

        # Project the drone onto the path, then sample a lookahead point used for heading.
        projection = self.path_manager.project(vehicle_state.position_local_ned_m)
        preview = self.path_manager.carrot_point(
            vehicle_state.position_local_ned_m,
            self.lookahead_m,
            self.speed_lookahead_m,
        )

        closest = np.asarray(projection.closest_point_local_ned_m, dtype=float)
        tangent = _unit(projection.tangent_local_ned, "projection.tangent_local_ned")
        heading = _unit(preview["tangent_local_ned"], "preview.tangent_local_ned")
        curvature = _nonnegative_float(
            preview.get("max_curvature_ahead", preview.get("curvature", 0.0)),
            "preview.max_curvature_ahead",
        )

        # Split position and velocity error into along-path and cross-path components.
        position_error = position - closest
        along_track_error = tangent * float(np.dot(position_error, tangent))
        cross_track_error = position_error - along_track_error

        along_track_speed_mps = float(np.dot(velocity, tangent))
        cross_track_velocity = velocity - along_track_speed_mps * tangent
        commanded_speed_mps = self._curvature_speed_mps(curvature)
        curvature_feedforward = self._curvature_feedforward_acceleration(
            tangent=tangent,
            heading=heading,
            along_track_speed_mps=along_track_speed_mps,
            preview_distance_m=float(preview["along_track_m"]) - float(projection.along_track_m),
        )

        speed_acceleration_mps2 = self.kp_speed * (
            commanded_speed_mps - along_track_speed_mps
        )
        # Cross-track feedback pulls back to the path; speed feedback pushes along it.
        desired_acceleration = (
            -self.kp_cross * cross_track_error
            - self.kd_cross * cross_track_velocity
            + speed_acceleration_mps2 * tangent
            + curvature_feedforward
        )

        if self.drag_coeff > 0.0:
            speed_mps = float(np.linalg.norm(velocity))
            desired_acceleration = desired_acceleration + self.drag_coeff * speed_mps * velocity

        desired_acceleration = self._filtered_acceleration(
            self._limited_acceleration(desired_acceleration)
        )
        desired_acceleration = self._tilt_limited_acceleration(
            self._vertical_limited_acceleration(desired_acceleration)
        )

        # Convert the desired inertial acceleration into attitude plus normalized thrust.
        thrust_acceleration_ned = desired_acceleration - self.gravity_ned
        thrust_norm_mps2 = float(np.linalg.norm(thrust_acceleration_ned))
        if thrust_norm_mps2 <= 1e-9:
            raise ValueError("thrust acceleration vector cannot be zero")

        body_z_down_ned = -thrust_acceleration_ned / thrust_norm_mps2
        heading_horizontal = heading.copy()
        heading_horizontal[2] = 0.0
        if float(np.linalg.norm(heading_horizontal)) <= 1e-9:
            heading_horizontal = np.array((1.0, 0.0, 0.0), dtype=float)
        quaternion = np.asarray(
            _attitude_from_body_z_and_heading(body_z_down_ned, heading_horizontal),
            dtype=float,
        )

        thrust = float(
            np.clip(
                self.hover_thrust * thrust_norm_mps2 / self.gravity_ned[2],
                self.min_normalized_thrust,
                self.max_normalized_thrust,
            )
        )

        payload = self.command_mapper.to_attitude_target(quaternion, thrust)
        payload["source"] = "geometric_path_follower"
        payload["path_follower"] = {
            "projection_closest_point_local_ned_m": [float(value) for value in closest],
            "projection_along_track_m": float(projection.along_track_m),
            "preview_position_local_ned_m": [
                float(value) for value in preview["position_local_ned_m"]
            ],
            "preview_along_track_m": float(preview["along_track_m"]),
            "cross_track_error_m": float(np.linalg.norm(cross_track_error)),
            "cross_track_error_local_ned_m": [float(value) for value in cross_track_error],
            "kp_cross": float(self.kp_cross),
            "kd_cross": float(self.kd_cross),
            "along_track_speed_mps": float(along_track_speed_mps),
            "commanded_speed_mps": float(commanded_speed_mps),
            "speed_acceleration_mps2": float(speed_acceleration_mps2),
            "curvature": float(preview.get("curvature", curvature)),
            "max_curvature_ahead": float(curvature),
            "curvature_feedforward_gain": float(self.curvature_feedforward_gain),
            "curvature_feedforward_max_acceleration_mps2": float(
                self.curvature_feedforward_max_acceleration_mps2
            ),
            "curvature_feedforward_acceleration_local_ned_mps2": [
                float(value) for value in curvature_feedforward
            ],
            "curvature_feedforward_acceleration_mps2": float(
                np.linalg.norm(curvature_feedforward)
            ),
            "curvature_speed_deadband": float(self.curvature_speed_deadband),
            "curvature_speed_ramp": float(self.curvature_speed_ramp),
            "acceleration_filter_alpha": float(self.acceleration_filter_alpha),
            "max_upward_acceleration_mps2": float(self.max_upward_acceleration_mps2),
            "max_downward_acceleration_mps2": float(self.max_downward_acceleration_mps2),
            "max_tilt_deg": float(math.degrees(self.max_tilt_rad)),
        }
        payload["desired_acceleration_local_ned_mps2"] = [
            float(value) for value in desired_acceleration
        ]
        payload["thrust_control"] = {
            "mode": "geometric_dynamic_inversion",
            "specific_thrust_mps2": float(thrust_norm_mps2),
            "hover_thrust": float(self.hover_thrust),
        }

        self.last_payload = payload
        return payload

    def _curvature_speed_mps(self, curvature: float) -> float:
        if curvature <= max(1e-9, self.curvature_speed_deadband):
            return float(self.v_max)

        curvature_limited = math.sqrt(self.a_lat_max / curvature)
        curvature_limited = float(np.clip(curvature_limited, self.v_min, self.v_max))
        if curvature_limited >= self.v_max:
            return float(self.v_max)

        excess_curvature = curvature - self.curvature_speed_deadband
        blend = 1.0 - math.exp(-excess_curvature / self.curvature_speed_ramp)
        blend = float(np.clip(blend, 0.0, 1.0))
        return float(self.v_max - blend * (self.v_max - curvature_limited))

    def _curvature_feedforward_acceleration(
        self,
        *,
        tangent: np.ndarray,
        heading: np.ndarray,
        along_track_speed_mps: float,
        preview_distance_m: float,
    ) -> np.ndarray:
        gain = float(self.curvature_feedforward_gain)
        if gain <= 0.0:
            return np.zeros(3, dtype=float)

        preview_distance = max(1e-6, float(preview_distance_m))
        dot = float(np.clip(np.dot(tangent, heading), -1.0, 1.0))
        turn_angle_rad = math.acos(dot)
        if turn_angle_rad <= 1e-9:
            return np.zeros(3, dtype=float)

        turn_normal = heading - dot * tangent
        normal_norm = float(np.linalg.norm(turn_normal))
        if normal_norm <= 1e-9:
            return np.zeros(3, dtype=float)
        turn_normal = turn_normal / normal_norm

        speed_mps = max(0.0, float(along_track_speed_mps))
        preview_curvature = turn_angle_rad / preview_distance
        acceleration = gain * speed_mps * speed_mps * preview_curvature * turn_normal

        limit = float(self.curvature_feedforward_max_acceleration_mps2)
        acceleration_norm = float(np.linalg.norm(acceleration))
        if limit > 0.0 and acceleration_norm > limit:
            acceleration = acceleration * (limit / acceleration_norm)
        return acceleration

    def _limited_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        limit = float(self.max_commanded_acceleration_mps2)
        if limit <= 0.0:
            return acceleration
        norm = float(np.linalg.norm(acceleration))
        if norm <= limit or norm <= 1e-12:
            return acceleration
        return acceleration * (limit / norm)

    def _vertical_limited_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        limited = acceleration.copy()
        limited[2] = float(
            np.clip(
                limited[2],
                -self.max_upward_acceleration_mps2,
                self.max_downward_acceleration_mps2,
            )
        )
        return limited

    def _tilt_limited_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        if self.max_tilt_rad <= 0.0:
            limited = acceleration.copy()
            limited[0:2] = 0.0
            return limited

        horizontal = acceleration[0:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= 1e-12:
            return acceleration

        upward_thrust_acceleration_mps2 = max(
            1e-6,
            float(self.gravity_ned[2] - acceleration[2]),
        )
        max_horizontal_mps2 = upward_thrust_acceleration_mps2 * math.tan(self.max_tilt_rad)
        if horizontal_norm <= max_horizontal_mps2:
            return acceleration

        limited = acceleration.copy()
        limited[0:2] = horizontal * (max_horizontal_mps2 / horizontal_norm)
        return limited

    def _validate_gains(self) -> None:
        if self.v_min < 0.0:
            raise ValueError("v_min cannot be negative")
        if self.v_max < self.v_min:
            raise ValueError("v_max cannot be less than v_min")
        if self.a_lat_max <= 0.0:
            raise ValueError("a_lat_max must be positive")
        if self.curvature_speed_deadband < 0.0:
            raise ValueError("curvature_speed_deadband cannot be negative")
        if self.curvature_speed_ramp <= 0.0:
            raise ValueError("curvature_speed_ramp must be positive")
        if self.curvature_feedforward_gain < 0.0:
            raise ValueError("curvature_feedforward_gain cannot be negative")
        if self.curvature_feedforward_max_acceleration_mps2 < 0.0:
            raise ValueError("curvature_feedforward_max_acceleration_mps2 cannot be negative")
        if self.hover_thrust <= 0.0:
            raise ValueError("hover_thrust must be positive")
        if self.max_upward_acceleration_mps2 < 0.0:
            raise ValueError("max_upward_acceleration_mps2 cannot be negative")
        if self.max_downward_acceleration_mps2 < 0.0:
            raise ValueError("max_downward_acceleration_mps2 cannot be negative")
        if not 0.0 <= self.max_tilt_rad < math.pi / 2.0:
            raise ValueError("max_tilt_deg must be in [0, 90)")
        if not 0.0 <= self.acceleration_filter_alpha <= 1.0:
            raise ValueError("acceleration_filter_alpha must be in [0, 1]")
        if self.min_normalized_thrust > self.max_normalized_thrust:
            raise ValueError("min_normalized_thrust cannot exceed max_normalized_thrust")

    def _filtered_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        alpha = float(self.acceleration_filter_alpha)
        if alpha >= 1.0 or self._filtered_acceleration_ned is None:
            self._filtered_acceleration_ned = acceleration.astype(float)
        elif alpha > 0.0:
            self._filtered_acceleration_ned = (
                (1.0 - alpha) * self._filtered_acceleration_ned + alpha * acceleration
            )
        else:
            return self._filtered_acceleration_ned
        return self._filtered_acceleration_ned.copy()


def _attitude_from_body_z_and_heading(
    body_z_down_ned: np.ndarray,
    heading_ned: np.ndarray,
) -> QuatWxyz:
    z_body = _unit(body_z_down_ned, "body_z_down_ned")
    x_projected = heading_ned - z_body * float(np.dot(heading_ned, z_body))
    x_norm = float(np.linalg.norm(x_projected))
    if x_norm <= 1e-9:
        x_body = _orthogonal_horizontal(z_body)
    else:
        x_body = x_projected / x_norm

    y_body = np.cross(z_body, x_body)
    y_body /= np.linalg.norm(y_body)
    x_body = np.cross(y_body, z_body)
    return quaternion_from_rotation_matrix(np.column_stack((x_body, y_body, z_body)))


def _orthogonal_horizontal(vector: np.ndarray) -> np.ndarray:
    candidate = np.array((1.0, 0.0, 0.0), dtype=float)
    candidate = candidate - vector * float(np.dot(candidate, vector))
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-9:
        candidate = np.array((0.0, 1.0, 0.0), dtype=float)
        candidate = candidate - vector * float(np.dot(candidate, vector))
        norm = float(np.linalg.norm(candidate))
    return candidate / norm


def _unit(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{name} cannot be zero")
    return array / norm


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return max(0.0, number)
