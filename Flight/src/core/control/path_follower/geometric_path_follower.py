"""Geometric path follower for local-NED quadrotor racing paths.

The controller consumes the runtime :class:`core.schema.VehicleState` and a
path provider with the same projection API as :class:`autonomy.pathing.PathManager`.
It emits the repository's standard attitude-target payload:

    {"quaternion": [w, x, y, z], "thrust": normalized_thrust, ...}

Coordinate conventions:
- Inertial: local NED, +z down, gravity = [0, 0, +g]
- Body: FRD, +z down, positive collective thrust accelerates along -body_z
- Quaternion: scalar-first [w, x, y, z], body-to-local-NED rotation
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol
import math

import numpy as np

from core.control.command_mapper import CommandMapper
from core.coordinates import quaternion_from_rotation_matrix
from core.schema import QuatWxyz, Vec3, VehicleState


class PathProvider(Protocol):
    """Projection/sampling surface used by PathManager."""

    def project(self, position_local_ned_m: Vec3) -> dict[str, Any]: ...

    def carrot_point(
        self,
        position_local_ned_m: Vec3,
        lookahead_m: float,
        speed_lookahead_m: float | None = None,
    ) -> dict[str, Any]: ...


@dataclass
class PathFollowerGains:
    # Cross-track position / velocity feedback.
    kp_cross: float = 1.2
    kd_cross: float = 1.4

    # Along-track speed regulation.
    kp_speed: float = 0.45

    # Command smoothing. This runs on the outer-loop command updates.
    acceleration_filter_alpha: float = 0.25

    # Preview distances used for path heading and curvature-limited speed.
    lookahead_m: float = 3.0
    speed_lookahead_m: float = 10.0

    # Speed planner.
    v_max: float = 12.0
    a_lat_max: float = 8.0
    v_min: float = 1.5
    curvature_speed_deadband: float = 0.04
    curvature_speed_ramp: float = 0.08

    # Proactive turn acceleration from path curvature preview. This banks/pitches
    # before cross-track error builds up.
    curvature_feedforward_gain: float = 1.0
    curvature_feedforward_max_acceleration_mps2: float = 8.0

    # Dynamic inversion / actuator limits.
    hover_thrust: float = 0.265
    max_commanded_acceleration_mps2: float = 6.0
    min_normalized_thrust: float = 0.05
    max_normalized_thrust: float = 1.0

    # Optional simple quadratic drag compensation. Set to 0 to disable.
    drag_coeff: float = 0.0


class GeometricPathFollower:
    """High-level path follower that produces attitude and normalized thrust."""

    def __init__(
        self,
        path_provider: PathProvider,
        gains: PathFollowerGains | None = None,
        *,
        gravity_mps2: float = 9.81,
        command_mapper: CommandMapper | None = None,
    ) -> None:
        self.path_provider = path_provider
        self.gains = gains or PathFollowerGains()
        self.gravity_ned = np.array((0.0, 0.0, float(gravity_mps2)), dtype=float)
        self.command_mapper = command_mapper or CommandMapper()
        self.last_payload: dict[str, Any] | None = None
        self._filtered_acceleration_ned: np.ndarray | None = None

        if gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        self._validate_gains()

    def set_path_provider(self, path_provider: PathProvider) -> None:
        self.path_provider = path_provider

    def compute_control(self, vehicle_state: VehicleState) -> dict[str, Any]:
        position = _vec3(vehicle_state.position_local_ned_m, "position_local_ned_m")
        velocity = _vec3(vehicle_state.velocity_local_ned_mps, "velocity_local_ned_mps")

        projection = self.path_provider.project(vehicle_state.position_local_ned_m)
        preview = self.path_provider.carrot_point(
            vehicle_state.position_local_ned_m,
            self.gains.lookahead_m,
            self.gains.speed_lookahead_m,
        )

        closest = _vec3(projection["position_local_ned_m"], "projection.position_local_ned_m")
        tangent = _unit(projection["tangent_local_ned"], "projection.tangent_local_ned")
        heading = _unit(preview["tangent_local_ned"], "preview.tangent_local_ned")
        curvature = _nonnegative_float(
            preview.get("max_curvature_ahead", preview.get("curvature", 0.0)),
            "preview.max_curvature_ahead",
        )

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
            preview_distance_m=float(preview["along_track_m"]) - float(projection["along_track_m"]),
        )

        speed_acceleration_mps2 = self.gains.kp_speed * (
            commanded_speed_mps - along_track_speed_mps
        )
        desired_acceleration = (
            -self.gains.kp_cross * cross_track_error
            - self.gains.kd_cross * cross_track_velocity
            + speed_acceleration_mps2 * tangent
            + curvature_feedforward
        )

        if self.gains.drag_coeff > 0.0:
            speed_mps = float(np.linalg.norm(velocity))
            desired_acceleration = desired_acceleration + self.gains.drag_coeff * speed_mps * velocity

        desired_acceleration = self._filtered_acceleration(
            self._limited_acceleration(desired_acceleration)
        )

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
                self.gains.hover_thrust * thrust_norm_mps2 / self.gravity_ned[2],
                self.gains.min_normalized_thrust,
                self.gains.max_normalized_thrust,
            )
        )

        payload = self.command_mapper.to_attitude_target(quaternion, thrust)
        payload["source"] = "geometric_path_follower"
        payload["path_follower"] = {
            "projection_position_local_ned_m": [float(value) for value in closest],
            "projection_along_track_m": float(projection["along_track_m"]),
            "preview_position_local_ned_m": [
                float(value) for value in _vec3(preview["position_local_ned_m"], "preview.position_local_ned_m")
            ],
            "preview_along_track_m": float(preview["along_track_m"]),
            "cross_track_error_m": float(np.linalg.norm(cross_track_error)),
            "cross_track_error_local_ned_m": [float(value) for value in cross_track_error],
            "kp_cross": float(self.gains.kp_cross),
            "kd_cross": float(self.gains.kd_cross),
            "along_track_speed_mps": float(along_track_speed_mps),
            "commanded_speed_mps": float(commanded_speed_mps),
            "speed_acceleration_mps2": float(speed_acceleration_mps2),
            "curvature": float(preview.get("curvature", curvature)),
            "max_curvature_ahead": float(curvature),
            "curvature_feedforward_gain": float(self.gains.curvature_feedforward_gain),
            "curvature_feedforward_max_acceleration_mps2": float(
                self.gains.curvature_feedforward_max_acceleration_mps2
            ),
            "curvature_feedforward_acceleration_local_ned_mps2": [
                float(value) for value in curvature_feedforward
            ],
            "curvature_feedforward_acceleration_mps2": float(
                np.linalg.norm(curvature_feedforward)
            ),
            "curvature_speed_deadband": float(self.gains.curvature_speed_deadband),
            "curvature_speed_ramp": float(self.gains.curvature_speed_ramp),
            "acceleration_filter_alpha": float(self.gains.acceleration_filter_alpha),
        }
        payload["desired_acceleration_local_ned_mps2"] = [
            float(value) for value in desired_acceleration
        ]
        payload["thrust_control"] = {
            "mode": "geometric_dynamic_inversion",
            "specific_thrust_mps2": float(thrust_norm_mps2),
            "hover_thrust": float(self.gains.hover_thrust),
        }

        self.last_payload = payload
        return payload

    def _curvature_speed_mps(self, curvature: float) -> float:
        if curvature <= max(1e-9, self.gains.curvature_speed_deadband):
            return float(self.gains.v_max)

        curvature_limited = math.sqrt(self.gains.a_lat_max / curvature)
        curvature_limited = float(np.clip(curvature_limited, self.gains.v_min, self.gains.v_max))
        if curvature_limited >= self.gains.v_max:
            return float(self.gains.v_max)

        excess_curvature = curvature - self.gains.curvature_speed_deadband
        blend = 1.0 - math.exp(-excess_curvature / self.gains.curvature_speed_ramp)
        blend = float(np.clip(blend, 0.0, 1.0))
        return float(self.gains.v_max - blend * (self.gains.v_max - curvature_limited))

    def _curvature_feedforward_acceleration(
        self,
        *,
        tangent: np.ndarray,
        heading: np.ndarray,
        along_track_speed_mps: float,
        preview_distance_m: float,
    ) -> np.ndarray:
        gain = float(self.gains.curvature_feedforward_gain)
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

        limit = float(self.gains.curvature_feedforward_max_acceleration_mps2)
        acceleration_norm = float(np.linalg.norm(acceleration))
        if limit > 0.0 and acceleration_norm > limit:
            acceleration = acceleration * (limit / acceleration_norm)
        return acceleration

    def _limited_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        limit = float(self.gains.max_commanded_acceleration_mps2)
        if limit <= 0.0:
            return acceleration
        norm = float(np.linalg.norm(acceleration))
        if norm <= limit or norm <= 1e-12:
            return acceleration
        return acceleration * (limit / norm)

    def _validate_gains(self) -> None:
        if self.gains.v_min < 0.0:
            raise ValueError("v_min cannot be negative")
        if self.gains.v_max < self.gains.v_min:
            raise ValueError("v_max cannot be less than v_min")
        if self.gains.a_lat_max <= 0.0:
            raise ValueError("a_lat_max must be positive")
        if self.gains.curvature_speed_deadband < 0.0:
            raise ValueError("curvature_speed_deadband cannot be negative")
        if self.gains.curvature_speed_ramp <= 0.0:
            raise ValueError("curvature_speed_ramp must be positive")
        if self.gains.curvature_feedforward_gain < 0.0:
            raise ValueError("curvature_feedforward_gain cannot be negative")
        if self.gains.curvature_feedforward_max_acceleration_mps2 < 0.0:
            raise ValueError("curvature_feedforward_max_acceleration_mps2 cannot be negative")
        if self.gains.hover_thrust <= 0.0:
            raise ValueError("hover_thrust must be positive")
        if not 0.0 <= self.gains.acceleration_filter_alpha <= 1.0:
            raise ValueError("acceleration_filter_alpha must be in [0, 1]")
        if self.gains.min_normalized_thrust > self.gains.max_normalized_thrust:
            raise ValueError("min_normalized_thrust cannot exceed max_normalized_thrust")

    def _filtered_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        alpha = float(self.gains.acceleration_filter_alpha)
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


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return max(0.0, number)
