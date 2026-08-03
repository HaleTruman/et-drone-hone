"""Geometric path follower for local-NED quadrotor racing paths.

The controller consumes the runtime :class:`core.schema.VehicleState` plus
path projection and preview samples from :mod:`autonomy.pathing`.
It emits the repository's standard attitude-target payload:

    {"quaternion": [w, x, y, z], "thrust": normalized_thrust, ...}

Coordinate conventions:
- Inertial: local NED, +z down, gravity = [0, 0, +g]
- Body: FRD, +z down, positive collective thrust accelerates along -body_z
- Quaternion: scalar-first [w, x, y, z], body-to-local-NED rotation

"Spirits of the Machine-God, aid your servant and free his weapon so he may use it to break his foes"
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import math

import numpy as np

from autonomy.pathing import PathCarrot
from core.control.command_mapper import CommandMapper
from core.coordinates import (
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
    GRAVITY_MPS2,
)
from core.schema import QuatWxyz, VehicleState


class GeometricPathFollower:
    """High-level path follower that produces attitude and normalized thrust."""

    def __init__(
        self,
        *,
        kp_cross: float = 1.2,
        kd_cross: float = 1.4,
        kp_cross_vertical: float | None = None,
        kd_cross_vertical: float | None = None,
        cross_gain_curvature_deadband: float = 0.0,
        cross_gain_curvature_ramp: float = 1.0,
        horizontal_cross_gain_curvature_boost: float = 0.0,
        vertical_cross_gain_curvature_boost: float = 0.0,
        kp_speed: float = 0.45,
        kd_speed: float = 0.0,
        acceleration_filter_alpha: float = 0.25,
        lookahead_m: float = 3.0,
        speed_lookahead_m: float = 10.0,
        v_max: float = 12.0,
        a_lat_max: float = 8.0,
        v_min: float = 1.5,
        curvature_speed_deadband: float = 0.04,
        curvature_speed_ramp: float = 0.08,
        cross_track_speed_derate_start_m: float = 0.5,
        cross_track_speed_derate_full_m: float = 2.0,
        cross_track_speed_derate_min_scale: float = 0.5,
        launch_speed_ramp_s: float = 1.0,
        curvature_feedforward_gain: float = 1.0,
        curvature_feedforward_max_acceleration_mps2: float = 8.0,
        hover_thrust: float = 0.265,
        max_commanded_acceleration_mps2: float = 6.0,
        max_commanded_jerk_mps3: float = 50.0,
        max_upward_acceleration_mps2: float = 8.0,
        max_downward_acceleration_mps2: float = 3.0,
        max_tilt_deg: float = 60.0,
        min_normalized_thrust: float = 0.05,
        max_normalized_thrust: float = 1.0,
        tilt_thrust_alignment_min: float = 0.35,
        drag_coeff: float = 0.0,
        gravity_mps2: float = GRAVITY_MPS2,
        command_mapper: CommandMapper | None = None,
    ) -> None:
        self.kp_cross = float(kp_cross)
        self.kd_cross = float(kd_cross)
        self.kp_cross_vertical = float(kp_cross if kp_cross_vertical is None else kp_cross_vertical)
        self.kd_cross_vertical = float(kd_cross if kd_cross_vertical is None else kd_cross_vertical)
        self.cross_gain_curvature_deadband = max(0.0, float(cross_gain_curvature_deadband))
        self.cross_gain_curvature_ramp = float(cross_gain_curvature_ramp)
        self.horizontal_cross_gain_curvature_boost = float(horizontal_cross_gain_curvature_boost)
        self.vertical_cross_gain_curvature_boost = float(vertical_cross_gain_curvature_boost)
        self.kp_speed = float(kp_speed)
        self.kd_speed = float(kd_speed)
        self.acceleration_filter_alpha = float(acceleration_filter_alpha)
        self.lookahead_m = float(lookahead_m)
        self.speed_lookahead_m = float(speed_lookahead_m)
        self.v_max = float(v_max)
        self.a_lat_max = float(a_lat_max)
        self.v_min = float(v_min)
        self.curvature_speed_deadband = float(curvature_speed_deadband)
        self.curvature_speed_ramp = float(curvature_speed_ramp)
        self.cross_track_speed_derate_start_m = float(cross_track_speed_derate_start_m)
        self.cross_track_speed_derate_full_m = float(cross_track_speed_derate_full_m)
        self.cross_track_speed_derate_min_scale = float(cross_track_speed_derate_min_scale)
        self.launch_speed_ramp_s = float(launch_speed_ramp_s)
        self.curvature_feedforward_gain = float(curvature_feedforward_gain)
        self.curvature_feedforward_max_acceleration_mps2 = float(
            curvature_feedforward_max_acceleration_mps2
        )
        self.hover_thrust = float(hover_thrust)
        self.max_commanded_acceleration_mps2 = float(max_commanded_acceleration_mps2)
        self.max_commanded_jerk_mps3 = float(max_commanded_jerk_mps3)
        self.max_upward_acceleration_mps2 = float(max_upward_acceleration_mps2)
        self.max_downward_acceleration_mps2 = float(max_downward_acceleration_mps2)
        self.max_tilt_rad = math.radians(float(max_tilt_deg))
        self.min_normalized_thrust = float(min_normalized_thrust)
        self.max_normalized_thrust = float(max_normalized_thrust)
        self.tilt_thrust_alignment_min = float(tilt_thrust_alignment_min)
        self.drag_coeff = float(drag_coeff)
        self.gravity_ned = np.array((0.0, 0.0, float(gravity_mps2)), dtype=float)
        self.command_mapper = command_mapper or CommandMapper()
        self.last_payload: dict[str, Any] | None = None
        self._filtered_acceleration_ned: np.ndarray | None = None
        self._last_commanded_acceleration_ned: np.ndarray | None = None
        self._last_commanded_acceleration_time_s: float | None = None
        self._last_jerk_limit_info: dict[str, Any] = {
            "active": False,
            "dt_s": None,
            "max_delta_acceleration_mps2": None,
            "requested_delta_acceleration_mps2": None,
        }

        if gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        self._validate_gains()

    def compute_control(
        self,
        vehicle_state: VehicleState,
        *,
        carrot: PathCarrot,
        time_since_takeoff_s: float | None = None,
    ) -> dict[str, Any]:
        velocity = np.asarray(vehicle_state.velocity_local_ned_mps, dtype=float)
        acceleration = np.asarray(vehicle_state.acceleration_local_ned_mps2, dtype=float)

        closest = np.asarray(carrot.projection_closest_point_local_ned_m, dtype=float)
        tangent = _unit(
            carrot.projection_tangent_local_ned,
            "carrot.projection_tangent_local_ned",
        )
        carrot_heading = _unit(carrot.tangent_local_ned, "carrot.tangent_local_ned")
        curvature = _nonnegative_float(
            carrot.max_curvature_ahead,
            "carrot.max_curvature_ahead",
        )

        cross_track_error = np.asarray(carrot.cross_track_error_local_ned_m, dtype=float)
        along_track_speed_mps = float(np.dot(velocity, tangent))
        cross_track_velocity = velocity - along_track_speed_mps * tangent
        horizontal_cross_track_error = cross_track_error.copy()
        horizontal_cross_track_error[2] = 0.0
        vertical_cross_track_error = np.array((0.0, 0.0, cross_track_error[2]), dtype=float)
        horizontal_cross_track_velocity = cross_track_velocity.copy()
        horizontal_cross_track_velocity[2] = 0.0
        vertical_cross_track_velocity = np.array((0.0, 0.0, cross_track_velocity[2]), dtype=float)
        cross_gain_curvature_alpha = self._cross_gain_curvature_alpha(curvature)
        horizontal_cross_gain_scale = (
            1.0 + self.horizontal_cross_gain_curvature_boost * cross_gain_curvature_alpha
        )
        vertical_cross_gain_scale = (
            1.0 + self.vertical_cross_gain_curvature_boost * cross_gain_curvature_alpha
        )
        kp_cross_effective = self.kp_cross * horizontal_cross_gain_scale
        kd_cross_effective = self.kd_cross * horizontal_cross_gain_scale
        kp_cross_vertical_effective = self.kp_cross_vertical * vertical_cross_gain_scale
        kd_cross_vertical_effective = self.kd_cross_vertical * vertical_cross_gain_scale
        cross_track_guidance_acceleration = (
            -kp_cross_effective * horizontal_cross_track_error
            - kd_cross_effective * horizontal_cross_track_velocity
            - kp_cross_vertical_effective * vertical_cross_track_error
            - kd_cross_vertical_effective * vertical_cross_track_velocity
        )
        curvature_commanded_speed_mps = self._curvature_speed_mps(curvature)
        cross_track_error_m = float(np.linalg.norm(cross_track_error))
        cross_track_speed_derate_alpha = self._cross_track_speed_derate_alpha(
            cross_track_error_m
        )
        cross_track_speed_derate_scale = (
            1.0
            - cross_track_speed_derate_alpha
            * (1.0 - self.cross_track_speed_derate_min_scale)
        )
        derated_commanded_speed_mps = float(
            np.clip(
                curvature_commanded_speed_mps * cross_track_speed_derate_scale,
                self.v_min,
                self.v_max,
            )
        )
        launch_speed_ramp_scale = self._launch_speed_ramp_scale(time_since_takeoff_s)
        commanded_speed_mps = float(
            np.clip(
                derated_commanded_speed_mps * launch_speed_ramp_scale,
                0.0,
                self.v_max,
            )
        )
        curvature_feedforward = self._curvature_feedforward_acceleration(
            tangent=tangent,
            heading=carrot_heading,
            along_track_speed_mps=along_track_speed_mps,
            preview_distance_m=(
                float(carrot.along_track_m)
                - float(carrot.projection_along_track_m)
            ),
        )

        along_track_acceleration_mps2 = float(np.dot(acceleration, tangent))
        speed_acceleration_mps2 = self.kp_speed * (
            commanded_speed_mps - along_track_speed_mps
        ) - self.kd_speed * along_track_acceleration_mps2
        # Cross-track feedback pulls back to the path; speed feedback pushes along it.
        desired_acceleration = (
            cross_track_guidance_acceleration
            + speed_acceleration_mps2 * tangent
            + curvature_feedforward
        )

        if self.drag_coeff > 0.0:
            speed_mps = float(np.linalg.norm(velocity))
            desired_acceleration = desired_acceleration + self.drag_coeff * speed_mps * velocity

        raw_desired_acceleration = desired_acceleration.astype(float)
        magnitude_limited_acceleration = self._limited_acceleration(raw_desired_acceleration)
        filtered_acceleration = self._filtered_acceleration(magnitude_limited_acceleration)
        constrained_acceleration = self._tilt_limited_acceleration(
            self._vertical_limited_acceleration(filtered_acceleration)
        )
        desired_acceleration = self._jerk_limited_acceleration(
            constrained_acceleration,
            sim_time_ns=vehicle_state.sim_time_ns,
        )

        # Convert the desired inertial acceleration into attitude plus normalized thrust.
        thrust_acceleration_ned = desired_acceleration - self.gravity_ned
        thrust_norm_mps2 = float(np.linalg.norm(thrust_acceleration_ned))
        if thrust_norm_mps2 <= 1e-9:
            raise ValueError("thrust acceleration vector cannot be zero")

        body_z_down_ned = -thrust_acceleration_ned / thrust_norm_mps2
        attitude_heading = carrot_heading.copy()
        attitude_heading[2] = 0.0
        if float(np.linalg.norm(attitude_heading)) <= 1e-9:
            attitude_heading = np.array((1.0, 0.0, 0.0), dtype=float)
        quaternion = np.asarray(
            _attitude_from_body_z_and_heading(body_z_down_ned, attitude_heading),
            dtype=float,
        )

        raw_thrust = float(
            np.clip(
                self.hover_thrust * thrust_norm_mps2 / self.gravity_ned[2],
                self.min_normalized_thrust,
                self.max_normalized_thrust,
            )
        )
        actual_body_z_down_ned = rotation_matrix_from_quaternion(
            vehicle_state.attitude_quaternion
        )[:, 2]
        thrust_alignment = float(
            np.clip(np.dot(actual_body_z_down_ned, body_z_down_ned), -1.0, 1.0)
        )
        thrust_alignment_scale = float(
            np.clip(thrust_alignment, self.tilt_thrust_alignment_min, 1.0)
        )
        tilt_compensated_thrust = float(raw_thrust * thrust_alignment_scale)
        actual_body_z_down_z = float(actual_body_z_down_ned[2])
        prevent_climb_thrust_cap = self.max_normalized_thrust
        prevent_climb_thrust_cap_active = False
        if desired_acceleration[2] >= 0.0 and actual_body_z_down_z > 1e-6:
            prevent_climb_thrust_cap = float(
                np.clip(
                    self.hover_thrust / actual_body_z_down_z,
                    self.min_normalized_thrust,
                    self.max_normalized_thrust,
                )
            )
            prevent_climb_thrust_cap_active = (
                tilt_compensated_thrust > prevent_climb_thrust_cap
            )
        max_allowed_thrust = min(self.max_normalized_thrust, prevent_climb_thrust_cap)
        thrust = float(
            np.clip(
                tilt_compensated_thrust,
                self.min_normalized_thrust,
                max_allowed_thrust,
            )
        )

        payload = self.command_mapper.to_attitude_target(quaternion, thrust)
        payload["source"] = "geometric_path_follower"
        payload["path_follower"] = {
            "projection_closest_point_local_ned_m": [float(value) for value in closest],
            "projection_along_track_m": float(carrot.projection_along_track_m),
            "preview_position_local_ned_m": [
                float(value) for value in carrot.position_local_ned_m
            ],
            "preview_along_track_m": float(carrot.along_track_m),
            "cross_track_error_m": float(cross_track_error_m),
            "cross_track_error_local_ned_m": [float(value) for value in cross_track_error],
            "horizontal_cross_track_error_local_ned_m": [
                float(value) for value in horizontal_cross_track_error
            ],
            "vertical_cross_track_error_local_ned_m": [
                float(value) for value in vertical_cross_track_error
            ],
            "cross_track_velocity_local_ned_mps": [
                float(value) for value in cross_track_velocity
            ],
            "horizontal_cross_track_velocity_local_ned_mps": [
                float(value) for value in horizontal_cross_track_velocity
            ],
            "vertical_cross_track_velocity_local_ned_mps": [
                float(value) for value in vertical_cross_track_velocity
            ],
            "cross_track_guidance_acceleration_local_ned_mps2": [
                float(value) for value in cross_track_guidance_acceleration
            ],
            "cross_track_guidance_acceleration_mps2": float(
                np.linalg.norm(cross_track_guidance_acceleration)
            ),
            "position_guidance_acceleration_local_ned_mps2": [
                float(value) for value in cross_track_guidance_acceleration
            ],
            "position_guidance_acceleration_mps2": float(
                np.linalg.norm(cross_track_guidance_acceleration)
            ),
            "attitude_heading_local_ned": [
                float(value) for value in _unit(attitude_heading, "attitude_heading")
            ],
            "attitude_heading_source": "carrot_tangent_local_ned",
            "kp_cross": float(self.kp_cross),
            "kd_cross": float(self.kd_cross),
            "kp_cross_vertical": float(self.kp_cross_vertical),
            "kd_cross_vertical": float(self.kd_cross_vertical),
            "kp_cross_effective": float(kp_cross_effective),
            "kd_cross_effective": float(kd_cross_effective),
            "kp_cross_vertical_effective": float(kp_cross_vertical_effective),
            "kd_cross_vertical_effective": float(kd_cross_vertical_effective),
            "cross_gain_curvature_alpha": float(cross_gain_curvature_alpha),
            "horizontal_cross_gain_scale": float(horizontal_cross_gain_scale),
            "vertical_cross_gain_scale": float(vertical_cross_gain_scale),
            "cross_gain_curvature_deadband": float(self.cross_gain_curvature_deadband),
            "cross_gain_curvature_ramp": float(self.cross_gain_curvature_ramp),
            "horizontal_cross_gain_curvature_boost": float(
                self.horizontal_cross_gain_curvature_boost
            ),
            "vertical_cross_gain_curvature_boost": float(
                self.vertical_cross_gain_curvature_boost
            ),
            "along_track_speed_mps": float(along_track_speed_mps),
            "along_track_acceleration_mps2": float(along_track_acceleration_mps2),
            "curvature_commanded_speed_mps": float(curvature_commanded_speed_mps),
            "derated_commanded_speed_mps": float(derated_commanded_speed_mps),
            "commanded_speed_mps": float(commanded_speed_mps),
            "cross_track_speed_derate_alpha": float(cross_track_speed_derate_alpha),
            "cross_track_speed_derate_scale": float(cross_track_speed_derate_scale),
            "cross_track_speed_derate_start_m": float(
                self.cross_track_speed_derate_start_m
            ),
            "cross_track_speed_derate_full_m": float(
                self.cross_track_speed_derate_full_m
            ),
            "cross_track_speed_derate_min_scale": float(
                self.cross_track_speed_derate_min_scale
            ),
            "launch_speed_ramp_s": float(self.launch_speed_ramp_s),
            "launch_speed_ramp_scale": float(launch_speed_ramp_scale),
            "time_since_takeoff_s": (
                None if time_since_takeoff_s is None else float(time_since_takeoff_s)
            ),
            "kp_speed": float(self.kp_speed),
            "kd_speed": float(self.kd_speed),
            "speed_acceleration_mps2": float(speed_acceleration_mps2),
            "curvature": float(carrot.curvature),
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
            "max_commanded_acceleration_mps2": float(self.max_commanded_acceleration_mps2),
            "max_commanded_jerk_mps3": float(self.max_commanded_jerk_mps3),
            "jerk_limit": self._last_jerk_limit_info,
            "raw_desired_acceleration_local_ned_mps2": [
                float(value) for value in raw_desired_acceleration
            ],
            "magnitude_limited_acceleration_local_ned_mps2": [
                float(value) for value in magnitude_limited_acceleration
            ],
            "filtered_acceleration_local_ned_mps2": [
                float(value) for value in filtered_acceleration
            ],
            "constrained_acceleration_local_ned_mps2": [
                float(value) for value in constrained_acceleration
            ],
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
            "raw_thrust": float(raw_thrust),
            "tilt_compensated_thrust": float(tilt_compensated_thrust),
            "prevent_climb_thrust_cap": float(prevent_climb_thrust_cap),
            "prevent_climb_thrust_cap_active": bool(prevent_climb_thrust_cap_active),
            "max_allowed_thrust": float(max_allowed_thrust),
            "final_thrust": float(thrust),
            "tilt_thrust_alignment": float(thrust_alignment),
            "tilt_thrust_alignment_scale": float(thrust_alignment_scale),
            "tilt_thrust_alignment_min": float(self.tilt_thrust_alignment_min),
            "actual_body_z_down_ned": [float(value) for value in actual_body_z_down_ned],
            "desired_body_z_down_ned": [float(value) for value in body_z_down_ned],
            "actual_desired_body_z_angle_deg": float(
                math.degrees(math.acos(thrust_alignment))
            ),
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

    def _cross_track_speed_derate_alpha(self, cross_track_error_m: float) -> float:
        start_m = float(self.cross_track_speed_derate_start_m)
        full_m = float(self.cross_track_speed_derate_full_m)
        error_m = max(0.0, float(cross_track_error_m))
        if error_m <= start_m:
            return 0.0
        if full_m <= start_m:
            return 1.0
        return float(np.clip((error_m - start_m) / (full_m - start_m), 0.0, 1.0))

    def _launch_speed_ramp_scale(self, time_since_takeoff_s: float | None) -> float:
        ramp_s = float(self.launch_speed_ramp_s)
        if ramp_s <= 0.0 or time_since_takeoff_s is None:
            return 1.0
        return float(np.clip(float(time_since_takeoff_s) / ramp_s, 0.0, 1.0))

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

    def _cross_gain_curvature_alpha(self, curvature: float) -> float:
        ramp = float(self.cross_gain_curvature_ramp)
        if ramp <= 0.0:
            return 1.0 if float(curvature) > self.cross_gain_curvature_deadband else 0.0
        return float(
            np.clip(
                (float(curvature) - self.cross_gain_curvature_deadband) / ramp,
                0.0,
                1.0,
            )
        )

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
        thrust_vertical_mps2 = float(self.gravity_ned[2] - acceleration[2])
        thrust_norm_mps2 = math.hypot(horizontal_norm, thrust_vertical_mps2)
        if thrust_norm_mps2 <= 1e-12:
            return acceleration

        tilt_rad = math.atan2(horizontal_norm, thrust_vertical_mps2)
        if tilt_rad <= self.max_tilt_rad:
            return acceleration

        limited = acceleration.copy()
        if self.max_tilt_rad < math.pi / 2.0:
            max_horizontal_mps2 = max(0.0, thrust_vertical_mps2) * math.tan(self.max_tilt_rad)
            if horizontal_norm <= 1e-12:
                return acceleration
            limited[0:2] = horizontal * (max_horizontal_mps2 / horizontal_norm)
        else:
            if horizontal_norm <= 1e-12:
                limited[2] = float(self.gravity_ned[2])
            else:
                limited_thrust_vertical_mps2 = horizontal_norm / math.tan(self.max_tilt_rad)
                limited[2] = float(self.gravity_ned[2] - limited_thrust_vertical_mps2)
        return limited

    def _jerk_limited_acceleration(
        self,
        acceleration: np.ndarray,
        *,
        sim_time_ns: int | float | None,
    ) -> np.ndarray:
        limit = float(self.max_commanded_jerk_mps3)
        command_time_s = None if sim_time_ns is None else float(sim_time_ns) * 1e-9
        previous = self._last_commanded_acceleration_ned
        previous_time_s = self._last_commanded_acceleration_time_s
        self._last_jerk_limit_info = {
            "active": False,
            "dt_s": None,
            "max_delta_acceleration_mps2": None,
            "requested_delta_acceleration_mps2": None,
        }

        if limit <= 0.0 or previous is None or command_time_s is None or previous_time_s is None:
            self._last_commanded_acceleration_ned = acceleration.astype(float)
            self._last_commanded_acceleration_time_s = command_time_s
            return acceleration.astype(float)

        dt_s = command_time_s - previous_time_s
        if dt_s <= 1e-6:
            self._last_commanded_acceleration_ned = acceleration.astype(float)
            self._last_commanded_acceleration_time_s = command_time_s
            return acceleration.astype(float)

        delta = acceleration - previous
        requested_delta = float(np.linalg.norm(delta))
        max_delta = limit * dt_s
        if requested_delta > max_delta and requested_delta > 1e-12:
            acceleration = previous + delta * (max_delta / requested_delta)
            self._last_jerk_limit_info["active"] = True

        self._last_jerk_limit_info.update(
            {
                "dt_s": float(dt_s),
                "max_delta_acceleration_mps2": float(max_delta),
                "requested_delta_acceleration_mps2": requested_delta,
            }
        )
        self._last_commanded_acceleration_ned = acceleration.astype(float)
        self._last_commanded_acceleration_time_s = command_time_s
        return acceleration.astype(float)

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
        if self.cross_track_speed_derate_start_m < 0.0:
            raise ValueError("cross_track_speed_derate_start_m cannot be negative")
        if self.cross_track_speed_derate_full_m < 0.0:
            raise ValueError("cross_track_speed_derate_full_m cannot be negative")
        if not 0.0 <= self.cross_track_speed_derate_min_scale <= 1.0:
            raise ValueError("cross_track_speed_derate_min_scale must be in [0, 1]")
        if self.launch_speed_ramp_s < 0.0:
            raise ValueError("launch_speed_ramp_s cannot be negative")
        if self.curvature_feedforward_gain < 0.0:
            raise ValueError("curvature_feedforward_gain cannot be negative")
        if self.curvature_feedforward_max_acceleration_mps2 < 0.0:
            raise ValueError("curvature_feedforward_max_acceleration_mps2 cannot be negative")
        if self.kp_cross < 0.0:
            raise ValueError("kp_cross cannot be negative")
        if self.kd_cross < 0.0:
            raise ValueError("kd_cross cannot be negative")
        if self.kp_cross_vertical < 0.0:
            raise ValueError("kp_cross_vertical cannot be negative")
        if self.kd_cross_vertical < 0.0:
            raise ValueError("kd_cross_vertical cannot be negative")
        if self.cross_gain_curvature_ramp < 0.0:
            raise ValueError("cross_gain_curvature_ramp cannot be negative")
        if self.horizontal_cross_gain_curvature_boost < -1.0:
            raise ValueError("horizontal_cross_gain_curvature_boost cannot be less than -1")
        if self.vertical_cross_gain_curvature_boost < -1.0:
            raise ValueError("vertical_cross_gain_curvature_boost cannot be less than -1")
        if self.kp_speed < 0.0:
            raise ValueError("kp_speed cannot be negative")
        if self.kd_speed < 0.0:
            raise ValueError("kd_speed cannot be negative")
        if self.hover_thrust <= 0.0:
            raise ValueError("hover_thrust must be positive")
        if self.max_upward_acceleration_mps2 < 0.0:
            raise ValueError("max_upward_acceleration_mps2 cannot be negative")
        if self.max_downward_acceleration_mps2 < 0.0:
            raise ValueError("max_downward_acceleration_mps2 cannot be negative")
        if self.max_commanded_jerk_mps3 < 0.0:
            raise ValueError("max_commanded_jerk_mps3 cannot be negative")
        if not 0.0 <= self.max_tilt_rad < math.pi:
            raise ValueError("max_tilt_deg must be in [0, 180)")
        if not 0.0 <= self.acceleration_filter_alpha <= 1.0:
            raise ValueError("acceleration_filter_alpha must be in [0, 1]")
        if self.min_normalized_thrust > self.max_normalized_thrust:
            raise ValueError("min_normalized_thrust cannot exceed max_normalized_thrust")
        if not 0.0 <= self.tilt_thrust_alignment_min <= 1.0:
            raise ValueError("tilt_thrust_alignment_min must be in [0, 1]")

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
