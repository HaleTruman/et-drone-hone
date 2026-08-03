"""Gate-aware acceleration controller for racing through discrete gates.

Coordinate conventions:
- Inertial: local NED, +z down, gravity = [0, 0, +g]
- Body: FRD, +z down, collective thrust accelerates along -body_z
- Quaternion: scalar-first [w, x, y, z], body-to-local-NED rotation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from core.control.command_mapper import CommandMapper
from core.coordinates import (
    GRAVITY_MPS2,
    quaternion_from_rotation_matrix,
    quat_wxyz,
)
from core.schema import QuatWxyz, Vec3, VehicleState
from mapping.gates import GateRecord


@dataclass(frozen=True)
class AutiPilotGains:
    v_max_mps: float = 15.0
    v_min_mps: float = 2.0
    a_lat_max_mps2: float = 8.0
    medium_turn_angle_deg: float = 45.0
    tight_turn_angle_deg: float = 95.0
    medium_turn_a_lat_mps2: float | None = None
    tight_turn_a_lat_mps2: float | None = None
    kp_speed: float = 2.0
    near_gate_speed_gain_near_m: float = 1.0
    near_gate_speed_gain_far_m: float = 5.0
    near_gate_speed_gain_min_scale: float = 0.25
    near_gate_speed_gain_ramp_s: float = 0.5
    kp_dir: float = 1.5
    kd_dir: float = 0.0
    kp_lat: float = 1.2
    kd_lat: float = 1.8
    kp_z: float = 3.0
    kd_z: float = 2.0
    lookahead_near_m: float = 1.0
    lookahead_far_m: float = 5.0
    fly_through_turn_angle_start_deg: float = 70.0
    fly_through_turn_angle_full_deg: float = 120.0
    fly_through_next_gate_distance_start_m: float = 12.0
    fly_through_next_gate_distance_full_m: float = 30.0
    fly_through_lookahead_min_scale: float = 0.35
    gate_cross_velocity_damping: float = 2.0
    gate_cross_velocity_cancel_near_m: float = 1.0
    gate_cross_velocity_cancel_far_m: float = 7.0
    approach_gain_min_scale: float = 0.3
    post_cross_turn_scale: float = 1.0
    post_cross_ramp_distance_m: float = 3.0
    launch_speed_ramp_s: float = 1.0
    max_specific_thrust_mps2: float = GRAVITY_MPS2 / 0.2644
    min_normalized_thrust: float = 0.05
    max_normalized_thrust: float = 0.95
    max_commanded_acceleration_mps2: float = 20.0
    gate_switch_acceleration_ramp_s: float = 0.35
    max_commanded_jerk_mps3: float = 40.0


class AutiPilot:
    """Produces attitude-error quaternion plus normalized thrust from gate geometry."""

    def __init__(
        self,
        gains: AutiPilotGains | None = None,
        *,
        gravity_mps2: float = GRAVITY_MPS2,
        command_mapper: CommandMapper | None = None,
    ) -> None:
        self.gains = gains or AutiPilotGains()
        self.gravity_ned = np.array((0.0, 0.0, float(gravity_mps2)), dtype=float)
        self.command_mapper = command_mapper or CommandMapper()
        self.last_payload: dict[str, Any] | None = None
        self._last_target_gate_key: str | None = None
        self._gate_switch_ramp_start_time_s: float | None = None
        self._gate_switch_ramp_start_acceleration_ned: np.ndarray | None = None
        self._last_gate_switch_ramp_info: dict[str, Any] = {
            "active": False,
            "alpha": 1.0,
            "elapsed_s": None,
            "duration_s": float(self.gains.gate_switch_acceleration_ramp_s),
            "from_gate_key": None,
            "to_gate_key": None,
        }
        self._last_commanded_acceleration_ned: np.ndarray | None = None
        self._last_commanded_acceleration_time_s: float | None = None
        self._last_speed_gain_scale: float | None = None
        self._last_speed_gain_scale_time_s: float | None = None
        self._last_speed_gain_scale_info: dict[str, Any] = {
            "raw_scale": 1.0,
            "scale": 1.0,
            "active": False,
            "ramp_s": float(self.gains.near_gate_speed_gain_ramp_s),
            "dt_s": None,
            "max_delta_scale": None,
        }
        self._last_jerk_limit_info: dict[str, Any] = {
            "active": False,
            "dt_s": None,
            "max_delta_acceleration_mps2": None,
            "requested_delta_acceleration_mps2": None,
            "held_for_nonpositive_dt": False,
        }
        self._validate_gains()

    def compute(
        self,
        state: VehicleState,
        target_gate: GateRecord,
        next_gate: GateRecord | None = None,
        *,
        time_since_takeoff_s: float | None = None,
    ) -> tuple[QuatWxyz, float]:
        payload = self._compute_payload(
            state,
            target_gate,
            next_gate,
            time_since_takeoff_s=time_since_takeoff_s,
        )
        return payload["error_quaternion"], float(payload["thrust"])

    def compute_control(
        self,
        state: VehicleState,
        target_gate: GateRecord,
        next_gate: GateRecord | None = None,
        *,
        time_since_takeoff_s: float | None = None,
    ) -> dict[str, Any]:
        """Return a diagnostic payload around the controller's native output."""
        payload = self._compute_payload(
            state,
            target_gate,
            next_gate,
            time_since_takeoff_s=time_since_takeoff_s,
        )
        command = {
            "source": "gate_aware_acceleration_controller",
            "error_quaternion": payload["error_quaternion"],
            "thrust": self.command_mapper.scale_thrust(float(payload["thrust"])),
            "gate_aware_acceleration_controller": payload[
                "gate_aware_acceleration_controller"
            ],
            "desired_acceleration_local_ned_mps2": payload[
                "desired_acceleration_local_ned_mps2"
            ],
            "thrust_control": payload["thrust_control"],
        }
        self.last_payload = command
        return command

    def _compute_payload(
        self,
        state: VehicleState,
        target_gate: GateRecord,
        next_gate: GateRecord | None = None,
        *,
        time_since_takeoff_s: float | None = None,
    ) -> dict[str, Any]:
        gains = self.gains
        position = _vec3(state.position_local_ned_m, "state.position_local_ned_m")
        velocity = _vec3(state.velocity_local_ned_mps, "state.velocity_local_ned_mps")
        acceleration = _vec3(
            state.acceleration_local_ned_mps2,
            "state.acceleration_local_ned_mps2",
        )
        current_gate_position = _vec3(
            target_gate.position_local_ned_m,
            "target_gate.position_local_ned_m",
        )

        los = current_gate_position - position
        distance_m = float(np.linalg.norm(los))
        los_dir = _unit_or(los, np.array((1.0, 0.0, 0.0), dtype=float))
        has_next_gate = next_gate is not None
        if has_next_gate:
            next_gate_position = _vec3(
                next_gate.position_local_ned_m,
                "next_gate.position_local_ned_m",
            )
            target_to_next_gate = next_gate_position - current_gate_position
            target_to_next_gate_distance_m = float(np.linalg.norm(target_to_next_gate))
            exit_dir = _unit_or(target_to_next_gate, los_dir)
            target_to_next_gate_horizontal_bearing_rad = math.atan2(
                float(target_to_next_gate[1]),
                float(target_to_next_gate[0]),
            )
            target_to_next_gate_horizontal_m = math.hypot(
                float(target_to_next_gate[0]),
                float(target_to_next_gate[1]),
            )
            target_to_next_gate_elevation_rad = math.atan2(
                -float(target_to_next_gate[2]),
                target_to_next_gate_horizontal_m,
            )
        else:
            next_gate_position = None
            target_to_next_gate = None
            target_to_next_gate_distance_m = None
            target_to_next_gate_horizontal_bearing_rad = None
            target_to_next_gate_elevation_rad = None
            exit_dir = los_dir
        entry_dir = _unit_or(los, np.array((1.0, 0.0, 0.0), dtype=float))

        turn_angle_rad = self._turn_angle(entry_dir, exit_dir)
        raw_lookahead_alpha = _smoothstep(
            distance_m,
            gains.lookahead_far_m,
            gains.lookahead_near_m,
        )
        fly_through_lookahead_scale = self._fly_through_lookahead_scale(
            turn_angle_rad=turn_angle_rad,
            target_to_next_gate_distance_m=target_to_next_gate_distance_m,
        )
        lookahead_alpha = raw_lookahead_alpha * fly_through_lookahead_scale
        aim_dir = _unit_or(
            (1.0 - lookahead_alpha) * los_dir + lookahead_alpha * exit_dir,
            exit_dir,
        )

        speed_plan = self._desired_speed_plan(distance_m, turn_angle_rad)
        unramped_v_des_mps = float(speed_plan["desired_speed_mps"])
        launch_speed_ramp_scale = self._launch_speed_ramp_scale(time_since_takeoff_s)
        v_des_mps = float(
            np.clip(
                unramped_v_des_mps * launch_speed_ramp_scale,
                0.0,
                gains.v_max_mps,
            )
        )
        v_des_vec = v_des_mps * aim_dir

        along_speed_mps = float(np.dot(velocity, aim_dir))
        speed_error_mps = v_des_mps - along_speed_mps
        speed_gain_scale = self._near_gate_speed_gain_scale(
            distance_m,
            sim_time_ns=state.sim_time_ns,
        )
        effective_kp_speed = gains.kp_speed * speed_gain_scale
        a_long = effective_kp_speed * speed_error_mps * aim_dir

        v_dir_error = v_des_vec - velocity
        v_dir_lateral_error = v_dir_error - float(np.dot(v_dir_error, aim_dir)) * aim_dir
        measured_lateral_acceleration = (
            acceleration - float(np.dot(acceleration, aim_dir)) * aim_dir
        )
        a_dir = (
            gains.kp_dir * v_dir_lateral_error
            - gains.kd_dir * measured_lateral_acceleration
        )

        e_lat = _lateral_position_error(position, current_gate_position, aim_dir)
        v_lat = _lateral_velocity(velocity, aim_dir)
        a_lat_pos = -gains.kp_lat * e_lat - gains.kd_lat * v_lat
        gate_cross_cancel_alpha = self._gate_cross_velocity_cancel_alpha(distance_m)
        gate_cross_velocity = _lateral_velocity(velocity, aim_dir)
        gate_cross_velocity_cancel_acceleration = (
            -gains.gate_cross_velocity_damping
            * gate_cross_cancel_alpha
            * gate_cross_velocity
        )

        lateral_schedule = self._lateral_schedule(
            distance_m=distance_m,
            crossed=target_gate.crossed,
            turn_angle_rad=turn_angle_rad,
            position=position,
            current_gate=current_gate_position,
            exit_dir=exit_dir,
        )
        scheduled_lateral = (
            lateral_schedule * (a_dir + a_lat_pos)
            + gate_cross_velocity_cancel_acceleration
        )

        target_altitude_m = -float(current_gate_position[2])
        altitude_m = -float(position[2])
        e_z_m = target_altitude_m - altitude_m
        a_z_vec = np.array(
            (0.0, 0.0, -gains.kp_z * e_z_m - gains.kd_z * velocity[2])
        )

        desired_acceleration = a_long + scheduled_lateral + a_z_vec
        desired_acceleration = _limit_norm(
            desired_acceleration,
            gains.max_commanded_acceleration_mps2,
        )
        desired_acceleration = self._gate_switch_ramped_acceleration(
            desired_acceleration,
            target_gate=target_gate,
            sim_time_ns=state.sim_time_ns,
        )
        desired_acceleration = self._jerk_limited_acceleration(
            desired_acceleration,
            sim_time_ns=state.sim_time_ns,
        )

        specific_force = desired_acceleration - self.gravity_ned
        specific_thrust_mps2 = float(np.linalg.norm(specific_force))
        body_z_down_ned = (
            np.array((0.0, 0.0, 1.0), dtype=float)
            if specific_thrust_mps2 <= 1e-9
            else -specific_force / specific_thrust_mps2
        )
        heading = aim_dir.copy()
        heading[2] = 0.0
        heading = _unit_or(heading, np.array((1.0, 0.0, 0.0), dtype=float))

        q_des = _attitude_from_body_z_and_heading(body_z_down_ned, heading)
        q_err = _quat_multiply(_quat_conjugate(state.attitude_quaternion), q_des)
        if q_err[0] < 0.0:
            q_err = tuple(-value for value in q_err)

        thrust = float(
            np.clip(
                specific_thrust_mps2 / gains.max_specific_thrust_mps2,
                gains.min_normalized_thrust,
                gains.max_normalized_thrust,
            )
        )
        payload = {
            "error_quaternion": quat_wxyz(q_err),
            "thrust": thrust,
            "desired_acceleration_local_ned_mps2": [
                float(value) for value in desired_acceleration
            ],
            "gate_aware_acceleration_controller": {
                "gate_id": target_gate.gate_id,
                "gate_sequence": target_gate.sequence,
                "next_gate_id": None if next_gate is None else next_gate.gate_id,
                "next_gate_sequence": None if next_gate is None else next_gate.sequence,
                "next_gate_available": bool(has_next_gate),
                "fallback_mode": None if has_next_gate else "target_gate_only",
                "distance_to_current_gate_m": distance_m,
                "crossed_current_gate": bool(target_gate.crossed),
                "raw_lookahead_alpha": float(raw_lookahead_alpha),
                "lookahead_alpha": float(lookahead_alpha),
                "fly_through_active": bool(fly_through_lookahead_scale < 0.999),
                "fly_through_lookahead_scale": float(fly_through_lookahead_scale),
                "fly_through_turn_angle_start_deg": float(
                    gains.fly_through_turn_angle_start_deg
                ),
                "fly_through_turn_angle_full_deg": float(
                    gains.fly_through_turn_angle_full_deg
                ),
                "fly_through_next_gate_distance_start_m": float(
                    gains.fly_through_next_gate_distance_start_m
                ),
                "fly_through_next_gate_distance_full_m": float(
                    gains.fly_through_next_gate_distance_full_m
                ),
                "fly_through_lookahead_min_scale": float(
                    gains.fly_through_lookahead_min_scale
                ),
                "aim_direction_local_ned": [float(value) for value in aim_dir],
                "exit_direction_local_ned": [float(value) for value in exit_dir],
                "target_to_next_gate_vector_local_ned_m": (
                    None
                    if target_to_next_gate is None
                    else [float(value) for value in target_to_next_gate]
                ),
                "target_to_next_gate_distance_m": target_to_next_gate_distance_m,
                "target_to_next_gate_horizontal_bearing_rad": (
                    None
                    if target_to_next_gate_horizontal_bearing_rad is None
                    else float(target_to_next_gate_horizontal_bearing_rad)
                ),
                "target_to_next_gate_horizontal_bearing_deg": (
                    None
                    if target_to_next_gate_horizontal_bearing_rad is None
                    else float(math.degrees(target_to_next_gate_horizontal_bearing_rad))
                ),
                "target_to_next_gate_elevation_rad": (
                    None
                    if target_to_next_gate_elevation_rad is None
                    else float(target_to_next_gate_elevation_rad)
                ),
                "target_to_next_gate_elevation_deg": (
                    None
                    if target_to_next_gate_elevation_rad is None
                    else float(math.degrees(target_to_next_gate_elevation_rad))
                ),
                "turn_angle_rad": float(turn_angle_rad),
                "turn_angle_deg": float(math.degrees(turn_angle_rad)),
                "target_to_next_gate_turn_angle_rad": float(turn_angle_rad),
                "target_to_next_gate_turn_angle_deg": float(math.degrees(turn_angle_rad)),
                "turn_speed_shape_alpha": float(speed_plan["turn_shape_alpha"]),
                "turn_speed_medium_angle_deg": float(gains.medium_turn_angle_deg),
                "turn_speed_tight_angle_deg": float(gains.tight_turn_angle_deg),
                "turn_speed_effective_lateral_acceleration_mps2": float(
                    speed_plan["effective_lateral_acceleration_mps2"]
                ),
                "turn_speed_medium_lateral_acceleration_mps2": float(
                    speed_plan["medium_lateral_acceleration_mps2"]
                ),
                "turn_speed_tight_lateral_acceleration_mps2": float(
                    speed_plan["tight_lateral_acceleration_mps2"]
                ),
                "curvature_estimate": float(speed_plan["curvature_estimate"]),
                "curvature_limited_speed_mps": float(
                    speed_plan["curvature_limited_speed_mps"]
                ),
                "unramped_desired_speed_mps": float(unramped_v_des_mps),
                "desired_speed_mps": float(v_des_mps),
                "launch_speed_ramp_s": float(gains.launch_speed_ramp_s),
                "launch_speed_ramp_scale": float(launch_speed_ramp_scale),
                "time_since_takeoff_s": (
                    None if time_since_takeoff_s is None else float(time_since_takeoff_s)
                ),
                "along_speed_mps": float(along_speed_mps),
                "speed_error_mps": float(speed_error_mps),
                "speed_gain": float(gains.kp_speed),
                "effective_speed_gain": float(effective_kp_speed),
                "near_gate_speed_gain_scale": float(speed_gain_scale),
                "near_gate_speed_gain": self._last_speed_gain_scale_info,
                "near_gate_speed_gain_near_m": float(gains.near_gate_speed_gain_near_m),
                "near_gate_speed_gain_far_m": float(gains.near_gate_speed_gain_far_m),
                "near_gate_speed_gain_min_scale": float(
                    gains.near_gate_speed_gain_min_scale
                ),
                "near_gate_speed_gain_ramp_s": float(
                    gains.near_gate_speed_gain_ramp_s
                ),
                "lateral_schedule": float(lateral_schedule),
                "lateral_error_local_ned_m": [float(value) for value in e_lat],
                "lateral_velocity_local_ned_mps": [float(value) for value in v_lat],
                "gate_cross_velocity_cancel_alpha": float(gate_cross_cancel_alpha),
                "gate_cross_velocity_damping": float(gains.gate_cross_velocity_damping),
                "gate_cross_velocity_cancel_near_m": float(
                    gains.gate_cross_velocity_cancel_near_m
                ),
                "gate_cross_velocity_cancel_far_m": float(
                    gains.gate_cross_velocity_cancel_far_m
                ),
                "gate_cross_velocity_local_ned_mps": [
                    float(value) for value in gate_cross_velocity
                ],
                "gate_cross_velocity_cancel_acceleration_local_ned_mps2": [
                    float(value) for value in gate_cross_velocity_cancel_acceleration
                ],
                "measured_lateral_acceleration_local_ned_mps2": [
                    float(value) for value in measured_lateral_acceleration
                ],
                "longitudinal_acceleration_local_ned_mps2": [
                    float(value) for value in a_long
                ],
                "direction_acceleration_local_ned_mps2": [
                    float(value) for value in a_dir
                ],
                "lateral_position_acceleration_local_ned_mps2": [
                    float(value) for value in a_lat_pos
                ],
                "target_gate_altitude_m": float(target_altitude_m),
                "vehicle_altitude_m": float(altitude_m),
                "vertical_error_m": float(e_z_m),
                "gate_switch_acceleration_ramp": self._last_gate_switch_ramp_info,
                "max_commanded_jerk_mps3": float(gains.max_commanded_jerk_mps3),
                "jerk_limit": self._last_jerk_limit_info,
            },
            "thrust_control": {
                "mode": "gate_aware_dynamic_inversion",
                "specific_thrust_mps2": specific_thrust_mps2,
                "max_specific_thrust_mps2": float(gains.max_specific_thrust_mps2),
                "desired_body_z_down_ned": [float(value) for value in body_z_down_ned],
                "final_thrust": thrust,
            },
        }
        self.last_payload = payload
        return payload

    def _desired_speed(self, distance_m: float, turn_angle_rad: float) -> float:
        return float(self._desired_speed_plan(distance_m, turn_angle_rad)["desired_speed_mps"])

    def _desired_speed_plan(self, distance_m: float, turn_angle_rad: float) -> dict[str, float]:
        gains = self.gains
        kappa_est = max(1e-6, turn_angle_rad / max(distance_m, 1.0))
        medium_a_lat_mps2 = (
            gains.a_lat_max_mps2
            if gains.medium_turn_a_lat_mps2 is None
            else gains.medium_turn_a_lat_mps2
        )
        tight_a_lat_mps2 = (
            gains.a_lat_max_mps2
            if gains.tight_turn_a_lat_mps2 is None
            else gains.tight_turn_a_lat_mps2
        )
        turn_alpha = _smoothstep(
            math.degrees(turn_angle_rad),
            gains.medium_turn_angle_deg,
            gains.tight_turn_angle_deg,
        )
        effective_a_lat_mps2 = (
            (1.0 - turn_alpha) * medium_a_lat_mps2
            + turn_alpha * tight_a_lat_mps2
        )
        v_curvature_mps = math.sqrt(effective_a_lat_mps2 / kappa_est)
        desired_speed_mps = float(
            np.clip(
                min(gains.v_max_mps, v_curvature_mps),
                gains.v_min_mps,
                gains.v_max_mps,
            )
        )
        return {
            "desired_speed_mps": desired_speed_mps,
            "curvature_estimate": float(kappa_est),
            "curvature_limited_speed_mps": float(v_curvature_mps),
            "effective_lateral_acceleration_mps2": float(effective_a_lat_mps2),
            "medium_lateral_acceleration_mps2": float(medium_a_lat_mps2),
            "tight_lateral_acceleration_mps2": float(tight_a_lat_mps2),
            "turn_shape_alpha": float(turn_alpha),
        }

    def _turn_angle(self, entry_dir: np.ndarray, exit_dir: np.ndarray) -> float:
        dot = float(np.clip(np.dot(entry_dir, exit_dir), -1.0, 1.0))
        return float(math.acos(dot))

    def _launch_speed_ramp_scale(self, time_since_takeoff_s: float | None) -> float:
        ramp_s = float(self.gains.launch_speed_ramp_s)
        if ramp_s <= 0.0 or time_since_takeoff_s is None:
            return 1.0
        return float(np.clip(float(time_since_takeoff_s) / ramp_s, 0.0, 1.0))

    def _gate_cross_velocity_cancel_alpha(self, distance_m: float) -> float:
        gains = self.gains
        return 1.0 - _smoothstep(
            distance_m,
            gains.gate_cross_velocity_cancel_near_m,
            gains.gate_cross_velocity_cancel_far_m,
        )

    def _fly_through_lookahead_scale(
        self,
        *,
        turn_angle_rad: float,
        target_to_next_gate_distance_m: float | None,
    ) -> float:
        if target_to_next_gate_distance_m is None:
            return 1.0

        gains = self.gains
        turn_alpha = _smoothstep(
            math.degrees(turn_angle_rad),
            gains.fly_through_turn_angle_start_deg,
            gains.fly_through_turn_angle_full_deg,
        )
        distance_alpha = _smoothstep(
            target_to_next_gate_distance_m,
            gains.fly_through_next_gate_distance_start_m,
            gains.fly_through_next_gate_distance_full_m,
        )
        fly_through_alpha = turn_alpha * distance_alpha
        return float(
            np.clip(
                1.0
                - fly_through_alpha
                * (1.0 - gains.fly_through_lookahead_min_scale),
                gains.fly_through_lookahead_min_scale,
                1.0,
            )
        )

    def _near_gate_speed_gain_scale(
        self,
        distance_m: float,
        *,
        sim_time_ns: int | float | None,
    ) -> float:
        gains = self.gains
        raw_scale = float(
            np.clip(
                gains.near_gate_speed_gain_min_scale
                + (1.0 - gains.near_gate_speed_gain_min_scale)
                * _smoothstep(
                    distance_m,
                    gains.near_gate_speed_gain_near_m,
                    gains.near_gate_speed_gain_far_m,
                ),
                gains.near_gate_speed_gain_min_scale,
                1.0,
            )
        )
        ramp_s = float(gains.near_gate_speed_gain_ramp_s)
        command_time_s = None if sim_time_ns is None else float(sim_time_ns) * 1e-9
        previous = self._last_speed_gain_scale
        previous_time_s = self._last_speed_gain_scale_time_s

        self._last_speed_gain_scale_info = {
            "raw_scale": raw_scale,
            "scale": raw_scale,
            "active": False,
            "ramp_s": ramp_s,
            "dt_s": None,
            "max_delta_scale": None,
        }

        if ramp_s <= 0.0 or previous is None or command_time_s is None or previous_time_s is None:
            self._last_speed_gain_scale = raw_scale
            self._last_speed_gain_scale_time_s = command_time_s
            return raw_scale

        dt_s = command_time_s - previous_time_s
        if dt_s <= 1e-6:
            self._last_speed_gain_scale_info.update(
                {
                    "scale": float(previous),
                    "active": abs(raw_scale - previous) > 1e-12,
                    "dt_s": float(dt_s),
                    "max_delta_scale": 0.0,
                }
            )
            return float(previous)

        max_delta = dt_s / ramp_s
        delta = raw_scale - previous
        if abs(delta) > max_delta:
            scale = previous + math.copysign(max_delta, delta)
            active = True
        else:
            scale = raw_scale
            active = False

        scale = float(np.clip(scale, gains.near_gate_speed_gain_min_scale, 1.0))
        self._last_speed_gain_scale_info.update(
            {
                "scale": scale,
                "active": active,
                "dt_s": float(dt_s),
                "max_delta_scale": float(max_delta),
            }
        )
        self._last_speed_gain_scale = scale
        self._last_speed_gain_scale_time_s = command_time_s
        return scale

    def _gate_switch_ramped_acceleration(
        self,
        acceleration: np.ndarray,
        *,
        target_gate: GateRecord,
        sim_time_ns: int | float | None,
    ) -> np.ndarray:
        ramp_s = float(self.gains.gate_switch_acceleration_ramp_s)
        command_time_s = None if sim_time_ns is None else float(sim_time_ns) * 1e-9
        target_key = _gate_key(target_gate)
        previous_key = self._last_target_gate_key
        switched = previous_key is not None and target_key != previous_key

        self._last_gate_switch_ramp_info = {
            "active": False,
            "alpha": 1.0,
            "elapsed_s": None,
            "duration_s": ramp_s,
            "from_gate_key": previous_key,
            "to_gate_key": target_key,
        }

        if switched and ramp_s > 0.0 and command_time_s is not None:
            self._gate_switch_ramp_start_time_s = command_time_s
            self._gate_switch_ramp_start_acceleration_ned = (
                self._last_commanded_acceleration_ned.copy()
                if self._last_commanded_acceleration_ned is not None
                else acceleration.astype(float)
            )
        elif switched:
            self._gate_switch_ramp_start_time_s = None
            self._gate_switch_ramp_start_acceleration_ned = None

        self._last_target_gate_key = target_key

        start_time_s = self._gate_switch_ramp_start_time_s
        start_acceleration = self._gate_switch_ramp_start_acceleration_ned
        if ramp_s <= 0.0 or command_time_s is None or start_time_s is None or start_acceleration is None:
            return acceleration.astype(float)

        elapsed_s = max(0.0, command_time_s - start_time_s)
        alpha = float(np.clip(elapsed_s / ramp_s, 0.0, 1.0))
        if alpha >= 1.0:
            self._gate_switch_ramp_start_time_s = None
            self._gate_switch_ramp_start_acceleration_ned = None
            self._last_gate_switch_ramp_info.update(
                {
                    "elapsed_s": float(elapsed_s),
                    "alpha": 1.0,
                }
            )
            return acceleration.astype(float)

        ramped = start_acceleration + alpha * (acceleration - start_acceleration)
        self._last_gate_switch_ramp_info.update(
            {
                "active": True,
                "elapsed_s": float(elapsed_s),
                "alpha": float(alpha),
            }
        )
        return ramped.astype(float)

    def _jerk_limited_acceleration(
        self,
        acceleration: np.ndarray,
        *,
        sim_time_ns: int | float | None,
    ) -> np.ndarray:
        limit = float(self.gains.max_commanded_jerk_mps3)
        command_time_s = None if sim_time_ns is None else float(sim_time_ns) * 1e-9
        previous = self._last_commanded_acceleration_ned
        previous_time_s = self._last_commanded_acceleration_time_s
        self._last_jerk_limit_info = {
            "active": False,
            "dt_s": None,
            "max_delta_acceleration_mps2": None,
            "requested_delta_acceleration_mps2": None,
            "held_for_nonpositive_dt": False,
        }

        if limit <= 0.0 or previous is None or command_time_s is None or previous_time_s is None:
            self._last_commanded_acceleration_ned = acceleration.astype(float)
            self._last_commanded_acceleration_time_s = command_time_s
            return acceleration.astype(float)

        dt_s = command_time_s - previous_time_s
        if dt_s <= 1e-6:
            requested_delta = float(np.linalg.norm(acceleration - previous))
            self._last_jerk_limit_info.update(
                {
                    "active": requested_delta > 1e-12,
                    "dt_s": float(dt_s),
                    "max_delta_acceleration_mps2": 0.0,
                    "requested_delta_acceleration_mps2": requested_delta,
                    "held_for_nonpositive_dt": True,
                }
            )
            return previous.astype(float)

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

    def _lateral_schedule(
        self,
        *,
        distance_m: float,
        crossed: bool,
        turn_angle_rad: float,
        position: np.ndarray,
        current_gate: np.ndarray,
        exit_dir: np.ndarray,
    ) -> float:
        gains = self.gains
        approach = gains.approach_gain_min_scale + (
            1.0 - gains.approach_gain_min_scale
        ) * _smoothstep(distance_m, gains.lookahead_near_m, gains.lookahead_far_m)
        if not crossed:
            return float(np.clip(approach, 0.0, 10.0))

        progress_m = max(0.0, float(np.dot(position - current_gate, exit_dir)))
        ramp = (
            1.0
            if gains.post_cross_ramp_distance_m <= 0.0
            else np.clip(progress_m / gains.post_cross_ramp_distance_m, 0.0, 1.0)
        )
        turn_boost = 1.0 + gains.post_cross_turn_scale * (turn_angle_rad / math.pi)
        return float(np.clip(approach * (1.0 + ramp * (turn_boost - 1.0)), 0.0, 10.0))

    def _validate_gains(self) -> None:
        gains = self.gains
        if self.gravity_ned[2] <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if gains.v_min_mps < 0.0:
            raise ValueError("v_min_mps cannot be negative")
        if gains.v_max_mps < gains.v_min_mps:
            raise ValueError("v_max_mps cannot be less than v_min_mps")
        if gains.a_lat_max_mps2 <= 0.0:
            raise ValueError("a_lat_max_mps2 must be positive")
        if gains.near_gate_speed_gain_near_m < 0.0:
            raise ValueError("near_gate_speed_gain_near_m cannot be negative")
        if gains.near_gate_speed_gain_far_m < gains.near_gate_speed_gain_near_m:
            raise ValueError(
                "near_gate_speed_gain_far_m cannot be less than near_gate_speed_gain_near_m"
            )
        if not 0.0 <= gains.near_gate_speed_gain_min_scale <= 1.0:
            raise ValueError("near_gate_speed_gain_min_scale must be in [0, 1]")
        if gains.near_gate_speed_gain_ramp_s < 0.0:
            raise ValueError("near_gate_speed_gain_ramp_s cannot be negative")
        if gains.medium_turn_angle_deg < 0.0:
            raise ValueError("medium_turn_angle_deg cannot be negative")
        if gains.tight_turn_angle_deg < gains.medium_turn_angle_deg:
            raise ValueError("tight_turn_angle_deg cannot be less than medium_turn_angle_deg")
        if gains.medium_turn_a_lat_mps2 is not None and gains.medium_turn_a_lat_mps2 <= 0.0:
            raise ValueError("medium_turn_a_lat_mps2 must be positive when set")
        if gains.tight_turn_a_lat_mps2 is not None and gains.tight_turn_a_lat_mps2 <= 0.0:
            raise ValueError("tight_turn_a_lat_mps2 must be positive when set")
        if gains.fly_through_turn_angle_start_deg < 0.0:
            raise ValueError("fly_through_turn_angle_start_deg cannot be negative")
        if gains.fly_through_turn_angle_full_deg < gains.fly_through_turn_angle_start_deg:
            raise ValueError(
                "fly_through_turn_angle_full_deg cannot be less than fly_through_turn_angle_start_deg"
            )
        if gains.fly_through_next_gate_distance_start_m < 0.0:
            raise ValueError("fly_through_next_gate_distance_start_m cannot be negative")
        if (
            gains.fly_through_next_gate_distance_full_m
            < gains.fly_through_next_gate_distance_start_m
        ):
            raise ValueError(
                "fly_through_next_gate_distance_full_m cannot be less than fly_through_next_gate_distance_start_m"
            )
        if not 0.0 <= gains.fly_through_lookahead_min_scale <= 1.0:
            raise ValueError("fly_through_lookahead_min_scale must be in [0, 1]")
        if gains.max_specific_thrust_mps2 <= 0.0:
            raise ValueError("max_specific_thrust_mps2 must be positive")
        if gains.max_commanded_acceleration_mps2 <= 0.0:
            raise ValueError("max_commanded_acceleration_mps2 must be positive")
        if gains.gate_switch_acceleration_ramp_s < 0.0:
            raise ValueError("gate_switch_acceleration_ramp_s cannot be negative")
        if gains.max_commanded_jerk_mps3 < 0.0:
            raise ValueError("max_commanded_jerk_mps3 cannot be negative")
        if gains.launch_speed_ramp_s < 0.0:
            raise ValueError("launch_speed_ramp_s cannot be negative")
        if gains.gate_cross_velocity_cancel_near_m < 0.0:
            raise ValueError("gate_cross_velocity_cancel_near_m cannot be negative")
        if gains.gate_cross_velocity_cancel_far_m < 0.0:
            raise ValueError("gate_cross_velocity_cancel_far_m cannot be negative")
        if gains.gate_cross_velocity_cancel_far_m < gains.gate_cross_velocity_cancel_near_m:
            raise ValueError(
                "gate_cross_velocity_cancel_far_m cannot be less than gate_cross_velocity_cancel_near_m"
            )
        if gains.min_normalized_thrust > gains.max_normalized_thrust:
            raise ValueError("min_normalized_thrust cannot exceed max_normalized_thrust")
        if not 0.0 <= gains.approach_gain_min_scale <= 1.0:
            raise ValueError("approach_gain_min_scale must be in [0, 1]")
        for name in (
            "kp_speed",
            "kp_dir",
            "kd_dir",
            "kp_lat",
            "kd_lat",
            "kp_z",
            "kd_z",
            "gate_cross_velocity_damping",
        ):
            if getattr(gains, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")


def _attitude_from_body_z_and_heading(
    body_z_down_ned: np.ndarray,
    heading_ned: np.ndarray,
) -> QuatWxyz:
    z_body = _unit_or(body_z_down_ned, np.array((0.0, 0.0, 1.0), dtype=float))
    x_projected = heading_ned - z_body * float(np.dot(heading_ned, z_body))
    x_body = _unit_or(x_projected, _orthogonal_horizontal(z_body))
    y_body = np.cross(z_body, x_body)
    y_body = _unit_or(y_body, np.array((0.0, 1.0, 0.0), dtype=float))
    x_body = np.cross(y_body, z_body)
    return quaternion_from_rotation_matrix(np.column_stack((x_body, y_body, z_body)))


def _lateral_position_error(
    position: np.ndarray,
    gate: np.ndarray,
    aim_dir: np.ndarray,
) -> np.ndarray:
    error = position - gate
    return error - float(np.dot(error, aim_dir)) * aim_dir


def _lateral_velocity(velocity: np.ndarray, aim_dir: np.ndarray) -> np.ndarray:
    return velocity - float(np.dot(velocity, aim_dir)) * aim_dir


def _quat_conjugate(quaternion: QuatWxyz) -> QuatWxyz:
    qw, qx, qy, qz = quat_wxyz(quaternion)
    return (qw, -qx, -qy, -qz)


def _quat_multiply(left: QuatWxyz, right: QuatWxyz) -> QuatWxyz:
    lw, lx, ly, lz = quat_wxyz(left)
    rw, rx, ry, rz = quat_wxyz(right)
    return quat_wxyz(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return 0.0 if value <= edge0 else 1.0
    t = float(np.clip((float(value) - edge0) / (edge1 - edge0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if limit <= 0.0 or norm <= limit or norm <= 1e-12:
        return vector.astype(float)
    return vector.astype(float) * (float(limit) / norm)


def _unit_or(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm > 1e-12:
        return array / norm
    fallback_array = np.asarray(fallback, dtype=float)
    fallback_norm = float(np.linalg.norm(fallback_array))
    if fallback_norm <= 1e-12:
        return np.array((1.0, 0.0, 0.0), dtype=float)
    return fallback_array / fallback_norm


def _orthogonal_horizontal(vector: np.ndarray) -> np.ndarray:
    candidate = np.array((1.0, 0.0, 0.0), dtype=float)
    candidate = candidate - vector * float(np.dot(candidate, vector))
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-9:
        candidate = np.array((0.0, 1.0, 0.0), dtype=float)
        candidate = candidate - vector * float(np.dot(candidate, vector))
        norm = float(np.linalg.norm(candidate))
    return candidate / norm


def _gate_key(gate: GateRecord) -> str:
    if gate.gate_id is not None:
        return str(gate.gate_id)
    if gate.sequence is not None:
        return f"sequence:{gate.sequence}"
    return "unknown"


def _vec3(value: Vec3, name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array
