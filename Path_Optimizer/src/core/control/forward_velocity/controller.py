from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardVelocityControllerGains:
    velocity_p: float = 0.35
    velocity_i: float = 0.0
    attitude_p: float = 0.8
    altitude_p: float = 1.4
    altitude_d: float = 1.6
    altitude_i: float = 0.12


class ForwardVelocityAltitudeController:
    """Cascaded local-NED forward-velocity and altitude-hold controller.

    The controller follows the standard multirotor pattern:
    horizontal velocity error -> desired horizontal acceleration -> desired
    roll/pitch, and altitude error -> desired vertical acceleration -> thrust.
    Thrust is computed from the configured hover trim and requested vertical
    acceleration, not from a hand-tuned fixed throttle value.
    """

    def __init__(
        self,
        *,
        dt_s: float = 1.0 / 30.0,
        neutral_thrust: float = 0.5,
        gravity_mps2: float = 9.80665,
        gains: ForwardVelocityControllerGains | None = None,
        max_tilt_rad: float = 0.10,
        max_tilt_rate_rps: float = 0.20,
        max_horizontal_accel_mps2: float = 1.0,
        max_vertical_accel_mps2: float = 8.0,
        max_target_velocity_rate_mps2: float = 0.5,
        altitude_priority_deadband_m: float = 0.15,
        climb_priority_deadband_mps: float = 0.2,
        max_velocity_integral_mps: float = 3.0,
        max_altitude_integral_m_s: float = 4.0,
    ) -> None:
        self.dt_s = float(dt_s)
        self.neutral_thrust = float(neutral_thrust)
        self.gravity_mps2 = float(gravity_mps2)
        self.gains = gains or ForwardVelocityControllerGains()
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_tilt_rate_rps = float(max_tilt_rate_rps)
        self.max_horizontal_accel_mps2 = float(max_horizontal_accel_mps2)
        self.max_vertical_accel_mps2 = float(max_vertical_accel_mps2)
        self.max_target_velocity_rate_mps2 = float(max_target_velocity_rate_mps2)
        self.altitude_priority_deadband_m = float(altitude_priority_deadband_m)
        self.climb_priority_deadband_mps = float(climb_priority_deadband_mps)
        self.max_velocity_integral_mps = float(max_velocity_integral_mps)
        self.max_altitude_integral_m_s = float(max_altitude_integral_m_s)
        self._velocity_integral = np.zeros(2, dtype=float)
        self._altitude_integral = 0.0
        self._last_roll_pitch = np.zeros(2, dtype=float)
        self._target_velocity_local_ned_mps = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self._velocity_integral[:] = 0.0
        self._altitude_integral = 0.0
        self._last_roll_pitch[:] = 0.0
        self._target_velocity_local_ned_mps[:] = 0.0

    def update(
        self,
        *,
        position_local_ned_m: np.ndarray | tuple[float, float, float],
        velocity_local_ned_mps: np.ndarray | tuple[float, float, float],
        attitude_quaternion: np.ndarray | tuple[float, float, float, float],
        target_altitude_ned_m: float,
        target_velocity_local_ned_mps: np.ndarray | tuple[float, float, float],
        trim_thrust: float | None = None,
        yaw_rad: float | None = None,
    ) -> dict[str, object]:
        position = np.asarray(position_local_ned_m, dtype=float)
        velocity = np.asarray(velocity_local_ned_mps, dtype=float)
        target_velocity_request = np.asarray(target_velocity_local_ned_mps, dtype=float)
        measured_roll, measured_pitch, measured_yaw = self._euler_from_quaternion(attitude_quaternion)
        yaw = measured_yaw if yaw_rad is None else float(yaw_rad)
        target_velocity = self._slew_target_velocity(target_velocity_request)

        velocity_error = target_velocity[:2] - velocity[:2]
        self._velocity_integral += velocity_error * self.dt_s
        self._velocity_integral = np.clip(
            self._velocity_integral,
            -self.max_velocity_integral_mps,
            self.max_velocity_integral_mps,
        )
        horizontal_accel = (
            self.gains.velocity_p * velocity_error
            + self.gains.velocity_i * self._velocity_integral
        )
        horizontal_accel = self._limit_norm(horizontal_accel, self.max_horizontal_accel_mps2)

        altitude_error_ned = float(target_altitude_ned_m) - float(position[2])
        vertical_velocity_error = 0.0 - float(velocity[2])
        self._altitude_integral = float(
            np.clip(
                self._altitude_integral + altitude_error_ned * self.dt_s,
                -self.max_altitude_integral_m_s,
                self.max_altitude_integral_m_s,
            )
        )
        vertical_accel_ned = (
            self.gains.altitude_p * altitude_error_ned
            + self.gains.altitude_d * vertical_velocity_error
            + self.gains.altitude_i * self._altitude_integral
        )
        vertical_accel_ned = float(
            np.clip(vertical_accel_ned, -self.max_vertical_accel_mps2, self.max_vertical_accel_mps2)
        )
        altitude_priority_active = (
            altitude_error_ned > self.altitude_priority_deadband_m
            or vertical_velocity_error > self.climb_priority_deadband_mps
        )
        if altitude_priority_active:
            self._velocity_integral[:] = 0.0

        roll_cmd, pitch_cmd = self._acceleration_to_roll_pitch(horizontal_accel, yaw)
        roll_cmd -= self.gains.attitude_p * measured_roll
        pitch_cmd -= self.gains.attitude_p * measured_pitch
        roll_cmd = float(np.clip(roll_cmd, -self.max_tilt_rad, self.max_tilt_rad))
        pitch_cmd = float(np.clip(pitch_cmd, -self.max_tilt_rad, self.max_tilt_rad))
        roll_cmd, pitch_cmd = self._slew_roll_pitch(roll_cmd, pitch_cmd)
        trim = self.neutral_thrust if trim_thrust is None else float(np.clip(trim_thrust, 0.0, 1.0))
        thrust = self._vertical_acceleration_to_thrust(vertical_accel_ned, roll_cmd, pitch_cmd, trim)
        if altitude_priority_active or altitude_error_ned > 0.0 or vertical_velocity_error > 0.0:
            thrust = min(thrust, trim)

        command_yaw = 0.0
        quaternion = self._quaternion_from_roll_pitch_yaw(roll_cmd, pitch_cmd, command_yaw)
        return {
            "quaternion": quaternion.tolist(),
            "thrust": thrust,
            "attitude_type_mask": 7,
            "mode": "forward_velocity_altitude_hold",
            "target_velocity_local_ned_mps": target_velocity.tolist(),
            "target_altitude_ned_m": float(target_altitude_ned_m),
            "trim_thrust": trim,
            "desired_acceleration_local_ned_mps2": [
                float(horizontal_accel[0]),
                float(horizontal_accel[1]),
                vertical_accel_ned,
            ],
            "roll_pitch_yaw_rad": [roll_cmd, pitch_cmd, command_yaw],
            "measured_roll_pitch_yaw_rad": [measured_roll, measured_pitch, measured_yaw],
            "target_velocity_request_local_ned_mps": target_velocity_request.tolist(),
            "altitude_priority_active": altitude_priority_active,
        }

    def _acceleration_to_roll_pitch(self, accel_ne_mps2: np.ndarray, yaw_rad: float) -> tuple[float, float]:
        north_accel, east_accel = np.asarray(accel_ne_mps2, dtype=float)
        forward_accel = north_accel * np.cos(yaw_rad) + east_accel * np.sin(yaw_rad)
        right_accel = -north_accel * np.sin(yaw_rad) + east_accel * np.cos(yaw_rad)
        roll = np.arctan2(right_accel, self.gravity_mps2)
        pitch = np.arctan2(-forward_accel, self.gravity_mps2)
        return float(roll), float(pitch)

    def _vertical_acceleration_to_thrust(self, accel_down_mps2: float, roll: float, pitch: float, trim_thrust: float) -> float:
        tilt_compensation = max(np.cos(roll) * np.cos(pitch), 0.5)
        thrust = float(trim_thrust) * (1.0 - float(accel_down_mps2) / self.gravity_mps2) / tilt_compensation
        return float(np.clip(thrust, 0.0, 1.0))

    def _slew_roll_pitch(self, roll: float, pitch: float) -> tuple[float, float]:
        desired = np.array([roll, pitch], dtype=float)
        max_delta = self.max_tilt_rate_rps * self.dt_s
        delta = np.clip(desired - self._last_roll_pitch, -max_delta, max_delta)
        limited = self._last_roll_pitch + delta
        self._last_roll_pitch = limited
        return float(limited[0]), float(limited[1])

    def _slew_target_velocity(self, target_velocity: np.ndarray) -> np.ndarray:
        desired = np.asarray(target_velocity, dtype=float).copy()
        desired[2] = 0.0
        max_delta = self.max_target_velocity_rate_mps2 * self.dt_s
        delta = desired[:2] - self._target_velocity_local_ned_mps[:2]
        delta = self._limit_norm(delta, max_delta)
        self._target_velocity_local_ned_mps[:2] += delta
        self._target_velocity_local_ned_mps[2] = 0.0
        return self._target_velocity_local_ned_mps.copy()

    @staticmethod
    def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm <= 1e-12:
            return vector
        return vector / norm * max_norm

    @staticmethod
    def _quaternion_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )

    @staticmethod
    def _euler_from_quaternion(quaternion: np.ndarray | tuple[float, float, float, float]) -> tuple[float, float, float]:
        qw, qx, qy, qz = np.asarray(quaternion, dtype=float)
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return float(roll), float(pitch), float(yaw)
