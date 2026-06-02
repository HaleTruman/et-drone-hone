"""Small XYZ-homing SET_ATTITUDE_TARGET producer for ARMED-mode hover."""

from __future__ import annotations

import numpy as np


class HoverPIDController:
    """Request attitude and collective thrust to return toward a target position."""

    def __init__(self, *, mass_kg: float, gravity_mps2: float, thrust_coefficient: float, dt_s: float):
        self.hover_motor_command = float(np.sqrt(mass_kg * gravity_mps2 / (4.0 * thrust_coefficient)))
        self.dt_s = float(dt_s)
        self.kp = 0.003
        self.ki = 0.0002
        self.kd = 0.0
        self.velocity_damping = 0.025
        self.target_position_local_ned_m = np.zeros(3, dtype=float)
        self.position_gain = np.array([0.6, 0.6, 0.8], dtype=float)
        self.velocity_gain = np.array([0.6, 0.6, 0.8], dtype=float)
        self._integral = 0.0
        self._previous_error = 0.0

    def update(
        self,
        acceleration_local_ned_mps2: tuple[float, float, float],
        position_local_ned_m: np.ndarray,
        velocity_local_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, float]:
        error = float(acceleration_local_ned_mps2[2])
        self._integral = float(np.clip(self._integral + error * self.dt_s, -1.0, 1.0))
        derivative = (error - self._previous_error) / self.dt_s
        self._previous_error = error
        position = np.asarray(position_local_ned_m, dtype=float)
        velocity = np.asarray(velocity_local_ned_mps, dtype=float)
        desired_acceleration = (
            self.position_gain * (self.target_position_local_ned_m - position)
            - self.velocity_gain * velocity
        )
        roll = float(np.clip(desired_acceleration[1] / 9.81, -0.25, 0.25))
        pitch = float(np.clip(-desired_acceleration[0] / 9.81, -0.25, 0.25))
        quaternion = self._quaternion_from_roll_pitch(roll, pitch)
        command = (
            self.hover_motor_command
            + self.kp * error
            + self.ki * self._integral
            + self.kd * derivative
            - 0.035 * desired_acceleration[2]
            + self.velocity_damping * float(velocity[2])
        )
        return quaternion, float(np.clip(command, 0.0, 1.0))

    @staticmethod
    def _quaternion_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=float)
