import numpy as np


class HoverPIDController:
    def __init__(self, dt_s=1.0 / 30.0, neutral_thrust=0.5):
        self.dt_s = float(dt_s)
        self.neutral_thrust = float(neutral_thrust)
        self.hover_motor_command = self.neutral_thrust
        self.accel_angle_gain = 0.025
        self.attitude_gain = 0.8
        self.thrust_gain = 0.04
        self.integral_gain = 0.002
        self.max_angle_rad = 0.25
        self._z_integral = 0.0

    def update(self, acceleration_local_ned_mps2, attitude_quaternion=(1.0, 0.0, 0.0, 0.0)):
        acceleration = np.asarray(acceleration_local_ned_mps2, dtype=float)
        roll, pitch, _ = self._euler_from_quaternion(attitude_quaternion)
        self._z_integral = float(np.clip(self._z_integral + acceleration[2] * self.dt_s, -5.0, 5.0))
        roll_cmd = np.clip(-self.accel_angle_gain * acceleration[1] - self.attitude_gain * roll, -self.max_angle_rad, self.max_angle_rad)
        pitch_cmd = np.clip(self.accel_angle_gain * acceleration[0] - self.attitude_gain * pitch, -self.max_angle_rad, self.max_angle_rad)
        thrust = self.neutral_thrust + self.thrust_gain * acceleration[2] + self.integral_gain * self._z_integral
        return self._quaternion_from_roll_pitch(roll_cmd, pitch_cmd), float(np.clip(thrust, 0.0, 1.0))

    @staticmethod
    def _quaternion_from_roll_pitch(roll, pitch):
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=float)

    @staticmethod
    def _euler_from_quaternion(quaternion):
        qw, qx, qy, qz = np.asarray(quaternion, dtype=float)
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return float(roll), float(pitch), float(yaw)
