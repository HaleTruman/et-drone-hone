from pathlib import Path

import numpy as np
import yaml


def load_params(path=None):
    params_path = Path(path) if path is not None else Path(__file__).with_name("params.yaml")
    params = yaml.safe_load(params_path.read_text())
    gains_path = params_path.with_name("gains.yaml")
    params.update(yaml.safe_load(gains_path.read_text()))
    return params


class RateController:
    def __init__(self, params, gains):
        self.kp = np.asarray(gains["kp"], dtype=float)
        self.ki = np.asarray(gains["ki"], dtype=float)
        self.kd = np.asarray(gains["kd"], dtype=float)
        self.max_rate_accel = np.asarray(params["max_rate_accel"], dtype=float)
        self.max_integral = np.asarray(params["max_integral"], dtype=float)
        self.hover_command = np.asarray(params["hover_motor_command"], dtype=float)
        self.motor_min = params["motor_min"]
        self.motor_max = params["motor_max"]
        self.dt = params["dt"]
        self.integral_error = np.zeros(3, dtype=float)
        self.previous_error = np.zeros(3, dtype=float)

    def reset(self):
        self.integral_error[:] = 0.0
        self.previous_error[:] = 0.0

    def update(self, state, desired_rates):
        current_rates = state[10:13]
        rate_error = desired_rates - current_rates
        self.integral_error += rate_error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)
        rate_error_dot = (rate_error - self.previous_error) / self.dt
        self.previous_error = rate_error

        pid = (
            self.kp * rate_error
            + self.ki * self.integral_error
            + self.kd * rate_error_dot
        )
        pid = np.clip(pid, -self.max_rate_accel, self.max_rate_accel)

        roll_cmd, pitch_cmd, yaw_cmd = pid
        motor_command = self.hover_command + np.array(
            [
                -roll_cmd + pitch_cmd - yaw_cmd,
                -roll_cmd - pitch_cmd + yaw_cmd,
                roll_cmd - pitch_cmd - yaw_cmd,
                roll_cmd + pitch_cmd + yaw_cmd,
            ],
            dtype=float,
        )
        return np.clip(motor_command, self.motor_min, self.motor_max)
