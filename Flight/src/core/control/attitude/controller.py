from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.coordinates import normalize_quaternion
from core.schemas import VehicleState

PARAMS_PATH = Path(__file__).resolve().parents[2] / "quadrotor" / "params.yaml"


@dataclass(frozen=True)
class AttitudeMotorConfig:
    attitude_accel_gain: tuple[float, float, float] = (1.8, 1.8, 0.0)
    rate_accel_damping: tuple[float, float, float] = (9.0, 10.0, 1.2)
    max_moment_nm: tuple[float, float, float] = (0.055, 0.080, 0.002)
    moment_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_motor_delta_per_update: float = 0.012
    min_motor_command: float = 0.12
    max_motor_command: float = 0.45


class AttitudeMotorController:
    """Simple attitude/thrust to normalized motor-command mixer."""

    def __init__(
        self,
        *,
        config: AttitudeMotorConfig | None = None,
        params_path: str | Path | None = None,
    ) -> None:
        self.config = config or AttitudeMotorConfig()
        params = yaml.safe_load(Path(params_path or PARAMS_PATH).read_text()) or {}
        self.kf = float(params["kf"])
        self.inertia = np.diag(
            np.array((float(params["Ixx"]), float(params["Iyy"]), float(params["Izz"])), dtype=float)
        )
        self.allocation = _allocation_matrix(params)
        self._previous_motor_commands: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_motor_commands = None

    def compute_motor_commands(self, vehicle_state: VehicleState, attitude_target: Mapping[str, Any]) -> list[float]:
        desired_quaternion = normalize_quaternion(attitude_target["quaternion"])
        current_quaternion = normalize_quaternion(vehicle_state.attitude_quaternion)
        attitude_error = _attitude_error_vector(current_quaternion, desired_quaternion)
        body_rates = np.asarray(vehicle_state.body_rates_frd_rps, dtype=float)

        desired_angular_accel = (
            np.asarray(self.config.attitude_accel_gain, dtype=float) * attitude_error
            - np.asarray(self.config.rate_accel_damping, dtype=float) * body_rates
        )
        moment = self.inertia @ desired_angular_accel
        moment = np.clip(
            moment,
            -np.asarray(self.config.max_moment_nm, dtype=float),
            np.asarray(self.config.max_moment_nm, dtype=float),
        )
        moment = moment * np.asarray(self.config.moment_sign, dtype=float)
        base = float(np.clip(float(attitude_target["thrust"]), 0.0, 1.0))
        total_thrust_n = 4.0 * self.kf * base * base
        motor_thrusts_n = np.linalg.solve(self.allocation, np.array((total_thrust_n, *moment), dtype=float))
        motors = np.sqrt(np.clip(motor_thrusts_n / self.kf, 0.0, None))
        motors = np.clip(motors, self.config.min_motor_command, self.config.max_motor_command)
        if self._previous_motor_commands is None:
            self._previous_motor_commands = motors
            return motors.tolist()

        max_delta = float(self.config.max_motor_delta_per_update)
        motors = self._previous_motor_commands + np.clip(
            motors - self._previous_motor_commands,
            -max_delta,
            max_delta,
        )
        motors = np.clip(motors, self.config.min_motor_command, self.config.max_motor_command)
        self._previous_motor_commands = motors
        return motors.tolist()


def _allocation_matrix(params: Mapping[str, Any]) -> np.ndarray:
    arm_lengths = np.asarray(params["l"], dtype=float)
    vertical_offsets = np.asarray(params.get("h", (0.0, 0.0, 0.0, 0.0)), dtype=float)
    spin_directions = np.asarray(params["d"], dtype=float)
    km_over_kf = float(params["km"]) / float(params["kf"])
    c = np.sqrt(2.0) / 2.0
    rho = np.array(
        [
            [arm_lengths[0] * c, arm_lengths[0] * c, vertical_offsets[0]],
            [-arm_lengths[1] * c, arm_lengths[1] * c, vertical_offsets[1]],
            [-arm_lengths[2] * c, -arm_lengths[2] * c, vertical_offsets[2]],
            [arm_lengths[3] * c, -arm_lengths[3] * c, vertical_offsets[3]],
        ],
        dtype=float,
    )
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            -rho[:, 1],
            rho[:, 0],
            -km_over_kf * spin_directions,
        ],
        dtype=float,
    )


def _attitude_error_vector(current_quaternion: np.ndarray, desired_quaternion: np.ndarray) -> np.ndarray:
    error = _quaternion_multiply(desired_quaternion, _quaternion_conjugate(current_quaternion))
    if error[0] < 0.0:
        error = -error
    return 2.0 * error[1:4]


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]), dtype=float)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=float,
    )
