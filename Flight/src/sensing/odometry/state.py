"""Stateful vehicle state estimator."""

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from core.coordinates import quat_wxyz, vec3
from core.coordinates import euler_from_quaternion, normalize_quaternion, rotate_vector
from core.schemas import MavlinkHighresImu, MavlinkTelemetry, QuatWxyz, Vec3, VehicleState


ZERO_VEC3 = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
GRAVITY_LOCAL_NED_MPS2 = (0.0, 0.0, 9.80665)


class VehicleStateEstimator:
    """Mutable owner for the latest estimated vehicle state."""

    def __init__(self, vehicle_state: VehicleState | None = None):
        self.sim_time_ns = 0
        self.initialized = False
        self.position_local_ned_m: Vec3 = ZERO_VEC3
        self.velocity_local_ned_mps: Vec3 = ZERO_VEC3
        self.attitude_quaternion: QuatWxyz = IDENTITY_QUATERNION
        self.angular_velocity_body_frd_rps: Vec3 = ZERO_VEC3
        self.angular_acceleration_body_frd_rps2: Vec3 = ZERO_VEC3
        self.acceleration_body_frd_mps2: Vec3 = ZERO_VEC3
        self.acceleration_rest_body_frd_mps2: Vec3 | None = None
        self.acceleration_local_ned_mps2: Vec3 = ZERO_VEC3
        self.gyro_bias_body_frd_rps: Vec3 = ZERO_VEC3
        self.last_imu_time_boot_us: int | None = None
        if vehicle_state is not None:
            self.update_state(vehicle_state)

    @property
    def state(self) -> VehicleState:
        return VehicleState(
            sim_time_ns=self.sim_time_ns,
            position_local_ned_m=self.position_local_ned_m,
            velocity_local_ned_mps=self.velocity_local_ned_mps,
            attitude_quaternion=self.attitude_quaternion,
            body_rates_frd_rps=self.angular_velocity_body_frd_rps,
            acceleration_local_ned_mps2=self.acceleration_local_ned_mps2,
        )

    def attitude_euler_local_ned(self, unit: str = "rad") -> Vec3:
        """Return current local-NED attitude as roll, pitch, yaw."""

        euler_rad = euler_from_quaternion(self.attitude_quaternion)
        if unit == "rad":
            return vec3(euler_rad)
        if unit == "deg":
            return vec3(math.degrees(value) for value in euler_rad)
        raise ValueError('unit must be "rad" or "deg"')

    def reset(self, sim_time_ns: int = 0, vehicle_state: VehicleState | None = None) -> VehicleState:
        self.sim_time_ns = sim_time_ns
        self.position_local_ned_m = ZERO_VEC3
        self.velocity_local_ned_mps = ZERO_VEC3
        self.attitude_quaternion = IDENTITY_QUATERNION
        self.angular_velocity_body_frd_rps = ZERO_VEC3
        self.angular_acceleration_body_frd_rps2 = ZERO_VEC3
        self.acceleration_body_frd_mps2 = ZERO_VEC3
        self.acceleration_rest_body_frd_mps2 = None
        self.acceleration_local_ned_mps2 = ZERO_VEC3
        self.gyro_bias_body_frd_rps = ZERO_VEC3
        self.last_imu_time_boot_us = None
        self.initialized = False
        if vehicle_state is not None:
            return self.update_state(vehicle_state)
        return self.state

    def update_state(self, vehicle_state: VehicleState) -> VehicleState:
        self.sim_time_ns = int(vehicle_state.sim_time_ns)
        self.position_local_ned_m = vec3(vehicle_state.position_local_ned_m)
        self.velocity_local_ned_mps = vec3(vehicle_state.velocity_local_ned_mps)
        self.attitude_quaternion = quat_wxyz(vehicle_state.attitude_quaternion)
        self.angular_velocity_body_frd_rps = vec3(vehicle_state.body_rates_frd_rps)
        self.acceleration_local_ned_mps2 = vec3(vehicle_state.acceleration_local_ned_mps2)
        return self.state

    def update(self, imu_data_t: MavlinkHighresImu | None) -> VehicleState | None:
        if imu_data_t is None:
            return None
        return self.update_from_imu(imu_data_t)

    def update_telemetry(self, telemetry: MavlinkTelemetry | None) -> MavlinkTelemetry | None:
        vehicle_state = self.update(None if telemetry is None else telemetry.imu)
        if telemetry is None or vehicle_state is None:
            return None
        return MavlinkTelemetry(
            sim_time_ns=vehicle_state.sim_time_ns,
            vehicle_state=vehicle_state,
            imu=telemetry.imu,
            system_status=telemetry.system_status,
            reset_count=telemetry.reset_count,
            raw={**telemetry.raw, "vehicle_state_source": "vehicle_state_estimator_highres_imu"},
        )

    def wait_for_update(
        self,
        telemetry_source: Callable[[], MavlinkTelemetry | None],
        *,
        timeout_s: float,
        idle_sleep_s: float = 0.02,
    ) -> MavlinkTelemetry:
        deadline_s = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline_s:
            telemetry = telemetry_source()
            estimated = self.update_telemetry(telemetry)
            if estimated is not None:
                return estimated
            time.sleep(idle_sleep_s)
        raise TimeoutError("No vehicle state estimate received before timeout.")

    def wait_until_stable(
        self,
        telemetry_source: Callable[[], MavlinkTelemetry | None],
        *,
        timeout_s: float,
        stable_s: float,
        stable_max_speed_mps: float,
        idle_sleep_s: float = 0.05,
    ) -> MavlinkTelemetry:
        deadline_s = time.perf_counter() + float(timeout_s)
        stable_start_s: float | None = None
        latest: MavlinkTelemetry | None = None
        while time.perf_counter() < deadline_s:
            telemetry = telemetry_source()
            estimated = self.update_telemetry(telemetry)
            if estimated is None or estimated.vehicle_state is None:
                stable_start_s = None
                time.sleep(idle_sleep_s)
                continue
            latest = estimated
            speed = float(np.linalg.norm(np.asarray(estimated.vehicle_state.velocity_local_ned_mps, dtype=float)))
            if speed <= float(stable_max_speed_mps):
                if stable_start_s is None:
                    stable_start_s = time.perf_counter()
                if time.perf_counter() - stable_start_s >= float(stable_s):
                    return latest
            else:
                stable_start_s = None
            time.sleep(idle_sleep_s)
        raise TimeoutError("Vehicle state did not remain stable before timeout.")

    def initialize_from_imu(self, imu_data_t: MavlinkHighresImu) -> VehicleState:
        """Initialize attitude from accelerometer gravity direction and cache the sample."""

        self.sim_time_ns = int(imu_data_t.time_boot_us) * 1_000
        self.last_imu_time_boot_us = int(imu_data_t.time_boot_us)
        self.acceleration_body_frd_mps2 = vec3(imu_data_t.acceleration_body_frd_mps2)
        self.angular_velocity_body_frd_rps = self._corrected_gyro(imu_data_t.gyro_body_frd_rps)
        self.attitude_quaternion = _attitude_from_accelerometer(self.acceleration_body_frd_mps2)
        self.acceleration_local_ned_mps2 = vec3(
            np.asarray(rotate_vector(self.attitude_quaternion, self.acceleration_body_frd_mps2), dtype=float)
            + np.asarray(GRAVITY_LOCAL_NED_MPS2, dtype=float)
        )
        self._set_initialized()
        print("Vehicle state initialized...")
        
        return self.state

    def initialize_from_stationary_imu_samples(self, imu_samples: Sequence[MavlinkHighresImu]) -> VehicleState:
        if not imu_samples:
            raise ValueError("imu_samples must contain at least one sample")

        accelerations = np.asarray([sample.acceleration_body_frd_mps2 for sample in imu_samples], dtype=float)
        gyros = np.asarray([sample.gyro_body_frd_rps for sample in imu_samples], dtype=float)
        mean_acceleration = accelerations.mean(axis=0)
        mean_gyro = gyros.mean(axis=0)
        latest_sample = imu_samples[-1]

        self.sim_time_ns = int(latest_sample.time_boot_us) * 1_000
        self.last_imu_time_boot_us = int(latest_sample.time_boot_us)
        self.acceleration_rest_body_frd_mps2 = vec3(mean_acceleration)
        self.acceleration_body_frd_mps2 = vec3(mean_acceleration)
        self.gyro_bias_body_frd_rps = vec3(mean_gyro)
        self.angular_velocity_body_frd_rps = ZERO_VEC3
        self.angular_acceleration_body_frd_rps2 = ZERO_VEC3
        self.attitude_quaternion = _attitude_from_accelerometer(self.acceleration_rest_body_frd_mps2)
        self.acceleration_local_ned_mps2 = vec3(
            np.asarray(rotate_vector(self.attitude_quaternion, self.acceleration_body_frd_mps2), dtype=float)
            + np.asarray(GRAVITY_LOCAL_NED_MPS2, dtype=float)
        )
        self._set_initialized()
        print("Vehicle state initialized from stationary IMU calibration...")
        return self.state
    
    def _set_initialized(self):
        self.initialized = True

    def update_from_imu(self, imu_data_t: MavlinkHighresImu) -> VehicleState:
        previous_time_boot_us = self.last_imu_time_boot_us
        previous_velocity = self.velocity_local_ned_mps
        previous_angular_velocity = self.angular_velocity_body_frd_rps

        if previous_time_boot_us is None:
            return self.initialize_from_imu(imu_data_t)

        self.sim_time_ns = int(imu_data_t.time_boot_us) * 1_000
        self.last_imu_time_boot_us = int(imu_data_t.time_boot_us)
        self.acceleration_body_frd_mps2 = vec3(imu_data_t.acceleration_body_frd_mps2)
        self.angular_velocity_body_frd_rps = self._corrected_gyro(imu_data_t.gyro_body_frd_rps)

        dt_s = (self.last_imu_time_boot_us - previous_time_boot_us) / 1_000_000
        if dt_s <= 0.0:
            return self.state

        angular_velocity = np.asarray(self.angular_velocity_body_frd_rps, dtype=float)
        previous_rates = np.asarray(previous_angular_velocity, dtype=float)
        self.angular_acceleration_body_frd_rps2 = vec3((angular_velocity - previous_rates) / dt_s)
        self.attitude_quaternion = _integrate_attitude_quaternion(
            self.attitude_quaternion,
            self.angular_velocity_body_frd_rps,
            dt_s,
        )

        acceleration_local_ned = np.asarray(
            rotate_vector(
                self.attitude_quaternion,
                self.acceleration_body_frd_mps2,
            ),
            dtype=float,
        ) + np.asarray(GRAVITY_LOCAL_NED_MPS2, dtype=float)
        self.acceleration_local_ned_mps2 = vec3(acceleration_local_ned)

        acceleration = np.asarray(self.acceleration_local_ned_mps2, dtype=float)
        velocity = np.asarray(previous_velocity, dtype=float)
        position = np.asarray(self.position_local_ned_m, dtype=float)

        self.position_local_ned_m = vec3(position + velocity * dt_s + 0.5 * acceleration * dt_s * dt_s)
        self.velocity_local_ned_mps = vec3(velocity + acceleration * dt_s)
        return self.state

    def _corrected_gyro(self, gyro_body_frd_rps: Vec3) -> Vec3:
        return vec3(np.asarray(gyro_body_frd_rps, dtype=float) - np.asarray(self.gyro_bias_body_frd_rps, dtype=float))


def _attitude_from_accelerometer(acceleration_body_frd_mps2: Vec3) -> QuatWxyz:
    acceleration = np.asarray(acceleration_body_frd_mps2, dtype=float)
    norm = float(np.linalg.norm(acceleration))
    if norm <= 1e-9:
        return IDENTITY_QUATERNION
    measured_up_local = np.array((0.0, 0.0, -1.0), dtype=float)
    measured_up_body = acceleration / norm
    return _quaternion_between_vectors(measured_up_body, measured_up_local)


def _integrate_attitude_quaternion(
    attitude_quaternion: QuatWxyz,
    angular_velocity_body_frd_rps: Vec3,
    dt_s: float,
) -> QuatWxyz:
    q = np.asarray(attitude_quaternion, dtype=float)
    omega = np.asarray((0.0, *angular_velocity_body_frd_rps), dtype=float)
    q_dot = 0.5 * _quaternion_multiply(q, omega)
    return quat_wxyz(normalize_quaternion(q + q_dot * float(dt_s)))


def _quaternion_between_vectors(source: np.ndarray, target: np.ndarray) -> QuatWxyz:
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    dot = float(np.dot(source, target))
    if dot < -0.999999:
        axis = np.cross(source, np.array((1.0, 0.0, 0.0), dtype=float))
        if np.linalg.norm(axis) <= 1e-9:
            axis = np.cross(source, np.array((0.0, 1.0, 0.0), dtype=float))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return quat_wxyz((0.0, *axis))
    axis = np.cross(source, target)
    quaternion = np.array((1.0 + dot, axis[0], axis[1], axis[2]), dtype=float)
    return quat_wxyz(quaternion)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )
