"""Stateful vehicle state estimator."""

import time
from collections.abc import Callable

import numpy as np

from core.coordinates import quat_wxyz, vec3
from core.coordinates import normalize_quaternion, rotate_vector
from core.schemas import MavlinkHighresImu, MavlinkTelemetry, QuatWxyz, Vec3, VehicleState


ZERO_VEC3 = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
GRAVITY_LOCAL_NED_MPS2 = (0.0, 0.0, 9.80665)


class VehicleStateEstimator:
    """Mutable owner for the latest estimated vehicle state."""

    def __init__(self, vehicle_state: VehicleState | None = None):
        self.sim_time_ns = 0
        self.position_local_ned_m: Vec3 = ZERO_VEC3
        self.velocity_local_ned_mps: Vec3 = ZERO_VEC3
        self.attitude_quaternion: QuatWxyz = IDENTITY_QUATERNION
        self.angular_velocity_body_frd_rps: Vec3 = ZERO_VEC3
        self.angular_acceleration_body_frd_rps2: Vec3 = ZERO_VEC3
        self.acceleration_body_frd_mps2: Vec3 = ZERO_VEC3
        self.acceleration_local_ned_mps2: Vec3 = ZERO_VEC3
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

    def reset(self, vehicle_state: VehicleState | None = None) -> VehicleState:
        self.sim_time_ns = 0
        self.position_local_ned_m = ZERO_VEC3
        self.velocity_local_ned_mps = ZERO_VEC3
        self.attitude_quaternion = IDENTITY_QUATERNION
        self.angular_velocity_body_frd_rps = ZERO_VEC3
        self.angular_acceleration_body_frd_rps2 = ZERO_VEC3
        self.acceleration_body_frd_mps2 = ZERO_VEC3
        self.acceleration_local_ned_mps2 = ZERO_VEC3
        self.last_imu_time_boot_us = None
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

    def update(self, imu: MavlinkHighresImu | None) -> VehicleState | None:
        if imu is None:
            return None
        return self.update_from_imu(imu)

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

    def initialize_from_imu(self, latest_imu: MavlinkHighresImu) -> VehicleState:
        """Initialize attitude from accelerometer gravity direction and cache the sample."""

        self.sim_time_ns = int(latest_imu.time_boot_us) * 1_000
        self.last_imu_time_boot_us = int(latest_imu.time_boot_us)
        self.acceleration_body_frd_mps2 = vec3(latest_imu.acceleration_body_frd_mps2)
        self.angular_velocity_body_frd_rps = vec3(latest_imu.gyro_body_frd_rps)
        self.attitude_quaternion = _attitude_from_accelerometer(self.acceleration_body_frd_mps2)
        self.acceleration_local_ned_mps2 = vec3(
            np.asarray(rotate_vector(self.attitude_quaternion, self.acceleration_body_frd_mps2), dtype=float)
            + np.asarray(GRAVITY_LOCAL_NED_MPS2, dtype=float)
        )
        return self.state

    def update_from_imu(self, latest_imu: MavlinkHighresImu) -> VehicleState:
        previous_time_boot_us = self.last_imu_time_boot_us
        previous_velocity = self.velocity_local_ned_mps
        previous_angular_velocity = self.angular_velocity_body_frd_rps

        if previous_time_boot_us is None:
            return self.initialize_from_imu(latest_imu)

        self.sim_time_ns = int(latest_imu.time_boot_us) * 1_000
        self.last_imu_time_boot_us = int(latest_imu.time_boot_us)
        self.acceleration_body_frd_mps2 = vec3(latest_imu.acceleration_body_frd_mps2)
        self.angular_velocity_body_frd_rps = vec3(latest_imu.gyro_body_frd_rps)
        acceleration_local_ned = np.asarray(
            rotate_vector(
                self.attitude_quaternion,
                self.acceleration_body_frd_mps2,
            ),
            dtype=float,
        ) + np.asarray(GRAVITY_LOCAL_NED_MPS2, dtype=float)
        self.acceleration_local_ned_mps2 = vec3(
            acceleration_local_ned,
        )

        dt_s = (self.last_imu_time_boot_us - previous_time_boot_us) / 1_000_000
        if dt_s <= 0.0:
            return self.state

        acceleration = np.asarray(self.acceleration_local_ned_mps2, dtype=float)
        velocity = np.asarray(previous_velocity, dtype=float)
        position = np.asarray(self.position_local_ned_m, dtype=float)
        angular_velocity = np.asarray(self.angular_velocity_body_frd_rps, dtype=float)
        previous_rates = np.asarray(previous_angular_velocity, dtype=float)

        self.position_local_ned_m = vec3(position + velocity * dt_s + 0.5 * acceleration * dt_s * dt_s)
        self.velocity_local_ned_mps = vec3(velocity + acceleration * dt_s)
        self.angular_acceleration_body_frd_rps2 = vec3((angular_velocity - previous_rates) / dt_s)
        self.attitude_quaternion = _integrate_attitude_quaternion(
            self.attitude_quaternion,
            self.angular_velocity_body_frd_rps,
            dt_s,
        )
        return self.state


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
