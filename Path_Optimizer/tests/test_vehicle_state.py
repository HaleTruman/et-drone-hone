import math

import numpy as np

from core.coordinates import euler_from_quaternion
from core.schemas import MavlinkHighresImu
from sensing.odometry import VehicleState


def imu_sample(
    time_boot_us: int,
    *,
    acceleration_body_frd_mps2=(0.0, 0.0, 0.0),
    gyro_body_frd_rps=(0.0, 0.0, 0.0),
) -> MavlinkHighresImu:
    return MavlinkHighresImu(
        time_boot_us=time_boot_us,
        acceleration_body_frd_mps2=acceleration_body_frd_mps2,
        gyro_body_frd_rps=gyro_body_frd_rps,
    )


def test_first_imu_sample_updates_sensor_fields_without_integrating_state() -> None:
    state = VehicleState()

    odometry = state.update_from_imu(
        imu_sample(
            1_000_000,
            acceleration_body_frd_mps2=(1.0, 2.0, 3.0),
            gyro_body_frd_rps=(0.1, 0.2, 0.3),
        )
    )

    assert state.last_imu_time_boot_us == 1_000_000
    assert state.acceleration_body_frd_mps2 == (1.0, 2.0, 3.0)
    assert state.angular_velocity_body_frd_rps == (0.1, 0.2, 0.3)
    assert odometry.position_local_ned_m == (0.0, 0.0, 0.0)
    assert odometry.velocity_local_ned_mps == (0.0, 0.0, 0.0)
    assert odometry.attitude_quaternion == (1.0, 0.0, 0.0, 0.0)


def test_imu_acceleration_integrates_position_and_velocity() -> None:
    state = VehicleState()
    state.update_from_imu(imu_sample(0))

    odometry = state.update_from_imu(
        imu_sample(
            1_000_000,
            acceleration_body_frd_mps2=(2.0, 0.0, -1.0),
        )
    )

    np.testing.assert_allclose(odometry.velocity_local_ned_mps, (2.0, 0.0, -1.0))
    np.testing.assert_allclose(odometry.position_local_ned_m, (1.0, 0.0, -0.5))
    np.testing.assert_allclose(odometry.acceleration_local_ned_mps2, (2.0, 0.0, -1.0))


def test_imu_gyro_integrates_yaw_attitude_and_stays_normalized() -> None:
    state = VehicleState()
    state.update_from_imu(imu_sample(0))

    odometry = state.update_from_imu(
        imu_sample(
            1_000_000,
            gyro_body_frd_rps=(0.0, 0.0, 1.0),
        )
    )

    _, _, yaw = euler_from_quaternion(odometry.attitude_quaternion)
    assert yaw > 0.0
    np.testing.assert_allclose(np.linalg.norm(odometry.attitude_quaternion), 1.0)


def test_imu_gyro_delta_updates_angular_acceleration() -> None:
    state = VehicleState()
    state.update_from_imu(imu_sample(0, gyro_body_frd_rps=(0.0, 0.0, 0.0)))

    state.update_from_imu(imu_sample(500_000, gyro_body_frd_rps=(1.0, -2.0, 3.0)))

    np.testing.assert_allclose(state.angular_acceleration_body_frd_rps2, (2.0, -4.0, 6.0))
    assert state.angular_acceleration_roll_body_frd_rps2 == 2.0
    assert state.angular_acceleration_pitch_body_frd_rps2 == -4.0
    assert state.angular_acceleration_yaw_body_frd_rps2 == 6.0


def test_non_increasing_imu_timestamp_does_not_integrate_state() -> None:
    state = VehicleState()
    state.update_from_imu(imu_sample(1_000_000))
    state.update_from_imu(imu_sample(2_000_000, acceleration_body_frd_mps2=(1.0, 0.0, 0.0)))
    position = state.position_local_ned_m
    velocity = state.velocity_local_ned_mps

    state.update_from_imu(imu_sample(2_000_000, acceleration_body_frd_mps2=(10.0, 0.0, 0.0)))

    assert state.position_local_ned_m == position
    assert state.velocity_local_ned_mps == velocity
    assert math.isfinite(state.attitude_w)
