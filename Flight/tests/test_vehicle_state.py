import math

import numpy as np

from core.coordinates import euler_from_quaternion
from core.schemas import MavlinkHighresImu, OdometryState
from sensing.odometry import VehicleState


GRAVITY_MPS2 = 9.80665


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


def test_update_odometry_outputs_odometry_state() -> None:
    state = VehicleState()

    odometry = state.update_odometry(
        OdometryState(
            sim_time_ns=1_234_000,
            position_local_ned_m=(1.0, 2.0, -3.0),
            velocity_local_ned_mps=(4.0, 5.0, 6.0),
            attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
            body_rates_frd_rps=(0.1, 0.2, 0.3),
            acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
        )
    )

    assert isinstance(odometry, OdometryState)
    assert state.odometry == odometry
    assert odometry.sim_time_ns == 1_234_000
    assert odometry.position_local_ned_m == (1.0, 2.0, -3.0)
    assert odometry.velocity_local_ned_mps == (4.0, 5.0, 6.0)
    assert odometry.body_rates_frd_rps == (0.1, 0.2, 0.3)


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
    np.testing.assert_allclose(np.linalg.norm(odometry.attitude_quaternion), 1.0)


def test_first_imu_sample_infers_level_attitude_from_gravity() -> None:
    state = VehicleState()

    odometry = state.update_from_imu(
        imu_sample(
            1_000_000,
            acceleration_body_frd_mps2=(0.0, 0.0, -GRAVITY_MPS2),
        )
    )

    np.testing.assert_allclose(odometry.attitude_quaternion, (1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(odometry.acceleration_local_ned_mps2, (0.0, 0.0, 0.0), atol=1e-9)


def test_update_accepts_highres_imu_and_returns_odometry_state() -> None:
    state = VehicleState()
    imu = imu_sample(
        1_000_000,
        acceleration_body_frd_mps2=(0.0, 0.0, -GRAVITY_MPS2),
        gyro_body_frd_rps=(0.1, 0.2, 0.3),
    )

    odometry = state.update(imu)

    assert odometry is not None
    assert isinstance(odometry, OdometryState)
    assert odometry == state.state
    assert odometry.sim_time_ns == 1_000_000_000
    assert odometry.body_rates_frd_rps == (0.1, 0.2, 0.3)


def test_imu_acceleration_integrates_position_and_velocity() -> None:
    state = VehicleState()
    state.update_from_imu(imu_sample(0))

    odometry = state.update_from_imu(
        imu_sample(
            1_000_000,
            acceleration_body_frd_mps2=(2.0, 0.0, -GRAVITY_MPS2),
        )
    )

    np.testing.assert_allclose(odometry.velocity_local_ned_mps, (2.0, 0.0, 0.0))
    np.testing.assert_allclose(odometry.position_local_ned_m, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(odometry.acceleration_local_ned_mps2, (2.0, 0.0, 0.0))


def test_stationary_specific_force_does_not_integrate_gravity() -> None:
    state = VehicleState()
    state.update_from_imu(imu_sample(0))

    odometry = state.update_from_imu(
        imu_sample(
            1_000_000,
            acceleration_body_frd_mps2=(0.0, 0.0, -GRAVITY_MPS2),
        )
    )

    np.testing.assert_allclose(odometry.acceleration_local_ned_mps2, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(odometry.velocity_local_ned_mps, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(odometry.position_local_ned_m, (0.0, 0.0, 0.0))


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

    state.update_from_imu(imu_sample(2_000_000, acceleration_body_frd_mps2=(10.0, 0.0, -GRAVITY_MPS2)))

    assert state.position_local_ned_m == position
    assert state.velocity_local_ned_mps == velocity
    assert math.isfinite(state.attitude_w)
