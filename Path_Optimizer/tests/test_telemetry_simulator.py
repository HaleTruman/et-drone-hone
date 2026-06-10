import numpy as np

from core.schemas import ImuSample
from sensing.odometry import initial_odometry_state, integrate_highres_imu
from simulator import TelemetrySimulator


def test_emits_expected_mavlink_telemetry_messages() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0, heartbeat_hz=2.0)

    messages = list(simulator.messages(duration_s=1.0))
    message_types = {message.message_type for message in messages}

    assert message_types == {
        "HEARTBEAT",
        "TIMESYNC",
        "ATTITUDE",
        "HIGHRES_IMU",
        "LOCAL_POSITION_NED",
        "ODOMETRY",
        "ACTUATOR_OUTPUT_STATUS",
    }
    assert sum(message.message_type == "HEARTBEAT" for message in messages) == 3
    assert sum(message.message_type == "ATTITUDE" for message in messages) == 11
    assert sum(message.message_type == "ODOMETRY" for message in messages) == 11


def test_bridge_samples_are_normalized_and_fall_under_gravity() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)

    samples = list(simulator.telemetry_samples(duration_s=0.2))

    assert len(samples) == 3
    np.testing.assert_allclose(simulator._harness.initial_state[0:3], np.zeros(3))
    np.testing.assert_allclose(np.linalg.norm(samples[0].attitude), 1.0)
    assert np.linalg.norm(samples[0].velocity_local_ned_mps) < 0.1
    assert samples[-1].velocity_local_ned_mps[2] > 0.0
    assert samples[-1].acceleration_local_ned_mps2[2] > 0.0


def test_local_ned_odometry_integrates_highres_imu_velocity() -> None:
    state = initial_odometry_state()
    imu = ImuSample(
        sim_time_ns=1_000_000_000,
        acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
        gyro_frd_rps=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(2.0, 0.0, -1.0),
    )

    updated = integrate_highres_imu(state, imu)

    np.testing.assert_allclose(updated.position_local_ned_m, (1.0, 0.0, -0.5))


def test_highres_imu_odometry_tracks_truth_when_initialized_in_simulator_frame() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)
    state = initial_odometry_state(position_local_ned_m=tuple(simulator._harness.initial_state[0:3]))

    for sample in simulator.telemetry_samples(duration_s=0.5):
        imu = ImuSample(
            sim_time_ns=sample.sim_time_ns,
            acceleration_local_ned_mps2=sample.acceleration_local_ned_mps2,
            gyro_frd_rps=sample.body_rates_rps,
            velocity_local_ned_mps=sample.velocity_local_ned_mps,
        )
        state = integrate_highres_imu(state, imu)

    np.testing.assert_allclose(state.position_local_ned_m, simulator._harness.state[0:3], atol=0.05)


def test_noise_is_repeatable_and_changes_ideal_velocity() -> None:
    first = list(TelemetrySimulator(telemetry_hz=10.0, noise_seed=11).telemetry_samples(0.1))
    second = list(TelemetrySimulator(telemetry_hz=10.0, noise_seed=11).telemetry_samples(0.1))

    assert first == second
    assert first[0].velocity_local_ned_mps != (0.0, 0.0, 0.0)
