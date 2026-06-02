import numpy as np

from core.simulator import TelemetrySimulator
from sensing.estimation.state_estimator import StateEstimator


def test_emits_only_expected_mavlink_telemetry_messages() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0, heartbeat_hz=2.0)

    messages = list(simulator.messages(duration_s=1.0))
    message_types = {message.message_type for message in messages}

    assert message_types == {"HEARTBEAT", "TIMESYNC", "ATTITUDE", "HIGHRES_IMU"}
    assert sum(message.message_type == "HEARTBEAT" for message in messages) == 3
    assert sum(message.message_type == "ATTITUDE" for message in messages) == 11
    assert all("position_local_ned_m" not in message.fields for message in messages)


def test_bridge_samples_are_normalized_and_fall_under_gravity() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)

    samples = list(simulator.telemetry_samples(duration_s=0.2))

    assert len(samples) == 3
    np.testing.assert_allclose(simulator._harness.initial_state[0:3], np.zeros(3))
    np.testing.assert_allclose(np.linalg.norm(samples[0].attitude), 1.0)
    assert np.linalg.norm(samples[0].velocity_local_ned_mps) < 0.1
    assert samples[-1].velocity_local_ned_mps[2] > 0.0
    assert samples[-1].raw["acceleration_local_ned_mps2"][2] > 0.0


def test_state_estimator_tracks_relative_fall_from_telemetry() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)
    estimator = StateEstimator()

    for sample in simulator.telemetry_samples(duration_s=0.5):
        estimator.update_from_telemetry(sample)

    estimated_relative_z = estimator.get_13_state()[2]
    model_relative_z = simulator._harness.state[2] - simulator._harness.initial_state[2]
    np.testing.assert_allclose(estimated_relative_z, model_relative_z, atol=0.05)


def test_state_estimator_tracks_truth_when_initialized_in_simulator_frame() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)
    estimator = StateEstimator(initial_state=simulator._harness.initial_state)

    for sample in simulator.telemetry_samples(duration_s=0.5):
        estimator.update_from_telemetry(sample)

    np.testing.assert_allclose(estimator.get_13_state()[0:3], simulator._harness.state[0:3], atol=0.05)


def test_noise_is_repeatable_and_changes_ideal_velocity() -> None:
    first = list(TelemetrySimulator(telemetry_hz=10.0, noise_seed=11).telemetry_samples(0.1))
    second = list(TelemetrySimulator(telemetry_hz=10.0, noise_seed=11).telemetry_samples(0.1))

    assert first == second
    assert first[0].velocity_local_ned_mps != (0.0, 0.0, 0.0)
