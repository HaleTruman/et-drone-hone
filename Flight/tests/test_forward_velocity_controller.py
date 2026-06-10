import numpy as np

from core.control.forward_velocity import ForwardVelocityAltitudeController
from simulator import TelemetrySimulator


def test_forward_velocity_error_commands_forward_pitch_without_fixed_thrust() -> None:
    controller = ForwardVelocityAltitudeController(dt_s=0.1, neutral_thrust=0.5)

    target = controller.update(
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(2.0, 0.0, 0.0),
        yaw_rad=0.0,
    )

    roll, pitch, _ = target["roll_pitch_yaw_rad"]
    assert abs(roll) < 1e-9
    assert pitch < 0.0
    assert target["thrust"] > 0.5
    assert abs(pitch) <= controller.max_tilt_rate_rps * controller.dt_s
    assert target["attitude_type_mask"] == 7
    assert target["target_velocity_local_ned_mps"][0] < 2.0


def test_measured_yaw_is_preserved_in_attitude_target() -> None:
    controller = ForwardVelocityAltitudeController(dt_s=0.1, neutral_thrust=0.5)

    target = controller.update(
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=TelemetrySimulator._quaternion_from_roll_pitch_yaw(0.0, 0.0, 1.2),
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(0.0, 2.0, 0.0),
    )

    assert np.isclose(target["measured_roll_pitch_yaw_rad"][2], 1.2)
    assert np.isclose(target["roll_pitch_yaw_rad"][2], 1.2)


def test_measured_forward_pitch_damps_forward_pitch_command() -> None:
    controller = ForwardVelocityAltitudeController(dt_s=0.1, neutral_thrust=0.5)
    pitched_forward = TelemetrySimulator._quaternion_from_roll_pitch_yaw(0.0, -0.12, 0.0)

    target = controller.update(
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=pitched_forward,
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(2.0, 0.0, 0.0),
        yaw_rad=0.0,
    )

    assert target["roll_pitch_yaw_rad"][1] > -controller.max_tilt_rate_rps * controller.dt_s


def test_altitude_error_increases_or_decreases_thrust_from_hover_trim() -> None:
    controller = ForwardVelocityAltitudeController(dt_s=0.1, neutral_thrust=0.5)

    above_target = controller.update(
        position_local_ned_m=(0.0, 0.0, -1.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
    )
    controller.reset()
    below_target = controller.update(
        position_local_ned_m=(0.0, 0.0, 1.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
    )

    assert above_target["thrust"] < 0.5
    assert below_target["thrust"] > 0.5


def test_climbing_or_above_target_reduces_thrust_below_live_trim() -> None:
    controller = ForwardVelocityAltitudeController(dt_s=0.1, neutral_thrust=0.5)

    target = controller.update(
        position_local_ned_m=(0.0, 0.0, -0.5),
        velocity_local_ned_mps=(0.0, 0.0, -0.4),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(2.0, 0.0, 0.0),
        trim_thrust=0.62,
        yaw_rad=0.0,
    )

    assert target["trim_thrust"] == 0.62
    assert target["thrust"] < target["trim_thrust"]
    assert target["altitude_priority_active"] is True
    assert target["desired_acceleration_local_ned_mps2"][0] > 0.0
    assert target["desired_acceleration_local_ned_mps2"][1] == 0.0


def test_small_altitude_noise_does_not_cancel_forward_control() -> None:
    controller = ForwardVelocityAltitudeController(dt_s=0.1, neutral_thrust=0.5)

    target = controller.update(
        position_local_ned_m=(0.0, 0.0, -0.02),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_altitude_ned_m=0.0,
        target_velocity_local_ned_mps=(2.0, 0.0, 0.0),
        trim_thrust=0.5,
        yaw_rad=0.0,
    )

    assert target["altitude_priority_active"] is False
    assert target["desired_acceleration_local_ned_mps2"][0] > 0.0


def test_controller_moves_forward_and_holds_altitude_in_local_simulator() -> None:
    simulator = TelemetrySimulator(telemetry_hz=30.0, physics_hz=120.0)
    simulator.arm()
    controller = ForwardVelocityAltitudeController(dt_s=1.0 / 30.0, neutral_thrust=0.5)
    sample = simulator.step()
    target_altitude = sample.position_local_ned_m[2]

    for _ in range(150):
        sample = simulator.step()
        target = controller.update(
            position_local_ned_m=sample.position_local_ned_m,
            velocity_local_ned_mps=sample.velocity_local_ned_mps,
            attitude_quaternion=sample.attitude,
            target_altitude_ned_m=target_altitude,
            target_velocity_local_ned_mps=(2.0, 0.0, 0.0),
            yaw_rad=0.0,
        )
        simulator.apply_attitude_target(target)

    sample = simulator.step()
    assert sample.position_local_ned_m[0] > 1.0
    assert abs(sample.position_local_ned_m[2] - target_altitude) < 2.0
    assert np.isclose(np.linalg.norm(sample.attitude), 1.0, atol=1e-6)
