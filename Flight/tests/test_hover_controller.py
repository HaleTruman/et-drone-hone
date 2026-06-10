import math

import numpy as np

from core.control.hover import HoverController, HoverPIDController
from core.coordinates import euler_from_quaternion, quaternion_from_roll_pitch_yaw
from sensing.odometry import VehicleState


def test_at_target_zero_velocity_returns_physical_hover_thrust() -> None:
    controller = HoverPIDController(dt_s=0.1)

    quaternion, thrust = controller.update(
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        target_position_local_ned_m=(0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    assert thrust == math.sqrt((controller.mass_kg * controller.gravity_mps2) / (4.0 * controller.thrust_coefficient_n))


def test_below_target_increases_physical_thrust() -> None:
    controller = HoverPIDController(dt_s=0.1)

    _, hover_thrust = controller.update((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    _, recovery_thrust = controller.update(
        position_local_ned_m=(0.0, 0.0, 1.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        target_position_local_ned_m=(0.0, 0.0, 0.0),
    )

    assert recovery_thrust > hover_thrust


def test_velocity_error_changes_thrust_without_neutral_offset() -> None:
    controller = HoverPIDController(dt_s=0.1)

    _, hover_thrust = controller.update((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    _, recovery_thrust = controller.update(
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 2.0),
        target_position_local_ned_m=(0.0, 0.0, 0.0),
    )

    assert recovery_thrust > hover_thrust
    assert not hasattr(controller, "neutral_thrust")


def test_lateral_error_changes_attitude_command() -> None:
    controller = HoverPIDController(dt_s=0.1)

    quaternion, _ = controller.update(
        position_local_ned_m=(1.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        target_position_local_ned_m=(0.0, 0.0, 0.0),
    )
    _, pitch, _ = euler_from_quaternion(quaternion)

    assert pitch > 0.0


def test_current_yaw_is_preserved_by_default() -> None:
    controller = HoverPIDController(dt_s=0.1)
    current_attitude = quaternion_from_roll_pitch_yaw(0.0, 0.0, 1.2)

    quaternion, _ = controller.update(
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=current_attitude,
        target_position_local_ned_m=(0.0, 0.0, 0.0),
    )
    _, _, yaw = euler_from_quaternion(quaternion)

    assert abs(yaw - 1.2) < 1e-12


def test_hover_controller_latches_initial_state_as_hover_target() -> None:
    state = VehicleState()
    state.position_local_ned_m = (1.0, 2.0, -3.0)
    state.velocity_local_ned_mps = (0.0, 0.0, 0.0)
    controller = HoverController(dt_s=0.1)

    control_outputs = controller.compute_control(state)

    assert controller.target_position_local_ned_m == (1.0, 2.0, -3.0)
    assert sorted(control_outputs) == ["quaternion", "thrust"]
    np.testing.assert_allclose(control_outputs["quaternion"], [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_large_error_is_limited_to_configured_thrust_range() -> None:
    controller = HoverPIDController(dt_s=0.1, min_thrust=0.35, max_thrust=0.65)

    _, thrust = controller.update(
        position_local_ned_m=(100.0, 100.0, 100.0),
        velocity_local_ned_mps=(50.0, 50.0, 50.0),
        target_position_local_ned_m=(0.0, 0.0, 0.0),
    )

    assert 0.35 <= thrust <= 0.65
