import numpy as np

from core.control.hover import HoverPIDController
from simulator import TelemetrySimulator


def test_level_zero_acceleration_returns_neutral_hover_command() -> None:
    controller = HoverPIDController(dt_s=0.1)

    quaternion, thrust = controller.update((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    assert quaternion.tolist() == [1.0, 0.0, 0.0, -0.0]
    assert thrust == 0.5


def test_downward_acceleration_increases_thrust() -> None:
    controller = HoverPIDController(dt_s=0.1)

    _, thrust = controller.update((0.0, 0.0, 3.0), (1.0, 0.0, 0.0, 0.0))

    assert thrust > controller.neutral_thrust


def test_lateral_acceleration_changes_attitude_command() -> None:
    controller = HoverPIDController(dt_s=0.1)

    quaternion, _ = controller.update((2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    assert quaternion[2] > 0.0


def test_measured_tilt_is_commanded_back_toward_level() -> None:
    simulator = TelemetrySimulator()
    controller = HoverPIDController(dt_s=0.1)
    tilted = simulator._quaternion_from_roll_pitch_yaw(0.1, 0.0, 0.0)

    quaternion, _ = controller.update((0.0, 0.0, 0.0), tilted)

    assert quaternion[1] < 0.0
