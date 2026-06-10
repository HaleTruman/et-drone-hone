import numpy as np

from core.coordinates import (
    quaternion_from_roll_pitch_yaw,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
    unreal_cm_to_local_ned_m,
)


def test_unreal_cm_to_local_ned_m() -> None:
    assert unreal_cm_to_local_ned_m(100.0, 200.0, 300.0) == (1.0, 2.0, -3.0)


def test_quaternion_helpers_use_wxyz_order() -> None:
    quaternion = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)

    assert quaternion == (1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(rotation_matrix_from_quaternion(quaternion), np.eye(3))


def test_rotation_matrix_round_trips_to_quaternion() -> None:
    quaternion = quaternion_from_roll_pitch_yaw(0.2, -0.1, 0.7)
    rotation = rotation_matrix_from_quaternion(quaternion)

    round_trip = quaternion_from_rotation_matrix(rotation)

    np.testing.assert_allclose(round_trip, quaternion, atol=1e-12)
