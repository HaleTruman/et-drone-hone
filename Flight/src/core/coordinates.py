"""Coordinate-frame helpers used by runtime code.

Canonical frames:
- LOCAL_NED: world/local frame in meters, +x north/forward, +y east/right, +z down.
- BODY_FRD: vehicle body frame, +x forward, +y right, +z down.
- CAMERA_OPTICAL: +x right, +y up, +z forward.
- UNREAL_WORLD: centimeters, +x forward, +y right, +z up.

Quaternions are always stored in [w, x, y, z] order.
"""

import math
from typing import Iterable

import numpy as np


def vec3(value: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError("expected three values")
    return tuple(float(item) for item in array)


def quat_wxyz(value: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (4,):
        raise ValueError("expected quaternion [w, x, y, z]")
    return tuple(float(item) for item in normalize_quaternion(array))


def normalize_quaternion(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quaternion), dtype=float)
    if q.shape != (4,):
        raise ValueError("expected quaternion [w, x, y, z]")
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("quaternion cannot be zero")
    return q / norm


def rotation_matrix_from_quaternion(quaternion: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = normalize_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def quaternion_from_rotation_matrix(rotation: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("expected 3x3 rotation matrix")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    return quat_wxyz((qw, qx, qy, qz))


def rotate_vector(quaternion: Iterable[float], vector: Iterable[float]) -> tuple[float, float, float]:
    return vec3(rotation_matrix_from_quaternion(quaternion) @ np.asarray(tuple(vector), dtype=float))


def quaternion_from_roll_pitch_yaw(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll_rad / 2.0), math.sin(roll_rad / 2.0)
    cp, sp = math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0)
    cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    return quat_wxyz(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def euler_from_quaternion(quaternion: Iterable[float]) -> tuple[float, float, float]:
    qw, qx, qy, qz = normalize_quaternion(quaternion)
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return float(roll), float(pitch), float(yaw)


def unreal_cm_to_local_ned_m(x_cm: float, y_cm: float, z_cm: float) -> tuple[float, float, float]:
    return float(x_cm) * 0.01, float(y_cm) * 0.01, -float(z_cm) * 0.01


def unreal_vector_cm_to_local_ned_m(value: dict[str, float] | Iterable[float]) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return unreal_cm_to_local_ned_m(value["x"], value["y"], value["z"])
    x_cm, y_cm, z_cm = tuple(value)
    return unreal_cm_to_local_ned_m(x_cm, y_cm, z_cm)


def camera_optical_to_body_frd(camera_tilt_deg: float = 20.0) -> np.ndarray:
    tilt = math.radians(float(camera_tilt_deg))
    c, s = math.cos(tilt), math.sin(tilt)
    optical_to_body_frd = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    upward_camera_tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    return upward_camera_tilt @ optical_to_body_frd
