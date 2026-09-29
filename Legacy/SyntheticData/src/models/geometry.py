"""Per-gate planar pose recovery from four detected image keypoints."""

from __future__ import annotations

import numpy as np

from SyntheticData.src.models.training.targets import CameraCalibration, wrap_yaw_180


def solve_gate_pose(
    corners_px: np.ndarray,
    calibration: CameraCalibration,
) -> dict[str, np.ndarray | float] | None:
    """Solve camera-relative position and side-invariant plane orientation.

    The keypoints must be image TL, TR, BL, BR. A normalized DLT homography is
    decomposed into a rigid planar pose. The gate normal is treated as
    unoriented, so yaw is always returned in [-90, 90).
    """

    image_points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(image_points).all():
        return None
    width = calibration.gate_width_m
    height = calibration.gate_height_m
    object_points = np.asarray(
        [
            [-width * 0.5, -height * 0.5],
            [width * 0.5, -height * 0.5],
            [-width * 0.5, height * 0.5],
            [width * 0.5, height * 0.5],
        ],
        dtype=np.float64,
    )
    try:
        homography = _homography_dlt(object_points, image_points)
        intrinsic = np.asarray(
            [
                [calibration.focal_x_px, 0.0, calibration.principal_x_px],
                [0.0, calibration.focal_y_px, calibration.principal_y_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        normalized = np.linalg.inv(intrinsic) @ homography
        scale = 2.0 / (
            np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1])
        )
        r1 = normalized[:, 0] * scale
        r2 = normalized[:, 1] * scale
        translation = normalized[:, 2] * scale
        if translation[2] < 0.0:
            r1, r2, translation = -r1, -r2, -translation
        rotation_approx = np.column_stack((r1, r2, np.cross(r1, r2)))
        u, _, vt = np.linalg.svd(rotation_approx)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
    except np.linalg.LinAlgError:
        return None

    normal = rotation[:, 2]
    if normal[2] < 0.0:
        normal = -normal
    gate_up = -rotation[:, 1]
    right, up, forward = normal[0], -normal[1], normal[2]
    yaw = float(wrap_yaw_180(np.degrees(np.arctan2(right, forward))))
    pitch = float(
        np.degrees(np.arctan2(up, np.sqrt(right * right + forward * forward)))
    )
    reference_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    reference_up -= normal * np.dot(reference_up, normal)
    reference_norm = np.linalg.norm(reference_up)
    if reference_norm > 1e-8:
        reference_up /= reference_norm
        roll = float(
            -np.degrees(
                np.arctan2(
                    np.dot(np.cross(reference_up, gate_up), normal),
                    np.dot(reference_up, gate_up),
                )
            )
        )
    else:
        roll = 0.0

    reprojection = _project_gate(rotation, translation, intrinsic, width, height)
    reprojection_error = float(
        np.linalg.norm(reprojection - image_points, axis=1).mean()
    )
    return {
        "position_m": np.asarray(
            [translation[0], -translation[1], translation[2]],
            dtype=np.float32,
        ),
        "orientation_deg": np.asarray([yaw, pitch, roll], dtype=np.float32),
        "rotation_matrix_cv": rotation.astype(np.float32),
        "reprojection_error_px": reprojection_error,
    }


def _homography_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows = []
    for (x_value, y_value), (u_value, v_value) in zip(source, destination):
        rows.extend(
            (
                [-x_value, -y_value, -1.0, 0.0, 0.0, 0.0, u_value * x_value, u_value * y_value, u_value],
                [0.0, 0.0, 0.0, -x_value, -y_value, -1.0, v_value * x_value, v_value * y_value, v_value],
            )
        )
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vt[-1].reshape(3, 3)
    if abs(homography[2, 2]) > 1e-12:
        homography /= homography[2, 2]
    return homography


def _project_gate(
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    points = np.asarray(
        [
            [-width * 0.5, -height * 0.5, 0.0],
            [width * 0.5, -height * 0.5, 0.0],
            [-width * 0.5, height * 0.5, 0.0],
            [width * 0.5, height * 0.5, 0.0],
        ]
    )
    camera = (rotation @ points.T).T + translation
    projected = (intrinsic @ camera.T).T
    return projected[:, :2] / projected[:, 2:3]
