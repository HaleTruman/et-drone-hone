"""Camera-relative PnP for a normalized gate-centerline quadrilateral."""

from __future__ import annotations

import cv2
import numpy as np

from .schema import (
    QUADRILATERAL_CORNER_ORDER,
    CameraCalibration,
    CameraPoseEstimate,
    PlanarGateModel,
    QuadrilateralEstimate,
)


SOLVER_NAME = "opencv_ippe_square_v1"
MAX_REPROJECTION_RMSE_PX = 8.0
CAMERA_CALIBRATION = CameraCalibration(
    calibration_id="sim-camera-640x360-pinhole-v1",
    image_shape=(360, 640),
    camera_matrix=((320.0, 0.0, 320.0),
                   (0.0, 320.0, 180.0),
                   (0.0, 0.0, 1.0)),
    distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
)
HALF_SIDE_M = 0.5 * ((2.70 + 1.50) / 2.0)
GATE_MODEL = PlanarGateModel(
    model_id="gate-face-centerline-square-2.10m-v1",
    side_length_m=2.0 * HALF_SIDE_M,
    corner_order=QUADRILATERAL_CORNER_ORDER,
    object_points_m=((-HALF_SIDE_M, HALF_SIDE_M, 0.0),
                     (HALF_SIDE_M, HALF_SIDE_M, 0.0),
                     (HALF_SIDE_M, -HALF_SIDE_M, 0.0),
                     (-HALF_SIDE_M, -HALF_SIDE_M, 0.0)),
)
K = np.asarray(CAMERA_CALIBRATION.camera_matrix, np.float64)
DISTORTION = np.asarray(
    CAMERA_CALIBRATION.distortion_coefficients, np.float64)
OBJECT_POINTS_M = np.asarray(GATE_MODEL.object_points_m, np.float64)


def _rejected(
    quad: QuadrilateralEstimate,
    reason: str,
    *,
    candidate_count: int = 0,
    rmse: float | None = None,
    secondary_rmse: float | None = None,
) -> CameraPoseEstimate:
    gap = (None if rmse is None or secondary_rmse is None
           else secondary_rmse - rmse)
    return CameraPoseEstimate(
        frame_id=quad.frame_id,
        sim_time_ns=quad.sim_time_ns,
        component_id=quad.component_id,
        route=quad.route,
        solver=SOLVER_NAME,
        camera_calibration_id=CAMERA_CALIBRATION.calibration_id,
        gate_model_id=GATE_MODEL.model_id,
        rotation_vector_model_to_camera=None,
        position_camera_m=None,
        candidate_count=candidate_count,
        reprojection_rmse_px=rmse,
        secondary_reprojection_rmse_px=secondary_rmse,
        ambiguity_gap_px=gap,
        position_confidence=0.0,
        orientation_confidence=0.0,
        accepted=False,
        rejection_reason=reason,
    )


def solve_gate_pose(quad: QuadrilateralEstimate) -> CameraPoseEstimate:
    """Solve a 2.10 m square; rotation maps gate-model into camera optical."""
    if not quad.accepted or quad.corners_uv is None:
        return _rejected(quad, "quadrilateral_not_accepted")
    if quad.image_shape != CAMERA_CALIBRATION.image_shape:
        return _rejected(quad, "camera_calibration_shape_mismatch")
    if quad.corner_order != QUADRILATERAL_CORNER_ORDER:
        return _rejected(quad, "quadrilateral_corner_order_mismatch")
    image_points = np.asarray(quad.corners_uv, np.float64)
    if image_points.shape != (4, 2) or not np.all(np.isfinite(image_points)):
        return _rejected(quad, "invalid_quadrilateral_points")

    try:
        solved = cv2.solvePnPGeneric(
            OBJECT_POINTS_M, image_points, K, DISTORTION,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return _rejected(quad, "pnp_solver_failed")
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]) if solved[0] else ():
        rotation = rvec.reshape(3)
        position = tvec.reshape(3)
        if (not np.all(np.isfinite(rotation)) or
                not np.all(np.isfinite(position)) or position[2] <= 0):
            continue
        projected, _ = cv2.projectPoints(
            OBJECT_POINTS_M, rvec, tvec, K, DISTORTION)
        rmse = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        candidates.append((rmse, rotation, position))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return _rejected(quad, "pnp_no_positive_depth_solution")

    rmse, rotation, position = candidates[0]
    secondary = candidates[1][0] if len(candidates) > 1 else None
    if rmse > MAX_REPROJECTION_RMSE_PX:
        return _rejected(
            quad, "pnp_reprojection_error", candidate_count=len(candidates),
            rmse=rmse, secondary_rmse=secondary)
    fit_confidence = 1.0 / (1.0 + (rmse / 3.0) ** 2)
    ambiguity_confidence = (1.0 if secondary is None else
                            (secondary - rmse) / (secondary - rmse + 1.0))
    return CameraPoseEstimate(
        frame_id=quad.frame_id,
        sim_time_ns=quad.sim_time_ns,
        component_id=quad.component_id,
        route=quad.route,
        solver=SOLVER_NAME,
        camera_calibration_id=CAMERA_CALIBRATION.calibration_id,
        gate_model_id=GATE_MODEL.model_id,
        rotation_vector_model_to_camera=tuple(map(float, rotation)),
        position_camera_m=tuple(map(float, position)),
        candidate_count=len(candidates),
        reprojection_rmse_px=rmse,
        secondary_reprojection_rmse_px=secondary,
        ambiguity_gap_px=(None if secondary is None else secondary - rmse),
        position_confidence=float(np.clip(fit_confidence, 0.0, 1.0)),
        orientation_confidence=float(np.clip(
            fit_confidence * ambiguity_confidence, 0.0, 1.0)),
        accepted=True,
        rejection_reason=None,
    )
