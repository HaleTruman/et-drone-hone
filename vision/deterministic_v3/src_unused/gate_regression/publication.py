"""Authoritative final-pose publication from one regression posterior."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..schema import (
    CameraCalibration,
    CameraPoseEstimate,
    GateRegressionConfiguration,
    GateRegressionEvidence,
    PlanarGateModel,
    PnPRelativePoseEstimate,
    QuadrilateralEstimate,
)
from .fusion import FusedPose, orientation_consistency, rotation_vector


def _reprojection_rmse(
    posterior: FusedPose,
    quadrilateral: QuadrilateralEstimate,
    calibration: CameraCalibration,
    gate_model: PlanarGateModel,
) -> float | None:
    if quadrilateral.corners_uv is None:
        return None
    image_points = np.asarray(quadrilateral.corners_uv, np.float64)
    object_points = np.asarray(gate_model.object_points_m, np.float64)
    if (
        image_points.shape != (4, 2)
        or object_points.shape != (4, 3)
        or not np.all(np.isfinite(image_points))
        or not np.all(np.isfinite(object_points))
    ):
        return None
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rotation_vector(posterior.rotation_model_to_camera)),
        posterior.position,
        np.asarray(calibration.camera_matrix, np.float64),
        np.asarray(calibration.distortion_coefficients, np.float64),
    )
    residuals = projected.reshape(4, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def _position_confidence(posterior: FusedPose) -> float:
    sigma = math.sqrt(max(0.0, float(np.trace(posterior.covariance))) / 3.0)
    distance = max(0.20, float(np.linalg.norm(posterior.position)))
    covariance_confidence = 1.0 / (1.0 + (sigma / (0.08 * distance)) ** 2)
    return float(np.clip(
        covariance_confidence * (0.5 + 0.5 * posterior.quality), 0.0, 1.0
    ))


def publish_pose(
    *,
    raw_pose: PnPRelativePoseEstimate,
    posterior: FusedPose,
    quadrilateral: QuadrilateralEstimate | None,
    calibration: CameraCalibration,
    gate_model: PlanarGateModel,
    configuration: GateRegressionConfiguration,
    track_id: str,
    selected_candidate_rank: int,
    observation_count: int,
    orientation_residual_deg: float | None,
) -> CameraPoseEstimate:
    """Build one final pose and recompute all metrics it claims."""
    evidence = GateRegressionEvidence(
        regression_version=configuration.configuration_version,
        track_id=track_id,
        selected_candidate_rank=selected_candidate_rank,
        observation_count=observation_count,
        effective_history_weight=float(posterior.effective_history_weight),
        orientation_residual_deg=orientation_residual_deg,
    )
    solver = raw_pose.solver
    if configuration.configuration_version not in solver.split("+"):
        solver = f"{solver}+{configuration.configuration_version}"
    rmse = None if quadrilateral is None else _reprojection_rmse(
        posterior, quadrilateral, calibration, gate_model
    )
    rejection_reason = None
    if posterior.position[2] <= configuration.minimum_camera_depth_m:
        rejection_reason = "regression_nonpositive_camera_depth"
    elif rmse is None or not math.isfinite(rmse):
        rejection_reason = "regression_reprojection_unavailable"
    elif rmse > configuration.maximum_reprojection_rmse_px:
        rejection_reason = "regression_reprojection_error"

    accepted = rejection_reason is None
    fit_confidence = (
        0.0 if rmse is None else 1.0 / (1.0 + (rmse / 3.0) ** 2)
    )
    return CameraPoseEstimate(
        frame_id=raw_pose.frame_id,
        sim_time_ns=raw_pose.sim_time_ns,
        component_id=raw_pose.component_id,
        route=raw_pose.route,
        solver=solver,
        camera_calibration_id=raw_pose.camera_calibration_id,
        gate_model_id=raw_pose.gate_model_id,
        rotation_vector_model_to_camera=(
            rotation_vector(posterior.rotation_model_to_camera)
            if accepted else None
        ),
        position_camera_m=(
            tuple(map(float, posterior.position)) if accepted else None
        ),
        reprojection_rmse_px=rmse,
        position_confidence=(
            _position_confidence(posterior) if accepted else 0.0
        ),
        orientation_confidence=(
            float(np.clip(fit_confidence * orientation_consistency(posterior), 0.0, 1.0))
            if accepted else 0.0
        ),
        accepted=accepted,
        rejection_reason=rejection_reason,
        gate_index=raw_pose.gate_index,
        regression_evidence=evidence,
    )
