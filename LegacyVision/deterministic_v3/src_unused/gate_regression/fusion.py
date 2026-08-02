"""Bounded incremental position and orientation fusion for gate poses."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


@dataclass(slots=True)
class FusedPose:
    """One active-track posterior expressed in its anchor camera frame."""

    position: np.ndarray
    covariance: np.ndarray
    rotation_model_to_camera: np.ndarray
    effective_history_weight: float
    orientation_residual_energy: float
    reprojection_rmse_px: float
    quality: float


def rotation_matrix(rotation_vector: tuple[float, float, float]) -> np.ndarray:
    """Return a proper rotation matrix for an OpenCV Rodrigues vector."""
    vector = np.asarray(rotation_vector, np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("invalid rotation vector")
    matrix, _ = cv2.Rodrigues(vector)
    return matrix


def rotation_vector(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return an OpenCV Rodrigues vector for a proper rotation matrix."""
    vector, _ = cv2.Rodrigues(np.asarray(rotation, np.float64))
    return tuple(map(float, vector.reshape(3)))


def rotation_residual_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest geodesic angle between two rotations."""
    relative = np.asarray(first, np.float64) @ np.asarray(second, np.float64).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def reframe(
    pose: FusedPose | None,
    old_origin: np.ndarray | None,
    old_world_from_camera: np.ndarray | None,
    new_origin: np.ndarray,
    new_world_from_camera: np.ndarray,
) -> FusedPose | None:
    """Express a cached posterior in a new camera frame."""
    if pose is None or old_origin is None or old_world_from_camera is None:
        return None
    camera_from_old = new_world_from_camera.T @ old_world_from_camera
    translation = new_world_from_camera.T @ (old_origin - new_origin)
    return replace(
        pose,
        position=translation + camera_from_old @ pose.position,
        covariance=camera_from_old @ pose.covariance @ camera_from_old.T,
        rotation_model_to_camera=(
            camera_from_old @ pose.rotation_model_to_camera
        ),
    )


def decay(pose: FusedPose, age_frames: int, half_life_frames: float) -> FusedPose:
    """Decay cached information without revisiting retained observations."""
    factor = 2.0 ** (-max(0, age_frames) / max(half_life_frames, 1e-6))
    factor = max(factor, 1e-9)
    return replace(
        pose,
        covariance=pose.covariance / factor,
        effective_history_weight=pose.effective_history_weight * factor,
        orientation_residual_energy=pose.orientation_residual_energy * factor,
        quality=pose.quality * factor,
    )


def initial_pose(
    *,
    position: np.ndarray,
    covariance: np.ndarray,
    rotation: np.ndarray,
    reprojection_rmse_px: float,
    quality: float,
) -> FusedPose:
    weight = float(np.clip(quality, 1e-3, 1.0))
    return FusedPose(
        position=np.asarray(position, np.float64).copy(),
        covariance=np.asarray(covariance, np.float64).copy() / weight,
        rotation_model_to_camera=np.asarray(rotation, np.float64).copy(),
        effective_history_weight=weight,
        orientation_residual_energy=0.0,
        reprojection_rmse_px=float(reprojection_rmse_px),
        quality=weight,
    )


def fuse(
    prior: FusedPose,
    *,
    position: np.ndarray,
    covariance: np.ndarray,
    rotation: np.ndarray,
    reprojection_rmse_px: float,
    quality: float,
) -> tuple[FusedPose, float]:
    """Fuse one independent observation into a decayed posterior in O(1)."""
    evidence_weight = float(np.clip(quality, 1e-3, 1.0))
    evidence_covariance = np.asarray(covariance, np.float64) / evidence_weight
    prior_information = np.linalg.pinv(prior.covariance, rcond=1e-10)
    evidence_information = np.linalg.pinv(evidence_covariance, rcond=1e-10)
    information = prior_information + evidence_information
    fused_covariance = np.linalg.pinv(information, rcond=1e-10)
    fused_position = fused_covariance @ (
        prior_information @ prior.position
        + evidence_information @ np.asarray(position, np.float64)
    )

    residual_deg = rotation_residual_degrees(rotation, prior.rotation_model_to_camera)
    total_weight = prior.effective_history_weight + evidence_weight
    alpha = evidence_weight / max(total_weight, 1e-9)
    relative = np.asarray(rotation, np.float64) @ prior.rotation_model_to_camera.T
    relative_vector, _ = cv2.Rodrigues(relative)
    step, _ = cv2.Rodrigues(alpha * relative_vector)
    fused_rotation = step @ prior.rotation_model_to_camera

    fused = FusedPose(
        position=fused_position,
        covariance=fused_covariance,
        rotation_model_to_camera=fused_rotation,
        effective_history_weight=total_weight,
        orientation_residual_energy=(
            prior.orientation_residual_energy
            + evidence_weight * residual_deg * residual_deg
        ),
        reprojection_rmse_px=(
            prior.effective_history_weight * prior.reprojection_rmse_px
            + evidence_weight * float(reprojection_rmse_px)
        ) / max(total_weight, 1e-9),
        quality=(
            prior.effective_history_weight * prior.quality
            + evidence_weight * evidence_weight
        ) / max(total_weight, 1e-9),
    )
    return fused, residual_deg


def robust_fuse(
    prior: FusedPose,
    *,
    position: np.ndarray,
    covariance: np.ndarray,
    rotation: np.ndarray,
    reprojection_rmse_px: float,
    quality: float,
    observation_count: int,
) -> tuple[FusedPose, float]:
    """Apply the inherited gross-outlier gate before ordinary fusion."""
    evidence_weight = float(np.clip(quality, 1e-3, 1.0))
    evidence_covariance = np.asarray(covariance, np.float64) / evidence_weight
    delta = np.asarray(position, np.float64) - prior.position
    combined = prior.covariance + evidence_covariance
    mahalanobis = float(
        delta @ np.linalg.pinv(combined, rcond=1e-10) @ delta
    )
    distance_gate = max(
        0.50, 0.08 * float(np.linalg.norm(np.asarray(position, np.float64)))
    )
    if mahalanobis <= 16.0 or float(np.linalg.norm(delta)) <= distance_gate:
        return fuse(
            prior,
            position=position,
            covariance=covariance,
            rotation=rotation,
            reprojection_rmse_px=reprojection_rmse_px,
            quality=quality,
        )

    residual_deg = rotation_residual_degrees(
        rotation, prior.rotation_model_to_camera
    )
    prior_score = prior.quality / max(
        1e-6, float(np.trace(prior.covariance))
    )
    evidence_score = evidence_weight / max(
        1e-6, float(np.trace(evidence_covariance))
    )
    stronger = (
        observation_count > 1
        and (
            reprojection_rmse_px < 1.25 * prior.reprojection_rmse_px
            or evidence_score > 1.5 * prior_score
        )
    )
    if stronger:
        return initial_pose(
            position=position,
            covariance=covariance,
            rotation=rotation,
            reprojection_rmse_px=reprojection_rmse_px,
            quality=quality,
        ), residual_deg
    return replace(
        prior,
        covariance=prior.covariance + np.eye(3) * 0.01,
        quality=0.90 * prior.quality,
    ), residual_deg


def orientation_consistency(pose: FusedPose) -> float:
    """Convert the decayed angular RMS into a bounded confidence factor."""
    rms_deg = math.sqrt(
        pose.orientation_residual_energy
        / max(pose.effective_history_weight, 1e-9)
    )
    return float(math.exp(-0.5 * (rms_deg / 15.0) ** 2))
