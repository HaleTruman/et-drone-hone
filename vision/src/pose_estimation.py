"""Square gate pose fitting stage."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

try:  # pragma: no cover - exercised by script-mode imports
    from .schema import ClippingFrame, ContourFrame, PipelinePreset, PoseFrame, PoseObservation, utc_now
except ImportError:  # pragma: no cover
    from schema import ClippingFrame, ContourFrame, PipelinePreset, PoseFrame, PoseObservation, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def _object_points(square_size_m: float) -> np.ndarray:
    half = float(square_size_m) / 2.0
    return np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float64,
    )


def _camera_matrix(camera: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(camera.get("fx", 320.0)), 0.0, float(camera.get("cx", 320.0))],
            [0.0, float(camera.get("fy", 320.0)), float(camera.get("cy", 180.0))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rpy_from_rotation(rotation: np.ndarray) -> list[float]:
    sy = math.sqrt(float(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(float(-rotation[2, 0]), sy)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(float(-rotation[2, 0]), sy)
        yaw = 0.0
    return [clean_float(math.degrees(value), 4) for value in (roll, pitch, yaw)]


def _invalid(bbox_id: str, reason: str, contour: Any = None) -> PoseObservation:
    return PoseObservation(
        bbox_id=bbox_id,
        available=False,
        reason=reason,
        fitQuality={"overall": 0.0},
        imagePointsPx=contour.quad_points_px if contour is not None else None,
    )


class PoseEstimator:
    def __init__(self, preset: PipelinePreset):
        self.preset = preset
        self.settings = dict(preset.pose)
        self.camera_matrix = _camera_matrix(preset.camera)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.object_points = _object_points(float(self.settings.get("squareSizeM", 2.7)))

    def process(self, contour_frame: ContourFrame, clipping_frame: ClippingFrame) -> PoseFrame:
        clipping_by_bbox = {item.bbox_id: item for item in clipping_frame.observations}
        observations: list[PoseObservation] = []
        min_points = int(self.settings.get("minContourPoints", 4))
        max_error = float(self.settings.get("maxReprojectionErrorPx", 8.0))
        clip_guard = bool(self.settings.get("clipPoseGuardEnabled", True))
        for contour in contour_frame.observations:
            if clip_guard:
                clip = clipping_by_bbox.get(contour.bbox_id)
                if clip is not None and clip.status == "clipped":
                    observations.append(_invalid(contour.bbox_id, "bbox-clipped", contour))
                    continue
            if not contour.outer and not contour.quad_points_px:
                observations.append(_invalid(contour.bbox_id, "missing-outer-contour", contour))
                continue
            outer_points = []
            if contour.outer and isinstance(contour.outer.get("pointsPx"), list):
                outer_points = contour.outer.get("pointsPx") or []
            if len(outer_points) < min_points and not contour.quad_points_px:
                observations.append(_invalid(contour.bbox_id, "too-few-contour-points", contour))
                continue
            if not contour.quad_points_px or len(contour.quad_points_px) != 4:
                observations.append(_invalid(contour.bbox_id, "missing-quad-points", contour))
                continue
            image_points = np.asarray(contour.quad_points_px, dtype=np.float64).reshape(4, 2)
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                ok = False
                rvec = np.zeros((3, 1), dtype=np.float64)
                tvec = np.zeros((3, 1), dtype=np.float64)
            if not ok:
                observations.append(_invalid(contour.bbox_id, "solvepnp-failed", contour))
                continue
            projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            projected = projected.reshape(-1, 2)
            errors = np.linalg.norm(projected - image_points, axis=1)
            mean_error = float(np.mean(errors))
            max_observed_error = float(np.max(errors))
            if max_observed_error > max_error:
                observations.append(
                    PoseObservation(
                        bbox_id=contour.bbox_id,
                        available=False,
                        reason="reprojection-error-too-high",
                        fitQuality={
                            "overall": clean_float(max(0.0, 1.0 - max_observed_error / max(0.1, max_error)), 6),
                            "meanReprojectionErrorPx": clean_float(mean_error, 4),
                            "maxReprojectionErrorPx": clean_float(max_observed_error, 4),
                        },
                        reprojectionErrorPx=clean_float(max_observed_error, 4),
                        solvePnPMethod="SOLVEPNP_ITERATIVE",
                        imagePointsPx=contour.quad_points_px,
                        objectPointsM=self.object_points.tolist(),
                    )
                )
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            xyz = [clean_float(float(value), 6) for value in tvec.reshape(3).tolist()]
            depth = float(xyz[2])
            quality = {
                "overall": clean_float(max(0.0, 1.0 - max_observed_error / max(0.1, max_error)), 6),
                "meanReprojectionErrorPx": clean_float(mean_error, 4),
                "maxReprojectionErrorPx": clean_float(max_observed_error, 4),
                "edgeCoverage": 1.0,
            }
            observations.append(
                PoseObservation(
                    bbox_id=contour.bbox_id,
                    available=bool(math.isfinite(depth) and depth > 0.0),
                    reason=None if math.isfinite(depth) and depth > 0.0 else "non-positive-depth",
                    xyzCameraM=xyz,
                    rpyCameraDeg=_rpy_from_rotation(rotation),
                    depthM=clean_float(depth, 6),
                    fitQuality=quality,
                    reprojectionErrorPx=clean_float(max_observed_error, 4),
                    solvePnPMethod="SOLVEPNP_ITERATIVE",
                    imagePointsPx=contour.quad_points_px,
                    objectPointsM=[[clean_float(float(v), 6) for v in row] for row in self.object_points.tolist()],
                )
            )
        return PoseFrame(
            run_id=contour_frame.run_id,
            frame_ordinal=contour_frame.frame_ordinal,
            frame_id=contour_frame.frame_id,
            source_path=contour_frame.source_path,
            image_width=contour_frame.image_width,
            image_height=contour_frame.image_height,
            created_at=utc_now(),
            timing_ms={},
            observations=observations,
            available_count=sum(1 for item in observations if item.available),
            settings=dict(self.settings),
        )

