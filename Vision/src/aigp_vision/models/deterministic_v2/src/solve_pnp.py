"""OpenCV SolvePnP for inner void geometry products."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

try:  # pragma: no cover
    from .config import ProjectionConfig
    from .schema import FrameMeta, utc_now
except ImportError:  # pragma: no cover
    from config import ProjectionConfig
    from schema import FrameMeta, utc_now


SOURCE_WEIGHTS: dict[str, float] = {
    "inner_void_quad_fit": 0.45,
    "inner_void_ellipse_fit": 0.30,
    "inner_void_detection_rect": 0.25,
}


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def clamp01(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(fallback)
    if not math.isfinite(number):
        number = float(fallback)
    return max(0.0, min(1.0, number))


def _float(settings: dict[str, Any], key: str, fallback: float) -> float:
    try:
        value = float(settings.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    cleaned: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            cleaned.append([clean_float(x, 4), clean_float(y, 4)])
    return cleaned


def camera_matrix(camera: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [float(camera.get("fx", 320.0)), 0.0, float(camera.get("cx", 320.0))],
            [0.0, float(camera.get("fy", 320.0)), float(camera.get("cy", 180.0))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def square_points(size_m: float) -> list[list[float]]:
    half = float(size_m) * 0.5
    return [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]]


def circle_cardinals(diameter_m: float) -> list[list[float]]:
    radius = float(diameter_m) * 0.5
    return [[radius, 0.0, 0.0], [0.0, radius, 0.0], [-radius, 0.0, 0.0], [0.0, -radius, 0.0]]


def rpy_from_rotation(rotation: np.ndarray) -> list[float]:
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


def solve_object_pose(
    *,
    pose_id: str,
    image_points_px: list[list[float]],
    object_points_m: list[list[float]],
    camera: dict[str, Any],
    max_reprojection_error_px: float,
    source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "solve_id": pose_id,
        "source": source,
        "available": False,
        "status": "rejected",
        "reason": None,
        "imagePointsPx": image_points_px,
        "objectPointsM": object_points_m,
        "fitQuality": {"overall": 0.0},
    }
    if len(image_points_px) < 4 or len(image_points_px) != len(object_points_m):
        payload["reason"] = "missing-points"
        return payload
    object_array = np.asarray(object_points_m, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    matrix = camera_matrix(camera)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(object_array, image_array, matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
    if not ok:
        payload["reason"] = "solvepnp-failed"
        return payload
    projected, _ = cv2.projectPoints(object_array, rvec, tvec, matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_array, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    quality = clamp01(1.0 - max_error / max(0.1, float(max_reprojection_error_px)))
    rotation, _ = cv2.Rodrigues(rvec)
    xyz = [clean_float(float(value), 6) for value in tvec.reshape(3).tolist()]
    payload.update(
        {
            "fitQuality": {
                "overall": clean_float(quality, 6),
                "meanReprojectionErrorPx": clean_float(mean_error, 4),
                "maxReprojectionErrorPx": clean_float(max_error, 4),
            },
            "reprojectionErrorPx": clean_float(max_error, 4),
            "projectedPointsPx": [[clean_float(float(x), 4), clean_float(float(y), 4)] for x, y in projected.tolist()],
            "solvePnPMethod": "SOLVEPNP_ITERATIVE",
            "xyzCameraM": xyz,
            "depthM": clean_float(float(xyz[2]), 6),
            "rpyCameraDeg": rpy_from_rotation(rotation),
        }
    )
    if max_error > float(max_reprojection_error_px):
        payload["reason"] = "reprojection-error-too-high"
        return payload
    if not math.isfinite(float(xyz[2])) or float(xyz[2]) <= 0.0:
        payload["reason"] = "non-positive-depth"
        return payload
    payload["available"] = True
    payload["status"] = "accepted"
    payload["reason"] = None
    return payload


def ellipse_axis_points(fit: dict[str, Any]) -> list[list[float]]:
    center = fit.get("ellipse_center_px") or fit.get("fit_center_px") or fit.get("center_px") or []
    axes = fit.get("ellipse_axes_px") or fit.get("fit_axes_px") or []
    if len(center) < 2 or len(axes) < 2:
        return []
    cx, cy = float(center[0]), float(center[1])
    axis_x = max(0.0, float(axes[0]) * 0.5)
    axis_y = max(0.0, float(axes[1]) * 0.5)
    angle = math.radians(float(fit.get("ellipse_angle_deg") or fit.get("fit_angle_deg") or 0.0))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [clean_float(cx + axis_x * cos_a, 4), clean_float(cy + axis_x * sin_a, 4)],
        [clean_float(cx - axis_y * sin_a, 4), clean_float(cy + axis_y * cos_a, 4)],
        [clean_float(cx - axis_x * cos_a, 4), clean_float(cy - axis_x * sin_a, 4)],
        [clean_float(cx + axis_y * sin_a, 4), clean_float(cy - axis_y * cos_a, 4)],
    ]


class SolvePnPRunner:
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.settings = config.section("innerVoidSolvePnP")
        self.camera = config.camera

    def _lookup_fits(self, fit_payload: dict[str, Any], group: str) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for observation in fit_payload.get(group, {}).get("observations", []):
            for fit in [*(observation.get("fits") or []), *(observation.get("rejected_fits") or [])]:
                void_id = str(fit.get("void_id") or "")
                if void_id and (void_id not in lookup or (fit.get("available") and not lookup[void_id].get("available"))):
                    lookup[void_id] = fit
        return lookup

    def _solve_candidate(
        self,
        *,
        void: dict[str, Any],
        source: str,
        image_points: list[list[float]],
        object_points: list[list[float]],
        fit: dict[str, Any] | None = None,
        source_geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        solve = solve_object_pose(
            pose_id=f"{void['void_id']}-{source}-pnp",
            image_points_px=image_points,
            object_points_m=object_points,
            camera=self.camera,
            max_reprojection_error_px=_float(self.settings, "maxReprojectionErrorPx", 8.0),
            source=source,
        )
        fit_quality = solve.get("fitQuality") if isinstance(solve.get("fitQuality"), dict) else {}
        solve.update(
            {
                "void_id": void["void_id"],
                "bbox_id": void["bbox_id"],
                "source_status": void.get("status"),
                "source_score": clean_float(float(void.get("score") or 0.0), 6),
                "fit_id": fit.get("fit_id") if isinstance(fit, dict) else None,
                "fit_score": clean_float(float(fit.get("score") or 0.0), 6) if isinstance(fit, dict) else None,
                "sourceGeometry": source_geometry or {},
                "quality": {
                    "sourceWeight": SOURCE_WEIGHTS.get(source, 0.0),
                    "fitQualityOverall": fit_quality.get("overall", 0.0),
                },
            }
        )
        return solve

    def process(self, meta: FrameMeta, inner_void_payload: dict[str, Any], fit_payload: dict[str, Any]) -> dict[str, Any]:
        quad_lookup = self._lookup_fits(fit_payload, "quad")
        ellipse_lookup = self._lookup_fits(fit_payload, "ellipse")
        include_rejected = bool(self.settings.get("includeRejectedSourceVoids", True))
        square_model = square_points(float(self.settings.get("innerSquareSizeM", self.config.inner_square_size_m)))
        circle_model = circle_cardinals(float(self.settings.get("ellipseDiameterM", self.config.ellipse_diameter_m)))
        observations: list[dict[str, Any]] = []
        total_available = 0
        total_rejected = 0
        for observation in inner_void_payload.get("observations", []):
            source_voids = list(observation.get("voids") or [])
            if include_rejected:
                source_voids.extend(list(observation.get("rejected_voids") or []))
            solves: list[dict[str, Any]] = []
            for void in source_voids:
                rect_points = _points((void.get("rotated_rect_px") or {}).get("pointsPx"))
                if len(rect_points) == 4:
                    solves.append(
                        self._solve_candidate(
                            void=void,
                            source="inner_void_detection_rect",
                            image_points=rect_points,
                            object_points=square_model,
                            source_geometry={"rotatedRectPx": void.get("rotated_rect_px") or {}},
                        )
                    )
                else:
                    solves.append(self._unavailable(void, "inner_void_detection_rect", "missing-rotated-rect-points", square_model))

                quad = quad_lookup.get(str(void.get("void_id")))
                quad_points = _points(quad.get("quad_points_px") if quad else None)
                if quad and quad.get("available") and len(quad_points) == 4:
                    solves.append(
                        self._solve_candidate(
                            void=void,
                            source="inner_void_quad_fit",
                            image_points=quad_points,
                            object_points=square_model,
                            fit=quad,
                            source_geometry={"quadPointsPx": quad_points, "approxPointsPx": quad.get("approx_points_px") or []},
                        )
                    )
                else:
                    solves.append(self._unavailable(void, "inner_void_quad_fit", (quad or {}).get("reason") or "missing-quad-fit", square_model, quad))

                ellipse = ellipse_lookup.get(str(void.get("void_id")))
                ellipse_points = ellipse_axis_points(ellipse) if ellipse else []
                if ellipse and ellipse.get("available") and len(ellipse_points) == 4:
                    solves.append(
                        self._solve_candidate(
                            void=void,
                            source="inner_void_ellipse_fit",
                            image_points=ellipse_points,
                            object_points=circle_model,
                            fit=ellipse,
                            source_geometry={
                                "ellipseCenterPx": ellipse.get("ellipse_center_px") or [],
                                "ellipseAxesPx": ellipse.get("ellipse_axes_px") or [],
                                "axisExtremePointsPx": ellipse_points,
                            },
                        )
                    )
                else:
                    solves.append(
                        self._unavailable(void, "inner_void_ellipse_fit", (ellipse or {}).get("reason") or "missing-ellipse-fit", circle_model, ellipse)
                    )
            available = sum(1 for solve in solves if solve.get("available"))
            total_available += available
            total_rejected += len(solves) - available
            observations.append(
                {
                    "bbox_id": observation.get("bbox_id"),
                    "bbox_px": list(observation.get("bbox_px") or []),
                    "source_void_count": len(source_voids),
                    "solve_count": len(solves),
                    "available_count": available,
                    "rejected_count": len(solves) - available,
                    "solves": solves,
                }
            )
        return {
            "schema": "projection-inner-void-solve-pnp.v1",
            "run_id": meta.run_id,
            "frame_ordinal": meta.frame_ordinal,
            "frame_id": meta.frame_id,
            "source_path": meta.source_path,
            "image_width": inner_void_payload["image_width"],
            "image_height": inner_void_payload["image_height"],
            "created_at": utc_now(),
            "observations": observations,
            "available_count": total_available,
            "rejected_count": total_rejected,
            "settings": dict(self.settings),
        }

    def _unavailable(
        self,
        void: dict[str, Any],
        source: str,
        reason: str,
        object_points: list[list[float]],
        fit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "solve_id": f"{void.get('void_id', '')}-{source}-pnp",
            "void_id": void.get("void_id", ""),
            "bbox_id": void.get("bbox_id", ""),
            "source": source,
            "available": False,
            "status": "rejected",
            "reason": reason,
            "source_status": void.get("status"),
            "source_score": clean_float(float(void.get("score") or 0.0), 6),
            "fit_id": fit.get("fit_id") if isinstance(fit, dict) else None,
            "fit_score": clean_float(float(fit.get("score") or 0.0), 6) if isinstance(fit, dict) else None,
            "imagePointsPx": [],
            "objectPointsM": object_points,
            "fitQuality": {"overall": 0.0},
            "sourceGeometry": {},
            "quality": {"sourceWeight": SOURCE_WEIGHTS.get(source, 0.0)},
        }
