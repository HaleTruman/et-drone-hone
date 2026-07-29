"""Quad and ellipse fits for inner void contours."""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

try:  # pragma: no cover
    from .bbox import order_quad, polygon_area
    from .config import ProjectionConfig
    from .schema import FrameMeta, utc_now
except ImportError:  # pragma: no cover
    from bbox import order_quad, polygon_area
    from config import ProjectionConfig
    from schema import FrameMeta, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float(settings: dict[str, Any], key: str, fallback: float) -> float:
    try:
        value = float(settings.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _int(settings: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(float(settings.get(key, fallback)))
    except (TypeError, ValueError):
        return int(fallback)


def _bool(settings: dict[str, Any], key: str, fallback: bool = False) -> bool:
    value = settings.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return [x, y] if math.isfinite(x) and math.isfinite(y) else None


def _points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    cleaned = [_point(item) for item in value]
    return [[clean_float(point[0]), clean_float(point[1])] for point in cleaned if point is not None]


def _contour(points: list[list[float]]) -> np.ndarray | None:
    cleaned = _points(points)
    if len(cleaned) < 3:
        return None
    return np.asarray(cleaned, dtype=np.float32).reshape(-1, 1, 2)


def _point_list(points: np.ndarray) -> list[list[float]]:
    return [[clean_float(float(x)), clean_float(float(y))] for x, y in points.reshape(-1, 2)]


def _distance(left: list[float], right: list[float]) -> float:
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _normalize(raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in raw_weights.values())
    if total <= 0.0:
        first = next(iter(raw_weights), "score")
        return {key: 1.0 if key == first else 0.0 for key in raw_weights}
    return {key: max(0.0, float(value)) / total for key, value in raw_weights.items()}


def _polygon_angles(points: np.ndarray) -> list[float]:
    polygon = points.reshape(-1, 2).astype(np.float64)
    angles: list[float] = []
    for index, point in enumerate(polygon):
        prev_point = polygon[(index - 1) % len(polygon)]
        next_point = polygon[(index + 1) % len(polygon)]
        left = prev_point - point
        right = next_point - point
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-9:
            angles.append(0.0)
            continue
        cosine = max(-1.0, min(1.0, float(np.dot(left, right) / denom)))
        angles.append(float(math.degrees(math.acos(cosine))))
    return angles


def _side_balance(points: np.ndarray) -> float:
    polygon = points.reshape(-1, 2)
    sides = [float(np.linalg.norm(polygon[index] - polygon[(index + 1) % len(polygon)])) for index in range(len(polygon))]
    return clamp01(min(sides) / max(1e-6, max(sides))) if sides else 0.0


def _ellipse_perimeter_points(
    center: tuple[float, float],
    axes_px: tuple[float, float],
    angle_deg: float,
    sample_count: int,
) -> list[tuple[float, float]]:
    cx, cy = center
    axis_x = max(0.0, float(axes_px[0]) * 0.5)
    axis_y = max(0.0, float(axes_px[1]) * 0.5)
    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    points: list[tuple[float, float]] = []
    for index in range(max(8, int(sample_count))):
        theta = (2.0 * math.pi * index) / max(8, int(sample_count))
        local_x = axis_x * math.cos(theta)
        local_y = axis_y * math.sin(theta)
        points.append((cx + local_x * cos_a - local_y * sin_a, cy + local_x * sin_a + local_y * cos_a))
    return points


def _ellipse_area(axes_px: tuple[float, float], scale: float = 1.0) -> float:
    width = max(0.0, float(axes_px[0]) * scale)
    height = max(0.0, float(axes_px[1]) * scale)
    return float(math.pi * width * height * 0.25)


class GeometryFitter:
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.quad_settings = config.section("innerVoidQuadFit")
        self.ellipse_settings = config.section("innerVoidEllipseFit")

    def _quad_fit(self, void: dict[str, Any], parent_bbox_px: list[int]) -> dict[str, Any]:
        settings = self.quad_settings
        fit_id = f"{void['void_id']}-quad"
        contour = _contour(void.get("raw_points_px") or void.get("points_px") or [])
        if contour is None:
            return self._rejected_quad(fit_id, void, "missing-points")
        hull = cv2.convexHull(contour)
        perimeter = float(cv2.arcLength(hull, True))
        epsilon = _float(settings, "epsilonPx", 0.0) or perimeter * max(0.0, _float(settings, "epsilonRatio", 0.02))
        approx = cv2.approxPolyDP(hull, epsilon, True) if perimeter > 0.0 else hull
        approx_count = int(approx.reshape(-1, 2).shape[0])
        is_convex = bool(len(approx) >= 3 and cv2.isContourConvex(approx))
        quad_area = abs(float(cv2.contourArea(approx))) if approx_count >= 3 else 0.0
        fill_ratio = float(void.get("area_px") or 0.0) / max(1e-6, quad_area)
        fit_polygon = approx.reshape(-1, 2)
        angle_skew = max((abs(angle - 90.0) for angle in _polygon_angles(fit_polygon)), default=90.0) if approx_count >= 3 else 90.0
        side_balance = _side_balance(fit_polygon) if approx_count >= 3 else 0.0
        quad_points: list[list[float]] = []
        quad_local: list[list[float]] = []
        if approx_count == 4:
            ordered = order_quad(approx)
            quad_points = _point_list(ordered)
            px0, py0 = int(parent_bbox_px[0]), int(parent_bbox_px[1])
            quad_local = [[clean_float(point[0] - px0), clean_float(point[1] - py0)] for point in quad_points]
            side_balance = _side_balance(ordered)
        score, score_breakdown = self._quad_score(approx_count, is_convex, quad_area, fill_ratio, angle_skew, side_balance)
        reasons: list[str] = []
        max_points = _int(settings, "maxPointCountBeforeReject", 5)
        if max_points > 0 and approx_count > max_points:
            reasons.append("too-many-points")
        if _bool(settings, "requireExactlyFourPoints", False) and approx_count != 4:
            reasons.append("not-four-points")
        if _bool(settings, "requireConvex", True) and not is_convex:
            reasons.append("not-convex")
        if quad_area < _float(settings, "minQuadAreaPx", 1.0):
            reasons.append("quad-area-too-low")
        if fill_ratio < _float(settings, "minFillRatio", 0.0):
            reasons.append("fill-ratio-too-low")
        available = not reasons
        return {
            "fit_id": fit_id,
            "void_id": void["void_id"],
            "bbox_id": void["bbox_id"],
            "available": available,
            "status": "accepted" if available else "rejected",
            "reason": None if available else reasons[0],
            "source_status": void.get("status"),
            "pixel_count": int(void.get("pixel_count") or 0),
            "area_px": clean_float(float(void.get("area_px") or 0.0), 3),
            "bbox_px": list(void.get("bbox_px") or []),
            "center_px": list(void.get("center_px") or []),
            "contour_points_px": list(void.get("points_px") or []),
            "raw_point_count": int(contour.reshape(-1, 2).shape[0]),
            "hull_points_px": _point_list(hull),
            "hull_point_count": int(hull.reshape(-1, 2).shape[0]),
            "approx_points_px": _point_list(approx),
            "approx_point_count": approx_count,
            "quad_points_px": quad_points,
            "quad_points_local_px": quad_local,
            "quad_area_px": clean_float(quad_area, 3),
            "fill_ratio": clean_float(fill_ratio, 6),
            "angle_skew_deg": clean_float(angle_skew, 6),
            "side_balance": clean_float(side_balance, 6),
            "score": score,
            "score_breakdown": score_breakdown,
            "quality": {
                "reasons": reasons,
                "fitKind": "quad" if approx_count == 4 else "polygon",
                "fitPointCount": approx_count,
                "sourceVoidScore": clean_float(float(void.get("score") or 0.0), 6),
            },
        }

    def _rejected_quad(self, fit_id: str, void: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "fit_id": fit_id,
            "void_id": void.get("void_id", ""),
            "bbox_id": void.get("bbox_id", ""),
            "available": False,
            "status": "rejected",
            "reason": reason,
            "source_status": void.get("status"),
            "pixel_count": int(void.get("pixel_count") or 0),
            "area_px": clean_float(float(void.get("area_px") or 0.0), 3),
            "bbox_px": list(void.get("bbox_px") or []),
            "center_px": list(void.get("center_px") or []),
            "contour_points_px": list(void.get("points_px") or []),
            "score": 0.0,
            "score_breakdown": {},
            "quality": {"reasons": [reason]},
        }

    def _quad_score(
        self,
        approx_count: int,
        is_convex: bool,
        quad_area: float,
        fill_ratio: float,
        angle_skew: float,
        side_balance: float,
    ) -> tuple[float, dict[str, Any]]:
        settings = self.quad_settings
        max_points = max(4, _int(settings, "maxPointCountBeforeReject", 5))
        if _bool(settings, "requireExactlyFourPoints", False):
            point_score = clamp01(1.0 - abs(float(approx_count) - 4.0) / max(1.0, float(max_points - 4)))
        else:
            point_score = 1.0 if 3 <= approx_count <= max_points else clamp01(float(max_points) / max(1.0, float(approx_count)))
        raw = {
            "pointCount": max(0.0, _float(settings, "scoreWeightPointCount", 0.25)),
            "convexity": max(0.0, _float(settings, "scoreWeightConvexity", 0.15)),
            "fill": max(0.0, _float(settings, "scoreWeightFill", 0.25)),
            "area": max(0.0, _float(settings, "scoreWeightArea", 0.10)),
            "angle": max(0.0, _float(settings, "scoreWeightAngle", 0.15)),
            "side": max(0.0, _float(settings, "scoreWeightSide", 0.10)),
        }
        weights = _normalize(raw)
        scores = {
            "pointCount": point_score,
            "convexity": 1.0 if is_convex else 0.0,
            "fill": clamp01(fill_ratio / max(1e-6, _float(settings, "minFillRatio", 0.01))),
            "area": clamp01(quad_area / max(1e-6, _float(settings, "minQuadAreaPx", 1.0))),
            "angle": clamp01(1.0 - angle_skew / max(1e-6, _float(settings, "maxAngleSkewDeg", 180.0))),
            "side": clamp01(side_balance),
        }
        score = sum(weights[key] * scores[key] for key in weights)
        return clean_float(score, 6), {
            **{f"{key}Score": clean_float(value, 6) for key, value in scores.items()},
            "weights": {key: clean_float(value, 6) for key, value in weights.items()},
        }

    def _source_voids_for_ellipse(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        source_mode = str(self.ellipse_settings.get("sourceVoidMode") or "accepted-only")
        voids = list(observation.get("voids") or [])
        if source_mode == "accepted-and-rejected":
            voids.extend(list(observation.get("rejected_voids") or []))
        return voids

    def _ellipse_source_points(self, void: dict[str, Any]) -> tuple[list[list[float]], str]:
        source = str(self.ellipse_settings.get("contourSource") or "raw")
        raw = list(void.get("raw_points_px") or [])
        simplified = list(void.get("points_px") or [])
        if source == "simplified":
            return simplified or raw, "simplified"
        if source == "hull":
            contour = _contour(raw or simplified)
            if contour is None:
                return raw or simplified, "hull"
            return _point_list(cv2.convexHull(contour)), "hull"
        return raw or simplified, "raw"

    def _fit_ellipse(self, contour: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float, str]:
        method_name = str(self.ellipse_settings.get("fitMethod") or "ams").strip().lower()
        method: Callable[[np.ndarray], Any]
        actual = method_name
        if method_name == "direct" and hasattr(cv2, "fitEllipseDirect"):
            method = cv2.fitEllipseDirect
        elif method_name == "standard":
            method = cv2.fitEllipse
        elif hasattr(cv2, "fitEllipseAMS"):
            method = cv2.fitEllipseAMS
            actual = "ams"
        else:
            method = cv2.fitEllipse
            actual = "standard"
        try:
            center, axes, angle = method(contour)
        except cv2.error:
            center, axes, angle = cv2.fitEllipse(contour)
            actual = "standard-fallback"
        return (float(center[0]), float(center[1])), (float(axes[0]), float(axes[1])), float(angle), actual

    def _ellipse_containment(
        self,
        contour: np.ndarray,
        center: tuple[float, float],
        axes: tuple[float, float],
        angle: float,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        sample_count = max(8, _int(self.ellipse_settings, "sampleCount", 96))
        tolerance = max(0.0, _float(self.ellipse_settings, "containmentTolerancePx", 0.75))
        perimeter_points = _ellipse_perimeter_points(center, axes, angle, sample_count)
        distances = [float(cv2.pointPolygonTest(contour, point, True)) for point in perimeter_points]
        perimeter_ratio = sum(1 for value in distances if value >= -tolerance) / max(1, len(distances))
        mean_residual = sum(abs(value) for value in distances) / max(1, len(distances))
        contour_mask = np.zeros((max(1, image_height), max(1, image_width)), dtype=np.uint8)
        cv2.drawContours(contour_mask, [np.rint(contour).astype(np.int32)], -1, 255, thickness=-1)
        ellipse_mask = np.zeros_like(contour_mask)
        cv2.ellipse(
            ellipse_mask,
            (int(round(center[0])), int(round(center[1]))),
            (max(1, int(round(axes[0] * 0.5))), max(1, int(round(axes[1] * 0.5)))),
            angle,
            0.0,
            360.0,
            255,
            thickness=-1,
        )
        ellipse_pixels = int(np.count_nonzero(ellipse_mask))
        leak = int(np.count_nonzero((ellipse_mask > 0) & (contour_mask == 0)))
        fill_ratio = 1.0 if ellipse_pixels <= 0 else clamp01(1.0 - leak / max(1, ellipse_pixels))
        return {
            "perimeterContainmentRatio": clean_float(perimeter_ratio, 6),
            "fillContainmentRatio": clean_float(fill_ratio, 6),
            "leakPixelCount": leak,
            "ellipsePixelCount": ellipse_pixels,
            "meanBoundaryResidualPx": clean_float(mean_residual, 6),
        }

    def _contained_ellipse(
        self,
        contour: np.ndarray,
        center: tuple[float, float],
        axes: tuple[float, float],
        angle: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, tuple[float, float], int, dict[str, Any]]:
        if not _bool(self.ellipse_settings, "shrinkEnabled", True):
            metrics = self._ellipse_containment(contour, center, axes, angle, image_width, image_height)
            return 1.0, axes, 0, metrics
        step = max(0.001, _float(self.ellipse_settings, "shrinkStep", 0.02))
        min_scale = max(0.01, min(1.0, _float(self.ellipse_settings, "minShrinkScale", 0.35)))
        max_iterations = max(0, _int(self.ellipse_settings, "maxShrinkIterations", 80))
        min_perimeter = clamp01(_float(self.ellipse_settings, "minPerimeterContainment", 0.98))
        min_fill = clamp01(_float(self.ellipse_settings, "minFillContainment", 0.98))
        best = (1.0, axes, 0, self._ellipse_containment(contour, center, axes, angle, image_width, image_height))
        for iteration in range(max_iterations + 1):
            scale = max(min_scale, 1.0 - step * iteration)
            scaled_axes = (axes[0] * scale, axes[1] * scale)
            metrics = self._ellipse_containment(contour, center, scaled_axes, angle, image_width, image_height)
            best = (scale, scaled_axes, iteration, metrics)
            if metrics["perimeterContainmentRatio"] >= min_perimeter and metrics["fillContainmentRatio"] >= min_fill:
                break
            if scale <= min_scale:
                break
        return best

    def _ellipse_fit(self, void: dict[str, Any], parent_bbox_px: list[int], image_width: int, image_height: int) -> dict[str, Any]:
        points, contour_source = self._ellipse_source_points(void)
        contour = _contour(points)
        fit_id = f"{void['void_id']}-ellipse"
        min_points = max(5, _int(self.ellipse_settings, "minContourPoints", 5))
        if contour is None or contour.reshape(-1, 2).shape[0] < min_points:
            return self._rejected_ellipse(fit_id, void, "not-enough-points", contour_source, points)
        try:
            fit_center, fit_axes, fit_angle, method = self._fit_ellipse(contour)
        except cv2.error:
            return self._rejected_ellipse(fit_id, void, "fit-failed", contour_source, _point_list(contour))
        if fit_axes[0] <= 0.0 or fit_axes[1] <= 0.0:
            return self._rejected_ellipse(fit_id, void, "invalid-axes", contour_source, _point_list(contour))
        center = tuple(fit_center) if str(self.ellipse_settings.get("centerMode") or "contour-centroid") == "fit-center" else (
            float(void["center_px"][0]),
            float(void["center_px"][1]),
        )
        scale, final_axes, iterations, containment = self._contained_ellipse(contour, center, fit_axes, fit_angle, image_width, image_height)
        area = _ellipse_area(final_axes)
        area_ratio = area / max(1e-6, float(void.get("area_px") or 0.0))
        axis_ratio = max(final_axes) / max(1e-6, min(final_axes))
        center_shift = _distance([fit_center[0], fit_center[1]], [center[0], center[1]])
        score, score_breakdown = self._ellipse_score(
            float(containment["perimeterContainmentRatio"]),
            float(containment["fillContainmentRatio"]),
            area_ratio,
            float(containment["meanBoundaryResidualPx"]),
            center_shift,
            axis_ratio,
            float(void.get("score") or 0.0),
        )
        reasons: list[str] = []
        if area < _float(self.ellipse_settings, "minEllipseAreaPx", 12.0):
            reasons.append("ellipse-area-too-low")
        if axis_ratio > _float(self.ellipse_settings, "maxAxisRatio", 6.0):
            reasons.append("axis-ratio-too-high")
        if float(containment["perimeterContainmentRatio"]) < clamp01(_float(self.ellipse_settings, "minPerimeterContainment", 0.98)):
            reasons.append("perimeter-containment-too-low")
        if float(containment["fillContainmentRatio"]) < clamp01(_float(self.ellipse_settings, "minFillContainment", 0.98)):
            reasons.append("fill-containment-too-low")
        available = not reasons
        return {
            "fit_id": fit_id,
            "void_id": void["void_id"],
            "bbox_id": void["bbox_id"],
            "available": available,
            "status": "accepted" if available else "rejected",
            "reason": None if available else reasons[0],
            "source_status": void.get("status"),
            "pixel_count": int(void.get("pixel_count") or 0),
            "area_px": clean_float(float(void.get("area_px") or 0.0), 3),
            "bbox_px": list(void.get("bbox_px") or []),
            "center_px": list(void.get("center_px") or []),
            "contour_points_px": _point_list(contour),
            "raw_point_count": int(contour.reshape(-1, 2).shape[0]),
            "fit_center_px": [clean_float(fit_center[0]), clean_float(fit_center[1])],
            "fit_axes_px": [clean_float(fit_axes[0]), clean_float(fit_axes[1])],
            "fit_angle_deg": clean_float(fit_angle, 6),
            "ellipse_center_px": [clean_float(center[0]), clean_float(center[1])],
            "ellipse_axes_px": [clean_float(final_axes[0]), clean_float(final_axes[1])],
            "ellipse_angle_deg": clean_float(fit_angle, 6),
            "ellipse_area_px": clean_float(area, 3),
            "shrink_scale": clean_float(scale, 6),
            "shrink_iterations": int(iterations),
            "perimeter_containment_ratio": clean_float(float(containment["perimeterContainmentRatio"]), 6),
            "fill_containment_ratio": clean_float(float(containment["fillContainmentRatio"]), 6),
            "leak_pixel_count": int(containment["leakPixelCount"]),
            "area_ratio": clean_float(area_ratio, 6),
            "axis_ratio": clean_float(axis_ratio, 6),
            "center_shift_px": clean_float(center_shift, 6),
            "mean_boundary_residual_px": clean_float(float(containment["meanBoundaryResidualPx"]), 6),
            "score": score,
            "score_breakdown": score_breakdown,
            "quality": {
                "reasons": reasons,
                "contourSource": contour_source,
                "fitMethod": method,
                "parentBboxPx": list(parent_bbox_px),
                "sourceVoidScore": clean_float(float(void.get("score") or 0.0), 6),
            },
        }

    def _ellipse_score(
        self,
        perimeter: float,
        fill: float,
        area_ratio: float,
        residual: float,
        center_shift: float,
        axis_ratio: float,
        source_score: float,
    ) -> tuple[float, dict[str, Any]]:
        raw = {
            "containment": max(0.0, _float(self.ellipse_settings, "scoreWeightContainment", 0.35)),
            "coverage": max(0.0, _float(self.ellipse_settings, "scoreWeightCoverage", 0.20)),
            "residual": max(0.0, _float(self.ellipse_settings, "scoreWeightResidual", 0.15)),
            "centerShift": max(0.0, _float(self.ellipse_settings, "scoreWeightCenterShift", 0.10)),
            "axisRatio": max(0.0, _float(self.ellipse_settings, "scoreWeightAxisRatio", 0.10)),
            "sourceVoid": max(0.0, _float(self.ellipse_settings, "scoreWeightSourceVoid", 0.10)),
        }
        weights = _normalize(raw)
        scores = {
            "containment": clamp01((perimeter + fill) * 0.5),
            "coverage": clamp01(area_ratio),
            "residual": clamp01(1.0 - residual / max(1e-6, _float(self.ellipse_settings, "maxBoundaryResidualPx", 12.0))),
            "centerShift": clamp01(1.0 - center_shift / max(1e-6, _float(self.ellipse_settings, "maxCenterShiftPx", 18.0))),
            "axisRatio": clamp01(1.0 - max(0.0, axis_ratio - 1.0) / max(1e-6, _float(self.ellipse_settings, "maxAxisRatio", 6.0) - 1.0)),
            "sourceVoid": clamp01(source_score),
        }
        score = sum(weights[key] * scores[key] for key in weights)
        return clean_float(score, 6), {
            **{f"{key}Score": clean_float(value, 6) for key, value in scores.items()},
            "weights": {key: clean_float(value, 6) for key, value in weights.items()},
        }

    def _rejected_ellipse(
        self,
        fit_id: str,
        void: dict[str, Any],
        reason: str,
        contour_source: str,
        points: list[list[float]],
    ) -> dict[str, Any]:
        return {
            "fit_id": fit_id,
            "void_id": void.get("void_id", ""),
            "bbox_id": void.get("bbox_id", ""),
            "available": False,
            "status": "rejected",
            "reason": reason,
            "source_status": void.get("status"),
            "pixel_count": int(void.get("pixel_count") or 0),
            "area_px": clean_float(float(void.get("area_px") or 0.0), 3),
            "bbox_px": list(void.get("bbox_px") or []),
            "center_px": list(void.get("center_px") or []),
            "contour_points_px": _points(points),
            "raw_point_count": len(points or []),
            "score": 0.0,
            "score_breakdown": {},
            "quality": {"reasons": [reason], "contourSource": contour_source},
        }

    def process(self, meta: FrameMeta, inner_void_payload: dict[str, Any]) -> dict[str, Any]:
        quad_observations: list[dict[str, Any]] = []
        ellipse_observations: list[dict[str, Any]] = []
        quad_available = quad_rejected = ellipse_available = ellipse_rejected = 0
        for observation in inner_void_payload.get("observations", []):
            quad_fits: list[dict[str, Any]] = []
            quad_rejections: list[dict[str, Any]] = []
            for void in observation.get("voids") or []:
                fit = self._quad_fit(void, observation.get("bbox_px") or [])
                if fit.get("available"):
                    quad_fits.append(fit)
                else:
                    quad_rejections.append(fit)
            quad_available += len(quad_fits)
            quad_rejected += len(quad_rejections)
            quad_observations.append(
                {
                    "bbox_id": observation.get("bbox_id"),
                    "bbox_px": list(observation.get("bbox_px") or []),
                    "source_void_count": len(observation.get("voids") or []),
                    "available_count": len(quad_fits),
                    "rejected_count": len(quad_rejections),
                    "fits": quad_fits,
                    "rejected_fits": quad_rejections if _bool(self.quad_settings, "includeRejected", True) else [],
                }
            )

            ellipse_fits: list[dict[str, Any]] = []
            ellipse_rejections: list[dict[str, Any]] = []
            for void in self._source_voids_for_ellipse(observation):
                fit = self._ellipse_fit(void, observation.get("bbox_px") or [], int(inner_void_payload["image_width"]), int(inner_void_payload["image_height"]))
                if fit.get("available"):
                    ellipse_fits.append(fit)
                else:
                    ellipse_rejections.append(fit)
            ellipse_available += len(ellipse_fits)
            ellipse_rejected += len(ellipse_rejections)
            ellipse_observations.append(
                {
                    "bbox_id": observation.get("bbox_id"),
                    "bbox_px": list(observation.get("bbox_px") or []),
                    "source_void_count": len(self._source_voids_for_ellipse(observation)),
                    "available_count": len(ellipse_fits),
                    "rejected_count": len(ellipse_rejections),
                    "fits": ellipse_fits,
                    "rejected_fits": ellipse_rejections if _bool(self.ellipse_settings, "includeRejected", True) else [],
                }
            )
        return {
            "schema": "projection-geometry-fits.v1",
            "run_id": meta.run_id,
            "frame_ordinal": meta.frame_ordinal,
            "frame_id": meta.frame_id,
            "source_path": meta.source_path,
            "image_width": inner_void_payload["image_width"],
            "image_height": inner_void_payload["image_height"],
            "created_at": utc_now(),
            "quad": {
                "observations": quad_observations,
                "available_count": quad_available,
                "rejected_count": quad_rejected,
                "settings": dict(self.quad_settings),
            },
            "ellipse": {
                "observations": ellipse_observations,
                "available_count": ellipse_available,
                "rejected_count": ellipse_rejected,
                "settings": dict(self.ellipse_settings),
            },
        }
