"""Workbench-equivalent rolling inner-void instance tracking for projection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover
    from .config import ProjectionConfig
    from .schema import FrameMeta, utc_now
except ImportError:  # pragma: no cover
    from config import ProjectionConfig
    from schema import FrameMeta, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == -0.0 else float(rounded)


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
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    return [x, y] if math.isfinite(x) and math.isfinite(y) else None


def _points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    cleaned: list[list[float]] = []
    for item in value:
        parsed = _point(item)
        if parsed is not None:
            cleaned.append(parsed)
    return cleaned


def _distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [int(round(float(item))) for item in value[:4]]
    except (TypeError, ValueError):
        return None


def _bbox_iou(left: list[int] | None, right: list[int] | None) -> float:
    if left is None or right is None:
        return 0.0
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    intersection = iw * ih
    area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
    area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
    return float(intersection) / float(max(1, area_a + area_b - intersection))


def _ratio_similarity(left: Any, right: Any) -> tuple[float, float]:
    try:
        a = abs(float(left))
        b = abs(float(right))
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not math.isfinite(a) or not math.isfinite(b) or max(a, b) <= 1e-9:
        return 0.0, 1.0
    similarity = min(a, b) / max(a, b)
    return clamp01(similarity), clamp01(1.0 - similarity)


def _angle_delta(left: Any, right: Any) -> float | None:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    delta = abs((a - b) % 180.0)
    return min(delta, 180.0 - delta)


def _rect(void: dict[str, Any]) -> dict[str, Any]:
    value = void.get("rotated_rect_px")
    return dict(value) if isinstance(value, dict) else {}


def _rect_angle(void: dict[str, Any]) -> float | None:
    try:
        value = float(_rect(void).get("angleDeg"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rect_size(void: dict[str, Any]) -> list[float] | None:
    size = _rect(void).get("sizePx")
    if not isinstance(size, list) or len(size) < 2:
        return None
    try:
        return [float(size[0]), float(size[1])]
    except (TypeError, ValueError):
        return None


def _rect_points(void: dict[str, Any]) -> list[list[float]]:
    points = _points(_rect(void).get("pointsPx"))
    return points if len(points) == 4 else []


def _mean_corner_distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_points = _rect_points(left)
    right_points = _rect_points(right)
    if len(left_points) != 4 or len(right_points) != 4:
        return None
    distances = [_distance(left_points[index], right_points[index]) for index in range(4)]
    if any(item is None for item in distances):
        return None
    return float(sum(float(item) for item in distances) / 4.0)


def _center_from_points(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    return [float(sum(point[0] for point in points) / len(points)), float(sum(point[1] for point in points) / len(points))]


def _points_angle(points: list[list[float]]) -> float | None:
    if len(points) < 2:
        return None
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def _mean_point_distance(left: list[list[float]], right: list[list[float]]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    distances = [_distance(left[index], right[index]) for index in range(len(left))]
    if any(item is None for item in distances):
        return None
    return float(sum(float(item) for item in distances) / len(distances))


def _fit_score(fit: dict[str, Any] | None) -> float:
    if not fit:
        return 0.0
    quality = fit.get("quality") if isinstance(fit.get("quality"), dict) else {}
    return clamp01(fit.get("score", quality.get("score", 0.0)))


def _fit_available(fit: dict[str, Any] | None) -> bool:
    return bool(fit and fit.get("available") and str(fit.get("status") or "accepted") == "accepted")


def _fit_reason(fit: dict[str, Any] | None, fallback: str) -> str:
    if not fit:
        return fallback
    quality = fit.get("quality") if isinstance(fit.get("quality"), dict) else {}
    reasons = quality.get("reasons") if isinstance(quality.get("reasons"), list) else []
    return str(fit.get("reason") or (reasons[0] if reasons else "") or fallback)


def _quad_points(fit: dict[str, Any] | None) -> list[list[float]]:
    if not fit:
        return []
    return _points(fit.get("quad_points_px")) or _points(fit.get("approx_points_px"))


def _quad_center(fit: dict[str, Any] | None) -> list[float] | None:
    if not fit:
        return None
    return _point(fit.get("center_px")) or _center_from_points(_quad_points(fit))


def _ellipse_center(fit: dict[str, Any] | None) -> list[float] | None:
    if not fit:
        return None
    return _point(fit.get("ellipse_center_px")) or _point(fit.get("fit_center_px")) or _point(fit.get("center_px"))


def _ellipse_axis_ratio(fit: dict[str, Any] | None) -> float | None:
    if not fit:
        return None
    axes = fit.get("ellipse_axes_px") or fit.get("fit_axes_px")
    if not isinstance(axes, list) or len(axes) < 2:
        return None
    try:
        a = abs(float(axes[0]))
        b = abs(float(axes[1]))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b) or min(a, b) <= 1e-9:
        return None
    return max(a, b) / min(a, b)


SOLVE_PNP_SOURCES = ("inner_void_detection_rect", "inner_void_quad_fit", "inner_void_ellipse_fit")
SOURCE_WEIGHTS = {"inner_void_quad_fit": 0.45, "inner_void_ellipse_fit": 0.30, "inner_void_detection_rect": 0.25}


def tuple3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def distance3(left: Any, right: Any) -> float | None:
    a = tuple3(left)
    b = tuple3(right)
    if a is None or b is None:
        return None
    return float(math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3))))


def _scalar_delta(left: Any, right: Any) -> float | None:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return abs(a - b)


def _rpy_delta(left: Any, right: Any) -> float | None:
    a = tuple3(left)
    b = tuple3(right)
    if a is None or b is None:
        return None
    deltas = []
    for index in range(3):
        delta = abs((a[index] - b[index]) % 360.0)
        deltas.append(min(delta, 360.0 - delta))
    return float(sum(deltas) / len(deltas))


def _plane_normal_from_rpy(value: Any) -> list[float] | None:
    rpy = tuple3(value)
    if rpy is None:
        return None
    roll = math.radians(rpy[0])
    pitch = math.radians(rpy[1])
    yaw = math.radians(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    normal = [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr]
    length = math.sqrt(sum(item * item for item in normal))
    if length <= 1e-9:
        return None
    return [item / length for item in normal]


def _plane_normal_delta_deg(left: Any, right: Any) -> float | None:
    a = _plane_normal_from_rpy(left)
    b = _plane_normal_from_rpy(right)
    if a is None or b is None:
        return None
    dot = abs(sum(a[index] * b[index] for index in range(3)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in raw_weights.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        first = next(iter(cleaned), "inner_void_detection")
        cleaned[first] = 1.0
        total = 1.0
    return {key: value / total for key, value in cleaned.items()}


def _void_score(void: dict[str, Any]) -> float:
    quality = void.get("quality") if isinstance(void.get("quality"), dict) else {}
    return clamp01(void.get("score", quality.get("score", 0.0)))


def _compact_void(void: dict[str, Any]) -> dict[str, Any]:
    return {
        "void_id": str(void.get("void_id") or ""),
        "bbox_id": str(void.get("bbox_id") or ""),
        "status": str(void.get("status") or ""),
        "score": clean_float(_void_score(void), 6),
        "pixel_count": int(void.get("pixel_count") or 0),
        "area_px": clean_float(float(void.get("area_px") or 0.0), 3),
        "center_px": list(void.get("center_px") or []),
        "bbox_px": list(void.get("bbox_px") or []),
    }


def _history_record(
    *,
    frame_ordinal: int,
    frame_id: str,
    record: dict[str, Any],
    association_score: float,
    total_confidence: float,
    tracking_layers: dict[str, Any],
    tracking_status: str,
) -> dict[str, Any]:
    void = record.get("void") if isinstance(record.get("void"), dict) else {}
    solve_pnp = record.get("solve_pnp") if isinstance(record.get("solve_pnp"), dict) else {}
    return {
        "frame_ordinal": int(frame_ordinal),
        "frame_id": frame_id,
        "void_id": str(void.get("void_id") or ""),
        "bbox_id": str(void.get("bbox_id") or ""),
        "bbox_px": list(void.get("bbox_px") or []),
        "center_px": list(void.get("center_px") or []),
        "score": clean_float(_void_score(void), 6),
        "solve_pnp": {
            "best_available_source": solve_pnp.get("best_available_source"),
            "available_count": int(solve_pnp.get("available_count") or 0),
            "rejected_count": int(solve_pnp.get("rejected_count") or 0),
        },
        "association_score": clean_float(float(association_score), 6),
        "total_confidence": clean_float(float(total_confidence), 6),
        "layers": {
            key: {
                "available": bool(layer.get("available")),
                "confidence": clean_float(float(layer.get("confidence") or 0.0), 6),
                "agreement": clean_float(float(layer.get("agreement") or 0.0), 6),
                "status": str(layer.get("status") or ""),
                "reason": layer.get("reason"),
            }
            for key, layer in tracking_layers.items()
            if isinstance(layer, dict)
        },
        "tracking_status": tracking_status,
    }


@dataclass
class _Track:
    instance_id: str
    void: dict[str, Any]
    last_frame_ordinal: int
    quad_fit: dict[str, Any] = field(default_factory=dict)
    ellipse_fit: dict[str, Any] = field(default_factory=dict)
    solve_pnp: dict[str, Any] = field(default_factory=dict)
    total_confidence: float = 0.0
    tracking_layers: dict[str, Any] = field(default_factory=dict)
    age_frames: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)


class InnerVoidTracker:
    def __init__(self, config: ProjectionConfig):
        self.settings = config.section("innerVoidInstanceTracking")
        self.tracks: dict[str, _Track] = {}
        self.next_index = 1

    def _new_id(self) -> str:
        instance_id = f"inner-void-{self.next_index:04d}"
        self.next_index += 1
        return instance_id

    def _active_tracks(self, frame_ordinal: int) -> list[_Track]:
        max_gap = _int(self.settings, "maxFrameGap", 5)
        return [track for track in self.tracks.values() if int(frame_ordinal) - int(track.last_frame_ordinal) <= max_gap]

    def _layer_weights(self, has_quad: bool, has_ellipse: bool, has_solve_pnp: bool) -> dict[str, float]:
        return _normalize_weights(
            {
                "inner_void_detection": max(0.0, _float(self.settings, "layerWeightDetection", 0.20)),
                "inner_void_ellipse_fit": 0.0 if not has_ellipse else max(0.0, _float(self.settings, "layerWeightEllipseFit", 0.15)),
                "inner_void_quad_fit": 0.0 if not has_quad else max(0.0, _float(self.settings, "layerWeightQuadFit", 0.30)),
                "inner_void_solve_pnp": 0.0 if not has_solve_pnp else max(0.0, _float(self.settings, "layerWeightSolvePnP", 0.25)),
            }
        )

    def _fit_lookup(self, fit_payload: dict[str, Any], group: str) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        frame = fit_payload.get(group) if isinstance(fit_payload.get(group), dict) else {}
        for observation in frame.get("observations", []) if isinstance(frame.get("observations"), list) else []:
            for fit in [*(observation.get("fits") or []), *(observation.get("rejected_fits") or [])]:
                if not isinstance(fit, dict):
                    continue
                void_id = str(fit.get("void_id") or "")
                previous = lookup.get(void_id)
                if void_id and (previous is None or (_fit_available(fit) and not _fit_available(previous))):
                    lookup[void_id] = fit
        return lookup

    def _compact_solve(self, solve: dict[str, Any]) -> dict[str, Any]:
        fit_quality = solve.get("fitQuality") if isinstance(solve.get("fitQuality"), dict) else {}
        quality = solve.get("quality") if isinstance(solve.get("quality"), dict) else {}
        return {
            "solve_id": str(solve.get("solve_id") or ""),
            "void_id": str(solve.get("void_id") or ""),
            "bbox_id": str(solve.get("bbox_id") or ""),
            "source": str(solve.get("source") or ""),
            "available": bool(solve.get("available")),
            "status": str(solve.get("status") or ""),
            "reason": solve.get("reason"),
            "source_status": str(solve.get("source_status") or ""),
            "source_score": clean_float(float(solve.get("source_score") or 0.0), 6),
            "fit_id": solve.get("fit_id"),
            "fit_score": None if solve.get("fit_score") is None else clean_float(float(solve.get("fit_score") or 0.0), 6),
            "geometry_type": str(solve.get("geometry_type") or ""),
            "xyzCameraM": tuple3(solve.get("xyzCameraM")),
            "rpyCameraDeg": tuple3(solve.get("rpyCameraDeg")),
            "depthM": None if solve.get("depthM") is None else clean_float(float(solve.get("depthM") or 0.0), 6),
            "reprojectionErrorPx": None
            if solve.get("reprojectionErrorPx") is None
            else clean_float(float(solve.get("reprojectionErrorPx") or 0.0), 6),
            "solvePnPMethod": solve.get("solvePnPMethod"),
            "fitQuality": dict(fit_quality),
            "quality": dict(quality),
        }

    def _solve_pnp_lookup(self, solve_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for observation in solve_payload.get("observations", []) if isinstance(solve_payload.get("observations"), list) else []:
            for solve in observation.get("solves") or []:
                if not isinstance(solve, dict):
                    continue
                void_id = str(solve.get("void_id") or "")
                source = str(solve.get("source") or "")
                if not void_id or not source:
                    continue
                entry = lookup.setdefault(
                    void_id,
                    {
                        "void_id": void_id,
                        "bbox_id": str(solve.get("bbox_id") or ""),
                        "sources": {},
                        "available_count": 0,
                        "rejected_count": 0,
                        "best_available_source": None,
                    },
                )
                compact = self._compact_solve(solve)
                previous = entry["sources"].get(source)
                if previous is None or (compact["available"] and not bool(previous.get("available"))):
                    entry["sources"][source] = compact
        for entry in lookup.values():
            sources = entry.get("sources") if isinstance(entry.get("sources"), dict) else {}
            available = {key: value for key, value in sources.items() if isinstance(value, dict) and value.get("available")}
            entry["available_count"] = len(available)
            entry["rejected_count"] = max(0, len(sources) - len(available))
            if available:
                entry["best_available_source"] = max(
                    available.items(),
                    key=lambda item: (
                        clamp01(item[1].get("fit_score", item[1].get("source_score", 0.0))),
                        -float(item[1].get("reprojectionErrorPx") or 1e9),
                    ),
                )[0]
        return lookup

    def _layer(
        self,
        *,
        label: str,
        available: bool,
        confidence: float,
        agreement: float,
        weight: float,
        status: str,
        reason: str | None,
        current: dict[str, Any] | None,
        prior: dict[str, Any] | None,
        signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        confidence_value = clamp01(confidence)
        agreement_value = clamp01(agreement)
        return {
            "label": label,
            "available": bool(available),
            "confidence": clean_float(confidence_value, 6),
            "agreement": clean_float(agreement_value, 6),
            "weight": clean_float(float(weight), 6),
            "score": clean_float(confidence_value * agreement_value, 6),
            "status": status,
            "reason": reason,
            "current": current or {},
            "prior": prior or {},
            "signals": signals or {},
        }

    def _detection_layer(self, record: dict[str, Any], prior_record: dict[str, Any] | None, weight: float) -> tuple[dict[str, Any], bool, list[str]]:
        void = record.get("void") if isinstance(record.get("void"), dict) else {}
        prior = prior_record.get("void") if isinstance(prior_record, dict) and isinstance(prior_record.get("void"), dict) else {}
        current_quality = _void_score(void)
        if not prior:
            min_void_score = _float(self.settings, "minVoidScore", 0.45)
            accepted = current_quality >= min_void_score
            return (
                self._layer(
                    label="Inner Void Detection",
                    available=bool(void),
                    confidence=current_quality,
                    agreement=0.0,
                    weight=weight,
                    status="current-only" if accepted else "rejected",
                    reason=None if accepted else "void-score-too-low",
                    current=void,
                    prior=None,
                    signals={"currentVoidScore": clean_float(current_quality, 6), "minVoidScore": clean_float(min_void_score, 6)},
                ),
                accepted,
                [] if accepted else ["void-score-too-low"],
            )
        center_distance = _distance(_point(void.get("center_px")), _point(prior.get("center_px")))
        max_center = max(1e-6, _float(self.settings, "maxCenterDistancePx", 80.0))
        center_score = 0.0 if center_distance is None else clamp01(1.0 - center_distance / max_center)
        iou = _bbox_iou(_bbox(void.get("bbox_px")), _bbox(prior.get("bbox_px")))
        min_iou = _float(self.settings, "minIou", 0.01)
        iou_score = 1.0 if min_iou <= 0.0 else clamp01(iou / max(1e-6, min_iou))
        area_score, area_delta = _ratio_similarity(void.get("area_px"), prior.get("area_px"))
        pixel_score, pixel_delta = _ratio_similarity(void.get("pixel_count"), prior.get("pixel_count"))
        size_score = (area_score + pixel_score) * 0.5
        current_size = _rect_size(void)
        prior_size = _rect_size(prior)
        if current_size and prior_size:
            width_score, _ = _ratio_similarity(current_size[0], prior_size[0])
            height_score, _ = _ratio_similarity(current_size[1], prior_size[1])
            size_score = (size_score + width_score + height_score) / 3.0
        delta_angle = _angle_delta(_rect_angle(void), _rect_angle(prior))
        max_angle = max(1e-6, _float(self.settings, "maxAngleDeltaDeg", 35.0))
        angle_score = 0.0 if delta_angle is None else clamp01(1.0 - delta_angle / max_angle)
        corner_distance = _mean_corner_distance(void, prior)
        max_corner = max(1e-6, _float(self.settings, "maxCornerDistancePx", 45.0))
        corner_score = 0.0 if corner_distance is None else clamp01(1.0 - corner_distance / max_corner)
        prior_quality = _void_score(prior)
        void_quality_score = clamp01((current_quality + prior_quality + (1.0 - abs(current_quality - prior_quality))) / 3.0)
        raw_weights = {
            "center": max(0.0, _float(self.settings, "scoreWeightCenter", 0.30)),
            "iou": max(0.0, _float(self.settings, "scoreWeightIou", 0.15)),
            "size": max(0.0, _float(self.settings, "scoreWeightSize", 0.15)),
            "angle": max(0.0, _float(self.settings, "scoreWeightAngle", 0.10)),
            "corners": max(0.0, _float(self.settings, "scoreWeightCorners", 0.20)),
            "voidQuality": max(0.0, _float(self.settings, "scoreWeightVoidQuality", 0.10)),
        }
        weights = _normalize_weights(raw_weights)
        reasons: list[str] = []
        if center_distance is None:
            reasons.append("missing-center")
        elif center_distance > max_center:
            reasons.append("center-distance-too-large")
        if area_delta > _float(self.settings, "maxAreaRatioDelta", 0.65):
            reasons.append("area-delta-too-large")
        if pixel_delta > _float(self.settings, "maxPixelRatioDelta", 0.65):
            reasons.append("pixel-delta-too-large")
        if delta_angle is not None and delta_angle > max_angle:
            reasons.append("angle-delta-too-large")
        if corner_distance is not None and corner_distance > max_corner:
            reasons.append("corner-distance-too-large")
        score = (
            weights["center"] * center_score
            + weights["iou"] * iou_score
            + weights["size"] * size_score
            + weights["angle"] * angle_score
            + weights["corners"] * corner_score
            + weights["voidQuality"] * void_quality_score
        )
        if reasons:
            score = 0.0
        layer = self._layer(
            label="Inner Void Detection",
            available=True,
            confidence=current_quality,
            agreement=score,
            weight=weight,
            status="accepted" if not reasons else "rejected",
            reason=None if not reasons else reasons[0],
            current=void,
            prior=prior,
            signals={
                "centerDistancePx": None if center_distance is None else clean_float(center_distance, 4),
                "centerScore": clean_float(center_score, 6),
                "iou": clean_float(iou, 6),
                "iouScore": clean_float(iou_score, 6),
                "areaDelta": clean_float(area_delta, 6),
                "pixelDelta": clean_float(pixel_delta, 6),
                "sizeScore": clean_float(size_score, 6),
                "angleDeltaDeg": None if delta_angle is None else clean_float(delta_angle, 4),
                "angleScore": clean_float(angle_score, 6),
                "cornerDistancePx": None if corner_distance is None else clean_float(corner_distance, 4),
                "cornerScore": clean_float(corner_score, 6),
                "currentVoidScore": clean_float(current_quality, 6),
                "priorVoidScore": clean_float(prior_quality, 6),
                "voidQualityScore": clean_float(void_quality_score, 6),
                "weights": {key: clean_float(value, 6) for key, value in weights.items()},
            },
        )
        return layer, not reasons, reasons

    def _quad_layer(self, record: dict[str, Any], prior_record: dict[str, Any] | None, weight: float) -> dict[str, Any]:
        fit = record.get("quad_fit") if isinstance(record.get("quad_fit"), dict) else {}
        prior = prior_record.get("quad_fit") if isinstance(prior_record, dict) and isinstance(prior_record.get("quad_fit"), dict) else {}
        confidence = _fit_score(fit) if _fit_available(fit) else 0.0
        if not fit:
            return self._layer(
                label="Inner Void Quad Fit",
                available=False,
                confidence=0.0,
                agreement=0.0,
                weight=weight,
                status="missing",
                reason="missing-quad-fit",
                current=None,
                prior=prior,
            )
        if not _fit_available(fit):
            return self._layer(
                label="Inner Void Quad Fit",
                available=False,
                confidence=0.0,
                agreement=0.0,
                weight=weight,
                status=str(fit.get("status") or "rejected"),
                reason=_fit_reason(fit, "quad-fit-unavailable"),
                current=fit,
                prior=prior,
            )
        min_quad_score = _float(self.settings, "minQuadScore", 0.50)
        if confidence < min_quad_score:
            return self._layer(
                label="Inner Void Quad Fit",
                available=True,
                confidence=confidence,
                agreement=0.0,
                weight=weight,
                status="rejected",
                reason="quad-score-too-low",
                current=fit,
                prior=prior,
                signals={"minQuadScore": clean_float(min_quad_score, 6)},
            )
        if not _fit_available(prior):
            return self._layer(
                label="Inner Void Quad Fit",
                available=True,
                confidence=confidence,
                agreement=0.0,
                weight=weight,
                status="current-only",
                reason="missing-prior-quad-fit",
                current=fit,
                prior=prior,
            )
        current_points = _quad_points(fit)
        prior_points = _quad_points(prior)
        center_distance = _distance(_quad_center(fit), _quad_center(prior))
        max_center = max(1e-6, _float(self.settings, "maxQuadCenterDistancePx", _float(self.settings, "maxCenterDistancePx", 80.0)))
        center_score = 0.0 if center_distance is None else clamp01(1.0 - center_distance / max_center)
        point_distance = _mean_point_distance(current_points, prior_points)
        max_points = max(1e-6, _float(self.settings, "maxQuadCornerDistancePx", _float(self.settings, "maxCornerDistancePx", 45.0)))
        point_score = 0.0 if point_distance is None else clamp01(1.0 - point_distance / max_points)
        area_score, area_delta = _ratio_similarity(fit.get("quad_area_px"), prior.get("quad_area_px"))
        angle_delta = _angle_delta(_points_angle(current_points), _points_angle(prior_points))
        max_angle = max(1e-6, _float(self.settings, "maxQuadAngleDeltaDeg", _float(self.settings, "maxAngleDeltaDeg", 35.0)))
        angle_score = 0.0 if angle_delta is None else clamp01(1.0 - angle_delta / max_angle)
        reasons: list[str] = []
        if center_distance is None:
            reasons.append("missing-quad-center")
        elif center_distance > max_center:
            reasons.append("quad-center-distance-too-large")
        if point_distance is not None and point_distance > max_points:
            reasons.append("quad-point-distance-too-large")
        if area_delta > _float(self.settings, "maxQuadAreaRatioDelta", _float(self.settings, "maxAreaRatioDelta", 0.65)):
            reasons.append("quad-area-delta-too-large")
        if angle_delta is not None and angle_delta > max_angle:
            reasons.append("quad-angle-delta-too-large")
        agreement = 0.35 * center_score + 0.35 * point_score + 0.20 * area_score + 0.10 * angle_score
        if reasons:
            agreement = 0.0
        return self._layer(
            label="Inner Void Quad Fit",
            available=True,
            confidence=confidence,
            agreement=agreement,
            weight=weight,
            status="accepted" if not reasons else "rejected",
            reason=None if not reasons else reasons[0],
            current=fit,
            prior=prior,
            signals={
                "centerDistancePx": None if center_distance is None else clean_float(center_distance, 4),
                "centerScore": clean_float(center_score, 6),
                "pointDistancePx": None if point_distance is None else clean_float(point_distance, 4),
                "pointScore": clean_float(point_score, 6),
                "areaDelta": clean_float(area_delta, 6),
                "areaScore": clean_float(area_score, 6),
                "angleDeltaDeg": None if angle_delta is None else clean_float(angle_delta, 4),
                "angleScore": clean_float(angle_score, 6),
            },
        )

    def _ellipse_layer(self, record: dict[str, Any], prior_record: dict[str, Any] | None, weight: float) -> dict[str, Any]:
        fit = record.get("ellipse_fit") if isinstance(record.get("ellipse_fit"), dict) else {}
        prior = prior_record.get("ellipse_fit") if isinstance(prior_record, dict) and isinstance(prior_record.get("ellipse_fit"), dict) else {}
        confidence = _fit_score(fit) if _fit_available(fit) else 0.0
        if not fit:
            return self._layer(
                label="Inner Void Ellipse Fit",
                available=False,
                confidence=0.0,
                agreement=0.0,
                weight=weight,
                status="missing",
                reason="missing-ellipse-fit",
                current=None,
                prior=prior,
            )
        if not _fit_available(fit):
            return self._layer(
                label="Inner Void Ellipse Fit",
                available=False,
                confidence=0.0,
                agreement=0.0,
                weight=weight,
                status=str(fit.get("status") or "rejected"),
                reason=_fit_reason(fit, "ellipse-fit-unavailable"),
                current=fit,
                prior=prior,
            )
        min_ellipse_score = _float(self.settings, "minEllipseScore", 0.40)
        if confidence < min_ellipse_score:
            return self._layer(
                label="Inner Void Ellipse Fit",
                available=True,
                confidence=confidence,
                agreement=0.0,
                weight=weight,
                status="rejected",
                reason="ellipse-score-too-low",
                current=fit,
                prior=prior,
                signals={"minEllipseScore": clean_float(min_ellipse_score, 6)},
            )
        if not _fit_available(prior):
            return self._layer(
                label="Inner Void Ellipse Fit",
                available=True,
                confidence=confidence,
                agreement=0.0,
                weight=weight,
                status="current-only",
                reason="missing-prior-ellipse-fit",
                current=fit,
                prior=prior,
            )
        center_distance = _distance(_ellipse_center(fit), _ellipse_center(prior))
        max_center = max(1e-6, _float(self.settings, "maxEllipseCenterDistancePx", _float(self.settings, "maxCenterDistancePx", 80.0)))
        center_score = 0.0 if center_distance is None else clamp01(1.0 - center_distance / max_center)
        area_score, area_delta = _ratio_similarity(fit.get("ellipse_area_px"), prior.get("ellipse_area_px"))
        axis_score, axis_delta = _ratio_similarity(_ellipse_axis_ratio(fit), _ellipse_axis_ratio(prior))
        angle_delta = _angle_delta(fit.get("ellipse_angle_deg"), prior.get("ellipse_angle_deg"))
        max_angle = max(1e-6, _float(self.settings, "maxEllipseAngleDeltaDeg", _float(self.settings, "maxAngleDeltaDeg", 35.0)))
        angle_score = 0.0 if angle_delta is None else clamp01(1.0 - angle_delta / max_angle)
        reasons: list[str] = []
        if center_distance is None:
            reasons.append("missing-ellipse-center")
        elif center_distance > max_center:
            reasons.append("ellipse-center-distance-too-large")
        if area_delta > _float(self.settings, "maxEllipseAreaRatioDelta", _float(self.settings, "maxAreaRatioDelta", 0.65)):
            reasons.append("ellipse-area-delta-too-large")
        if axis_delta > _float(self.settings, "maxEllipseAxisRatioDelta", 0.65):
            reasons.append("ellipse-axis-delta-too-large")
        if angle_delta is not None and angle_delta > max_angle:
            reasons.append("ellipse-angle-delta-too-large")
        agreement = 0.40 * center_score + 0.25 * area_score + 0.20 * angle_score + 0.15 * axis_score
        if reasons:
            agreement = 0.0
        return self._layer(
            label="Inner Void Ellipse Fit",
            available=True,
            confidence=confidence,
            agreement=agreement,
            weight=weight,
            status="accepted" if not reasons else "rejected",
            reason=None if not reasons else reasons[0],
            current=fit,
            prior=prior,
            signals={
                "centerDistancePx": None if center_distance is None else clean_float(center_distance, 4),
                "centerScore": clean_float(center_score, 6),
                "areaDelta": clean_float(area_delta, 6),
                "areaScore": clean_float(area_score, 6),
                "axisDelta": clean_float(axis_delta, 6),
                "axisScore": clean_float(axis_score, 6),
                "angleDeltaDeg": None if angle_delta is None else clean_float(angle_delta, 4),
                "angleScore": clean_float(angle_score, 6),
            },
        )

    def _solve_source_weights(self) -> dict[str, float]:
        return _normalize_weights(
            {
                "inner_void_detection_rect": _float(self.settings, "solvePnPSourceWeightRect", 1.0),
                "inner_void_quad_fit": _float(self.settings, "solvePnPSourceWeightQuad", 1.0),
                "inner_void_ellipse_fit": _float(self.settings, "solvePnPSourceWeightEllipse", 1.0),
            }
        )

    def _solve_confidence(self, solve: dict[str, Any] | None) -> float:
        if not solve or not solve.get("available"):
            return 0.0
        fit_quality = solve.get("fitQuality") if isinstance(solve.get("fitQuality"), dict) else {}
        max_reprojection = max(1e-6, _float(self.settings, "maxSolvePnPReprojectionErrorPx", 8.0))
        reprojection_score = 0.0
        if solve.get("reprojectionErrorPx") is not None:
            reprojection_score = clamp01(1.0 - float(solve.get("reprojectionErrorPx") or 0.0) / max_reprojection)
        quality_score = clamp01(fit_quality.get("overall", reprojection_score))
        raw_score = solve.get("fit_score")
        if raw_score is None:
            raw_score = solve.get("source_score", quality_score)
        return clamp01((quality_score + reprojection_score + clamp01(raw_score)) / 3.0)

    def _solve_quality_score(self, solve: dict[str, Any] | None) -> float:
        if not solve or not solve.get("available"):
            return 0.0
        fit_quality = solve.get("fitQuality") if isinstance(solve.get("fitQuality"), dict) else {}
        if fit_quality.get("overall") is not None:
            return clamp01(fit_quality.get("overall"))
        return self._solve_confidence(solve)

    def _same_frame_solve_agreement(self, available_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
        sources = [(key, value) for key, value in available_sources.items() if isinstance(value, dict) and value.get("available")]
        max_xyz = max(1e-6, _float(self.settings, "maxSolvePnPXyzDistanceM", 3.0))
        max_depth = max(1e-6, _float(self.settings, "maxSolvePnPDepthDeltaM", 5.0))
        max_normal = max(1e-6, _float(self.settings, "maxSolvePnPRpyDeltaDeg", 50.0))
        if len(sources) < 2:
            return {
                "available": False,
                "agreement": 1.0 if len(sources) == 1 else 0.0,
                "reason": "single-solvepnp-source" if len(sources) == 1 else "no-solvepnp-sources",
                "pairs": [],
            }
        pairs: list[dict[str, Any]] = []
        for index, (left_name, left) in enumerate(sources):
            for right_name, right in sources[index + 1 :]:
                xyz_distance = distance3(left.get("xyzCameraM"), right.get("xyzCameraM"))
                depth_delta = _scalar_delta(left.get("depthM"), right.get("depthM"))
                normal_delta = _plane_normal_delta_deg(left.get("rpyCameraDeg"), right.get("rpyCameraDeg"))
                xyz_score = 0.0 if xyz_distance is None else clamp01(1.0 - xyz_distance / max_xyz)
                depth_score = 0.0 if depth_delta is None else clamp01(1.0 - depth_delta / max_depth)
                normal_score = 0.0 if normal_delta is None else clamp01(1.0 - normal_delta / max_normal)
                reasons: list[str] = []
                if xyz_distance is None:
                    reasons.append("same-frame-missing-xyz")
                elif xyz_distance > max_xyz:
                    reasons.append("same-frame-xyz-distance-too-large")
                if depth_delta is None:
                    reasons.append("same-frame-missing-depth")
                elif depth_delta > max_depth:
                    reasons.append("same-frame-depth-delta-too-large")
                if normal_delta is None:
                    reasons.append("same-frame-missing-plane-normal")
                elif normal_delta > max_normal:
                    reasons.append("same-frame-plane-normal-delta-too-large")
                score = 0.45 * xyz_score + 0.20 * depth_score + 0.35 * normal_score
                if reasons:
                    score = 0.0
                pairs.append(
                    {
                        "sources": [left_name, right_name],
                        "status": "accepted" if not reasons else "rejected",
                        "reason": None if not reasons else reasons[0],
                        "score": clean_float(score, 6),
                        "signals": {
                            "xyzDistanceM": None if xyz_distance is None else clean_float(xyz_distance, 6),
                            "xyzScore": clean_float(xyz_score, 6),
                            "depthDeltaM": None if depth_delta is None else clean_float(depth_delta, 6),
                            "depthScore": clean_float(depth_score, 6),
                            "planeNormalDeltaDeg": None if normal_delta is None else clean_float(normal_delta, 6),
                            "planeNormalScore": clean_float(normal_score, 6),
                        },
                    }
                )
        agreement = sum(float(pair["score"]) for pair in pairs) / max(1, len(pairs))
        rejected = [pair for pair in pairs if str(pair.get("status")) == "rejected"]
        return {
            "available": True,
            "agreement": clean_float(agreement, 6),
            "reason": None if not rejected else str(rejected[0].get("reason") or "same-frame-solvepnp-disagreement"),
            "pairs": pairs,
        }

    def _solve_source_score(self, current: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
        max_xyz = max(1e-6, _float(self.settings, "maxSolvePnPXyzDistanceM", 3.0))
        max_depth = max(1e-6, _float(self.settings, "maxSolvePnPDepthDeltaM", 5.0))
        max_rpy = max(1e-6, _float(self.settings, "maxSolvePnPRpyDeltaDeg", 50.0))
        max_reprojection = max(1e-6, _float(self.settings, "maxSolvePnPReprojectionErrorPx", 8.0))
        min_quality = _float(self.settings, "minPoseFitQuality", 0.45)
        xyz_distance = distance3(current.get("xyzCameraM"), prior.get("xyzCameraM"))
        depth_delta = _scalar_delta(current.get("depthM"), prior.get("depthM"))
        rpy_delta = _rpy_delta(current.get("rpyCameraDeg"), prior.get("rpyCameraDeg"))
        reprojection_values = [float(value) for value in (current.get("reprojectionErrorPx"), prior.get("reprojectionErrorPx")) if value is not None]
        reprojection_error = sum(reprojection_values) / max(1, len(reprojection_values)) if reprojection_values else None
        xyz_score = 0.0 if xyz_distance is None else clamp01(1.0 - xyz_distance / max_xyz)
        depth_score = 0.0 if depth_delta is None else clamp01(1.0 - depth_delta / max_depth)
        rpy_score = 0.0 if rpy_delta is None else clamp01(1.0 - rpy_delta / max_rpy)
        reprojection_score = 0.0 if reprojection_error is None else clamp01(1.0 - reprojection_error / max_reprojection)
        confidence = self._solve_confidence(current)
        quality_score = self._solve_quality_score(current)
        reasons: list[str] = []
        if quality_score < min_quality:
            reasons.append("solvepnp-fit-quality-too-low")
        if xyz_distance is None:
            reasons.append("solvepnp-missing-xyz")
        elif xyz_distance > max_xyz:
            reasons.append("solvepnp-xyz-distance-too-large")
        if depth_delta is None:
            reasons.append("solvepnp-missing-depth")
        elif depth_delta > max_depth:
            reasons.append("solvepnp-depth-delta-too-large")
        if rpy_delta is None:
            reasons.append("solvepnp-missing-rpy")
        elif rpy_delta > max_rpy:
            reasons.append("solvepnp-rpy-delta-too-large")
        if reprojection_error is not None and reprojection_error > max_reprojection:
            reasons.append("solvepnp-reprojection-error-too-large")
        agreement = 0.40 * xyz_score + 0.20 * depth_score + 0.25 * rpy_score + 0.15 * reprojection_score
        if reasons:
            agreement = 0.0
        return {
            "available": True,
            "status": "accepted" if not reasons else "rejected",
            "reason": None if not reasons else reasons[0],
            "confidence": clean_float(confidence, 6),
            "agreement": clean_float(agreement, 6),
            "current": current,
            "prior": prior,
            "signals": {
                "xyzDistanceM": None if xyz_distance is None else clean_float(xyz_distance, 6),
                "xyzScore": clean_float(xyz_score, 6),
                "depthDeltaM": None if depth_delta is None else clean_float(depth_delta, 6),
                "depthScore": clean_float(depth_score, 6),
                "rpyDeltaDeg": None if rpy_delta is None else clean_float(rpy_delta, 6),
                "rpyScore": clean_float(rpy_score, 6),
                "reprojectionErrorPx": None if reprojection_error is None else clean_float(reprojection_error, 6),
                "reprojectionScore": clean_float(reprojection_score, 6),
                "fitQualityScore": clean_float(quality_score, 6),
                "minPoseFitQuality": clean_float(min_quality, 6),
            },
        }

    def _solve_pnp_layer(self, record: dict[str, Any], prior_record: dict[str, Any] | None, weight: float) -> tuple[dict[str, Any], bool, list[str]]:
        solve_pnp = record.get("solve_pnp") if isinstance(record.get("solve_pnp"), dict) else {}
        prior_solve_pnp = (
            prior_record.get("solve_pnp")
            if isinstance(prior_record, dict) and isinstance(prior_record.get("solve_pnp"), dict)
            else {}
        )
        sources = solve_pnp.get("sources") if isinstance(solve_pnp.get("sources"), dict) else {}
        prior_sources = prior_solve_pnp.get("sources") if isinstance(prior_solve_pnp.get("sources"), dict) else {}
        current_available = {key: value for key, value in sources.items() if isinstance(value, dict) and value.get("available")}
        source_weights = self._solve_source_weights()
        confidence = sum(source_weights.get(key, 0.0) * self._solve_confidence(value) for key, value in current_available.items())
        same_frame = self._same_frame_solve_agreement(current_available)
        common_signals = {
            "sourceWeights": {key: clean_float(value, 6) for key, value in source_weights.items()},
            "sameFrameAgreement": same_frame,
        }
        if weight <= 0.0:
            return (
                self._layer(
                    label="Inner Void SolvePnP",
                    available=bool(current_available),
                    confidence=confidence,
                    agreement=0.0,
                    weight=weight,
                    status="disabled",
                    reason="zero-layer-weight",
                    current=solve_pnp,
                    prior=prior_solve_pnp,
                    signals=common_signals,
                ),
                True,
                [],
            )
        if not solve_pnp:
            return (
                self._layer(
                    label="Inner Void SolvePnP",
                    available=False,
                    confidence=0.0,
                    agreement=0.0,
                    weight=weight,
                    status="missing",
                    reason="missing-solvepnp",
                    current=None,
                    prior=prior_solve_pnp,
                    signals=common_signals,
                ),
                True,
                [],
            )
        if not prior_solve_pnp:
            return (
                self._layer(
                    label="Inner Void SolvePnP",
                    available=bool(current_available),
                    confidence=confidence,
                    agreement=0.0,
                    weight=weight,
                    status="current-only",
                    reason="missing-prior-solvepnp",
                    current=solve_pnp,
                    prior=None,
                    signals=common_signals,
                ),
                True,
                [],
            )
        source_scores: dict[str, Any] = {}
        comparable: dict[str, Any] = {}
        for source in SOLVE_PNP_SOURCES:
            current = sources.get(source) if isinstance(sources.get(source), dict) else {}
            prior = prior_sources.get(source) if isinstance(prior_sources.get(source), dict) else {}
            if not current or not current.get("available"):
                source_scores[source] = {
                    "available": False,
                    "status": str(current.get("status") or "missing"),
                    "reason": current.get("reason") or "missing-current-solvepnp",
                    "current": current,
                    "prior": prior,
                }
                continue
            if not prior or not prior.get("available"):
                source_scores[source] = {
                    "available": False,
                    "status": str(prior.get("status") or "missing-prior"),
                    "reason": prior.get("reason") or "missing-prior-solvepnp",
                    "current": current,
                    "prior": prior,
                }
                continue
            score = self._solve_source_score(current, prior)
            source_scores[source] = score
            comparable[source] = score
        if not comparable:
            signals = dict(common_signals)
            signals["sources"] = source_scores
            return (
                self._layer(
                    label="Inner Void SolvePnP",
                    available=bool(current_available),
                    confidence=confidence,
                    agreement=0.0,
                    weight=weight,
                    status="not-comparable",
                    reason="no-comparable-solvepnp-sources",
                    current=solve_pnp,
                    prior=prior_solve_pnp,
                    signals=signals,
                ),
                True,
                [],
            )
        comparable_weights = _normalize_weights({key: source_weights.get(key, 0.0) for key in comparable})
        temporal_agreement = sum(comparable_weights.get(key, 0.0) * float(value.get("agreement") or 0.0) for key, value in comparable.items())
        same_frame_agreement = float(same_frame.get("agreement") or 0.0)
        agreement = 0.65 * temporal_agreement + 0.35 * same_frame_agreement if bool(same_frame.get("available")) else temporal_agreement
        accepted_sources = [key for key, value in comparable.items() if str(value.get("status")) == "accepted"]
        reasons = [] if accepted_sources else ["solvepnp-all-comparable-sources-rejected"]
        if bool(same_frame.get("available")) and same_frame_agreement <= 0.0 and same_frame.get("reason"):
            reasons.append(str(same_frame.get("reason")))
        signals = dict(common_signals)
        signals.update(
            {
                "comparableWeights": {key: clean_float(value, 6) for key, value in comparable_weights.items()},
                "temporalAgreement": clean_float(temporal_agreement, 6),
                "sources": source_scores,
                "acceptedSources": accepted_sources,
            }
        )
        return (
            self._layer(
                label="Inner Void SolvePnP",
                available=bool(current_available),
                confidence=confidence,
                agreement=agreement,
                weight=weight,
                status="accepted" if accepted_sources else "rejected",
                reason=None if accepted_sources else reasons[0],
                current=solve_pnp,
                prior=prior_solve_pnp,
                signals=signals,
            ),
            bool(accepted_sources),
            reasons,
        )

    def _total_layer_value(self, layers: dict[str, dict[str, Any]], field_name: str) -> float:
        total = 0.0
        for layer in layers.values():
            if isinstance(layer, dict):
                total += float(layer.get("weight") or 0.0) * clamp01(layer.get(field_name, 0.0))
        return clean_float(total, 6)

    def _final_layer_score(self, layer: dict[str, Any]) -> tuple[float, str]:
        confidence = clamp01(layer.get("confidence", 0.0))
        agreement = clamp01(layer.get("agreement", 0.0))
        status = str(layer.get("status") or "")
        if status in {"missing", "rejected", "disabled"} or not bool(layer.get("available")):
            return 0.0, "not-contributing"
        if status == "current-only":
            signals = layer.get("signals") if isinstance(layer.get("signals"), dict) else {}
            same_frame = signals.get("sameFrameAgreement") if isinstance(signals.get("sameFrameAgreement"), dict) else {}
            if same_frame.get("available"):
                return clean_float(confidence * clamp01(same_frame.get("agreement", 0.0)), 6), "current-confidence-with-same-frame-agreement"
            return clean_float(confidence, 6), "current-confidence"
        if status == "not-comparable":
            return clean_float(confidence, 6), "current-confidence-not-comparable"
        return clean_float(confidence * agreement, 6), "confidence-times-agreement"

    def _final_score_breakdown(self, layers: dict[str, dict[str, Any]]) -> dict[str, Any]:
        min_final = _float(self.settings, "minFinalInstanceScore", 0.55)
        missing_penalty = max(0.0, _float(self.settings, "missingSourcePenalty", 0.15))
        normalize_available = _bool(self.settings, "normalizeAvailableWeights", True)
        weighted_total = 0.0
        available_weight = 0.0
        penalty_total = 0.0
        layer_payload: dict[str, Any] = {}
        for key, layer in layers.items():
            if not isinstance(layer, dict):
                continue
            weight = max(0.0, float(layer.get("weight") or 0.0))
            score, mode = self._final_layer_score(layer)
            contributes = score > 0.0 and bool(layer.get("available"))
            if contributes:
                available_weight += weight
            elif weight > 0.0 and str(layer.get("status") or "") != "disabled":
                penalty_total += missing_penalty * weight
            weighted_total += weight * score
            layer_payload[key] = {
                "label": layer.get("label"),
                "available": bool(layer.get("available")),
                "confidence": clean_float(float(layer.get("confidence") or 0.0), 6),
                "agreement": clean_float(float(layer.get("agreement") or 0.0), 6),
                "weight": clean_float(weight, 6),
                "score": clean_float(score, 6),
                "scoreMode": mode,
                "status": layer.get("status"),
                "reason": layer.get("reason"),
            }
        raw_score = weighted_total / available_weight if normalize_available and available_weight > 0.0 else weighted_total
        final_score = clamp01(raw_score - penalty_total)
        return {
            "schema": "projection-inner-void-instance-final-score.v1",
            "status": "accepted-preview" if final_score >= min_final else "below-threshold",
            "finalInstanceScore": clean_float(final_score, 6),
            "minFinalInstanceScore": clean_float(min_final, 6),
            "rawScore": clean_float(raw_score, 6),
            "missingSourcePenalty": clean_float(missing_penalty, 6),
            "penaltyTotal": clean_float(penalty_total, 6),
            "normalizeAvailableWeights": normalize_available,
            "availableWeight": clean_float(available_weight, 6),
            "layers": layer_payload,
        }

    def _track_record(self, track: _Track) -> dict[str, Any]:
        return {
            "void": track.void,
            "quad_fit": track.quad_fit,
            "ellipse_fit": track.ellipse_fit,
            "solve_pnp": track.solve_pnp,
        }

    def _layers_for_record(
        self,
        record: dict[str, Any],
        prior_record: dict[str, Any] | None,
        layer_weights: dict[str, float],
    ) -> tuple[dict[str, Any], bool, list[str]]:
        detection, eligible, reasons = self._detection_layer(record, prior_record, layer_weights.get("inner_void_detection", 1.0))
        solve_pnp, solve_eligible, solve_reasons = self._solve_pnp_layer(record, prior_record, layer_weights.get("inner_void_solve_pnp", 0.0))
        return (
            {
                "inner_void_detection": detection,
                "inner_void_ellipse_fit": self._ellipse_layer(record, prior_record, layer_weights.get("inner_void_ellipse_fit", 0.0)),
                "inner_void_quad_fit": self._quad_layer(record, prior_record, layer_weights.get("inner_void_quad_fit", 0.0)),
                "inner_void_solve_pnp": solve_pnp,
            },
            eligible and solve_eligible,
            [*reasons, *solve_reasons],
        )

    def _compact_layers(self, layers: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, layer in layers.items():
            if not isinstance(layer, dict):
                continue
            compact[key] = {
                "label": layer.get("label"),
                "available": bool(layer.get("available")),
                "confidence": clean_float(float(layer.get("confidence") or 0.0), 6),
                "agreement": clean_float(float(layer.get("agreement") or 0.0), 6),
                "weight": clean_float(float(layer.get("weight") or 0.0), 6),
                "score": clean_float(float(layer.get("score") or 0.0), 6),
                "status": layer.get("status"),
                "reason": layer.get("reason"),
                "signals": layer.get("signals") if isinstance(layer.get("signals"), dict) else {},
            }
        return compact

    def _compact_candidate_score(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "instanceId": candidate.get("instanceId"),
            "score": clean_float(float(candidate.get("score") or 0.0), 6),
            "eligible": bool(candidate.get("eligible")),
            "reasons": list(candidate.get("reasons") or []),
            "totalConfidence": clean_float(float(candidate.get("totalConfidence") or 0.0), 6),
            "totalAgreement": clean_float(float(candidate.get("totalAgreement") or 0.0), 6),
            "layers": self._compact_layers(candidate.get("layers") if isinstance(candidate.get("layers"), dict) else {}),
        }

    def _score(self, record: dict[str, Any], track: _Track, layer_weights: dict[str, float]) -> dict[str, Any]:
        layers, eligible, reasons = self._layers_for_record(record, self._track_record(track), layer_weights)
        total_confidence = self._total_layer_value(layers, "confidence")
        total_agreement = self._total_layer_value(layers, "agreement")
        score = total_agreement if eligible else 0.0
        return {
            "instanceId": track.instance_id,
            "score": clean_float(score, 6),
            "eligible": eligible,
            "reasons": reasons,
            "totalConfidence": total_confidence,
            "totalAgreement": clean_float(score, 6),
            "layers": layers,
            "layerWeights": {key: clean_float(value, 6) for key, value in layer_weights.items()},
        }

    def _accepted_records(self, inner_void_payload: dict[str, Any], fit_payload: dict[str, Any], solve_payload: dict[str, Any]) -> list[dict[str, Any]]:
        quad_fits = self._fit_lookup(fit_payload, "quad")
        ellipse_fits = self._fit_lookup(fit_payload, "ellipse")
        solve_pnp = self._solve_pnp_lookup(solve_payload)
        records: list[dict[str, Any]] = []
        for observation in inner_void_payload.get("observations", []) if isinstance(inner_void_payload.get("observations"), list) else []:
            for void in observation.get("voids") or []:
                if isinstance(void, dict) and str(void.get("status")) == "accepted":
                    void_id = str(void.get("void_id") or "")
                    records.append(
                        {
                            "void": void,
                            "quad_fit": quad_fits.get(void_id, {}),
                            "ellipse_fit": ellipse_fits.get(void_id, {}),
                            "solve_pnp": solve_pnp.get(void_id, {}),
                        }
                    )
        return sorted(records, key=lambda item: float((item.get("void") or {}).get("score") or 0.0), reverse=True)

    def _available_objects(self, solve_pnp: dict[str, Any]) -> list[dict[str, Any]]:
        sources = solve_pnp.get("sources") if isinstance(solve_pnp.get("sources"), dict) else {}
        return [dict(sources[source]) for source in SOLVE_PNP_SOURCES if isinstance(sources.get(source), dict) and sources[source].get("available")]

    def _compact_solve_pnp_summary(self, solve_pnp: dict[str, Any]) -> dict[str, Any]:
        return {
            "best_available_source": solve_pnp.get("best_available_source"),
            "available_count": int(solve_pnp.get("available_count") or 0),
            "rejected_count": int(solve_pnp.get("rejected_count") or 0),
        }

    def process(self, meta: FrameMeta, inner_void_payload: dict[str, Any], fit_payload: dict[str, Any], solve_payload: dict[str, Any]) -> dict[str, Any]:
        enabled = _bool(self.settings, "enabled", True)
        min_void_score = _float(self.settings, "minVoidScore", 0.35)
        min_association_score = _float(self.settings, "minAssociationScore", 0.55)
        history_limit = max(1, _int(self.settings, "historyLimit", 30))
        layer_weights = self._layer_weights(bool(fit_payload.get("quad")), bool(fit_payload.get("ellipse")), bool(solve_payload))
        active_tracks = self._active_tracks(meta.frame_ordinal)
        claimed: set[str] = set()
        observations: list[dict[str, Any]] = []
        accepted_records = self._accepted_records(inner_void_payload, fit_payload, solve_payload)

        for index, record in enumerate(accepted_records):
            void = record.get("void") if isinstance(record.get("void"), dict) else {}
            void_score = _void_score(void)
            instance_confidence = 0.0
            candidate_scores = sorted(
                [self._score(record, track, layer_weights) for track in active_tracks if track.instance_id not in claimed],
                key=lambda item: float(item["score"]),
                reverse=True,
            )
            tracking_layers, _, _ = self._layers_for_record(record, None, layer_weights)
            total_confidence = self._total_layer_value(tracking_layers, "confidence")
            total_agreement = 0.0
            match_breakdown: dict[str, Any] = {
                "enabled": enabled,
                "minVoidScore": clean_float(min_void_score, 6),
                "minAssociationScore": clean_float(min_association_score, 6),
                "voidScore": clean_float(void_score, 6),
                "layerWeights": {key: clean_float(value, 6) for key, value in layer_weights.items()},
            }
            if not enabled:
                instance_id = ""
                tracking_status = "unmatched"
                association_score = 0.0
                age = 1
                gap = 0
                history: list[dict[str, Any]] = []
                match_breakdown["reason"] = "tracking-disabled"
            elif void_score < min_void_score:
                instance_id = ""
                tracking_status = "unmatched"
                association_score = 0.0
                age = 1
                gap = 0
                history = []
                match_breakdown["reason"] = "void-score-too-low"
            else:
                chosen = candidate_scores[0] if candidate_scores and float(candidate_scores[0]["score"]) >= min_association_score else None
                if chosen is None:
                    instance_id = self._new_id()
                    tracking_status = "new"
                    association_score = 0.0
                    age = 1
                    gap = 0
                    current_history = _history_record(
                        frame_ordinal=meta.frame_ordinal,
                        frame_id=meta.frame_id,
                        record=record,
                        association_score=association_score,
                        total_confidence=total_confidence,
                        tracking_layers=tracking_layers,
                        tracking_status=tracking_status,
                    )
                    history = [current_history]
                    self.tracks[instance_id] = _Track(
                        instance_id=instance_id,
                        void=void,
                        last_frame_ordinal=meta.frame_ordinal,
                        quad_fit=record.get("quad_fit") if isinstance(record.get("quad_fit"), dict) else {},
                        ellipse_fit=record.get("ellipse_fit") if isinstance(record.get("ellipse_fit"), dict) else {},
                        solve_pnp=record.get("solve_pnp") if isinstance(record.get("solve_pnp"), dict) else {},
                        total_confidence=total_confidence,
                        tracking_layers=tracking_layers,
                        age_frames=age,
                        history=history,
                    )
                    match_breakdown["reason"] = "new-track"
                else:
                    instance_id = str(chosen["instanceId"])
                    claimed.add(instance_id)
                    previous = self.tracks[instance_id]
                    tracking_status = "matched"
                    association_score = float(chosen["score"])
                    total_confidence = float(chosen.get("totalConfidence") or total_confidence)
                    total_agreement = float(chosen.get("totalAgreement") or association_score)
                    tracking_layers = chosen.get("layers") if isinstance(chosen.get("layers"), dict) else tracking_layers
                    age = previous.age_frames + 1
                    gap = meta.frame_ordinal - previous.last_frame_ordinal
                    prior_scores = [
                        float(item.get("association_score", 0.0))
                        for item in previous.history
                        if str(item.get("tracking_status")) == "matched"
                    ]
                    confidence_samples = [*prior_scores, association_score]
                    instance_confidence = sum(confidence_samples) / max(1, len(confidence_samples))
                    current_history = _history_record(
                        frame_ordinal=meta.frame_ordinal,
                        frame_id=meta.frame_id,
                        record=record,
                        association_score=association_score,
                        total_confidence=total_confidence,
                        tracking_layers=tracking_layers,
                        tracking_status=tracking_status,
                    )
                    history = [*previous.history, current_history][-history_limit:]
                    self.tracks[instance_id] = _Track(
                        instance_id=instance_id,
                        void=void,
                        last_frame_ordinal=meta.frame_ordinal,
                        quad_fit=record.get("quad_fit") if isinstance(record.get("quad_fit"), dict) else {},
                        ellipse_fit=record.get("ellipse_fit") if isinstance(record.get("ellipse_fit"), dict) else {},
                        solve_pnp=record.get("solve_pnp") if isinstance(record.get("solve_pnp"), dict) else {},
                        total_confidence=total_confidence,
                        tracking_layers=tracking_layers,
                        age_frames=age,
                        history=history,
                    )
                    match_breakdown.update({"reason": "matched", "selected": self._compact_candidate_score(chosen)})

            match_breakdown["totalConfidence"] = clean_float(total_confidence, 6)
            match_breakdown["totalAgreement"] = clean_float(total_agreement if total_agreement else association_score, 6)
            final_score_breakdown = self._final_score_breakdown(tracking_layers)
            final_instance_score = float(final_score_breakdown.get("finalInstanceScore") or 0.0)
            final_instance_status = str(final_score_breakdown.get("status") or "")
            match_breakdown["finalInstanceScore"] = clean_float(final_instance_score, 6)
            match_breakdown["finalInstanceStatus"] = final_instance_status
            solve_summary = self._compact_solve_pnp_summary(record.get("solve_pnp") if isinstance(record.get("solve_pnp"), dict) else {})
            objects = self._available_objects(record.get("solve_pnp") if isinstance(record.get("solve_pnp"), dict) else {})
            observations.append(
                {
                    "frame_ordinal": int(meta.frame_ordinal),
                    "frame_id": meta.frame_id,
                    "observation_id": f"{meta.frame_id}-inner-void-track-{index + 1:03d}",
                    "instance_id": instance_id,
                    "void_id": str(void.get("void_id") or ""),
                    "bbox_id": str(void.get("bbox_id") or ""),
                    "void": _compact_void(void),
                    "tracking_status": tracking_status,
                    "association_score": clean_float(association_score, 6),
                    "agreement_score": clean_float(association_score, 6),
                    "instance_confidence": clean_float(instance_confidence, 6),
                    "total_confidence": clean_float(total_confidence, 6),
                    "total_agreement": clean_float(total_agreement if total_agreement else association_score, 6),
                    "finalInstanceScore": clean_float(final_instance_score, 6),
                    "finalInstanceStatus": final_instance_status,
                    "age_frames": age,
                    "missed_frame_gap": gap,
                    "match_breakdown": match_breakdown,
                    "solve_pnp": solve_summary,
                    "objects": objects,
                    "finalScoreBreakdown": final_score_breakdown,
                    "tracking_layers": self._compact_layers(tracking_layers),
                    "candidate_scores": [self._compact_candidate_score(item) for item in candidate_scores[:5]],
                    "history": history,
                    "smoothed_center_px": None,
                    "gap_fill_candidate": False,
                }
            )

        stale = [
            instance_id
            for instance_id, track in self.tracks.items()
            if meta.frame_ordinal - track.last_frame_ordinal > _int(self.settings, "maxFrameGap", 5)
        ]
        for instance_id in stale:
            del self.tracks[instance_id]
        instances = [dict(item) for item in observations]
        return {
            "schema": "projection-inner-void-instance-tracking.v1",
            "run_id": meta.run_id,
            "frame_ordinal": meta.frame_ordinal,
            "frame_id": meta.frame_id,
            "source_path": meta.source_path,
            "image_width": inner_void_payload["image_width"],
            "image_height": inner_void_payload["image_height"],
            "created_at": utc_now(),
            "observations": observations,
            "instances": instances,
            "active_track_count": len(self.tracks),
            "settings": dict(self.settings),
            "summary": {
                "enabled": enabled,
                "acceptedVoidCount": len(accepted_records),
                "trackedObservationCount": len(observations),
                "matchedCount": sum(1 for item in observations if item["tracking_status"] == "matched"),
                "newCount": sum(1 for item in observations if item["tracking_status"] == "new"),
                "unmatchedCount": sum(1 for item in observations if item["tracking_status"] == "unmatched"),
                "activeTrackCount": len(self.tracks),
                "finalAcceptedPreviewCount": sum(1 for item in observations if item["finalInstanceStatus"] == "accepted-preview"),
                "finalBelowThresholdCount": sum(1 for item in observations if item["finalInstanceStatus"] == "below-threshold"),
                "layerWeights": {key: clean_float(value, 6) for key, value in layer_weights.items()},
            },
        }
