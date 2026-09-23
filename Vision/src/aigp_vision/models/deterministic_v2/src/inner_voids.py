"""Inner void detection from accepted bbox regions."""

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


def _kernel(size: int) -> np.ndarray | None:
    if int(size) <= 1:
        return None
    return np.ones((int(size), int(size)), dtype=np.uint8)


def _depth(hierarchy: np.ndarray, index: int) -> int:
    depth = 0
    parent = int(hierarchy[index][3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def _ordered_box_points(points: np.ndarray) -> list[list[float]]:
    pts = points.reshape(-1, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.asarray(
        [
            pts[int(np.argmin(sums))],
            pts[int(np.argmax(diffs))],
            pts[int(np.argmax(sums))],
            pts[int(np.argmin(diffs))],
        ],
        dtype=np.float32,
    )
    return [[clean_float(float(x)), clean_float(float(y))] for x, y in ordered.tolist()]


def _point_lists(contour: np.ndarray, offset_x: int, offset_y: int) -> tuple[list[list[float]], list[list[float]]]:
    pts = contour.reshape(-1, 2)
    absolute = [[clean_float(float(x + offset_x)), clean_float(float(y + offset_y))] for x, y in pts]
    local = [[clean_float(float(x)), clean_float(float(y))] for x, y in pts]
    return absolute, local


def _bbox_area(bbox_px: list[int]) -> int:
    x0, y0, x1, y1 = [int(value) for value in bbox_px[:4]]
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def _distance(left: list[float], right: list[float]) -> float:
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _corner_scores(points: list[list[float]], angle_tolerance_deg: float) -> tuple[float, float]:
    if len(points) != 4:
        return 0.0, 0.0
    side_lengths = [_distance(points[index], points[(index + 1) % 4]) for index in range(4)]
    side_balance = min(side_lengths) / max(1e-6, max(side_lengths))
    angle_scores: list[float] = []
    for index, point in enumerate(points):
        prev_point = points[(index - 1) % 4]
        next_point = points[(index + 1) % 4]
        left = np.asarray([prev_point[0] - point[0], prev_point[1] - point[1]], dtype=np.float64)
        right = np.asarray([next_point[0] - point[0], next_point[1] - point[1]], dtype=np.float64)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-9:
            angle_scores.append(0.0)
            continue
        cosine = max(-1.0, min(1.0, float(np.dot(left, right) / denom)))
        angle = math.degrees(math.acos(cosine))
        angle_scores.append(clamp01(1.0 - abs(angle - 90.0) / max(1e-6, angle_tolerance_deg)))
    return clamp01(sum(angle_scores) / max(1, len(angle_scores))), clamp01(side_balance)


def _range_score(value: float, min_value: float, max_value: float) -> float:
    low_score = 1.0 if min_value <= 0.0 or value >= min_value else value / max(1e-6, min_value)
    high_score = 1.0 if max_value <= 0.0 or value <= max_value else max_value / max(1e-6, value)
    return clamp01(min(low_score, high_score))


class InnerVoidDetector:
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.settings = config.section("innerVoids")

    def _source_mask(self, mask_bits: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
        source = str(self.settings.get("maskSource") or "enabled-layers")
        if source == "enabled-layers":
            return selected_mask.astype(np.bool_)
        prefix = source[:-5] if source.endswith("-only") else source
        bit = next((int(layer["bit"]) for layer in self.config.mask_layers if str(layer.get("prefix")) == prefix), None)
        if bit is None:
            return np.zeros(mask_bits.shape, dtype=np.bool_)
        return (mask_bits & np.uint8(1 << bit)) != 0

    def _score(self, candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        if not bool(self.settings.get("scoreEnabled", True)):
            return 0.0, {"enabled": False, "reason": "score-disabled"}
        min_pixels = _float(self.settings, "minVoidPixels", 20.0)
        max_pixels = _float(self.settings, "maxVoidPixels", 0.0)
        min_area = _float(self.settings, "minVoidAreaPx", 20.0)
        max_area = _float(self.settings, "maxVoidAreaPx", 0.0)
        size_score = 0.5 * (
            _range_score(float(candidate["pixel_count"]), min_pixels, max_pixels)
            + _range_score(float(candidate["area_px"]), min_area, max_area)
        )
        rect = candidate.get("rotated_rect_px") or {}
        size = rect.get("sizePx") if isinstance(rect, dict) else None
        width = float(size[0]) if isinstance(size, list) and len(size) >= 2 else 0.0
        height = float(size[1]) if isinstance(size, list) and len(size) >= 2 else 0.0
        aspect = max(width, height) / max(1e-6, min(width, height)) if max(width, height) > 0.0 else 0.0
        square_aspect = clamp01(1.0 - abs(aspect - 1.0) / max(1e-6, _float(self.settings, "scoreAspectTolerance", 0.6)))
        rect_area = max(1e-6, width * height)
        rect_fit = clamp01(float(candidate["area_px"]) / rect_area / max(1e-6, _float(self.settings, "scoreMinRectFillRatio", 0.65)))
        corner_count = int(candidate.get("simplified_point_count") or 0)
        corner_count_score = clamp01(1.0 - abs(corner_count - 4.0) / max(1.0, _float(self.settings, "scoreCornerCountTolerance", 3.0)))
        angle_score, side_balance = _corner_scores(candidate.get("points_px") or [], _float(self.settings, "scoreAngleToleranceDeg", 25.0))
        corner_score = 0.65 * corner_count_score if corner_count != 4 else 0.4 * corner_count_score + 0.3 * angle_score + 0.3 * side_balance
        ideal_perimeter = 4.0 * math.sqrt(max(0.0, float(candidate["area_px"])))
        perimeter_score = clamp01(ideal_perimeter / max(1e-6, float(candidate["perimeter_px"])))
        simplification_score = clamp01(1.0 - max(0.0, corner_count - 4.0) / max(1.0, _float(self.settings, "scoreCornerCountTolerance", 3.0) * 2.0))
        smoothness = 0.75 * perimeter_score + 0.25 * simplification_score
        placement = 0.0 if bool(candidate["quality"].get("bboxEdgeTouching")) and bool(self.settings.get("rejectBboxEdgeTouching", True)) else 1.0
        raw_weights = {
            "size": max(0.0, _float(self.settings, "scoreWeightSize", 0.15)),
            "squareAspect": max(0.0, _float(self.settings, "scoreWeightSquareAspect", 0.20)),
            "rectFit": max(0.0, _float(self.settings, "scoreWeightRectFit", 0.15)),
            "corners": max(0.0, _float(self.settings, "scoreWeightCorners", 0.25)),
            "smoothness": max(0.0, _float(self.settings, "scoreWeightSmoothness", 0.20)),
            "placement": max(0.0, _float(self.settings, "scoreWeightPlacement", 0.05)),
        }
        total = sum(raw_weights.values()) or 1.0
        weights = {key: value / total for key, value in raw_weights.items()}
        score = (
            weights["size"] * size_score
            + weights["squareAspect"] * square_aspect
            + weights["rectFit"] * rect_fit
            + weights["corners"] * corner_score
            + weights["smoothness"] * smoothness
            + weights["placement"] * placement
        )
        return clean_float(score, 6), {
            "sizeScore": clean_float(size_score, 6),
            "squareAspectScore": clean_float(square_aspect, 6),
            "rotatedRectFitScore": clean_float(rect_fit, 6),
            "cornerScore": clean_float(corner_score, 6),
            "smoothnessScore": clean_float(smoothness, 6),
            "placementScore": clean_float(placement, 6),
            "weights": {key: clean_float(value, 6) for key, value in weights.items()},
        }

    def _candidate(
        self,
        contour: np.ndarray,
        contour_index: int,
        hierarchy: np.ndarray,
        offset_x: int,
        offset_y: int,
        roi_mask: np.ndarray,
        bbox: dict[str, Any],
        epsilon: float,
    ) -> dict[str, Any] | None:
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            return None
        simplified = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
        sx, sy, sw, sh = cv2.boundingRect(contour)
        x1_local = int(sx + sw - 1)
        y1_local = int(sy + sh - 1)
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(sx + sw * 0.5)
            cy = float(sy + sh * 0.5)
        points_px, points_local = _point_lists(simplified, offset_x, offset_y)
        raw_points_px, raw_points_local = _point_lists(contour, offset_x, offset_y)
        filled = np.zeros(roi_mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], 0, 1, thickness=-1)
        empty_pixel_count = int(np.count_nonzero((filled != 0) & ~roi_mask))
        filled_pixel_count = int(np.count_nonzero(filled))
        pixel_count = empty_pixel_count if empty_pixel_count > 0 else filled_pixel_count
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_abs = box.copy()
        box_abs[:, 0] += float(offset_x)
        box_abs[:, 1] += float(offset_y)
        roi_height, roi_width = roi_mask.shape
        touches_edge = sx <= 0 or sy <= 0 or x1_local >= roi_width - 1 or y1_local >= roi_height - 1
        bbox_px = list(bbox["bbox_px"])
        candidate = {
            "void_id": f"{bbox['bbox_id']}-void-{contour_index + 1:03d}",
            "bbox_id": bbox["bbox_id"],
            "status": "candidate",
            "reason": None,
            "source_contour_index": int(contour_index),
            "parent_contour_index": int(hierarchy[contour_index][3]),
            "depth": int(_depth(hierarchy, contour_index)),
            "pixel_count": int(pixel_count),
            "area_px": clean_float(area, 3),
            "perimeter_px": clean_float(perimeter, 3),
            "bbox_px": [int(offset_x + sx), int(offset_y + sy), int(offset_x + x1_local), int(offset_y + y1_local)],
            "bbox_local_px": [int(sx), int(sy), int(x1_local), int(y1_local)],
            "center_px": [clean_float(cx + offset_x), clean_float(cy + offset_y)],
            "center_local_px": [clean_float(cx), clean_float(cy)],
            "rotated_rect_px": {
                "centerPx": [clean_float(float(rect[0][0] + offset_x)), clean_float(float(rect[0][1] + offset_y))],
                "centerLocalPx": [clean_float(float(rect[0][0])), clean_float(float(rect[0][1]))],
                "sizePx": [clean_float(float(rect[1][0])), clean_float(float(rect[1][1]))],
                "angleDeg": clean_float(float(rect[2])),
                "pointsPx": _ordered_box_points(box_abs),
                "pointsLocalPx": _ordered_box_points(box),
            },
            "points_px": points_px,
            "points_local_px": points_local,
            "raw_points_px": raw_points_px,
            "raw_points_local_px": raw_points_local,
            "raw_point_count": int(contour.reshape(-1, 2).shape[0]),
            "simplified_point_count": int(simplified.reshape(-1, 2).shape[0]),
            "quality": {
                "bboxEdgeTouching": bool(touches_edge),
                "areaRatioToBbox": clean_float(area / max(1, _bbox_area(bbox_px)), 6),
                "pixelRatioToBbox": clean_float(pixel_count / max(1, _bbox_area(bbox_px)), 6),
                "emptyPixelCount": empty_pixel_count,
                "filledPixelCount": filled_pixel_count,
            },
        }
        return self._classify(candidate)

    def _classify(self, candidate: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        if int(candidate["pixel_count"]) < int(self.settings.get("minVoidPixels", 20)):
            reasons.append("pixel-count-too-low")
        max_pixels = int(self.settings.get("maxVoidPixels", 0))
        if max_pixels > 0 and int(candidate["pixel_count"]) > max_pixels:
            reasons.append("pixel-count-too-high")
        if float(candidate["area_px"]) < float(self.settings.get("minVoidAreaPx", 20)):
            reasons.append("area-too-low")
        max_area = float(self.settings.get("maxVoidAreaPx", 0))
        if max_area > 0 and float(candidate["area_px"]) > max_area:
            reasons.append("area-too-high")
        if bool(self.settings.get("rejectBboxEdgeTouching", True)) and bool(candidate["quality"].get("bboxEdgeTouching")):
            reasons.append("touches-bbox-edge")
        score, breakdown = self._score(candidate)
        candidate["status"] = "rejected" if reasons else "accepted"
        candidate["reason"] = reasons[0] if reasons else None
        candidate["score"] = score
        candidate["score_breakdown"] = breakdown
        candidate["quality"]["reasons"] = reasons
        candidate["quality"]["score"] = score
        candidate["quality"]["scoreBreakdown"] = breakdown
        return candidate

    def process(
        self,
        meta: FrameMeta,
        bbox_payload: dict[str, Any],
        labels: np.ndarray,
        selected_mask: np.ndarray,
        mask_bits: np.ndarray,
    ) -> dict[str, Any]:
        source_mask = self._source_mask(mask_bits, selected_mask)
        observations: list[dict[str, Any]] = []
        max_voids = int(self.settings.get("maxVoidsPerBbox", 12))
        epsilon = float(self.settings.get("simplifyEpsilonPx", 1.5))
        close_kernel = _kernel(int(self.settings.get("closeKernelPx", 0)))
        open_kernel = _kernel(int(self.settings.get("openKernelPx", 0)))
        include_rejected = bool(self.settings.get("includeRejected", True))
        total_accepted = 0
        total_rejected = 0
        for bbox in bbox_payload.get("observations", []):
            x0, y0, x1, y1 = [int(value) for value in bbox["bbox_px"]]
            component_mask = labels == int(bbox["component_label"])
            roi = (component_mask & source_mask)[y0 : y1 + 1, x0 : x1 + 1]
            working = roi.astype(np.uint8) * 255
            if close_kernel is not None:
                working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, close_kernel)
            if open_kernel is not None:
                working = cv2.morphologyEx(working, cv2.MORPH_OPEN, open_kernel)
            roi_bool = working > 0
            contours, hierarchy = cv2.findContours(working, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            candidates: list[dict[str, Any]] = []
            if hierarchy is not None:
                rows = hierarchy[0]
                for index, contour in enumerate(contours):
                    if _depth(rows, index) % 2 == 0:
                        continue
                    candidate = self._candidate(contour, index, rows, x0, y0, roi_bool, bbox, epsilon)
                    if candidate is not None:
                        candidates.append(candidate)
            accepted = sorted(
                [item for item in candidates if item["status"] == "accepted"],
                key=lambda item: (-float(item["area_px"]), item["bbox_px"][1], item["bbox_px"][0]),
            )
            rejected = sorted(
                [item for item in candidates if item["status"] != "accepted"],
                key=lambda item: (-float(item["area_px"]), item["bbox_px"][1], item["bbox_px"][0]),
            )
            if max_voids > 0 and len(accepted) > max_voids:
                overflow = accepted[max_voids:]
                accepted = accepted[:max_voids]
                for item in overflow:
                    item["status"] = "rejected"
                    item["reason"] = "max-voids-per-bbox"
                    item["quality"]["reasons"] = ["max-voids-per-bbox"]
                rejected.extend(overflow)
            total_accepted += len(accepted)
            total_rejected += len(rejected)
            best = max(accepted, key=lambda item: float(item["score"]), default=None)
            observations.append(
                {
                    "bbox_id": bbox["bbox_id"],
                    "bbox_px": list(bbox["bbox_px"]),
                    "selected_pixel_count": int(np.count_nonzero(roi_bool)),
                    "contour_count": len(contours),
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "voids": accepted,
                    "rejected_voids": rejected if include_rejected else [],
                    "metrics": {
                        "candidateCount": len(candidates),
                        "acceptedCount": len(accepted),
                        "rejectedCount": len(rejected),
                        "bestVoidId": best["void_id"] if best else None,
                        "bestScore": clean_float(float(best["score"]), 6) if best else 0.0,
                    },
                }
            )
        return {
            "schema": "projection-inner-voids.v1",
            "run_id": meta.run_id,
            "frame_ordinal": meta.frame_ordinal,
            "frame_id": meta.frame_id,
            "source_path": meta.source_path,
            "image_width": int(mask_bits.shape[1]),
            "image_height": int(mask_bits.shape[0]),
            "created_at": utc_now(),
            "observations": observations,
            "accepted_count": total_accepted,
            "rejected_count": total_rejected,
            "summary": {
                "bboxCount": len(observations),
                "acceptedVoidCount": total_accepted,
                "rejectedVoidCount": total_rejected,
                "bboxWithVoidCount": sum(1 for item in observations if int(item["accepted_count"]) > 0),
            },
            "settings": dict(self.settings),
        }
