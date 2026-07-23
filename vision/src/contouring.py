"""BBox-local contour extraction stage."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:  # pragma: no cover - exercised by script-mode imports
    from .schema import BBoxFrame, ColorMaskFrame, ContourFrame, ContourObservation, PipelinePreset, utc_now
except ImportError:  # pragma: no cover
    from schema import BBoxFrame, ColorMaskFrame, ContourFrame, ContourObservation, PipelinePreset, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def _depth(hierarchy: np.ndarray, index: int) -> int:
    depth = 0
    parent = int(hierarchy[index][3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def _point_lists(contour: np.ndarray, offset_x: int, offset_y: int) -> tuple[list[list[float]], list[list[float]]]:
    pts = contour.reshape(-1, 2)
    absolute = [[clean_float(float(x + offset_x)), clean_float(float(y + offset_y))] for x, y in pts]
    local = [[clean_float(float(x)), clean_float(float(y))] for x, y in pts]
    return absolute, local


def _feature(
    contour: np.ndarray,
    contour_index: int,
    hierarchy: np.ndarray,
    role: str,
    offset_x: int,
    offset_y: int,
    bbox_id: str,
    epsilon: float,
) -> dict[str, Any] | None:
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return None
    simplified = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
    sx, sy, sw, sh = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(sx + sw * 0.5)
        cy = float(sy + sh * 0.5)
    points_px, points_local = _point_lists(simplified, offset_x, offset_y)
    raw_points_px, raw_points_local = _point_lists(contour, offset_x, offset_y)
    parent = int(hierarchy[contour_index][3])
    return {
        "contourId": f"{bbox_id}-contour-{contour_index + 1:03d}",
        "bboxId": bbox_id,
        "role": role,
        "depth": int(_depth(hierarchy, contour_index)),
        "parentContourIndex": parent,
        "areaPx": clean_float(area, 3),
        "perimeterPx": clean_float(perimeter, 3),
        "bboxPx": [int(offset_x + sx), int(offset_y + sy), int(offset_x + sx + sw - 1), int(offset_y + sy + sh - 1)],
        "bboxLocalPx": [int(sx), int(sy), int(sx + sw - 1), int(sy + sh - 1)],
        "centerPx": [clean_float(cx + offset_x), clean_float(cy + offset_y)],
        "centerLocalPx": [clean_float(cx), clean_float(cy)],
        "rawPointCount": int(contour.reshape(-1, 2).shape[0]),
        "simplifiedPointCount": int(simplified.reshape(-1, 2).shape[0]),
        "pointsPx": points_px,
        "pointsLocalPx": points_local,
        "rawPointsPx": raw_points_px,
        "rawPointsLocalPx": raw_points_local,
    }


def _quad_from_outer(outer: dict[str, Any] | None) -> list[list[float]] | None:
    if not outer:
        return None
    points = outer.get("rawPointsPx") or outer.get("pointsPx")
    if not isinstance(points, list) or len(points) < 4:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect)
    sums = box[:, 0] + box[:, 1]
    diffs = box[:, 0] - box[:, 1]
    ordered = np.array(
        [box[int(np.argmin(sums))], box[int(np.argmax(diffs))], box[int(np.argmax(sums))], box[int(np.argmin(diffs))]],
        dtype=np.float32,
    )
    return [[clean_float(float(x)), clean_float(float(y))] for x, y in ordered.tolist()]


class Contourer:
    def __init__(self, preset: PipelinePreset):
        self.preset = preset
        self.settings = dict(preset.contours)

    def _source_mask(self, mask_frame: ColorMaskFrame, bbox_frame: BBoxFrame) -> np.ndarray:
        mask_bits = np.asarray(mask_frame.mask_bits, dtype=np.uint8)
        if self.settings.get("maskSource") == "002-only":
            bit = next((int(layer["bit"]) for layer in self.preset.maskLayers if str(layer["prefix"]) == "002"), None)
            if bit is None:
                return np.zeros(mask_bits.shape, dtype=np.bool_)
            return (mask_bits & np.uint8(1 << bit)) != 0
        if bbox_frame.selected_mask is not None:
            return np.asarray(bbox_frame.selected_mask, dtype=np.bool_)
        selected = np.zeros(mask_bits.shape, dtype=np.bool_)
        for bit in self.preset.enabled_bits():
            selected |= (mask_bits & np.uint8(1 << bit)) != 0
        return selected

    def process(self, bbox_frame: BBoxFrame, mask_frame: ColorMaskFrame) -> ContourFrame:
        source_mask = self._source_mask(mask_frame, bbox_frame)
        labels = bbox_frame.component_labels
        observations: list[ContourObservation] = []
        min_outer = float(self.settings.get("minOuterAreaPx", 10))
        min_void = float(self.settings.get("minVoidAreaPx", 20))
        max_voids = int(self.settings.get("maxVoidsPerBbox", 12))
        epsilon = float(self.settings.get("simplifyEpsilonPx", 1.5))
        for bbox in bbox_frame.observations:
            x0, y0, x1, y1 = [int(value) for value in bbox.bbox_px]
            if labels is not None:
                component_mask = labels == int(bbox.component_label)
            else:
                component_mask = np.zeros(source_mask.shape, dtype=np.bool_)
                component_mask[y0 : y1 + 1, x0 : x1 + 1] = True
            analysis = component_mask & source_mask
            roi = analysis[y0 : y1 + 1, x0 : x1 + 1]
            selected_pixel_count = int(np.count_nonzero(roi))
            if selected_pixel_count <= 0:
                observations.append(
                    ContourObservation(
                        bbox_id=bbox.bbox_id,
                        outer=None,
                        selected_pixel_count=0,
                        contour_count=0,
                        metrics={"outerCount": 0, "voidCount": 0},
                        quad_points_px=bbox.quad_fit.get("pointsPx") if isinstance(bbox.quad_fit, dict) else None,
                        bbox_px=list(bbox.bbox_px),
                    )
                )
                continue
            contours, hierarchy = cv2.findContours((roi.astype(np.uint8) * 255), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            features: list[dict[str, Any]] = []
            if hierarchy is not None:
                rows = hierarchy[0]
                for index, contour in enumerate(contours):
                    depth = _depth(rows, index)
                    role = "outer" if depth % 2 == 0 else "void"
                    item = _feature(contour, index, rows, role, x0, y0, bbox.bbox_id, epsilon)
                    if not item:
                        continue
                    area = float(item["areaPx"])
                    if role == "outer" and area >= min_outer:
                        features.append(item)
                    elif role == "void" and area >= min_void:
                        features.append(item)
            outers = sorted(
                [item for item in features if item["role"] == "outer"],
                key=lambda item: (-float(item["areaPx"]), item["bboxPx"][1], item["bboxPx"][0]),
            )
            voids = sorted(
                [item for item in features if item["role"] == "void"],
                key=lambda item: (-float(item["areaPx"]), item["bboxPx"][1], item["bboxPx"][0]),
            )[:max_voids]
            outer = outers[0] if outers else None
            quad_points = bbox.quad_fit.get("pointsPx") if isinstance(bbox.quad_fit, dict) else None
            if not quad_points:
                quad_points = _quad_from_outer(outer)
            observations.append(
                ContourObservation(
                    bbox_id=bbox.bbox_id,
                    outer=outer,
                    voids=voids,
                    additional_outers=outers[1:],
                    selected_pixel_count=selected_pixel_count,
                    contour_count=len(features),
                    metrics={
                        "outerCount": len(outers),
                        "voidCount": len(voids),
                        "additionalOuterCount": max(0, len(outers) - 1),
                    },
                    quad_points_px=quad_points,
                    bbox_px=list(bbox.bbox_px),
                )
            )
        summary = {
            "bboxCount": len(observations),
            "contourCount": sum(item.contour_count for item in observations),
            "outerCount": sum(1 for item in observations if item.outer),
            "voidCount": sum(len(item.voids) for item in observations),
        }
        return ContourFrame(
            run_id=bbox_frame.run_id,
            frame_ordinal=bbox_frame.frame_ordinal,
            frame_id=bbox_frame.frame_id,
            source_path=bbox_frame.source_path,
            image_width=bbox_frame.image_width,
            image_height=bbox_frame.image_height,
            created_at=utc_now(),
            timing_ms={},
            observations=observations,
            summary=summary,
            settings=dict(self.settings),
        )

