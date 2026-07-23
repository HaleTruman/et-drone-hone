"""Connected-component bbox stage."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # pragma: no cover - exercised by script-mode imports
    from .schema import BBoxFrame, BBoxObservation, ColorMaskFrame, PipelinePreset, utc_now
except ImportError:  # pragma: no cover
    from schema import BBoxFrame, BBoxObservation, ColorMaskFrame, PipelinePreset, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def _bbox_uv(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> list[float]:
    return [
        clean_float(x0 / max(1, width), 6),
        clean_float(y0 / max(1, height), 6),
        clean_float((x1 + 1) / max(1, width), 6),
        clean_float((y1 + 1) / max(1, height), 6),
    ]


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("quad ordering requires four points")
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[int(np.argmin(sums))],
            pts[int(np.argmax(diffs))],
            pts[int(np.argmax(sums))],
            pts[int(np.argmin(diffs))],
        ],
        dtype=np.float32,
    )


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    x_values = pts[:, 0]
    y_values = pts[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def _axis_quad(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.array(
        [[float(x0), float(y0)], [float(x1 + 1), float(y0)], [float(x1 + 1), float(y1 + 1)], [float(x0), float(y1 + 1)]],
        dtype=np.float32,
    )


def _quad_fit(points_xy: np.ndarray, bbox_px: list[int], mode: str) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox_px
    fallback = None
    if mode == "axis-bbox" or points_xy.shape[0] < 4:
        quad = _axis_quad(x0, y0, x1, y1)
        if mode != "axis-bbox":
            fallback = "too-few-points-axis"
    else:
        rect = cv2.minAreaRect(points_xy.astype(np.float32).reshape(-1, 1, 2))
        quad = cv2.boxPoints(rect)
        if _polygon_area(quad) <= 0.1:
            quad = _axis_quad(x0, y0, x1, y1)
            fallback = "degenerate-rotated-rect-axis"
    quad = _order_quad(quad)
    side_lengths = [
        float(np.linalg.norm(quad[(index + 1) % 4] - quad[index]))
        for index in range(4)
    ]
    width_px = (side_lengths[0] + side_lengths[2]) / 2.0
    height_px = (side_lengths[1] + side_lengths[3]) / 2.0
    aspect = width_px / max(height_px, 1e-6)
    return {
        "enabled": True,
        "mode": mode,
        "fallbackReason": fallback,
        "pointsPx": [[clean_float(x), clean_float(y)] for x, y in quad.tolist()],
        "areaPx": clean_float(_polygon_area(quad), 3),
        "widthPx": clean_float(width_px, 3),
        "heightPx": clean_float(height_px, 3),
        "aspect": clean_float(aspect, 6),
    }


class BBoxer:
    def __init__(self, preset: PipelinePreset):
        self.preset = preset
        self.settings = dict(preset.bbox)
        self.enabled_bits = preset.enabled_bits()
        self.layers = list(preset.maskLayers)

    def _selected_mask(self, mask_bits: np.ndarray) -> np.ndarray:
        selected = np.zeros(mask_bits.shape, dtype=np.bool_)
        for bit in self.enabled_bits:
            selected |= (mask_bits & np.uint8(1 << int(bit))) != 0
        return selected

    def process(self, mask_frame: ColorMaskFrame) -> BBoxFrame:
        if mask_frame.mask_bits is None:
            raise ValueError("BBoxer requires ColorMaskFrame.mask_bits")
        mask_bits = np.asarray(mask_frame.mask_bits, dtype=np.uint8)
        selected = self._selected_mask(mask_bits)
        connectivity = int(self.settings.get("connectivity", 8))
        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            selected.astype(np.uint8),
            connectivity=connectivity,
        )
        min_pixels = int(self.settings.get("minPixels", 50))
        max_bboxes = int(self.settings.get("maxBboxesPerFrame", 1000))
        candidates: list[tuple[int, int]] = []
        for label in range(1, int(label_count)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_pixels:
                candidates.append((label, area))
        candidates.sort(key=lambda item: item[1], reverse=True)

        observations: list[BBoxObservation] = []
        width = int(mask_frame.image_width)
        height = int(mask_frame.image_height)
        target_aspect = float(self.settings.get("targetAspect", 1.0))
        aspect_tolerance = float(self.settings.get("aspectTolerance", 0.6))
        for index, (label, pixel_count) in enumerate(candidates[:max_bboxes]):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            x0, y0, x1, y1 = x, y, x + w - 1, y + h - 1
            component_mask = labels == label
            ys, xs = np.nonzero(component_mask)
            layer_counts: dict[str, int] = {}
            for layer in self.layers:
                prefix = str(layer["prefix"])
                bit = int(layer["bit"])
                count = int(np.count_nonzero(mask_bits[component_mask] & np.uint8(1 << bit)))
                if count:
                    layer_counts[prefix] = count
            dominant_prefix = max(layer_counts.items(), key=lambda item: item[1])[0] if layer_counts else None
            bbox_area = int(w * h)
            fill_ratio = pixel_count / max(1, bbox_area)
            aspect = float(w) / max(1.0, float(h))
            aspect_low = target_aspect * max(0.0, 1.0 - aspect_tolerance)
            aspect_high = target_aspect * (1.0 + aspect_tolerance)
            points_xy = np.column_stack([xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5])
            if bool(self.settings.get("quadFitEnabled", True)):
                quad_fit = _quad_fit(points_xy, [x0, y0, x1, y1], str(self.settings.get("quadFitMode", "rotated-rect")))
            else:
                quad_fit = {"enabled": False}
            cx, cy = centroids[label]
            observations.append(
                BBoxObservation(
                    bbox_id=f"{mask_frame.frame_id}-bbox-{index + 1:03d}",
                    component_label=int(label),
                    bbox_px=[x0, y0, x1, y1],
                    bbox_uv=_bbox_uv(x0, y0, x1, y1, width, height),
                    center_px=[clean_float(float(cx)), clean_float(float(cy))],
                    center_uv=[clean_float(float(cx) / max(1, width), 6), clean_float(float(cy) / max(1, height), 6)],
                    width_px=w,
                    height_px=h,
                    area_px=bbox_area,
                    pixel_count=int(pixel_count),
                    layer_counts=layer_counts,
                    dominant_prefix=dominant_prefix,
                    quality={
                        "fillRatio": clean_float(fill_ratio, 6),
                        "aspect": clean_float(aspect, 6),
                        "aspectInRange": bool(aspect_low <= aspect <= aspect_high),
                        "extentTouchesFrame": bool(x0 == 0 or y0 == 0 or x1 >= width - 1 or y1 >= height - 1),
                    },
                    quad_fit=quad_fit,
                )
            )
        return BBoxFrame(
            run_id=mask_frame.run_id,
            frame_ordinal=mask_frame.frame_ordinal,
            frame_id=mask_frame.frame_id,
            source_path=mask_frame.source_path,
            image_width=width,
            image_height=height,
            created_at=utc_now(),
            timing_ms={},
            observations=observations,
            bbox_count=len(observations),
            selected_pixel_count=int(np.count_nonzero(selected)),
            enabled_bits=[int(bit) for bit in self.enabled_bits],
            settings=dict(self.settings),
            component_labels=labels,
            selected_mask=selected,
        )

