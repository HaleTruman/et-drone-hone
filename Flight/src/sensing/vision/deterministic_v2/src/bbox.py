"""Connected-component bboxing and clipping diagnostics."""

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


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    return np.asarray(
        [
            pts[int(np.argmin(sums))],
            pts[int(np.argmax(diffs))],
            pts[int(np.argmax(sums))],
            pts[int(np.argmin(diffs))],
        ],
        dtype=np.float32,
    )


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x_values = pts[:, 0]
    y_values = pts[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def bbox_iou(left: list[int] | None, right: list[int] | None) -> float:
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [int(value) for value in left[:4]]
    bx0, by0, bx1, by1 = [int(value) for value in right[:4]]
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


def _axis_quad(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.asarray(
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
        if polygon_area(quad) <= 0.1:
            quad = _axis_quad(x0, y0, x1, y1)
            fallback = "degenerate-rotated-rect-axis"
    quad = order_quad(quad)
    side_lengths = [float(np.linalg.norm(quad[(index + 1) % 4] - quad[index])) for index in range(4)]
    width_px = (side_lengths[0] + side_lengths[2]) / 2.0
    height_px = (side_lengths[1] + side_lengths[3]) / 2.0
    return {
        "enabled": True,
        "mode": mode,
        "fallbackReason": fallback,
        "pointsPx": [[clean_float(x), clean_float(y)] for x, y in quad.tolist()],
        "areaPx": clean_float(polygon_area(quad), 3),
        "widthPx": clean_float(width_px, 3),
        "heightPx": clean_float(height_px, 3),
        "aspect": clean_float(width_px / max(height_px, 1e-6), 6),
    }


class BBoxer:
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.settings = config.section("bbox")
        self.clipping_settings = config.section("clipping")
        self.enabled_bits = config.enabled_bits
        self.layers = config.mask_layers

    def selected_mask(self, mask_bits: np.ndarray) -> np.ndarray:
        selected = np.zeros(mask_bits.shape, dtype=np.bool_)
        for bit in self.enabled_bits:
            selected |= (mask_bits & np.uint8(1 << int(bit))) != 0
        return selected

    def _clipping(self, bbox_px: list[int], width: int, height: int) -> dict[str, Any]:
        settings = self.clipping_settings
        margin = int(settings.get("marginPx", 6))
        x0, y0, x1, y1 = bbox_px
        contacts = {
            "left": x0 <= margin,
            "top": y0 <= margin,
            "right": x1 >= width - 1 - margin,
            "bottom": y1 >= height - 1 - margin,
        }
        return {
            "enabled": bool(settings.get("enabled", True)),
            "marginPx": margin,
            "touchesFrame": bool(any(contacts.values())),
            "contacts": contacts,
            "warnOnly": bool(settings.get("warnOnly", True)),
        }

    def process(self, meta: FrameMeta, mask_bits: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        mask_bits = np.asarray(mask_bits, dtype=np.uint8)
        selected = self.selected_mask(mask_bits)
        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            selected.astype(np.uint8),
            connectivity=int(self.settings.get("connectivity", 8)),
        )
        min_pixels = int(self.settings.get("minPixels", 250))
        max_bboxes = int(self.settings.get("maxBboxesPerFrame", 4))
        candidates = [
            (label, int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, int(label_count))
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        width, height = int(mask_bits.shape[1]), int(mask_bits.shape[0])
        observations: list[dict[str, Any]] = []
        target_aspect = float(self.settings.get("targetAspect", 1.0))
        aspect_tolerance = float(self.settings.get("aspectTolerance", 0.6))
        aspect_low = target_aspect * max(0.0, 1.0 - aspect_tolerance)
        aspect_high = target_aspect * (1.0 + aspect_tolerance)
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
            bbox_area = int(w * h)
            aspect = float(w) / max(1.0, float(h))
            points_xy = np.column_stack([xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5])
            quad_fit = (
                _quad_fit(points_xy, [x0, y0, x1, y1], str(self.settings.get("quadFitMode", "rotated-rect")))
                if bool(self.settings.get("quadFitEnabled", True))
                else {"enabled": False}
            )
            cx, cy = centroids[label]
            observations.append(
                {
                    "bbox_id": f"{meta.frame_id}-bbox-{index + 1:03d}",
                    "component_label": int(label),
                    "bbox_px": [x0, y0, x1, y1],
                    "center_px": [clean_float(float(cx)), clean_float(float(cy))],
                    "width_px": w,
                    "height_px": h,
                    "area_px": bbox_area,
                    "pixel_count": int(pixel_count),
                    "layer_counts": layer_counts,
                    "dominant_prefix": max(layer_counts.items(), key=lambda item: item[1])[0] if layer_counts else None,
                    "quality": {
                        "fillRatio": clean_float(pixel_count / max(1, bbox_area), 6),
                        "aspect": clean_float(aspect, 6),
                        "aspectInRange": bool(aspect_low <= aspect <= aspect_high),
                        "extentTouchesFrame": bool(x0 == 0 or y0 == 0 or x1 >= width - 1 or y1 >= height - 1),
                    },
                    "quad_fit": quad_fit,
                    "clipping": self._clipping([x0, y0, x1, y1], width, height),
                }
            )
        return (
            {
                "schema": "projection-bbox.v1",
                "run_id": meta.run_id,
                "frame_ordinal": meta.frame_ordinal,
                "frame_id": meta.frame_id,
                "source_path": meta.source_path,
                "image_width": width,
                "image_height": height,
                "created_at": utc_now(),
                "observations": observations,
                "bbox_count": len(observations),
                "selected_pixel_count": int(np.count_nonzero(selected)),
                "enabled_bits": list(self.enabled_bits),
                "settings": dict(self.settings),
                "clipping_settings": dict(self.clipping_settings),
            },
            labels,
            selected,
        )
