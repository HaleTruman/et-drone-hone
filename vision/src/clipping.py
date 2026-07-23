"""Frame-edge clipping diagnostics stage."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised by script-mode imports
    from .schema import BBoxFrame, ClippingFrame, ClippingObservation, ColorMaskFrame, PipelinePreset, utc_now
except ImportError:  # pragma: no cover
    from schema import BBoxFrame, ClippingFrame, ClippingObservation, ColorMaskFrame, PipelinePreset, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


class ClippingAnalyzer:
    def __init__(self, preset: PipelinePreset):
        self.settings = dict(preset.clipping)

    def process(self, bbox_frame: BBoxFrame, mask_frame: ColorMaskFrame) -> ClippingFrame:
        enabled = bool(self.settings.get("enabled", False))
        margin = int(self.settings.get("marginPx", 6))
        min_contact_pixels = int(self.settings.get("minContactPixels", 12))
        min_contact_ratio = float(self.settings.get("minContactRatio", 0.01))
        require_touch = bool(self.settings.get("requireBboxTouch", True))
        warn_only = bool(self.settings.get("warnOnly", True))
        width = int(bbox_frame.image_width)
        height = int(bbox_frame.image_height)
        labels = bbox_frame.component_labels

        observations: list[ClippingObservation] = []
        summary = {
            "enabled": enabled,
            "clippedBboxCount": 0,
            "nearEdgeBboxCount": 0,
            "sideCounts": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "maxSeverity": 0.0,
            "meanSeverity": 0.0,
        }
        severities: list[float] = []
        for bbox in bbox_frame.observations:
            x0, y0, x1, y1 = [int(value) for value in bbox.bbox_px]
            if labels is not None:
                ys, xs = np.nonzero(labels == int(bbox.component_label))
            else:
                yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
                ys, xs = yy.reshape(-1), xx.reshape(-1)
            pixel_count = max(1, int(xs.size))
            bbox_touches = {
                "left": x0 <= margin,
                "right": x1 >= width - 1 - margin,
                "top": y0 <= margin,
                "bottom": y1 >= height - 1 - margin,
            }
            contact_masks = {
                "left": xs <= margin,
                "right": xs >= width - 1 - margin,
                "top": ys <= margin,
                "bottom": ys >= height - 1 - margin,
            }
            contact_pixels = {side: int(np.count_nonzero(mask)) for side, mask in contact_masks.items()}
            contact_ratio = {
                side: clean_float(float(contact_pixels[side]) / float(pixel_count), 6)
                for side in contact_pixels
            }
            clipped_sides: list[str] = []
            near_sides: list[str] = []
            side_severity: dict[str, float] = {}
            side_trust: dict[str, float] = {}
            for side in ("left", "right", "top", "bottom"):
                pixel_score = min(1.0, float(contact_pixels[side]) / max(1, min_contact_pixels))
                ratio_score = 1.0 if min_contact_ratio <= 0 else min(1.0, float(contact_ratio[side]) / min_contact_ratio)
                touch_score = 1.0 if bbox_touches[side] else 0.0
                active = bbox_touches[side] or contact_pixels[side] > 0
                severity = (0.45 * touch_score + 0.35 * pixel_score + 0.20 * ratio_score) if active else 0.0
                side_severity[side] = clean_float(severity, 4)
                side_trust[side] = clean_float(1.0 - severity, 4)
                touch_gate = bbox_touches[side] or not require_touch
                is_clipped = enabled and touch_gate and contact_pixels[side] >= min_contact_pixels and float(contact_ratio[side]) >= min_contact_ratio
                if is_clipped:
                    clipped_sides.append(side)
                elif enabled and active:
                    near_sides.append(side)
            status = "disabled"
            if enabled:
                status = "clipped" if clipped_sides else "near-edge" if near_sides else "clear"
            severity_value = max(side_severity.values()) if enabled and side_severity else 0.0
            if status == "clipped":
                summary["clippedBboxCount"] += 1
            if status == "near-edge":
                summary["nearEdgeBboxCount"] += 1
            for side in clipped_sides:
                summary["sideCounts"][side] += 1
            severities.append(severity_value)
            observations.append(
                ClippingObservation(
                    bbox_id=bbox.bbox_id,
                    enabled=enabled,
                    status=status,
                    severity=clean_float(severity_value, 4),
                    sides=clipped_sides,
                    near_sides=near_sides,
                    bbox_touches=bbox_touches,
                    contact_pixels=contact_pixels,
                    contact_ratio=contact_ratio,
                    side_severity=side_severity,
                    side_trust=side_trust,
                    warn_only=warn_only,
                )
            )
        summary["maxSeverity"] = clean_float(max(severities) if severities else 0.0, 4)
        summary["meanSeverity"] = clean_float(sum(severities) / len(severities) if severities else 0.0, 4)
        return ClippingFrame(
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

