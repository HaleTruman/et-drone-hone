from __future__ import annotations

import cv2
import numpy as np

from schema import DEFAULT_MIN_MASK_REGION_PX, VoidAnalysis

CLIPPED_VOID_MIN_PX = 1
VOID_EDGE_MARGIN_PX = 3  # a void's own bbox rarely reaches the literal edge before dropping out of detection


def _touches_frame(stats_row: np.ndarray, width: int, height: int, margin: int = 0) -> bool:
    x, y, w, h = [int(value) for value in stats_row[:4]]
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def annotate_mask_clipping(
    analysis: VoidAnalysis,
    min_mask_region_px: int = DEFAULT_MIN_MASK_REGION_PX,
) -> VoidAnalysis:
    final = analysis.bridge.final_mask.astype(bool)
    height, width = final.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(final.astype(np.uint8), 8)
    region_index = 0
    for label in range(1, count):
        clipped = _touches_frame(stats[label], width, height)
        if int(stats[label, cv2.CC_STAT_AREA]) < int(min_mask_region_px):
            continue
        if region_index >= len(analysis.regions):
            break
        region = analysis.regions[region_index]
        region.mask_clipping = clipped
        for void in region.voids:
            void.mask_clipping = _touches_frame(void.bbox, width, height, margin=VOID_EDGE_MARGIN_PX)
        region_index += 1
    return analysis
