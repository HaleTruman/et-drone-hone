from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from mask import frame_to_mask
from mask_fill import fill_close_pixels
from schema import DEFAULT_BRIDGE_MAX_PX, DEFAULT_MIN_MASK_REGION_PX, BridgeFrame, FillFrame, mask_count

POST_HULL_CLOSE_PX = 3
MAX_HULL_BBOX_FRACTION = 0.9  # skip the hull for components spanning nearly the whole frame (background noise)


def _hull_outline(component: np.ndarray) -> np.ndarray:
    source = component.astype(np.uint8)
    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(component, dtype=bool)
    points = np.vstack(contours).reshape(-1, 2).astype(np.int32)
    if len(points) < 3:
        return np.zeros_like(component, dtype=bool)
    hull = cv2.convexHull(points)
    outline = np.zeros_like(source, dtype=np.uint8)
    cv2.polylines(outline, [hull.reshape(-1, 2)], True, 1, 1)
    return outline.astype(bool)


def _post_hull_close(base: np.ndarray, green: np.ndarray) -> np.ndarray:
    seed = base | green
    kernel = np.ones((POST_HULL_CLOSE_PX, POST_HULL_CLOSE_PX), dtype=np.uint8)
    closed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return closed & ~base


def bridge_outer_hull(
    fill: FillFrame,
    bridge_max_px: int = DEFAULT_BRIDGE_MAX_PX,
    min_mask_region_px: int = DEFAULT_MIN_MASK_REGION_PX,
) -> BridgeFrame:
    base = fill.filled_mask.astype(bool)
    original = fill.frame.base_mask.astype(bool)
    height, width = original.shape
    green = np.zeros_like(base, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(original.astype(np.uint8), 8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= int(min_mask_region_px):
            continue
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        if w >= width * MAX_HULL_BBOX_FRACTION and h >= height * MAX_HULL_BBOX_FRACTION:
            continue
        green |= _hull_outline(labels == label)
    green |= _post_hull_close(base, green)
    return BridgeFrame(
        fill=fill,
        green_mask=green,
        final_mask=base | green,
        bridge_max_px=int(bridge_max_px),
        min_mask_region_px=int(min_mask_region_px),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview original-component convex hull outlines.")
    parser.add_argument("frame")
    parser.add_argument("--lut", default="assets/color_lut_v1.npz")
    parser.add_argument("--fill-radius", type=int, default=3)
    parser.add_argument("--bridge-max", type=int, default=DEFAULT_BRIDGE_MAX_PX)
    args = parser.parse_args()
    fill = fill_close_pixels(frame_to_mask(args.frame, args.lut), args.fill_radius)
    result = bridge_outer_hull(fill, args.bridge_max)
    print(json.dumps({
        "frame_path": result.fill.frame.frame_path,
        "green_pixels": mask_count(result.green_mask),
        "final_pixels": mask_count(result.final_mask),
    }, indent=2))


if __name__ == "__main__":
    main()
