from __future__ import annotations

import argparse
import json
import math

import cv2
import numpy as np

from mask import frame_to_mask
from mask_bridge import bridge_outer_hull
from mask_fill import fill_close_pixels
from schema import (DEFAULT_BRIDGE_MAX_PX, DEFAULT_FILL_RADIUS_PX, DEFAULT_LUT_PATH,
                    DEFAULT_MIN_MASK_REGION_PX, DEFAULT_MIN_ORIGINAL_COVERAGE_DEG,
                    DEFAULT_MIN_VOID_DISTANCE_PX, DEFAULT_MIN_VOID_PX,
                    BridgeFrame, RegionRecord, VoidAnalysis, VoidRecord,
                    as_bool_mask, bbox_from_mask, mask_count)

ORIGINAL_HALO_PX = 5


# ---------------------------------------------------------------------------
# Contour hierarchy: label each contour's depth so outer (region) contours
# and inner (void/hole) contours can be told apart (odd depth = a hole).
# ---------------------------------------------------------------------------

def _contour_depths(hierarchy: np.ndarray) -> list[int]:
    depths = [-1] * len(hierarchy)

    def depth_of(index: int) -> int:
        if depths[index] != -1:
            return depths[index]
        parent = int(hierarchy[index][3])
        depths[index] = 0 if parent == -1 else depth_of(parent) + 1
        return depths[index]

    return [depth_of(i) for i in range(len(hierarchy))]


def _void_pixel_mask(contours, hierarchy: np.ndarray, index: int, component: np.ndarray) -> np.ndarray:
    filled = np.zeros(component.shape, dtype=np.uint8)
    cv2.drawContours(filled, contours, index, 1, -1)
    for child in range(len(hierarchy)):
        if int(hierarchy[child][3]) == index:
            cv2.drawContours(filled, contours, child, 0, -1)
    return filled.astype(bool) & ~component


# ---------------------------------------------------------------------------
# Validation A -- original-mask perimeter coverage: march 360 one-degree rays
# outward from a void's center until each exits the void, then look a small
# halo further for a true original-mask pixel. A void genuinely enclosed by
# the object should be surrounded by real mask almost all the way around; a
# void mostly "enclosed" only because the convex hull bridged over open
# space is not (run 37Z frame 200/226-241: confirmed-bad voids covered
# ~217-234 of 360 degrees, the confirmed-good void covered 360/360).
# ---------------------------------------------------------------------------

def _original_coverage_degrees(center: tuple[int, int], void_mask: np.ndarray, original_mask: np.ndarray,
                                halo_px: int = ORIGINAL_HALO_PX) -> int:
    height, width = void_mask.shape
    cx, cy = center
    hits = 0
    for degree in range(360):
        angle = math.radians(degree)
        dx, dy = math.cos(angle), math.sin(angle)
        radius = 0.0
        x, y = cx, cy
        while 0 <= int(round(x)) < width and 0 <= int(round(y)) < height and void_mask[int(round(y)), int(round(x))]:
            radius += 1.0
            x, y = cx + dx * radius, cy + dy * radius
        for extra in range(halo_px + 1):
            xi, yi = int(round(cx + dx * (radius + extra))), int(round(cy + dy * (radius + extra)))
            if 0 <= xi < width and 0 <= yi < height and original_mask[yi, xi]:
                hits += 1
                break
    return hits


def _has_original_support(coverage_degrees: int, min_coverage_deg: float) -> bool:
    return coverage_degrees >= float(min_coverage_deg)


# ---------------------------------------------------------------------------
# Validation B -- minimum depth: a void must reach some minimum distance
# from its boundary to count as a real opening. Small segmentation-noise
# pockets can clear the min_void_px area threshold while staying thin and
# shallow (run 37Z frame 332-343: noise holes peaked at distance_px ~2-6,
# vs ~28+ for the one genuine opening in that range).
# ---------------------------------------------------------------------------

def _is_real_opening(distance_px: float, min_void_distance_px: float) -> bool:
    return distance_px >= float(min_void_distance_px)


def _voids_for_component(
    component: np.ndarray,
    original_mask: np.ndarray,
    min_void_px: int,
    min_void_distance_px: float,
    min_coverage_deg: float,
    start_seq: int,
) -> tuple[list[VoidRecord], int]:
    contours, hierarchy = cv2.findContours(component.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], start_seq
    hierarchy = hierarchy[0]
    depths = _contour_depths(hierarchy)
    dist_field = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
    voids: list[VoidRecord] = []
    seq = start_seq
    for index, depth in enumerate(depths):
        if depth % 2 == 0:
            continue
        void_mask = _void_pixel_mask(contours, hierarchy, index, component)
        pixel_count = mask_count(void_mask)
        if pixel_count < int(min_void_px):
            continue
        _, max_val, _, _ = cv2.minMaxLoc(dist_field, mask=void_mask.astype(np.uint8))
        if not _is_real_opening(max_val, min_void_distance_px):  # Validation B
            continue
        ridge_ys, ridge_xs = np.where(void_mask & (dist_field >= max_val * 0.95))
        center = (int(round(ridge_xs.mean())), int(round(ridge_ys.mean())))
        coverage = _original_coverage_degrees(center, void_mask, original_mask)
        if not _has_original_support(coverage, min_coverage_deg):  # Validation A
            continue
        voids.append(VoidRecord(
            void_id=f"void_{seq:04d}",
            pixel_count=pixel_count,
            center=center,
            bbox=bbox_from_mask(void_mask),
            distance_px=float(max_val),
        ))
        seq += 1
    return voids, seq


def find_void_centers(
    bridge: BridgeFrame,
    min_void_px: int = DEFAULT_MIN_VOID_PX,
    min_mask_region_px: int = DEFAULT_MIN_MASK_REGION_PX,
    min_void_distance_px: float = DEFAULT_MIN_VOID_DISTANCE_PX,
    min_coverage_deg: float = DEFAULT_MIN_ORIGINAL_COVERAGE_DEG,
) -> VoidAnalysis:
    final = as_bool_mask(bridge.final_mask)
    original = as_bool_mask(bridge.fill.frame.base_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(final.astype(np.uint8), 8)
    regions: list[RegionRecord] = []
    seq = 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_mask_region_px):
            continue
        component = labels == label
        voids, seq = _voids_for_component(
            component, original, int(min_void_px), min_void_distance_px, min_coverage_deg, seq,
        )
        regions.append(RegionRecord(
            region_id=f"region_{len(regions) + 1:04d}",
            pixel_count=area,
            bbox=bbox_from_mask(component),
            voids=voids,
        ))
    return VoidAnalysis(bridge=bridge, regions=regions, min_void_px=int(min_void_px))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview void-center detection for one frame.")
    parser.add_argument("frame")
    parser.add_argument("--lut", default=DEFAULT_LUT_PATH)
    parser.add_argument("--fill-radius", type=float, default=DEFAULT_FILL_RADIUS_PX)
    parser.add_argument("--bridge-max", type=int, default=DEFAULT_BRIDGE_MAX_PX)
    parser.add_argument("--min-void-px", type=int, default=DEFAULT_MIN_VOID_PX)
    parser.add_argument("--min-mask-region-px", type=int, default=DEFAULT_MIN_MASK_REGION_PX)
    parser.add_argument("--min-void-distance-px", type=float, default=DEFAULT_MIN_VOID_DISTANCE_PX)
    parser.add_argument("--min-coverage-deg", type=float, default=DEFAULT_MIN_ORIGINAL_COVERAGE_DEG)
    args = parser.parse_args()
    fill = fill_close_pixels(frame_to_mask(args.frame, args.lut), args.fill_radius)
    bridged = bridge_outer_hull(fill, args.bridge_max, args.min_mask_region_px)
    analysis = find_void_centers(bridged, args.min_void_px, args.min_mask_region_px,
                                  args.min_void_distance_px, args.min_coverage_deg)
    print(json.dumps({
        "frame_path": analysis.bridge.fill.frame.frame_path,
        "region_count": len(analysis.regions),
        "void_count": sum(len(region.voids) for region in analysis.regions),
    }, indent=2))


if __name__ == "__main__":
    main()
