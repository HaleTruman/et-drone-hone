"""Approved C-shape geometry baseline pinned on 2026-08-01.

This geometry-local copy preserves the reviewed contour-ray exit and tailored
extension policy. The upstream density and line-fitting modules remain shared;
all accepted refinement settings are passed explicitly here.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..c_shape_bounded_angle import (
    CShapeBoundedAngleResult, fit_bounded_angle_c_shape)
from ..c_shape_three_line_pose import CShapeDensityInput


CONTOUR_ANGULAR_INFLUENCE = 0.50
MAX_CONTOUR_CORRECTION_DEGREES = 15.0
LONGEST_GREEN_TARGET_SCALE = 0.75


@dataclass(frozen=True, slots=True)
class TailoredExtensionCandidate:
    extension_scales: tuple[float, float]
    observed_exits_uv: tuple[tuple[float, float], ...]
    extended_endpoints_uv: tuple[tuple[float, float], ...]
    quadrilateral_uv: tuple[tuple[float, float], ...]
    convex: bool
    area_px2: float


@dataclass(frozen=True, slots=True)
class CShapeGeometryBaselineResult:
    instance_id: str
    bounded_result: CShapeBoundedAngleResult
    candidate: TailoredExtensionCandidate


def _cross(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def ray_contour_exit(start, direction, contour):
    """Return the first forward intersection with a closed contour."""
    start = np.asarray(start, np.float64)
    direction = np.asarray(direction, np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    contour = np.asarray(contour, np.float64)
    intersections = []
    for index, segment_start in enumerate(contour):
        edge = contour[(index + 1) % len(contour)] - segment_start
        denominator = _cross(direction, edge)
        if abs(denominator) < 1e-9:
            continue
        relative = segment_start - start
        ray_position = _cross(relative, edge) / denominator
        edge_position = _cross(relative, direction) / denominator
        if ray_position > 1e-6 and -1e-9 <= edge_position <= 1.0 + 1e-9:
            intersections.append(ray_position)
    return None if not intersections else start + min(intersections) * direction


def contour_exits_from_paths(extent, contour):
    """Intersect the two adjusted outward arm rays with the outer contour."""
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    direction_hints = np.asarray(extent.outer_exits_uv, np.float64)
    exits = [ray_contour_exit(start, hint - start, contour)
             for start, hint in zip(closed, direction_hints)]
    if any(exit_point is None for exit_point in exits):
        return None
    return np.asarray(exits)


def _candidate(extent, exits, extension_lengths):
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    exits = np.asarray(exits, np.float64)
    directions = exits - closed
    green_lengths = np.linalg.norm(directions, axis=1)
    if np.any(green_lengths <= 1e-9):
        return None
    directions /= green_lengths[:, None]
    endpoints = exits + extension_lengths[:, None] * directions
    missing = extent.missing_side_index
    corners = np.empty((4, 2), np.float64)
    corners[(missing - 1) % 4] = closed[0]
    corners[missing] = endpoints[0]
    corners[(missing + 1) % 4] = endpoints[1]
    corners[(missing + 2) % 4] = closed[1]
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    scales = extension_lengths / max(extent.half_width_px, 1e-9)
    return TailoredExtensionCandidate(
        tuple(map(float, scales)), tuple(map(tuple, exits)),
        tuple(map(tuple, endpoints)), tuple(map(tuple, corners)),
        bool(cv2.isContourConvex(contour)), abs(float(cv2.contourArea(contour))))


def extend_by_scaled_green_shortfall(extent, observed_exits_uv):
    """Apply max(0, 0.75 * longest green - this green arm)."""
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    exits = np.asarray(observed_exits_uv, np.float64)
    arm_lengths = np.linalg.norm(exits - closed, axis=1)
    spine_length = float(np.linalg.norm(closed[1] - closed[0]))
    longest = max(spine_length, *map(float, arm_lengths))
    target = LONGEST_GREEN_TARGET_SCALE * longest
    extension_lengths = np.maximum(target - arm_lengths, 0.0)
    return _candidate(extent, exits, extension_lengths)


def fit_c_shape_geometry_baseline(source: CShapeDensityInput):
    """Run the accepted C-shape geometry baseline and return one candidate."""
    bounded = fit_bounded_angle_c_shape(
        source, influence=CONTOUR_ANGULAR_INFLUENCE,
        maximum_degrees=MAX_CONTOUR_CORRECTION_DEGREES)
    if bounded is None:
        return None
    contour = bounded.parallel_result.contour_result.outer_contour_uv
    exits = contour_exits_from_paths(bounded.bounded_extent, contour)
    if exits is None:
        return None
    candidate = extend_by_scaled_green_shortfall(
        bounded.bounded_extent, exits)
    if candidate is None:
        return None
    return CShapeGeometryBaselineResult(
        source.instance_id, bounded, candidate)
