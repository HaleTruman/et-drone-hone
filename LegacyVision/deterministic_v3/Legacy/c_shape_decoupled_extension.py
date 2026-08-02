"""Decouple predicted C-shape corners from the observed mask exits."""

from dataclasses import dataclass

import cv2
import numpy as np

from .c_shape_bounded_angle import (
    CShapeBoundedAngleResult, fit_bounded_angle_c_shape)
from .c_shape_three_line_pose import CShapeDensityInput


LONGEST_GREEN_TARGET_SCALE = 0.75


@dataclass(frozen=True, slots=True)
class DecoupledExtensionCandidate:
    extension_scales: tuple[float, float]
    observed_exits_uv: tuple[tuple[float, float], ...]
    extended_endpoints_uv: tuple[tuple[float, float], ...]
    quadrilateral_uv: tuple[tuple[float, float], ...]
    convex: bool
    area_px2: float


@dataclass(frozen=True, slots=True)
class CShapeDecoupledExtensionResult:
    instance_id: str
    bounded_result: CShapeBoundedAngleResult
    candidates: tuple[DecoupledExtensionCandidate, ...]


def _cross(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def ray_contour_exit(start, direction, contour):
    """Return the first forward intersection with a closed polygon."""
    start = np.asarray(start, np.float64)
    direction = np.asarray(direction, np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    contour = np.asarray(contour, np.float64)
    intersections = []
    for index, segment_start in enumerate(contour):
        segment_end = contour[(index + 1) % len(contour)]
        edge = segment_end - segment_start
        denominator = _cross(direction, edge)
        if abs(denominator) < 1e-9:
            continue
        relative = segment_start - start
        ray_position = _cross(relative, edge) / denominator
        edge_position = _cross(relative, direction) / denominator
        if ray_position > 1e-6 and -1e-9 <= edge_position <= 1.0 + 1e-9:
            intersections.append(ray_position)
    if not intersections:
        return None
    return start + min(intersections) * direction


def contour_exits_from_paths(extent, contour):
    """Intersect each fitted outward arm ray with the simplified contour."""
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    mask_exits = np.asarray(extent.outer_exits_uv, np.float64)
    exits = [ray_contour_exit(start, hint - start, contour)
             for start, hint in zip(closed, mask_exits)]
    if any(exit_point is None for exit_point in exits):
        return None
    return np.asarray(exits)


def extend_from_observed_paths(extent, first_scale, second_scale,
                               observed_exits_uv=None):
    """Hold observed green paths fixed and extend their endpoints along-line."""
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    exits = np.asarray(
        extent.outer_exits_uv if observed_exits_uv is None else observed_exits_uv,
        np.float64)
    directions = exits - closed
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= 1e-9):
        return None
    directions /= lengths[:, None]
    scales = np.asarray((first_scale, second_scale), np.float64)
    endpoints = exits + scales[:, None] * extent.half_width_px * directions
    missing = extent.missing_side_index
    corners = np.empty((4, 2), np.float64)
    corners[(missing - 1) % 4] = closed[0]
    corners[missing] = endpoints[0]
    corners[(missing + 1) % 4] = endpoints[1]
    corners[(missing + 2) % 4] = closed[1]
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    return DecoupledExtensionCandidate(
        (float(first_scale), float(second_scale)), tuple(map(tuple, exits)),
        tuple(map(tuple, endpoints)), tuple(map(tuple, corners)),
        bool(cv2.isContourConvex(contour)), abs(float(cv2.contourArea(contour))))


def extend_to_longest_green_path(extent, observed_exits_uv):
    """Extend each protruding arm by its shortfall from longest green."""
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    exits = np.asarray(observed_exits_uv, np.float64)
    arm_lengths = np.linalg.norm(exits - closed, axis=1)
    spine_length = float(np.linalg.norm(closed[1] - closed[0]))
    target_length = LONGEST_GREEN_TARGET_SCALE * max(
        spine_length, *map(float, arm_lengths))
    shortfalls = np.maximum(target_length - arm_lengths, 0.0)
    extension_lengths = shortfalls
    scales = extension_lengths / max(extent.half_width_px, 1e-9)
    return extend_from_observed_paths(
        extent, float(scales[0]), float(scales[1]), exits)


def fit_decoupled_extension_sweep(
        source: CShapeDensityInput, scales=(0.5, 1.0, 2.0, 3.0)):
    """Evaluate extension lengths without refitting the green evidence paths."""
    bounded = fit_bounded_angle_c_shape(source)
    if bounded is None:
        return None
    contour = bounded.parallel_result.contour_result.outer_contour_uv
    exits = contour_exits_from_paths(bounded.bounded_extent, contour)
    if exits is None:
        return None
    candidates = tuple(
        extend_from_observed_paths(
            bounded.bounded_extent, scale, scale, exits)
        for scale in scales)
    candidates = tuple(candidate for candidate in candidates
                       if candidate is not None)
    return CShapeDecoupledExtensionResult(
        source.instance_id, bounded, candidates)


def fit_longest_green_extension(source: CShapeDensityInput):
    """Extend each protruding arm by its shortfall from longest green."""
    bounded = fit_bounded_angle_c_shape(source)
    if bounded is None:
        return None
    contour = bounded.parallel_result.contour_result.outer_contour_uv
    exits = contour_exits_from_paths(bounded.bounded_extent, contour)
    if exits is None:
        return None
    candidate = extend_to_longest_green_path(
        bounded.bounded_extent, exits)
    if candidate is None:
        return None
    return CShapeDecoupledExtensionResult(
        source.instance_id, bounded, (candidate,))
