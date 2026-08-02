"""Complete one C-shaped centerline from mask extent; no PnP is performed."""

from dataclasses import dataclass

import cv2
import numpy as np

from .c_shape_three_line_pose import (
    CShapeDensityInput, CShapeLineFit, VisibleLine, fit_c_shape_lines)


@dataclass(frozen=True, slots=True)
class CShapeExtentResult:
    instance_id: str
    threshold: float
    missing_side_index: int
    spine_side_index: int
    visible_lines: tuple[VisibleLine, ...]
    closed_intersections_uv: tuple[tuple[float, float], ...]
    half_width_px: float
    full_width_px: float
    arm_extents_px: tuple[float, float]
    outer_exits_uv: tuple[tuple[float, float], ...]
    extended_endpoints_uv: tuple[tuple[float, float], ...]
    quadrilateral_uv: tuple[tuple[float, float], ...]


def _intersection(first, second):
    matrix = np.asarray((first[:2], second[:2]), np.float64)
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None
    return np.linalg.solve(matrix, -np.asarray((first[2], second[2])))


def _half_width(mask, start, end, origin):
    distance = cv2.distanceTransform(
        (mask != 0).astype(np.uint8), cv2.DIST_L2, 5)
    local_start, local_end = start - origin, end - origin
    sample_count = max(2, int(np.ceil(np.linalg.norm(local_end - local_start))) * 2)
    points = np.linspace(local_start, local_end, sample_count)
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, mask.shape[0] - 1)
    values = distance[y, x]
    values = values[values > 0]
    return None if not len(values) else float(np.median(values))


def _arm_extent(mask_points, intersection, line, direction_hint, half_width):
    direction = np.asarray((-line[1], line[0]), np.float64)
    if float(direction @ direction_hint) < 0:
        direction = -direction
    perpendicular = np.abs(
        np.column_stack((mask_points, np.ones(len(mask_points)))) @ line)
    projections = (mask_points - intersection) @ direction
    supported = projections[(perpendicular <= half_width) & (projections >= 0)]
    if not len(supported):
        return None
    extent = float(np.max(supported))
    outer_exit = intersection + extent * direction
    endpoint = intersection + (extent + half_width) * direction
    return extent, outer_exit, endpoint


def estimate_c_shape_extent_from_lines(
        source: CShapeDensityInput, line_fit: CShapeLineFit):
    """Complete a C shape from an explicit three-line fit and mask extent."""
    missing = line_fit.missing_side_index
    spine_index = (missing + 2) % 4
    first_arm, second_arm = (missing - 1) % 4, (missing + 1) % 4
    lines = {item.side_index: np.asarray(item.coefficients, np.float64)
             for item in line_fit.visible_lines}
    first_closed = _intersection(lines[spine_index], lines[first_arm])
    second_closed = _intersection(lines[spine_index], lines[second_arm])
    if first_closed is None or second_closed is None:
        return None
    origin = np.asarray(source.image_origin_uv, np.float64)
    half_width = _half_width(
        source.mask, first_closed, second_closed, origin)
    if half_width is None:
        return None
    yy, xx = np.nonzero(source.mask)
    mask_points = np.column_stack((xx, yy)).astype(np.float64) + origin
    scaffold = np.asarray(line_fit.scaffold_uv, np.float64)
    first = _arm_extent(
        mask_points, first_closed, lines[first_arm],
        scaffold[missing] - first_closed, half_width)
    second = _arm_extent(
        mask_points, second_closed, lines[second_arm],
        scaffold[(missing + 1) % 4] - second_closed, half_width)
    if first is None or second is None:
        return None
    corners = np.empty((4, 2), np.float64)
    corners[(missing - 1) % 4] = first_closed
    corners[missing] = first[2]
    corners[(missing + 1) % 4] = second[2]
    corners[(missing + 2) % 4] = second_closed
    return CShapeExtentResult(
        source.instance_id, line_fit.threshold, missing, spine_index,
        line_fit.visible_lines,
        tuple(map(tuple, (first_closed, second_closed))), half_width,
        2.0 * half_width, (first[0], second[0]),
        tuple(map(tuple, (first[1], second[1]))),
        tuple(map(tuple, (first[2], second[2]))), tuple(map(tuple, corners)))


def estimate_c_shape_extent(source: CShapeDensityInput):
    """Use P90 lines, mask exits, and distance-transform half-width."""
    line_fit = fit_c_shape_lines(source)
    return (None if line_fit is None else
            estimate_c_shape_extent_from_lines(source, line_fit))
