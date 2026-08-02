"""Apply a small, bounded contour-angle correction to C-shape P90 lines."""

from dataclasses import dataclass

import numpy as np

from .c_shape_finite_extent import (
    CShapeExtentResult, estimate_c_shape_extent_from_lines)
from .c_shape_parallel_aligned import (
    CShapeParallelAlignedResult, fit_parallel_aligned_c_shape)
from .c_shape_three_line_pose import (
    CShapeDensityInput, CShapeLineFit, VisibleLine, fit_c_shape_lines)


CONTOUR_ANGULAR_INFLUENCE = 0.50
MAX_CONTOUR_CORRECTION_DEGREES = 15.0


@dataclass(frozen=True, slots=True)
class BoundedAngularLine:
    side_index: int
    p90_coefficients: tuple[float, float, float]
    contour_coefficients: tuple[float, float, float]
    bounded_coefficients: tuple[float, float, float]
    raw_contour_delta_degrees: float
    weighted_delta_degrees: float
    applied_delta_degrees: float
    correction_capped: bool
    p90_rmse_px: float


@dataclass(frozen=True, slots=True)
class CShapeBoundedAngleResult:
    instance_id: str
    contour_angular_influence: float
    maximum_contour_correction_degrees: float
    parallel_result: CShapeParallelAlignedResult
    bounded_extent: CShapeExtentResult
    bounded_lines: tuple[BoundedAngularLine, ...]


def _segment_distances(points, start, end):
    edge = end - start
    position = np.clip(
        ((points - start) @ edge) / max(float(edge @ edge), 1e-9), 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * edge), axis=1)


def _direction(line):
    return np.asarray((-line[1], line[0]), np.float64)


def bounded_axial_direction(p90_direction, contour_direction,
                            influence=CONTOUR_ANGULAR_INFLUENCE,
                            maximum_degrees=MAX_CONTOUR_CORRECTION_DEGREES):
    """Interpolate unoriented line angles and cap the applied correction."""
    p90_angle = float(np.arctan2(p90_direction[1], p90_direction[0]))
    contour_angle = float(np.arctan2(contour_direction[1], contour_direction[0]))
    difference = contour_angle - p90_angle
    raw = 0.5 * np.arctan2(np.sin(2 * difference), np.cos(2 * difference))
    weighted = influence * raw
    limit = np.radians(maximum_degrees)
    applied = float(np.clip(weighted, -limit, limit))
    direction = np.asarray((np.cos(p90_angle + applied),
                            np.sin(p90_angle + applied)))
    return direction, float(raw), float(weighted), applied


def fit_bounded_angle_c_shape(
        source: CShapeDensityInput,
        influence=CONTOUR_ANGULAR_INFLUENCE,
        maximum_degrees=MAX_CONTOUR_CORRECTION_DEGREES):
    """Keep P90 offsets while applying a limited contour-angle influence."""
    parallel_result = fit_parallel_aligned_c_shape(source)
    line_fit = fit_c_shape_lines(source)
    if parallel_result is None or line_fit is None:
        return None
    baseline = parallel_result.contour_result.baseline_extent
    missing = baseline.missing_side_index
    first, second, spine = (missing - 1) % 4, (missing + 1) % 4, (missing + 2) % 4
    closed = np.asarray(baseline.closed_intersections_uv)
    exits = np.asarray(baseline.outer_exits_uv)
    segments = {first: (closed[0], exits[0]), second: (closed[1], exits[1]),
                spine: (closed[0], closed[1])}
    origin = np.asarray(source.image_origin_uv, np.float64)
    yy, xx = np.nonzero(
        (source.mask != 0) & (source.density >= line_fit.threshold))
    evidence = np.column_stack((xx, yy)).astype(np.float64) + origin
    distances = np.column_stack([
        _segment_distances(evidence, *segments[line.side_index])
        for line in line_fit.visible_lines])
    assignments = np.argmin(distances, axis=1)
    contour_lines = {line.side_index: line
                     for line in parallel_result.parallel_lines}
    records, visible_lines = [], []
    for position, p90_line in enumerate(line_fit.visible_lines):
        contour_line = contour_lines[p90_line.side_index]
        p90 = np.asarray(p90_line.coefficients)
        contour = np.asarray(contour_line.parallel_coefficients)
        direction, raw, weighted, applied = bounded_axial_direction(
            _direction(p90), _direction(contour), influence, maximum_degrees)
        normal = np.asarray((-direction[1], direction[0]))
        if float(normal @ p90[:2]) < 0:
            normal = -normal
        p90_points = evidence[assignments == position]
        offset = float(np.median(p90_points @ normal))
        bounded = np.asarray((normal[0], normal[1], -offset))
        homogeneous = np.column_stack((p90_points, np.ones(len(p90_points))))
        rmse = float(np.sqrt(np.mean((homogeneous @ bounded) ** 2)))
        record = BoundedAngularLine(
            p90_line.side_index, p90_line.coefficients,
            contour_line.parallel_coefficients, tuple(map(float, bounded)),
            float(np.degrees(raw)), float(np.degrees(weighted)),
            float(np.degrees(applied)),
            bool(abs(weighted) > np.radians(maximum_degrees)),
            rmse)
        records.append(record)
        visible_lines.append(VisibleLine(
            p90_line.side_index, record.bounded_coefficients,
            p90_line.support_points))
    bounded_fit = CShapeLineFit(
        line_fit.instance_id, line_fit.threshold, line_fit.scaffold_uv, missing,
        tuple(visible_lines))
    extent = estimate_c_shape_extent_from_lines(source, bounded_fit)
    if extent is None:
        return None
    return CShapeBoundedAngleResult(
        source.instance_id, influence, maximum_degrees, parallel_result,
        extent, tuple(records))
