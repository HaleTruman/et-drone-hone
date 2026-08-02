"""Align C-shape density lines to a parallel consensus of contour rails."""

from dataclasses import dataclass

import numpy as np

from .c_shape_contour_guided import (
    CShapeContourGuidedResult, fit_contour_guided_c_shape)
from .c_shape_finite_extent import (
    CShapeExtentResult, estimate_c_shape_extent_from_lines)
from .c_shape_three_line_pose import (
    CShapeDensityInput, CShapeLineFit, VisibleLine, fit_c_shape_lines)


@dataclass(frozen=True, slots=True)
class ParallelAlignedLine:
    side_index: int
    p90_coefficients: tuple[float, float, float]
    parallel_coefficients: tuple[float, float, float]
    rail_angle_gap_degrees: float
    p90_angle_delta_degrees: float
    p90_rmse_px: float


@dataclass(frozen=True, slots=True)
class CShapeParallelAlignedResult:
    instance_id: str
    contour_result: CShapeContourGuidedResult
    parallel_extent: CShapeExtentResult
    parallel_lines: tuple[ParallelAlignedLine, ...]


def _segment_distances(points, start, end):
    edge = end - start
    position = np.clip(
        ((points - start) @ edge) / max(float(edge @ edge), 1e-9), 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * edge), axis=1)


def _direction(line):
    return np.asarray((-line[1], line[0]), np.float64)


def _axial_angle(first, second):
    cosine = np.clip(abs(float(first @ second)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def fit_parallel_aligned_c_shape(source: CShapeDensityInput):
    """Use contour rails only for parallel angle and P90 only for offset."""
    contour_result = fit_contour_guided_c_shape(source)
    line_fit = fit_c_shape_lines(source)
    if contour_result is None or line_fit is None:
        return None
    baseline = contour_result.baseline_extent
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
                     for line in contour_result.guided_lines}
    records, visible_lines = [], []
    for position, p90_line in enumerate(line_fit.visible_lines):
        contour_line = contour_lines[p90_line.side_index]
        rails = [np.asarray(value) for value in contour_line.rail_coefficients]
        directions = [_direction(line) for line in rails]
        moment = sum(np.outer(value, value) for value in directions)
        _, vectors = np.linalg.eigh(moment)
        direction = vectors[:, -1]
        normal = np.asarray((-direction[1], direction[0]))
        baseline_line = np.asarray(p90_line.coefficients)
        if float(normal @ baseline_line[:2]) < 0:
            normal = -normal
        p90_points = evidence[assignments == position]
        offset = float(np.median(p90_points @ normal))
        aligned = np.asarray((normal[0], normal[1], -offset))
        homogeneous = np.column_stack((p90_points, np.ones(len(p90_points))))
        rmse = float(np.sqrt(np.mean((homogeneous @ aligned) ** 2)))
        record = ParallelAlignedLine(
            p90_line.side_index, p90_line.coefficients,
            tuple(map(float, aligned)), _axial_angle(*directions),
            _axial_angle(_direction(baseline_line), direction), rmse)
        records.append(record)
        visible_lines.append(VisibleLine(
            p90_line.side_index, record.parallel_coefficients,
            p90_line.support_points))
    parallel_fit = CShapeLineFit(
        line_fit.instance_id, line_fit.threshold, line_fit.scaffold_uv, missing,
        tuple(visible_lines))
    extent = estimate_c_shape_extent_from_lines(source, parallel_fit)
    if extent is None:
        return None
    return CShapeParallelAlignedResult(
        source.instance_id, contour_result, extent, tuple(records))
