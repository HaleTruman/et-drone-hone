"""Refine C-shape density lines with paired outer-contour rails."""

from dataclasses import dataclass

import cv2
import numpy as np

from .c_shape_finite_extent import (
    CShapeExtentResult, estimate_c_shape_extent_from_lines)
from .c_shape_three_line_pose import (
    CShapeDensityInput, CShapeLineFit, VisibleLine, fit_c_shape_lines)


CONTOUR_SIMPLIFY_EPSILON_PX = 3.0


@dataclass(frozen=True, slots=True)
class ContourGuidedLine:
    side_index: int
    p90_coefficients: tuple[float, float, float]
    guided_coefficients: tuple[float, float, float]
    rail_coefficients: tuple[tuple[float, float, float], ...]
    midpoint_samples_uv: tuple[tuple[float, float], ...]
    contour_support: tuple[int, int]
    p90_support: int
    angle_delta_degrees: float
    p90_rmse_px: float


@dataclass(frozen=True, slots=True)
class CShapeContourGuidedResult:
    instance_id: str
    raw_outer_contour_uv: tuple[tuple[float, float], ...]
    outer_contour_uv: tuple[tuple[float, float], ...]
    baseline_extent: CShapeExtentResult
    guided_extent: CShapeExtentResult
    guided_lines: tuple[ContourGuidedLine, ...]


def _segment_distances(points, start, end):
    edge = end - start
    position = np.clip(
        ((points - start) @ edge) / max(float(edge @ edge), 1e-9), 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * edge), axis=1)


def _fit_line(points):
    vx, vy, x, y = cv2.fitLine(
        np.asarray(points, np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    line = np.array((-vy, vx, vy * x - vx * y), np.float64)
    return line / np.linalg.norm(line[:2])


def _densify_closed_contour(vertices):
    samples = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        count = max(1, int(np.ceil(np.linalg.norm(end - start))))
        samples.extend(start + fraction * (end - start)
                       for fraction in np.arange(count) / count)
    return np.asarray(samples, np.float64)


def _guide_line(contour, p90_points, start, end, half_width, baseline):
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 2.0 * half_width:
        return None
    direction /= length
    normal = np.array((-direction[1], direction[0]))
    relative = contour - start
    longitudinal, perpendicular = relative @ direction, relative @ normal
    selected = ((longitudinal >= half_width) &
                (longitudinal <= length - half_width) &
                (np.abs(perpendicular) <= 2.0 * half_width))
    negative = contour[selected & (perpendicular < 0)]
    positive = contour[selected & (perpendicular >= 0)]
    if len(negative) < 2 or len(positive) < 2 or len(p90_points) < 2:
        return None
    rails = (_fit_line(negative), _fit_line(positive))
    positions = np.linspace(
        half_width, length - half_width,
        max(2, int(np.ceil(length - 2.0 * half_width)) + 1))
    centers = []
    for position in positions:
        base = start + position * direction
        crossings = []
        for rail in rails:
            denominator = float(rail[:2] @ normal)
            if abs(denominator) < 1e-8:
                return None
            distance = -float(rail[:2] @ base + rail[2]) / denominator
            crossings.append(base + distance * normal)
        centers.append(0.5 * (crossings[0] + crossings[1]))
    contour_center = _fit_line(centers)
    guided_normal = contour_center[:2]
    offset = float(np.median(p90_points @ guided_normal))
    guided = np.array((guided_normal[0], guided_normal[1], -offset))
    if float(guided[:2] @ baseline[:2]) < 0:
        guided = -guided
    direction_before = np.array((-baseline[1], baseline[0]))
    direction_after = np.array((-guided[1], guided[0]))
    cosine = np.clip(abs(float(direction_before @ direction_after)), 0, 1)
    angle = float(np.degrees(np.arccos(cosine)))
    homogeneous = np.column_stack((p90_points, np.ones(len(p90_points))))
    rmse = float(np.sqrt(np.mean((homogeneous @ guided) ** 2)))
    return guided, rails, centers, (len(negative), len(positive)), angle, rmse


def fit_contour_guided_c_shape(source: CShapeDensityInput):
    """Use contour rails for angle and P90 pixels for center offset."""
    line_fit = fit_c_shape_lines(source)
    if line_fit is None:
        return None
    baseline = estimate_c_shape_extent_from_lines(source, line_fit)
    if baseline is None:
        return None
    contours, _ = cv2.findContours(
        (source.mask != 0).astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    origin = np.asarray(source.image_origin_uv, np.float64)
    raw_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    simplified = cv2.approxPolyDP(
        raw_contour.astype(np.float32).reshape(-1, 1, 2),
        CONTOUR_SIMPLIFY_EPSILON_PX, True).reshape(-1, 2)
    contour = _densify_closed_contour(simplified) + origin
    yy, xx = np.nonzero(
        (source.mask != 0) & (source.density >= line_fit.threshold))
    evidence = np.column_stack((xx, yy)).astype(np.float64) + origin
    missing = line_fit.missing_side_index
    first_arm, second_arm, spine = (
        (missing - 1) % 4, (missing + 1) % 4, (missing + 2) % 4)
    closed = np.asarray(baseline.closed_intersections_uv)
    exits = np.asarray(baseline.outer_exits_uv)
    segments = {first_arm: (closed[0], exits[0]),
                second_arm: (closed[1], exits[1]),
                spine: (closed[0], closed[1])}
    distances = np.column_stack([
        _segment_distances(evidence, *segments[item.side_index])
        for item in line_fit.visible_lines])
    assignments = np.argmin(distances, axis=1)
    guided_records, guided_visible = [], []
    for position, visible in enumerate(line_fit.visible_lines):
        baseline_line = np.asarray(visible.coefficients)
        guided = _guide_line(
            contour, evidence[assignments == position], *segments[visible.side_index],
            baseline.half_width_px, baseline_line)
        if guided is None:
            return None
        line, rails, centers, support, angle, rmse = guided
        guided_records.append(ContourGuidedLine(
            visible.side_index, visible.coefficients, tuple(map(float, line)),
            tuple(tuple(map(float, rail)) for rail in rails),
            tuple(tuple(map(float, center)) for center in centers), support,
            visible.support_points, angle, rmse))
        guided_visible.append(VisibleLine(
            visible.side_index, tuple(map(float, line)), visible.support_points))
    guided_fit = CShapeLineFit(
        line_fit.instance_id, line_fit.threshold, line_fit.scaffold_uv, missing,
        tuple(guided_visible))
    guided_extent = estimate_c_shape_extent_from_lines(source, guided_fit)
    if guided_extent is None:
        return None
    return CShapeContourGuidedResult(
        source.instance_id,
        tuple(map(tuple, (raw_contour.astype(float) + origin))),
        tuple(map(tuple, (simplified.astype(float) + origin))), baseline,
        guided_extent, tuple(guided_records))
