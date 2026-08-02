"""Coverage-optimized four-line P70 fitting for a standard-gate candidate."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..quadrilateral import (
    normalize_quadrilateral,
    ordered_quadrilateral_corners,
)
from ..schema import StandardGateConfiguration, StandardGateSideEvidence


FITTER_NAME = "standard_independent_p70_coverage_v1"
MAXIMUM_REFINEMENT_ITERATIONS = 8


@dataclass(frozen=True, slots=True)
class StandardQuadrilateralFit:
    """Internal fit result converted to the schema-owned process result."""

    initial_corners_uv: tuple[tuple[float, float], ...] | None
    fitted_corners_uv: tuple[tuple[float, float], ...] | None
    side_evidence: tuple[StandardGateSideEvidence, ...]
    aperture_center_offset_ratio: float | None
    area_px2: float | None
    p70_coverage_ratio: float
    fit_confidence: float
    accepted: bool
    rejection_reason: str | None


def _rejected(
    reason: str,
    *,
    initial_corners_uv=None,
    fitted_corners_uv=None,
    side_evidence=(),
    aperture_center_offset_ratio=None,
    area_px2=None,
    p70_coverage_ratio=0.0,
    fit_confidence=0.0,
) -> StandardQuadrilateralFit:
    return StandardQuadrilateralFit(
        initial_corners_uv=initial_corners_uv,
        fitted_corners_uv=fitted_corners_uv,
        side_evidence=tuple(side_evidence),
        aperture_center_offset_ratio=aperture_center_offset_ratio,
        area_px2=area_px2,
        p70_coverage_ratio=float(p70_coverage_ratio),
        fit_confidence=float(fit_confidence),
        accepted=False,
        rejection_reason=reason,
    )


def _segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    edge = end - start
    length_squared = float(edge @ edge)
    if length_squared <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    position = np.clip(((points - start) @ edge) / length_squared, 0.0, 1.0)
    closest = start + position[:, None] * edge
    return np.linalg.norm(points - closest, axis=1)


def _intersection(first, second) -> np.ndarray | None:
    point_a, direction_a = first
    point_b, direction_b = second
    denominator = float(
        direction_a[0] * direction_b[1]
        - direction_a[1] * direction_b[0]
    )
    if abs(denominator) < 1e-6:
        return None
    delta = point_b - point_a
    distance = float(
        (delta[0] * direction_b[1] - delta[1] * direction_b[0])
        / denominator
    )
    return point_a + distance * direction_a


def _fit_four_lines(
    evidence: np.ndarray,
    scaffold: np.ndarray,
    minimum_side_support_points: int,
) -> tuple[np.ndarray | None, str | None]:
    distances = np.column_stack(
        [
            _segment_distance(
                evidence, scaffold[index], scaffold[(index + 1) % 4]
            )
            for index in range(4)
        ]
    )
    assignments = np.argmin(distances, axis=1)
    lines = []
    for index in range(4):
        side_points = evidence[assignments == index]
        if len(side_points) < minimum_side_support_points:
            return None, "standard_side_support_insufficient"
        vx, vy, x, y = cv2.fitLine(
            side_points.astype(np.float32),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        )
        lines.append(
            (
                np.array((float(x.item()), float(y.item())), np.float64),
                np.array((float(vx.item()), float(vy.item())), np.float64),
            )
        )
    intersections = tuple(
        _intersection(lines[index], lines[(index + 1) % 4])
        for index in range(4)
    )
    if any(point is None for point in intersections):
        return None, "standard_line_intersection_failed"
    return ordered_quadrilateral_corners(np.asarray(intersections)), None


def _coverage_objective(
    evidence: np.ndarray,
    corners: np.ndarray,
    p70_mask: np.ndarray,
    support_radius_px: int,
) -> tuple[float, float]:
    """Score direct P70 perimeter coverage without filling or hulling it."""
    distances = np.column_stack(
        [
            _segment_distance(
                evidence, corners[index], corners[(index + 1) % 4]
            )
            for index in range(4)
        ]
    )
    assignments = np.argmin(distances, axis=1)
    nearest = distances[np.arange(len(evidence)), assignments]
    radius = float(max(1, support_radius_px))
    covered = nearest <= radius
    point_proximity = float(np.mean(covered))

    longitudinal_coverages = []
    side_counts = []
    for index in range(4):
        selected = evidence[(assignments == index) & covered]
        side_counts.append(len(selected))
        start = corners[index]
        edge = corners[(index + 1) % 4] - start
        length_squared = float(edge @ edge)
        if len(selected) < 2 or length_squared <= 1e-9:
            longitudinal_coverages.append(0.0)
            continue
        projections = np.clip(
            ((selected - start) @ edge) / length_squared, 0.0, 1.0
        )
        low, high = np.percentile(projections, (5, 95))
        longitudinal_coverages.append(float(max(0.0, high - low)))

    target_count = max(len(evidence) / 4.0, 1.0)
    balance = min(1.0, min(side_counts) / target_count)
    side_mask_coverages = []
    height, width = p70_mask.shape
    for index in range(4):
        start = corners[index]
        edge = corners[(index + 1) % 4] - start
        sample_count = max(2, int(np.ceil(np.linalg.norm(edge))) + 1)
        samples = start + np.linspace(0.0, 1.0, sample_count)[:, None] * edge
        sample_xy = np.rint(samples).astype(np.int32)
        in_bounds = (
            (sample_xy[:, 0] >= 0)
            & (sample_xy[:, 0] < width)
            & (sample_xy[:, 1] >= 0)
            & (sample_xy[:, 1] < height)
        )
        supported = np.zeros(sample_count, bool)
        supported[in_bounds] = p70_mask[
            sample_xy[in_bounds, 1], sample_xy[in_bounds, 0]
        ] != 0
        side_mask_coverages.append(float(np.mean(supported)))
    perimeter_coverage = float(np.mean(side_mask_coverages))
    objective = (
        0.40 * perimeter_coverage
        + 0.25 * min(side_mask_coverages)
        + 0.15 * point_proximity
        + 0.10 * min(longitudinal_coverages)
        + 0.10 * balance
    )
    return float(objective), perimeter_coverage


def _fit_coverage_optimized_quad(
    evidence: np.ndarray,
    p70_mask: np.ndarray,
    configuration: StandardGateConfiguration,
) -> tuple[np.ndarray | None, np.ndarray | None, float, str | None]:
    """Iteratively fit four lines directly to P70 pixels and retain best."""
    rectangle = cv2.minAreaRect(evidence.astype(np.float32))
    scaffold = ordered_quadrilateral_corners(cv2.boxPoints(rectangle))
    best = scaffold
    best_objective, best_coverage = _coverage_objective(
        evidence, scaffold, p70_mask, configuration.mask_support_radius_px
    )
    current = scaffold
    fit_error = None
    for _ in range(MAXIMUM_REFINEMENT_ITERATIONS):
        fitted, fit_error = _fit_four_lines(
            evidence, current, configuration.minimum_side_support_points
        )
        if fitted is None:
            break
        contour = fitted.astype(np.float32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 4.0:
            fit_error = "standard_p70_fit_degenerate"
            break
        objective, coverage = _coverage_objective(
            evidence, fitted, p70_mask,
            configuration.mask_support_radius_px,
        )
        if objective > best_objective:
            best, best_objective, best_coverage = fitted, objective, coverage
        movement = float(np.max(np.linalg.norm(fitted - current, axis=1)))
        current = fitted
        if movement < 0.05:
            break
    return scaffold, best, best_coverage, fit_error


def _side_evidence(
    evidence: np.ndarray,
    corners: np.ndarray,
    closed_mask: np.ndarray,
    configuration: StandardGateConfiguration,
) -> tuple[StandardGateSideEvidence, ...]:
    distances = np.column_stack(
        [
            _segment_distance(
                evidence, corners[index], corners[(index + 1) % 4]
            )
            for index in range(4)
        ]
    )
    assignments = np.argmin(distances, axis=1)
    radius = configuration.mask_support_radius_px
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        support_mask = cv2.dilate(
            (closed_mask != 0).astype(np.uint8), kernel
        )
    else:
        support_mask = closed_mask != 0
    height, width = support_mask.shape
    component_scale = float(max(height, width, 1))
    count = max(len(evidence), 1)
    sides = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        selected = evidence[assignments == index]
        if edge_length <= 1e-9 or len(selected) == 0:
            coverage, rmse = 0.0, float("inf")
        else:
            projections = np.clip(
                ((selected - start) @ edge) / (edge_length * edge_length),
                0.0,
                1.0,
            )
            low, high = np.percentile(projections, (5, 95))
            coverage = float(max(0.0, high - low))
            perpendicular = np.abs(
                edge[0] * (start[1] - selected[:, 1])
                - (start[0] - selected[:, 0]) * edge[1]
            ) / edge_length
            rmse = float(np.sqrt(np.mean(np.square(perpendicular))))

        samples = start + np.linspace(0.0, 1.0, 33)[:, None] * edge
        sample_xy = np.rint(samples).astype(np.int32)
        in_bounds = (
            (sample_xy[:, 0] >= 0)
            & (sample_xy[:, 0] < width)
            & (sample_xy[:, 1] >= 0)
            & (sample_xy[:, 1] < height)
        )
        supported = np.zeros(len(sample_xy), bool)
        supported[in_bounds] = support_mask[
            sample_xy[in_bounds, 1], sample_xy[in_bounds, 0]
        ] != 0
        sides.append(
            StandardGateSideEvidence(
                side_index=index,
                support_points=int(len(selected)),
                support_fraction=float(len(selected) / count),
                longitudinal_coverage_ratio=coverage,
                rmse_px=rmse,
                normalized_rmse=float(rmse / component_scale),
                mask_support_ratio=float(np.mean(supported)),
            )
        )
    return tuple(sides)


def _aperture_center_offset_ratio(
    aperture_contour_uv: np.ndarray,
    corners_uv: np.ndarray,
    area_px2: float,
) -> float:
    moments = cv2.moments(aperture_contour_uv)
    if abs(moments["m00"]) > 1e-9:
        aperture_center = np.array(
            (moments["m10"] / moments["m00"],
             moments["m01"] / moments["m00"]),
            np.float64,
        )
    else:
        aperture_center = aperture_contour_uv[:, 0, :].mean(axis=0)
    gate_center = corners_uv.mean(axis=0)
    return float(
        np.linalg.norm(aperture_center - gate_center)
        / max(np.sqrt(area_px2), 1.0)
    )


def fit_standard_quadrilateral(
    *,
    p70_mask: np.ndarray,
    closed_mask: np.ndarray,
    image_origin_uv: tuple[int, int],
    image_shape: tuple[int, int],
    aperture_contour_uv: np.ndarray,
    configuration: StandardGateConfiguration,
) -> StandardQuadrilateralFit:
    """Fit and confidence-gate one proposal directly from cached P70."""
    evidence_yx = np.argwhere(p70_mask)
    evidence = evidence_yx[:, ::-1].astype(np.float64)
    if len(evidence) < configuration.minimum_p70_evidence_points:
        return _rejected("standard_p70_evidence_insufficient")

    scaffold, fitted_local, p70_coverage, fit_error = \
        _fit_coverage_optimized_quad(evidence, p70_mask, configuration)
    if scaffold is None or fitted_local is None:
        return _rejected(
            fit_error or "standard_p70_fit_degenerate",
            p70_coverage_ratio=p70_coverage,
        )
    origin = np.asarray(image_origin_uv, np.float64)
    initial_uv, _ = normalize_quadrilateral(scaffold + origin, image_shape)
    if initial_uv is None:
        return _rejected("standard_quadrilateral_validation_failed")
    fitted_uv, area = normalize_quadrilateral(
        fitted_local + origin, image_shape
    )
    if fitted_uv is None or area is None:
        return _rejected(
            "standard_quadrilateral_validation_failed",
            initial_corners_uv=initial_uv,
        )

    corners_uv_array = np.asarray(fitted_uv, np.float64)
    fitted_local = corners_uv_array - origin
    sides = _side_evidence(
        evidence, fitted_local, closed_mask, configuration
    )
    aperture_offset = _aperture_center_offset_ratio(
        aperture_contour_uv, corners_uv_array, area
    )
    minimum_support = min(side.support_points for side in sides)
    minimum_coverage = min(
        side.longitudinal_coverage_ratio for side in sides
    )
    maximum_rmse = max(side.normalized_rmse for side in sides)
    minimum_mask_support = min(side.mask_support_ratio for side in sides)

    support_score = min(
        1.0,
        min(side.support_fraction for side in sides) / 0.12,
    )
    coverage_score = min(1.0, minimum_coverage / 0.55)
    residual_score = float(np.clip(
        1.0 - maximum_rmse / configuration.maximum_line_rmse_scale,
        0.0,
        1.0,
    ))
    mask_score = min(1.0, minimum_mask_support / 0.80)
    aperture_score = float(np.clip(
        1.0
        - aperture_offset
        / configuration.maximum_aperture_center_offset_ratio,
        0.0,
        1.0,
    ))
    p70_coverage_score = min(1.0, p70_coverage / 0.80)
    confidence = float(
        0.15 * support_score
        + 0.15 * coverage_score
        + 0.15 * residual_score
        + 0.20 * p70_coverage_score
        + 0.20 * mask_score
        + 0.15 * aperture_score
    )
    evidence_values = dict(
        initial_corners_uv=initial_uv,
        fitted_corners_uv=fitted_uv,
        side_evidence=sides,
        aperture_center_offset_ratio=aperture_offset,
        area_px2=area,
        p70_coverage_ratio=p70_coverage,
        fit_confidence=confidence,
    )
    if minimum_support < configuration.minimum_side_support_points:
        return _rejected(
            "standard_side_support_insufficient", **evidence_values
        )
    if minimum_coverage < configuration.minimum_side_coverage_ratio:
        return _rejected(
            "standard_side_coverage_insufficient", **evidence_values
        )
    if maximum_rmse > configuration.maximum_line_rmse_scale:
        return _rejected(
            "standard_line_residual_excessive", **evidence_values
        )
    if minimum_mask_support < configuration.minimum_side_mask_support_ratio:
        return _rejected(
            "standard_side_mask_support_insufficient", **evidence_values
        )
    if aperture_offset > configuration.maximum_aperture_center_offset_ratio:
        return _rejected("standard_aperture_unbalanced", **evidence_values)
    if confidence < configuration.minimum_fit_confidence:
        return _rejected(
            "standard_fit_confidence_below_minimum", **evidence_values
        )
    return StandardQuadrilateralFit(
        **evidence_values,
        accepted=True,
        rejection_reason=None,
    )
