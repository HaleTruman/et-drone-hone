"""Offline clipped-component density reconstruction and quadrilateral trial.

This module is intentionally review-only.  It consumes the profiles recorded
by a historic ``DensityBankConfiguration`` and either reuses serialized
``DensityEvidence`` or reconstructs the same field from the serialized closed
component mask.  The clipped-specific reconstruction changes only the density
window denominator: pixels outside the camera frame are unobserved rather than
background.

The resulting quadrilateral is a visual-review candidate.  It is not a
pipeline result, is never routing eligible, and performs no pose estimation.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


DENSITY_FIT_VERSION = "clipped-density-quad-review-v1"
PROFILE_SELECTION_VERSION = "all-recorded-profiles-highest-review-score-v1"
BOUNDARY_DENOMINATOR_VERSION = "observed-frame-domain-denominator-v1"
MINIMUM_P90_POINTS = 12
MINIMUM_SIDE_POINTS = 2


def _integer_tuple(value: Any, length: int, field: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain {length} integers") from error
    if len(result) != length:
        raise ValueError(f"{field} must contain {length} integers")
    return result


def _box_sum(values: np.ndarray, radius: int, ddepth: int = -1) -> np.ndarray:
    window = 2 * radius + 1
    return cv2.boxFilter(
        values,
        ddepth,
        (window, window),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def _profile_id(profile: Mapping[str, Any]) -> str:
    return str(profile["profile_id"])


def _profile_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    maximum_area = profile.get("maximum_component_area_px")
    return {
        "profile_id": _profile_id(profile),
        "calibration_version": str(profile["calibration_version"]),
        "maximum_component_area_px": (
            None if maximum_area is None else int(maximum_area)
        ),
        "density_radius_px": int(profile["density_radius_px"]),
        "ridge_radius_px": int(profile["ridge_radius_px"]),
        "relative_cap": float(profile["relative_cap"]),
        "inverse_gamma": float(profile["inverse_gamma"]),
        "ridge_gamma": float(profile["ridge_gamma"]),
    }


def _component_local_context(
    frame: Mapping[str, Any], component: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Rebuild the shared component-local closed mask and observed domain."""
    image_shape = _integer_tuple(frame["image_shape"], 2, "image_shape")
    analysis_shape = _integer_tuple(
        component["analysis_shape"], 2, "analysis_shape"
    )
    origin_u, origin_v = _integer_tuple(
        component["image_origin_uv"], 2, "image_origin_uv"
    )
    x, y, width, height = _integer_tuple(
        component["bbox_xywh"], 4, "bbox_xywh"
    )
    frame_height, frame_width = image_shape
    labels = np.asarray(frame["component_labels"])
    if labels.shape != image_shape:
        raise ValueError("serialized component_labels shape differs from image_shape")
    if not (
        0 <= x <= frame_width
        and 0 <= y <= frame_height
        and 0 <= x + width <= frame_width
        and 0 <= y + height <= frame_height
    ):
        raise ValueError("component bbox lies outside the serialized frame")

    offset_x, offset_y = x - origin_u, y - origin_v
    local_height, local_width = analysis_shape
    if not (
        0 <= offset_x
        and 0 <= offset_y
        and offset_x + width <= local_width
        and offset_y + height <= local_height
    ):
        raise ValueError("component bbox does not fit its recorded analysis window")

    selected = labels[y:y + height, x:x + width] == int(
        component["component_id"]
    )
    closed = np.zeros(analysis_shape, np.uint8)
    closed[offset_y:offset_y + height, offset_x:offset_x + width] = selected

    local_y, local_x = np.indices(analysis_shape, dtype=np.int32)
    observed = (
        (local_x + origin_u >= 0)
        & (local_x + origin_u < frame_width)
        & (local_y + origin_v >= 0)
        & (local_y + origin_v < frame_height)
    )
    return closed, observed, (origin_u, origin_v), image_shape


def _reconstruct_density(
    closed_mask: np.ndarray,
    observed_domain: np.ndarray,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce DensityBank math with a clipped-aware observed denominator."""
    density_radius = int(profile["density_radius_px"])
    ridge_radius = int(profile["ridge_radius_px"])
    if density_radius < 1 or ridge_radius < 1:
        raise ValueError("density profile radii must be positive")
    inside = closed_mask != 0
    foreground = inside.astype(np.float32)
    density = _box_sum(foreground, density_radius, cv2.CV_32F)
    denominator = _box_sum(
        observed_domain.astype(np.float32), density_radius, cv2.CV_32F
    )
    density = np.divide(
        density,
        denominator,
        out=np.zeros_like(density),
        where=denominator > 0,
    )
    density[~inside] = 0

    normalized = np.zeros_like(density)
    mean_density = float(density[inside].mean()) if np.any(inside) else 0.0
    if mean_density > 0:
        normalized[inside] = np.clip(
            density[inside]
            / (mean_density * float(profile["relative_cap"])),
            0,
            1,
        )
    inverse = np.zeros_like(density)
    inverse[inside] = np.power(
        1.0 - normalized[inside],
        1.0 / float(profile["inverse_gamma"]),
    )

    yy, xx = np.indices(closed_mask.shape, dtype=np.float64)
    moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
    sums = _box_sum(moments, ridge_radius)
    mass, weighted_x, weighted_y = cv2.split(sums)
    centroid_x = np.divide(
        weighted_x, mass, out=np.zeros_like(mass), where=mass > 0
    )
    centroid_y = np.divide(
        weighted_y, mass, out=np.zeros_like(mass), where=mass > 0
    )
    distance = cv2.magnitude(xx - centroid_x, yy - centroid_y)
    ridge = np.power(
        1.0 - np.clip(distance / ridge_radius, 0, 1),
        float(profile["ridge_gamma"]),
    )
    ridge[(~inside) | (mass <= 0)] = 0
    field = inverse * ridge

    positive = inside & (field > 0)
    weights = field[positive].astype(np.float32)
    p90_threshold = float(np.percentile(weights, 90)) if weights.size else 0.0
    p90_mask = (positive & (field >= p90_threshold)).astype(np.uint8)
    return {
        "field": field,
        "p90_mask": p90_mask,
        "p90_threshold": p90_threshold,
        "positive_point_count": int(weights.size),
        "source": "reconstructed_from_recorded_profile_and_closed_mask",
        "boundary_handling": BOUNDARY_DENOMINATOR_VERSION,
    }


def _serialized_density(
    evidence: Mapping[str, Any], expected_shape: tuple[int, int]
) -> dict[str, Any] | None:
    field = np.asarray(evidence.get("final_field"))
    p90_mask = np.asarray(evidence.get("p90_mask"))
    if field.shape != expected_shape or p90_mask.shape != expected_shape:
        return None
    positive_points = np.asarray(evidence.get("positive_points_xy", ()))
    return {
        "field": field.astype(np.float64, copy=False),
        "p90_mask": (p90_mask != 0).astype(np.uint8),
        "p90_threshold": float(evidence.get("p90_threshold", 0.0)),
        "positive_point_count": int(
            len(positive_points) if positive_points.ndim > 0 else 0
        ),
        "source": "serialized_density_evidence",
        "boundary_handling": "recorded-density-bank-denominator",
    }


def _ordered_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    contour = ordered.astype(np.float32).reshape(-1, 1, 2)
    if cv2.contourArea(contour, oriented=True) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


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


def _line_intersection(first: tuple[np.ndarray, np.ndarray], second):
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


def _fit_lines(
    evidence: np.ndarray, scaffold: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    distances = np.column_stack([
        _segment_distance(
            evidence, scaffold[index], scaffold[(index + 1) % 4]
        )
        for index in range(4)
    ])
    assignments = np.argmin(distances, axis=1)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(4):
        side_points = evidence[assignments == index]
        if len(side_points) < MINIMUM_SIDE_POINTS:
            return None, assignments, "clipped_side_support_insufficient"
        vx, vy, x, y = cv2.fitLine(
            side_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        )
        lines.append((
            np.array((float(x.item()), float(y.item())), np.float64),
            np.array((float(vx.item()), float(vy.item())), np.float64),
        ))
    intersections = tuple(
        _line_intersection(lines[index], lines[(index + 1) % 4])
        for index in range(4)
    )
    if any(point is None for point in intersections):
        return None, assignments, "clipped_line_intersection_failed"
    return _ordered_corners(np.asarray(intersections)), assignments, None


def _side_metrics(
    evidence: np.ndarray,
    corners: np.ndarray,
    closed_mask: np.ndarray,
    image_origin_uv: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    distances = np.column_stack([
        _segment_distance(evidence, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    ])
    assignments = np.argmin(distances, axis=1)
    support_mask = cv2.dilate(
        (closed_mask != 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    local_height, local_width = closed_mask.shape
    frame_height, frame_width = image_shape
    origin = np.asarray(image_origin_uv, np.float64)
    component_scale = float(max(np.sqrt(abs(cv2.contourArea(
        corners.astype(np.float32).reshape(-1, 1, 2)
    ))), 1.0))
    count = max(len(evidence), 1)
    result = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        selected = evidence[assignments == index]
        if edge_length <= 1e-9 or not len(selected):
            coverage = 0.0
            rmse = float("inf")
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

        samples = start + np.linspace(0.0, 1.0, 65)[:, None] * edge
        sample_xy = np.rint(samples).astype(np.int32)
        local_in_bounds = (
            (sample_xy[:, 0] >= 0)
            & (sample_xy[:, 0] < local_width)
            & (sample_xy[:, 1] >= 0)
            & (sample_xy[:, 1] < local_height)
        )
        supported = np.zeros(len(samples), bool)
        supported[local_in_bounds] = support_mask[
            sample_xy[local_in_bounds, 1], sample_xy[local_in_bounds, 0]
        ] != 0
        full_samples = samples + origin
        frame_observed = (
            (full_samples[:, 0] >= 0)
            & (full_samples[:, 0] < frame_width)
            & (full_samples[:, 1] >= 0)
            & (full_samples[:, 1] < frame_height)
        )
        observed_count = int(np.count_nonzero(frame_observed))
        result.append({
            "side_index": index,
            "support_points": int(len(selected)),
            "support_fraction": float(len(selected) / count),
            "longitudinal_coverage_ratio": coverage,
            "rmse_px": rmse,
            "normalized_rmse": float(rmse / component_scale),
            "observed_fraction": float(np.mean(frame_observed)),
            "observed_mask_support_ratio": (
                float(np.mean(supported[frame_observed]))
                if observed_count else None
            ),
        })
    return result


def _rejected_fit(
    reason: str,
    *,
    evidence_points: int,
    initial_corners: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "status": "not_fitted",
        "candidate_for_review": False,
        "rejection_reason": reason,
        "p90_evidence_points": int(evidence_points),
        "initial_corners_local_xy": (
            None if initial_corners is None else initial_corners.tolist()
        ),
        "corners_uv": None,
        "quadrilateral_area_px2": None,
        "fit_score": 0.0,
        "off_frame_corner_count": None,
        "side_evidence": [],
    }


def _fit_quadrilateral(
    p90_mask: np.ndarray,
    closed_mask: np.ndarray,
    image_origin_uv: tuple[int, int],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    evidence_yx = np.argwhere(p90_mask != 0)
    evidence = evidence_yx[:, ::-1].astype(np.float64)
    if len(evidence) < MINIMUM_P90_POINTS:
        return _rejected_fit(
            "clipped_p90_evidence_insufficient", evidence_points=len(evidence)
        )
    hull = cv2.convexHull(evidence.astype(np.float32).reshape(-1, 1, 2))
    if len(hull) < 4 or cv2.contourArea(hull) < 4.0:
        return _rejected_fit(
            "clipped_p90_hull_degenerate", evidence_points=len(evidence)
        )
    if hasattr(cv2, "approxPolyN"):
        scaffold = cv2.approxPolyN(
            hull, 4, epsilon_percentage=-1, ensure_convex=True
        ).reshape(-1, 2)
    else:
        scaffold = cv2.boxPoints(cv2.minAreaRect(hull))
    if len(scaffold) != 4:
        return _rejected_fit(
            "clipped_p90_hull_degenerate", evidence_points=len(evidence)
        )
    scaffold = _ordered_corners(scaffold)
    fitted_local, _, fit_error = _fit_lines(evidence, scaffold)
    if fitted_local is None:
        return _rejected_fit(
            str(fit_error),
            evidence_points=len(evidence),
            initial_corners=scaffold,
        )

    contour = fitted_local.astype(np.float32).reshape(-1, 1, 2)
    area = float(abs(cv2.contourArea(contour)))
    origin = np.asarray(image_origin_uv, np.float64)
    corners_uv = fitted_local + origin
    frame_height, frame_width = image_shape
    extrapolation_limit = 2.0 * max(frame_height, frame_width, 1)
    structurally_valid = bool(
        fitted_local.shape == (4, 2)
        and np.all(np.isfinite(fitted_local))
        and cv2.isContourConvex(contour)
        and area >= 4.0
        and np.all(corners_uv[:, 0] >= -extrapolation_limit)
        and np.all(corners_uv[:, 0] <= frame_width + extrapolation_limit)
        and np.all(corners_uv[:, 1] >= -extrapolation_limit)
        and np.all(corners_uv[:, 1] <= frame_height + extrapolation_limit)
    )
    if not structurally_valid:
        return _rejected_fit(
            "clipped_quadrilateral_structurally_invalid",
            evidence_points=len(evidence),
            initial_corners=scaffold,
        )

    sides = _side_metrics(
        evidence, fitted_local, closed_mask, image_origin_uv, image_shape
    )
    minimum_support_fraction = min(
        side["support_fraction"] for side in sides
    )
    minimum_coverage = min(
        side["longitudinal_coverage_ratio"] for side in sides
    )
    maximum_normalized_rmse = max(side["normalized_rmse"] for side in sides)
    observed_supports = [
        side["observed_mask_support_ratio"]
        for side in sides
        if side["observed_mask_support_ratio"] is not None
    ]
    minimum_observed_support = min(observed_supports, default=0.0)
    balance_score = min(1.0, minimum_support_fraction / 0.12)
    coverage_score = min(1.0, minimum_coverage / 0.55)
    residual_score = float(np.clip(
        1.0 - maximum_normalized_rmse / 0.08, 0.0, 1.0
    ))
    mask_score = min(1.0, minimum_observed_support / 0.75)
    fit_score = float(
        0.25 * balance_score
        + 0.30 * coverage_score
        + 0.25 * residual_score
        + 0.20 * mask_score
    )
    off_frame = ~(
        (corners_uv[:, 0] >= 0)
        & (corners_uv[:, 0] < frame_width)
        & (corners_uv[:, 1] >= 0)
        & (corners_uv[:, 1] < frame_height)
    )
    return {
        "status": "candidate_for_review",
        "candidate_for_review": True,
        "rejection_reason": None,
        "p90_evidence_points": int(len(evidence)),
        "initial_corners_local_xy": scaffold.tolist(),
        "corners_uv": corners_uv.tolist(),
        "quadrilateral_area_px2": area,
        "fit_score": fit_score,
        "off_frame_corner_count": int(np.count_nonzero(off_frame)),
        "side_evidence": sides,
    }


def _full_frame_runs(
    local_mask: np.ndarray,
    image_origin_uv: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[list[int]]:
    frame = np.zeros(image_shape, bool)
    local_y, local_x = np.nonzero(local_mask)
    full_x = local_x + image_origin_uv[0]
    full_y = local_y + image_origin_uv[1]
    in_frame = (
        (full_x >= 0)
        & (full_x < image_shape[1])
        & (full_y >= 0)
        & (full_y < image_shape[0])
    )
    frame[full_y[in_frame], full_x[in_frame]] = True
    runs: list[list[int]] = []
    for row in np.flatnonzero(np.any(frame, axis=1)):
        columns = np.flatnonzero(frame[row])
        breaks = np.flatnonzero(np.diff(columns) > 1)
        starts = np.concatenate((columns[:1], columns[breaks + 1]))
        ends = np.concatenate((columns[breaks] + 1, columns[-1:] + 1))
        runs.extend(
            [int(row), int(start), int(end)]
            for start, end in zip(starts, ends)
        )
    return runs


def _heatmap(
    field: np.ndarray,
    closed_mask: np.ndarray,
    observed_domain: np.ndarray,
    image_origin_uv: tuple[int, int],
) -> dict[str, Any] | None:
    visible = (closed_mask != 0) & observed_domain
    rows, columns = np.nonzero(visible)
    if not len(rows):
        return None
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    crop_field = np.asarray(field[y0:y1, x0:x1], np.float64)
    crop_visible = visible[y0:y1, x0:x1]
    maximum = float(np.max(crop_field[crop_visible]))
    intensity = np.zeros(crop_field.shape, np.uint8)
    if maximum > 0:
        intensity[crop_visible] = np.rint(
            np.clip(crop_field[crop_visible] / maximum, 0, 1) * 255
        ).astype(np.uint8)
    bgr = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    alpha = np.where(crop_visible, 210, 0).astype(np.uint8)
    bgra = np.dstack((bgr, alpha))
    encoded, png = cv2.imencode(".png", bgra)
    if not encoded:
        return None
    return {
        "encoding": "png-base64-v1",
        "origin_uv": [image_origin_uv[0] + x0, image_origin_uv[1] + y0],
        "shape": [y1 - y0, x1 - x0],
        "field_max": maximum,
        "data": base64.b64encode(png.tobytes()).decode("ascii"),
    }


def fit_clipped_density(
    *,
    frame: Mapping[str, Any],
    component: Mapping[str, Any],
    density_configuration: Mapping[str, Any] | None,
    serialized_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate recorded density profiles and return one review candidate."""
    closed, observed, origin, image_shape = _component_local_context(
        frame, component
    )
    component_id = int(component["component_id"])
    serialized_by_profile: dict[str, Mapping[str, Any]] = {}
    for evidence in serialized_evidence:
        if int(evidence.get("component_id", -1)) != component_id:
            continue
        profile = evidence.get("profile")
        if isinstance(profile, Mapping) and "profile_id" in profile:
            serialized_by_profile[_profile_id(profile)] = evidence

    profiles: list[Mapping[str, Any]] = []
    if density_configuration is not None:
        profiles.extend(
            profile for profile in density_configuration.get("profiles", ())
            if isinstance(profile, Mapping)
        )
    if not profiles:
        profiles.extend(
            evidence["profile"] for evidence in serialized_by_profile.values()
            if isinstance(evidence.get("profile"), Mapping)
        )
    if not profiles:
        return {
            "process_version": DENSITY_FIT_VERSION,
            "status": "density_configuration_unavailable",
            "rejection_reason": "recorded_density_profiles_missing",
            "routing_eligible": False,
            "pose_eligible": False,
            "selected_profile": None,
            "quadrilateral": None,
            "profile_trials": [],
        }

    trials: list[dict[str, Any]] = []
    working: list[tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]] = []
    for profile in profiles:
        profile_id = _profile_id(profile)
        density = None
        serialized = serialized_by_profile.get(profile_id)
        if serialized is not None:
            density = _serialized_density(serialized, closed.shape)
        if density is None:
            density = _reconstruct_density(closed, observed, profile)
        fit = _fit_quadrilateral(
            density["p90_mask"], closed, origin, image_shape
        )
        trial = {
            "profile_id": profile_id,
            "density_source": density["source"],
            "boundary_handling": density["boundary_handling"],
            "positive_evidence_points": density["positive_point_count"],
            "p90_threshold": density["p90_threshold"],
            "p90_evidence_points": int(np.count_nonzero(density["p90_mask"])),
            "candidate_for_review": fit["candidate_for_review"],
            "fit_score": fit["fit_score"],
            "off_frame_corner_count": fit["off_frame_corner_count"],
            "rejection_reason": fit["rejection_reason"],
        }
        trials.append(trial)
        working.append((density, fit, profile))

    selected_index = max(
        range(len(working)),
        key=lambda item: (
            int(working[item][1]["candidate_for_review"]),
            float(working[item][1]["fit_score"]),
            int(np.count_nonzero(working[item][0]["p90_mask"])),
            -item,
        ),
    )
    density, fit, profile = working[selected_index]
    return {
        "process_version": DENSITY_FIT_VERSION,
        "status": fit["status"],
        "rejection_reason": fit["rejection_reason"],
        "evidence_scope": "offline_historic_review_only",
        "profile_selection": PROFILE_SELECTION_VERSION,
        "selected_profile": _profile_summary(profile),
        "density_source": density["source"],
        "boundary_handling": density["boundary_handling"],
        "positive_evidence_points": density["positive_point_count"],
        "p90_threshold": density["p90_threshold"],
        "p90_evidence_points": int(np.count_nonzero(density["p90_mask"])),
        "p90_runs": _full_frame_runs(
            density["p90_mask"], origin, image_shape
        ),
        "heatmap": _heatmap(density["field"], closed, observed, origin),
        "quadrilateral": fit,
        "profile_trials": trials,
        "routing_eligible": False,
        "pose_eligible": False,
        "promotion_status": "not_evaluated",
    }
