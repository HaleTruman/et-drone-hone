"""Component-wide exploratory identification of overlapping-gate candidates.

This module is the production-local counterpart to the isolated
``profile_calibration/local_thickness`` discovery work.  It intentionally does
not import that review package.  The result is proposal evidence: only the
authoritative topology decision can make a proposal ready for the multi-gate
fitter.
"""

from __future__ import annotations

from math import atan2, ceil, degrees, hypot, pi, sqrt
from typing import Iterable

import cv2
import numpy as np

from ..configurations import (
    DEFAULT_MULTI_GATE_CONFIGURATION,
    DEFAULT_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
)
from ..density_bank import DensityBank
from ..preprocessing import component_mask
from ..schema import (
    MULTI_GATE_ROUTE,
    MULTI_VOID_TOPOLOGY,
    ComponentObservation,
    FrameObservation,
    MultiGateCandidateAssessment,
    MultiGateConfiguration,
    MultiGateDensityOverlapEvidence,
    MultiGateIdentificationConfiguration,
    MultiGateIdentificationResult,
    MultiGateJunctionEvidence,
    MultiGateThicknessPopulationEvidence,
    TopologyDecision,
)
from .profile_selection import select_multi_gate_density_profile_id


def _validate_configuration(
    configuration: MultiGateIdentificationConfiguration,
) -> None:
    positive = (
        configuration.regional_smoothing_sigma_px,
        configuration.excess_ratio,
        configuration.minimum_excess_area_scale,
        configuration.opposing_tolerance_degrees,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("multi-gate identification scales must be positive")
    if configuration.excess_ratio <= 1.0:
        raise ValueError("multi-gate excess ratio must be greater than one")
    if configuration.angle_bin_count < 4:
        raise ValueError("multi-gate identification requires at least four bins")
    if configuration.minimum_branch_bins < 1:
        raise ValueError("minimum branch width must be positive")
    if configuration.minimum_branch_count < 2:
        raise ValueError("minimum branch count must be at least two")
    boundaries = configuration.population_boundaries_px
    if (
        not boundaries
        or any(not np.isfinite(value) or value <= 0 for value in boundaries)
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("population boundaries must be positive and increasing")
    fractions = (
        configuration.population_minimum_fraction,
        configuration.population_minimum_explained_variance,
    )
    if any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("population fractions must be in (0, 1]")
    if configuration.population_minimum_pixels < 1:
        raise ValueError("population minimum pixels must be positive")
    if configuration.population_minimum_median_ratio <= 1.0:
        raise ValueError("population median ratio must be greater than one")


def padded_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Return precise component-local L2 distance with a known zero border."""
    inside = (np.asarray(mask) != 0).astype(np.uint8)
    padded = np.pad(inside, 1, mode="constant")
    distance = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return distance[1:-1, 1:-1].astype(np.float32, copy=False)


def _maximal_disk_centers(distance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep foreground disks not contained by an adjacent larger disk."""
    selector = np.asarray(distance, np.float32)
    padded = np.pad(selector, 1, mode="constant")
    contained = np.zeros(selector.shape, dtype=bool)
    height, width = selector.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = padded[
                1 + dy:1 + dy + height,
                1 + dx:1 + dx + width,
            ]
            contained |= neighbor >= selector + hypot(dx, dy) - 1e-4
    centers = (np.asarray(mask) != 0) & (selector > 0) & ~contained
    if np.any(mask) and not np.any(centers):
        centers.flat[int(np.argmax(selector))] = True
    return centers


def covering_local_radius(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the largest retained foreground disk over every mask pixel."""
    inside = np.asarray(mask) != 0
    distance = padded_distance_transform(inside)
    centers = _maximal_disk_centers(distance, inside)
    radius_map = np.zeros(distance.shape, np.float32)
    ys, xs = np.nonzero(centers)
    radii = distance[ys, xs]
    for index in np.argsort(radii)[::-1]:
        center_y = int(ys[index])
        center_x = int(xs[index])
        radius = float(radii[index])
        extent = int(ceil(radius))
        y0 = max(0, center_y - extent)
        y1 = min(distance.shape[0], center_y + extent + 1)
        x0 = max(0, center_x - extent)
        x1 = min(distance.shape[1], center_x + extent + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        covered = (
            (yy - center_y) ** 2 + (xx - center_x) ** 2
            <= radius * radius + 1e-6
        )
        covered &= inside[y0:y1, x0:x1]
        target = radius_map[y0:y1, x0:x1]
        target[covered] = np.maximum(target[covered], radius)
    missing = inside & (radius_map <= 0)
    radius_map[missing] = distance[missing]
    return distance, radius_map


def regional_thickness(
    thickness_px: np.ndarray,
    mask: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """Smooth one thickness map without leaking through mask background."""
    inside = np.asarray(mask) != 0
    values = np.asarray(thickness_px, np.float32)
    numerator = cv2.GaussianBlur(
        values * inside.astype(np.float32),
        (0, 0),
        sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )
    denominator = cv2.GaussianBlur(
        inside.astype(np.float32),
        (0, 0),
        sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )
    output = np.zeros_like(values)
    np.divide(numerator, denominator, out=output, where=denominator > 1e-6)
    output[~inside] = 0
    return output


def _empty_population(reason: str) -> MultiGateThicknessPopulationEvidence:
    return MultiGateThicknessPopulationEvidence(
        available=False,
        split_threshold_px=None,
        thinner_pixel_count=0,
        thicker_pixel_count=0,
        thinner_fraction=0.0,
        thicker_fraction=0.0,
        thinner_median_px=0.0,
        thicker_median_px=0.0,
        median_ratio=0.0,
        explained_variance=0.0,
        reason=reason,
    )


def segment_thickness_populations(
    mask: np.ndarray,
    thickness_px: np.ndarray,
    configuration: MultiGateIdentificationConfiguration =
        DEFAULT_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
) -> MultiGateThicknessPopulationEvidence:
    """Measure a possible thinner/thicker split without assigning gate IDs."""
    inside = np.asarray(mask) != 0
    values = np.asarray(thickness_px, np.float32)[inside]
    minimum_count = max(
        configuration.population_minimum_pixels,
        int(ceil(values.size * configuration.population_minimum_fraction)),
    )
    if values.size < 2 * minimum_count:
        return _empty_population("insufficient_population_support")
    total_variance = float(np.var(values))
    if total_variance <= 1e-9:
        median = float(np.median(values))
        return MultiGateThicknessPopulationEvidence(
            False, None, 0, 0, 0.0, 0.0, median, median, 1.0, 0.0,
            "uniform_thickness",
        )

    best = None
    for boundary in configuration.population_boundaries_px:
        thinner = values < boundary
        thinner_count = int(np.count_nonzero(thinner))
        thicker_count = int(values.size - thinner_count)
        if thinner_count < minimum_count or thicker_count < minimum_count:
            continue
        thinner_values = values[thinner]
        thicker_values = values[~thinner]
        thinner_median = float(np.median(thinner_values))
        thicker_median = float(np.median(thicker_values))
        median_ratio = thicker_median / max(thinner_median, 1e-6)
        within_variance = (
            thinner_count * float(np.var(thinner_values))
            + thicker_count * float(np.var(thicker_values))
        ) / values.size
        explained = float(np.clip(
            1.0 - within_variance / total_variance, 0.0, 1.0
        ))
        balance = 2.0 * min(thinner_count, thicker_count) / values.size
        proposal = (
            explained * balance,
            float(boundary),
            thinner_count,
            thicker_count,
            thinner_median,
            thicker_median,
            median_ratio,
            explained,
        )
        if best is None or proposal[0] > best[0]:
            best = proposal

    if best is None:
        return _empty_population("no_band_boundary_with_two_supported_sides")
    (
        _, boundary, thinner_count, thicker_count, thinner_median,
        thicker_median, median_ratio, explained,
    ) = best
    available = (
        median_ratio >= configuration.population_minimum_median_ratio
        and explained >= configuration.population_minimum_explained_variance
    )
    return MultiGateThicknessPopulationEvidence(
        available=available,
        split_threshold_px=boundary,
        thinner_pixel_count=thinner_count,
        thicker_pixel_count=thicker_count,
        thinner_fraction=thinner_count / values.size,
        thicker_fraction=thicker_count / values.size,
        thinner_median_px=thinner_median,
        thicker_median_px=thicker_median,
        median_ratio=median_ratio,
        explained_variance=explained,
        reason=(
            "two_supported_thickness_populations"
            if available
            else "population_separation_below_threshold"
        ),
    )


def _circular_branch_runs(
    occupied: np.ndarray,
    minimum_branch_bins: int,
) -> tuple[tuple[int, ...], ...]:
    values = np.asarray(occupied, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        return ()
    values = values | (np.roll(values, 1) & np.roll(values, -1))
    if np.all(values):
        return (tuple(range(values.size)),)
    starts = np.flatnonzero(values & ~np.roll(values, 1))
    runs = []
    for start in starts:
        run = []
        index = int(start)
        while values[index]:
            run.append(index)
            index = (index + 1) % values.size
        if len(run) >= minimum_branch_bins:
            runs.append(tuple(run))
    return tuple(runs)


def _run_angle_degrees(run: tuple[int, ...], bin_count: int) -> float:
    angles = (np.asarray(run, np.float64) + 0.5) * (2.0 * pi / bin_count)
    mean = atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    return degrees(mean) % 360.0


def _has_opposing_pair(
    angles_degrees: tuple[float, ...],
    tolerance_degrees: float,
) -> bool:
    for first_index, first in enumerate(angles_degrees):
        for second in angles_degrees[first_index + 1:]:
            separation = abs(first - second) % 360.0
            separation = min(separation, 360.0 - separation)
            if abs(separation - 180.0) <= tolerance_degrees:
                return True
    return False


def _annulus_branches(
    mask: np.ndarray,
    center_xy: tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
    configuration: MultiGateIdentificationConfiguration,
) -> tuple[tuple[float, ...], np.ndarray, tuple[tuple[int, ...], ...]]:
    radii = np.linspace(
        max(1.0, inner_radius_px),
        max(inner_radius_px + 1.0, outer_radius_px),
        max(3, int(ceil(outer_radius_px - inner_radius_px)) + 1),
    )
    angles = (
        np.arange(configuration.angle_bin_count, dtype=np.float32) + 0.5
    ) * (2.0 * pi / configuration.angle_bin_count)
    xs = np.rint(
        center_xy[0] + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.int32)
    ys = np.rint(
        center_xy[1] + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.int32)
    valid = (
        (xs >= 0)
        & (xs < mask.shape[1])
        & (ys >= 0)
        & (ys < mask.shape[0])
    )
    samples = np.zeros(xs.shape, bool)
    samples[valid] = np.asarray(mask, bool)[ys[valid], xs[valid]]
    occupied = samples.mean(axis=1) >= 0.5
    runs = _circular_branch_runs(
        occupied, configuration.minimum_branch_bins
    )
    branch_angles = tuple(
        _run_angle_degrees(run, configuration.angle_bin_count) for run in runs
    )
    return branch_angles, occupied, runs


def _boundary_safe_mask(
    component: ComponentObservation,
    mask: np.ndarray,
    thickness_px: np.ndarray,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    inside = np.asarray(mask) != 0
    if not component.touches_frame:
        return inside
    x, y, width, height = component.bbox_xywh
    origin_u, origin_v = component.image_origin_uv
    frame_height, frame_width = frame_shape
    yy, xx = np.indices(inside.shape, dtype=np.float32)
    uu = xx + origin_u
    vv = yy + origin_v
    safe = inside.copy()
    thickness = np.asarray(thickness_px, np.float32)
    if x == 0:
        safe &= uu > thickness
    if y == 0:
        safe &= vv > thickness
    if x + width >= frame_width:
        safe &= (frame_width - 1 - uu) > thickness
    if y + height >= frame_height:
        safe &= (frame_height - 1 - vv) > thickness
    return safe


def _density_overlap_evidence(
    component: ComponentObservation,
    density_bank: DensityBank,
    supported_excess: np.ndarray,
    multi_gate_configuration: MultiGateConfiguration,
) -> MultiGateDensityOverlapEvidence | None:
    if not density_bank.includes_component(component):
        return None
    profile_id = select_multi_gate_density_profile_id(
        component, multi_gate_configuration
    )
    density = density_bank.get(component, profile_id)
    p90 = np.asarray(density.p90_mask) != 0
    component_count, _ = cv2.connectedComponents(
        p90.astype(np.uint8), connectivity=8
    )
    supported_count = int(np.count_nonzero(supported_excess))
    intersection_count = int(np.count_nonzero(p90 & supported_excess))
    return MultiGateDensityOverlapEvidence(
        density_profile_id=profile_id,
        p70_threshold=float(density.p70_threshold),
        p80_threshold=float(density.p80_threshold),
        p90_threshold=float(density.p90_threshold),
        p70_pixel_count=int(np.count_nonzero(density.p70_mask)),
        p80_pixel_count=int(np.count_nonzero(density.p80_mask)),
        p90_pixel_count=int(np.count_nonzero(density.p90_mask)),
        p90_component_count=max(0, component_count - 1),
        supported_excess_pixel_count=supported_count,
        p90_supported_excess_pixel_count=intersection_count,
        supported_excess_p90_fraction=(
            intersection_count / supported_count if supported_count else 0.0
        ),
    )


def _readiness_reason(
    component: ComponentObservation,
    decision: TopologyDecision,
    candidate: bool,
    candidate_reason: str,
) -> tuple[bool, str]:
    if component.touches_frame:
        return False, "frame_edge_clipped"
    if not candidate:
        return False, f"overlap_identification_not_candidate:{candidate_reason}"
    if (
        not decision.accepted
        or decision.route != MULTI_GATE_ROUTE
        or decision.topology_label != MULTI_VOID_TOPOLOGY
    ):
        return False, "authoritative_topology_route_not_multi_gate"
    if (
        decision.closed_significant_holes != 2
        or len(decision.significant_parent_child_contour_ids) != 2
    ):
        return False, "multi_gate_fitter_requires_two_aperture_seeds"
    return True, "ready_for_multi_gate_fitter"


def assess_multi_gate_candidate(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
    density_bank: DensityBank,
    configuration: MultiGateIdentificationConfiguration =
        DEFAULT_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
    multi_gate_configuration: MultiGateConfiguration =
        DEFAULT_MULTI_GATE_CONFIGURATION,
) -> MultiGateCandidateAssessment:
    """Assess one component without changing its authoritative topology route."""
    _validate_configuration(configuration)
    if not any(owned is component for owned in frame.components):
        raise ValueError("component does not belong to frame")
    identity = (frame.frame_id, frame.sim_time_ns, component.component_id)
    if identity != (
        decision.frame_id,
        decision.sim_time_ns,
        decision.component_id,
    ):
        raise ValueError("topology decision belongs to another component")
    if density_bank.frame is not frame:
        raise ValueError("density bank belongs to another frame")

    if component.touches_frame and not configuration.analyze_clipped_components:
        return MultiGateCandidateAssessment(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            component_id=component.component_id,
            bbox_xywh=component.bbox_xywh,
            touches_frame=True,
            topology_label=decision.topology_label,
            topology_route=decision.route,
            topology_accepted=decision.accepted,
            topology_classification_rule=decision.classification_rule,
            closed_significant_apertures=decision.closed_significant_holes,
            foreground_px=component.area_px,
            thickness_reference_px=0.0,
            thickness_p90_px=0.0,
            thickness_p95_px=0.0,
            thickness_max_px=0.0,
            peak_ratio=0.0,
            excess_pixel_count=0,
            excess_island_count=0,
            supported_excess_island_count=0,
            minimum_excess_island_area_px=0,
            assessment="indeterminate",
            experimental_overlap_candidate=False,
            evidence_scope="frame_edge_clipped_not_analyzed",
            routing_eligible=False,
            fitter_ready=False,
            candidate_reason="frame_edge_clipped_review_disabled",
            fitter_readiness_reason="frame_edge_clipped",
            population=_empty_population("clipped_review_evidence_disabled"),
            junctions=(),
            density_overlap=None,
        )

    mask = component_mask(frame, component, stage="closed")
    _, local_radius = covering_local_radius(mask)
    local_thickness = np.zeros_like(local_radius)
    inside = mask != 0
    local_thickness[inside] = 2.0 * local_radius[inside]
    local_regional = regional_thickness(
        local_thickness, mask, configuration.regional_smoothing_sigma_px
    )
    observable = _boundary_safe_mask(
        component, mask, local_regional, frame.image_shape
    )
    values = local_regional[observable]
    population = segment_thickness_populations(
        observable, local_regional, configuration
    )
    foreground_px = int(np.count_nonzero(inside))
    supported_excess = np.zeros(inside.shape, bool)
    junctions: list[MultiGateJunctionEvidence] = []

    if values.size:
        reference = max(float(np.percentile(values, 50)), 1e-6)
        p90 = float(np.percentile(values, 90))
        p95 = float(np.percentile(values, 95))
        maximum = float(values.max())
        ratio = np.zeros(inside.shape, np.float32)
        ratio[observable] = local_regional[observable] / reference
        excess = observable & (ratio >= configuration.excess_ratio)
        excess_pixel_count = int(np.count_nonzero(excess))
        minimum_area = max(
            3,
            int(ceil(
                configuration.minimum_excess_area_scale
                * reference
                * reference
            )),
        )
        label_count = 1
        labels = np.zeros(inside.shape, np.int32)
        stats = np.zeros((1, 5), np.int32)
        if np.any(excess):
            label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                excess.astype(np.uint8), connectivity=8
            )
        supported_island_count = 0
        observed_supported_excess = np.zeros(inside.shape, bool)
        for island_id in range(1, label_count):
            area = int(stats[island_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            supported_island_count += 1
            island = labels == island_id
            supported_excess |= island
            ys, xs = np.nonzero(island)
            weights = np.maximum(
                ratio[island] - configuration.excess_ratio, 1e-3
            )
            center_x = float(np.average(xs, weights=weights))
            center_y = float(np.average(ys, weights=weights))
            equivalent_radius = sqrt(area / pi)
            inner_radius = equivalent_radius + 0.50 * reference
            outer_radius = equivalent_radius + 1.00 * reference
            branch_angles, occupied, branch_runs = _annulus_branches(
                inside,
                (center_x, center_y),
                inner_radius,
                outer_radius,
                configuration,
            )
            origin_u, origin_v = component.image_origin_uv
            center_u = float(origin_u + center_x)
            center_v = float(origin_v + center_y)
            if not component.touches_frame:
                support_fully_observed = True
            else:
                frame_height, frame_width = frame.image_shape
                support_fully_observed = (
                    center_u - outer_radius >= 0
                    and center_v - outer_radius >= 0
                    and center_u + outer_radius < frame_width
                    and center_v + outer_radius < frame_height
                )
            if support_fully_observed:
                observed_supported_excess |= island
            geometric_candidate = (
                len(branch_angles) >= configuration.minimum_branch_count
            )
            junctions.append(MultiGateJunctionEvidence(
                center_uv=(int(round(center_u)), int(round(center_v))),
                center_local_xy=(round(center_x, 3), round(center_y, 3)),
                peak_ratio=float(ratio[island].max()),
                excess_area_px=area,
                excess_area_scale=float(area / (reference * reference)),
                annulus_inner_radius_px=round(inner_radius, 3),
                annulus_outer_radius_px=round(outer_radius, 3),
                occupied_angle_bins=tuple(
                    int(index) for index in np.flatnonzero(occupied)
                ),
                branch_count=len(branch_angles),
                branch_angles_deg=tuple(
                    round(value, 2) for value in branch_angles
                ),
                branch_width_bins=tuple(len(run) for run in branch_runs),
                opposing_pair=_has_opposing_pair(
                    branch_angles, configuration.opposing_tolerance_degrees
                ),
                support_fully_observed=support_fully_observed,
                candidate=geometric_candidate and support_fully_observed,
            ))

        junction_candidate = any(item.candidate for item in junctions)
        aperture_candidate = (
            decision.closed_significant_holes >= 2
            and any(item.support_fully_observed for item in junctions)
        )
        candidate = junction_candidate or aperture_candidate
        if junction_candidate:
            assessment = (
                "candidate_clipped_review_only"
                if component.touches_frame
                else "candidate"
            )
            candidate_reason = "thickness_excess_with_multi_arm_junction"
        elif aperture_candidate:
            assessment = (
                "candidate_clipped_review_only"
                if component.touches_frame
                else "candidate"
            )
            candidate_reason = "thickness_excess_with_multiple_apertures"
        elif component.touches_frame:
            assessment = "indeterminate"
            candidate_reason = (
                "frame_edge_clipped_support_incomplete"
                if junctions
                else "frame_edge_clipped_without_supported_proposal"
            )
        elif not excess_pixel_count:
            assessment = "not_candidate"
            candidate_reason = "no_component_local_thickness_excess"
        elif not junctions:
            assessment = "not_candidate"
            candidate_reason = "thickness_excess_below_scale_support"
        else:
            assessment = "not_candidate"
            candidate_reason = "thickness_excess_without_multi_arm_junction"
        excess_island_count = max(0, label_count - 1)
    else:
        reference = p90 = p95 = maximum = 0.0
        excess_pixel_count = 0
        excess_island_count = 0
        supported_island_count = 0
        candidate = False
        assessment = "indeterminate" if component.touches_frame else "not_candidate"
        candidate_reason = (
            "no_boundary_safe_evidence" if foreground_px else "empty_component"
        )

    density_overlap = _density_overlap_evidence(
        component, density_bank, supported_excess, multi_gate_configuration
    )
    routing_eligible = candidate and not component.touches_frame
    fitter_ready, readiness_reason = _readiness_reason(
        component, decision, candidate, candidate_reason
    )
    return MultiGateCandidateAssessment(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        bbox_xywh=component.bbox_xywh,
        touches_frame=component.touches_frame,
        topology_label=decision.topology_label,
        topology_route=decision.route,
        topology_accepted=decision.accepted,
        topology_classification_rule=decision.classification_rule,
        closed_significant_apertures=decision.closed_significant_holes,
        foreground_px=foreground_px,
        thickness_reference_px=reference,
        thickness_p90_px=p90,
        thickness_p95_px=p95,
        thickness_max_px=maximum,
        peak_ratio=maximum / max(reference, 1e-6),
        excess_pixel_count=excess_pixel_count,
        excess_island_count=excess_island_count,
        supported_excess_island_count=supported_island_count,
        minimum_excess_island_area_px=(
            max(
                3,
                int(ceil(
                    configuration.minimum_excess_area_scale
                    * reference
                    * reference
                )),
            )
            if reference > 0
            else 0
        ),
        assessment=assessment,
        experimental_overlap_candidate=candidate,
        evidence_scope=(
            "frame_edge_clipped_partial"
            if component.touches_frame
            else "complete_component"
        ),
        routing_eligible=routing_eligible,
        fitter_ready=fitter_ready,
        candidate_reason=candidate_reason,
        fitter_readiness_reason=readiness_reason,
        population=population,
        junctions=tuple(junctions),
        density_overlap=density_overlap,
    )


def identify_multi_gate_candidates(
    frame: FrameObservation,
    decisions: Iterable[TopologyDecision],
    density_bank: DensityBank,
    configuration: MultiGateIdentificationConfiguration =
        DEFAULT_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
    multi_gate_configuration: MultiGateConfiguration =
        DEFAULT_MULTI_GATE_CONFIGURATION,
) -> MultiGateIdentificationResult:
    """Skim every incoming component and retain all overlap assessments."""
    _validate_configuration(configuration)
    if density_bank.frame is not frame:
        raise ValueError("density bank belongs to another frame")
    decision_values = tuple(decisions)
    decision_map = {item.component_id: item for item in decision_values}
    if len(decision_map) != len(decision_values):
        raise ValueError("topology decisions must have unique component IDs")
    component_ids = {component.component_id for component in frame.components}
    if set(decision_map) != component_ids:
        raise ValueError("topology decisions must cover every frame component")
    assessments = tuple(
        assess_multi_gate_candidate(
            frame,
            component,
            decision_map[component.component_id],
            density_bank,
            configuration,
            multi_gate_configuration,
        )
        for component in frame.components
    )
    return MultiGateIdentificationResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        configuration_version=configuration.configuration_version,
        component_assessments=assessments,
        candidate_component_ids=tuple(
            item.component_id
            for item in assessments
            if item.experimental_overlap_candidate
        ),
        routing_eligible_component_ids=tuple(
            item.component_id for item in assessments if item.routing_eligible
        ),
        fitter_ready_component_ids=tuple(
            item.component_id for item in assessments if item.fitter_ready
        ),
    )
