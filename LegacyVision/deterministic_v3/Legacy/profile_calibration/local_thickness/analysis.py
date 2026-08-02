"""Component-constrained local thickness and adaptive-profile experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, ceil, degrees, hypot, pi, sqrt

import cv2
import numpy as np

from ...configurations import DEFAULT_DENSITY_CONFIGURATION
from ...density_bank import DensityBank
from ...preprocessing import component_mask
from ...schema import DensityBankConfiguration, FrameObservation


ASSIGNMENT_MODES = (
    "equivalent_area_thickness",
    "calibrated_thickness_bands",
    "l2_covering_disk",
    "square_window",
)
# Tie-preserving physical bands measured across 2,441 retained, non-clipped
# components from the three-run spectrum-calibration corpus.  The ascending
# order maps directly to the current scale_01 (finest) .. scale_10 (coarsest)
# density deck without splitting identical thickness values between profiles.
THICKNESS_PROFILE_BOUNDARIES_PX = (
    6.25, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 25.0,
)
REGIONAL_THICKNESS_SMOOTHING_SIGMA_PX = 1.0
POPULATION_MIN_FRACTION = 0.10
POPULATION_MIN_PIXELS = 24
POPULATION_MIN_MEDIAN_RATIO = 1.25
POPULATION_MIN_EXPLAINED_VARIANCE = 0.35
OVERLAP_EXCESS_RATIO = 1.20
OVERLAP_MIN_EXCESS_AREA_SCALE = 0.15
OVERLAP_ANGLE_BINS = 36
OVERLAP_MIN_BRANCH_BINS = 2
OVERLAP_MIN_BRANCHES = 3
OVERLAP_OPPOSING_TOLERANCE_DEG = 35.0
PROFILE_RADII = tuple(
    profile.density_radius_px
    for profile in DEFAULT_DENSITY_CONFIGURATION.profiles
)


@dataclass(frozen=True, slots=True)
class ThicknessJunctionEvidence:
    """One component-local thickness proposal and its directional support."""

    center_uv: tuple[int, int]
    center_local_xy: tuple[float, float]
    peak_ratio: float
    excess_area_px: int
    excess_area_scale: float
    annulus_inner_radius_px: float
    annulus_outer_radius_px: float
    occupied_angle_bins: tuple[int, ...]
    branch_count: int
    branch_angles_deg: tuple[float, ...]
    branch_width_bins: tuple[int, ...]
    opposing_pair: bool
    support_fully_observed: bool
    candidate: bool


@dataclass(frozen=True, slots=True)
class ThicknessPopulationEvidence:
    """Deterministic two-population split of one component's wall scale."""

    available: bool
    split_threshold_px: float | None
    thinner_pixel_count: int
    thicker_pixel_count: int
    thinner_fraction: float
    thicker_fraction: float
    thinner_median_px: float
    thicker_median_px: float
    median_ratio: float
    explained_variance: float
    reason: str


@dataclass(frozen=True, slots=True)
class ComponentThicknessEvidence:
    """Auditable, review-only thickness evidence for one connected component."""

    component_id: int
    bbox_xywh: tuple[int, int, int, int]
    touches_frame: bool
    closed_hole_count: int
    foreground_px: int
    thickness_reference_px: float
    thickness_display_cap_px: float
    thickness_p50_px: float
    thickness_p90_px: float
    thickness_p95_px: float
    thickness_max_px: float
    peak_ratio: float
    excess_pixel_count: int
    excess_island_count: int
    minimum_excess_island_area_px: int
    assessment: str
    experimental_overlap_candidate: bool
    evidence_scope: str
    routing_eligible: bool
    thickness_split_ready: bool
    population: ThicknessPopulationEvidence
    reason: str
    junctions: tuple[ThicknessJunctionEvidence, ...]


@dataclass(frozen=True, slots=True)
class ThicknessAnalysis:
    distance_to_background_px: np.ndarray
    covering_local_radius_px: np.ndarray
    covering_local_thickness_px: np.ndarray
    regional_local_thickness_px: np.ndarray
    component_local_normalized_thickness: np.ndarray
    component_relative_thickness: np.ndarray
    maximal_disk_centers: np.ndarray
    overlap_candidate_mask: np.ndarray
    overlap_candidate_cores: np.ndarray
    thinner_population_mask: np.ndarray
    thicker_population_mask: np.ndarray
    thickness_band_index: np.ndarray
    adaptive_profile_index: np.ndarray
    profile_seams: np.ndarray
    adaptive_final_field: np.ndarray
    adaptive_p70_mask: np.ndarray
    adaptive_p80_mask: np.ndarray
    adaptive_p90_mask: np.ndarray
    component_evidence: tuple[ComponentThicknessEvidence, ...]
    metrics: dict


def padded_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Return precise L2 distance with an explicit zero boundary."""
    inside = (np.asarray(mask) != 0).astype(np.uint8)
    padded = np.pad(inside, 1, mode="constant")
    distance = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance[1:-1, 1:-1].astype(np.float32, copy=False)


def _maximal_disk_centers(
    distance: np.ndarray,
    mask: np.ndarray,
    smoothing_sigma_px: float,
) -> np.ndarray:
    """Keep disks not contained by an immediately adjacent larger disk."""
    selector = distance
    if smoothing_sigma_px > 0:
        selector = cv2.GaussianBlur(
            distance, (0, 0), smoothing_sigma_px,
            borderType=cv2.BORDER_CONSTANT)
        selector[np.asarray(mask) == 0] = 0
    padded = np.pad(selector, 1, mode="constant")
    contained = np.zeros(distance.shape, dtype=bool)
    height, width = distance.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = padded[
                1 + dy:1 + dy + height,
                1 + dx:1 + dx + width,
            ]
            contained |= neighbor >= selector + hypot(dx, dy) - 1e-4
    centers = (np.asarray(mask) != 0) & (distance > 0) & ~contained
    if np.any(mask) and not np.any(centers):
        centers.flat[int(np.argmax(distance))] = True
    return centers


def covering_local_radius(
    mask: np.ndarray,
    *,
    smoothing_sigma_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize the largest retained L2 disk covering each mask pixel."""
    inside = np.asarray(mask) != 0
    distance = padded_distance_transform(inside)
    centers = _maximal_disk_centers(
        distance, inside, smoothing_sigma_px)
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
    return distance, radius_map, centers


def regional_thickness(
    thickness_px: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_px: float = REGIONAL_THICKNESS_SMOOTHING_SIGMA_PX,
) -> np.ndarray:
    """Smooth thickness inside one component without leaking through holes."""
    inside = np.asarray(mask) != 0
    values = np.asarray(thickness_px, np.float32)
    if sigma_px <= 0:
        output = np.zeros_like(values)
        output[inside] = values[inside]
        return output
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
    np.divide(
        numerator,
        denominator,
        out=output,
        where=denominator > 1e-6,
    )
    output[~inside] = 0
    return output


def thickness_band_assignment(
    mask: np.ndarray,
    thickness_px: np.ndarray,
) -> np.ndarray:
    """Map measured physical thickness to the tie-preserving ten-profile deck."""
    inside = np.asarray(mask) != 0
    assignment = np.full(inside.shape, -1, np.int16)
    assignment[inside] = np.digitize(
        np.asarray(thickness_px, np.float32)[inside],
        THICKNESS_PROFILE_BOUNDARIES_PX,
        right=False,
    ).astype(np.int16)
    return assignment


def segment_thickness_populations(
    mask: np.ndarray,
    thickness_px: np.ndarray,
) -> tuple[ThicknessPopulationEvidence, np.ndarray, np.ndarray]:
    """Attempt a deterministic low/high wall-scale split for review.

    Candidate thresholds are restricted to the corpus-derived physical band
    boundaries.  The selected threshold maximizes one-dimensional explained
    variance while requiring meaningful support and median separation on both
    sides.  The two returned masks are thickness populations, not gate IDs.
    """
    inside = np.asarray(mask) != 0
    values = np.asarray(thickness_px, np.float32)[inside]
    empty = np.zeros(inside.shape, bool)
    minimum_count = max(
        POPULATION_MIN_PIXELS,
        int(ceil(values.size * POPULATION_MIN_FRACTION)),
    )
    if values.size < 2 * minimum_count:
        return ThicknessPopulationEvidence(
            False, None, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "insufficient_population_support",
        ), empty, empty.copy()

    total_variance = float(np.var(values))
    if total_variance <= 1e-9:
        return ThicknessPopulationEvidence(
            False, None, 0, 0, 0.0, 0.0,
            float(np.median(values)), float(np.median(values)), 1.0, 0.0,
            "uniform_thickness",
        ), empty, empty.copy()

    best = None
    for boundary in THICKNESS_PROFILE_BOUNDARIES_PX:
        thinner = values < boundary
        thinner_count = int(np.count_nonzero(thinner))
        thicker_count = int(values.size - thinner_count)
        if thinner_count < minimum_count or thicker_count < minimum_count:
            continue
        thin_values = values[thinner]
        thick_values = values[~thinner]
        thin_median = float(np.median(thin_values))
        thick_median = float(np.median(thick_values))
        median_ratio = thick_median / max(thin_median, 1e-6)
        within_variance = (
            thinner_count * float(np.var(thin_values))
            + thicker_count * float(np.var(thick_values))
        ) / values.size
        explained = float(np.clip(
            1.0 - within_variance / total_variance, 0.0, 1.0))
        balance = 2.0 * min(thinner_count, thicker_count) / values.size
        score = explained * balance
        proposal = (
            score,
            float(boundary),
            thinner_count,
            thicker_count,
            thin_median,
            thick_median,
            median_ratio,
            explained,
        )
        if best is None or proposal[0] > best[0]:
            best = proposal

    if best is None:
        return ThicknessPopulationEvidence(
            False, None, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "no_band_boundary_with_two_supported_sides",
        ), empty, empty.copy()

    (_, boundary, thinner_count, thicker_count, thin_median, thick_median,
     median_ratio, explained) = best
    available = (
        median_ratio >= POPULATION_MIN_MEDIAN_RATIO
        and explained >= POPULATION_MIN_EXPLAINED_VARIANCE
    )
    reason = ("two_supported_thickness_populations" if available else
              "population_separation_below_threshold")
    evidence = ThicknessPopulationEvidence(
        available=available,
        split_threshold_px=boundary,
        thinner_pixel_count=thinner_count,
        thicker_pixel_count=thicker_count,
        thinner_fraction=thinner_count / values.size,
        thicker_fraction=thicker_count / values.size,
        thinner_median_px=thin_median,
        thicker_median_px=thick_median,
        median_ratio=median_ratio,
        explained_variance=explained,
        reason=reason,
    )
    if not available:
        return evidence, empty, empty.copy()
    thinner_mask = inside & (thickness_px < boundary)
    thicker_mask = inside & ~thinner_mask
    return evidence, thinner_mask, thicker_mask


def _circular_branch_runs(occupied: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Return stable circular runs after bridging one empty angular bin."""
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
        if len(run) >= OVERLAP_MIN_BRANCH_BINS:
            runs.append(tuple(run))
    return tuple(runs)


def _run_angle_deg(run: tuple[int, ...], bin_count: int) -> float:
    angles = (np.asarray(run, np.float64) + 0.5) * (2.0 * pi / bin_count)
    mean = atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    return degrees(mean) % 360.0


def _has_opposing_pair(angles_deg: tuple[float, ...]) -> bool:
    for first_index, first in enumerate(angles_deg):
        for second in angles_deg[first_index + 1:]:
            separation = abs(first - second) % 360.0
            separation = min(separation, 360.0 - separation)
            if abs(separation - 180.0) <= OVERLAP_OPPOSING_TOLERANCE_DEG:
                return True
    return False


def _annulus_branches(
    mask: np.ndarray,
    center_xy: tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
) -> tuple[tuple[float, ...], np.ndarray]:
    """Sample stable foreground arms around one excess-thickness island."""
    radii = np.linspace(
        max(1.0, inner_radius_px),
        max(inner_radius_px + 1.0, outer_radius_px),
        max(3, int(ceil(outer_radius_px - inner_radius_px)) + 1),
    )
    angle_values = (
        np.arange(OVERLAP_ANGLE_BINS, dtype=np.float32) + 0.5
    ) * (2.0 * pi / OVERLAP_ANGLE_BINS)
    xs = np.rint(
        center_xy[0] + np.cos(angle_values)[:, None] * radii[None, :]
    ).astype(np.int32)
    ys = np.rint(
        center_xy[1] + np.sin(angle_values)[:, None] * radii[None, :]
    ).astype(np.int32)
    valid = (
        (xs >= 0) & (xs < mask.shape[1])
        & (ys >= 0) & (ys < mask.shape[0])
    )
    samples = np.zeros(xs.shape, bool)
    samples[valid] = np.asarray(mask, bool)[ys[valid], xs[valid]]
    occupied = samples.mean(axis=1) >= 0.5
    runs = _circular_branch_runs(occupied)
    angles = tuple(_run_angle_deg(run, OVERLAP_ANGLE_BINS) for run in runs)
    return angles, occupied


def _boundary_safe_evidence_mask(
    mask: np.ndarray,
    thickness_px: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    image_origin_uv: tuple[int, int],
    frame_shape: tuple[int, int] | None,
    touches_frame: bool,
) -> np.ndarray:
    """Exclude clipped-edge pixels whose supporting disk is unobservable."""
    inside = np.asarray(mask) != 0
    if not touches_frame or frame_shape is None:
        return inside
    x, y, width, height = bbox_xywh
    origin_u, origin_v = image_origin_uv
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


def component_thickness_evidence(
    component_id: int,
    bbox_xywh: tuple[int, int, int, int],
    image_origin_uv: tuple[int, int],
    touches_frame: bool,
    mask: np.ndarray,
    thickness_px: np.ndarray,
    *,
    closed_hole_count: int = 0,
    display_percentile: float = 99.0,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[ComponentThicknessEvidence, np.ndarray]:
    """Assess a component using thickness contrast plus multi-arm geometry.

    Thickness peaks are proposals only.  A proposal is flagged when its
    surrounding annulus contains at least three stable arms.  A component with
    two or more already-observed closed apertures is also flagged when it has a
    component-local thickness excess.  The aperture count is supporting mask
    structure, not a gate label; the result remains an experimental candidate.
    """
    inside = np.asarray(mask) != 0
    evidence_inside = _boundary_safe_evidence_mask(
        inside,
        thickness_px,
        bbox_xywh,
        image_origin_uv,
        frame_shape,
        touches_frame,
    )
    values = np.asarray(thickness_px, np.float32)[evidence_inside]
    foreground_px = int(np.count_nonzero(inside))
    core_mask = np.zeros(inside.shape, bool)
    population, _, _ = segment_thickness_populations(
        evidence_inside, thickness_px)
    if not values.size:
        record = ComponentThicknessEvidence(
            component_id=component_id,
            bbox_xywh=bbox_xywh,
            touches_frame=touches_frame,
            closed_hole_count=closed_hole_count,
            foreground_px=foreground_px,
            thickness_reference_px=0.0,
            thickness_display_cap_px=0.0,
            thickness_p50_px=0.0,
            thickness_p90_px=0.0,
            thickness_p95_px=0.0,
            thickness_max_px=0.0,
            peak_ratio=0.0,
            excess_pixel_count=0,
            excess_island_count=0,
            minimum_excess_island_area_px=0,
            assessment="not_candidate",
            experimental_overlap_candidate=False,
            evidence_scope=("frame_edge_clipped_partial" if touches_frame
                            else "complete_component"),
            routing_eligible=False,
            thickness_split_ready=False,
            population=population,
            reason=("no_boundary_safe_evidence" if foreground_px
                    else "empty_component"),
            junctions=(),
        )
        return record, core_mask

    reference = max(float(np.percentile(values, 50)), 1e-6)
    display_cap = max(float(np.percentile(values, display_percentile)), 1e-6)
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))
    maximum = float(values.max())
    ratio_map = np.zeros(inside.shape, np.float32)
    ratio_map[evidence_inside] = thickness_px[evidence_inside] / reference
    excess = evidence_inside & (ratio_map >= OVERLAP_EXCESS_RATIO)
    excess_pixel_count = int(np.count_nonzero(excess))
    junctions: list[ThicknessJunctionEvidence] = []
    island_count = 0
    minimum_area = max(
        3,
        int(ceil(OVERLAP_MIN_EXCESS_AREA_SCALE * reference * reference)),
    )
    supported_excess = np.zeros(inside.shape, bool)
    observed_supported_excess = np.zeros(inside.shape, bool)
    if np.any(excess):
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            excess.astype(np.uint8), connectivity=8)
        island_count = label_count - 1
        for island_id in range(1, label_count):
            area = int(stats[island_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            island = labels == island_id
            supported_excess |= island
            ys, xs = np.nonzero(island)
            weights = np.maximum(
                ratio_map[island] - OVERLAP_EXCESS_RATIO, 1e-3)
            center_x = float(np.average(xs, weights=weights))
            center_y = float(np.average(ys, weights=weights))
            equivalent_radius = sqrt(area / pi)
            inner_radius = equivalent_radius + 0.50 * reference
            outer_radius = equivalent_radius + 1.00 * reference
            branch_angles, occupied = _annulus_branches(
                inside,
                (center_x, center_y),
                inner_radius,
                outer_radius,
            )
            opposing = _has_opposing_pair(branch_angles)
            branch_runs = _circular_branch_runs(occupied)
            geometric_candidate = len(branch_angles) >= OVERLAP_MIN_BRANCHES
            peak_ratio = float(ratio_map[island].max())
            origin_u, origin_v = image_origin_uv
            center_u = float(origin_u + center_x)
            center_v = float(origin_v + center_y)
            if not touches_frame:
                support_fully_observed = True
            elif frame_shape is None:
                support_fully_observed = False
            else:
                frame_height, frame_width = frame_shape
                support_fully_observed = (
                    center_u - outer_radius >= 0
                    and center_v - outer_radius >= 0
                    and center_u + outer_radius < frame_width
                    and center_v + outer_radius < frame_height
                )
            if support_fully_observed:
                observed_supported_excess |= island
            candidate = geometric_candidate and support_fully_observed
            junctions.append(ThicknessJunctionEvidence(
                center_uv=(
                    int(round(center_u)),
                    int(round(center_v)),
                ),
                center_local_xy=(round(center_x, 3), round(center_y, 3)),
                peak_ratio=peak_ratio,
                excess_area_px=area,
                excess_area_scale=float(area / (reference * reference)),
                annulus_inner_radius_px=round(inner_radius, 3),
                annulus_outer_radius_px=round(outer_radius, 3),
                occupied_angle_bins=tuple(
                    int(index) for index in np.flatnonzero(occupied)),
                branch_count=len(branch_angles),
                branch_angles_deg=tuple(round(value, 2)
                                        for value in branch_angles),
                branch_width_bins=tuple(len(run) for run in branch_runs),
                opposing_pair=opposing,
                support_fully_observed=support_fully_observed,
                candidate=candidate,
            ))
            if candidate:
                core_mask |= island

    junction_candidate = any(junction.candidate for junction in junctions)
    aperture_supported_candidate = (
        closed_hole_count >= 2
        and any(junction.support_fully_observed for junction in junctions)
    )
    candidate = junction_candidate or aperture_supported_candidate
    if aperture_supported_candidate and not junction_candidate:
        core_mask |= observed_supported_excess
    if junction_candidate:
        assessment = ("candidate_clipped_review_only" if touches_frame
                      else "candidate")
        reason = "thickness_excess_with_multi_arm_junction"
    elif aperture_supported_candidate:
        assessment = ("candidate_clipped_review_only" if touches_frame
                      else "candidate")
        reason = "thickness_excess_with_multiple_apertures"
    elif touches_frame:
        assessment = "indeterminate"
        reason = ("frame_edge_clipped_support_incomplete" if junctions
                  else "frame_edge_clipped_without_supported_proposal")
    elif not excess_pixel_count:
        assessment = "not_candidate"
        reason = "no_component_local_thickness_excess"
    elif not junctions:
        assessment = "not_candidate"
        reason = "thickness_excess_below_scale_support"
    else:
        assessment = "not_candidate"
        reason = "thickness_excess_without_multi_arm_junction"
    record = ComponentThicknessEvidence(
        component_id=component_id,
        bbox_xywh=bbox_xywh,
        touches_frame=touches_frame,
        closed_hole_count=closed_hole_count,
        foreground_px=foreground_px,
        thickness_reference_px=reference,
        thickness_display_cap_px=display_cap,
        thickness_p50_px=reference,
        thickness_p90_px=p90,
        thickness_p95_px=p95,
        thickness_max_px=maximum,
        peak_ratio=maximum / reference,
        excess_pixel_count=excess_pixel_count,
        excess_island_count=island_count,
        minimum_excess_island_area_px=minimum_area,
        assessment=assessment,
        experimental_overlap_candidate=candidate,
        evidence_scope=("frame_edge_clipped_partial" if touches_frame
                        else "complete_component"),
        routing_eligible=(candidate and not touches_frame),
        thickness_split_ready=(candidate and population.available),
        population=population,
        reason=reason,
        junctions=tuple(junctions),
    )
    return record, core_mask


def _l2_profile_assignment(
    mask: np.ndarray,
    local_radius: np.ndarray,
    minimum_support_ratio: float,
) -> np.ndarray:
    assignment = np.full(mask.shape, -1, np.int16)
    inside = np.asarray(mask) != 0
    assignment[inside] = -3
    for index, profile_radius in enumerate(PROFILE_RADII):
        supported = inside & (
            local_radius / float(profile_radius) >= minimum_support_ratio)
        assignment[supported] = index
    return assignment


def _square_profile_assignment(
    mask: np.ndarray,
    minimum_support_ratio: float,
) -> np.ndarray:
    inside = (np.asarray(mask) != 0).astype(np.uint8)
    assignment = np.full(mask.shape, -1, np.int16)
    assignment[inside != 0] = -3
    maximum_required = int(ceil(max(PROFILE_RADII) * minimum_support_ratio))
    padded = np.pad(inside, maximum_required, mode="constant")
    for index, profile_radius in enumerate(PROFILE_RADII):
        required = max(1, int(ceil(
            profile_radius * minimum_support_ratio)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * required + 1, 2 * required + 1))
        centers = cv2.erode(
            padded, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        covered = cv2.dilate(
            centers, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        covered = covered[
            maximum_required:maximum_required + inside.shape[0],
            maximum_required:maximum_required + inside.shape[1],
        ]
        assignment[(inside != 0) & (covered != 0)] = index
    return assignment


def _area_profile_index(area_px: np.ndarray | float) -> np.ndarray:
    """Resolve area evidence through the current ordered production limits."""
    areas = np.asarray(area_px, np.float64)
    result = np.full(areas.shape, len(
        DEFAULT_DENSITY_CONFIGURATION.profiles) - 1, np.int16)
    unresolved = np.ones(areas.shape, bool)
    for index, profile in enumerate(DEFAULT_DENSITY_CONFIGURATION.profiles):
        limit = profile.maximum_component_area_px
        if limit is None:
            result[unresolved] = index
            break
        selected = unresolved & (areas <= limit)
        result[selected] = index
        unresolved &= ~selected
    return result


def equivalent_area_profile_assignment(
    mask: np.ndarray,
    thickness_px: np.ndarray,
    component_area_px: int,
    *,
    maximum_profile_delta: int = 2,
) -> np.ndarray:
    """Select local profiles from scale-equivalent area, bounded for review.

    The production deck is calibrated by component area.  Under uniform image
    scaling, area changes with the square of linear wall thickness, so local
    thickness can be translated back into the deck's native evidence domain.
    """
    inside = np.asarray(mask) != 0
    assignment = np.full(inside.shape, -1, np.int16)
    values = np.asarray(thickness_px, np.float32)[inside]
    if not values.size:
        return assignment
    reference = max(float(np.median(values)), 1e-6)
    equivalent_area = component_area_px * np.square(values / reference)
    selected = _area_profile_index(equivalent_area)
    baseline = int(_area_profile_index(float(component_area_px)))
    selected = np.clip(
        selected,
        max(0, baseline - maximum_profile_delta),
        min(len(DEFAULT_DENSITY_CONFIGURATION.profiles) - 1,
            baseline + maximum_profile_delta),
    )
    assignment[inside] = selected
    return assignment


def assign_profiles(
    mask: np.ndarray,
    local_radius: np.ndarray,
    *,
    assignment_mode: str,
    minimum_support_ratio: float,
    thickness_px: np.ndarray | None = None,
    component_area_px: int | None = None,
) -> np.ndarray:
    """Return experimental profile indices, with -3 for unsupported pixels."""
    if assignment_mode not in ASSIGNMENT_MODES:
        raise ValueError(f"unknown assignment mode: {assignment_mode}")
    if not np.isfinite(minimum_support_ratio) or minimum_support_ratio <= 0:
        raise ValueError("minimum_support_ratio must be finite and positive")
    if assignment_mode == "calibrated_thickness_bands":
        values = (2.0 * np.asarray(local_radius, np.float32)
                  if thickness_px is None else thickness_px)
        return thickness_band_assignment(mask, values)
    if assignment_mode == "equivalent_area_thickness":
        if thickness_px is None or component_area_px is None:
            raise ValueError(
                "equivalent-area assignment requires thickness and area")
        return equivalent_area_profile_assignment(
            mask, thickness_px, component_area_px)
    if assignment_mode == "l2_covering_disk":
        return _l2_profile_assignment(
            mask, local_radius, minimum_support_ratio)
    return _square_profile_assignment(mask, minimum_support_ratio)


def _component_slices(component):
    x, y, width, height = component.bbox_xywh
    origin_u, origin_v = component.image_origin_uv
    local_x = x - origin_u
    local_y = y - origin_v
    local = np.s_[local_y:local_y + height, local_x:local_x + width]
    frame = np.s_[y:y + height, x:x + width]
    return local, frame


def _seams(profile_index: np.ndarray) -> np.ndarray:
    valid = profile_index >= 0
    seams = np.zeros(profile_index.shape, bool)
    horizontal = (
        valid[:, 1:] & valid[:, :-1]
        & (profile_index[:, 1:] != profile_index[:, :-1])
    )
    vertical = (
        valid[1:, :] & valid[:-1, :]
        & (profile_index[1:, :] != profile_index[:-1, :])
    )
    seams[:, 1:] |= horizontal
    seams[:, :-1] |= horizontal
    seams[1:, :] |= vertical
    seams[:-1, :] |= vertical
    return seams


def _regional_percentile_masks(
    field: np.ndarray,
    profile_index: np.ndarray,
    component_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Threshold each connected same-profile region independently."""
    outputs = [np.zeros(field.shape, np.uint8) for _ in range(3)]
    for component_id in np.unique(component_labels):
        if component_id <= 0:
            continue
        owned = component_labels == component_id
        for profile_id in np.unique(profile_index[owned]):
            if profile_id < 0:
                continue
            selection = owned & (profile_index == profile_id) & (field > 0)
            count, labels = cv2.connectedComponents(
                selection.astype(np.uint8), connectivity=8)
            for region_id in range(1, count):
                region = labels == region_id
                values = field[region]
                if not values.size:
                    continue
                for output, percentile in zip(outputs, (70, 80, 90)):
                    threshold = float(np.percentile(values, percentile))
                    output[region & (field >= threshold)] = 1
    return tuple(outputs)


def analyze_frame(
    frame: FrameObservation,
    *,
    assignment_mode: str = "equivalent_area_thickness",
    minimum_support_ratio: float = 1.0,
    smoothing_sigma_px: float = 0.0,
    include_clipped_components: bool = True,
) -> ThicknessAnalysis:
    """Build isolated whole-frame evidence and an experimental field mosaic."""
    if frame.gated:
        raise ValueError(
            f"frame preprocessing was gated: {frame.rejection_reason}")
    height, width = frame.image_shape
    distance_map = np.zeros((height, width), np.float32)
    radius_map = np.zeros((height, width), np.float32)
    raw_thickness_map = np.zeros((height, width), np.float32)
    regional_thickness_map = np.zeros((height, width), np.float32)
    normalized_thickness_map = np.zeros((height, width), np.float32)
    relative_thickness_map = np.zeros((height, width), np.float32)
    center_map = np.zeros((height, width), bool)
    overlap_candidate_map = np.zeros((height, width), bool)
    overlap_core_map = np.zeros((height, width), bool)
    thinner_population_map = np.zeros((height, width), bool)
    thicker_population_map = np.zeros((height, width), bool)
    thickness_band_map = np.full((height, width), -1, np.int16)
    profile_map = np.full((height, width), -1, np.int16)
    adaptive_field = np.zeros((height, width), np.float32)
    component_evidence: list[ComponentThicknessEvidence] = []
    review_configuration = DensityBankConfiguration(
        calibration_version=DEFAULT_DENSITY_CONFIGURATION.calibration_version,
        ignore_frame_edge_clipped=False,
        profiles=DEFAULT_DENSITY_CONFIGURATION.profiles,
    )
    bank = DensityBank(frame, review_configuration)

    for component in frame.components:
        local_mask = component_mask(frame, component, stage="closed")
        local_distance, local_radius, local_centers = covering_local_radius(
            local_mask, smoothing_sigma_px=smoothing_sigma_px)
        local_thickness = np.zeros_like(local_radius)
        local_inside = local_mask != 0
        local_thickness[local_inside] = 2.0 * local_radius[local_inside]
        local_regional_thickness = regional_thickness(
            local_thickness, local_mask)
        evidence, local_overlap_cores = component_thickness_evidence(
            component.component_id,
            component.bbox_xywh,
            component.image_origin_uv,
            component.touches_frame,
            local_mask,
            local_regional_thickness,
            closed_hole_count=component.topology.closed_hole_count,
            frame_shape=frame.image_shape,
        )
        component_evidence.append(evidence)
        local_bands = thickness_band_assignment(
            local_mask, local_regional_thickness)
        local_assignment = assign_profiles(
            local_mask,
            local_radius,
            assignment_mode=assignment_mode,
            minimum_support_ratio=minimum_support_ratio,
            thickness_px=local_regional_thickness,
            component_area_px=component.area_px,
        )
        local_slice, frame_slice = _component_slices(component)
        owned = (
            frame.component_labels[frame_slice] == component.component_id)
        for target, local_values in (
            (distance_map, local_distance),
            (radius_map, local_radius),
            (raw_thickness_map, local_thickness),
            (regional_thickness_map, local_regional_thickness),
        ):
            target_crop = target[frame_slice]
            values_crop = local_values[local_slice]
            target_crop[owned] = values_crop[owned]
        center_crop = center_map[frame_slice]
        center_values = local_centers[local_slice]
        center_crop[owned] = center_values[owned]
        thickness_crop = local_regional_thickness[local_slice]
        local_values = local_regional_thickness[local_inside]
        display_cap = max(
            evidence.thickness_display_cap_px,
            float(np.percentile(local_values, 99)) if local_values.size else 0.0,
            1e-6,
        )
        reference = max(
            evidence.thickness_reference_px,
            float(np.median(local_values)) if local_values.size else 0.0,
            1e-6,
        )
        normalized_crop = normalized_thickness_map[frame_slice]
        normalized_crop[owned] = np.clip(
            thickness_crop[owned] / display_cap,
            0.0,
            1.0,
        )
        relative_crop = relative_thickness_map[frame_slice]
        relative_crop[owned] = (
            thickness_crop[owned] / reference)
        overlap_core_crop = overlap_core_map[frame_slice]
        local_core_crop = local_overlap_cores[local_slice]
        overlap_core_crop[owned] = local_core_crop[owned]
        if evidence.experimental_overlap_candidate:
            candidate_crop = overlap_candidate_map[frame_slice]
            candidate_crop[owned] = True
        if (evidence.thickness_split_ready
                and evidence.population.split_threshold_px is not None):
            boundary = evidence.population.split_threshold_px
            local_thinner = local_inside & (
                local_regional_thickness < boundary)
            local_thicker = local_inside & ~local_thinner
            thinner_crop = thinner_population_map[frame_slice]
            thicker_crop = thicker_population_map[frame_slice]
            thinner_values = local_thinner[local_slice]
            thicker_values = local_thicker[local_slice]
            thinner_crop[owned] = thinner_values[owned]
            thicker_crop[owned] = thicker_values[owned]
        band_crop = thickness_band_map[frame_slice]
        local_band_crop = local_bands[local_slice]
        band_crop[owned] = local_band_crop[owned]
        profile_crop = profile_map[frame_slice]
        if component.touches_frame and not include_clipped_components:
            profile_crop[owned] = -2
            continue
        assignment_crop = local_assignment[local_slice]
        profile_crop[owned] = assignment_crop[owned]
        adaptive_crop = adaptive_field[frame_slice]
        for profile_index, profile in enumerate(
                DEFAULT_DENSITY_CONFIGURATION.profiles):
            selected = owned & (assignment_crop == profile_index)
            if not np.any(selected):
                continue
            evidence = bank.get(component, profile.profile_id)
            field_crop = evidence.final_field[local_slice]
            adaptive_crop[selected] = field_crop[selected]

    inside = frame.closed_mask != 0
    values = raw_thickness_map[inside]
    adaptive_p70, adaptive_p80, adaptive_p90 = _regional_percentile_masks(
        adaptive_field, profile_map, frame.component_labels)
    counts = {
        profile.profile_id: int(np.count_nonzero(profile_map == index))
        for index, profile in enumerate(
            DEFAULT_DENSITY_CONFIGURATION.profiles)
    }
    thickness_band_counts = {
        f"thickness_band_{index + 1:02d}": int(np.count_nonzero(
            thickness_band_map == index))
        for index in range(len(THICKNESS_PROFILE_BOUNDARIES_PX) + 1)
    }
    metrics = {
        "component_count": len(frame.components),
        "clipped_component_count": sum(
            component.touches_frame for component in frame.components),
        "foreground_px": int(np.count_nonzero(inside)),
        "unsupported_px": int(np.count_nonzero(profile_map == -3)),
        "clipped_px": int(np.count_nonzero(profile_map == -2)),
        "local_thickness_p50_px": (
            float(np.percentile(values, 50)) if values.size else 0.0),
        "local_thickness_p90_px": (
            float(np.percentile(values, 90)) if values.size else 0.0),
        "local_thickness_max_px": (
            float(values.max()) if values.size else 0.0),
        "assignment_mode": assignment_mode,
        "minimum_support_ratio": minimum_support_ratio,
        "distance_ridge_smoothing_sigma_px": smoothing_sigma_px,
        "regional_thickness_smoothing_sigma_px": (
            REGIONAL_THICKNESS_SMOOTHING_SIGMA_PX),
        "thickness_profile_boundaries_px": list(
            THICKNESS_PROFILE_BOUNDARIES_PX),
        "include_clipped_components": include_clipped_components,
        "overlap_candidate_rule": {
            "status": "experimental_discovery_only",
            "excess_ratio": OVERLAP_EXCESS_RATIO,
            "minimum_excess_island_area": (
                f"max(3, ceil({OVERLAP_MIN_EXCESS_AREA_SCALE} * "
                "component_median_thickness_px^2))"),
            "angular_bins": OVERLAP_ANGLE_BINS,
            "minimum_branch_bins": OVERLAP_MIN_BRANCH_BINS,
            "minimum_branches": OVERLAP_MIN_BRANCHES,
            "opposing_tolerance_deg": OVERLAP_OPPOSING_TOLERANCE_DEG,
            "opposing_pair_role": "reported_support_not_required",
            "multiple_aperture_support": (
                "closed_hole_count>=2 with thickness excess"),
            "clipped_policy": (
                "review candidate only when the complete supporting annulus "
                "is inside recorded frame bounds; never routing eligible"),
        },
        "overlap_candidate_count": sum(
            item.experimental_overlap_candidate
            for item in component_evidence),
        "clipped_overlap_candidate_count": sum(
            item.experimental_overlap_candidate and item.touches_frame
            for item in component_evidence),
        "thickness_split_ready_count": sum(
            item.thickness_split_ready for item in component_evidence),
        "overlap_candidates": [
            asdict(item) for item in component_evidence
            if item.experimental_overlap_candidate
        ],
        "component_thickness_evidence": [
            asdict(item) for item in component_evidence
        ],
        "thickness_band_pixel_counts": thickness_band_counts,
        "profile_pixel_counts": counts,
    }
    return ThicknessAnalysis(
        distance_map,
        radius_map,
        raw_thickness_map,
        regional_thickness_map,
        normalized_thickness_map,
        relative_thickness_map,
        center_map,
        overlap_candidate_map,
        overlap_core_map,
        thinner_population_map,
        thicker_population_map,
        thickness_band_map,
        profile_map,
        _seams(profile_map),
        adaptive_field,
        adaptive_p70,
        adaptive_p80,
        adaptive_p90,
        tuple(component_evidence),
        metrics,
    )


def _heat(
    values: np.ndarray,
    mask: np.ndarray,
    percentile: float,
    *,
    unit: str,
) -> tuple[np.ndarray, float]:
    positive = values[(mask != 0) & np.isfinite(values) & (values > 0)]
    cap = float(np.percentile(positive, percentile)) if positive.size else 1.0
    cap = max(cap, 1e-9)
    normalized = np.rint(np.clip(values / cap, 0, 1) * 255).astype(np.uint8)
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[mask == 0] = 0
    bar_width = min(240, max(100, output.shape[1] // 3))
    gradient = np.arange(256, dtype=np.uint8)[None, :]
    gradient = cv2.resize(gradient, (bar_width, 10), interpolation=cv2.INTER_LINEAR)
    gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    y0 = output.shape[0] - 18
    output[y0:y0 + 10, 10:10 + bar_width] = gradient
    cv2.putText(output, f"0 .. {cap:.2f} {unit} (P{percentile:g})",
                (14 + bar_width, output.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1,
                cv2.LINE_AA)
    return output, cap


def _profile_colors(
    profile_index: np.ndarray,
    seams: np.ndarray,
    *,
    legend: str = "scale_01 -> scale_10 | white=seam",
) -> np.ndarray:
    ramp = cv2.applyColorMap(
        np.linspace(20, 245, 10, dtype=np.uint8)[None, :],
        cv2.COLORMAP_TURBO,
    )[0]
    output = np.zeros((*profile_index.shape, 3), np.uint8)
    for index, color in enumerate(ramp):
        output[profile_index == index] = color
    output[profile_index == -2] = (105, 105, 105)
    output[profile_index == -3] = (210, 0, 210)
    output[seams] = (255, 255, 255)
    segment_width = max(12, min(28, (output.shape[1] - 130) // 10))
    y0 = output.shape[0] - 18
    for index, color in enumerate(ramp):
        x0 = 10 + index * segment_width
        output[y0:y0 + 10, x0:x0 + segment_width] = color
    cv2.putText(output, legend,
                (18 + 10 * segment_width, output.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 235, 235), 1,
                cv2.LINE_AA)
    return output


def _component_colors(
    labels: np.ndarray,
    components,
    candidate_ids: frozenset[int],
    show_candidates: bool,
) -> np.ndarray:
    """Render deterministic connected-component identities in frame UV."""
    output = np.zeros((*labels.shape, 3), np.uint8)
    for component in components:
        hue = (int(component.component_id) * 47) % 180
        hsv = np.uint8([[[hue, 190, 225]]])
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        output[labels == component.component_id] = color
        if show_candidates and component.component_id in candidate_ids:
            component_mask = (labels == component.component_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 165, 255), 2)
    return output


def _relative_heat(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    cap_ratio: float = 2.0,
) -> np.ndarray:
    normalized = np.rint(
        np.clip(values / cap_ratio, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[mask == 0] = 0
    return output


def _black_selection_over_white_mask(
    mask: np.ndarray,
    selection: np.ndarray,
) -> np.ndarray:
    """Show one inferred population as black over the exact white mask."""
    inside = np.asarray(mask) != 0
    selected = np.asarray(selection, bool) & inside
    output = np.full((*inside.shape, 3), 34, np.uint8)
    output[inside] = (255, 255, 255)
    output[selected] = (0, 0, 0)
    return output


def _binary_evidence(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        (np.asarray(mask) != 0).astype(np.uint8) * 255,
        cv2.COLOR_GRAY2BGR,
    )


def _overlap_overlay(
    source_bgr: np.ndarray,
    frame: FrameObservation,
    analysis: ThicknessAnalysis,
    show_candidates: bool,
) -> np.ndarray:
    output = source_bgr.copy()
    if not show_candidates:
        cv2.putText(
            output,
            "experimental candidate display disabled",
            (10, max(20, output.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 180, 195),
            1,
            cv2.LINE_AA,
        )
        return output
    core_contours, _ = cv2.findContours(
        analysis.overlap_candidate_cores.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, core_contours, -1, (255, 255, 255), 1)
    for evidence in analysis.component_evidence:
        if (evidence.experimental_overlap_candidate
                and evidence.touches_frame):
            color = (255, 0, 255)
        elif evidence.assessment == "indeterminate":
            color = (115, 115, 115)
        elif evidence.experimental_overlap_candidate:
            color = (0, 165, 255)
        else:
            continue
        component_mask_image = (
            frame.component_labels == evidence.component_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
        x, y, width, height = evidence.bbox_xywh
        cv2.putText(
            output,
            (f"c{evidence.component_id} clipped candidate / review only" if
             evidence.experimental_overlap_candidate and evidence.touches_frame
             else f"c{evidence.component_id} candidate" if
             evidence.experimental_overlap_candidate else
             f"c{evidence.component_id} clipped / indeterminate"),
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        for junction in evidence.junctions:
            if not junction.candidate:
                continue
            cv2.drawMarker(
                output,
                junction.center_uv,
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=11,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
    return output


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = 44
    output = np.zeros((image.shape[0] + header, image.shape[1], 3), np.uint8)
    output[header:] = image
    cv2.putText(output, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, subtitle, (8, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (165, 175, 190), 1, cv2.LINE_AA)
    return output


def render_analysis(
    source_bgr: np.ndarray,
    frame: FrameObservation,
    analysis: ThicknessAnalysis,
    *,
    source_label: str,
    heat_cap_percentile: float,
    show_overlap_candidate_flags: bool = False,
) -> np.ndarray:
    """Render a same-UV 4x4 waterfall for visual discovery."""
    mask = (frame.closed_mask != 0).astype(np.uint8)
    distance_heat, distance_cap = _heat(
        analysis.distance_to_background_px,
        mask,
        heat_cap_percentile,
        unit="px to background",
    )
    thickness_heat, thickness_cap = _heat(
        analysis.covering_local_thickness_px,
        mask,
        heat_cap_percentile,
        unit="px thickness",
    )
    regional_heat, regional_cap = _heat(
        analysis.regional_local_thickness_px,
        mask,
        heat_cap_percentile,
        unit="px regional thickness",
    )
    candidate_ids = frozenset(
        evidence.component_id for evidence in analysis.component_evidence
        if evidence.experimental_overlap_candidate)
    component_image = _component_colors(
        frame.component_labels,
        frame.components,
        candidate_ids,
        show_overlap_candidate_flags,
    )
    relative_heat = _relative_heat(
        analysis.component_relative_thickness, mask)
    overlap_overlay = _overlap_overlay(
        source_bgr,
        frame,
        analysis,
        show_overlap_candidate_flags,
    )
    thickness_band_image = _profile_colors(
        analysis.thickness_band_index,
        _seams(analysis.thickness_band_index),
        legend="thin band 1 -> thick band 10 | white=boundary",
    )
    profile_image = _profile_colors(
        analysis.adaptive_profile_index, analysis.profile_seams)
    adaptive_heat, adaptive_cap = _heat(
        analysis.adaptive_final_field,
        mask & (analysis.adaptive_profile_index >= 0),
        heat_cap_percentile,
        unit="field",
    )
    binary = _binary_evidence(mask)
    thinner_knockout = _black_selection_over_white_mask(
        mask, analysis.thinner_population_mask)
    thicker_knockout = _black_selection_over_white_mask(
        mask, analysis.thicker_population_mask)
    mode = analysis.metrics["assignment_mode"]
    ratio = analysis.metrics["minimum_support_ratio"]
    clipped_policy = ("included" if analysis.metrics[
        "include_clipped_components"] else "excluded")
    profiles = DEFAULT_DENSITY_CONFIGURATION.profiles
    panels = (
        _panel(source_bgr, "recorded source frame", source_label),
        _panel(
            binary,
            "FrameObservation.closed_mask",
            f"components={len(frame.components)}; exact post-close foreground",
        ),
        _panel(
            component_image,
            "FrameObservation.component_labels",
            "component-segmented in the same frame UV; deterministic colors",
        ),
        _panel(
            distance_heat,
            "distance_to_background_px",
            f"component-local padded precise L2; display cap={distance_cap:.2f}",
        ),
        _panel(
            thickness_heat,
            "covering_local_thickness_px",
            f"2 x largest covering disk radius at every mask pixel; cap={thickness_cap:.2f}",
        ),
        _panel(
            regional_heat,
            "regional_local_thickness_px",
            f"component-constrained normalized Gaussian; sigma=1 px; cap={regional_cap:.2f}",
        ),
        _panel(
            relative_heat,
            "component_relative_thickness",
            "regional thickness / that component's median; fixed 0..2x scale",
        ),
        _panel(
            thickness_band_image,
            "empirical ten-band thickness map",
            "cuts=6.25/8/10/12/14/16/18/21/25 px; equal values stay together",
        ),
        _panel(
            thinner_knockout,
            "thinner thickness population",
            "candidate pixels are black over the exact white closed mask",
        ),
        _panel(
            thicker_knockout,
            "thicker thickness population",
            "candidate pixels are black over the exact white closed mask",
        ),
        _panel(
            overlap_overlay,
            "experimental overlap-candidate evidence",
            (f"flag display={'on' if show_overlap_candidate_flags else 'off'}; "
             f"candidates={len(candidate_ids)}; orange=complete; "
             "magenta=clipped review-only; white=core; gray=indeterminate"),
        ),
        _panel(
            profile_image,
            "experimental adaptive profile_id map",
            f"mode={mode}; support ratio={ratio:g}; "
            f"clipped={clipped_policy}; magenta=unsupported; gray=excluded",
        ),
        _panel(
            adaptive_heat,
            "adaptive composite of DensityEvidence.final_field",
            f"per-pixel selection; {profiles[0].profile_id} r{profiles[0].density_radius_px} -> "
            f"{profiles[-1].profile_id} r{profiles[-1].density_radius_px}; cap={adaptive_cap:.3f}",
        ),
        _panel(
            _binary_evidence(analysis.adaptive_p70_mask),
            "regional adaptive P70 evidence",
            "percentile recomputed per connected same-profile region",
        ),
        _panel(
            _binary_evidence(analysis.adaptive_p80_mask),
            "regional adaptive P80 evidence",
            "locally calibrated evidence; subset of regional P70",
        ),
        _panel(
            _binary_evidence(analysis.adaptive_p90_mask),
            "regional adaptive P90 evidence",
            "strongest locally calibrated evidence; subset of regional P80",
        ),
    )
    return np.vstack(tuple(
        np.hstack(panels[index:index + 4])
        for index in range(0, len(panels), 4)
    ))
