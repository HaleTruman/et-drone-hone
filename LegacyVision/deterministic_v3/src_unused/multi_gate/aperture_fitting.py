"""Topology-seeded adapter around the pinned two-aperture coverage kernel.

The historical solver remains untouched.  This adapter deliberately supplies
the two authoritative aperture contours itself, preventing the fitter from
rediscovering or reclassifying topology from the component mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..schema import DensityEvidence
from . import overlapping_gate_aperture_solver as baseline


@dataclass(frozen=True, slots=True)
class ApertureSeed:
    parent_contour_id: int
    child_contour_id: int
    area_px2: float
    contour_local: np.ndarray


@dataclass(frozen=True, slots=True)
class ApertureCoverageFit:
    seed: ApertureSeed
    initial_fit: baseline.GateFit
    optimized_fit: baseline.CoverageCandidate
    evidence_point_count: int


@dataclass(frozen=True, slots=True)
class TwoApertureCoverageResult:
    aperture_fits: tuple[ApertureCoverageFit, ...]
    residual_evidence_points: int
    additions: np.ndarray
    union_covered: np.ndarray
    weighted_union_coverage: float


def _initial_fits_from_apertures(
    aperture_seeds: tuple[ApertureSeed, ...],
    seed_mask: np.ndarray,
    maximum_offset: int,
) -> tuple[baseline.GateFit, ...]:
    """Fit supplied aperture seeds without running mask-hole discovery."""
    evidence_field = (np.asarray(seed_mask) != 0).astype(np.float64)
    evidence_mask = (np.asarray(seed_mask) != 0).astype(np.uint8)
    fits = []
    prior_sides = []
    for aperture in aperture_seeds:
        fit = baseline.solve_gate(
            (aperture.area_px2, aperture.contour_local),
            evidence_field,
            evidence_mask,
            maximum_offset,
            tuple(prior_sides),
        )
        if fit is None:
            raise RuntimeError("unable_to_fit_aperture_seed")
        fits.append(fit)
        prior_sides.extend(fit.sides)
    return tuple(fits)


def fit_two_aperture_coverage(
    component_mask: np.ndarray,
    density: DensityEvidence,
    aperture_seeds: tuple[ApertureSeed, ...],
) -> TwoApertureCoverageResult:
    """Fit larger then smaller aperture against cached P90/P70 evidence."""
    if len(aperture_seeds) != 2:
        raise ValueError("two aperture seeds are required")
    if aperture_seeds[0].area_px2 < aperture_seeds[1].area_px2:
        raise ValueError("aperture seeds must be ordered larger first")
    mask = (np.asarray(component_mask) != 0).astype(np.uint8)
    field = np.asarray(density.final_field)
    seed = (np.asarray(density.p90_mask) != 0).astype(np.uint8)
    limit = (np.asarray(density.p70_mask) != 0).astype(np.uint8)
    if mask.shape != field.shape or mask.shape != seed.shape:
        raise ValueError("component and density evidence shapes must match")
    maximum_offset = max(3, int(density.profile.ridge_radius_px) + 2)
    initial = _initial_fits_from_apertures(
        aperture_seeds, seed, maximum_offset
    )

    yy, xx = np.nonzero(seed)
    points = np.column_stack((xx, yy)).astype(np.float64)
    weights = field[yy, xx].astype(np.float64)
    larger, _ = baseline.optimize_coverage_candidate(
        seed, limit, initial[0], points, weights
    )
    covered_by_larger = (
        baseline.distances_to_quadrilateral(points, larger.corners)
        <= baseline.COVERAGE_TOLERANCE_PX
    )
    residual = ~covered_by_larger
    if not np.any(residual):
        residual = np.ones(len(points), dtype=bool)
    smaller, _ = baseline.optimize_coverage_candidate(
        seed,
        limit,
        initial[1],
        points[residual],
        weights[residual],
    )

    additions = cv2.bitwise_or(larger.additions, smaller.additions)
    union_distances = np.minimum(
        baseline.distances_to_quadrilateral(points, larger.corners),
        baseline.distances_to_quadrilateral(points, smaller.corners),
    )
    union_covered = union_distances <= baseline.COVERAGE_TOLERANCE_PX
    return TwoApertureCoverageResult(
        aperture_fits=(
            ApertureCoverageFit(
                aperture_seeds[0], initial[0], larger, len(points)
            ),
            ApertureCoverageFit(
                aperture_seeds[1], initial[1], smaller,
                int(np.count_nonzero(residual)),
            ),
        ),
        residual_evidence_points=int(np.count_nonzero(residual)),
        additions=additions,
        union_covered=union_covered,
        weighted_union_coverage=float(
            weights[union_covered].sum() / max(weights.sum(), 1e-9)
        ),
    )
