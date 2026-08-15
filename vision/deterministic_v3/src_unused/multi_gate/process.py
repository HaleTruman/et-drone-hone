"""Route adapter for topology-approved two-aperture multi-gate fitting."""

from __future__ import annotations

import cv2
import numpy as np

from ..configurations import DEFAULT_MULTI_GATE_CONFIGURATION
from ..density_bank import DensityBank
from ..preprocessing import component_mask
from ..quadrilateral import normalize_quadrilateral
from ..schema import (
    MULTI_GATE_ROUTE,
    MULTI_VOID_TOPOLOGY,
    QUADRILATERAL_CORNER_ORDER,
    ComponentObservation,
    FrameObservation,
    MultiGateApertureFit,
    MultiGateCandidateAssessment,
    MultiGateConfiguration,
    MultiGateResult,
    QuadrilateralEstimate,
    TopologyDecision,
)
from .aperture_fitting import ApertureSeed, fit_two_aperture_coverage
from .profile_selection import select_multi_gate_density_profile_id


FITTER_NAME = "multi_gate_two_aperture_coverage_adapter_v1"


def _points(values: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(point[0]), float(point[1]))
        for point in np.asarray(values).reshape(-1, 2)
    )


def _rejected_result(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
    density,
    configuration: MultiGateConfiguration,
    reason: str,
    identification: MultiGateCandidateAssessment | None,
) -> MultiGateResult:
    return MultiGateResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        topology_label=decision.topology_label,
        route=MULTI_GATE_ROUTE,
        fitter=FITTER_NAME,
        configuration_version=configuration.configuration_version,
        selected_density_profile=density.profile,
        identification_reason=(
            None if identification is None else identification.candidate_reason
        ),
        p70_threshold=float(density.p70_threshold),
        p90_threshold=float(density.p90_threshold),
        p70_evidence_points=int(np.count_nonzero(density.p70_mask)),
        p90_evidence_points=int(np.count_nonzero(density.p90_mask)),
        aperture_fits=(),
        residual_evidence_points=0,
        p70_added_pixel_count=0,
        union_covered_evidence_points=0,
        weighted_union_coverage=0.0,
        accepted=False,
        rejection_reason=reason,
    )


def _aperture_seeds(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
) -> tuple[ApertureSeed, ...]:
    nodes = {node.contour_id: node for node in frame.closed_contour_nodes}
    origin = np.asarray(component.image_origin_uv, np.float64)
    seeds = []
    for parent_id, child_id in decision.significant_parent_child_contour_ids:
        try:
            child = nodes[child_id]
        except KeyError as error:
            raise ValueError("multi_gate_aperture_contour_missing") from error
        if (
            child.component_id != component.component_id
            or child.parent_contour_id != parent_id
            or not child.is_hole
        ):
            raise ValueError("multi_gate_aperture_contour_ownership_mismatch")
        contour = np.asarray(child.points_uv, np.float64).copy()
        contour[:, 0, :] -= origin
        seeds.append(ApertureSeed(
            parent_contour_id=parent_id,
            child_contour_id=child_id,
            area_px2=float(child.area_px2),
            contour_local=contour.astype(np.float32),
        ))
    return tuple(sorted(
        seeds, key=lambda seed: (-seed.area_px2, seed.child_contour_id)
    ))


def process_multi_gate(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
    density_bank: DensityBank,
    configuration: MultiGateConfiguration = DEFAULT_MULTI_GATE_CONFIGURATION,
    identification: MultiGateCandidateAssessment | None = None,
) -> MultiGateResult:
    """Run the two-aperture fitter for one accepted ``multi_gate`` route."""
    if not any(owned is component for owned in frame.components):
        raise ValueError("component does not belong to frame")
    identity = (frame.frame_id, frame.sim_time_ns, component.component_id)
    if identity != (
        decision.frame_id,
        decision.sim_time_ns,
        decision.component_id,
    ):
        raise ValueError("topology decision belongs to another component")
    if (
        not decision.accepted
        or decision.topology_label != MULTI_VOID_TOPOLOGY
        or decision.route != MULTI_GATE_ROUTE
    ):
        raise ValueError("multi-gate fitter requires an accepted multi_gate route")
    if density_bank.frame is not frame:
        raise ValueError("density bank belongs to another frame")
    if configuration.minimum_p90_evidence_points < 4:
        raise ValueError("multi-gate minimum P90 evidence must be at least four")
    if identification is not None:
        if identity != (
            identification.frame_id,
            identification.sim_time_ns,
            identification.component_id,
        ):
            raise ValueError("multi-gate identification belongs to another component")

    profile_id = select_multi_gate_density_profile_id(component, configuration)
    density = density_bank.get(component, profile_id)
    pairs = decision.significant_parent_child_contour_ids
    if decision.closed_significant_holes != 2 or len(pairs) != 2:
        return _rejected_result(
            frame,
            component,
            decision,
            density,
            configuration,
            "multi_gate_fitter_requires_two_aperture_seeds",
            identification,
        )
    if identification is not None and not identification.fitter_ready:
        return _rejected_result(
            frame,
            component,
            decision,
            density,
            configuration,
            identification.fitter_readiness_reason,
            identification,
        )
    p90_count = int(np.count_nonzero(density.p90_mask))
    if p90_count < configuration.minimum_p90_evidence_points:
        return _rejected_result(
            frame,
            component,
            decision,
            density,
            configuration,
            "multi_gate_insufficient_p90_evidence",
            identification,
        )

    try:
        seeds = _aperture_seeds(frame, component, decision)
        coverage = fit_two_aperture_coverage(
            component_mask(frame, component, stage="closed"),
            density,
            seeds,
        )
    except (cv2.error, ValueError, RuntimeError, FloatingPointError,
            np.linalg.LinAlgError) as error:
        return _rejected_result(
            frame,
            component,
            decision,
            density,
            configuration,
            f"multi_gate_geometry_error:{type(error).__name__}",
            identification,
        )

    origin = np.asarray(component.image_origin_uv, np.float64)
    aperture_fits = []
    for gate_index, fit in enumerate(coverage.aperture_fits):
        initial = fit.initial_fit
        optimized = fit.optimized_fit
        aperture_uv = np.asarray(initial.aperture_corners, np.float64) + origin
        initial_uv = np.asarray(initial.corners, np.float64) + origin
        fitted_uv = np.asarray(optimized.corners, np.float64) + origin
        normalized, area = normalize_quadrilateral(
            fitted_uv, frame.image_shape
        )
        accepted = normalized is not None
        aperture_fits.append(MultiGateApertureFit(
            gate_index=gate_index,
            aperture_role="larger" if gate_index == 0 else "smaller",
            parent_contour_id=fit.seed.parent_contour_id,
            child_contour_id=fit.seed.child_contour_id,
            aperture_area_px2=fit.seed.area_px2,
            aperture_corners_uv=_points(aperture_uv),
            initial_corners_uv=_points(initial_uv),
            fitted_corners_uv=_points(fitted_uv),
            normalized_corners_uv=normalized,
            scale=float(optimized.scale),
            rotation_degrees=float(optimized.rotation_degrees),
            translation_xy_px=tuple(map(float, optimized.translation)),
            weighted_coverage=float(optimized.weighted_coverage),
            side_coverages=tuple(map(float, optimized.side_coverages)),
            evidence_point_count=fit.evidence_point_count,
            p70_added_pixel_count=int(cv2.countNonZero(optimized.additions)),
            quadrilateral_area_px2=area,
            accepted=accepted,
            rejection_reason=(
                None if accepted
                else "multi_gate_quadrilateral_validation_failed"
            ),
        ))
    accepted = len(aperture_fits) == 2 and all(
        fit.accepted for fit in aperture_fits
    )
    return MultiGateResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        topology_label=decision.topology_label,
        route=MULTI_GATE_ROUTE,
        fitter=FITTER_NAME,
        configuration_version=configuration.configuration_version,
        selected_density_profile=density.profile,
        identification_reason=(
            None if identification is None else identification.candidate_reason
        ),
        p70_threshold=float(density.p70_threshold),
        p90_threshold=float(density.p90_threshold),
        p70_evidence_points=int(np.count_nonzero(density.p70_mask)),
        p90_evidence_points=p90_count,
        aperture_fits=tuple(aperture_fits),
        residual_evidence_points=coverage.residual_evidence_points,
        p70_added_pixel_count=int(cv2.countNonZero(coverage.additions)),
        union_covered_evidence_points=int(
            np.count_nonzero(coverage.union_covered)
        ),
        weighted_union_coverage=coverage.weighted_union_coverage,
        accepted=accepted,
        rejection_reason=(
            None if accepted else "multi_gate_quadrilateral_validation_failed"
        ),
    )


def quadrilaterals_from_multi_gate(
    result: MultiGateResult,
) -> tuple[QuadrilateralEstimate, ...]:
    """Publish each fitted gate with a stable within-component identity."""
    return tuple(QuadrilateralEstimate(
        frame_id=result.frame_id,
        sim_time_ns=result.sim_time_ns,
        component_id=result.component_id,
        image_shape=result.image_shape,
        route=MULTI_GATE_ROUTE,
        fitter=result.fitter,
        selected_density_profile=result.selected_density_profile,
        corner_order=QUADRILATERAL_CORNER_ORDER,
        corners_uv=fit.normalized_corners_uv,
        p90_threshold=result.p90_threshold,
        p90_evidence_points=fit.evidence_point_count,
        area_px2=fit.quadrilateral_area_px2,
        accepted=fit.accepted,
        rejection_reason=fit.rejection_reason,
        gate_index=fit.gate_index,
    ) for fit in result.aperture_fits)
