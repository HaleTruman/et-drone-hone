"""Frame-scoped adapter for independent standard-gate processing."""

from __future__ import annotations

import cv2
import numpy as np

from ..configurations import DEFAULT_STANDARD_GATE_CONFIGURATION
from ..density_bank import DensityBank
from ..preprocessing import component_mask
from ..quadrilateral import normalize_quadrilateral
from ..schema import (
    QUADRILATERAL_CORNER_ORDER,
    STANDARD_ROUTE,
    ComponentObservation,
    ContourNodeEvidence,
    FrameObservation,
    QuadrilateralEstimate,
    StandardGateConfiguration,
    StandardGateResult,
)
from .quadrilateral_fitter import FITTER_NAME, fit_standard_quadrilateral


def _significant_pairs(
    frame: FrameObservation,
    component: ComponentObservation,
    configuration: StandardGateConfiguration,
) -> tuple[
    tuple[tuple[int, int], ...], dict[int, ContourNodeEvidence]
]:
    nodes = {node.contour_id: node for node in frame.closed_contour_nodes}
    if len(nodes) != len(frame.closed_contour_nodes):
        raise ValueError("closed contour IDs must be unique")
    pairs = []
    for parent_id, child_id in component.contours.parent_child_contour_ids:
        parent = nodes.get(parent_id)
        child = nodes.get(child_id)
        if parent is None or child is None:
            raise ValueError("component references an unknown contour node")
        if (parent.component_id != component.component_id
                or child.component_id != component.component_id):
            raise ValueError("component references a foreign contour node")
        if child.area_px2 >= configuration.minimum_hole_area_px2:
            pairs.append((parent_id, child_id))
    return tuple(pairs), nodes


def _rejected_result(
    frame: FrameObservation,
    component: ComponentObservation,
    profile,
    configuration: StandardGateConfiguration,
    reason: str,
    *,
    pairs=(),
    density=None,
    fit=None,
) -> StandardGateResult:
    return StandardGateResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        route=STANDARD_ROUTE,
        fitter=FITTER_NAME,
        configuration_version=configuration.configuration_version,
        selected_density_profile=profile,
        significant_parent_child_contour_ids=tuple(pairs),
        p70_threshold=(0.0 if density is None
                       else float(density.p70_threshold)),
        p70_evidence_points=(0 if density is None
                             else int(np.count_nonzero(density.p70_mask))),
        p90_threshold=(0.0 if density is None
                       else float(density.p90_threshold)),
        p90_evidence_points=(0 if density is None
                             else int(np.count_nonzero(density.p90_mask))),
        initial_corners_uv=(None if fit is None
                            else fit.initial_corners_uv),
        fitted_corners_uv=(None if fit is None else fit.fitted_corners_uv),
        side_evidence=(() if fit is None else fit.side_evidence),
        aperture_center_offset_ratio=(
            None if fit is None else fit.aperture_center_offset_ratio),
        quadrilateral_area_px2=(None if fit is None else fit.area_px2),
        p70_coverage_ratio=(0.0 if fit is None
                            else fit.p70_coverage_ratio),
        fit_confidence=(0.0 if fit is None else fit.fit_confidence),
        high_confidence_threshold=configuration.minimum_fit_confidence,
        high_confidence=False,
        accepted=False,
        rejection_reason=reason,
    )


def process_standard_gate(
    frame: FrameObservation,
    component: ComponentObservation,
    density_bank: DensityBank,
    configuration: StandardGateConfiguration =
        DEFAULT_STANDARD_GATE_CONFIGURATION,
) -> StandardGateResult:
    """Evaluate one live component without requiring a topology label."""
    if not any(owned is component for owned in frame.components):
        raise ValueError("component does not belong to frame")
    if density_bank.frame is not frame:
        raise ValueError("density bank belongs to another frame")
    profile = density_bank.recommended_standard_profile(component)
    if component.touches_frame:
        return _rejected_result(
            frame, component, profile, configuration,
            "standard_frame_edge_clipped",
        )
    if not density_bank.includes_component(component):
        return _rejected_result(
            frame, component, profile, configuration,
            "standard_density_ineligible",
        )

    pairs, nodes = _significant_pairs(
        frame, component, configuration
    )
    if not pairs:
        return _rejected_result(
            frame, component, profile, configuration,
            "standard_aperture_missing", pairs=pairs,
        )
    if len(pairs) != 1:
        return _rejected_result(
            frame, component, profile, configuration,
            "standard_multiple_apertures", pairs=pairs,
        )

    density = density_bank.get(component, profile.profile_id)
    try:
        fit = fit_standard_quadrilateral(
            p70_mask=density.p70_mask,
            closed_mask=component_mask(frame, component, stage="closed"),
            image_origin_uv=component.image_origin_uv,
            image_shape=frame.image_shape,
            aperture_contour_uv=nodes[pairs[0][1]].points_uv,
            configuration=configuration,
        )
    except (cv2.error, ValueError, FloatingPointError, np.linalg.LinAlgError) \
            as error:
        return _rejected_result(
            frame, component, profile, configuration,
            f"standard_geometry_error:{type(error).__name__}",
            pairs=pairs, density=density,
        )
    if not fit.accepted:
        return _rejected_result(
            frame, component, profile, configuration,
            fit.rejection_reason or "standard_quadrilateral_fit_failed",
            pairs=pairs, density=density, fit=fit,
        )
    return StandardGateResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        route=STANDARD_ROUTE,
        fitter=FITTER_NAME,
        configuration_version=configuration.configuration_version,
        selected_density_profile=profile,
        significant_parent_child_contour_ids=pairs,
        p70_threshold=float(density.p70_threshold),
        p70_evidence_points=int(np.count_nonzero(density.p70_mask)),
        p90_threshold=float(density.p90_threshold),
        p90_evidence_points=int(np.count_nonzero(density.p90_mask)),
        initial_corners_uv=fit.initial_corners_uv,
        fitted_corners_uv=fit.fitted_corners_uv,
        side_evidence=fit.side_evidence,
        aperture_center_offset_ratio=fit.aperture_center_offset_ratio,
        quadrilateral_area_px2=fit.area_px2,
        p70_coverage_ratio=fit.p70_coverage_ratio,
        fit_confidence=fit.fit_confidence,
        high_confidence_threshold=configuration.minimum_fit_confidence,
        high_confidence=True,
        accepted=True,
        rejection_reason=None,
    )


def process_standard_components(
    frame: FrameObservation,
    density_bank: DensityBank,
    configuration: StandardGateConfiguration =
        DEFAULT_STANDARD_GATE_CONFIGURATION,
) -> tuple[StandardGateResult, ...]:
    """Evaluate every frame component once, retaining accepted and rejected."""
    if density_bank.frame is not frame:
        raise ValueError("density bank belongs to another frame")
    return tuple(
        process_standard_gate(frame, component, density_bank, configuration)
        for component in frame.components
    )


def quadrilateral_from_standard(
    result: StandardGateResult,
) -> QuadrilateralEstimate:
    """Adapt one high-confidence result to the shared PnP input contract."""
    corners, area = (
        normalize_quadrilateral(result.fitted_corners_uv, result.image_shape)
        if result.accepted and result.high_confidence
        else (None, None)
    )
    accepted = bool(
        result.accepted and result.high_confidence and corners is not None
    )
    if not result.accepted:
        rejection_reason = result.rejection_reason
    elif not accepted:
        rejection_reason = "standard_quadrilateral_validation_failed"
    else:
        rejection_reason = None
    return QuadrilateralEstimate(
        frame_id=result.frame_id,
        sim_time_ns=result.sim_time_ns,
        component_id=result.component_id,
        image_shape=result.image_shape,
        route=STANDARD_ROUTE,
        fitter=result.fitter,
        selected_density_profile=result.selected_density_profile,
        corner_order=QUADRILATERAL_CORNER_ORDER,
        corners_uv=corners,
        p90_threshold=result.p90_threshold,
        p90_evidence_points=result.p90_evidence_points,
        area_px2=area,
        accepted=accepted,
        rejection_reason=rejection_reason,
        fit_confidence=result.fit_confidence,
    )
