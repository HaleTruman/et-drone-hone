"""Route adapter for the approved C-shape geometry baseline."""

from __future__ import annotations

import cv2
import numpy as np

from ..configurations import DEFAULT_C_SHAPE_CONFIGURATION
from ..density_bank import DensityBank
from ..preprocessing import component_mask
from ..quadrilateral import normalize_quadrilateral
from ..schema import (
    C_SHAPE_ROUTE, C_SHAPE_TOPOLOGY, QUADRILATERAL_CORNER_ORDER,
    CShapeConfiguration, CShapeRefinedLine, CShapeResult,
    ComponentObservation, FrameObservation, QuadrilateralEstimate,
    TopologyDecision)
from ..c_shape_three_line_pose import CShapeDensityInput
from .c_shape_tailored_shortfall_baseline import fit_c_shape_geometry_baseline


FITTER_NAME = "c_shape_tailored_shortfall_v1"
REFINED_MASK_LINE_WIDTH_PX = 2


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _points(values) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in values)


def select_c_shape_density_profile_id(
    component: ComponentObservation,
    configuration: CShapeConfiguration = DEFAULT_C_SHAPE_CONFIGURATION,
) -> str:
    """Apply the accepted legacy dimension bands to named bank profiles."""
    rules = configuration.density_profile_rules
    if not rules:
        raise ValueError("C-shape configuration requires profile rules")
    limits = tuple(rule.maximum_component_dimension_px for rule in rules)
    if (limits[-1] is not None or any(limit is None for limit in limits[:-1])
            or any(limit is not None and limit < 1 for limit in limits)
            or any(left >= right for left, right in zip(
                limits[:-2], limits[1:-1]))):
        raise ValueError(
            "C-shape profile limits must increase and end unbounded")
    maximum_dimension = max(component.bbox_xywh[2:])
    for rule in rules:
        if (rule.maximum_component_dimension_px is None or
                maximum_dimension <= rule.maximum_component_dimension_px):
            return rule.density_profile_id
    raise AssertionError("the final C-shape profile rule must be unbounded")


def _mask_from_refined_lines(image_shape, extent, candidate):
    closed = np.asarray(extent.closed_intersections_uv, np.float64)
    endpoints = np.asarray(candidate.extended_endpoints_uv, np.float64)
    points = np.vstack((closed, endpoints))
    line_width = REFINED_MASK_LINE_WIDTH_PX
    margin = (line_width + 1) // 2 + 1
    height, width = image_shape
    x0 = max(0, int(np.floor(points[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(points[:, 1].min())) - margin)
    x1 = min(width, int(np.ceil(points[:, 0].max())) + margin + 1)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0), line_width, _readonly(np.zeros((1, 1), np.uint8))
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    origin = np.asarray((x0, y0), np.float64)
    segments = (
        (closed[0], closed[1]),
        (closed[0], endpoints[0]),
        (closed[1], endpoints[1]),
        (endpoints[0], endpoints[1]),
    )
    for start, end in segments:
        start_xy = tuple(np.rint(start - origin).astype(int))
        end_xy = tuple(np.rint(end - origin).astype(int))
        cv2.line(mask, start_xy, end_xy, 1, line_width, cv2.LINE_8)
    return (x0, y0), line_width, _readonly(mask)


def _rejected_result(
    frame, component, density, configuration, reason,
) -> CShapeResult:
    return CShapeResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        topology_label=C_SHAPE_TOPOLOGY,
        route=C_SHAPE_ROUTE,
        fitter=FITTER_NAME,
        configuration_version=configuration.configuration_version,
        selected_density_profile=density.profile,
        p90_threshold=float(density.p90_threshold),
        p90_evidence_points=int(np.count_nonzero(density.p90_mask)),
        visible_lines=(),
        missing_side_index=None,
        closed_intersections_uv=None,
        contour_exits_uv=None,
        extension_lengths_px=None,
        extended_endpoints_uv=None,
        completed_quadrilateral_uv=None,
        quadrilateral_convex=None,
        quadrilateral_area_px2=None,
        refined_mask_origin_uv=component.image_origin_uv,
        refined_mask_line_width_px=None,
        refined_mask=_readonly(np.zeros(component.analysis_shape, np.uint8)),
        accepted=False,
        rejection_reason=reason,
    )


def process_c_shape(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
    density_bank: DensityBank,
    configuration: CShapeConfiguration = DEFAULT_C_SHAPE_CONFIGURATION,
) -> CShapeResult:
    """Run the pinned fitter for one accepted ``c_shape`` route."""
    if not any(owned is component for owned in frame.components):
        raise ValueError("component does not belong to frame")
    identity = (frame.frame_id, frame.sim_time_ns, component.component_id)
    if identity != (
            decision.frame_id, decision.sim_time_ns, decision.component_id):
        raise ValueError("topology decision belongs to another component")
    if (not decision.accepted or decision.topology_label != C_SHAPE_TOPOLOGY
            or decision.route != C_SHAPE_ROUTE):
        raise ValueError("C-shape fitter requires an accepted c_shape route")
    if density_bank.frame is not frame:
        raise ValueError("density bank belongs to another frame")

    profile_id = select_c_shape_density_profile_id(component, configuration)
    density = density_bank.get(component, profile_id)
    source = CShapeDensityInput(
        instance_id=f"{frame.frame_id}:{component.component_id}",
        mask=component_mask(frame, component, stage="closed"),
        density=density.final_field,
        image_origin_uv=component.image_origin_uv,
    )
    try:
        baseline = fit_c_shape_geometry_baseline(source)
    except (cv2.error, ValueError, FloatingPointError, np.linalg.LinAlgError) \
            as error:
        return _rejected_result(
            frame, component, density, configuration,
            f"c_shape_geometry_error:{type(error).__name__}")
    if baseline is None:
        return _rejected_result(
            frame, component, density, configuration,
            "c_shape_geometry_fit_failed")

    bounded = baseline.bounded_result
    extent = bounded.bounded_extent
    candidate = baseline.candidate
    support = {
        line.side_index: line.support_points for line in extent.visible_lines}
    lines = tuple(CShapeRefinedLine(
        side_index=int(line.side_index),
        support_points=int(support[line.side_index]),
        p90_coefficients=tuple(map(float, line.p90_coefficients)),
        contour_coefficients=tuple(map(float, line.contour_coefficients)),
        refined_coefficients=tuple(map(float, line.bounded_coefficients)),
        raw_contour_delta_degrees=float(line.raw_contour_delta_degrees),
        weighted_contour_delta_degrees=float(line.weighted_delta_degrees),
        applied_contour_delta_degrees=float(line.applied_delta_degrees),
        correction_capped=bool(line.correction_capped),
        p90_rmse_px=float(line.p90_rmse_px),
    ) for line in bounded.bounded_lines)
    exits = np.asarray(candidate.observed_exits_uv, np.float64)
    endpoints = np.asarray(candidate.extended_endpoints_uv, np.float64)
    extension_lengths = np.linalg.norm(endpoints - exits, axis=1)
    mask_origin, line_width, refined_mask = _mask_from_refined_lines(
        frame.image_shape, extent, candidate)
    return CShapeResult(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        topology_label=C_SHAPE_TOPOLOGY,
        route=C_SHAPE_ROUTE,
        fitter=FITTER_NAME,
        configuration_version=configuration.configuration_version,
        selected_density_profile=density.profile,
        p90_threshold=float(extent.threshold),
        p90_evidence_points=int(np.count_nonzero(density.p90_mask)),
        visible_lines=lines,
        missing_side_index=int(extent.missing_side_index),
        closed_intersections_uv=_points(extent.closed_intersections_uv),
        contour_exits_uv=_points(candidate.observed_exits_uv),
        extension_lengths_px=tuple(map(float, extension_lengths)),
        extended_endpoints_uv=_points(candidate.extended_endpoints_uv),
        completed_quadrilateral_uv=_points(candidate.quadrilateral_uv),
        quadrilateral_convex=bool(candidate.convex),
        quadrilateral_area_px2=float(candidate.area_px2),
        refined_mask_origin_uv=mask_origin,
        refined_mask_line_width_px=line_width,
        refined_mask=refined_mask,
        accepted=True,
        rejection_reason=None,
    )


def quadrilateral_from_c_shape(result: CShapeResult) -> QuadrilateralEstimate:
    """Normalize one C-shape completion to the shared quadrilateral contract."""
    corners, area = (normalize_quadrilateral(
        result.completed_quadrilateral_uv, result.image_shape)
        if result.accepted else (None, None))
    accepted = result.accepted and corners is not None
    if not result.accepted:
        rejection_reason = result.rejection_reason
    elif not accepted:
        rejection_reason = "c_shape_quadrilateral_validation_failed"
    else:
        rejection_reason = None
    return QuadrilateralEstimate(
        frame_id=result.frame_id,
        sim_time_ns=result.sim_time_ns,
        component_id=result.component_id,
        image_shape=result.image_shape,
        route=C_SHAPE_ROUTE,
        fitter=result.fitter,
        selected_density_profile=result.selected_density_profile,
        corner_order=QUADRILATERAL_CORNER_ORDER,
        corners_uv=corners,
        p90_threshold=result.p90_threshold,
        p90_evidence_points=result.p90_evidence_points,
        area_px2=area,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )
