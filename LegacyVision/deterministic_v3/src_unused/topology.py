"""Closed-mask contour-hierarchy topology assessment and routing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schema import (
    CLIPPED_TOPOLOGY,
    C_SHAPE_ROUTE,
    C_SHAPE_TOPOLOGY,
    MULTI_GATE_ROUTE,
    MULTI_VOID_TOPOLOGY,
    STANDARD_ROUTE,
    STANDARD_TOPOLOGY,
    UNKNOWN_TOPOLOGY,
    ApertureCenterEvidence,
    ComponentObservation,
    ContourNodeEvidence,
    FrameObservation,
    TopologyDecision,
)


@dataclass(frozen=True, slots=True)
class TopologyPolicy:
    """First closed-mask hierarchy policy; values remain directly tunable."""

    minimum_hole_area_px2: float = 12.0
    standard_minimum_distance_peak_px: float = 3.0
    standard_minimum_diameter_ratio: float = 0.50
    standard_maximum_center_offset_ratio: float = 0.20
    standard_minimum_radial_balance_ratio: float = 0.55
    standard_minimum_radial_balance_median: float = 0.70
    standard_radial_sample_count: int = 36
    mixed_opening_minimum_span_px: int = 75

    def __post_init__(self) -> None:
        positive = (
            "minimum_hole_area_px2",
            "standard_minimum_distance_peak_px",
            "standard_minimum_diameter_ratio",
            "mixed_opening_minimum_span_px",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "standard_maximum_center_offset_ratio",
            "standard_minimum_radial_balance_ratio",
            "standard_minimum_radial_balance_median",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.standard_radial_sample_count < 4:
            raise ValueError("standard_radial_sample_count must be at least four")


DEFAULT_POLICY = TopologyPolicy()


@dataclass(frozen=True, slots=True)
class _OpeningEvidence:
    area_px: int = 0
    area_ratio: float = 0.0
    max_distance_px: float = 0.0
    depth_ratio: float = 0.0
    span_px: int = 0
    span_axis: str | None = None
    segment_uv: tuple[tuple[float, float], ...] | None = None


def _node_map(frame: FrameObservation) -> dict[int, ContourNodeEvidence]:
    nodes = {node.contour_id: node for node in frame.closed_contour_nodes}
    if len(nodes) != len(frame.closed_contour_nodes):
        raise ValueError("closed contour IDs must be unique")
    return nodes


def _significant_pairs(
    component: ComponentObservation,
    nodes: dict[int, ContourNodeEvidence],
    policy: TopologyPolicy,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    pairs = tuple(
        (parent_id, child_id)
        for parent_id, child_id in component.contours.parent_child_contour_ids
        if nodes[child_id].area_px2 >= policy.minimum_hole_area_px2
    )
    counts = tuple(
        (parent_id, sum(pair_parent == parent_id
                        for pair_parent, _ in pairs))
        for parent_id in component.contours.parent_contour_ids
    )
    return pairs, counts


def _longest_bounded_gap(
    plane: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Return ``(span, line, left, right)`` without Python scan-line loops."""
    foreground = plane != 0
    line_count, width = foreground.shape
    if line_count == 0 or width < 3:
        return None
    indices = np.arange(width, dtype=np.int32)[None, :]
    left = np.maximum.accumulate(
        np.where(foreground, indices, -1), axis=1)
    right = np.minimum.accumulate(
        np.where(foreground, indices, width)[:, ::-1], axis=1)[:, ::-1]
    valid = (~foreground) & (left >= 0) & (right < width)
    spans = np.where(valid, right - left - 1, 0)
    flat_index = int(np.argmax(spans))
    line_index, column = divmod(flat_index, width)
    span = int(spans[line_index, column])
    if span <= 0:
        return None
    return (span, line_index, int(left[line_index, column]),
            int(right[line_index, column]))


def _bounded_opening(component: ComponentObservation) -> _OpeningEvidence:
    """Measure an exterior non-mask run bracketed by parent foreground."""
    outer = component.contours.outer_raw
    if outer is None or len(outer) < 3:
        return _OpeningEvidence()
    x, y, width, height = cv2.boundingRect(outer)
    if width < 1 or height < 1:
        return _OpeningEvidence()
    local = outer.copy()
    local[:, :, 0] -= x
    local[:, :, 1] -= y
    outer_fill = np.zeros((height, width), np.uint8)
    cv2.drawContours(outer_fill, [local], -1, 1, cv2.FILLED)

    best_span, best_axis, best_segment = 0, None, None
    for axis, plane in (("horizontal", outer_fill),
                        ("vertical", outer_fill.T)):
        gap = _longest_bounded_gap(plane)
        if gap is None:
            continue
        span, line_index, left, right = gap
        if span > best_span:
            best_span, best_axis = span, axis
            best_segment = (((left, line_index), (right, line_index))
                            if axis == "horizontal"
                            else ((line_index, left), (line_index, right)))

    hull_fill = np.zeros_like(outer_fill)
    hull = cv2.convexHull(local.copy())
    cv2.drawContours(hull_fill, [hull], -1, 1, cv2.FILLED)
    concavity = ((hull_fill != 0) & (outer_fill == 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        concavity, connectivity=8)
    dominant_area, dominant_depth = 0, 0.0
    if count > 1:
        concavity_distance = cv2.distanceTransform(
            concavity, cv2.DIST_L2, 5)
        _, dominant_depth, _, maximum_xy = cv2.minMaxLoc(concavity_distance)
        dominant_label = int(labels[maximum_xy[1], maximum_xy[0]])
        if dominant_label > 0:
            dominant_area = int(stats[dominant_label, cv2.CC_STAT_AREA])

    hull_area = int(np.count_nonzero(hull_fill))
    _, _, bbox_width, bbox_height = component.bbox_xywh
    origin_x, origin_y = component.image_origin_uv
    segment_uv = (None if best_segment is None else tuple(
        (float(origin_x + x + point_x), float(origin_y + y + point_y))
        for point_x, point_y in best_segment))
    return _OpeningEvidence(
        area_px=dominant_area,
        area_ratio=dominant_area / max(hull_area, 1),
        max_distance_px=dominant_depth,
        depth_ratio=dominant_depth / max(bbox_width, bbox_height, 1),
        span_px=best_span,
        span_axis=best_axis,
        segment_uv=segment_uv,
    )


def _aperture_center(
    node: ContourNodeEvidence, policy: TopologyPolicy
) -> ApertureCenterEvidence:
    """Confirm a child aperture around its distance-transform center."""
    contour = node.points_uv
    x, y, width, height = cv2.boundingRect(contour)
    local = contour.copy()
    local[:, :, 0] -= x - 1
    local[:, :, 1] -= y - 1
    child_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.drawContours(child_mask, [local], -1, 1, cv2.FILLED)
    distance = cv2.distanceTransform(child_mask, cv2.DIST_L2, 5)
    _, peak, _, _ = cv2.minMaxLoc(distance)
    core_y, core_x = np.nonzero(distance >= 0.90 * peak)
    core_weights = distance[core_y, core_x]
    core_mass = float(core_weights.sum())
    if core_mass > 0:
        peak_x = float(np.dot(core_x, core_weights) / core_mass)
        peak_y = float(np.dot(core_y, core_weights) / core_mass)
    else:
        peak_x, peak_y = 1.0, 1.0
    unpadded_x, unpadded_y = peak_x - 1.0, peak_y - 1.0

    diameter_ratio = 2.0 * peak / max(min(width, height), 1)
    center_uv = np.array((x + unpadded_x, y + unpadded_y), np.float64)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        contour_center = np.array((
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ))
    else:
        contour_center = contour[:, 0, :].mean(axis=0)
    contour_area = max(float(cv2.contourArea(contour)), 1.0)
    center_offset = float(
        np.linalg.norm(center_uv - contour_center) / np.sqrt(contour_area))

    points = contour[:, 0, :].astype(np.float64) - center_uv
    angles = np.linspace(
        0.0, np.pi, policy.standard_radial_sample_count, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    projections = points @ directions.T
    positive = projections.max(axis=0)
    negative = -projections.min(axis=0)
    radial_ratios = np.divide(
        np.minimum(positive, negative),
        np.maximum(positive, negative),
        out=np.zeros_like(positive),
        where=np.maximum(positive, negative) > 1e-9,
    )
    radial_balance = float(np.percentile(radial_ratios, 25))
    radial_median = float(np.median(radial_ratios))

    failed = []
    if peak < policy.standard_minimum_distance_peak_px:
        failed.append("distance_peak_below_minimum")
    if diameter_ratio < policy.standard_minimum_diameter_ratio:
        failed.append("diameter_ratio_below_minimum")
    if center_offset > policy.standard_maximum_center_offset_ratio:
        failed.append("center_offset_above_maximum")
    if radial_balance < policy.standard_minimum_radial_balance_ratio:
        failed.append("radial_balance_below_minimum")
    if radial_median < policy.standard_minimum_radial_balance_median:
        failed.append("radial_balance_median_below_minimum")
    return ApertureCenterEvidence(
        parent_contour_id=(node.parent_contour_id
                           if node.parent_contour_id is not None else -1),
        child_contour_id=node.contour_id,
        center_uv=(float(center_uv[0]), float(center_uv[1])),
        distance_peak_px=float(peak),
        diameter_ratio=float(diameter_ratio),
        center_offset_ratio=center_offset,
        radial_balance_ratio=radial_balance,
        radial_balance_median=radial_median,
        accepted=not failed,
        failed_checks=tuple(failed),
    )


def _decision(
    frame: FrameObservation,
    component: ComponentObservation,
    raw_holes: int,
    pairs: tuple[tuple[int, int], ...],
    counts: tuple[tuple[int, int], ...],
    centers: tuple[ApertureCenterEvidence, ...],
    opening: _OpeningEvidence,
    *,
    topology_label: str,
    route: str | None,
    accepted: bool,
    classification_rule: str,
    rejection_reason: str | None,
) -> TopologyDecision:
    return TopologyDecision(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        topology_label=topology_label,
        route=route,
        accepted=accepted,
        touches_frame=component.touches_frame,
        raw_significant_holes=raw_holes,
        closed_significant_holes=len(pairs),
        significant_parent_child_contour_ids=pairs,
        significant_child_counts_by_parent=counts,
        aperture_center_evidence=centers,
        foreground_distance_max_px=float(component.distance_transform.max()),
        exterior_void_area_px=opening.area_px,
        exterior_void_area_ratio=opening.area_ratio,
        exterior_void_max_distance_px=opening.max_distance_px,
        exterior_void_depth_ratio=opening.depth_ratio,
        exterior_void_span_px=opening.span_px,
        exterior_void_span_axis=opening.span_axis,
        exterior_void_segment_uv=opening.segment_uv,
        density_profile_id=None,
        density_p90_pixel_count=0,
        density_p90_component_count=0,
        classification_rule=classification_rule,
        rejection_reason=rejection_reason,
    )


def assess_component(
    frame: FrameObservation,
    component: ComponentObservation,
    *,
    policy: TopologyPolicy = DEFAULT_POLICY,
    _nodes: dict[int, ContourNodeEvidence] | None = None,
) -> TopologyDecision:
    """Assign one route from final closed-mask hierarchy evidence only."""
    if not any(owned is component for owned in frame.components):
        raise ValueError("component does not belong to frame")
    nodes = _node_map(frame) if _nodes is None else _nodes
    pairs, counts = _significant_pairs(component, nodes, policy)
    raw_holes = sum(
        area >= policy.minimum_hole_area_px2
        for area in component.topology.raw_hole_areas_px2)
    empty = _OpeningEvidence()
    if component.touches_frame:
        return _decision(
            frame, component, raw_holes, pairs, counts, (), empty,
            topology_label=CLIPPED_TOPOLOGY, route=None, accepted=False,
            classification_rule="frame_edge_clipped",
            rejection_reason="frame_edge_clipped")
    if not counts:
        return _decision(
            frame, component, raw_holes, pairs, counts, (), empty,
            topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False,
            classification_rule="missing_parent_contour",
            rejection_reason="missing_parent_contour")

    child_counts = tuple(count for _, count in counts)
    if any(count > 2 for count in child_counts):
        return _decision(
            frame, component, raw_holes, pairs, counts, (), empty,
            topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False,
            classification_rule="more_than_two_direct_children",
            rejection_reason="unsupported_direct_child_count")
    if any(count == 2 for count in child_counts):
        return _decision(
            frame, component, raw_holes, pairs, counts, (), empty,
            topology_label=MULTI_VOID_TOPOLOGY, route=MULTI_GATE_ROUTE,
            accepted=True, classification_rule="two_direct_children",
            rejection_reason=None)

    if any(count == 1 for count in child_counts):
        if not all(count == 1 for count in child_counts):
            return _decision(
                frame, component, raw_holes, pairs, counts, (), empty,
                topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False,
                classification_rule="mixed_parent_child_structure",
                rejection_reason="unsupported_parent_structure")
        opening = _bounded_opening(component)
        centers = tuple(_aperture_center(nodes[child_id], policy)
                        for _, child_id in pairs)
        if opening.span_px >= policy.mixed_opening_minimum_span_px:
            return _decision(
                frame, component, raw_holes, pairs, counts, centers, opening,
                topology_label=MULTI_VOID_TOPOLOGY, route=MULTI_GATE_ROUTE,
                accepted=True,
                classification_rule="one_child_plus_large_opening",
                rejection_reason=None)
        if all(center.accepted for center in centers):
            return _decision(
                frame, component, raw_holes, pairs, counts, centers, opening,
                topology_label=STANDARD_TOPOLOGY, route=STANDARD_ROUTE,
                accepted=True,
                classification_rule="one_child_balanced_aperture",
                rejection_reason=None)
        return _decision(
            frame, component, raw_holes, pairs, counts, centers, opening,
            topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False,
            classification_rule="one_child_unbalanced_aperture",
            rejection_reason="standard_aperture_center_unbalanced")

    opening = _bounded_opening(component)
    if len(counts) == 1 and opening.span_px > 0:
        return _decision(
            frame, component, raw_holes, pairs, counts, (), opening,
            topology_label=C_SHAPE_TOPOLOGY, route=C_SHAPE_ROUTE,
            accepted=True, classification_rule="zero_child_bounded_opening",
            rejection_reason=None)
    if len(counts) == 1:
        return _decision(
            frame, component, raw_holes, pairs, counts, (), opening,
            topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False,
            classification_rule="zero_child_solid",
            rejection_reason="solid_without_void")
    return _decision(
        frame, component, raw_holes, pairs, counts, (), opening,
        topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False,
        classification_rule="multiple_zero_child_parents",
        rejection_reason="unsupported_parent_structure")


def assess_frame(
    frame: FrameObservation,
    *,
    density_bank: object | None = None,
    policy: TopologyPolicy = DEFAULT_POLICY,
) -> tuple[TopologyDecision, ...]:
    """Assess every component; ``density_bank`` is compatibility-only."""
    if density_bank is not None and getattr(density_bank, "frame", frame) is not frame:
        raise ValueError("density bank belongs to another frame")
    nodes = _node_map(frame)
    return tuple(assess_component(
        frame, component, policy=policy, _nodes=nodes)
                 for component in frame.components)


def classify_component(
    frame: FrameObservation,
    component: ComponentObservation,
    *,
    c_shape_supported: bool = False,
    policy: TopologyPolicy = DEFAULT_POLICY,
) -> TopologyDecision:
    """Compatibility entry point; active code should use :func:`assess_frame`."""
    del c_shape_supported
    return assess_component(frame, component, policy=policy)
