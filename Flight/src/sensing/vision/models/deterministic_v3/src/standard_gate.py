"""Calibrated P90 quadrilateral fitting for one standard gate component."""

from __future__ import annotations

import cv2
import numpy as np

from .schema import (
    QUADRILATERAL_CORNER_ORDER,
    STANDARD_ROUTE,
    STANDARD_TOPOLOGY,
    ComponentObservation,
    DensityEvidence,
    FrameObservation,
    QuadrilateralEstimate,
    TopologyDecision,
)


FITTER_NAME = "standard_p90_lines_v1"


def _segment_distance(points: np.ndarray, start: np.ndarray,
                      end: np.ndarray) -> np.ndarray:
    edge = end - start
    length_squared = float(edge @ edge)
    if length_squared <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    position = np.clip(((points - start) @ edge) / length_squared, 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * edge), axis=1)


def _intersection(first, second) -> np.ndarray | None:
    point_a, direction_a = first
    point_b, direction_b = second
    denominator = (direction_a[0] * direction_b[1] -
                   direction_a[1] * direction_b[0])
    if abs(denominator) < 1e-6:
        return None
    delta = point_b - point_a
    distance = (delta[0] * direction_b[1] -
                delta[1] * direction_b[0]) / denominator
    return point_a + distance * direction_a


def _ordered_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1],
                        points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    contour = ordered.astype(np.float32).reshape(-1, 1, 2)
    if cv2.contourArea(contour, oriented=True) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def _valid_corners(corners: np.ndarray | None,
                   image_shape: tuple[int, int]) -> bool:
    if corners is None or corners.shape != (4, 2) or \
            not np.all(np.isfinite(corners)):
        return False
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 4.0:
        return False
    height, width = image_shape
    return bool(
        np.all((0 <= corners[:, 0]) & (corners[:, 0] < width)) and
        np.all((0 <= corners[:, 1]) & (corners[:, 1] < height)))


def _fit_p90_corners(evidence: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(evidence.astype(np.float32).reshape(-1, 1, 2))
    if len(hull) < 4 or cv2.contourArea(hull) < 4.0:
        return None
    initial = cv2.approxPolyN(
        hull, 4, epsilon_percentage=-1, ensure_convex=True).reshape(-1, 2)
    if len(initial) != 4:
        return None
    initial = _ordered_corners(initial)
    distances = np.column_stack([
        _segment_distance(evidence, initial[index], initial[(index + 1) % 4])
        for index in range(4)
    ])
    assignments = np.argmin(distances, axis=1)
    lines = []
    for index in range(4):
        side_points = evidence[assignments == index]
        if len(side_points) < 2:
            side_points = initial[[index, (index + 1) % 4]]
        vx, vy, x, y = cv2.fitLine(
            side_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        lines.append((np.array((float(x[0]), float(y[0]))),
                      np.array((float(vx[0]), float(vy[0])))))
    fitted = tuple(_intersection(lines[index], lines[(index + 1) % 4])
                   for index in range(4))
    return (initial if any(point is None for point in fitted)
            else _ordered_corners(np.asarray(fitted)))


def fit_standard_gate(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
    density: DensityEvidence,
) -> QuadrilateralEstimate:
    """Fit a normalized image-space quadrilateral from cached P90 evidence."""
    if not any(owned is component for owned in frame.components):
        raise ValueError("component does not belong to frame")
    identity = (frame.frame_id, frame.sim_time_ns, component.component_id)
    if identity != (decision.frame_id, decision.sim_time_ns,
                    decision.component_id):
        raise ValueError("topology decision belongs to another component")
    if (not decision.accepted or decision.topology_label != STANDARD_TOPOLOGY
            or decision.route != STANDARD_ROUTE):
        raise ValueError("standard fitter requires an accepted standard route")
    if density.component_id != component.component_id:
        raise ValueError("density evidence belongs to another component")

    evidence_yx = np.argwhere(density.p90_mask)
    evidence = evidence_yx[:, ::-1].astype(np.float64)
    corners = None if len(evidence) < 4 else _fit_p90_corners(evidence)
    if corners is not None:
        corners = _ordered_corners(
            corners + np.asarray(component.image_origin_uv, np.float64))
    accepted = _valid_corners(corners, frame.image_shape)
    corner_values = (None if not accepted else tuple(
        (float(point[0]), float(point[1])) for point in corners))
    area = (None if not accepted else float(cv2.contourArea(
        corners.astype(np.float32).reshape(-1, 1, 2))))
    return QuadrilateralEstimate(
        frame_id=frame.frame_id,
        sim_time_ns=frame.sim_time_ns,
        component_id=component.component_id,
        image_shape=frame.image_shape,
        route=STANDARD_ROUTE,
        fitter=FITTER_NAME,
        selected_density_profile=density.profile,
        corner_order=QUADRILATERAL_CORNER_ORDER,
        corners_uv=corner_values,
        p90_threshold=float(density.p90_threshold),
        p90_evidence_points=len(evidence),
        area_px2=area,
        accepted=accepted,
        rejection_reason=None if accepted else "standard_quadrilateral_fit_failed",
    )
