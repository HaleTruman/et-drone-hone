import argparse
import colorsys
import json
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .instance_tracking import InstanceTracker
from .legacy_inverse_density import (
    DENSITY_RADIUS_PX as CALIBRATED_DENSITY_RADIUS_PX,
    INVERSE_GAMMA as CALIBRATED_INVERSE_GAMMA,
    RELATIVE_CAP as CALIBRATED_RELATIVE_CAP,
    RIDGE_GAMMA as CALIBRATED_RIDGE_GAMMA,
    RIDGE_RADIUS_PX as CALIBRATED_RIDGE_RADIUS_PX,
    compute_density_fields as compute_calibrated_inverse_density,
    heat_layer as calibrated_inverse_density_heat,
)
from .schema import (CornerGeometry, EllipseEstimate, VoidDetectionFeatures,
                     VoidDetectionFrame, VoidDetectionGeometry,
                     geometry_source)
from .void_geometry import (compute_geometry, draw_geometry, draw_inner_corners,
                            draw_outer_corners, draw_parent_obtuse_corners)


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT.parent
RUNS = MODELS_ROOT / "detection_v2" / "runs"
LOG_RUNS = ROOT.parents[5] / "Logs" / "flight" / "runs"
REVIEWS = ROOT / "review_runs"
MANIFEST = ROOT / "ui" / "frontend" / "data" / "runs-manifest.json"
LUT_PATH = ROOT / "assets" / "color_lut_v1.npz"
EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_DISTANCE_GAMMA = 0.35
DEFAULT_LOCAL_DISTANCE_RADIUS = 9.0
OPEN_APERTURE_MIN_COMPONENT_AREA = 120
OPEN_APERTURE_MIN_RECT_SIDE_PX = 18.0
OPEN_APERTURE_MAX_FILL_RATIO = 0.38
OPEN_APERTURE_MIN_EXTENT_RATIO = 0.18
OPEN_APERTURE_INNER_AXIS_SCALE = 0.35
OPEN_APERTURE_MAX_CANDIDATES = 8
C_SHAPE_MIN_SIDE_COVERAGE = 0.18
C_SHAPE_VISIBLE_SIDE_COUNT = 3
C_SHAPE_MIN_QUAD_AREA = 120.0
MULTI_GATE_MIN_HOLE_AREA = 12.0
MULTI_GATE_MAX_APERTURES = 2
MULTI_GATE_OUTER_SCALE = 1.8
INVERSE_DENSITY_RADIUS_PX = 9
DENSE_PIXEL_PERCENTILE = 80.0
DENSE_LINE_MIN_COMPONENT_AREA = 20
DENSE_LINE_MIN_SEGMENTS = 2
DENSE_LINE_MAX_SEGMENTS = 8
DENSE_LINE_MIN_SPLIT_POINTS = 10
DENSE_LINE_MIN_SPLIT_IMPROVEMENT = 0.15
DENSE_LINE_ENDPOINT_PERCENTILES = (5.0, 95.0)
DENSE_QUADRILATERAL_MIN_AREA = 20.0
CALIBRATED_COMPONENT_MIN_EQUIVALENT_PX = 20.0
CALIBRATED_COMPONENT_MAX_EQUIVALENT_PX = 90.0
CALIBRATED_COMPONENT_RADIUS_PROFILES = (
    ("r20", 20, 17),
    ("r16", 16, 13),
    ("r11", 11, 9),
    ("r7", 7, 6),
    ("r2", 2, 2),
)
LOCAL_INVERSE_DENSITY_RADIUS_PX = 5
LOCAL_INVERSE_DENSITY_GAMMA = 3.0
LEGACY_INVERSE_DENSITY_RADIUS_PX = 5
LEGACY_INVERSE_DENSITY_GAMMA = 1.5
LEGACY_INVERSE_DENSITY_SWEEP = tuple(
    (radius, gamma)
    for radius in (3, 5, 7)
    for gamma in (1.0, 1.5, 3.0)
)
REVIEW_LAYERS = (
    ("base_mask", "LUT base mask", "void_detection.py::image_to_mask"),
    ("small_components", "Rejected small components",
     "void_detection.py::compute_mask_stages [area < 100 px]"),
    ("size_filtered", "Size-filtered mask",
     "void_detection.py::compute_mask_stages [size_filtered_mask]"),
    ("close_additions", "Morphological close additions",
     "void_detection.py::compute_mask_stages [MORPH_CLOSE 5x5]"),
    ("closed_mask", "Closed mask",
     "void_detection.py::compute_mask_stages [closed_mask]"),
    ("contour_mask", "Final contour input",
     "void_detection.py::compute_mask_stages [contour_mask]"),
    ("calibrated_final_inverse_density", "Calibrated final inverse density",
     "void_detection.py::calibrated_final_inverse_density_layer "
     "[native mask, density r12, mean cap 2, inverse gamma 1.65, "
     "ridge r10/gamma 4.15]"),
    ("calibrated_component_scaled_r20", "Component-scaled density · max r20",
     "void_detection.py::calibrated_component_scaled_layer "
     "[sqrt component pixels, density 2–20, ridge 2–17]"),
    ("calibrated_component_scaled_r16", "Component-scaled density · max r16",
     "void_detection.py::calibrated_component_scaled_layer "
     "[sqrt component pixels, density 2–16, ridge 2–13]"),
    ("calibrated_component_scaled_r11", "Component-scaled density · max r11",
     "void_detection.py::calibrated_component_scaled_layer "
     "[sqrt component pixels, density 2–11, ridge 2–9]"),
    ("calibrated_component_scaled_r7", "Component-scaled density · max r7",
     "void_detection.py::calibrated_component_scaled_layer "
     "[sqrt component pixels, density 2–7, ridge 2–6]"),
    ("calibrated_component_scaled_r2", "Component-scaled density · r2",
     "void_detection.py::calibrated_component_scaled_layer "
     "[all components density r2, ridge r2]"),
    ("inverse_mask_density", "Inverse mask pixel density",
     "void_detection.py::inverse_mask_density_layer [fixed 9 px radius]"),
    ("dense_pixel_line", "Adaptive dense line segments",
     "void_detection.py::dense_pixel_line_layer "
     "[radius 9, top 20%, adaptive 2–8 fits, endpoints 5–95%]"),
    ("dense_contour_quadrilateral", "Dense contour quadrilaterals",
     "void_detection.py::dense_contour_quadrilateral_layer "
     "[radius 9, top 20%, approxPolyN 4 sides, minAreaRect fallback]"),
    ("local_inverse_mask_density", "Local inverse mask pixel density",
     "void_detection.py::local_inverse_mask_density_layer "
     "[fixed 5 px radius, gamma 3]"),
    ("legacy_inverse_mask_density", "Legacy inverse mask density",
     "void_detection.py::legacy_inverse_mask_density_layer "
     "[mask_neighbor_density radius 5 square, gamma 1.5]"),
    ("contours", "Contour edges", "void_detection.py::find_contours"),
    ("distance_transform", "Distance transform inside contours",
     "void_detection.py::distance_transform_layer [--distance-gamma]"),
    ("local_distance_transform", "Local distance transform (9 px radius)",
     "void_detection.py::local_distance_transform_layer "
     "[--local-distance-radius]"),
    ("contour_to_local_maxima", "Contour to local distance maxima",
     "void_detection.py::contour_to_local_maxima_layer"),
    ("ellipses", "Fitted ellipses",
     "void_detection.py::analyze_ellipses + draw_ellipses"),
    ("outer_corners", "Ellipse-dependent outer corners",
     "void_geometry.py::compute_geometry [outer_corners]"),
    ("inner_corners", "Ellipse-dependent inner corners",
     "void_geometry.py::inner_corners"),
    ("parent_obtuse_corners", "Parent contour corners > 160°",
     "void_geometry.py::parent_obtuse_corners"),
    ("composite", "Combined detector render",
     "void_detection.py::render_composite"),
)


@dataclass(frozen=True, slots=True)
class MaskStages:
    """Intermediate masks used to produce the detector's review composite."""

    base_mask: np.ndarray
    small_pixels: np.ndarray
    size_filtered_mask: np.ndarray
    closed_mask: np.ndarray
    close_added_mask: np.ndarray
    contour_mask: np.ndarray


def frame_paths(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def include_latest_logged_run(runs, logs_root=LOG_RUNS):
    """Append the newest valid flight log run unless already selected."""
    if not logs_root.is_dir():
        return list(runs)
    candidates = [
        run for run in logs_root.iterdir()
        if run.is_dir() and (run / "vision_frames").is_dir()
        and frame_paths(run / "vision_frames")
    ]
    latest = max(candidates, key=lambda run: run.name, default=None)
    selected = list(runs)
    if latest is not None and latest.name not in {run.name for run in selected}:
        selected.append(latest)
    return selected


def load_lut(path=LUT_PATH):
    with np.load(path) as archive:
        lut = archive["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"Unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut


def image_to_mask(image, lut):
    bgr = image.astype(np.uint32)
    key = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return (lut[key] != 0).astype(np.uint8) * 255


def compute_mask_stages(image, lut):
    """Return every mask that participates in the active detector path."""
    binary = image_to_mask(image, lut)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 100
    small[0] = False
    small_pixels = small[labels]

    size_filtered = binary.copy()
    size_filtered[small_pixels] = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(size_filtered, cv2.MORPH_CLOSE, close_kernel)
    added = cv2.subtract(closed, size_filtered)
    contour_mask = closed.copy()
    contour_mask[small_pixels] = 0
    return MaskStages(
        base_mask=binary,
        small_pixels=small_pixels,
        size_filtered_mask=size_filtered,
        closed_mask=closed,
        close_added_mask=added,
        contour_mask=contour_mask,
    )


def next_ellipse_color(rng, used):
    while True:
        rgb = colorsys.hsv_to_rgb(rng.random(), 0.9, 1.0)
        candidate = tuple(round(value * 255) for value in reversed(rgb))
        if candidate not in used:
            used.add(candidate)
            return candidate


def fit_contour(component):
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) >= 5:
        center, size, angle = cv2.fitEllipse(contour)
        center = tuple(round(value) for value in center)
        axes = tuple(max(1, round(value / 2)) for value in size)
    else:
        x, y, width, height = cv2.boundingRect(contour)
        center = (x + width // 2, y + height // 2)
        axes, angle = (max(1, width // 2), max(1, height // 2)), 0
    return center, axes, angle


def geometry_record(ellipse, color, features):
    return VoidDetectionGeometry(
        geometry_source("void_detection", ellipse), ellipse,
        tuple(map(int, color)), features.outer_corners, features.inner_corners,
        features.parent_obtuse_corners)


GEOMETRY_FACTORIES = {"void_geometry.json": geometry_record}


def analyze_ellipses(contours, hierarchy, context, frame_shape, contour_mask=None):
    if hierarchy is None:
        return [], [], [], {name: [] for name in GEOMETRY_FACTORIES}
    rng = random.Random(f"ellipses:{context['frame_id']}")
    used = {(0, 0, 0), (255, 255, 255), (128, 128, 128),
            (0, 255, 0), (255, 0, 255)}
    blank = np.zeros(frame_shape[:2], np.uint8)
    ellipses = []
    children_by_parent = {}
    depths = {}
    for index, node in enumerate(hierarchy[0]):
        depth = 0
        parent = int(node[3])
        if parent >= 0:
            children_by_parent.setdefault(parent, []).append(index)
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        depths[index] = depth
        if depth % 2 == 0:
            continue
        component = blank.copy()
        cv2.drawContours(component, contours, index, 255, cv2.FILLED)
        area = cv2.countNonZero(component)
        fitted = fit_contour(component) if area >= 75 else None
        if fitted:
            color = next_ellipse_color(rng, used)
            parent_id = int(node[3])
            if unranked_multi_gate_child(
                    contours, children_by_parent, depths, index, parent_id):
                continue
            feature = multi_gate_aperture_feature(
                contours, children_by_parent, depths, index, parent_id)
            ellipses.append((area, *fitted, color, parent_id, feature))
    if contour_mask is not None:
        ellipses.extend(open_aperture_candidates(
            contours, children_by_parent, depths, blank, contour_mask, rng, used))
    estimates, groups = [], []
    ordered = sorted(ellipses, key=lambda item: item[0], reverse=True)
    for area, center, axes, angle, color, parent, _ in ordered:
        groups.append((parent, center, axes, color))
        estimates.append(EllipseEstimate(
            ellipse_id=context["id_start"] + len(estimates),
            run_id=context["run_id"], frame_id=context["frame_id"],
            frame_index=context["frame_index"], frame_count=context["frame_count"],
            frame_size_px=(frame_shape[1], frame_shape[0]),
            center_px=tuple(map(float, center)),
            semi_axes_px=tuple(map(float, axes)),
            angle_degrees=float(angle), source_area_px=area,
        ))
    features = compute_geometry(contours, groups)
    features = [
        item[6] if item[6] is not None else feature
        for item, feature in zip(ordered, features)
    ]
    records = {
        name: [factory(estimate, item[4], feature)
               for estimate, item, feature in zip(estimates, ordered, features)]
        for name, factory in GEOMETRY_FACTORIES.items()
    }
    return estimates, ordered, features, records


def open_aperture_candidates(
        contours, children_by_parent, depths, blank, contour_mask, rng, used):
    """Infer C-shape/open-aperture gates when an occluder removes the hole.

    The existing detector publishes enclosed child contours as voids.  Occluded
    gates can lose that enclosed contour entirely, leaving a sparse rectangular
    stroke as one parent component.  This conservative fallback only runs on
    contours with no child holes and requires a low fill ratio inside a stable
    minimum-area rectangle.
    """
    candidates = []
    for index, contour in enumerate(contours):
        if depths.get(index, 0) % 2 != 0:
            continue
        child_ids = children_by_parent.get(index, ())
        if any(depths.get(child, 0) % 2 != 0 for child in child_ids):
            continue
        if len(contour) < 5:
            continue
        filled = blank.copy()
        cv2.drawContours(filled, contours, index, 255, cv2.FILLED)
        component = cv2.bitwise_and(contour_mask, contour_mask, mask=filled)
        pixel_area = cv2.countNonZero(component)
        if pixel_area < OPEN_APERTURE_MIN_COMPONENT_AREA:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        long_side, short_side = max(width, height), min(width, height)
        if short_side < OPEN_APERTURE_MIN_RECT_SIDE_PX:
            continue
        rect_area = max(1.0, float(width * height))
        fill_ratio = pixel_area / rect_area
        extent_ratio = cv2.contourArea(contour) / rect_area
        if (fill_ratio > OPEN_APERTURE_MAX_FILL_RATIO or
                extent_ratio < OPEN_APERTURE_MIN_EXTENT_RATIO):
            continue
        center = (round(cx), round(cy))
        axes = (
            max(1, round(width * OPEN_APERTURE_INNER_AXIS_SCALE)),
            max(1, round(height * OPEN_APERTURE_INNER_AXIS_SCALE)),
        )
        color = next_ellipse_color(rng, used)
        score_area = int(round(pixel_area * (1.0 - fill_ratio) * long_side / short_side))
        feature = c_shape_feature(contour, component, rect)
        if feature is None:
            continue
        candidates.append((score_area, center, axes, float(angle), color, index,
                           feature))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[
        :OPEN_APERTURE_MAX_CANDIDATES]


def _ordered_quad(points):
    """Return image corners as upper-left, upper-right, lower-right, lower-left."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    total = points.sum(axis=1)
    diff = points[:, 0] - points[:, 1]
    upper_left = points[int(np.argmin(total))]
    lower_right = points[int(np.argmax(total))]
    upper_right = points[int(np.argmax(diff))]
    lower_left = points[int(np.argmin(diff))]
    return np.asarray((upper_left, upper_right, lower_right, lower_left),
                      np.float64)


def _corner_features(outer_quad, inner_quad=None, source="quadrilateral"):
    outer_order = (
        ("upper_left", outer_quad[0]),
        ("upper_right", outer_quad[1]),
        ("lower_left", outer_quad[3]),
        ("lower_right", outer_quad[2]),
    )
    outer = tuple(CornerGeometry(
        f"outer_{role}", tuple(map(int, np.rint(point))), source)
        for role, point in outer_order)
    inner = ()
    if inner_quad is not None:
        inner_order = (
            ("upper_left", inner_quad[0]),
            ("upper_right", inner_quad[1]),
            ("lower_left", inner_quad[3]),
            ("lower_right", inner_quad[2]),
        )
        inner = tuple(CornerGeometry(
            f"inner_{role}", tuple(map(int, np.rint(point))), source)
            for role, point in inner_order)
    return VoidDetectionFeatures(outer, inner, ())


def _segment_distances(points, start, end):
    edge = end - start
    position = np.clip(
        ((points - start) @ edge) / max(float(edge @ edge), 1e-9), 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * edge), axis=1)


def _line_from_points(points):
    if len(points) < 2:
        return None
    vx, vy, x, y = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    line = np.array((-vy, vx, vy * x - vx * y), np.float64)
    norm = np.linalg.norm(line[:2])
    return None if norm <= 1e-9 else line / norm


def _line_from_segment(start, end):
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm <= 1e-9:
        return None
    direction /= norm
    line = np.array((-direction[1], direction[0],
                     direction[1] * start[0] - direction[0] * start[1]),
                    np.float64)
    return line / np.linalg.norm(line[:2])


def _line_intersection(first, second):
    denominator = first[0] * second[1] - second[0] * first[1]
    if abs(denominator) <= 1e-9:
        return None
    x = (first[1] * second[2] - second[1] * first[2]) / denominator
    y = (second[0] * first[2] - first[0] * second[2]) / denominator
    return np.array((x, y), np.float64)


def c_shape_feature(contour, component, rect):
    """Complete three supported sides into route-specific outer corners."""
    del rect
    yy, xx = np.nonzero(component)
    if len(xx) < OPEN_APERTURE_MIN_COMPONENT_AREA:
        return None
    evidence = np.column_stack((xx, yy)).astype(np.float64)
    quad = _ordered_quad(cv2.boxPoints(cv2.minAreaRect(contour)))
    sides = ((0, 1), (1, 2), (2, 3), (3, 0))
    distances = np.column_stack([
        _segment_distances(evidence, quad[start], quad[end])
        for start, end in sides
    ])
    assignments = np.argmin(distances, axis=1)
    counts = np.bincount(assignments, minlength=4)
    side_lengths = np.asarray(
        [np.linalg.norm(quad[end] - quad[start]) for start, end in sides])
    coverage = counts / np.maximum(side_lengths, 1.0)
    visible = coverage >= C_SHAPE_MIN_SIDE_COVERAGE
    if np.count_nonzero(visible) < C_SHAPE_VISIBLE_SIDE_COUNT:
        return None
    missing = int(np.argmin(coverage))
    lines = []
    for side_index, (start, end) in enumerate(sides):
        if side_index == missing:
            line = _line_from_segment(quad[start], quad[end])
        else:
            line = _line_from_points(evidence[assignments == side_index])
            if line is None:
                line = _line_from_segment(quad[start], quad[end])
        if line is None:
            return None
        lines.append(line)
    fitted = []
    for side_index in range(4):
        point = _line_intersection(lines[side_index], lines[(side_index + 1) % 4])
        if point is None or not np.all(np.isfinite(point)):
            return None
        fitted.append(point)
    fitted = _ordered_quad(np.asarray(fitted))
    if cv2.contourArea(fitted.astype(np.float32)) < C_SHAPE_MIN_QUAD_AREA:
        return None
    return _corner_features(fitted, source="c_shape_three_line")


def multi_gate_aperture_feature(
        contours, children_by_parent, depths, child_index, parent_index):
    """Use aperture-local geometry for two-hole merged-gate components."""
    if parent_index < 0:
        return None
    child_ids = [
        child for child in children_by_parent.get(parent_index, ())
        if depths.get(child, 0) % 2 != 0
        and cv2.contourArea(contours[child]) >= MULTI_GATE_MIN_HOLE_AREA
    ]
    if len(child_ids) < MULTI_GATE_MAX_APERTURES or child_index not in child_ids:
        return None
    ranked = sorted(child_ids, key=lambda item: cv2.contourArea(contours[item]),
                    reverse=True)[:MULTI_GATE_MAX_APERTURES]
    if child_index not in ranked:
        return None
    inner = _ordered_quad(cv2.boxPoints(cv2.minAreaRect(contours[child_index])))
    center = inner.mean(axis=0)
    outer = center + (inner - center) * MULTI_GATE_OUTER_SCALE
    if cv2.contourArea(outer.astype(np.float32)) < C_SHAPE_MIN_QUAD_AREA:
        return None
    return _corner_features(outer, inner, source="multi_gate_aperture")


def unranked_multi_gate_child(
        contours, children_by_parent, depths, child_index, parent_index):
    if parent_index < 0:
        return False
    child_ids = [
        child for child in children_by_parent.get(parent_index, ())
        if depths.get(child, 0) % 2 != 0
        and cv2.contourArea(contours[child]) >= MULTI_GATE_MIN_HOLE_AREA
    ]
    if len(child_ids) < MULTI_GATE_MAX_APERTURES:
        return False
    ranked = sorted(child_ids, key=lambda item: cv2.contourArea(contours[item]),
                    reverse=True)[:MULTI_GATE_MAX_APERTURES]
    return child_index not in ranked


def draw_ellipses(image, ordered):
    for _, center, axes, angle, color, *_ in ordered:
        cv2.ellipse(image, center, axes, angle, 0, 360, color, cv2.FILLED)


def find_contours(contour_mask):
    contours, hierarchy = cv2.findContours(
        contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(contour, 1, True) for contour in contours], hierarchy


def analyze_image(image, context, lut):
    stages = compute_mask_stages(image, lut)
    contours, hierarchy = find_contours(stages.contour_mask)
    estimates, ordered, features, geometry = analyze_ellipses(
        contours, hierarchy, context, image.shape, stages.contour_mask)
    return stages, contours, ordered, features, estimates, geometry


def render_composite(stages, contours, ordered, features):
    output = cv2.cvtColor(stages.base_mask, cv2.COLOR_GRAY2BGR)
    output[stages.close_added_mask > 0] = (0, 255, 0)
    output[stages.small_pixels] = (128, 128, 128)
    cv2.drawContours(output, contours, -1, (255, 0, 255), 1)
    output[stages.small_pixels] = (128, 128, 128)
    draw_ellipses(output, ordered)
    draw_geometry(output, features, [item[4] for item in ordered])
    return output


def detect_image(image, context, lut):
    parts = analyze_image(image, context, lut)
    stages, contours, ordered, features, estimates, geometry = parts
    return render_composite(stages, contours, ordered, features), estimates, geometry


def _mask_density(contour_mask, radius):
    inside = (contour_mask > 0).astype(np.float32)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x * x + y * y <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    density = cv2.filter2D(
        inside, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    return inside, density


def _inverse_mask_density(contour_mask, radius, gamma=1.0):
    inside, density = _mask_density(contour_mask, radius)
    inverse = np.power(np.clip(1.0 - density, 0.0, 1.0), gamma)
    intensity = np.clip(inverse * 255, 0, 255).astype(np.uint8)
    intensity[inside == 0] = 0
    return cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)


def inverse_mask_density_layer(contour_mask):
    """Show sparse mask pixels within a fixed 9 px circular neighborhood."""
    return _inverse_mask_density(contour_mask, INVERSE_DENSITY_RADIUS_PX)


def calibrated_final_inverse_density_layer(contour_mask):
    """Render the documented calibrated field without resizing the input mask."""
    fields = compute_calibrated_inverse_density(contour_mask)
    return calibrated_inverse_density_heat(
        fields.final_inverse_density, contour_mask > 0)


def calibrated_component_radii(
        component_pixel_count, maximum_density_radius,
        maximum_ridge_radius):
    """Scale radius as a length derived from connected-component pixel area."""
    equivalent_pixels = np.sqrt(max(float(component_pixel_count), 0.0))
    span = (CALIBRATED_COMPONENT_MAX_EQUIVALENT_PX
            - CALIBRATED_COMPONENT_MIN_EQUIVALENT_PX)
    proportion = np.clip(
        (equivalent_pixels - CALIBRATED_COMPONENT_MIN_EQUIVALENT_PX) / span,
        0.0, 1.0)
    density_radius = round(2 + proportion * (maximum_density_radius - 2))
    ridge_radius = round(2 + proportion * (maximum_ridge_radius - 2))
    return max(2, density_radius), max(2, ridge_radius)


def calibrated_component_scaled_field(
        contour_mask, maximum_density_radius, maximum_ridge_radius):
    """Compute calibrated density per component with proportional local radii."""
    binary = (contour_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    output = np.zeros(contour_mask.shape, np.float64)
    for label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[label])
        density_radius, ridge_radius = calibrated_component_radii(
            area, maximum_density_radius, maximum_ridge_radius)
        padding = max(density_radius, ridge_radius)
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(contour_mask.shape[1], x + width + padding)
        y1 = min(contour_mask.shape[0], y + height + padding)
        component = (labels[y0:y1, x0:x1] == label).astype(np.uint8) * 255
        local = compute_calibrated_inverse_density(
            component, density_radius, ridge_radius).final_inverse_density
        destination = output[y0:y1, x0:x1]
        selected = component > 0
        destination[selected] = local[selected]
    return output


def calibrated_component_scaled_layer(
        contour_mask, maximum_density_radius, maximum_ridge_radius):
    field = calibrated_component_scaled_field(
        contour_mask, maximum_density_radius, maximum_ridge_radius)
    return calibrated_inverse_density_heat(field, contour_mask > 0)


def _fit_dense_points(points):
    if len(points) < 2:
        return None
    vx, vy, x0, y0 = (
        float(value) for value in
        cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1))
    direction = np.array([vx, vy], np.float32)
    origin = np.array([x0, y0], np.float32)
    offsets = points - origin
    projections = offsets @ direction
    low, high = np.percentile(
        projections, DENSE_LINE_ENDPOINT_PERCENTILES)
    if high - low < 1.0:
        return None
    perpendicular = np.abs(offsets[:, 0] * direction[1]
                           - offsets[:, 1] * direction[0])
    return {
        "endpoints": origin + np.outer([low, high], direction),
        "error": float(np.sqrt(np.mean(perpendicular ** 2))),
        "projections": projections,
    }


def _split_dense_group(group):
    projections = group["fit"]["projections"]
    midpoint = float(np.median(projections))
    partitions = (projections <= midpoint, projections > midpoint)
    if any(np.count_nonzero(partition) < DENSE_LINE_MIN_SPLIT_POINTS
           for partition in partitions):
        return None
    children = []
    for partition in partitions:
        points = group["points"][partition]
        fit = _fit_dense_points(points)
        if fit is None:
            return None
        children.append({
            "component": group["component"],
            "points": points,
            "fit": fit,
        })
    count = sum(len(child["points"]) for child in children)
    child_error = np.sqrt(sum(
        len(child["points"]) * child["fit"]["error"] ** 2
        for child in children) / count)
    parent_error = group["fit"]["error"]
    improvement = ((parent_error - child_error) / parent_error
                   if parent_error > 0 else 0.0)
    return children, improvement


def dense_pixel_line_segments(
        contour_mask, radius=INVERSE_DENSITY_RADIUS_PX,
        percentile=DENSE_PIXEL_PERCENTILE,
        minimum_area=DENSE_LINE_MIN_COMPONENT_AREA):
    """Return dense labels and two to eight adaptive local line fits."""
    inside, density = _mask_density(contour_mask, radius)
    foreground = inside > 0
    values = density[foreground]
    if values.size < 2:
        return np.zeros(contour_mask.shape, np.int32), []

    threshold = float(np.percentile(values, percentile))
    candidates = foreground & (density >= threshold)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8)
    accepted = sorted(
        (label for label in range(1, count)
         if stats[label, cv2.CC_STAT_AREA] >= minimum_area),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True,
    )[:DENSE_LINE_MAX_SEGMENTS]
    groups = []
    for label in accepted:
        ys, xs = np.nonzero(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float32)
        fit = _fit_dense_points(points)
        if fit is not None:
            groups.append({"component": label, "points": points, "fit": fit})

    while len(groups) < DENSE_LINE_MAX_SEGMENTS:
        options = []
        for index, group in enumerate(groups):
            split = _split_dense_group(group)
            if split is not None:
                options.append((split[1], index, split[0]))
        if not options:
            break
        improvement, index, children = max(options, key=lambda item: item[0])
        force_minimum = len(groups) < DENSE_LINE_MIN_SEGMENTS
        if not force_minimum and improvement < DENSE_LINE_MIN_SPLIT_IMPROVEMENT:
            break
        groups[index:index + 1] = children
    return labels, groups


def dense_pixel_line_layer(
        contour_mask, radius=INVERSE_DENSITY_RADIUS_PX,
        percentile=DENSE_PIXEL_PERCENTILE,
        minimum_area=DENSE_LINE_MIN_COMPONENT_AREA):
    """Render adaptive local line fits through dense mask evidence."""
    layer = np.zeros((*contour_mask.shape, 3), np.uint8)
    labels, groups = dense_pixel_line_segments(
        contour_mask, radius, percentile, minimum_area)
    components = sorted({group["component"] for group in groups})
    for label in components:
        component = labels == label
        hue = (label * 0.61803398875) % 1.0
        candidate_rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.42)
        layer[component] = tuple(
            round(channel * 255) for channel in reversed(candidate_rgb))

    for index, group in enumerate(groups):
        endpoints = np.rint(group["fit"]["endpoints"]).astype(int)
        endpoints[:, 0] = np.clip(
            endpoints[:, 0], 0, contour_mask.shape[1] - 1)
        endpoints[:, 1] = np.clip(
            endpoints[:, 1], 0, contour_mask.shape[0] - 1)
        hue = ((index + 1) * 0.61803398875) % 1.0
        line_rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        line_bgr = tuple(
            round(channel * 255) for channel in reversed(line_rgb))
        cv2.line(layer, tuple(endpoints[0]), tuple(endpoints[1]),
                 line_bgr, 1, cv2.LINE_AA)
    return layer


def dense_contour_quadrilaterals(
        contour_mask, radius=INVERSE_DENSITY_RADIUS_PX,
        percentile=DENSE_PIXEL_PERCENTILE,
        minimum_area=DENSE_QUADRILATERAL_MIN_AREA):
    """Extract external dense contours and force each to four convex sides."""
    inside, density = _mask_density(contour_mask, radius)
    foreground = inside > 0
    values = density[foreground]
    if values.size < 4:
        return np.zeros(contour_mask.shape, np.uint8), [], []

    threshold = float(np.percentile(values, percentile))
    candidates = (foreground & (density >= threshold)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = []
    quadrilaterals = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < minimum_area:
            continue
        hull = cv2.convexHull(contour)
        if len(hull) >= 4:
            quadrilateral = cv2.approxPolyN(
                hull, 4, epsilon_percentage=-1, ensure_convex=False)
            points = quadrilateral.reshape(-1, 2)
        else:
            points = cv2.boxPoints(cv2.minAreaRect(contour))
        if len(points) != 4:
            continue
        accepted.append(contour)
        quadrilaterals.append(points.astype(np.int32))
    return candidates, accepted, quadrilaterals


def dense_contour_quadrilateral_layer(contour_mask):
    """Render dense evidence, its external contours, and four-sided fits."""
    candidates, contours, quadrilaterals = dense_contour_quadrilaterals(
        contour_mask)
    layer = np.zeros((*contour_mask.shape, 3), np.uint8)
    layer[candidates > 0] = (48, 48, 48)
    cv2.drawContours(layer, contours, -1, (180, 180, 180), 1)
    for index, quadrilateral in enumerate(quadrilaterals):
        hue = ((index + 1) * 0.61803398875) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        bgr = tuple(round(channel * 255) for channel in reversed(rgb))
        cv2.polylines(layer, [quadrilateral.reshape(-1, 1, 2)], True,
                      bgr, 1, cv2.LINE_AA)
    return layer


def local_inverse_mask_density_layer(contour_mask):
    """Show sparse mask pixels using a fixed 5 px radius and 3.0 gamma."""
    return _inverse_mask_density(
        contour_mask, LOCAL_INVERSE_DENSITY_RADIUS_PX,
        LOCAL_INVERSE_DENSITY_GAMMA)


def legacy_sweep_id(radius, gamma):
    return f"r{radius}_g{gamma:g}".replace(".", "p")


def _legacy_density_context(contour_mask):
    mask = (contour_mask > 0).astype(np.float32)
    yy, xx = np.indices(mask.shape)
    integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return mask, yy, xx, integral


def _legacy_inverse_density(context, radius, gamma):
    mask, yy, xx, integral = context
    height, width = mask.shape
    y0, y1 = np.maximum(0, yy - radius), np.minimum(height - 1, yy + radius)
    x0, x1 = np.maximum(0, xx - radius), np.minimum(width - 1, xx + radius)
    local_sum = (integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1]
                 - integral[y1 + 1, x0] + integral[y0, x0])
    density = local_sum / ((y1 - y0 + 1) * (x1 - x0 + 1))
    inverse = mask * np.power(1.0 - density, gamma)
    rgb = np.empty((*mask.shape, 3), np.uint8)
    rgb[..., 0] = np.round(20 + 235 * inverse).astype(np.uint8)
    rgb[..., 1] = np.round(40 + 178 * np.sqrt(inverse)).astype(np.uint8)
    rgb[..., 2] = np.round(52 - 42 * inverse).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def legacy_inverse_mask_density_layer(
        contour_mask, radius=LEGACY_INVERSE_DENSITY_RADIUS_PX,
        gamma=LEGACY_INVERSE_DENSITY_GAMMA):
    """Render the legacy square-neighborhood inverse-density view."""
    return _legacy_inverse_density(
        _legacy_density_context(contour_mask), radius, gamma)


def legacy_inverse_density_sweep(contour_mask):
    context = _legacy_density_context(contour_mask)
    return {
        legacy_sweep_id(radius, gamma):
            _legacy_inverse_density(context, radius, gamma)
        for radius, gamma in LEGACY_INVERSE_DENSITY_SWEEP
    }


def distance_transform_layer(contour_mask, contours, gamma=DEFAULT_DISTANCE_GAMMA):
    inside = (contour_mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    peak = float(distance.max())
    intensity = np.power(distance / peak, gamma) if peak else distance
    layer = cv2.applyColorMap(
        np.clip(intensity * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    layer[inside == 0] = 0
    cv2.drawContours(layer, contours, -1, (255, 255, 255), 1)
    return layer


def local_distance_transform_layer(
        contour_mask, contours, radius=DEFAULT_LOCAL_DISTANCE_RADIUS,
        gamma=DEFAULT_DISTANCE_GAMMA):
    """Render distance against a fixed radius, independent of remote peaks."""
    inside = (contour_mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    intensity = np.power(np.clip(distance / radius, 0, 1), gamma)
    layer = cv2.applyColorMap(
        np.clip(intensity * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    layer[inside == 0] = 0
    cv2.drawContours(layer, contours, -1, (255, 255, 255), 1)
    return layer


def contour_to_local_maxima_layer(
        contour_mask, contours, gamma=DEFAULT_DISTANCE_GAMMA):
    """Map progress from contour lines to each component's distance maxima."""
    inside = (contour_mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    maxima = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))

    _, labels = cv2.connectedComponents(inside, connectivity=8)
    distance_to_maximum = np.zeros_like(distance)
    for label in range(1, int(labels.max()) + 1):
        component = labels == label
        ys, xs = np.where(component)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        seeds = np.ones((y1 - y0, x1 - x0), np.uint8)
        seeds[maxima[y0:y1, x0:x1] & component[y0:y1, x0:x1]] = 0
        peak_distance = cv2.distanceTransform(
            seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distance_to_maximum[y0:y1, x0:x1][component[y0:y1, x0:x1]] = (
            peak_distance[component[y0:y1, x0:x1]])

    contour_seeds = np.full(inside.shape, 255, np.uint8)
    cv2.drawContours(contour_seeds, contours, -1, 0, 1)
    distance_from_contour = cv2.distanceTransform(
        contour_seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    total = distance_from_contour + distance_to_maximum
    progress = np.divide(distance_from_contour, total, out=np.zeros_like(total),
                         where=(inside > 0) & (total > 0))
    intensity = np.power(progress, gamma)
    layer = cv2.applyColorMap(
        np.clip(intensity * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    layer[inside == 0] = 0
    cv2.drawContours(layer, contours, -1, (255, 255, 255), 1)
    return layer


def render_review_layers(image, context, lut,
                         distance_gamma=DEFAULT_DISTANCE_GAMMA,
                         local_distance_radius=DEFAULT_LOCAL_DISTANCE_RADIUS):
    parts = analyze_image(image, context, lut)
    stages, contours, ordered, features, estimates, geometry = parts
    blank = lambda: np.zeros_like(image)
    gray = lambda mask: cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    layers = {
        "base_mask": gray(stages.base_mask),
        "small_components": blank(),
        "size_filtered": gray(stages.size_filtered_mask),
        "close_additions": blank(),
        "closed_mask": gray(stages.closed_mask),
        "contour_mask": gray(stages.contour_mask),
        "calibrated_final_inverse_density":
            calibrated_final_inverse_density_layer(stages.contour_mask),
        "inverse_mask_density": inverse_mask_density_layer(stages.contour_mask),
        "dense_pixel_line": dense_pixel_line_layer(stages.contour_mask),
        "dense_contour_quadrilateral": dense_contour_quadrilateral_layer(
            stages.contour_mask),
        "local_inverse_mask_density": local_inverse_mask_density_layer(
            stages.contour_mask),
        "legacy_inverse_mask_density": legacy_inverse_mask_density_layer(
            stages.contour_mask),
        "contours": blank(),
        "distance_transform": blank(),
        "local_distance_transform": blank(),
        "contour_to_local_maxima": blank(),
        "ellipses": blank(),
        "outer_corners": blank(),
        "inner_corners": blank(),
        "parent_obtuse_corners": blank(),
    }
    for profile_id, density_radius, ridge_radius in (
            CALIBRATED_COMPONENT_RADIUS_PROFILES):
        layers[f"calibrated_component_scaled_{profile_id}"] = (
            calibrated_component_scaled_layer(
                stages.contour_mask, density_radius, ridge_radius))
    layers["small_components"][stages.small_pixels] = (128, 128, 128)
    layers["close_additions"][stages.close_added_mask > 0] = (0, 255, 0)
    cv2.drawContours(layers["contours"], contours, -1, (255, 0, 255), 1)
    layers["distance_transform"] = distance_transform_layer(
        stages.contour_mask, contours, distance_gamma)
    layers["local_distance_transform"] = local_distance_transform_layer(
        stages.contour_mask, contours, local_distance_radius, distance_gamma)
    layers["contour_to_local_maxima"] = contour_to_local_maxima_layer(
        stages.contour_mask, contours, distance_gamma)
    colors = [item[4] for item in ordered]
    draw_ellipses(layers["ellipses"], ordered)
    draw_outer_corners(layers["outer_corners"], features, colors)
    draw_inner_corners(layers["inner_corners"], features, colors)
    draw_parent_obtuse_corners(layers["parent_obtuse_corners"], features)
    layers["composite"] = render_composite(
        stages, contours, ordered, features)
    return layers, estimates, geometry


def render_frame(source, destination, context, lut):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read {source}")
    output, estimates, geometry = detect_image(image, context, lut)
    if not cv2.imwrite(str(destination), output):
        raise ValueError(f"Unable to write {destination}")
    return estimates, geometry


class VoidDetector:
    def __init__(self, run_id="live", lut_path=LUT_PATH):
        self.run_id, self.lut = str(run_id), load_lut(lut_path)
        self.tracker, self.frame_count, self.ellipse_count = InstanceTracker(), 0, 0

    def process_frame(self, frame, vehicle_state=None):
        image = frame.image
        if image is None:
            encoded = np.frombuffer(frame.jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode VisionFrame {frame.frame_id}")
        index, self.frame_count = self.frame_count, self.frame_count + 1
        label = f"frame-{int(frame.frame_id):08d}-{int(frame.sim_time_ns)}"
        context = {"id_start": self.ellipse_count, "run_id": self.run_id,
                   "frame_id": label, "frame_index": index,
                   "frame_count": self.frame_count}
        _, estimates, records = detect_image(image, context, self.lut)
        self.ellipse_count += len(estimates)
        detections = tuple(
            self.tracker.update_geometry(records["void_geometry.json"]))
        return VoidDetectionFrame(
            int(frame.frame_id), int(frame.sim_time_ns),
            self.frame_count, detections)


def process_run(run, output_root, lut, start=0, limit=None,
                distance_gamma=DEFAULT_DISTANCE_GAMMA,
                local_distance_radius=DEFAULT_LOCAL_DISTANCE_RADIUS):
    source = run / "vision_frames"
    if not source.is_dir():
        raise ValueError(f"Run has no vision_frames directory: {run}")
    frames = frame_paths(source)
    if not frames:
        raise ValueError(f"Run has no supported vision frames: {source}")
    if start >= len(frames):
        raise ValueError(
            f"Start index {start} is outside {run.name} ({len(frames)} frames)")
    selected = frames[start:start + limit if limit is not None else None]
    output = output_root / run.name / "layers"
    for layer_id, _, _ in REVIEW_LAYERS:
        (output / layer_id).mkdir(parents=True)
    sweep_output = (output_root / run.name / "sweeps" /
                    "legacy_inverse_mask_density")
    for radius, gamma in LEGACY_INVERSE_DENSITY_SWEEP:
        (sweep_output / legacy_sweep_id(radius, gamma)).mkdir(parents=True)
    for offset, frame in enumerate(selected):
        frame_index = start + offset
        context = {"id_start": 0, "run_id": run.name,
                   "frame_id": frame.stem, "frame_index": frame_index,
                   "frame_count": len(frames)}
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read {frame}")
        layers, _, _ = render_review_layers(
            image, context, lut, distance_gamma, local_distance_radius)
        for layer_id, layer in layers.items():
            destination = output / layer_id / f"{frame.stem}.png"
            if not cv2.imwrite(str(destination), layer):
                raise ValueError(f"Unable to write {destination}")
        sweep_layers = legacy_inverse_density_sweep(
            layers["contour_mask"][:, :, 0])
        for preset_id, layer in sweep_layers.items():
            destination = sweep_output / preset_id / f"{frame.stem}.png"
            if not cv2.imwrite(str(destination), layer):
                raise ValueError(f"Unable to write {destination}")
    return len(selected)


def static_url(path, static_root=MODELS_ROOT):
    try:
        relative = path.resolve().relative_to(static_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Static review output must be inside {static_root}: {path}") from exc
    return f"/{quote(relative.as_posix(), safe='/')}"


def write_manifest(run_names, output_root=REVIEWS, manifest=MANIFEST,
                   static_root=MODELS_ROOT,
                   distance_gamma=DEFAULT_DISTANCE_GAMMA,
                   local_distance_radius=DEFAULT_LOCAL_DISTANCE_RADIUS):
    runs = []
    for run_name in run_names:
        output = output_root / run_name / "layers"
        layer_files = {
            layer_id: {path.stem: path for path in frame_paths(output / layer_id)}
            for layer_id, _, _ in REVIEW_LAYERS
        }
        sweep_root = (output_root / run_name / "sweeps" /
                      "legacy_inverse_mask_density")
        sweep_files = {
            legacy_sweep_id(radius, gamma): {
                path.stem: path for path in frame_paths(
                    sweep_root / legacy_sweep_id(radius, gamma))
            }
            for radius, gamma in LEGACY_INVERSE_DENSITY_SWEEP
        }
        frame_ids = sorted(next(iter(layer_files.values())))
        frames = [{
            "id": frame_id,
            "layers": {
                layer_id: (f"{static_url(files[frame_id], static_root)}"
                           f"?v={files[frame_id].stat().st_mtime_ns}")
                for layer_id, files in layer_files.items()
            },
            "sweeps": {
                "legacy_inverse_mask_density": {
                    preset_id: (f"{static_url(files[frame_id], static_root)}"
                                f"?v={files[frame_id].stat().st_mtime_ns}")
                    for preset_id, files in sweep_files.items()
                }
            },
        } for frame_id in frame_ids]
        runs.append({"id": run_name, "name": run_name, "frames": frames})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(f"{manifest.suffix}.tmp")
    layers = [{"id": layer_id, "label": label, "reference": reference}
              for layer_id, label, reference in REVIEW_LAYERS]
    for layer in layers:
        if layer["id"] == "distance_transform":
            layer["reference"] += f" · gamma={distance_gamma:g}"
        elif layer["id"] == "local_distance_transform":
            layer["label"] = (
                f"Local distance transform ({local_distance_radius:g} px radius)")
            layer["reference"] += (
                f" · radius={local_distance_radius:g}px · gamma={distance_gamma:g}")
        elif layer["id"] == "contour_to_local_maxima":
            layer["reference"] += f" · component-isolated · gamma={distance_gamma:g}"
    sweep_presets = [{
        "id": legacy_sweep_id(radius, gamma),
        "label": f"radius {radius}px · gamma {gamma:g}",
        "radius": radius,
        "gamma": gamma,
    } for radius, gamma in LEGACY_INVERSE_DENSITY_SWEEP]
    payload = {"settings": {"distance_gamma": distance_gamma,
                            "local_distance_radius": local_distance_radius},
               "sweeps": [{
                   "layer_id": "legacy_inverse_mask_density",
                   "label": "Legacy density sweep",
                   "default_preset": legacy_sweep_id(
                       LEGACY_INVERSE_DENSITY_RADIUS_PX,
                       LEGACY_INVERSE_DENSITY_GAMMA),
                   "presets": sweep_presets,
               }],
               "layers": layers, "runs": runs}
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest)


def selected_runs(runs_root, run_names=None, all_runs=False):
    if not runs_root.is_dir():
        raise ValueError(f"Runs root does not exist: {runs_root}")
    if all_runs:
        runs = sorted(
            run for run in runs_root.iterdir()
            if run.is_dir() and (run / "vision_frames").is_dir())
        if not runs:
            raise ValueError(f"No runs with vision_frames found in {runs_root}")
        return runs
    runs = []
    seen = set()
    for name in run_names or ():
        if name in seen:
            continue
        seen.add(name)
        run = runs_root / name
        if not run.is_dir():
            raise ValueError(f"Run does not exist: {run}")
        if not (run / "vision_frames").is_dir():
            raise ValueError(f"Run has no vision_frames directory: {run}")
        runs.append(run)
    return runs


def generate_review(runs, output_root, manifest, lut, start=0, limit=None,
                    static_root=MODELS_ROOT,
                    distance_gamma=DEFAULT_DISTANCE_GAMMA,
                    local_distance_radius=DEFAULT_LOCAL_DISTANCE_RADIUS):
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}-", dir=output_root.parent))
    counts = {}
    try:
        for run in runs:
            counts[run.name] = process_run(
                run, staging, lut, start, limit, distance_gamma,
                local_distance_radius)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
        write_manifest(
            [run.name for run in runs], output_root, manifest, static_root,
            distance_gamma, local_distance_radius)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return counts


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render deterministic_v3_2 component layers for static review.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run", action="append", dest="run_names", metavar="RUN_ID",
        help="Run ID to process; repeat to process multiple runs.")
    selection.add_argument(
        "--all", action="store_true", dest="all_runs",
        help="Process every run containing vision_frames.")
    parser.add_argument("--start", type=int, default=0,
                        help="Zero-based source frame index to start at.")
    parser.add_argument("--limit", type=int,
                        help="Maximum frames to render from each selected run.")
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--output-root", type=Path, default=REVIEWS)
    parser.add_argument("--lut", type=Path, default=LUT_PATH)
    parser.add_argument(
        "--distance-gamma", type=float, default=DEFAULT_DISTANCE_GAMMA,
        help="Distance-transform display exponent; lower values are brighter.")
    parser.add_argument(
        "--local-distance-radius", type=float,
        default=DEFAULT_LOCAL_DISTANCE_RADIUS,
        help="Local distance normalization radius in pixels (default: 9).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start < 0:
        parser.error("--start must be zero or greater")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.distance_gamma <= 0:
        parser.error("--distance-gamma must be greater than zero")
    if args.local_distance_radius <= 0:
        parser.error("--local-distance-radius must be greater than zero")
    try:
        runs = selected_runs(args.runs_root, args.run_names, args.all_runs)
        runs = include_latest_logged_run(runs)
        lut = load_lut(args.lut)
        counts = generate_review(
            runs, args.output_root, MANIFEST, lut, args.start, args.limit,
            distance_gamma=args.distance_gamma,
            local_distance_radius=args.local_distance_radius)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    total = sum(counts.values())
    print(f"Rendered {total} frames from {len(counts)} runs into {args.output_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
