"""Minimal production path for the calibrated legacy inverse-density field."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LUT_PATH = ROOT / "assets" / "color_lut_v1.npz"
EXTENSIONS = {".png", ".jpg", ".jpeg"}

SMALL_COMPONENT_AREA = 100
MAX_CONNECTED_COMPONENTS = 500
CLOSE_KERNEL_PX = 5
DENSITY_RADIUS_PX = 20
RELATIVE_CAP = 2.0
INVERSE_GAMMA = 2.10
RIDGE_RADIUS_PX = 16
RIDGE_GAMMA = 3.00
COMPONENT_RADIUS_BUCKETS = (
    ("compact", 35, 2, 2),
    ("small", 59, 4, 3),
    ("medium", 89, 7, 6),
    ("large", None, DENSITY_RADIUS_PX, RIDGE_RADIUS_PX),
)
CENTERLINE_OFFSET_SAMPLES = 15
CENTERLINE_EDGE_MARGIN = 0.15
CENTERLINE_MIDPOINT_PENALTY = 0.025
CENTERLINE_TUBE_SIGMA_PX = 0.75
DENSITY_MARKER_PERCENTILE = 90.0


@dataclass
class DensityGuidedCenterline:
    """Straight-sided contour centerline before and after density guidance."""

    outer_contour: np.ndarray
    inner_contour: np.ndarray
    geometric_corners: np.ndarray
    density_corners: np.ndarray
    density_offsets: tuple
    density_coverages: tuple


@dataclass
class DensityPercentileMarker:
    """Four straight sides fitted only to the highest-density pixels."""

    corners: np.ndarray
    threshold: float
    point_count: int


def load_lut(path=LUT_PATH):
    with np.load(path) as archive:
        lut = archive["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"Unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut


def _box_sum(values, radius, ddepth=-1):
    window = 2 * radius + 1
    return cv2.boxFilter(
        values, ddepth, (window, window), normalize=False,
        borderType=cv2.BORDER_CONSTANT)


@lru_cache(maxsize=64)
def _coordinate_grids(height, width):
    return np.indices((height, width), dtype=np.float64)


@lru_cache(maxsize=64)
def _density_denominator(height, width, radius):
    return _box_sum(
        np.ones((height, width), np.float32), radius, cv2.CV_32F)


def component_radius_bucket(max_dimension):
    for bucket in COMPONENT_RADIUS_BUCKETS:
        if bucket[1] is None or max_dimension <= bucket[1]:
            return bucket
    raise AssertionError("The final component-radius bucket must be unbounded")


def _field_from_mask(mask, density_radius, ridge_radius):
    inside = mask != 0
    if not np.any(inside):
        return np.zeros(mask.shape, np.float64)

    foreground = inside.astype(np.float32)
    density = _box_sum(foreground, density_radius, cv2.CV_32F)
    density /= _density_denominator(*mask.shape, density_radius)
    density[~inside] = 0

    normalized = np.zeros_like(density, np.float32)
    mean_density = float(density[inside].mean())
    normalized[inside] = np.clip(
        density[inside] / (mean_density * RELATIVE_CAP), 0, 1)
    inverse = np.zeros_like(density, np.float32)
    inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / INVERSE_GAMMA)

    yy, xx = _coordinate_grids(*mask.shape)
    moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
    moment_sums = _box_sum(moments, ridge_radius)
    mass = moment_sums[:, :, 0]
    weighted_x = moment_sums[:, :, 1]
    weighted_y = moment_sums[:, :, 2]
    centroid_x = np.divide(
        weighted_x, mass, out=np.zeros_like(mass), where=mass > 0)
    centroid_y = np.divide(
        weighted_y, mass, out=np.zeros_like(mass), where=mass > 0)
    distance = cv2.magnitude(xx - centroid_x, yy - centroid_y)
    ridge = np.power(
        1.0 - np.clip(distance / ridge_radius, 0, 1), RIDGE_GAMMA)
    ridge[(~inside) | (mass <= 0)] = 0
    return inverse * ridge


def _apply_component_buckets(mask, global_field):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    for label in range(1, count):
        x, y, width, height, _ = (
            int(value) for value in stats[label])
        bucket_name, _, density_radius, ridge_radius = (
            component_radius_bucket(max(width, height)))
        if bucket_name == "large":
            continue

        component = labels[y:y + height, x:x + width] == label
        side = max(width, height)
        offset_x = (side - width) // 2
        offset_y = (side - height) // 2
        square = np.zeros((side, side), np.uint8)
        square[offset_y:offset_y + height,
               offset_x:offset_x + width] = component
        local_field = _field_from_mask(
            square, density_radius, ridge_radius)
        local_crop = local_field[
            offset_y:offset_y + height, offset_x:offset_x + width]
        destination = global_field[y:y + height, x:x + width]
        destination[component] = local_crop[component]
    return global_field


def _prepared_mask(image, lut):
    bgr = image.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    mask = (lut[keys] != 0).astype(np.uint8)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if stats.shape[0] - 1 >= MAX_CONNECTED_COMPONENTS:
        return np.zeros(mask.shape, np.uint8), True
    keep = stats[:, cv2.CC_STAT_AREA] >= SMALL_COMPONENT_AREA
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, False


def _field_for_prepared_mask(mask):
    if not np.any(mask):
        return np.zeros(mask.shape, np.float64)

    global_field = _field_from_mask(
        mask, DENSITY_RADIUS_PX, RIDGE_RADIUS_PX)
    return _apply_component_buckets(mask, global_field)


def final_inverse_density(image, lut):
    """Return the fixed-bucket calibrated field at native pixel resolution."""
    mask, _ = _prepared_mask(image, lut)
    return _field_for_prepared_mask(mask)


def _ordered_quadrilateral(contour):
    hull = cv2.convexHull(contour)
    if len(hull) >= 4:
        points = cv2.approxPolyN(
            hull, 4, epsilon_percentage=-1, ensure_convex=True).reshape(-1, 2)
    else:
        points = cv2.boxPoints(cv2.minAreaRect(contour))
    if len(points) != 4:
        return None
    points = points.astype(np.float64)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _point_segment_distance(points, start, end):
    edge = end - start
    length_squared = float(np.dot(edge, edge))
    if length_squared <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    position = np.clip(((points - start) @ edge) / length_squared, 0, 1)
    projection = start + position[:, None] * edge
    return np.linalg.norm(points - projection, axis=1)


def _fit_point_sides(points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    quadrilateral = _ordered_quadrilateral(contour)
    if quadrilateral is None:
        return None
    distances = np.column_stack([
        _point_segment_distance(
            points, quadrilateral[index], quadrilateral[(index + 1) % 4])
        for index in range(4)
    ])
    assignments = np.argmin(distances, axis=1)
    sides = []
    for index in range(4):
        side_points = points[assignments == index]
        if len(side_points) < 2:
            side_points = quadrilateral[[index, (index + 1) % 4]]
        vx, vy, x, y = cv2.fitLine(
            side_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        direction = np.array((float(vx[0]), float(vy[0])), np.float64)
        direction /= np.linalg.norm(direction)
        sides.append((np.array((float(x[0]), float(y[0]))), direction,
                      side_points))
    return sides


def _fit_contour_sides(contour):
    return _fit_point_sides(contour.reshape(-1, 2))


def _paired_sides(outer_sides, inner_sides, shape):
    orders = []
    forward = list(range(4))
    reverse = list(reversed(forward))
    for base in (forward, reverse):
        orders.extend(base[offset:] + base[:offset] for offset in range(4))
    diagonal = max(float(np.hypot(*shape)), 1.0)

    def score(order):
        total = 0.0
        for outer, inner_index in zip(outer_sides, order):
            inner = inner_sides[inner_index]
            separation = np.linalg.norm(outer[0] - inner[0]) / diagonal
            alignment = 1.0 - abs(float(np.dot(outer[1], inner[1])))
            total += separation + 2.0 * alignment
        return total

    order = min(orders, key=score)
    return [(outer, inner_sides[inner_index])
            for outer, inner_index in zip(outer_sides, order)]


def _average_side(outer, inner):
    outer_center, outer_direction, _ = outer
    inner_center, inner_direction, _ = inner
    if np.dot(outer_direction, inner_direction) < 0:
        inner_direction = -inner_direction
    direction = outer_direction + inner_direction
    direction /= np.linalg.norm(direction)
    normal = np.array((-direction[1], direction[0]))
    offset = 0.5 * (np.dot(outer_center, normal) +
                    np.dot(inner_center, normal))
    tangent = 0.5 * (np.dot(outer_center, direction) +
                     np.dot(inner_center, direction))
    return direction * tangent + normal * offset, direction


def _line_intersection(first, second):
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


def _line_corners(lines):
    corners = [_line_intersection(lines[index], lines[(index + 1) % 4])
               for index in range(4)]
    if any(corner is None or not np.all(np.isfinite(corner))
           for corner in corners):
        return None
    return np.asarray(corners, np.float64)


def _density_guided_side(outer, inner, component, density):
    geometric_point, direction = _average_side(outer, inner)
    normal = np.array((-direction[1], direction[0]))
    outer_offset = float(np.dot(outer[0], normal))
    inner_offset = float(np.dot(inner[0], normal))
    if abs(outer_offset - inner_offset) < 1e-6:
        return (geometric_point, direction), 0.5, 0.0

    points = np.column_stack(np.nonzero(component))[:, ::-1].astype(np.float64)
    values = density[component].astype(np.float64)
    tangent_positions = points @ direction
    outer_tangent = outer[2] @ direction
    inner_tangent = inner[2] @ direction
    start = max(float(outer_tangent.min()), float(inner_tangent.min())) - 1.0
    end = min(float(outer_tangent.max()), float(inner_tangent.max())) + 1.0
    if start >= end:
        start = min(float(outer_tangent.min()), float(inner_tangent.min()))
        end = max(float(outer_tangent.max()), float(inner_tangent.max()))
    low_offset, high_offset = sorted((outer_offset, inner_offset))
    normal_positions = points @ normal
    band = ((tangent_positions >= start) & (tangent_positions <= end) &
            (normal_positions >= low_offset - 1.0) &
            (normal_positions <= high_offset + 1.0))
    if not np.any(band) or float(values[band].sum()) <= 1e-12:
        return (geometric_point, direction), 0.5, 0.0

    band_normal = normal_positions[band]
    band_values = values[band]
    total_density = float(band_values.sum())
    candidates = np.linspace(
        CENTERLINE_EDGE_MARGIN, 1.0 - CENTERLINE_EDGE_MARGIN,
        CENTERLINE_OFFSET_SAMPLES)
    best = None
    for position in candidates:
        offset = inner_offset + position * (outer_offset - inner_offset)
        weights = np.exp(-0.5 * np.square(
            (band_normal - offset) / CENTERLINE_TUBE_SIGMA_PX))
        coverage = float(np.dot(band_values, weights) / total_density)
        score = coverage - CENTERLINE_MIDPOINT_PENALTY * np.square(
            2.0 * (position - 0.5))
        if best is None or score > best[0]:
            best = score, position, offset, coverage
    _, position, offset, coverage = best
    tangent = float(np.dot(geometric_point, direction))
    return (direction * tangent + normal * offset, direction), \
        float(position), float(coverage)


def fit_density_guided_centerline(component, density):
    """Fit a four-sided centerline bounded by contours and guided by density."""
    component = (component != 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    outer_indices = [index for index, item in enumerate(hierarchy)
                     if item[3] < 0]
    if not outer_indices:
        return None
    outer_index = max(outer_indices, key=lambda index: cv2.contourArea(
        contours[index]))
    inner_indices = [index for index, item in enumerate(hierarchy)
                     if item[3] == outer_index]
    if not inner_indices:
        return None
    inner_index = max(inner_indices, key=lambda index: cv2.contourArea(
        contours[index]))
    outer_contour = contours[outer_index]
    inner_contour = contours[inner_index]
    if cv2.contourArea(inner_contour) < 4:
        return None

    outer_sides = _fit_contour_sides(outer_contour)
    inner_sides = _fit_contour_sides(inner_contour)
    if outer_sides is None or inner_sides is None:
        return None
    pairs = _paired_sides(outer_sides, inner_sides, component.shape)
    geometric_lines = [_average_side(*pair) for pair in pairs]
    guided = [_density_guided_side(
        *pair, component=component != 0, density=density) for pair in pairs]
    geometric_corners = _line_corners(geometric_lines)
    density_corners = _line_corners([item[0] for item in guided])
    if geometric_corners is None or density_corners is None:
        return None
    height, width = component.shape
    margin = max(height, width)
    if np.any(density_corners[:, 0] < -margin) or \
            np.any(density_corners[:, 0] > width + margin) or \
            np.any(density_corners[:, 1] < -margin) or \
            np.any(density_corners[:, 1] > height + margin):
        return None
    return DensityGuidedCenterline(
        outer_contour, inner_contour, geometric_corners, density_corners,
        tuple(item[1] for item in guided),
        tuple(item[2] for item in guided))


def fit_density_percentile_marker(
        component, density, percentile=DENSITY_MARKER_PERCENTILE):
    """Fit a quadrilateral directly to the highest-density component pixels."""
    component = component != 0
    values = density[component]
    positive = values[values > 0]
    if positive.size < 4:
        return None
    threshold = float(np.percentile(positive, percentile))
    yy, xx = np.nonzero(component & (density >= threshold))
    points = np.column_stack((xx, yy)).astype(np.float64)
    if len(points) < 4 or cv2.contourArea(
            cv2.convexHull(points.astype(np.float32))) < 1.0:
        return None
    sides = _fit_point_sides(points)
    if sides is None:
        return None
    corners = _line_corners([(side[0], side[1]) for side in sides])
    if corners is None or abs(cv2.contourArea(
            corners.astype(np.float32))) < 1.0:
        return None
    height, width = component.shape
    margin = max(height, width)
    if np.any(corners[:, 0] < -margin) or \
            np.any(corners[:, 0] > width + margin) or \
            np.any(corners[:, 1] < -margin) or \
            np.any(corners[:, 1] > height + margin):
        return None
    return DensityPercentileMarker(corners, threshold, len(points))


def frame_paths(folder):
    return sorted(path for path in folder.iterdir()
                  if path.suffix.lower() in EXTENSIONS)


def _centerline_tile(component, density, fit, title, subtitle, size=320):
    header = 48
    tile = np.zeros((size, size, 3), np.uint8)
    maximum = float(density[component].max()) if np.any(component) else 0.0
    normalized = np.zeros(component.shape, np.uint8)
    if maximum > 0:
        normalized[component] = np.clip(
            density[component] * 255.0 / maximum, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heat[~component] = 0
    available = size - header - 8
    scale = min((size - 8) / component.shape[1],
                available / component.shape[0])
    scaled_size = (max(1, int(round(component.shape[1] * scale))),
                   max(1, int(round(component.shape[0] * scale))))
    rendered = cv2.resize(heat, scaled_size, interpolation=cv2.INTER_NEAREST)
    origin = ((size - scaled_size[0]) // 2,
              header + (available - scaled_size[1]) // 2)
    x0, y0 = origin
    tile[y0:y0 + scaled_size[1], x0:x0 + scaled_size[0]] = rendered

    def transform(points):
        values = np.asarray(points, np.float64).reshape(-1, 2)
        values = values * scale + np.array((x0, y0))
        return np.rint(values).astype(np.int32).reshape(-1, 1, 2)

    cv2.drawContours(tile, [transform(fit.outer_contour)], -1,
                     (210, 210, 210), 1, cv2.LINE_AA)
    cv2.drawContours(tile, [transform(fit.inner_contour)], -1,
                     (255, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(tile, [transform(fit.geometric_corners)], True,
                  (255, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(tile, [transform(fit.density_corners)], True,
                  (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, title, (5, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(tile, subtitle, (5, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.29, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(tile, "midpoint=magenta  density=yellow", (5, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (175, 175, 175), 1,
                cv2.LINE_AA)
    return tile


def _density_marker_tile(component, density, marker, title, subtitle, size=320):
    header = 48
    tile = np.zeros((size, size, 3), np.uint8)
    maximum = float(density[component].max()) if np.any(component) else 0.0
    normalized = np.zeros(component.shape, np.uint8)
    if maximum > 0:
        normalized[component] = np.clip(
            density[component] * 255.0 / maximum, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heat[~component] = 0
    available = size - header - 8
    scale = min((size - 8) / component.shape[1],
                available / component.shape[0])
    scaled_size = (max(1, int(round(component.shape[1] * scale))),
                   max(1, int(round(component.shape[0] * scale))))
    rendered = cv2.resize(heat, scaled_size, interpolation=cv2.INTER_NEAREST)
    x0 = (size - scaled_size[0]) // 2
    y0 = header + (available - scaled_size[1]) // 2
    tile[y0:y0 + scaled_size[1], x0:x0 + scaled_size[0]] = rendered
    corners = np.rint(
        marker.corners * scale + np.array((x0, y0))).astype(np.int32)
    cv2.polylines(tile, [corners.reshape(-1, 1, 2)], True,
                  (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, title, (5, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(tile, subtitle, (5, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.29, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(tile, "P90 highest-density marker=yellow", (5, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (175, 175, 175), 1,
                cv2.LINE_AA)
    return tile


def generate_centerline_review(run, lut, output, start=0, limit=None,
                               columns=5, rows=4):
    """Write every valid density-guided centerline and paged contact sheets."""
    frames_folder = run / "vision_frames" if (run / "vision_frames").is_dir() else run
    selected = frame_paths(frames_folder)[
        start:start + limit if limit is not None else None]
    if not selected:
        raise ValueError(f"No selected frames in {frames_folder}")
    instances_folder = output / "instances"
    sheets_folder = output / "contact_sheets"
    instances_folder.mkdir(parents=True, exist_ok=True)
    sheets_folder.mkdir(parents=True, exist_ok=True)

    page_capacity = columns * rows
    page_tiles = []
    page_number = 0
    manifest_instances = []
    total_components = skipped_without_hole = gated_frames = 0

    def flush_page():
        nonlocal page_number
        if not page_tiles:
            return
        sheet = np.zeros((rows * 320, columns * 320, 3), np.uint8)
        for tile_index, tile in enumerate(page_tiles):
            row, column = divmod(tile_index, columns)
            sheet[row * 320:(row + 1) * 320,
                  column * 320:(column + 1) * 320] = tile
        page_number += 1
        cv2.imwrite(str(sheets_folder / f"page-{page_number:03d}.png"), sheet)
        page_tiles.clear()

    for frame in selected:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read {frame}")
        mask, gated = _prepared_mask(image, lut)
        if gated:
            gated_frames += 1
            continue
        density = _field_for_prepared_mask(mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        total_components += count - 1
        for label in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[label])
            component = labels[y:y + height, x:x + width] == label
            local_density = density[y:y + height, x:x + width]
            fit = fit_density_guided_centerline(component, local_density)
            if fit is None:
                skipped_without_hole += 1
                continue
            bucket = component_radius_bucket(max(width, height))[0]
            number = len(manifest_instances) + 1
            name = (f"{number:05d}__{frame.stem}__label-{label:03d}__"
                    f"{bucket}-{width}x{height}.png")
            offsets = np.asarray(fit.density_offsets)
            frame_id = frame.stem.split("-")[1]
            title = f"{number:05d}  label={label:03d}  frame={frame_id}"
            subtitle = (f"{bucket} {width}x{height} area={area}  "
                        f"offset={offsets.mean():.2f} [{offsets.min():.2f},"
                        f"{offsets.max():.2f}]")
            tile = _centerline_tile(
                component, local_density, fit, title, subtitle)
            if not cv2.imwrite(str(instances_folder / name), tile):
                raise OSError(f"Unable to write {instances_folder / name}")
            page_tiles.append(tile)
            manifest_instances.append({
                "file": f"instances/{name}", "frame": frame.name,
                "label": label, "bucket": bucket, "bbox": [x, y, width, height],
                "area": area, "density_offsets": list(fit.density_offsets),
                "density_coverages": list(fit.density_coverages),
            })
            if len(page_tiles) == page_capacity:
                flush_page()
    flush_page()
    summary = {
        "run": str(run), "frames": len(selected),
        "gated_frames": gated_frames, "components": total_components,
        "eligible_instances": len(manifest_instances),
        "skipped_without_valid_inner_contour": skipped_without_hole,
        "profile": {
            "offset_samples": CENTERLINE_OFFSET_SAMPLES,
            "edge_margin": CENTERLINE_EDGE_MARGIN,
            "midpoint_penalty": CENTERLINE_MIDPOINT_PENALTY,
            "tube_sigma_px": CENTERLINE_TUBE_SIGMA_PX,
        },
        "instances": manifest_instances,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def generate_density_percentile_review(
        run, lut, output, start=0, limit=None,
        percentile=DENSITY_MARKER_PERCENTILE, columns=5, rows=4):
    """Write P90 markers grouped by the one calibrated radius bucket used."""
    frames_folder = run / "vision_frames" if (run / "vision_frames").is_dir() else run
    selected = frame_paths(frames_folder)[
        start:start + limit if limit is not None else None]
    if not selected:
        raise ValueError(f"No selected frames in {frames_folder}")
    profiles = {
        name: {"density_radius": density_radius, "ridge_radius": ridge_radius}
        for name, _, density_radius, ridge_radius in COMPONENT_RADIUS_BUCKETS
    }
    instances_folder = output / "instances_by_radius"
    sheets_folder = output / "contact_sheets_by_radius"
    for bucket, profile in profiles.items():
        suffix = (f"{bucket}__density-r{profile['density_radius']}__"
                  f"ridge-r{profile['ridge_radius']}")
        (instances_folder / suffix).mkdir(parents=True, exist_ok=True)
        (sheets_folder / suffix).mkdir(parents=True, exist_ok=True)

    page_capacity = columns * rows
    page_tiles = {bucket: [] for bucket in profiles}
    page_numbers = {bucket: 0 for bucket in profiles}
    manifest_instances = []
    total_components = skipped_without_inner = skipped_without_fit = gated_frames = 0

    def flush_page(bucket):
        tiles = page_tiles[bucket]
        if not tiles:
            return
        sheet = np.zeros((rows * 320, columns * 320, 3), np.uint8)
        for tile_index, tile in enumerate(tiles):
            row, column = divmod(tile_index, columns)
            sheet[row * 320:(row + 1) * 320,
                  column * 320:(column + 1) * 320] = tile
        page_numbers[bucket] += 1
        profile = profiles[bucket]
        suffix = (f"{bucket}__density-r{profile['density_radius']}__"
                  f"ridge-r{profile['ridge_radius']}")
        cv2.imwrite(str(sheets_folder / suffix /
                        f"page-{page_numbers[bucket]:03d}.png"), sheet)
        tiles.clear()

    for frame in selected:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read {frame}")
        mask, gated = _prepared_mask(image, lut)
        if gated:
            gated_frames += 1
            continue
        density = _field_for_prepared_mask(mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        total_components += count - 1
        for label in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[label])
            component = labels[y:y + height, x:x + width] == label
            local_density = density[y:y + height, x:x + width]
            if fit_density_guided_centerline(component, local_density) is None:
                skipped_without_inner += 1
                continue
            marker = fit_density_percentile_marker(
                component, local_density, percentile)
            if marker is None:
                skipped_without_fit += 1
                continue
            bucket = component_radius_bucket(max(width, height))[0]
            profile = profiles[bucket]
            suffix = (f"{bucket}__density-r{profile['density_radius']}__"
                      f"ridge-r{profile['ridge_radius']}")
            number = len(manifest_instances) + 1
            name = (f"{number:05d}__{frame.stem}__label-{label:03d}__"
                    f"{bucket}-{width}x{height}.png")
            frame_id = frame.stem.split("-")[1]
            title = f"{number:05d}  label={label:03d}  frame={frame_id}"
            subtitle = (f"{bucket} {width}x{height} area={area}  "
                        f"density-r={profile['density_radius']} "
                        f"ridge-r={profile['ridge_radius']} P{percentile:g}")
            tile = _density_marker_tile(
                component, local_density, marker, title, subtitle)
            instance_path = instances_folder / suffix / name
            if not cv2.imwrite(str(instance_path), tile):
                raise OSError(f"Unable to write {instance_path}")
            page_tiles[bucket].append(tile)
            manifest_instances.append({
                "file": str(instance_path.relative_to(output)),
                "frame": frame.name,
                "label": label, "bucket": bucket, "bbox": [x, y, width, height],
                "area": area, "threshold": marker.threshold,
                "point_count": marker.point_count,
                "density_radius": profile["density_radius"],
                "ridge_radius": profile["ridge_radius"],
                "corners": marker.corners.tolist(),
            })
            if len(page_tiles[bucket]) == page_capacity:
                flush_page(bucket)
    for bucket in profiles:
        flush_page(bucket)
    summary = {
        "run": str(run), "frames": len(selected),
        "gated_frames": gated_frames, "components": total_components,
        "fitted_instances": len(manifest_instances),
        "skipped_outside_ring_comparison_population": skipped_without_inner,
        "skipped_without_density_fit": skipped_without_fit,
        "profile": {
            "percentile": percentile, "evidence": "positive field values",
            "radius_buckets": profiles,
        },
        "instances": manifest_instances,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def process_run(run, lut, start=0, limit=None, output=None):
    frames_folder = run / "vision_frames" if (run / "vision_frames").is_dir() else run
    frames = frame_paths(frames_folder)
    selected = frames[start:start + limit if limit is not None else None]
    if not selected:
        raise ValueError(f"No selected frames in {frames_folder}")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    total_mass = 0.0
    for frame in selected:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read {frame}")
        field = final_inverse_density(image, lut)
        total_mass += float(field.sum(dtype=np.float64))
        if output is not None:
            np.save(output / f"{frame.stem}.npy", field)
    return len(selected), total_mass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or vision_frames directory")
    parser.add_argument("--lut", type=Path, default=LUT_PATH)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path,
                        help="Optional directory for float64 .npy fields")
    parser.add_argument("--centerline-review", type=Path,
                        help="Write density-guided instance crops and contact sheets")
    parser.add_argument("--density-percentile-review", type=Path,
                        help="Write density-only P90 markers and contact sheets")
    args = parser.parse_args(argv)
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("start must be nonnegative and limit must be positive")
    output_modes = (args.output, args.centerline_review,
                    args.density_percentile_review)
    if sum(mode is not None for mode in output_modes) > 1:
        parser.error("output and review options are separate output modes")
    try:
        if args.density_percentile_review is not None:
            summary = generate_density_percentile_review(
                args.run, load_lut(args.lut), args.density_percentile_review,
                args.start, args.limit)
            print(f"Reviewed {summary['fitted_instances']} P90 markers from "
                  f"{summary['components']} components across "
                  f"{summary['frames']} frames")
            return 0
        if args.centerline_review is not None:
            summary = generate_centerline_review(
                args.run, load_lut(args.lut), args.centerline_review,
                args.start, args.limit)
            print(f"Reviewed {summary['eligible_instances']} centerlines from "
                  f"{summary['components']} components across "
                  f"{summary['frames']} frames")
            return 0
        count, total_mass = process_run(
            args.run, load_lut(args.lut), args.start, args.limit, args.output)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Computed {count} final fields; checksum={total_mass:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
