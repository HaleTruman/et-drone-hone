"""Minimal calibrated mask-to-P90-quadrilateral production path."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LUT_PATH = ROOT / "assets" / "color_lut_v1.npz"

SMALL_COMPONENT_AREA = 100
MAX_CONNECTED_COMPONENTS = 500
CLOSE_KERNEL_PX = 5
RELATIVE_CAP = 2.0
INVERSE_GAMMA = 2.10
RIDGE_GAMMA = 3.00
P90_PERCENTILE = 90.0


@dataclass(frozen=True, slots=True)
class RadiusProfile:
    name: str
    maximum_dimension: int | None
    density_radius: int
    ridge_radius: int


RADIUS_PROFILES = (
    RadiusProfile("compact", 35, 2, 2),
    RadiusProfile("small", 59, 4, 3),
    RadiusProfile("medium", 89, 7, 6),
    RadiusProfile("large", None, 20, 16),
)


@dataclass(frozen=True, slots=True)
class ComponentInput:
    label: int
    bbox: tuple[int, int, int, int]
    area: int
    profile: RadiusProfile
    mask: np.ndarray
    offset: tuple[int, int]


@dataclass(frozen=True, slots=True)
class P90Fit:
    threshold: float
    evidence_points: int
    corners: np.ndarray


@dataclass(frozen=True, slots=True)
class DensityQuadrilateral:
    label: int
    bbox: tuple[int, int, int, int]
    area: int
    profile: RadiusProfile
    threshold: float
    evidence_points: int
    corners: np.ndarray


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    gated: bool
    input_components: int
    retained_components: int
    quadrilaterals: tuple[DensityQuadrilateral, ...]


def load_lut(path=LUT_PATH):
    with np.load(path) as archive:
        lut = archive["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"Unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut


def radius_profile(maximum_dimension):
    for profile in RADIUS_PROFILES:
        if (profile.maximum_dimension is None or
                maximum_dimension <= profile.maximum_dimension):
            return profile
    raise AssertionError("The final radius profile must be unbounded")


def image_to_mask(image, lut):
    bgr = image.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return (lut[keys] != 0).astype(np.uint8)


def prepare_mask(image, lut):
    """Apply the component gate, area filter, and one square close."""
    mask = image_to_mask(image, lut)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    input_components = count - 1
    if input_components >= MAX_CONNECTED_COMPONENTS:
        return np.zeros(mask.shape, np.uint8), True, input_components
    keep = stats[:, cv2.CC_STAT_AREA] >= SMALL_COMPONENT_AREA
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    return (cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), False,
            input_components)


def iter_component_inputs(mask):
    """Yield isolated square component masks with their calibrated profile."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    for label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[label])
        side = max(width, height)
        offset_x = (side - width) // 2
        offset_y = (side - height) // 2
        component = labels[y:y + height, x:x + width] == label
        square = np.zeros((side, side), np.uint8)
        square[offset_y:offset_y + height,
               offset_x:offset_x + width] = component
        yield ComponentInput(
            label, (x, y, width, height), area, radius_profile(side), square,
            (offset_x, offset_y))


def _box_sum(values, radius, ddepth=-1):
    side = 2 * radius + 1
    return cv2.boxFilter(
        values, ddepth, (side, side), normalize=False,
        borderType=cv2.BORDER_CONSTANT)


@lru_cache(maxsize=64)
def _coordinate_grids(height, width):
    return np.indices((height, width), dtype=np.float64)


@lru_cache(maxsize=64)
def _square_denominator(height, width, radius):
    return _box_sum(
        np.ones((height, width), np.float32), radius, cv2.CV_32F)


def raw_square_density(mask, radius):
    """Return foreground density in a clipped (2r+1) by (2r+1) square."""
    inside = mask != 0
    foreground = inside.astype(np.float32)
    density = _box_sum(foreground, radius, cv2.CV_32F)
    density /= _square_denominator(*mask.shape, radius)
    density[~inside] = 0
    return density


def inverse_density_field(mask, density_radius, ridge_radius):
    """Compute the optimized original inverse-density and ridge formula."""
    inside = mask != 0
    if not np.any(inside):
        return np.zeros(mask.shape, np.float64)

    density = raw_square_density(mask, density_radius)
    normalized = np.zeros_like(density, np.float32)
    mean_density = float(density[inside].mean())
    normalized[inside] = np.clip(
        density[inside] / (mean_density * RELATIVE_CAP), 0, 1)
    inverse = np.zeros_like(density, np.float32)
    inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / INVERSE_GAMMA)

    yy, xx = _coordinate_grids(*mask.shape)
    moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
    sums = _box_sum(moments, ridge_radius)
    mass, weighted_x, weighted_y = cv2.split(sums)
    centroid_x = np.divide(
        weighted_x, mass, out=np.zeros_like(mass), where=mass > 0)
    centroid_y = np.divide(
        weighted_y, mass, out=np.zeros_like(mass), where=mass > 0)
    distance = cv2.magnitude(xx - centroid_x, yy - centroid_y)
    ridge = np.power(
        1.0 - np.clip(distance / ridge_radius, 0, 1), RIDGE_GAMMA)
    ridge[(~inside) | (mass <= 0)] = 0
    return inverse * ridge


def _ordered(points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1],
                        points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _segment_distance(points, start, end):
    edge = end - start
    length_squared = float(np.dot(edge, edge))
    if length_squared <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    position = np.clip(((points - start) @ edge) / length_squared, 0, 1)
    projection = start + position[:, None] * edge
    return np.linalg.norm(points - projection, axis=1)


def _intersect(first, second):
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


def _valid_quadrilateral(corners, shape):
    if corners is None or len(corners) != 4 or not np.all(np.isfinite(corners)):
        return False
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1.0:
        return False
    height, width = shape
    margin = max(height, width)
    return bool(
        np.all(corners[:, 0] >= -margin) and
        np.all(corners[:, 0] <= width + margin) and
        np.all(corners[:, 1] >= -margin) and
        np.all(corners[:, 1] <= height + margin))


def fit_p90_quadrilateral(mask, field, percentile=P90_PERCENTILE):
    """Fit four ordered sides to the highest positive field percentile."""
    positive = field[(mask != 0) & (field > 0)]
    if positive.size < 4:
        return None
    threshold = float(np.percentile(positive, percentile))
    yy, xx = np.nonzero((mask != 0) & (field >= threshold))
    evidence = np.column_stack((xx, yy)).astype(np.float64)
    if len(evidence) < 4:
        return None
    hull = cv2.convexHull(evidence.astype(np.float32).reshape(-1, 1, 2))
    if len(hull) < 4 or cv2.contourArea(hull) < 1.0:
        return None
    initial = cv2.approxPolyN(
        hull, 4, epsilon_percentage=-1, ensure_convex=True).reshape(-1, 2)
    if len(initial) != 4:
        return None
    initial = _ordered(initial)

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
    fitted = [_intersect(lines[index], lines[(index + 1) % 4])
              for index in range(4)]
    if all(point is not None for point in fitted):
        fitted = _ordered(fitted)
    else:
        fitted = None
    corners = fitted if _valid_quadrilateral(fitted, mask.shape) else initial
    if not _valid_quadrilateral(corners, mask.shape):
        return None
    return P90Fit(threshold, len(evidence), corners)


def detect_density_quadrilaterals(image, lut):
    """Return calibrated P90 quadrilaterals without rendering or file output."""
    mask, gated, input_components = prepare_mask(image, lut)
    if gated:
        return DetectionBatch(True, input_components, 0, ())

    components = list(iter_component_inputs(mask))
    results = []
    for component in components:
        profile = component.profile
        field = inverse_density_field(
            component.mask, profile.density_radius, profile.ridge_radius)
        fit = fit_p90_quadrilateral(component.mask, field)
        if fit is None:
            continue
        x, y, _, _ = component.bbox
        offset_x, offset_y = component.offset
        corners = (fit.corners - np.array((offset_x, offset_y)) +
                   np.array((x, y)))
        results.append(DensityQuadrilateral(
            component.label, component.bbox, component.area, profile,
            fit.threshold, fit.evidence_points, corners))
    return DetectionBatch(
        False, input_components, len(components), tuple(results))
