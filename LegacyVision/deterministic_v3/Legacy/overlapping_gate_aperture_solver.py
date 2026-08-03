"""Standalone first-pass solver for known two-aperture gate overlaps.

Each visible aperture seeds one quadrilateral. Its four sides are moved outward
through a bounded one-dimensional search for coherent, parallel density support.
The larger aperture is solved first; the smaller aperture may reuse a larger
gate side only through a parallel-line compatibility bonus. No pixels are
subtracted or exclusively assigned.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT.parents[4] / "Logs" / "flight" / "runs" / \
    "run-20260801T031401Z" / "vision_frames"
DEFAULT_OUTPUT = ROOT / "production_samples" / \
    "two-aperture-overlap-solver-v1"
LUT_PATH = ROOT / "assets" / "color_lut_v1.npz"
DEFAULT_SCENARIOS = ("00106710:3", "00107025:1", "00107026:1")

SMALL_COMPONENT_AREA = 100
MAX_CONNECTED_COMPONENTS = 500
CLOSE_KERNEL_PX = 5
RELATIVE_CAP = 2.0
INVERSE_GAMMA = 2.10
RIDGE_GAMMA = 3.00
MIN_HOLE_AREA = 12.0
SMALL_APERTURE_RECT_AREA = 250.0
CONCAVE_APERTURE_SOLIDITY = 0.88
SHARED_ANGLE_DEGREES = 12.0
SHARED_LINE_DISTANCE_PX = 3.5
CONSTRAINED_CLOSE_SEED_PERCENTILE = 90.0
CONSTRAINED_CLOSE_LIMIT_PERCENTILE = 70.0
CONSTRAINED_CLOSE_KERNEL_PX = 3
SIDE_BRIDGE_MAX_GAP_PX = 3
SIDE_BRIDGE_HALF_WIDTH_PX = 1
COVERAGE_TOLERANCE_PX = 1.25
COVERAGE_INITIAL_SCALES = (0.95, 1.00, 1.05, 1.10, 1.15)
COVERAGE_EXTENSION_SCALES = (1.20, 1.25)
COVERAGE_ROTATIONS_DEG = (-4.0, 0.0, 4.0)
COVERAGE_TRANSLATIONS_PX = (-2.0, 0.0, 2.0)
RADIUS_PROFILES = (
    ("compact", 35, 2, 2),
    ("small", 59, 4, 3),
    ("medium", 89, 7, 6),
    ("large", None, 20, 16),
)

PANEL = 260
HEADER = 88
GATE_COLORS = ((0, 255, 255), (255, 0, 255))


@dataclass(frozen=True, slots=True)
class SideFit:
    point: np.ndarray
    direction: np.ndarray
    offset: float
    score: float
    coverage: float
    continuity: float
    shared_parallel: bool


@dataclass(frozen=True, slots=True)
class GateFit:
    aperture_area: float
    aperture_corners: np.ndarray
    sides: tuple
    corners: np.ndarray


@dataclass(frozen=True, slots=True)
class CoverageCandidate:
    corners: np.ndarray
    scale: float
    rotation_degrees: float
    translation: tuple
    weighted_coverage: float
    covered_points: int
    side_coverages: tuple
    additions: np.ndarray
    bridge_records: tuple


def load_lut(path=LUT_PATH):
    with np.load(path) as archive:
        lut = archive["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"Unexpected LUT: {lut.shape} {lut.dtype}")
    return lut


def radius_profile(maximum_dimension):
    for profile in RADIUS_PROFILES:
        if profile[1] is None or maximum_dimension <= profile[1]:
            return profile
    raise AssertionError("Final profile must be unbounded")


def prepare_mask(image, lut):
    bgr = image.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    mask = (lut[keys] != 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if count - 1 >= MAX_CONNECTED_COMPONENTS:
        return np.zeros(mask.shape, np.uint8), True
    keep = stats[:, cv2.CC_STAT_AREA] >= SMALL_COMPONENT_AREA
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), False


def isolate_component(image, lut, label):
    mask, gated = prepare_mask(image, lut)
    if gated:
        raise RuntimeError("Frame exceeded the connected-component gate")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if label <= 0 or label >= count:
        raise ValueError(f"Component label {label} not present")
    x, y, width, height, area = (int(value) for value in stats[label])
    side = max(width, height)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    component = labels[y:y + height, x:x + width] == label
    square = np.zeros((side, side), np.uint8)
    square[offset_y:offset_y + height,
           offset_x:offset_x + width] = component
    source = np.zeros((side, side, 3), np.uint8)
    source[offset_y:offset_y + height,
           offset_x:offset_x + width] = image[y:y + height, x:x + width]
    return square, source, (x, y, width, height, area)


def box_sum(values, radius, ddepth=-1):
    window = 2 * radius + 1
    return cv2.boxFilter(
        values, ddepth, (window, window), normalize=False,
        borderType=cv2.BORDER_CONSTANT)


def density_field(mask, density_radius, ridge_radius):
    inside = mask != 0
    foreground = inside.astype(np.float32)
    density = box_sum(foreground, density_radius, cv2.CV_32F)
    denominator = box_sum(
        np.ones(mask.shape, np.float32), density_radius, cv2.CV_32F)
    density = np.divide(
        density, denominator, out=np.zeros_like(density), where=denominator > 0)
    density[~inside] = 0
    normalized = np.zeros_like(density)
    mean_density = float(density[inside].mean())
    normalized[inside] = np.clip(
        density[inside] / (mean_density * RELATIVE_CAP), 0, 1)
    inverse = np.zeros_like(density)
    inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / INVERSE_GAMMA)

    yy, xx = np.indices(mask.shape, dtype=np.float64)
    moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
    sums = box_sum(moments, ridge_radius)
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


def ordered(points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1],
                        points[:, 0] - center[0])
    return points[np.argsort(angles)]


def aperture_quadrilateral(contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = cv2.contourArea(contour) / max(hull_area, 1e-6)
    if (hull_area < SMALL_APERTURE_RECT_AREA or
            solidity < CONCAVE_APERTURE_SOLIDITY):
        points = cv2.boxPoints(cv2.minAreaRect(hull))
    elif len(hull) >= 4 and hasattr(cv2, "approxPolyN"):
        points = cv2.approxPolyN(
            hull, 4, epsilon_percentage=-1,
            ensure_convex=True).reshape(-1, 2)
    else:
        points = cv2.boxPoints(cv2.minAreaRect(hull))
    if len(points) != 4:
        return None
    return ordered(points)


def significant_apertures(mask):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    apertures = [
        (cv2.contourArea(contour), contour)
        for contour, item in zip(contours, hierarchy[0])
        if item[3] >= 0 and cv2.contourArea(contour) >= MIN_HOLE_AREA
    ]
    return sorted(apertures, key=lambda item: item[0], reverse=True)


def sample_line(field, mask, start, end):
    length = max(int(np.ceil(np.linalg.norm(end - start))) + 1, 5)
    positions = np.linspace(0.0, 1.0, length)
    points = start + positions[:, None] * (end - start)
    xx = np.clip(np.rint(points[:, 0]).astype(int), 0, field.shape[1] - 1)
    yy = np.clip(np.rint(points[:, 1]).astype(int), 0, field.shape[0] - 1)
    return field[yy, xx], mask[yy, xx] != 0


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best / max(len(values), 1)


def shared_compatibility(point, direction, prior_sides):
    angle_limit = np.cos(np.deg2rad(SHARED_ANGLE_DEGREES))
    for prior in prior_sides:
        alignment = abs(float(np.dot(direction, prior.direction)))
        delta = point - prior.point
        distance = abs(float(
            delta[0] * prior.direction[1] -
            delta[1] * prior.direction[0]))
        if alignment >= angle_limit and distance <= SHARED_LINE_DISTANCE_PX:
            return True
    return False


def fit_side(field, mask, start, end, center, maximum_offset, prior_sides):
    direction = end - start
    direction /= np.linalg.norm(direction)
    midpoint = 0.5 * (start + end)
    normal = np.array((-direction[1], direction[0]))
    if np.dot(normal, midpoint - center) < 0:
        normal = -normal
    positive = field[field > 0]
    field_scale = max(float(positive.max()) if positive.size else 0.0, 1e-6)
    support_threshold = float(np.percentile(positive, 65)) \
        if positive.size else field_scale

    best = None
    for offset in np.linspace(0.5, maximum_offset, maximum_offset * 2):
        candidate_start = start + normal * offset
        candidate_end = end + normal * offset
        values, foreground = sample_line(
            field, mask, candidate_start, candidate_end)
        evidence = values >= support_threshold
        coverage = float(np.mean(evidence))
        continuity = float(longest_run(evidence))
        foreground_coverage = float(np.mean(foreground))
        shared = shared_compatibility(
            midpoint + normal * offset, direction, prior_sides)
        score = (
            float(values.mean()) +
            field_scale * (0.38 * coverage + 0.24 * continuity +
                           0.12 * foreground_coverage +
                           (0.12 if shared else 0.0)))
        candidate = SideFit(
            midpoint + normal * offset, direction.copy(), float(offset),
            score, coverage, continuity, shared)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def line_intersection(first, second):
    denominator = (first.direction[0] * second.direction[1] -
                   first.direction[1] * second.direction[0])
    if abs(denominator) < 1e-6:
        return None
    delta = second.point - first.point
    distance = (delta[0] * second.direction[1] -
                delta[1] * second.direction[0]) / denominator
    return first.point + distance * first.direction


def solve_gate(aperture, field, mask, maximum_offset, prior_sides):
    area, contour = aperture
    corners = aperture_quadrilateral(contour)
    if corners is None:
        return None
    center = corners.mean(axis=0)
    sides = tuple(
        fit_side(
            field, mask, corners[index], corners[(index + 1) % 4],
            center, maximum_offset, prior_sides)
        for index in range(4)
    )
    fitted = [
        line_intersection(sides[index], sides[(index + 1) % 4])
        for index in range(4)
    ]
    if any(point is None or not np.all(np.isfinite(point)) for point in fitted):
        return None
    return GateFit(area, corners, sides, ordered(fitted))


def solve_two_apertures(mask):
    profile = radius_profile(max(mask.shape))
    _, _, density_radius, ridge_radius = profile
    field = density_field(mask, density_radius, ridge_radius)
    apertures = significant_apertures(mask)
    if len(apertures) != 2:
        raise ValueError(
            f"Expected exactly two significant apertures, found {len(apertures)}")
    maximum_offset = max(3, ridge_radius + 2)
    fits = []
    prior_sides = []
    for aperture in apertures:
        fit = solve_gate(
            aperture, field, mask, maximum_offset, tuple(prior_sides))
        if fit is None:
            raise RuntimeError("Unable to fit aperture-seeded quadrilateral")
        fits.append(fit)
        prior_sides.extend(fit.sides)
    return profile, field, tuple(fits)


def solve_two_apertures_from_evidence(component_mask, evidence):
    profile = radius_profile(max(component_mask.shape))
    evidence_mask = (evidence != 0).astype(np.uint8)
    apertures = significant_apertures(component_mask)
    if len(apertures) != 2:
        raise ValueError(
            "Expected exactly two original aperture seeds, "
            f"found {len(apertures)}")
    evidence_field = evidence_mask.astype(np.float64)
    maximum_offset = max(3, profile[3] + 2)
    fits = []
    prior_sides = []
    for aperture in apertures:
        fit = solve_gate(
            aperture, evidence_field, evidence_mask, maximum_offset,
            tuple(prior_sides))
        if fit is None:
            raise RuntimeError("Unable to fit quadrilateral from evidence")
        fits.append(fit)
        prior_sides.extend(fit.sides)
    return profile, tuple(fits)


def solve_two_apertures_from_constrained_close(
        component_mask, closed_support):
    return solve_two_apertures_from_evidence(component_mask, closed_support)


def sampled_side_coordinates(start, end, half_width):
    vector = end - start
    length = max(int(np.ceil(np.linalg.norm(vector))) + 1, 2)
    direction = vector / max(np.linalg.norm(vector), 1e-6)
    normal = np.array((-direction[1], direction[0]))
    positions = np.linspace(0.0, 1.0, length)
    centers = start + positions[:, None] * vector
    samples = []
    for center in centers:
        row = []
        for offset in range(-half_width, half_width + 1):
            point = center + normal * offset
            row.append(tuple(np.rint(point).astype(int)))
        if not samples or row != samples[-1]:
            samples.append(row)
    return samples


def bridge_short_side_gaps(seed, limit, start, end):
    samples = sampled_side_coordinates(
        start, end, SIDE_BRIDGE_HALF_WIDTH_PX)
    height, width = seed.shape

    def in_bounds(point):
        return 0 <= point[0] < width and 0 <= point[1] < height

    supported = []
    allowed = []
    for row in samples:
        valid = [point for point in row if in_bounds(point)]
        supported.append(any(seed[y, x] != 0 for x, y in valid))
        candidates = [point for point in valid if limit[point[1], point[0]]]
        candidates.sort(key=lambda point: abs(row.index(point) - len(row) // 2))
        allowed.append(candidates[0] if candidates else None)

    additions = []
    gaps = []
    cursor = 0
    while cursor < len(samples):
        if supported[cursor]:
            cursor += 1
            continue
        gap_start = cursor
        while cursor < len(samples) and not supported[cursor]:
            cursor += 1
        gap_end = cursor
        gap_length = gap_end - gap_start
        bounded = gap_start > 0 and gap_end < len(samples)
        bridgeable = (
            bounded and gap_length <= SIDE_BRIDGE_MAX_GAP_PX and
            all(allowed[index] is not None
                for index in range(gap_start, gap_end)))
        if bridgeable:
            pixels = [allowed[index] for index in range(gap_start, gap_end)]
            additions.extend(pixels)
            gaps.append({
                "length_px": gap_length,
                "pixels": [[int(point[0]), int(point[1])]
                           for point in pixels],
            })
    return additions, gaps


def density_support_side_guided_bridge(component_mask, seed, limit):
    _, initial_fits = solve_two_apertures_from_evidence(component_mask, seed)
    additions = np.zeros(seed.shape, np.uint8)
    records = []
    for gate_index, fit in enumerate(initial_fits):
        for side_index in range(4):
            start = fit.corners[side_index]
            end = fit.corners[(side_index + 1) % 4]
            pixels, gaps = bridge_short_side_gaps(seed, limit, start, end)
            for x, y in pixels:
                if seed[y, x] == 0:
                    additions[y, x] = 255
            for gap in gaps:
                records.append({
                    "gate": gate_index + 1,
                    "side": side_index + 1,
                    **gap,
                })
    bridged = cv2.bitwise_or(seed, additions)
    _, refined_fits = solve_two_apertures_from_evidence(
        component_mask, bridged)
    return initial_fits, bridged, additions, refined_fits, records


def distances_to_quadrilateral(points, corners):
    distances = np.full(len(points), np.inf, np.float64)
    for index in range(4):
        start = corners[index]
        vector = corners[(index + 1) % 4] - start
        denominator = max(float(np.dot(vector, vector)), 1e-9)
        offsets = points - start
        positions = np.clip(offsets @ vector / denominator, 0.0, 1.0)
        projections = start + positions[:, None] * vector
        distances = np.minimum(
            distances, np.linalg.norm(points - projections, axis=1))
    return distances


def transform_quadrilateral(corners, scale, rotation_degrees, dx, dy):
    center = corners.mean(axis=0)
    angle = np.deg2rad(rotation_degrees)
    rotation = np.array(
        ((np.cos(angle), -np.sin(angle)),
         (np.sin(angle), np.cos(angle))))
    return (corners - center) @ rotation.T * scale + center + (dx, dy)


def quadrilateral_contains(points, corners):
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(
        contour, (float(point[0]), float(point[1])), False) >= 0
        for point in points)


def side_bridge_evidence(seed, limit, corners):
    additions = np.zeros(seed.shape, np.uint8)
    records = []
    for side_index in range(4):
        pixels, gaps = bridge_short_side_gaps(
            seed, limit, corners[side_index], corners[(side_index + 1) % 4])
        for x, y in pixels:
            if seed[y, x] == 0:
                additions[y, x] = 255
        for gap in gaps:
            records.append({"side": side_index + 1, **gap})
    evidence = cv2.bitwise_or(seed, additions)
    side_coverages = []
    height, width = seed.shape
    for side_index in range(4):
        samples = sampled_side_coordinates(
            corners[side_index], corners[(side_index + 1) % 4],
            SIDE_BRIDGE_HALF_WIDTH_PX)
        supported = []
        for row in samples:
            supported.append(any(
                0 <= x < width and 0 <= y < height and evidence[y, x] != 0
                for x, y in row))
        side_coverages.append(float(np.mean(supported)) if supported else 0.0)
    return additions, tuple(records), tuple(side_coverages)


def optimize_coverage_candidate(
        seed, limit, initial_fit, evidence_points, evidence_weights):
    best = None
    height, width = seed.shape

    def evaluate_scale(scale, current_best):
        for rotation in COVERAGE_ROTATIONS_DEG:
            for dx in COVERAGE_TRANSLATIONS_PX:
                for dy in COVERAGE_TRANSLATIONS_PX:
                    corners = transform_quadrilateral(
                        initial_fit.corners, scale, rotation, dx, dy)
                    if (np.any(corners[:, 0] < -1) or
                            np.any(corners[:, 0] > width) or
                            np.any(corners[:, 1] < -1) or
                            np.any(corners[:, 1] > height)):
                        continue
                    if not quadrilateral_contains(
                            initial_fit.aperture_corners, corners):
                        continue
                    distances = distances_to_quadrilateral(
                        evidence_points, corners)
                    covered = distances <= COVERAGE_TOLERANCE_PX
                    weight_sum = float(evidence_weights.sum())
                    weighted_coverage = (
                        float(evidence_weights[covered].sum()) / weight_sum
                        if weight_sum > 0 else 0.0)
                    additions, bridge_records, side_coverages = \
                        side_bridge_evidence(seed, limit, corners)
                    change = (
                        abs(scale - 1.0) / 0.05 +
                        abs(rotation) / 4.0 +
                        np.hypot(dx, dy) / 2.0)
                    key = (
                        weighted_coverage,
                        min(side_coverages),
                        float(np.mean(side_coverages)),
                        -cv2.countNonZero(additions),
                        -change,
                    )
                    if current_best is None or key > current_best[0]:
                        current_best = (key, CoverageCandidate(
                            corners=corners,
                            scale=scale,
                            rotation_degrees=rotation,
                            translation=(dx, dy),
                            weighted_coverage=weighted_coverage,
                            covered_points=int(np.count_nonzero(covered)),
                            side_coverages=side_coverages,
                            additions=additions,
                            bridge_records=bridge_records,
                        ))
        return current_best

    evaluated_scales = []
    for scale in COVERAGE_INITIAL_SCALES:
        best = evaluate_scale(scale, best)
        evaluated_scales.append(scale)
    for scale in COVERAGE_EXTENSION_SCALES:
        if best is None or not np.isclose(best[1].scale, evaluated_scales[-1]):
            break
        best = evaluate_scale(scale, best)
        evaluated_scales.append(scale)
    if best is None:
        raise RuntimeError("No valid bounded coverage candidate")
    return best[1], tuple(evaluated_scales)


def solve_two_apertures_by_coverage(component_mask, field, seed, limit):
    _, initial_fits = solve_two_apertures_from_evidence(component_mask, seed)
    yy, xx = np.nonzero(seed)
    points = np.column_stack((xx, yy)).astype(np.float64)
    weights = field[yy, xx].astype(np.float64)

    larger, larger_scales = optimize_coverage_candidate(
        seed, limit, initial_fits[0], points, weights)
    covered_by_larger = (
        distances_to_quadrilateral(points, larger.corners) <=
        COVERAGE_TOLERANCE_PX)
    residual = ~covered_by_larger
    if not np.any(residual):
        residual = np.ones(len(points), dtype=bool)
    smaller, smaller_scales = optimize_coverage_candidate(
        seed, limit, initial_fits[1], points[residual], weights[residual])

    additions = cv2.bitwise_or(larger.additions, smaller.additions)
    bridged = cv2.bitwise_or(seed, additions)
    union_distances = np.minimum(
        distances_to_quadrilateral(points, larger.corners),
        distances_to_quadrilateral(points, smaller.corners))
    union_covered = union_distances <= COVERAGE_TOLERANCE_PX
    return {
        "initial_fits": initial_fits,
        "larger": larger,
        "larger_evaluated_scales": larger_scales,
        "smaller": smaller,
        "smaller_evaluated_scales": smaller_scales,
        "additions": additions,
        "bridged": bridged,
        "points": points,
        "union_covered": union_covered,
        "weighted_union_coverage": float(
            weights[union_covered].sum() / max(weights.sum(), 1e-9)),
    }


def heat_layer(field, mask):
    image = np.zeros((*field.shape, 3), np.uint8)
    inside = mask != 0
    normalized = np.clip(field, 0, 1)
    image[inside, 0] = 240
    image[inside, 1] = np.rint(55 + 125 * normalized[inside]).astype(np.uint8)
    image[inside, 2] = np.rint(35 + 175 * normalized[inside]).astype(np.uint8)
    return image


def density_support_constrained_close(field, component_mask):
    inside = component_mask != 0
    positive = field[inside & (field > 0)]
    if positive.size == 0:
        empty = np.zeros(component_mask.shape, np.uint8)
        return 0.0, 0.0, empty, empty.copy(), empty.copy(), 0
    seed_threshold = float(np.percentile(
        positive, CONSTRAINED_CLOSE_SEED_PERCENTILE))
    limit_threshold = float(np.percentile(
        positive, CONSTRAINED_CLOSE_LIMIT_PERCENTILE))
    seed = ((field >= seed_threshold) & inside).astype(np.uint8) * 255
    limit = ((field >= limit_threshold) & inside).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (CONSTRAINED_CLOSE_KERNEL_PX, CONSTRAINED_CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    closed = cv2.bitwise_and(closed, limit)
    closed = cv2.bitwise_or(closed, seed)
    return seed_threshold, limit_threshold, seed, limit, closed


def draw_polyline(image, corners, color, source_side, thickness=1):
    scale = PANEL / source_side
    points = np.rint(corners * scale).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)


def render_scenario(name, source, mask, field, profile, fits, stats):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    seed_panel = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    seed_panel = cv2.resize(
        seed_panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    heat = heat_layer(field, mask)
    density_panel = cv2.resize(
        heat, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    side_panel = density_panel.copy()
    result_panel = source_panel.copy()

    for index, fit in enumerate(fits):
        color = GATE_COLORS[index]
        draw_polyline(seed_panel, fit.aperture_corners, color, mask.shape[0], 1)
        draw_polyline(side_panel, fit.corners, color, mask.shape[0], 1)
        draw_polyline(result_panel, fit.corners, color, mask.shape[0], 2)
        for side in fit.sides:
            center = side.point * (PANEL / mask.shape[0])
            cv2.circle(side_panel, tuple(np.rint(center).astype(int)), 2,
                       color, -1, cv2.LINE_AA)

    panels = (source_panel, seed_panel, density_panel, side_panel, result_panel)
    labels = ("source", "aperture seeds", "density", "selected side lines",
              "two-gate result")
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index],
                    (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 205, 205), 1,
                    cv2.LINE_AA)
    _, _, width, height, area = stats
    title = (
        f"{name} bbox={width}x{height} area={area} {profile[0]} "
        f"density-r={profile[2]} ridge-r={profile[3]} "
        f"gamma={INVERSE_GAMMA:.2f}/{RIDGE_GAMMA:.2f}")
    cv2.putText(sheet, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (240, 240, 240), 1, cv2.LINE_AA)
    details = []
    for index, fit in enumerate(fits):
        offsets = "/".join(f"{side.offset:.1f}" for side in fit.sides)
        shared = sum(side.shared_parallel for side in fit.sides)
        details.append(
            f"gate{index + 1} aperture={fit.aperture_area:.0f}px "
            f"offsets={offsets} shared={shared}")
    cv2.putText(sheet, "    ".join(details), (7, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (185, 185, 185), 1,
                cv2.LINE_AA)
    cv2.putText(
        sheet,
        "yellow=larger aperture first   magenta=smaller aperture   "
        "side search is parallel and bounded",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def render_constrained_close_comparison(
        name, source, mask, field, seed_threshold, limit_threshold,
        seed, limit, closed):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    density_panel = cv2.resize(
        heat_layer(field, mask), (PANEL, PANEL),
        interpolation=cv2.INTER_NEAREST)

    def binary_panel(values):
        panel = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        return cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    additions = (closed != 0) & (seed == 0)
    delta = np.zeros((*seed.shape, 3), np.uint8)
    delta[seed != 0] = (105, 105, 105)
    delta[additions] = (0, 255, 0)
    delta = cv2.resize(
        delta, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    panels = (
        source_panel,
        density_panel,
        binary_panel(seed),
        binary_panel(limit),
        binary_panel(closed),
        delta,
    )
    labels = (
        "source",
        "inverse density",
        f"P{CONSTRAINED_CLOSE_SEED_PERCENTILE:g} seeds",
        f"P{CONSTRAINED_CLOSE_LIMIT_PERCENTILE:g} allowed support",
        "one-pass constrained close",
        "close additions = green",
    )
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index],
                    (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 205, 205), 1,
                    cv2.LINE_AA)
    seed_px = cv2.countNonZero(seed)
    limit_px = cv2.countNonZero(limit)
    closed_px = cv2.countNonZero(closed)
    cv2.putText(
        sheet,
        f"{name}  P{CONSTRAINED_CLOSE_SEED_PERCENTILE:g} seeds: one "
        f"3x3 close constrained to P{CONSTRAINED_CLOSE_LIMIT_PERCENTILE:g}",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"seed threshold={seed_threshold:.4f} limit threshold="
        f"{limit_threshold:.4f}  seed={seed_px}px limit={limit_px}px "
        f"result={closed_px}px added={closed_px - seed_px}px",
        (7, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        "one close pass only; no directional close, no iterative dilation, "
        "and no pixels outside the P70 limit",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def render_constrained_close_fit(
        name, source, mask, seed, limit, closed, original_fits, close_fits):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    def binary_panel(values):
        panel = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        return cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    high_seed_panel = binary_panel(seed)
    limit_panel = binary_panel(limit)
    closed_panel = binary_panel(closed)
    aperture_panel = closed_panel.copy()
    fit_panel = closed_panel.copy()
    result_panel = source_panel.copy()
    original_panel = source_panel.copy()
    for index, fit in enumerate(close_fits):
        color = GATE_COLORS[index]
        draw_polyline(
            aperture_panel, fit.aperture_corners, color, mask.shape[0], 1)
        draw_polyline(fit_panel, fit.corners, color, mask.shape[0], 1)
        draw_polyline(result_panel, fit.corners, color, mask.shape[0], 2)
    for index, fit in enumerate(original_fits):
        draw_polyline(
            original_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 2)

    panels = (
        high_seed_panel,
        limit_panel,
        aperture_panel,
        fit_panel,
        original_panel,
        result_panel,
    )
    labels = (
        f"P{CONSTRAINED_CLOSE_SEED_PERCENTILE:g} seeds",
        f"P{CONSTRAINED_CLOSE_LIMIT_PERCENTILE:g} allowed support",
        "constrained close + apertures",
        "quad fit to closed seed",
        "previous density fit",
        "constrained-close quad fit",
    )
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index],
                    (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 205, 205), 1,
                    cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"{name}  two quadrilaterals fit from one-pass P90 close "
        "constrained to P70 evidence",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
        cv2.LINE_AA)
    details = []
    for index, fit in enumerate(close_fits):
        offsets = "/".join(f"{side.offset:.1f}" for side in fit.sides)
        details.append(f"gate{index + 1} offsets={offsets}")
    cv2.putText(sheet, "    ".join(details), (7, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
                cv2.LINE_AA)
    cv2.putText(
        sheet,
        "yellow=larger aperture first   magenta=smaller aperture   "
        "constrained-close pixels are the only fitting evidence",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def render_side_guided_bridge(
        name, source, mask, seed, limit, additions, bridged,
        density_fits, initial_fits, refined_fits, bridge_records):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    def binary_panel(values):
        panel = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        return cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    initial_panel = binary_panel(seed)
    bridge_panel = np.zeros((*seed.shape, 3), np.uint8)
    bridge_panel[seed != 0] = (105, 105, 105)
    bridge_panel[additions != 0] = (0, 255, 0)
    bridge_panel = cv2.resize(
        bridge_panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    refined_panel = binary_panel(bridged)
    density_panel = source_panel.copy()
    result_panel = source_panel.copy()
    for index, fit in enumerate(initial_fits):
        draw_polyline(
            initial_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 1)
    for index, fit in enumerate(refined_fits):
        draw_polyline(
            refined_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 1)
        draw_polyline(
            result_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 2)
    for index, fit in enumerate(density_fits):
        draw_polyline(
            density_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 2)

    panels = (
        binary_panel(seed),
        binary_panel(limit),
        initial_panel,
        bridge_panel,
        refined_panel,
        density_panel,
        result_panel,
    )
    labels = (
        "P90 seed",
        "P70 allowed evidence",
        "initial P90 quad fit",
        "accepted bridges = green",
        "bridged P90 + refined fit",
        "previous density fit",
        "side-guided bridge fit",
    )
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index],
                    (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 205, 205), 1,
                    cv2.LINE_AA)
    lengths = [record["length_px"] for record in bridge_records]
    cv2.putText(
        sheet,
        f"{name}  P90 fit -> short aligned side gaps -> P70-only bridge -> "
        "refit",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"bridges={len(bridge_records)} lengths={lengths} "
        f"added={cv2.countNonZero(additions)}px max-gap="
        f"{SIDE_BRIDGE_MAX_GAP_PX}px corridor=+/-"
        f"{SIDE_BRIDGE_HALF_WIDTH_PX}px",
        (7, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        "bridges are one pixel wide, bounded by P90 on both ends, and every "
        "added pixel already exists in P70",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def render_coverage_optimized_fit(
        name, source, mask, seed, limit, density_fits, result):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    def binary_panel(values):
        panel = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        return cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    initial_panel = binary_panel(seed)
    larger_panel = binary_panel(seed)
    residual = np.zeros((*seed.shape, 3), np.uint8)
    points = result["points"].astype(int)
    larger_distances = distances_to_quadrilateral(
        result["points"], result["larger"].corners)
    larger_covered = larger_distances <= COVERAGE_TOLERANCE_PX
    for (x, y), covered in zip(points, larger_covered):
        residual[y, x] = (90, 90, 90) if covered else (255, 255, 255)
    smaller_panel = cv2.resize(
        residual, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    coverage = np.zeros((*seed.shape, 3), np.uint8)
    for (x, y), covered in zip(points, result["union_covered"]):
        coverage[y, x] = (0, 210, 0) if covered else (0, 0, 255)
    coverage[result["additions"] != 0] = (0, 255, 255)
    coverage_panel = cv2.resize(
        coverage, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    evidence_panel = binary_panel(result["bridged"])
    density_panel = source_panel.copy()
    final_panel = source_panel.copy()

    for index, fit in enumerate(result["initial_fits"]):
        draw_polyline(
            initial_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 1)
    draw_polyline(
        larger_panel, result["larger"].corners, GATE_COLORS[0],
        mask.shape[0], 1)
    draw_polyline(
        smaller_panel, result["smaller"].corners, GATE_COLORS[1],
        mask.shape[0], 1)
    draw_polyline(
        evidence_panel, result["larger"].corners, GATE_COLORS[0],
        mask.shape[0], 1)
    draw_polyline(
        evidence_panel, result["smaller"].corners, GATE_COLORS[1],
        mask.shape[0], 1)
    for index, fit in enumerate(density_fits):
        draw_polyline(
            density_panel, fit.corners, GATE_COLORS[index], mask.shape[0], 2)
    draw_polyline(
        final_panel, result["larger"].corners, GATE_COLORS[0],
        mask.shape[0], 2)
    draw_polyline(
        final_panel, result["smaller"].corners, GATE_COLORS[1],
        mask.shape[0], 2)

    panels = (
        binary_panel(seed),
        initial_panel,
        larger_panel,
        smaller_panel,
        coverage_panel,
        evidence_panel,
        density_panel,
        final_panel,
    )
    labels = (
        "P90 evidence",
        "initial P90 fits",
        "larger gate solved first",
        "smaller fit on residual P90",
        "green=covered red=missed yellow=P70",
        "selected quads + bridges",
        "previous density fit",
        "coverage-optimized fit",
    )
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index],
                    (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (205, 205, 205), 1,
                    cv2.LINE_AA)

    larger = result["larger"]
    smaller = result["smaller"]
    cv2.putText(
        sheet,
        f"{name}  larger-first bounded P90 perimeter coverage; P70 only "
        "bridges short internal gaps",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"union={100 * result['weighted_union_coverage']:.1f}% "
        f"({np.count_nonzero(result['union_covered'])}/{len(points)} points) "
        f"P70-added={cv2.countNonZero(result['additions'])}px  "
        f"large s={larger.scale:.2f} rot={larger.rotation_degrees:+.0f} "
        f"move={larger.translation}",
        (7, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"small s={smaller.scale:.2f} rot={smaller.rotation_degrees:+.0f} "
        f"move={smaller.translation}; staged scales="
        f"{result['smaller_evaluated_scales']}",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def coverage_candidate_record(candidate):
    return {
        "scale": candidate.scale,
        "rotation_degrees": candidate.rotation_degrees,
        "translation": list(candidate.translation),
        "weighted_coverage": candidate.weighted_coverage,
        "covered_points": candidate.covered_points,
        "side_coverages": list(candidate.side_coverages),
        "p70_added_px": cv2.countNonZero(candidate.additions),
        "bridges": list(candidate.bridge_records),
        "corners": candidate.corners.tolist(),
    }


def parse_scenario(value):
    try:
        frame_id, label = value.split(":", 1)
        return frame_id, int(label)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "scenario must be FRAME_ID:LABEL") from error


def frame_path(run, frame_id):
    matches = sorted(run.glob(f"frame-*{frame_id}*"))
    if not matches:
        matches = sorted(run.glob(f"frame-{frame_id}-*"))
    if not matches:
        raise FileNotFoundError(f"No frame containing {frame_id} in {run}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scenario", action="append", type=parse_scenario,
        help="Known two-aperture component as FRAME_ID:LABEL; repeatable")
    args = parser.parse_args()
    scenarios = args.scenario or [parse_scenario(item)
                                  for item in DEFAULT_SCENARIOS]
    args.output.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    records = []
    sheets = []
    constrained_close_sheets = []
    constrained_close_fit_sheets = []
    side_bridge_sheets = []
    coverage_fit_sheets = []

    for index, (frame_id, label) in enumerate(scenarios, 1):
        path = frame_path(args.run, frame_id)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Unable to read {path}")
        mask, source, stats = isolate_component(image, lut, label)
        profile, field, fits = solve_two_apertures(mask)
        (seed_threshold, limit_threshold, seed, limit,
         closed) = density_support_constrained_close(field, mask)
        _, close_fits = solve_two_apertures_from_constrained_close(
            mask, closed)
        (initial_bridge_fits, bridged, bridge_additions, refined_bridge_fits,
         bridge_records) = density_support_side_guided_bridge(
             mask, seed, limit)
        coverage_result = solve_two_apertures_by_coverage(
            mask, field, seed, limit)
        name = f"frame={frame_id} label={label}"
        sheet = render_scenario(name, source, mask, field, profile, fits, stats)
        output = args.output / f"{index:02d}-frame-{frame_id}-label-{label:03d}.png"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"Unable to write {output}")
        print(output)
        sheets.append(sheet)
        constrained_close_sheet = render_constrained_close_comparison(
            name, source, mask, field, seed_threshold, limit_threshold,
            seed, limit, closed)
        constrained_close_output = args.output / (
            f"constrained-close-p90-in-p70-k3-{index:02d}-frame-{frame_id}-"
            f"label-{label:03d}.png")
        if not cv2.imwrite(
                str(constrained_close_output), constrained_close_sheet):
            raise OSError(f"Unable to write {constrained_close_output}")
        print(constrained_close_output)
        constrained_close_sheets.append(constrained_close_sheet)
        constrained_close_fit_sheet = render_constrained_close_fit(
            name, source, mask, seed, limit, closed, fits, close_fits)
        constrained_close_fit_output = args.output / (
            f"fit-constrained-close-p90-in-p70-k3-{index:02d}-"
            f"frame-{frame_id}-label-{label:03d}.png")
        if not cv2.imwrite(
                str(constrained_close_fit_output),
                constrained_close_fit_sheet):
            raise OSError(
                f"Unable to write {constrained_close_fit_output}")
        print(constrained_close_fit_output)
        constrained_close_fit_sheets.append(constrained_close_fit_sheet)
        side_bridge_sheet = render_side_guided_bridge(
            name, source, mask, seed, limit, bridge_additions, bridged,
            fits, initial_bridge_fits, refined_bridge_fits, bridge_records)
        side_bridge_output = args.output / (
            f"side-guided-p90-p70-{index:02d}-frame-{frame_id}-"
            f"label-{label:03d}.png")
        if not cv2.imwrite(str(side_bridge_output), side_bridge_sheet):
            raise OSError(f"Unable to write {side_bridge_output}")
        print(side_bridge_output)
        side_bridge_sheets.append(side_bridge_sheet)
        coverage_fit_sheet = render_coverage_optimized_fit(
            name, source, mask, seed, limit, fits, coverage_result)
        coverage_fit_output = args.output / (
            f"coverage-staged-scale-p90-p70-{index:02d}-frame-{frame_id}-"
            f"label-{label:03d}.png")
        if not cv2.imwrite(str(coverage_fit_output), coverage_fit_sheet):
            raise OSError(f"Unable to write {coverage_fit_output}")
        print(coverage_fit_output)
        coverage_fit_sheets.append(coverage_fit_sheet)
        records.append({
            "file": output.name,
            "frame": path.name,
            "label": label,
            "bbox": list(stats[:4]),
            "area_px": stats[4],
            "profile": profile[0],
            "density_radius": profile[2],
            "ridge_radius": profile[3],
            "constrained_close": {
                "seed_percentile": CONSTRAINED_CLOSE_SEED_PERCENTILE,
                "limit_percentile": CONSTRAINED_CLOSE_LIMIT_PERCENTILE,
                "seed_threshold": seed_threshold,
                "limit_threshold": limit_threshold,
                "kernel_px": CONSTRAINED_CLOSE_KERNEL_PX,
                "passes": 1,
                "seed_px": cv2.countNonZero(seed),
                "limit_px": cv2.countNonZero(limit),
                "closed_px": cv2.countNonZero(closed),
                "added_px": (
                    cv2.countNonZero(closed) - cv2.countNonZero(seed)),
            },
            "constrained_close_fit": [
                {
                    "aperture_area_px": fit.aperture_area,
                    "side_offsets_px": [side.offset for side in fit.sides],
                    "corners": fit.corners.tolist(),
                }
                for fit in close_fits
            ],
            "side_guided_bridge": {
                "seed_percentile": CONSTRAINED_CLOSE_SEED_PERCENTILE,
                "allowed_percentile": CONSTRAINED_CLOSE_LIMIT_PERCENTILE,
                "max_gap_px": SIDE_BRIDGE_MAX_GAP_PX,
                "corridor_half_width_px": SIDE_BRIDGE_HALF_WIDTH_PX,
                "added_px": cv2.countNonZero(bridge_additions),
                "bridges": bridge_records,
                "initial_fit": [
                    {
                        "aperture_area_px": fit.aperture_area,
                        "side_offsets_px": [
                            side.offset for side in fit.sides],
                        "corners": fit.corners.tolist(),
                    }
                    for fit in initial_bridge_fits
                ],
                "refined_fit": [
                    {
                        "aperture_area_px": fit.aperture_area,
                        "side_offsets_px": [
                            side.offset for side in fit.sides],
                        "corners": fit.corners.tolist(),
                    }
                    for fit in refined_bridge_fits
                ],
            },
            "coverage_optimized_fit": {
                "coverage_tolerance_px": COVERAGE_TOLERANCE_PX,
                "initial_scales": list(COVERAGE_INITIAL_SCALES),
                "extension_scales": list(COVERAGE_EXTENSION_SCALES),
                "rotations_degrees": list(COVERAGE_ROTATIONS_DEG),
                "translations_px": list(COVERAGE_TRANSLATIONS_PX),
                "weighted_union_coverage":
                    coverage_result["weighted_union_coverage"],
                "covered_points": int(np.count_nonzero(
                    coverage_result["union_covered"])),
                "total_points": len(coverage_result["points"]),
                "p70_added_px": cv2.countNonZero(
                    coverage_result["additions"]),
                "larger": coverage_candidate_record(
                    coverage_result["larger"]),
                "larger_evaluated_scales": list(
                    coverage_result["larger_evaluated_scales"]),
                "smaller": coverage_candidate_record(
                    coverage_result["smaller"]),
                "smaller_evaluated_scales": list(
                    coverage_result["smaller_evaluated_scales"]),
            },
            "gates": [
                {
                    "aperture_area_px": fit.aperture_area,
                    "side_offsets_px": [side.offset for side in fit.sides],
                    "shared_parallel_sides": sum(
                        side.shared_parallel for side in fit.sides),
                    "corners": fit.corners.tolist(),
                }
                for fit in fits
            ],
        })

    contact = np.vstack(sheets)
    contact_path = args.output / "three-known-two-gate-scenarios.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise OSError(f"Unable to write {contact_path}")
    constrained_close_contact = np.vstack(constrained_close_sheets)
    constrained_close_contact_path = args.output / \
        "three-known-two-gate-scenarios-constrained-close-p90-in-p70-k3.png"
    if not cv2.imwrite(
            str(constrained_close_contact_path), constrained_close_contact):
        raise OSError(f"Unable to write {constrained_close_contact_path}")
    constrained_close_fit_contact = np.vstack(constrained_close_fit_sheets)
    constrained_close_fit_contact_path = args.output / \
        "three-known-two-gate-scenarios-fit-constrained-close-p90-in-p70-k3.png"
    if not cv2.imwrite(
            str(constrained_close_fit_contact_path),
            constrained_close_fit_contact):
        raise OSError(
            f"Unable to write {constrained_close_fit_contact_path}")
    side_bridge_contact = np.vstack(side_bridge_sheets)
    side_bridge_contact_path = args.output / \
        "three-known-two-gate-scenarios-side-guided-p90-p70.png"
    if not cv2.imwrite(str(side_bridge_contact_path), side_bridge_contact):
        raise OSError(f"Unable to write {side_bridge_contact_path}")
    coverage_fit_contact = np.vstack(coverage_fit_sheets)
    coverage_fit_contact_path = args.output / \
        "three-known-two-gate-scenarios-coverage-staged-scale-p90-p70.png"
    if not cv2.imwrite(str(coverage_fit_contact_path), coverage_fit_contact):
        raise OSError(f"Unable to write {coverage_fit_contact_path}")
    manifest = {
        "solver": "aperture-seeded parallel side search v1",
        "run": str(args.run),
        "inverse_gamma": INVERSE_GAMMA,
        "ridge_gamma": RIDGE_GAMMA,
        "relative_cap": RELATIVE_CAP,
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(contact_path)
    print(constrained_close_contact_path)
    print(constrained_close_fit_contact_path)
    print(side_bridge_contact_path)
    print(coverage_fit_contact_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
