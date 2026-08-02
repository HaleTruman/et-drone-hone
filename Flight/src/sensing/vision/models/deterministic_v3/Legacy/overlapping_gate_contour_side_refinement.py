"""Bounded per-side raw-contour refinement of the overlap checkpoint."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_aperture_solver as base
from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_contour_refinement as contour_review


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "production_samples" / \
    "two-aperture-contour-side-refinement-v3-outer-only"

INFLUENCE_STRENGTHS = (0.0, 0.50)
MAX_SIDE_ANGLE_DEGREES = 15.0
P90_COVERAGE_SLACK = 0.01
CONTOUR_SCORE_WEIGHT = 0.01
MIN_RUN_POINTS = 4
MIN_ALIGNMENT = float(np.cos(np.deg2rad(25.0)))
MIN_SPAN_RATIO = 0.20
TANGENT_MARGIN_PX = 2.0

PANEL = 260
HEADER = 88
OUTER_COLOR = (255, 255, 0)
CENTERLINE_COLOR = (0, 140, 255)


@dataclass(frozen=True, slots=True)
class LocalLine:
    point: np.ndarray
    direction: np.ndarray
    start: np.ndarray
    end: np.ndarray
    alignment: float
    span_ratio: float
    point_count: int


@dataclass(frozen=True, slots=True)
class SideGuide:
    outer: LocalLine | None


@dataclass(frozen=True, slots=True)
class SideRefinement:
    corners: np.ndarray
    strength: float
    weighted_p90_coverage: float
    contour_alignment: float
    guided_sides: int
    lines: tuple
    guides: tuple


def circular_runs(selected):
    selected = np.asarray(selected, dtype=bool)
    if not np.any(selected):
        return ()
    if np.all(selected):
        return (np.arange(len(selected)),)
    start = (int(np.flatnonzero(~selected)[0]) + 1) % len(selected)
    ordered = (start + np.arange(len(selected))) % len(selected)
    runs = []
    current = []
    for index in ordered:
        if selected[index]:
            current.append(index)
        elif current:
            runs.append(np.asarray(current, dtype=int))
            current = []
    if current:
        runs.append(np.asarray(current, dtype=int))
    return tuple(runs)


def fit_run(points, side_direction, side_length):
    vx, vy, x, y = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    direction = np.array((float(vx[0]), float(vy[0])), np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    if np.dot(direction, side_direction) < 0:
        direction = -direction
    alignment = float(np.dot(direction, side_direction))
    if alignment < MIN_ALIGNMENT:
        return None
    point = np.array((float(x[0]), float(y[0])), np.float64)
    positions = (points - point) @ direction
    start = point + direction * float(positions.min())
    end = point + direction * float(positions.max())
    span_ratio = min(float(positions.max() - positions.min()) /
                     max(side_length, 1e-6), 1.0)
    if span_ratio < MIN_SPAN_RATIO:
        return None
    return LocalLine(
        point, direction, start, end, alignment, span_ratio, len(points))


def best_local_run(
        contour, start, end, center, expected_sign, maximum_distance):
    side_vector = end - start
    side_length = max(float(np.linalg.norm(side_vector)), 1e-6)
    side_direction = side_vector / side_length
    midpoint = 0.5 * (start + end)
    normal = np.array((-side_direction[1], side_direction[0]))
    if np.dot(normal, midpoint - center) < 0:
        normal = -normal
    points = contour.reshape(-1, 2).astype(np.float64)
    tangent = (points - midpoint) @ side_direction
    signed = (points - midpoint) @ normal
    selected = (
        (tangent >= -0.5 * side_length - TANGENT_MARGIN_PX) &
        (tangent <= 0.5 * side_length + TANGENT_MARGIN_PX) &
        (expected_sign * signed >= -0.5) &
        (expected_sign * signed <= maximum_distance))
    best = None
    for indices in circular_runs(selected):
        if len(indices) < MIN_RUN_POINTS:
            continue
        fitted = fit_run(points[indices], side_direction, side_length)
        if fitted is None:
            continue
        score = fitted.alignment * fitted.span_ratio
        if best is None or score > best[0]:
            best = score, fitted
    return None if best is None else best[1]


def outer_contour_lines(corners, outer_contour, max_distance):
    center = corners.mean(axis=0)
    guides = []
    for side_index in range(4):
        start = corners[side_index]
        end = corners[(side_index + 1) % 4]
        guides.append(SideGuide(best_local_run(
            outer_contour, start, end, center, 1.0, max_distance)))
    return tuple(guides)


def bounded_direction(baseline, target, strength):
    if np.dot(baseline, target) < 0:
        target = -target
    cross = baseline[0] * target[1] - baseline[1] * target[0]
    dot = float(np.clip(np.dot(baseline, target), -1.0, 1.0))
    delta = float(np.arctan2(cross, dot))
    limit = np.deg2rad(MAX_SIDE_ANGLE_DEGREES)
    correction = np.clip(strength * delta, -limit, limit)
    cosine, sine = np.cos(correction), np.sin(correction)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return rotation @ baseline


def intersect_lines(first, second):
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


def proposal_from_guides(baseline, guides, strength):
    lines = []
    for side_index, guide in enumerate(guides):
        start = baseline.corners[side_index]
        end = baseline.corners[(side_index + 1) % 4]
        direction = end - start
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        midpoint = 0.5 * (start + end)
        if guide.outer is None or strength == 0:
            lines.append((midpoint, direction))
            continue
        refined_direction = bounded_direction(
            direction, guide.outer.direction, strength)
        lines.append((midpoint, refined_direction))
    intersections = [
        intersect_lines(lines[index], lines[(index + 1) % 4])
        for index in range(4)
    ]
    if any(point is None or not np.all(np.isfinite(point))
           for point in intersections):
        return None, tuple(lines)
    return base.ordered(np.asarray(intersections)), tuple(lines)


def valid_corners(corners, aperture_corners, shape):
    if corners is None:
        return False
    polygon = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 1.0:
        return False
    if not base.quadrilateral_contains(aperture_corners, corners):
        return False
    height, width = shape
    return bool(
        np.all(corners[:, 0] >= -1) and np.all(corners[:, 0] <= width) and
        np.all(corners[:, 1] >= -1) and np.all(corners[:, 1] <= height))


def contour_alignment(lines, guides):
    scores = []
    for line, guide in zip(lines, guides):
        if guide.outer is not None:
            scores.append(abs(float(np.dot(line[1], guide.outer.direction))))
    return float(np.mean(scores)) if scores else 0.0


def refine_gate(
        baseline, aperture_corners, outer_contour, evidence_points,
        evidence_weights, shape, maximum_distance):
    guides = outer_contour_lines(
        baseline.corners, outer_contour, maximum_distance)
    candidates = []
    for strength in INFLUENCE_STRENGTHS:
        corners, lines = proposal_from_guides(baseline, guides, strength)
        if not valid_corners(corners, aperture_corners, shape):
            continue
        distances = base.distances_to_quadrilateral(
            evidence_points, corners)
        covered = distances <= base.COVERAGE_TOLERANCE_PX
        coverage = float(evidence_weights[covered].sum() /
                         max(evidence_weights.sum(), 1e-9))
        candidates.append(SideRefinement(
            corners, strength, coverage, contour_alignment(lines, guides),
            sum(guide.outer is not None for guide in guides),
            lines, guides))
    baseline_candidate = next(
        candidate for candidate in candidates if candidate.strength == 0.0)
    eligible = [
        candidate for candidate in candidates
        if candidate.weighted_p90_coverage >=
        baseline_candidate.weighted_p90_coverage - P90_COVERAGE_SLACK]
    return max(eligible, key=lambda candidate: (
        candidate.weighted_p90_coverage +
        CONTOUR_SCORE_WEIGHT * candidate.contour_alignment,
        candidate.weighted_p90_coverage,
        candidate.contour_alignment,
        -candidate.strength,
    ))


def solve(mask, field, seed, limit):
    baseline = base.solve_two_apertures_by_coverage(mask, field, seed, limit)
    outer, inners, _, _ = contour_review.component_boundaries(mask)
    yy, xx = np.nonzero(seed)
    points = np.column_stack((xx, yy)).astype(np.float64)
    weights = field[yy, xx].astype(np.float64)
    profile = base.radius_profile(max(mask.shape))
    maximum_distance = max(6.0, float(profile[3] + 4))

    larger = refine_gate(
        baseline["larger"], baseline["initial_fits"][0].aperture_corners,
        outer, points, weights, mask.shape, maximum_distance)
    larger_covered = base.distances_to_quadrilateral(
        points, larger.corners) <= base.COVERAGE_TOLERANCE_PX
    residual = ~larger_covered
    if not np.any(residual):
        residual = np.ones(len(points), dtype=bool)
    smaller = refine_gate(
        baseline["smaller"], baseline["initial_fits"][1].aperture_corners,
        outer, points[residual], weights[residual], mask.shape,
        maximum_distance)

    additions = cv2.bitwise_or(
        base.side_bridge_evidence(seed, limit, larger.corners)[0],
        base.side_bridge_evidence(seed, limit, smaller.corners)[0])
    union_distance = np.minimum(
        base.distances_to_quadrilateral(points, larger.corners),
        base.distances_to_quadrilateral(points, smaller.corners))
    union_covered = union_distance <= base.COVERAGE_TOLERANCE_PX
    return {
        "baseline": baseline,
        "outer": outer,
        "inners": inners,
        "larger": larger,
        "smaller": smaller,
        "additions": additions,
        "points": points,
        "union_covered": union_covered,
        "weighted_union_coverage": float(
            weights[union_covered].sum() / max(weights.sum(), 1e-9)),
    }


def draw_raw_contour(panel, contour, color, source_side):
    points = contour_review.transformed_points(
        contour.reshape(-1, 2), source_side)
    cv2.polylines(panel, [points.reshape(-1, 1, 2)], True, color, 1,
                  cv2.LINE_AA)


def draw_pair_lines(panel, refinement, source_side):
    for guide, line in zip(refinement.guides, refinement.lines):
        if guide.outer is not None:
            points = contour_review.transformed_points(
                np.asarray((guide.outer.start, guide.outer.end)), source_side)
            cv2.line(panel, tuple(points[0]), tuple(points[1]), OUTER_COLOR, 2,
                     cv2.LINE_AA)
            point, direction = line
            endpoints = np.asarray((point - direction * source_side,
                                    point + direction * source_side))
            points = contour_review.transformed_points(endpoints, source_side)
            cv2.line(panel, tuple(points[0]), tuple(points[1]),
                     CENTERLINE_COLOR, 1, cv2.LINE_AA)


def render(name, source, mask, seed, result):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    def binary_panel(values):
        return cv2.resize(
            cv2.cvtColor(values, cv2.COLOR_GRAY2BGR), (PANEL, PANEL),
            interpolation=cv2.INTER_NEAREST)

    raw_panel = binary_panel(mask * 70)
    draw_raw_contour(raw_panel, result["outer"], OUTER_COLOR, mask.shape[0])
    draw_raw_contour(raw_panel, result["inners"][0], base.GATE_COLORS[0],
                     mask.shape[0])
    draw_raw_contour(raw_panel, result["inners"][1], base.GATE_COLORS[1],
                     mask.shape[0])
    baseline_panel = binary_panel(seed)
    pair_panel = binary_panel(seed)
    refined_panel = binary_panel(seed)
    baseline_source = source_panel.copy()
    refined_source = source_panel.copy()
    for index, candidate in enumerate((
            result["baseline"]["larger"], result["baseline"]["smaller"])):
        base.draw_polyline(baseline_panel, candidate.corners,
                           base.GATE_COLORS[index], mask.shape[0], 1)
        base.draw_polyline(baseline_source, candidate.corners,
                           base.GATE_COLORS[index], mask.shape[0], 2)
    for index, candidate in enumerate((result["larger"], result["smaller"])):
        draw_pair_lines(pair_panel, candidate, mask.shape[0])
        base.draw_polyline(pair_panel, candidate.corners,
                           base.GATE_COLORS[index], mask.shape[0], 1)
        base.draw_polyline(refined_panel, candidate.corners,
                           base.GATE_COLORS[index], mask.shape[0], 1)
        base.draw_polyline(refined_source, candidate.corners,
                           base.GATE_COLORS[index], mask.shape[0], 2)

    coverage = np.zeros((*seed.shape, 3), np.uint8)
    for (x, y), covered in zip(
            result["points"].astype(int), result["union_covered"]):
        coverage[y, x] = (0, 210, 0) if covered else (0, 0, 255)
    coverage[result["additions"] != 0] = (0, 255, 255)
    coverage_panel = cv2.resize(
        coverage, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    panels = (
        source_panel, raw_panel, baseline_panel, pair_panel, refined_panel,
        coverage_panel, baseline_source, refined_source)
    labels = (
        "source", "raw contours (apertures are context only)",
        "P90/P70 checkpoint", "outer cyan / guided side orange",
        "outer-only per-side refinement",
        "green=covered red=missed yellow=P70", "checkpoint overlay",
        "outer-only refined overlay")
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index], (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (205, 205, 205), 1,
                    cv2.LINE_AA)
    large, small = result["larger"], result["smaller"]
    cv2.putText(
        sheet,
        f"{name}  raw outer-contour runs guide side angles only; "
        "side positions and missing sides retain P90 geometry",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"P90 union {100 * result['baseline']['weighted_union_coverage']:.1f}%"
        f" -> {100 * result['weighted_union_coverage']:.1f}%  large "
        f"strength={large.strength:.2f} guided={large.guided_sides}/4 "
        f"align={large.contour_alignment:.3f}",
        (7, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"small strength={small.strength:.2f} guided={small.guided_sides}/4 "
        f"align={small.contour_alignment:.3f} P70-added="
        f"{cv2.countNonZero(result['additions'])}px",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def refinement_record(refinement, baseline):
    shifts = np.linalg.norm(refinement.corners - baseline.corners, axis=1)
    return {
        "strength": refinement.strength,
        "weighted_p90_coverage": refinement.weighted_p90_coverage,
        "contour_alignment": refinement.contour_alignment,
        "guided_sides": refinement.guided_sides,
        "mean_corner_shift_px": float(shifts.mean()),
        "maximum_corner_shift_px": float(shifts.max()),
        "outer_run_points": [
            0 if guide.outer is None else guide.outer.point_count
            for guide in refinement.guides],
        "corners": refinement.corners.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scenario", action="append", type=base.parse_scenario,
        help="Known two-aperture component as FRAME_ID:LABEL; repeatable")
    args = parser.parse_args()
    scenarios = args.scenario or [base.parse_scenario(item)
                                  for item in base.DEFAULT_SCENARIOS]
    args.output.mkdir(parents=True, exist_ok=True)
    lut = base.load_lut()
    sheets = []
    records = []
    for index, (frame_id, label) in enumerate(scenarios, 1):
        path = base.frame_path(args.run, frame_id)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Unable to read {path}")
        mask, source, stats = base.isolate_component(image, lut, label)
        _, field, _ = base.solve_two_apertures(mask)
        _, _, seed, limit, _ = base.density_support_constrained_close(
            field, mask)
        result = solve(mask, field, seed, limit)
        name = f"frame={frame_id} label={label}"
        sheet = render(name, source, mask, seed, result)
        output = args.output / (
            f"{index:02d}-frame-{frame_id}-label-{label:03d}.png")
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"Unable to write {output}")
        sheets.append(sheet)
        records.append({
            "file": output.name,
            "frame": path.name,
            "label": label,
            "bbox": list(stats[:4]),
            "area_px": stats[4],
            "baseline_weighted_union_coverage":
                result["baseline"]["weighted_union_coverage"],
            "refined_weighted_union_coverage":
                result["weighted_union_coverage"],
            "p70_added_px": cv2.countNonZero(result["additions"]),
            "larger": refinement_record(
                result["larger"], result["baseline"]["larger"]),
            "smaller": refinement_record(
                result["smaller"], result["baseline"]["smaller"]),
        })
        print(output)

    contact = np.vstack(sheets)
    contact_path = args.output / \
        "three-known-two-gate-scenarios-outer-only-refinement.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise OSError(f"Unable to write {contact_path}")
    manifest = {
        "experiment": "bounded outer-contour-only side-angle refinement",
        "run": str(args.run),
        "influence_strengths": list(INFLUENCE_STRENGTHS),
        "maximum_side_angle_degrees": MAX_SIDE_ANGLE_DEGREES,
        "p90_coverage_slack": P90_COVERAGE_SLACK,
        "contour_score_weight": CONTOUR_SCORE_WEIGHT,
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(contact_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
