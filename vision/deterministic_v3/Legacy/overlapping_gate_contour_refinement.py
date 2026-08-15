"""Weak contour refinement for the two-aperture P90/P70 checkpoint."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from Vision.deterministic_v3.Legacy import \
    overlapping_gate_aperture_solver as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "production_samples" / \
    "two-aperture-contour-angular-refinement-v1"

DISPLAY_SIMPLIFY_EPSILON_PX = 1.0
ROTATION_DELTAS_DEGREES = (-2.0, -1.5, -1.0, -0.5, 0.0,
                           0.5, 1.0, 1.5, 2.0)
P90_COVERAGE_SLACK = 0.01
CONTOUR_SCORE_WEIGHT = 0.005
MIN_LOCAL_POINTS = 4
MIN_PARALLEL_ALIGNMENT = float(np.cos(np.deg2rad(20.0)))
MIN_TANGENT_OVERLAP = 0.20
LOCAL_TANGENT_MARGIN_PX = 2.0

PANEL = 260
HEADER = 88
OUTER_COLOR = (255, 255, 0)
INNER_COLORS = ((0, 255, 255), (255, 0, 255))


@dataclass(frozen=True, slots=True)
class SegmentMatch:
    start: np.ndarray
    end: np.ndarray
    alignment: float
    overlap: float
    signed_distance: float
    point_count: int


@dataclass(frozen=True, slots=True)
class ContourCandidate:
    corners: np.ndarray
    rotation_delta_degrees: float
    weighted_p90_coverage: float
    contour_score: float
    outer_parallel_score: float
    outer_matches: tuple


def simplified_contour(contour):
    return cv2.approxPolyDP(contour, DISPLAY_SIMPLIFY_EPSILON_PX, True)


def component_boundaries(mask):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        raise ValueError("Component has no contour hierarchy")
    hierarchy = hierarchy[0]
    parents = [index for index, item in enumerate(hierarchy) if item[3] < 0]
    outer_index = max(parents, key=lambda index: cv2.contourArea(
        contours[index]))
    holes = [contours[index] for index, item in enumerate(hierarchy)
             if item[3] == outer_index and cv2.contourArea(contours[index]) >=
             base.MIN_HOLE_AREA]
    holes.sort(key=cv2.contourArea, reverse=True)
    if len(holes) != 2:
        raise ValueError(f"Expected two aperture contours, found {len(holes)}")
    outer = contours[outer_index]
    return (
        outer,
        tuple(holes),
        simplified_contour(outer),
        tuple(simplified_contour(contour) for contour in holes),
    )


def fit_local_outer_line(start, end, center, contour, maximum_distance):
    vector = end - start
    length = max(float(np.linalg.norm(vector)), 1e-6)
    direction = vector / length
    midpoint = 0.5 * (start + end)
    normal = np.array((-direction[1], direction[0]))
    if np.dot(normal, midpoint - center) < 0:
        normal = -normal
    points = contour.reshape(-1, 2).astype(np.float64)
    tangent = (points - midpoint) @ direction
    signed = (points - midpoint) @ normal
    selected = points[
        (tangent >= -0.5 * length - LOCAL_TANGENT_MARGIN_PX) &
        (tangent <= 0.5 * length + LOCAL_TANGENT_MARGIN_PX) &
        (signed >= -0.5) & (signed <= maximum_distance)]
    if len(selected) < MIN_LOCAL_POINTS:
        return None
    vx, vy, x, y = cv2.fitLine(
        selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    fit_direction = np.array((float(vx[0]), float(vy[0])), np.float64)
    fit_direction /= max(float(np.linalg.norm(fit_direction)), 1e-9)
    alignment = abs(float(np.dot(direction, fit_direction)))
    if alignment < MIN_PARALLEL_ALIGNMENT:
        return None
    fit_point = np.array((float(x[0]), float(y[0])), np.float64)
    positions = (selected - fit_point) @ fit_direction
    fit_start = fit_point + fit_direction * float(positions.min())
    fit_end = fit_point + fit_direction * float(positions.max())
    projected_span = float(positions.max() - positions.min())
    overlap = min(projected_span / length, 1.0)
    if overlap < MIN_TANGENT_OVERLAP:
        return None
    signed_distance = float(np.dot(fit_point - midpoint, normal))
    return SegmentMatch(
        fit_start, fit_end, alignment, overlap, signed_distance,
        len(selected))


def contour_influence(corners, outer_contour, max_distance):
    center = corners.mean(axis=0)
    outer_matches = []
    outer_scores = []
    for side_index in range(4):
        start = corners[side_index]
        end = corners[(side_index + 1) % 4]
        outer = fit_local_outer_line(
            start, end, center, outer_contour, max_distance)
        outer_matches.append(outer)
        outer_scores.append(
            outer.alignment * outer.overlap if outer is not None else 0.0)
    outer_score = float(np.mean(outer_scores))
    return outer_score, tuple(outer_matches)


def refine_candidate(
        baseline, aperture_corners, evidence_points, evidence_weights,
        outer_contour, shape, maximum_distance):
    candidates = []
    height, width = shape
    for rotation in ROTATION_DELTAS_DEGREES:
        corners = base.transform_quadrilateral(
            baseline.corners, 1.0, rotation, 0.0, 0.0)
        if (np.any(corners[:, 0] < -1) or
                np.any(corners[:, 0] > width) or
                np.any(corners[:, 1] < -1) or
                np.any(corners[:, 1] > height)):
            continue
        if not base.quadrilateral_contains(aperture_corners, corners):
            continue
        distances = base.distances_to_quadrilateral(
            evidence_points, corners)
        covered = distances <= base.COVERAGE_TOLERANCE_PX
        coverage = float(evidence_weights[covered].sum() /
                         max(evidence_weights.sum(), 1e-9))
        contour_score, outer_matches = contour_influence(
            corners, outer_contour, maximum_distance)
        candidates.append(ContourCandidate(
            corners, rotation, coverage, contour_score, contour_score,
            outer_matches))
    if not candidates:
        raise RuntimeError("No valid contour-refinement candidate")
    baseline_candidate = next(candidate for candidate in candidates
                              if candidate.rotation_delta_degrees == 0.0)
    eligible = [candidate for candidate in candidates
                if candidate.weighted_p90_coverage >=
                baseline_candidate.weighted_p90_coverage -
                P90_COVERAGE_SLACK]

    def score(candidate):
        change = abs(candidate.rotation_delta_degrees) / 0.5
        return (
            candidate.weighted_p90_coverage +
            CONTOUR_SCORE_WEIGHT * candidate.contour_score,
            candidate.weighted_p90_coverage,
            candidate.contour_score,
            -change,
        )

    density_only = max(candidates, key=lambda candidate: (
        candidate.weighted_p90_coverage,
        -abs(candidate.rotation_delta_degrees)))
    return max(eligible, key=score), density_only


def solve_contour_refinement(mask, field, seed, limit):
    baseline = base.solve_two_apertures_by_coverage(mask, field, seed, limit)
    (outer_contour, inner_contours, display_outer_contour,
     display_inner_contours) = component_boundaries(mask)
    yy, xx = np.nonzero(seed)
    points = np.column_stack((xx, yy)).astype(np.float64)
    weights = field[yy, xx].astype(np.float64)
    profile = base.radius_profile(max(mask.shape))
    maximum_distance = max(6.0, float(profile[3] + 4))

    larger, larger_density_only = refine_candidate(
        baseline["larger"], baseline["initial_fits"][0].aperture_corners,
        points, weights, outer_contour, mask.shape, maximum_distance)
    larger_covered = base.distances_to_quadrilateral(
        points, larger.corners) <= base.COVERAGE_TOLERANCE_PX
    residual = ~larger_covered
    if not np.any(residual):
        residual = np.ones(len(points), dtype=bool)
    smaller, smaller_density_only = refine_candidate(
        baseline["smaller"], baseline["initial_fits"][1].aperture_corners,
        points[residual], weights[residual], outer_contour, mask.shape,
        maximum_distance)

    larger_additions, _, _ = base.side_bridge_evidence(
        seed, limit, larger.corners)
    smaller_additions, _, _ = base.side_bridge_evidence(
        seed, limit, smaller.corners)
    additions = cv2.bitwise_or(larger_additions, smaller_additions)
    union_distances = np.minimum(
        base.distances_to_quadrilateral(points, larger.corners),
        base.distances_to_quadrilateral(points, smaller.corners))
    union_covered = union_distances <= base.COVERAGE_TOLERANCE_PX
    return {
        "baseline": baseline,
        "outer_contour": outer_contour,
        "inner_contours": inner_contours,
        "display_outer_contour": display_outer_contour,
        "display_inner_contours": display_inner_contours,
        "larger": larger,
        "larger_density_only": larger_density_only,
        "smaller": smaller,
        "smaller_density_only": smaller_density_only,
        "additions": additions,
        "points": points,
        "union_covered": union_covered,
        "weighted_union_coverage": float(
            weights[union_covered].sum() / max(weights.sum(), 1e-9)),
    }


def transformed_points(points, source_side):
    return np.rint(points * (PANEL / source_side)).astype(np.int32)


def draw_contour(panel, contour, color, source_side, thickness=1):
    points = transformed_points(contour.reshape(-1, 2), source_side)
    cv2.polylines(
        panel, [points.reshape(-1, 1, 2)], True, color, thickness,
        cv2.LINE_AA)


def draw_matches(panel, candidate, source_side):
    for match in candidate.outer_matches:
        if match is not None:
            points = transformed_points(
                np.asarray((match.start, match.end)), source_side)
            cv2.line(panel, tuple(points[0]), tuple(points[1]), OUTER_COLOR,
                     2, cv2.LINE_AA)


def render_review(name, source, mask, seed, result):
    source_panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    def binary_panel(values):
        return cv2.resize(
            cv2.cvtColor(values, cv2.COLOR_GRAY2BGR), (PANEL, PANEL),
            interpolation=cv2.INTER_NEAREST)

    contour_panel = binary_panel(mask * 70)
    draw_contour(
        contour_panel, result["display_outer_contour"], OUTER_COLOR,
        mask.shape[0], 2)
    for index, contour in enumerate(result["display_inner_contours"]):
        draw_contour(
            contour_panel, contour, INNER_COLORS[index], mask.shape[0], 2)

    baseline_panel = binary_panel(seed)
    refined_panel = binary_panel(seed)
    match_panel = binary_panel(seed)
    baseline_source = source_panel.copy()
    refined_source = source_panel.copy()
    for index, candidate in enumerate((
            result["baseline"]["larger"],
            result["baseline"]["smaller"])):
        base.draw_polyline(
            baseline_panel, candidate.corners, base.GATE_COLORS[index],
            mask.shape[0], 1)
        base.draw_polyline(
            baseline_source, candidate.corners, base.GATE_COLORS[index],
            mask.shape[0], 2)
    for index, candidate in enumerate((result["larger"], result["smaller"])):
        base.draw_polyline(
            refined_panel, candidate.corners, base.GATE_COLORS[index],
            mask.shape[0], 1)
        base.draw_polyline(
            match_panel, candidate.corners, base.GATE_COLORS[index],
            mask.shape[0], 1)
        draw_matches(match_panel, candidate, mask.shape[0])
        base.draw_polyline(
            refined_source, candidate.corners, base.GATE_COLORS[index],
            mask.shape[0], 2)

    coverage = np.zeros((*seed.shape, 3), np.uint8)
    for (x, y), covered in zip(
            result["points"].astype(int), result["union_covered"]):
        coverage[y, x] = (0, 210, 0) if covered else (0, 0, 255)
    coverage[result["additions"] != 0] = (0, 255, 255)
    coverage_panel = cv2.resize(
        coverage, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    panels = (
        source_panel,
        contour_panel,
        baseline_panel,
        match_panel,
        refined_panel,
        coverage_panel,
        baseline_source,
        refined_source,
    )
    labels = (
        "source",
        "1px display contours; raw points drive fit",
        "committed P90/P70 checkpoint",
        "accepted contour segments",
        "safe angular contour refinement",
        "green=covered red=missed yellow=P70",
        "checkpoint overlay",
        "angular-refined overlay",
    )
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
        cv2.putText(sheet, labels[index], (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (205, 205, 205), 1,
                    cv2.LINE_AA)
    baseline_coverage = result["baseline"]["weighted_union_coverage"]
    large = result["larger"]
    small = result["smaller"]
    cv2.putText(
        sheet,
        f"{name}  raw local outer-contour lines weakly refine angular "
        "alignment after the P90/P70 fit",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"P90 union {100 * baseline_coverage:.1f}% -> "
        f"{100 * result['weighted_union_coverage']:.1f}%  "
        f"large rot={large.rotation_delta_degrees:+.1f} "
        f"contour={large.contour_score:.3f}",
        (7, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"small rot={small.rotation_delta_degrees:+.1f} "
        f"contour={small.contour_score:.3f} "
        f"P70-added={cv2.countNonZero(result['additions'])}px  contour-changed="
        f"{candidate_changed(large, result['larger_density_only'])}/"
        f"{candidate_changed(small, result['smaller_density_only'])}",
        (7, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def candidate_record(candidate):
    return {
        "rotation_delta_degrees": candidate.rotation_delta_degrees,
        "weighted_p90_coverage": candidate.weighted_p90_coverage,
        "contour_score": candidate.contour_score,
        "outer_parallel_score": candidate.outer_parallel_score,
        "outer_sides_matched": sum(
            match is not None for match in candidate.outer_matches),
        "outer_match_points": [
            0 if match is None else match.point_count
            for match in candidate.outer_matches],
        "corners": candidate.corners.tolist(),
    }


def candidate_changed(selected, density_only):
    return bool(
        selected.rotation_delta_degrees !=
        density_only.rotation_delta_degrees)


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
        result = solve_contour_refinement(mask, field, seed, limit)
        name = f"frame={frame_id} label={label}"
        sheet = render_review(name, source, mask, seed, result)
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
            "raw_outer_points": len(result["outer_contour"]),
            "display_outer_vertices": len(result["display_outer_contour"]),
            "raw_inner_points": [
                len(contour) for contour in result["inner_contours"]],
            "display_inner_vertices": [
                len(contour) for contour in result["display_inner_contours"]],
            "larger": candidate_record(result["larger"]),
            "larger_density_only": candidate_record(
                result["larger_density_only"]),
            "larger_contour_changed_selection": candidate_changed(
                result["larger"], result["larger_density_only"]),
            "smaller": candidate_record(result["smaller"]),
            "smaller_density_only": candidate_record(
                result["smaller_density_only"]),
            "smaller_contour_changed_selection": candidate_changed(
                result["smaller"], result["smaller_density_only"]),
        })
        print(output)

    contact = np.vstack(sheets)
    contact_path = args.output / \
        "three-known-two-gate-scenarios-safe-angular-contour-refinement.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise OSError(f"Unable to write {contact_path}")
    manifest = {
        "experiment": "safe raw-contour angular refinement",
        "run": str(args.run),
        "display_simplify_epsilon_px": DISPLAY_SIMPLIFY_EPSILON_PX,
        "rotation_deltas_degrees": list(ROTATION_DELTAS_DEGREES),
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
