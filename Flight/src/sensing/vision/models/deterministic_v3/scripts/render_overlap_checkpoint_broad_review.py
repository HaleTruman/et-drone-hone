"""REVIEW-ONLY broad sweep of the locked two-aperture overlap checkpoint.

This is an offline catalog and renderer, not production or test code. It does
not alter calibration or fitting behavior. Components with exactly two visible
apertures are passed to the existing outer-only checkpoint; clipped components
and components with more than two apertures are reported but never forced
through that solver.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_aperture_solver as base
from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_contour_side_refinement as checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "production_samples" / \
    "multi-gate-overlap-checkpoint-broad-sweep-v1"

PANEL = 240
HEADER = 70
ROWS_PER_PAGE = 8
REPRESENTATIVE_COUNT = 12


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    frame_id: str
    frame_index: int
    label: int
    bbox: tuple
    area: int
    side: int
    holes: int
    touches_frame: bool


def discover(paths, lut):
    candidates = []
    gated_frames = []
    for frame_index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        prepared, gated = base.prepare_mask(image, lut)
        if gated:
            gated_frames.append(path.name)
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            prepared, connectivity=8)
        image_height, image_width = prepared.shape
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
            holes = len(base.significant_apertures(square))
            if holes < 2:
                continue
            touches = bool(
                x == 0 or y == 0 or x + width == image_width or
                y + height == image_height)
            candidates.append(Candidate(
                path=path,
                frame_id=path.name.split("-")[1],
                frame_index=frame_index,
                label=label,
                bbox=(x, y, width, height),
                area=area,
                side=side,
                holes=holes,
                touches_frame=touches,
            ))
    return candidates, gated_frames


def load_component(candidate, lut):
    image = cv2.imread(str(candidate.path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Unable to read {candidate.path}")
    prepared, gated = base.prepare_mask(image, lut)
    if gated:
        raise RuntimeError("Frame became component-gated during review")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        prepared, connectivity=8)
    if candidate.label >= count:
        raise RuntimeError("Component label changed during review")
    x, y, width, height = candidate.bbox
    side = candidate.side
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    component = labels[y:y + height, x:x + width] == candidate.label
    mask = np.zeros((side, side), np.uint8)
    mask[offset_y:offset_y + height,
         offset_x:offset_x + width] = component
    source = np.zeros((side, side, 3), np.uint8)
    source[offset_y:offset_y + height,
           offset_x:offset_x + width] = image[y:y + height, x:x + width]
    return mask, source


def draw_quad(image, corners, color, source_side, thickness=2):
    points = np.rint(corners * (PANEL / source_side)).astype(np.int32)
    cv2.polylines(
        image, [points.reshape(-1, 1, 2)], True, color, thickness,
        cv2.LINE_AA)


def result_panel(source, larger, smaller, source_side):
    panel = cv2.resize(
        source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    draw_quad(panel, larger.corners, base.GATE_COLORS[0], source_side)
    draw_quad(panel, smaller.corners, base.GATE_COLORS[1], source_side)
    return panel


def coverage_panel(seed, result):
    panel = np.zeros((*seed.shape, 3), np.uint8)
    for (x, y), covered in zip(
            result["points"].astype(int), result["union_covered"]):
        panel[y, x] = (0, 210, 0) if covered else (0, 0, 255)
    panel[result["additions"] != 0] = (0, 255, 255)
    return cv2.resize(
        panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def rejection_reason(
        selected, baseline, aperture_corners, outer, evidence_points,
        evidence_weights, shape, maximum_distance):
    if selected.strength > 0:
        return "accepted"
    guides = checkpoint.outer_contour_lines(
        baseline.corners, outer, maximum_distance)
    if not any(guide.outer is not None for guide in guides):
        return "no-outer-guides"
    corners, lines = checkpoint.proposal_from_guides(
        baseline, guides, checkpoint.INFLUENCE_STRENGTHS[-1])
    if corners is None:
        return "parallel-or-invalid-line-intersection"
    polygon = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 1.0:
        return "non-convex-or-empty"
    if not base.quadrilateral_contains(aperture_corners, corners):
        return "aperture-containment"
    height, width = shape
    if not (
            np.all(corners[:, 0] >= -1) and
            np.all(corners[:, 0] <= width) and
            np.all(corners[:, 1] >= -1) and
            np.all(corners[:, 1] <= height)):
        return "crop-bounds"
    distances = base.distances_to_quadrilateral(evidence_points, corners)
    covered = distances <= base.COVERAGE_TOLERANCE_PX
    coverage = float(evidence_weights[covered].sum() /
                     max(evidence_weights.sum(), 1e-9))
    if coverage < selected.weighted_p90_coverage - \
            checkpoint.P90_COVERAGE_SLACK:
        return "p90-safety"
    alignment = checkpoint.contour_alignment(lines, guides)
    candidate_score = coverage + checkpoint.CONTOUR_SCORE_WEIGHT * alignment
    selected_score = (
        selected.weighted_p90_coverage +
        checkpoint.CONTOUR_SCORE_WEIGHT * selected.contour_alignment)
    if candidate_score <= selected_score:
        return "baseline-score"
    return "baseline-tie"


def classify(result):
    large = result["larger"].strength > 0
    small = result["smaller"].strength > 0
    if large and small:
        return "both-adjusted"
    if large:
        return "large-only"
    if small:
        return "small-only"
    return "unchanged"


def solve_candidate(candidate, mask):
    profile = base.radius_profile(candidate.side)
    density_started = perf_counter()
    field = base.density_field(mask, profile[2], profile[3])
    _, _, seed, limit, _ = base.density_support_constrained_close(field, mask)
    density_ms = 1000.0 * (perf_counter() - density_started)
    solve_started = perf_counter()
    result = checkpoint.solve(mask, field, seed, limit)
    solve_ms = 1000.0 * (perf_counter() - solve_started)

    points = result["points"]
    yy = points[:, 1].astype(int)
    xx = points[:, 0].astype(int)
    weights = field[yy, xx].astype(np.float64)
    maximum_distance = max(6.0, float(profile[3] + 4))
    larger_reason = rejection_reason(
        result["larger"], result["baseline"]["larger"],
        result["baseline"]["initial_fits"][0].aperture_corners,
        result["outer"], points, weights, mask.shape, maximum_distance)
    larger_covered = base.distances_to_quadrilateral(
        points, result["larger"].corners) <= base.COVERAGE_TOLERANCE_PX
    residual = ~larger_covered
    if not np.any(residual):
        residual = np.ones(len(points), dtype=bool)
    smaller_reason = rejection_reason(
        result["smaller"], result["baseline"]["smaller"],
        result["baseline"]["initial_fits"][1].aperture_corners,
        result["outer"], points[residual], weights[residual], mask.shape,
        maximum_distance)
    return profile, field, seed, result, density_ms, solve_ms, \
        larger_reason, smaller_reason


def render_card(candidate, source, seed, result, category, reasons):
    panels = (
        cv2.resize(source, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST),
        result_panel(
            source, result["baseline"]["larger"],
            result["baseline"]["smaller"], candidate.side),
        result_panel(
            source, result["larger"], result["smaller"], candidate.side),
        coverage_panel(seed, result),
    )
    labels = (
        "source", "locked P90/P70 baseline", "outer-only checkpoint",
        "green=covered red=missed yellow=P70")
    card = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * PANEL
        card[HEADER:, left:left + PANEL] = panel
        cv2.putText(
            card, label, (left + 7, HEADER - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.31, (205, 205, 205), 1,
            cv2.LINE_AA)
    baseline = result["baseline"]["weighted_union_coverage"]
    refined = result["weighted_union_coverage"]
    title = (
        f"{candidate.side}px frame={candidate.frame_id} label={candidate.label} "
        f"bbox={candidate.bbox[2]}x{candidate.bbox[3]} area={candidate.area} "
        f"{category} P90={100 * baseline:.1f}->{100 * refined:.1f}%")
    cv2.putText(
        card, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        card,
        f"large={reasons[0]} strength={result['larger'].strength:.2f} "
        f"small={reasons[1]} strength={result['smaller'].strength:.2f}",
        (7, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (185, 185, 185), 1,
        cv2.LINE_AA)
    return card


def refinement_record(refinement, baseline):
    movement = np.linalg.norm(refinement.corners - baseline.corners, axis=1)
    return {
        "strength": refinement.strength,
        "guided_sides": refinement.guided_sides,
        "weighted_p90_coverage": refinement.weighted_p90_coverage,
        "contour_alignment": refinement.contour_alignment,
        "mean_corner_movement_px": float(movement.mean()),
        "maximum_corner_movement_px": float(movement.max()),
        "corners": refinement.corners.tolist(),
    }


def flush_page(output, category, page_number, cards):
    if not cards:
        return
    folder = output / "contact-sheets" / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"page-{page_number:03d}.png"
    if not cv2.imwrite(str(path), np.vstack(cards)):
        raise OSError(f"Unable to write {path}")
    print(path, flush=True)


def representative_records(records, count):
    if len(records) <= count:
        return records
    ordered = sorted(records, key=lambda record: record["maximum_dimension_px"])
    targets = np.linspace(0, len(ordered) - 1, count)
    return [ordered[int(round(target))] for target in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instances = args.output / "instances"
    instances.mkdir(parents=True, exist_ok=True)

    lut = base.load_lut()
    paths = sorted(args.run.glob("frame-*"))
    candidates, gated_frames = discover(paths, lut)
    candidates.sort(key=lambda item: (
        item.side, item.frame_index, item.label))
    skipped = [
        candidate for candidate in candidates
        if candidate.touches_frame or candidate.holes != 2]
    eligible = [
        candidate for candidate in candidates
        if not candidate.touches_frame and candidate.holes == 2]
    print(json.dumps({
        "frames": len(paths),
        "multi_aperture_components": len(candidates),
        "eligible_exact_two": len(eligible),
        "skipped": len(skipped),
    }), flush=True)

    records = []
    errors = []
    buffers = {}
    page_numbers = {}
    for index, candidate in enumerate(eligible, 1):
        try:
            mask, source = load_component(candidate, lut)
            profile, _, seed, result, density_ms, solve_ms, \
                larger_reason, smaller_reason = solve_candidate(candidate, mask)
            category = classify(result)
            card = render_card(
                candidate, source, seed, result, category,
                (larger_reason, smaller_reason))
            filename = (
                f"{index:05d}-{candidate.side:03d}px-frame-"
                f"{candidate.frame_id}-label-{candidate.label:03d}.png")
            relative_file = Path("instances") / filename
            if not cv2.imwrite(str(args.output / relative_file), card):
                raise OSError(f"Unable to write {relative_file}")

            buffer = buffers.setdefault(category, [])
            buffer.append(card)
            if len(buffer) == ROWS_PER_PAGE:
                page_numbers[category] = page_numbers.get(category, 0) + 1
                flush_page(
                    args.output, category, page_numbers[category], buffer)
                buffer.clear()

            records.append({
                "file": str(relative_file),
                "frame": candidate.path.name,
                "label": candidate.label,
                "bbox": list(candidate.bbox),
                "area_px": candidate.area,
                "maximum_dimension_px": candidate.side,
                "radius_profile": profile[0],
                "density_radius": profile[2],
                "ridge_radius": profile[3],
                "category": category,
                "baseline_weighted_union_coverage":
                    result["baseline"]["weighted_union_coverage"],
                "refined_weighted_union_coverage":
                    result["weighted_union_coverage"],
                "p70_added_px": cv2.countNonZero(result["additions"]),
                "density_and_threshold_ms": density_ms,
                "checkpoint_solve_ms": solve_ms,
                "larger_reason": larger_reason,
                "smaller_reason": smaller_reason,
                "larger": refinement_record(
                    result["larger"], result["baseline"]["larger"]),
                "smaller": refinement_record(
                    result["smaller"], result["baseline"]["smaller"]),
            })
        except (ValueError, RuntimeError, OSError) as error:
            errors.append({
                "frame": candidate.path.name,
                "label": candidate.label,
                "error": str(error),
            })
        if index % 10 == 0 or index == len(eligible):
            print(f"processed {index}/{len(eligible)}", flush=True)

    for category, cards in buffers.items():
        if cards:
            page_numbers[category] = page_numbers.get(category, 0) + 1
            flush_page(args.output, category, page_numbers[category], cards)

    representative = representative_records(records, REPRESENTATIVE_COUNT)
    if representative:
        cards = [cv2.imread(str(args.output / record["file"]))
                 for record in representative]
        path = args.output / "representative-by-pixel-size.png"
        if not cv2.imwrite(str(path), np.vstack(cards)):
            raise OSError(f"Unable to write {path}")
        print(path, flush=True)

    category_counts = {}
    larger_reasons = {}
    smaller_reasons = {}
    for record in records:
        category_counts[record["category"]] = (
            category_counts.get(record["category"], 0) + 1)
        larger_reasons[record["larger_reason"]] = (
            larger_reasons.get(record["larger_reason"], 0) + 1)
        smaller_reasons[record["smaller_reason"]] = (
            smaller_reasons.get(record["smaller_reason"], 0) + 1)
    summary = {
        "review_only": True,
        "checkpoint": (
            "outer-only influence=0.50 angle-cap=15deg; locked solver "
            "modules are imported unchanged"),
        "run": str(args.run),
        "frames_scanned": len(paths),
        "gated_frames": gated_frames,
        "multi_aperture_components": len(candidates),
        "eligible_exact_two_non_clipped": len(eligible),
        "processed": len(records),
        "solver_errors": len(errors),
        "skipped_clipped": sum(item.touches_frame for item in skipped),
        "skipped_more_than_two_apertures": sum(
            item.holes > 2 for item in skipped),
        "category_counts": category_counts,
        "larger_decisions": larger_reasons,
        "smaller_decisions": smaller_reasons,
        "mean_density_and_threshold_ms": float(np.mean([
            record["density_and_threshold_ms"] for record in records
        ])) if records else 0.0,
        "mean_checkpoint_solve_ms": float(np.mean([
            record["checkpoint_solve_ms"] for record in records
        ])) if records else 0.0,
    }
    manifest = {
        "summary": summary,
        "skipped": [{
            "frame": item.path.name,
            "label": item.label,
            "bbox": list(item.bbox),
            "area_px": item.area,
            "maximum_dimension_px": item.side,
            "significant_apertures": item.holes,
            "touches_frame": item.touches_frame,
        } for item in skipped],
        "errors": errors,
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
