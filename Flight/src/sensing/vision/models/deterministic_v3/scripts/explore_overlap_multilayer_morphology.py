"""EXPERIMENT-ONLY side track for multi-layer overlap evidence.

The locked P90/P70 and outer-contour checkpoint modules are imported unchanged
and rendered as the baseline. This script tests one isolated alternative:

* one-pixel-radius (3x3) close of P90, constrained to P70;
* one-pixel-radius (3x3) open of P40-and-above support;
* independently normalized layer influences 1.00 / 0.70 / 0.40; and
* the existing staged scale/rotation/translation search, larger gate first and
  smaller gate on residual evidence, followed by the unchanged outer post-pass.

This is not production code, calibration, or a unit test. Its results must not
replace the locked checkpoint without explicit review and approval.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.scripts import \
    render_overlap_checkpoint_broad_review as broad
from sensing.vision.models.deterministic_v3.scripts import \
    render_overlap_checkpoint_stage_audit as audit
from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_aperture_solver as base
from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_contour_side_refinement as checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "production_samples" / \
    "overlap-multilayer-morphology-side-track-v2-layer-normalized"

MORPH_KERNEL_PX = 3
PERCENTILES = (90.0, 70.0, 40.0)
LAYER_WEIGHTS = (1.00, 0.70, 0.40)
SAMPLE_COUNT = 12
SHEETS_PER_PAGE = 3


@dataclass(frozen=True, slots=True)
class LayerCandidate:
    corners: np.ndarray
    scale: float
    rotation_degrees: float
    translation: tuple
    weighted_coverage: float
    p90_coverage: float


def percentile_layers(field, mask):
    inside = mask != 0
    positive = field[inside & (field > 0)]
    if positive.size == 0:
        raise ValueError("Density field has no positive component evidence")
    thresholds = tuple(float(np.percentile(positive, value))
                       for value in PERCENTILES)
    p90 = ((field >= thresholds[0]) & inside).astype(np.uint8) * 255
    p70 = ((field >= thresholds[1]) & inside).astype(np.uint8) * 255
    p40 = ((field >= thresholds[2]) & inside).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_PX, MORPH_KERNEL_PX))

    closed = cv2.morphologyEx(p90, cv2.MORPH_CLOSE, kernel)
    closed = cv2.bitwise_and(closed, p70)
    closed = cv2.bitwise_or(closed, p90)
    opened = cv2.morphologyEx(p40, cv2.MORPH_OPEN, kernel)
    opened = cv2.bitwise_and(opened, mask.astype(np.uint8) * 255)

    core = p90 != 0
    close_additions = (closed != 0) & ~core
    p40_additions = (opened != 0) & ~(closed != 0)
    weights = np.zeros(mask.shape, np.float64)
    for layer, influence in zip(
            (core, close_additions, p40_additions), LAYER_WEIGHTS):
        count = int(np.count_nonzero(layer))
        if count:
            weights[layer] = influence / count
    return {
        "thresholds": thresholds,
        "p90": p90,
        "p70": p70,
        "p40": p40,
        "closed": closed,
        "opened": opened,
        "core": core,
        "close_additions": close_additions,
        "p40_additions": p40_additions,
        "weights": weights,
    }


def candidate_coverage(points, weights, corners):
    distances = base.distances_to_quadrilateral(points, corners)
    covered = distances <= base.COVERAGE_TOLERANCE_PX
    return float(weights[covered].sum() / max(weights.sum(), 1e-9)), covered


def p90_coverage(p90_points, corners):
    if not len(p90_points):
        return 0.0
    distances = base.distances_to_quadrilateral(p90_points, corners)
    return float(np.mean(distances <= base.COVERAGE_TOLERANCE_PX))


def optimize_candidate(initial_fit, points, weights, p90_points, shape):
    height, width = shape
    best = None

    def evaluate_scale(scale, current_best):
        for rotation in base.COVERAGE_ROTATIONS_DEG:
            for dx in base.COVERAGE_TRANSLATIONS_PX:
                for dy in base.COVERAGE_TRANSLATIONS_PX:
                    corners = base.transform_quadrilateral(
                        initial_fit.corners, scale, rotation, dx, dy)
                    if (np.any(corners[:, 0] < -1) or
                            np.any(corners[:, 0] > width) or
                            np.any(corners[:, 1] < -1) or
                            np.any(corners[:, 1] > height)):
                        continue
                    if not base.quadrilateral_contains(
                            initial_fit.aperture_corners, corners):
                        continue
                    weighted, _ = candidate_coverage(points, weights, corners)
                    core = p90_coverage(p90_points, corners)
                    change = (
                        abs(scale - 1.0) / 0.05 +
                        abs(rotation) / 4.0 +
                        np.hypot(dx, dy) / 2.0)
                    key = (weighted, core, -change)
                    if current_best is None or key > current_best[0]:
                        current_best = (key, LayerCandidate(
                            corners=corners,
                            scale=scale,
                            rotation_degrees=rotation,
                            translation=(dx, dy),
                            weighted_coverage=weighted,
                            p90_coverage=core,
                        ))
        return current_best

    evaluated_scales = []
    for scale in base.COVERAGE_INITIAL_SCALES:
        best = evaluate_scale(scale, best)
        evaluated_scales.append(scale)
    for scale in base.COVERAGE_EXTENSION_SCALES:
        if best is None or not np.isclose(best[1].scale, evaluated_scales[-1]):
            break
        best = evaluate_scale(scale, best)
        evaluated_scales.append(scale)
    if best is None:
        raise RuntimeError("No valid multi-layer candidate")
    return best[1], tuple(evaluated_scales)


def solve_multilayer(mask, field, layers):
    _, initial_fits = base.solve_two_apertures_from_evidence(
        mask, layers["closed"])
    yy, xx = np.nonzero(layers["weights"] > 0)
    points = np.column_stack((xx, yy)).astype(np.float64)
    weights = layers["weights"][yy, xx]
    p90_y, p90_x = np.nonzero(layers["p90"])
    p90_points = np.column_stack((p90_x, p90_y)).astype(np.float64)
    p90_field_weights = field[p90_y, p90_x].astype(np.float64)

    larger, larger_scales = optimize_candidate(
        initial_fits[0], points, weights, p90_points, mask.shape)
    _, larger_covered = candidate_coverage(points, weights, larger.corners)
    residual = ~larger_covered
    if not np.any(residual):
        residual = np.ones(len(points), dtype=bool)
    small_p90 = p90_points[
        base.distances_to_quadrilateral(
            p90_points, larger.corners) > base.COVERAGE_TOLERANCE_PX]
    smaller, smaller_scales = optimize_candidate(
        initial_fits[1], points[residual], weights[residual], small_p90,
        mask.shape)

    outer, _, _, _ = checkpoint.contour_review.component_boundaries(mask)
    profile = base.radius_profile(max(mask.shape))
    maximum_distance = max(6.0, float(profile[3] + 4))
    refined_larger = checkpoint.refine_gate(
        larger, initial_fits[0].aperture_corners, outer, points, weights,
        mask.shape, maximum_distance)
    refined_large_distance = base.distances_to_quadrilateral(
        points, refined_larger.corners)
    refined_residual = refined_large_distance > base.COVERAGE_TOLERANCE_PX
    if not np.any(refined_residual):
        refined_residual = np.ones(len(points), dtype=bool)
    refined_smaller = checkpoint.refine_gate(
        smaller, initial_fits[1].aperture_corners, outer,
        points[refined_residual], weights[refined_residual], mask.shape,
        maximum_distance)

    base_union = np.minimum(
        base.distances_to_quadrilateral(points, larger.corners),
        base.distances_to_quadrilateral(points, smaller.corners))
    refined_union = np.minimum(
        base.distances_to_quadrilateral(points, refined_larger.corners),
        base.distances_to_quadrilateral(points, refined_smaller.corners))
    p90_union = np.minimum(
        base.distances_to_quadrilateral(p90_points, refined_larger.corners),
        base.distances_to_quadrilateral(p90_points, refined_smaller.corners))
    p90_covered = p90_union <= base.COVERAGE_TOLERANCE_PX
    return {
        "initial_fits": initial_fits,
        "larger": larger,
        "smaller": smaller,
        "refined_larger": refined_larger,
        "refined_smaller": refined_smaller,
        "larger_scales": larger_scales,
        "smaller_scales": smaller_scales,
        "points": points,
        "weights": weights,
        "base_covered": base_union <= base.COVERAGE_TOLERANCE_PX,
        "refined_covered": refined_union <= base.COVERAGE_TOLERANCE_PX,
        "weighted_base_coverage": float(
            weights[base_union <= base.COVERAGE_TOLERANCE_PX].sum() /
            max(weights.sum(), 1e-9)),
        "weighted_refined_coverage": float(
            weights[refined_union <= base.COVERAGE_TOLERANCE_PX].sum() /
            max(weights.sum(), 1e-9)),
        "refined_p90_coverage": float(np.mean(
            p90_covered)),
        "refined_weighted_p90_coverage": float(
            p90_field_weights[p90_covered].sum() /
            max(p90_field_weights.sum(), 1e-9)),
    }


def source_fit_panel(source, candidates, source_side):
    panel = cv2.resize(
        source, (audit.CELL, audit.CELL), interpolation=cv2.INTER_NEAREST)
    for index, candidate in enumerate(candidates):
        audit.draw_quad(
            panel, candidate.corners, base.GATE_COLORS[index], source_side)
    return panel


def composite_panel(layers):
    panel = np.zeros((*layers["p90"].shape, 3), np.uint8)
    panel[layers["p40_additions"]] = (235, 90, 25)
    panel[layers["close_additions"]] = (0, 220, 255)
    panel[layers["core"]] = (245, 245, 245)
    return panel


def multilayer_coverage_panel(mask, result):
    panel = np.zeros((*mask.shape, 3), np.uint8)
    maximum_weight = max(float(result["weights"].max()), 1e-9)
    for (x, y), weight, covered in zip(
            result["points"].astype(int), result["weights"],
            result["refined_covered"]):
        intensity = int(round(90 + 165 * weight / maximum_weight))
        panel[y, x] = (0, intensity, 0) if covered else (0, 0, intensity)
    audit.draw_quad(
        panel, result["refined_larger"].corners, base.GATE_COLORS[0],
        mask.shape[0], 1)
    audit.draw_quad(
        panel, result["refined_smaller"].corners, base.GATE_COLORS[1],
        mask.shape[0], 1)
    return panel


def render_sheet(candidate, source, mask, field, layers, locked, experiment,
                 reasons, profile):
    composite = composite_panel(layers)
    initial_panel = composite.copy()
    for index, fit in enumerate(experiment["initial_fits"]):
        audit.draw_quad(
            initial_panel, fit.corners, base.GATE_COLORS[index],
            candidate.side, 1)
    panels = (
        (source, "01 source"),
        (audit.field_panel(field, mask), "02 locked density field"),
        (audit.binary_panel(layers["p90"]), "03 original P90"),
        (audit.binary_panel(layers["p70"]), "04 P70 constraint"),
        (audit.binary_panel(layers["closed"]), "05 P90 close -> P70"),
        (audit.binary_panel(layers["p40"]), "06 raw P40 support"),
        (audit.binary_panel(layers["opened"]), "07 P40 3x3 open"),
        (composite, "08 weighted disjoint layers"),
        (initial_panel, "09 aperture-seeded initial"),
        (source_fit_panel(source, (
            locked["baseline"]["larger"],
            locked["baseline"]["smaller"]), candidate.side),
         "10 locked P90/P70 base"),
        (source_fit_panel(source, (
            locked["larger"], locked["smaller"]), candidate.side),
         "11 locked outer checkpoint"),
        (source_fit_panel(source, (
            experiment["larger"], experiment["smaller"]), candidate.side),
         "12 multi-layer large->small"),
        (source_fit_panel(source, (
            experiment["refined_larger"], experiment["refined_smaller"]),
            candidate.side), "13 multi-layer + same outer"),
        (multilayer_coverage_panel(mask, experiment),
         "14 multi-layer coverage"),
        (audit.text_panel((
            "SIDE TRACK ONLY",
            "3x3 = 1px radius",
            "layer mix 1/.7/.4",
            f"locked P90 {100 * locked['weighted_union_coverage']:.1f}%",
            f"multi all {100 * experiment['weighted_refined_coverage']:.1f}%",
            f"multi P90w {100 * experiment['refined_weighted_p90_coverage']:.1f}%",
            f"locked L/S {reasons[0]}/{reasons[1]}",
        )), "15 comparison summary"),
    )
    cells = [audit.labeled_cell(panel, label) for panel, label in panels]
    rows = len(cells) // audit.COLUMNS
    sheet = np.zeros((
        audit.HEADER + rows * (audit.LABEL_HEIGHT + audit.CELL),
        audit.COLUMNS * audit.CELL, 3), np.uint8)
    for index, cell in enumerate(cells):
        row, column = divmod(index, audit.COLUMNS)
        top = audit.HEADER + row * (audit.LABEL_HEIGHT + audit.CELL)
        left = column * audit.CELL
        sheet[top:top + cell.shape[0], left:left + audit.CELL] = cell
    cv2.putText(
        sheet,
        f"MULTI-LAYER MORPHOLOGY SIDE TRACK  {candidate.side}px "
        f"frame={candidate.frame_id} label={candidate.label} "
        f"area={candidate.area} profile={profile[0]}",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
        (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "Locked baseline preserved | experimental: P90 close->P70 + "
        "P40 open + weighted large-first residual fit",
        (7, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
        (185, 185, 185), 1, cv2.LINE_AA)
    return sheet


def select_candidates(candidates):
    eligible = sorted((
        candidate for candidate in candidates
        if candidate.holes == 2 and not candidate.touches_frame
    ), key=lambda candidate: (candidate.side, candidate.frame_index,
                              candidate.label))
    selected = {}
    for scenario in base.DEFAULT_SCENARIOS:
        frame_id, label = base.parse_scenario(scenario)
        matches = [candidate for candidate in eligible
                   if candidate.frame_id == frame_id and
                   candidate.label == label]
        if matches:
            selected[(matches[0].path, matches[0].label)] = matches[0]
    remaining_count = max(SAMPLE_COUNT - len(selected), 0)
    remaining = [candidate for candidate in eligible
                 if (candidate.path, candidate.label) not in selected]
    if remaining_count and remaining:
        targets = np.linspace(0, len(remaining) - 1, remaining_count)
        for target in targets:
            candidate = remaining[int(round(target))]
            selected[(candidate.path, candidate.label)] = candidate
    return sorted(selected.values(), key=lambda candidate: (
        candidate.side, candidate.frame_index, candidate.label))


def record_candidate(candidate, profile, layers, locked, experiment, path):
    return {
        "file": path.name,
        "frame": candidate.path.name,
        "label": candidate.label,
        "maximum_dimension_px": candidate.side,
        "area_px": candidate.area,
        "radius_profile": profile[0],
        "thresholds": {
            "p90": layers["thresholds"][0],
            "p70": layers["thresholds"][1],
            "p40": layers["thresholds"][2],
        },
        "pixel_counts": {
            "p90": cv2.countNonZero(layers["p90"]),
            "p90_close_p70": cv2.countNonZero(layers["closed"]),
            "p40_raw": cv2.countNonZero(layers["p40"]),
            "p40_open": cv2.countNonZero(layers["opened"]),
            "weighted_union": int(np.count_nonzero(layers["weights"])),
        },
        "locked_p90_union_coverage": locked["weighted_union_coverage"],
        "experimental_multilayer_base_coverage":
            experiment["weighted_base_coverage"],
        "experimental_multilayer_refined_coverage":
            experiment["weighted_refined_coverage"],
        "experimental_original_p90_coverage":
            experiment["refined_p90_coverage"],
        "experimental_weighted_p90_coverage":
            experiment["refined_weighted_p90_coverage"],
        "experimental_larger": {
            "scale": experiment["larger"].scale,
            "rotation_degrees": experiment["larger"].rotation_degrees,
            "translation": list(experiment["larger"].translation),
            "outer_strength": experiment["refined_larger"].strength,
        },
        "experimental_smaller": {
            "scale": experiment["smaller"].scale,
            "rotation_degrees": experiment["smaller"].rotation_degrees,
            "translation": list(experiment["smaller"].translation),
            "outer_strength": experiment["refined_smaller"].strength,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    instances = args.output / "instances"
    instances.mkdir(parents=True, exist_ok=True)

    lut = base.load_lut()
    paths = sorted(args.run.glob("frame-*"))
    candidates, _ = broad.discover(paths, lut)
    selected = select_candidates(candidates)
    sheets = []
    records = []
    errors = []
    for index, candidate in enumerate(selected, 1):
        try:
            mask, source = broad.load_component(candidate, lut)
            profile, field, _, locked, _, _, larger_reason, smaller_reason = \
                broad.solve_candidate(candidate, mask)
            layers = percentile_layers(field, mask)
            experiment = solve_multilayer(mask, field, layers)
            sheet = render_sheet(
                candidate, source, mask, field, layers, locked, experiment,
                (larger_reason, smaller_reason), profile)
            path = instances / (
                f"{index:02d}-{candidate.side:03d}px-frame-"
                f"{candidate.frame_id}-label-{candidate.label:03d}.png")
            if not cv2.imwrite(str(path), sheet):
                raise OSError(f"Unable to write {path}")
            sheets.append(sheet)
            records.append(record_candidate(
                candidate, profile, layers, locked, experiment, path))
            print(path, flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            errors.append({
                "frame": candidate.path.name,
                "label": candidate.label,
                "error": str(error),
            })

    contact_folder = args.output / "contact-sheets"
    for offset in range(0, len(sheets), SHEETS_PER_PAGE):
        audit.flush_page(
            contact_folder, offset // SHEETS_PER_PAGE + 1,
            sheets[offset:offset + SHEETS_PER_PAGE])

    manifest = {
        "experiment_only": True,
        "locked_modules_modified": False,
        "interpretation": {
            "one_by_one": "one-pixel radius, implemented as 3x3 morphology",
            "high_layer": "P90 close constrained to P70",
            "lower_layer": "P40-and-above support opened with 3x3 kernel",
            "normalized_layer_influence": {
                "original_p90": LAYER_WEIGHTS[0],
                "p90_close_additions": LAYER_WEIGHTS[1],
                "opened_p40_additions": LAYER_WEIGHTS[2],
            },
        },
        "run": str(args.run),
        "selected_examples": len(selected),
        "rendered": len(records),
        "errors": errors,
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path, flush=True)
    print(json.dumps({
        "selected": len(selected),
        "rendered": len(records),
        "errors": len(errors),
        "pages": int(np.ceil(len(sheets) / SHEETS_PER_PAGE)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
