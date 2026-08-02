"""ARCHIVED stage audit for the locked two-aperture overlap checkpoint.

This offline renderer imports the checkpoint unchanged and exposes intermediate
preprocessing, density, evidence, and fitting stages. It is neither production
code nor a test. Panels explicitly identify values that are computed for review
but are not consumed by the locked solver.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.ui.backend import \
    render_overlap_checkpoint_broad_review as broad
from sensing.vision.models.deterministic_v3.src import \
    overlapping_gate_aperture_solver as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "production_samples" / \
    "multi-gate-overlap-checkpoint-stage-audit-v1"

COLUMNS = 5
CELL = 210
LABEL_HEIGHT = 24
HEADER = 74
SHEETS_PER_PAGE = 3
REPRESENTATIVE_COUNT = 12


def raw_mask(image, lut):
    bgr = image.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return (lut[keys] != 0).astype(np.uint8)


def area_filtered(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= base.SMALL_COMPONENT_AREA
    keep[0] = False
    return keep[labels].astype(np.uint8)


def closed_mask(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (base.CLOSE_KERNEL_PX, base.CLOSE_KERNEL_PX))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def square_crop(values, candidate):
    x, y, width, height = candidate.bbox
    side = candidate.side
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    shape = (side, side) if values.ndim == 2 else (side, side, values.shape[2])
    square = np.zeros(shape, values.dtype)
    square[offset_y:offset_y + height,
           offset_x:offset_x + width] = values[y:y + height, x:x + width]
    return square


def density_stages(mask, density_radius, ridge_radius):
    inside = mask != 0
    foreground = inside.astype(np.float32)
    density = base.box_sum(foreground, density_radius, cv2.CV_32F)
    denominator = base.box_sum(
        np.ones(mask.shape, np.float32), density_radius, cv2.CV_32F)
    density = np.divide(
        density, denominator, out=np.zeros_like(density), where=denominator > 0)
    density[~inside] = 0
    normalized = np.zeros_like(density)
    mean_density = float(density[inside].mean())
    normalized[inside] = np.clip(
        density[inside] / (mean_density * base.RELATIVE_CAP), 0, 1)
    inverse = np.zeros_like(density)
    inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / base.INVERSE_GAMMA)

    yy, xx = np.indices(mask.shape, dtype=np.float64)
    moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
    sums = base.box_sum(moments, ridge_radius)
    mass, weighted_x, weighted_y = cv2.split(sums)
    centroid_x = np.divide(
        weighted_x, mass, out=np.zeros_like(mass), where=mass > 0)
    centroid_y = np.divide(
        weighted_y, mass, out=np.zeros_like(mass), where=mass > 0)
    distance = cv2.magnitude(xx - centroid_x, yy - centroid_y)
    ridge = np.power(
        1.0 - np.clip(distance / ridge_radius, 0, 1), base.RIDGE_GAMMA)
    ridge[(~inside) | (mass <= 0)] = 0
    return inverse, ridge, inverse * ridge


def binary_panel(values, scale=255):
    gray = (values != 0).astype(np.uint8) * scale
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def field_panel(values, mask):
    return base.heat_layer(values, mask != 0)


def draw_quad(panel, corners, color, source_side, thickness=2):
    height, width = panel.shape[:2]
    points = corners.copy().astype(np.float64)
    points[:, 0] *= width / source_side
    points[:, 1] *= height / source_side
    points = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(panel, [points], True, color, thickness, cv2.LINE_AA)


def fit_panel(background, fits, source_side):
    panel = background.copy()
    for index, fit in enumerate(fits):
        draw_quad(panel, fit.corners, base.GATE_COLORS[index], source_side)
    return panel


def residual_panel(seed, result):
    panel = np.zeros((*seed.shape, 3), np.uint8)
    points = result["points"].astype(int)
    covered = base.distances_to_quadrilateral(
        result["points"], result["baseline"]["larger"].corners)
    for (x, y), distance in zip(points, covered):
        panel[y, x] = (70, 70, 70) if \
            distance <= base.COVERAGE_TOLERANCE_PX else (235, 235, 235)
    draw_quad(
        panel, result["baseline"]["smaller"].corners,
        base.GATE_COLORS[1], seed.shape[0], 1)
    return panel


def outer_guide_panel(seed, result):
    panel = binary_panel(seed, 75)
    source_side = seed.shape[0]
    contour = result["outer"].reshape(-1, 2).astype(np.float64)
    contour[:, 0] *= panel.shape[1] / source_side
    contour[:, 1] *= panel.shape[0] / source_side
    cv2.polylines(
        panel, [np.rint(contour).astype(np.int32).reshape(-1, 1, 2)], True,
        (255, 255, 0), 1, cv2.LINE_AA)
    for refinement in (result["larger"], result["smaller"]):
        for guide in refinement.guides:
            if guide.outer is None:
                continue
            endpoints = np.asarray((guide.outer.start, guide.outer.end), float)
            endpoints[:, 0] *= panel.shape[1] / source_side
            endpoints[:, 1] *= panel.shape[0] / source_side
            endpoints = np.rint(endpoints).astype(np.int32)
            cv2.line(
                panel, tuple(endpoints[0]), tuple(endpoints[1]),
                (0, 165, 255), 2, cv2.LINE_AA)
    draw_quad(
        panel, result["larger"].corners, base.GATE_COLORS[0], source_side, 1)
    draw_quad(
        panel, result["smaller"].corners, base.GATE_COLORS[1], source_side, 1)
    return panel


def text_panel(lines):
    panel = np.zeros((CELL, CELL, 3), np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            panel, line, (7, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 205, 205), 1,
            cv2.LINE_AA)
    return panel


def labeled_cell(image, label):
    resized = cv2.resize(
        image, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
    cell = np.zeros((LABEL_HEIGHT + CELL, CELL, 3), np.uint8)
    cell[LABEL_HEIGHT:] = resized
    cv2.putText(
        cell, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
        (210, 210, 210), 1, cv2.LINE_AA)
    return cell


def render_sheet(candidate, source, masks, field_values, evidence, result,
                 category, reasons, profile):
    raw, filtered, closed, component = masks
    inverse, ridge, field = field_values
    limit, seed, constrained_close = evidence
    source_panel = source.copy()
    aperture_panel = fit_panel(
        binary_panel(component, 70), [
            type("Aperture", (), {"corners": fit.aperture_corners})
            for fit in result["baseline"]["initial_fits"]
        ], candidate.side)
    initial_panel = fit_panel(
        binary_panel(seed, 70), result["baseline"]["initial_fits"],
        candidate.side)
    large_panel = fit_panel(
        binary_panel(seed, 70), [result["baseline"]["larger"]],
        candidate.side)
    baseline_panel = fit_panel(
        cv2.resize(source, (CELL, CELL), interpolation=cv2.INTER_NEAREST),
        [result["baseline"]["larger"], result["baseline"]["smaller"]],
        candidate.side)
    final_panel = fit_panel(
        cv2.resize(source, (CELL, CELL), interpolation=cv2.INTER_NEAREST),
        [result["larger"], result["smaller"]], candidate.side)
    coverage = broad.coverage_panel(seed, result)

    panels = (
        (source_panel, "01 source crop"),
        (binary_panel(raw), "02 raw LUT mask"),
        (binary_panel(filtered), "03 area filter >=100px"),
        (binary_panel(closed), "04 full-frame 5x5 close"),
        (binary_panel(component), "05 isolated component"),
        (field_panel(inverse, component), "06 inverse density"),
        (field_panel(ridge, component), "07 local ridge response"),
        (field_panel(field, component), "08 final density product"),
        (binary_panel(limit), "09 P70 allowed support"),
        (binary_panel(seed), "10 P90 primary evidence"),
        (binary_panel(constrained_close), "11 3x3 close (UNUSED)"),
        (aperture_panel, "12 aperture seed boxes"),
        (initial_panel, "13 initial P90 quads"),
        (large_panel, "14 larger optimized first"),
        (residual_panel(seed, result), "15 residual + smaller fit"),
        (baseline_panel, "16 locked P90/P70 result"),
        (outer_guide_panel(seed, result), "17 raw outer guidance"),
        (final_panel, "18 accepted checkpoint"),
        (coverage, "19 final evidence coverage"),
        (text_panel((
            "LOCKED REVIEW", "influence 0.50", "angle cap 15 deg",
            "inner contours unused", f"large: {reasons[0]}",
            f"small: {reasons[1]}", f"profile: {profile[0]}",
        )), "20 decision summary"),
    )
    cells = [labeled_cell(panel, label) for panel, label in panels]
    rows = len(cells) // COLUMNS
    sheet = np.zeros(
        (HEADER + rows * (LABEL_HEIGHT + CELL), COLUMNS * CELL, 3),
        np.uint8)
    for index, cell in enumerate(cells):
        row, column = divmod(index, COLUMNS)
        top = HEADER + row * (LABEL_HEIGHT + CELL)
        left = column * CELL
        sheet[top:top + cell.shape[0], left:left + CELL] = cell
    baseline = result["baseline"]["weighted_union_coverage"]
    refined = result["weighted_union_coverage"]
    cv2.putText(
        sheet,
        f"STAGE AUDIT  {candidate.side}px frame={candidate.frame_id} "
        f"label={candidate.label} bbox={candidate.bbox[2]}x{candidate.bbox[3]} "
        f"area={candidate.area} {category}",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
        (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"P90 union {100 * baseline:.1f}->{100 * refined:.1f}%  "
        f"density-r={profile[2]} ridge-r={profile[3]}  "
        f"large={reasons[0]} small={reasons[1]}",
        (7, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
        (185, 185, 185), 1, cv2.LINE_AA)
    return sheet


def flush_page(folder, page_number, sheets):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"page-{page_number:03d}.png"
    if not cv2.imwrite(str(path), np.vstack(sheets)):
        raise OSError(f"Unable to write {path}")
    print(path, flush=True)


def select_representatives(records, count):
    ordered = sorted(records, key=lambda item: item[0].side)
    if len(ordered) <= count:
        return ordered
    targets = np.linspace(0, len(ordered) - 1, count)
    return [ordered[int(round(target))] for target in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    instances = args.output / "instances"
    instances.mkdir(parents=True, exist_ok=True)

    lut = base.load_lut()
    paths = sorted(args.run.glob("frame-*"))
    candidates, gated_frames = broad.discover(paths, lut)
    eligible = sorted((
        item for item in candidates
        if item.holes == 2 and not item.touches_frame
    ), key=lambda item: (item.side, item.frame_index, item.label))
    category_sheets = defaultdict(list)
    category_pages = defaultdict(int)
    records = []
    errors = []

    for index, candidate in enumerate(eligible, 1):
        try:
            image = cv2.imread(str(candidate.path), cv2.IMREAD_COLOR)
            source_mask = raw_mask(image, lut)
            filtered_mask = area_filtered(source_mask)
            prepared_mask = closed_mask(filtered_mask)
            component, source = broad.load_component(candidate, lut)
            profile, field, seed, result, density_ms, solve_ms, \
                larger_reason, smaller_reason = broad.solve_candidate(
                    candidate, component)
            inverse, ridge, reconstructed_field = density_stages(
                component, profile[2], profile[3])
            if not np.allclose(field, reconstructed_field, atol=1e-7):
                raise RuntimeError("Stage density does not match locked field")
            _, _, verified_seed, limit, constrained_close = \
                base.density_support_constrained_close(field, component)
            if not np.array_equal(seed, verified_seed):
                raise RuntimeError("Stage P90 seed does not match locked seed")
            category = broad.classify(result)
            sheet = render_sheet(
                candidate, source,
                (square_crop(source_mask, candidate),
                 square_crop(filtered_mask, candidate),
                 square_crop(prepared_mask, candidate), component),
                (inverse, ridge, field),
                (limit, seed, constrained_close), result, category,
                (larger_reason, smaller_reason), profile)
            filename = (
                f"{index:05d}-{candidate.side:03d}px-frame-"
                f"{candidate.frame_id}-label-{candidate.label:03d}.png")
            path = instances / filename
            if not cv2.imwrite(str(path), sheet):
                raise OSError(f"Unable to write {path}")
            records.append((candidate, category, path, density_ms, solve_ms))
            category_sheets[category].append(sheet)
            if len(category_sheets[category]) == SHEETS_PER_PAGE:
                category_pages[category] += 1
                flush_page(
                    args.output / "contact-sheets" / category,
                    category_pages[category], category_sheets[category])
                category_sheets[category].clear()
        except (ValueError, RuntimeError, OSError) as error:
            errors.append({
                "frame": candidate.path.name,
                "label": candidate.label,
                "error": str(error),
            })
        if index % 10 == 0 or index == len(eligible):
            print(f"processed {index}/{len(eligible)}", flush=True)

    for category, sheets in category_sheets.items():
        if sheets:
            category_pages[category] += 1
            flush_page(
                args.output / "contact-sheets" / category,
                category_pages[category], sheets)

    representatives = select_representatives(records, REPRESENTATIVE_COUNT)
    representative_folder = args.output / "representative-by-pixel-size"
    for page_index in range(0, len(representatives), SHEETS_PER_PAGE):
        sheets = [cv2.imread(str(record[2]))
                  for record in representatives[
                      page_index:page_index + SHEETS_PER_PAGE]]
        flush_page(
            representative_folder,
            page_index // SHEETS_PER_PAGE + 1, sheets)

    manifest = {
        "review_only": True,
        "locked_modules_modified": False,
        "run": str(args.run),
        "frames_scanned": len(paths),
        "gated_frames": gated_frames,
        "eligible_exact_two_non_clipped": len(eligible),
        "rendered": len(records),
        "errors": errors,
        "stage_panels": [
            "source crop", "raw LUT mask", "area filter >=100px",
            "full-frame 5x5 close", "isolated component",
            "inverse density", "local ridge response",
            "final density product", "P70 allowed support",
            "P90 primary evidence", "3x3 P90 close constrained to P70 "
            "(computed but unused)", "aperture seed boxes",
            "initial P90 quadrilaterals", "larger optimized first",
            "residual P90 plus smaller fit", "locked P90/P70 result",
            "raw outer-contour guidance", "accepted checkpoint",
            "final evidence coverage", "decision summary",
        ],
        "records": [{
            "file": str(path.relative_to(args.output)),
            "frame": candidate.path.name,
            "label": candidate.label,
            "maximum_dimension_px": candidate.side,
            "category": category,
            "density_and_threshold_ms": density_ms,
            "checkpoint_solve_ms": solve_ms,
        } for candidate, category, path, density_ms, solve_ms in records],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path, flush=True)
    print(json.dumps({
        "rendered": len(records),
        "errors": len(errors),
        "category_pages": dict(category_pages),
        "representative_pages": int(np.ceil(
            len(representatives) / SHEETS_PER_PAGE)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
