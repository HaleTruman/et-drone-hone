"""Discover and render a diverse non-clipped C-shape checkpoint sweep."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.ui.legacy.render_c_shape_contour_guided_review import (
    square_source)
from sensing.vision.models.deterministic_v3.ui.legacy.render_c_shape_decoupled_extension_review import (
    extension_panel)
from sensing.vision.models.deterministic_v3.src.c_shape_decoupled_extension import (
    LONGEST_GREEN_TARGET_SCALE, extend_from_observed_paths,
    fit_longest_green_extension)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)
from sensing.vision.models.deterministic_v3.src.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, load_lut, prepare_mask)


RUNS = Path("Flight/logs/runs")
OUTPUT = Path("Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
              "c-shape-checkpoint-75pct-tailored-shortfall-v6")
MIN_DIMENSION_PX = 20
MIN_FILL_RATIO = 0.32
MAX_SOLIDITY = 0.80
MIN_FRAME_GAP = 12
MAX_SAMPLES = 24
PANEL = 220
HEADER = 76
REJECTED_NON_GATE_COMPONENTS = {
    ("run-20260731T093159Z", "00259933", 3),
    ("run-20260731T093732Z", "00269987", 2),
    ("run-20260731T093732Z", "00269991", 1),
    ("run-20260731T093732Z", "00269999", 2),
    ("run-20260731T094021Z", "00275067", 2),
    ("run-20260731T094136Z", "00277262", 3),
    ("run-20260731T101628Z", "00001004", 1),
    ("run-20260731T101628Z", "00001070", 2),
    ("run-20260801T031401Z", "00106779", 1),
    ("run-20260801T031401Z", "00106876", 4),
    ("run-20260801T031401Z", "00106884", 4),
    ("run-20260801T031401Z", "00106920", 1),
    ("run-20260801T031401Z", "00106927", 1),
    ("run-20260801T031401Z", "00106991", 3),
}


@dataclass(frozen=True, slots=True)
class Candidate:
    run_id: str
    path: str
    frame_id: str
    frame_number: int
    label: int
    bbox: tuple[int, int, int, int]
    area: int
    dimension: int
    aspect_ratio: float
    fill_ratio: float
    solidity: float
    missing_side_index: int
    maximum_angle_delta_degrees: float


def topology(component_mask):
    mask = component_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return 0, 1.0
    hierarchy = hierarchy[0]
    outer_indices = [index for index, item in enumerate(hierarchy)
                     if item[3] < 0]
    outer = max(outer_indices, key=lambda index: cv2.contourArea(contours[index]))
    hull_area = cv2.contourArea(cv2.convexHull(contours[outer]))
    holes = sum(item[3] >= 0 and cv2.contourArea(contour) >= 12.0
                for contour, item in zip(contours, hierarchy))
    area = cv2.countNonZero(mask)
    return holes, float(area / hull_area) if hull_area > 0 else 1.0


def frame_id(path):
    return path.name.split("-")[1]


def scan(lut):
    records, statistics = [], {
        "frames": 0, "components": 0, "edge_clipped": 0,
        "topology_candidates": 0, "checkpoint_fits": 0,
        "fit_failures": 0}
    for run_path in sorted(RUNS.glob("run-*/vision_frames")):
        run_id = run_path.parent.name
        for path in sorted(run_path.glob("frame-*")):
            statistics["frames"] += 1
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            mask, gated, _ = prepare_mask(image, lut)
            if gated:
                continue
            image_height, image_width = mask.shape
            for component in iter_component_inputs(mask):
                statistics["components"] += 1
                x, y, width, height = component.bbox
                clipped = (x == 0 or y == 0 or x + width == image_width or
                           y + height == image_height)
                if clipped:
                    statistics["edge_clipped"] += 1
                    continue
                holes, solidity = topology(component.mask)
                fill = component.area / max(width * height, 1)
                if (holes != 0 or min(width, height) < MIN_DIMENSION_PX or
                        fill < MIN_FILL_RATIO or solidity > MAX_SOLIDITY):
                    continue
                statistics["topology_candidates"] += 1
                profile = component.profile
                field = inverse_density_field(
                    component.mask, profile.density_radius,
                    profile.ridge_radius)
                origin = np.asarray(component.bbox[:2]) - np.asarray(
                    component.offset)
                try:
                    result = fit_longest_green_extension(CShapeDensityInput(
                        f"{frame_id(path)}:{component.label}", component.mask,
                        field, tuple(origin)))
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    result = None
                if result is None or not result.candidates:
                    statistics["fit_failures"] += 1
                    continue
                statistics["checkpoint_fits"] += 1
                angles = result.bounded_result.bounded_lines
                records.append(Candidate(
                    run_id, str(path), frame_id(path), int(frame_id(path)),
                    component.label, tuple(map(int, component.bbox)),
                    int(component.area), max(width, height), width / height,
                    float(fill), solidity,
                    result.bounded_result.bounded_extent.missing_side_index,
                    max(abs(line.raw_contour_delta_degrees)
                        for line in angles)))
    return records, statistics


def temporal_deduplicate(candidates):
    retained = []
    for candidate in sorted(candidates, key=lambda item: (
            item.run_id, item.frame_number, item.label)):
        nearby = [item for item in retained
                  if item.run_id == candidate.run_id and
                  abs(item.frame_number - candidate.frame_number) < MIN_FRAME_GAP]
        if not nearby:
            retained.append(candidate)
            continue
        existing = nearby[-1]
        feature_change = max(
            abs(candidate.dimension - existing.dimension) /
            max(existing.dimension, 1),
            abs(candidate.solidity - existing.solidity),
            abs(candidate.aspect_ratio - existing.aspect_ratio) / 2.0)
        if (candidate.missing_side_index != existing.missing_side_index or
                feature_change >= 0.15):
            retained.append(candidate)
    return retained


def feature(candidate, ranges):
    values = np.asarray((
        np.log(max(candidate.dimension, 1)), candidate.aspect_ratio,
        candidate.solidity, candidate.fill_ratio,
        candidate.maximum_angle_delta_degrees), np.float64)
    normalized = (values - ranges[0]) / np.maximum(ranges[1] - ranges[0], 1e-9)
    missing = np.eye(4)[candidate.missing_side_index]
    return np.concatenate((normalized, missing))


def diverse_subset(candidates, count):
    if len(candidates) <= count:
        return candidates
    values = np.asarray([(
        np.log(max(item.dimension, 1)), item.aspect_ratio, item.solidity,
        item.fill_ratio, item.maximum_angle_delta_degrees)
        for item in candidates])
    ranges = (values.min(axis=0), values.max(axis=0))
    features = [feature(item, ranges) for item in candidates]
    reference = next((index for index, item in enumerate(candidates)
                      if item.frame_id == "00106766" and item.label == 3),
                     int(np.argmax(values[:, 0])))
    selected = [reference]
    represented_runs = {candidates[reference].run_id}
    while len(selected) < count:
        available = [index for index in range(len(candidates))
                     if index not in selected]
        unseen = [index for index in available
                  if candidates[index].run_id not in represented_runs]
        pool = unseen if unseen else available
        choice = max(pool, key=lambda index: min(
            float(np.linalg.norm(features[index] - features[prior]))
            for prior in selected))
        selected.append(choice)
        represented_runs.add(candidates[choice].run_id)
    return sorted((candidates[index] for index in selected),
                  key=lambda item: (item.dimension, item.run_id,
                                    item.frame_number))


def load_component(candidate, lut):
    image = cv2.imread(candidate.path, cv2.IMREAD_COLOR)
    mask, gated, _ = prepare_mask(image, lut)
    if gated:
        return None
    component = next((item for item in iter_component_inputs(mask)
                      if item.label == candidate.label), None)
    if component is None:
        return None
    profile = component.profile
    field = inverse_density_field(
        component.mask, profile.density_radius, profile.ridge_radius)
    origin = np.asarray(component.bbox[:2]) - np.asarray(component.offset)
    result = fit_longest_green_extension(CShapeDensityInput(
        f"{candidate.frame_id}:{candidate.label}", component.mask, field,
        tuple(origin)))
    return image, component, origin, result


def render_catalog(candidates, lut):
    columns, side, header = 6, 140, 34
    rows = int(np.ceil(len(candidates) / columns))
    catalog = np.zeros((rows * (side + header), columns * side, 3), np.uint8)
    for index, candidate in enumerate(candidates):
        image = cv2.imread(candidate.path, cv2.IMREAD_COLOR)
        mask, gated, _ = prepare_mask(image, lut)
        component = None if gated else next(
            (item for item in iter_component_inputs(mask)
             if item.label == candidate.label), None)
        if component is None:
            continue
        source = cv2.resize(square_source(image, component), (side, side),
                            interpolation=cv2.INTER_NEAREST)
        row, column = divmod(index, columns)
        x, y = column * side, row * (side + header)
        catalog[y + header:y + header + side, x:x + side] = source
        cv2.putText(catalog,
                    f"{index + 1:02d} {candidate.frame_id}:{candidate.label} "
                    f"{candidate.dimension}px",
                    (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(catalog,
                    f"{candidate.run_id[4:12]} m={candidate.missing_side_index} "
                    f"s={candidate.solidity:.2f}",
                    (x + 4, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                    (175, 175, 175), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUTPUT / "deduplicated-candidate-catalog.png"), catalog)


def render(candidate, lut):
    loaded = load_component(candidate, lut)
    if loaded is None:
        return None
    image, component, origin, result = loaded
    bounded = result.bounded_result
    extent = bounded.bounded_extent
    contour = bounded.parallel_result.contour_result.outer_contour_uv
    source = square_source(image, component)
    adaptive = result.candidates[0]
    exits = adaptive.observed_exits_uv
    fixed_candidates = tuple(
        extend_from_observed_paths(extent, scale, scale, exits)
        for scale in (1.0, 2.0, 3.0))
    panels = [source, extension_panel(
        source, extent, contour, origin, exits)]
    panels.extend(extension_panel(
        source, extent, contour, origin, exits, candidate_result)
        for candidate_result in fixed_candidates)
    green_arm_lengths = np.linalg.norm(
        np.asarray(exits) - np.asarray(extent.closed_intersections_uv), axis=1)
    shortest_arm = int(np.argmin(green_arm_lengths))
    adaptive_colors = [(0, 255, 255), (0, 255, 255)]
    adaptive_colors[shortest_arm] = (0, 0, 255)
    panels.append(extension_panel(
        source, extent, contour, origin, exits, adaptive, adaptive_colors))
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    labels = ["source", "green paths + contour exits"]
    labels.extend(("yellow=1x half-width", "yellow=2x half-width",
                   "yellow=3x half-width", "adaptive 75%: red=shorter yellow=longer"))
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 6, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (220, 220, 220), 1,
                    cv2.LINE_AA)
    deltas = ",".join(
        f"{line.applied_delta_degrees:.1f}"
        for line in bounded.bounded_lines)
    cv2.putText(sheet,
                f"{candidate.run_id} frame={candidate.frame_id} "
                f"label={candidate.label} bbox={candidate.bbox[2]}x{candidate.bbox[3]} "
                f"missing={candidate.missing_side_index} applied={deltas}deg",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(sheet,
                f"non-clipped solidity={candidate.solidity:.2f} "
                f"fill={candidate.fill_ratio:.2f} adaptive scales="
                f"{adaptive.extension_scales[0]:.2f},{adaptive.extension_scales[1]:.2f}",
                (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.31,
                (190, 190, 190), 1, cv2.LINE_AA)
    closed = np.asarray(extent.closed_intersections_uv)
    observed_exits = np.asarray(exits)
    arm_lengths = np.linalg.norm(observed_exits - closed, axis=1)
    spine_length = float(np.linalg.norm(closed[1] - closed[0]))
    longest_green = max(spine_length, *map(float, arm_lengths))
    target = LONGEST_GREEN_TARGET_SCALE * longest_green
    shortfalls = np.maximum(target - arm_lengths, 0.0)
    yellow_lengths = shortfalls
    metrics = {
        "green_arm_lengths_px": list(map(float, arm_lengths)),
        "green_spine_length_px": spine_length,
        "longest_green_length_px": longest_green,
        "longest_green_target_scale": LONGEST_GREEN_TARGET_SCALE,
        "target_length_px": target,
        "green_arm_shortfalls_px": list(map(float, shortfalls)),
        "yellow_extension_lengths_px": list(map(float, yellow_lengths)),
        "completed_arm_lengths_px": list(map(
            float, arm_lengths + yellow_lengths)),
        "yellow_half_width_scales": list(adaptive.extension_scales),
        "convex": adaptive.convex,
        "area_px2": adaptive.area_px2}
    return sheet, metrics


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    scanned, statistics = scan(lut)
    gate_candidates = [
        item for item in scanned
        if (item.run_id, item.frame_id, item.label)
        not in REJECTED_NON_GATE_COMPONENTS]
    statistics["visually_rejected_non_gate"] = len(scanned) - len(gate_candidates)
    deduplicated = temporal_deduplicate(gate_candidates)
    render_catalog(deduplicated, lut)
    selected = diverse_subset(deduplicated, MAX_SAMPLES)
    sheets, rendered = [], []
    for index, candidate in enumerate(selected, 1):
        rendered_result = render(candidate, lut)
        if rendered_result is None:
            continue
        sheet, adaptive_metrics = rendered_result
        output = OUTPUT / (
            f"{index:02d}-{candidate.run_id}-{candidate.frame_id}-"
            f"label-{candidate.label}.png")
        cv2.imwrite(str(output), sheet)
        sheets.append(sheet)
        rendered.append({**asdict(candidate),
                         "adaptive_extension": adaptive_metrics})
        print(output)
    if sheets:
        cv2.imwrite(str(OUTPUT / "all-selected-c-shapes.png"), np.vstack(sheets))
    manifest = {
        "checkpoint": {
            "contour_angular_influence": 0.50,
            "maximum_contour_correction_degrees": 15.0,
            "exit_rule": "first forward adjusted-line intersection with simplified outer contour",
            "extension_rule": "each arm extension equals max(0, 0.75 times longest green minus that arm's green length)",
            "comparison_scales": [1.0, 2.0, 3.0]},
        "selection": {
            "edge_clipped_components_excluded": True,
            "holes": 0, "minimum_dimension_px": MIN_DIMENSION_PX,
            "minimum_fill_ratio": MIN_FILL_RATIO,
            "maximum_solidity": MAX_SOLIDITY,
            "minimum_same_run_frame_gap": MIN_FRAME_GAP,
            "maximum_samples": MAX_SAMPLES,
            "strategy": "temporal deduplication then greedy feature diversity"},
        "statistics": {**statistics, "deduplicated": len(deduplicated),
                       "selected": len(rendered)},
        "deduplicated_candidates": [asdict(item) for item in deduplicated],
        "samples": rendered}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["statistics"], sort_keys=True))


if __name__ == "__main__":
    main()
