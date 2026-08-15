"""Archived: render bounded contour-angle influence beside prior fits."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from Vision.deterministic_v3.Legacy.legacy_ui.render_c_shape_contour_guided_review import (
    completion_panel, draw_outer_contour, local_points, segments, square_source)
from Vision.deterministic_v3.src_unused.c_shape_bounded_angle import (
    fit_bounded_angle_c_shape)
from Vision.deterministic_v3.src_unused.c_shape_three_line_pose import (
    CShapeDensityInput)
from Vision.deterministic_v3.src_unused.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, load_lut, prepare_mask)
from Vision.deterministic_v3.src_unused.legacy_inverse_density import heat_layer


RUN = (Path(__file__).resolve().parents[4] / "Viewer" / "logs" / "flight" / "runs" / "run-20260801T031401Z" / "vision_frames")
OUTPUT = Path("Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
              "c-shape-bounded-contour-angle-50pct-15deg-v1")
FRAME_ID = "00106766"
COMPONENT_LABEL = 3
PANEL = 300
HEADER = 88


def alignment_panel(component, field, result, origin):
    panel = heat_layer(field, component.mask != 0)
    contour = result.parallel_result.contour_result
    draw_outer_contour(panel, contour.outer_contour_uv, origin)
    baseline_segments = segments(contour.baseline_extent)
    bounded_segments = segments(result.bounded_extent)
    for record in result.bounded_lines:
        before = local_points(baseline_segments[record.side_index], origin)
        after = local_points(bounded_segments[record.side_index], origin)
        cv2.line(panel, tuple(before[0]), tuple(before[1]), (0, 255, 0), 1,
                 cv2.LINE_AA)
        cv2.line(panel, tuple(after[0]), tuple(after[1]), (255, 255, 255), 1,
                 cv2.LINE_AA)
    return panel


def main():
    frame_path = next(RUN.glob(f"frame-{FRAME_ID}-*"))
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask, gated, _ = prepare_mask(image, load_lut())
    if gated:
        raise RuntimeError("Selected frame was component-gated")
    component = next(item for item in iter_component_inputs(mask)
                     if item.label == COMPONENT_LABEL)
    x, y, width, height = component.bbox
    if x == 0 or y == 0 or x + width == mask.shape[1] or y + height == mask.shape[0]:
        raise RuntimeError("Selected component is frame-clipped")
    profile = component.profile
    field = inverse_density_field(
        component.mask, profile.density_radius, profile.ridge_radius)
    origin = np.asarray(component.bbox[:2]) - np.asarray(component.offset)
    result = fit_bounded_angle_c_shape(CShapeDensityInput(
        f"{FRAME_ID}:{COMPONENT_LABEL}", component.mask, field, tuple(origin)))
    if result is None:
        raise RuntimeError("No bounded-angle result")

    contour = result.parallel_result.contour_result
    source = square_source(image, component)
    contour_source = source.copy()
    draw_outer_contour(contour_source, contour.outer_contour_uv, origin)
    panels = (
        source, contour_source, alignment_panel(component, field, result, origin),
        completion_panel(source, contour.baseline_extent,
                         contour.outer_contour_uv, origin),
        completion_panel(source, result.parallel_result.parallel_extent,
                         contour.outer_contour_uv, origin),
        completion_panel(source, result.bounded_extent,
                         contour.outer_contour_uv, origin))
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    labels = ("isolated source", "simplified outer contour",
              "green=P90 white=bounded angle", "approved baseline",
              "full-parallel (100%)", "bounded contour angle (50%, max 15deg)")
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 7, HEADER - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1,
                    cv2.LINE_AA)
    raw = ",".join(f"{line.raw_contour_delta_degrees:.2f}"
                   for line in result.bounded_lines)
    applied = ",".join(f"{line.applied_delta_degrees:.2f}"
                       for line in result.bounded_lines)
    cv2.putText(sheet,
                f"frame={FRAME_ID} label={COMPONENT_LABEL} bbox={width}x{height} "
                f"raw P90-to-contour deltas={raw} deg",
                (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1,
                cv2.LINE_AA)
    cv2.putText(sheet,
                f"applied deltas={applied} deg  influence=50% cap=15deg  "
                "P90 offset=median  cyan=outer contour magenta=missing line",
                (7, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1,
                cv2.LINE_AA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"frame-{FRAME_ID}-label-{COMPONENT_LABEL}"
    cv2.imwrite(str(OUTPUT / f"{stem}.png"), sheet)
    (OUTPUT / f"{stem}.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
