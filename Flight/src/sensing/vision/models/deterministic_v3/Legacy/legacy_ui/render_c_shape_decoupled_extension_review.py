"""Archived: render fixed green paths and independently scaled extensions."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.ui.legacy.render_c_shape_contour_guided_review import (
    draw_dashed, draw_outer_contour, local_points, square_source)
from sensing.vision.models.deterministic_v3.src.c_shape_decoupled_extension import (
    fit_decoupled_extension_sweep)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)
from sensing.vision.models.deterministic_v3.src.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, load_lut, prepare_mask)


RUN = Path("Flight/logs/runs/run-20260801T031401Z/vision_frames")
OUTPUT = Path("Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
              "c-shape-decoupled-contour-exit-sweep-50pct-15deg-v2")
FRAME_ID = "00106766"
COMPONENT_LABEL = 3
PANEL = 300
HEADER = 90


def extension_panel(source, extent, contour, origin, observed_exits,
                    candidate=None, extension_colors=None):
    panel = source.copy()
    draw_outer_contour(panel, contour, origin)
    closed = local_points(extent.closed_intersections_uv, origin)
    exits = local_points(observed_exits, origin)
    cv2.line(panel, tuple(closed[0]), tuple(closed[1]), (0, 255, 0), 1,
             cv2.LINE_AA)
    for start, exit_point in zip(closed, exits):
        cv2.line(panel, tuple(start), tuple(exit_point), (0, 255, 0), 1,
                 cv2.LINE_AA)
        cv2.circle(panel, tuple(exit_point), 2, (255, 255, 255), -1,
                   cv2.LINE_AA)
    if candidate is not None:
        endpoints = local_points(candidate.extended_endpoints_uv, origin)
        colors = extension_colors or ((0, 255, 255), (0, 255, 255))
        for exit_point, endpoint, color in zip(exits, endpoints, colors):
            cv2.line(panel, tuple(exit_point), tuple(endpoint), color,
                     1, cv2.LINE_AA)
        draw_dashed(panel, endpoints[0], endpoints[1], (255, 0, 255))
    return panel


def main():
    frame_path = next(RUN.glob(f"frame-{FRAME_ID}-*"))
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask, gated, _ = prepare_mask(image, load_lut())
    if gated:
        raise RuntimeError("Selected frame was component-gated")
    component = next(item for item in iter_component_inputs(mask)
                     if item.label == COMPONENT_LABEL)
    profile = component.profile
    field = inverse_density_field(
        component.mask, profile.density_radius, profile.ridge_radius)
    origin = np.asarray(component.bbox[:2]) - np.asarray(component.offset)
    result = fit_decoupled_extension_sweep(CShapeDensityInput(
        f"{FRAME_ID}:{COMPONENT_LABEL}", component.mask, field, tuple(origin)))
    if result is None:
        raise RuntimeError("No decoupled-extension result")

    bounded = result.bounded_result
    extent = bounded.bounded_extent
    contour = bounded.parallel_result.contour_result.outer_contour_uv
    source = square_source(image, component)
    observed_exits = result.candidates[0].observed_exits_uv
    panels = [extension_panel(
        source, extent, contour, origin, observed_exits)]
    panels.extend(extension_panel(
        source, extent, contour, origin, observed_exits, candidate)
                  for candidate in result.candidates)
    panels.insert(0, source)
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    labels = ["isolated source", "green fitted paths only"]
    labels.extend(
        f"yellow={candidate.extension_scales[0]:g}x half-width"
        for candidate in result.candidates)
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 7, HEADER - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1,
                    cv2.LINE_AA)
    cv2.putText(sheet,
                f"frame={FRAME_ID} label={COMPONENT_LABEL} influence=50% "
                f"cap=15deg half-width={extent.half_width_px:.3f}px",
                (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1,
                cv2.LINE_AA)
    cv2.putText(sheet,
                "green paths and white exits are fixed; yellow length changes "
                "along-line; magenta joins inferred corners",
                (7, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1,
                cv2.LINE_AA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"frame-{FRAME_ID}-label-{COMPONENT_LABEL}"
    cv2.imwrite(str(OUTPUT / f"{stem}.png"), sheet)
    (OUTPUT / f"{stem}.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
