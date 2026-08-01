"""Render outer-contour guidance of the three C-shape density lines."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_contour_guided import (
    CONTOUR_SIMPLIFY_EPSILON_PX, fit_contour_guided_c_shape)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)
from sensing.vision.models.deterministic_v3.src.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, load_lut, prepare_mask)
from sensing.vision.models.deterministic_v3.src.legacy_inverse_density import heat_layer


RUN = Path("Flight/logs/runs/run-20260801T031401Z/vision_frames")
OUTPUT = Path("Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
              "c-shape-contour-guided-simplify-3px-v1")
FRAME_ID = "00106766"
COMPONENT_LABEL = 3
PANEL = 320
HEADER = 86


def square_source(image, component):
    x, y, width, height = component.bbox
    offset_x, offset_y = component.offset
    side = component.mask.shape[0]
    square = np.zeros((side, side, 3), np.uint8)
    square[offset_y:offset_y + height, offset_x:offset_x + width] = (
        image[y:y + height, x:x + width])
    return square


def local_points(values, origin):
    return np.rint(np.asarray(values) - origin).astype(np.int32)


def segments(extent):
    missing = extent.missing_side_index
    first, second, spine = (missing - 1) % 4, (missing + 1) % 4, (missing + 2) % 4
    closed = np.asarray(extent.closed_intersections_uv)
    exits = np.asarray(extent.outer_exits_uv)
    return {first: (closed[0], exits[0]), second: (closed[1], exits[1]),
            spine: (closed[0], closed[1])}


def draw_dashed(image, start, end, color):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    for begin in np.arange(0, length, 6.0):
        a, b = begin / length, min(begin + 3.0, length) / length
        cv2.line(image, tuple(np.rint(start + a * (end - start)).astype(int)),
                 tuple(np.rint(start + b * (end - start)).astype(int)),
                 color, 1, cv2.LINE_AA)


def draw_outer_contour(panel, contour, origin):
    points = local_points(contour, origin).reshape(-1, 1, 2)
    cv2.polylines(panel, [points], True, (255, 255, 0), 1, cv2.LINE_AA)


def line_comparison_panel(component, field, result, origin):
    panel = heat_layer(field, component.mask != 0)
    draw_outer_contour(panel, result.outer_contour_uv, origin)
    baseline_segments = segments(result.baseline_extent)
    guided_segments = segments(result.guided_extent)
    for record in result.guided_lines:
        before = local_points(baseline_segments[record.side_index], origin)
        after = local_points(guided_segments[record.side_index], origin)
        cv2.line(panel, tuple(before[0]), tuple(before[1]), (0, 255, 0), 1,
                 cv2.LINE_AA)
        cv2.line(panel, tuple(after[0]), tuple(after[1]), (255, 255, 255), 1,
                 cv2.LINE_AA)
        for midpoint in local_points(record.midpoint_samples_uv, origin):
            cv2.circle(panel, tuple(midpoint), 1, (0, 255, 255), -1)
    return panel


def completion_panel(source, extent, contour, origin):
    panel = source.copy()
    draw_outer_contour(panel, contour, origin)
    closed = local_points(extent.closed_intersections_uv, origin)
    exits = local_points(extent.outer_exits_uv, origin)
    endpoints = local_points(extent.extended_endpoints_uv, origin)
    cv2.line(panel, tuple(closed[0]), tuple(closed[1]), (0, 255, 0), 1,
             cv2.LINE_AA)
    for start, exit_point, endpoint in zip(closed, exits, endpoints):
        cv2.line(panel, tuple(start), tuple(exit_point), (0, 255, 0), 1,
                 cv2.LINE_AA)
        cv2.line(panel, tuple(exit_point), tuple(endpoint), (0, 255, 255), 1,
                 cv2.LINE_AA)
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
    x, y, width, height = component.bbox
    if x == 0 or y == 0 or x + width == mask.shape[1] or y + height == mask.shape[0]:
        raise RuntimeError("Selected component is frame-clipped")
    profile = component.profile
    field = inverse_density_field(
        component.mask, profile.density_radius, profile.ridge_radius)
    origin = np.asarray(component.bbox[:2]) - np.asarray(component.offset)
    result = fit_contour_guided_c_shape(CShapeDensityInput(
        f"{FRAME_ID}:{COMPONENT_LABEL}", component.mask, field, tuple(origin)))
    if result is None:
        raise RuntimeError("No contour-guided line result")

    source = square_source(image, component)
    contour_source = source.copy()
    draw_outer_contour(contour_source, result.outer_contour_uv, origin)
    panels = (source, contour_source,
              line_comparison_panel(component, field, result, origin),
              completion_panel(source, result.baseline_extent,
                               result.outer_contour_uv, origin),
              completion_panel(source, result.guided_extent,
                               result.outer_contour_uv, origin))
    sheet = np.zeros((HEADER + PANEL, PANEL * len(panels), 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    labels = ("isolated source", "outer contour simplified 3px",
              "green=P90 white=guided yellow=midpoints",
              "approved finite-extent baseline", "contour-guided completion")
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 7, HEADER - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1,
                    cv2.LINE_AA)
    angles = ",".join(f"{line.angle_delta_degrees:.2f}" for line in result.guided_lines)
    rmses = ",".join(f"{line.p90_rmse_px:.2f}" for line in result.guided_lines)
    cv2.putText(sheet,
                f"frame={FRAME_ID} label={COMPONENT_LABEL} bbox={width}x{height} "
                f"contour epsilon={CONTOUR_SIMPLIFY_EPSILON_PX:g}px "
                f"angle deltas={angles} deg",
                (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1,
                cv2.LINE_AA)
    cv2.putText(sheet,
                f"guided-line P90 RMSE={rmses}px  cyan=outer contour "
                "green=observed yellow=outward extension magenta=missing line",
                (7, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1,
                cv2.LINE_AA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"frame-{FRAME_ID}-label-{COMPONENT_LABEL}"
    cv2.imwrite(str(OUTPUT / f"{stem}.png"), sheet)
    (OUTPUT / f"{stem}.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
