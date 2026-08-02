"""Archived: render an isolated C-shape and its unranked pose solutions."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from LegacyVision.deterministic_v3.src_unused.c_shape_three_line_pose import (
    CShapeDensityInput, solve_c_shape_pose)
from LegacyVision.deterministic_v3.src_unused.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, load_lut, prepare_mask)
from LegacyVision.deterministic_v3.src_unused.legacy_inverse_density import heat_layer


RUN = (Path(__file__).resolve().parents[4] / "Logs" / "flight" / "runs" / "run-20260801T031401Z" / "vision_frames")
OUTPUT = Path("Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
              "c-shape-three-line-pose-v1")
FRAME_ID = "00106766"
COMPONENT_LABEL = 3
PANEL = 360
HEADER = 76


def square_source(image, component):
    x, y, width, height = component.bbox
    offset_x, offset_y = component.offset
    side = component.mask.shape[0]
    square = np.zeros((side, side, 3), np.uint8)
    square[offset_y:offset_y + height, offset_x:offset_x + width] = (
        image[y:y + height, x:x + width])
    return square


def draw_dashed(image, start, end, color):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    for begin in np.arange(0, length, 6.0):
        a, b = begin / length, min(begin + 3.0, length) / length
        cv2.line(image, tuple(np.rint(start + a * (end - start)).astype(int)),
                 tuple(np.rint(start + b * (end - start)).astype(int)),
                 color, 1, cv2.LINE_AA)


def draw_candidate(source, candidate, missing, origin):
    panel = source.copy()
    corners = np.asarray(candidate.quadrilateral_uv) - origin
    for side in range(4):
        start, end = corners[side], corners[(side + 1) % 4]
        if side == missing:
            draw_dashed(panel, start, end, (255, 0, 255))
        else:
            cv2.line(panel, tuple(np.rint(start).astype(int)),
                     tuple(np.rint(end).astype(int)), (0, 255, 0), 1,
                     cv2.LINE_AA)
    return panel


def draw_lines(component, field, result, origin):
    panel = heat_layer(field, component.mask != 0)
    panel[(component.mask != 0) & (field >= result.threshold)] = (245, 245, 245)
    side = component.mask.shape[0]
    for visible in result.visible_lines:
        a, b, c = visible.coefficients
        local_c = c + a * origin[0] + b * origin[1]
        point = -local_c * np.array((a, b))
        direction = np.array((-b, a))
        start = tuple(np.rint(point - 2 * side * direction).astype(int))
        end = tuple(np.rint(point + 2 * side * direction).astype(int))
        clipped, start, end = cv2.clipLine((0, 0, side, side), start, end)
        if clipped:
            cv2.line(panel, start, end, (0, 255, 0), 1, cv2.LINE_AA)
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
    result = solve_c_shape_pose(CShapeDensityInput(
        f"{FRAME_ID}:{COMPONENT_LABEL}", component.mask, field, tuple(origin)))
    if result is None or not result.candidates:
        raise RuntimeError("No three-line pose candidates")

    source = square_source(image, component)
    panels = [source, draw_lines(component, field, result, origin)]
    panels.extend(draw_candidate(source, candidate, result.missing_side_index, origin)
                  for candidate in result.candidates[:2])
    while len(panels) < 4:
        panels.append(np.zeros_like(source))
    sheet = np.zeros((HEADER + PANEL, PANEL * 4, 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    labels = ("isolated source", "P90 + three fitted lines",
              "unranked solution A", "unranked solution B")
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 8, HEADER - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)
    supports = ",".join(str(line.support_points) for line in result.visible_lines)
    cv2.putText(sheet,
                f"frame={FRAME_ID} label={COMPONENT_LABEL} bbox={width}x{height} "
                f"missing-side={result.missing_side_index} supports={supports}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1,
                cv2.LINE_AA)
    cv2.putText(sheet,
                "green=observed side fit  magenta dashed=pose-projected missing side  "
                "c_shape_three_line_pose.py::solve_c_shape_pose",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1,
                cv2.LINE_AA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT / f"frame-{FRAME_ID}-label-{COMPONENT_LABEL}.png"), sheet)
    (OUTPUT / f"frame-{FRAME_ID}-label-{COMPONENT_LABEL}.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
