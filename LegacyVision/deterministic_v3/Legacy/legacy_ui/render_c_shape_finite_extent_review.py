"""Archived: render mask-extent C-shape completion for one component."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from LegacyVision.deterministic_v3.src_unused.c_shape_finite_extent import (
    estimate_c_shape_extent)
from LegacyVision.deterministic_v3.src_unused.c_shape_three_line_pose import (
    CShapeDensityInput)
from LegacyVision.deterministic_v3.src_unused.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, load_lut, prepare_mask)
from LegacyVision.deterministic_v3.src_unused.legacy_inverse_density import heat_layer


RUN = (Path(__file__).resolve().parents[4] / "Logs" / "flight" / "runs" / "run-20260801T031401Z" / "vision_frames")
OUTPUT = Path("Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
              "c-shape-finite-extent-v1")
FRAME_ID = "00106766"
COMPONENT_LABEL = 3
PANEL = 360
HEADER = 82


def square_source(image, component):
    x, y, width, height = component.bbox
    offset_x, offset_y = component.offset
    side = component.mask.shape[0]
    square = np.zeros((side, side, 3), np.uint8)
    square[offset_y:offset_y + height, offset_x:offset_x + width] = (
        image[y:y + height, x:x + width])
    return square


def point(value, origin):
    return tuple(np.rint(np.asarray(value) - origin).astype(int))


def dashed(image, start, end, color):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    for begin in np.arange(0, length, 6.0):
        a, b = begin / length, min(begin + 3.0, length) / length
        cv2.line(image, tuple(np.rint(start + a * (end - start)).astype(int)),
                 tuple(np.rint(start + b * (end - start)).astype(int)),
                 color, 1, cv2.LINE_AA)


def distance_panel(component, result, origin):
    distance = cv2.distanceTransform(
        (component.mask != 0).astype(np.uint8), cv2.DIST_L2, 5)
    panel = heat_layer(distance, component.mask != 0)
    closed = [point(value, origin) for value in result.closed_intersections_uv]
    cv2.line(panel, closed[0], closed[1], (0, 255, 0), 1, cv2.LINE_AA)
    for location in closed:
        cv2.circle(panel, location, 2, (255, 255, 0), -1, cv2.LINE_AA)
    return panel


def construction_panel(component, field, result, origin):
    panel = heat_layer(field, component.mask != 0)
    panel[(component.mask != 0) & (field >= result.threshold)] = (245, 245, 245)
    closed = [point(value, origin) for value in result.closed_intersections_uv]
    exits = [point(value, origin) for value in result.outer_exits_uv]
    endpoints = [point(value, origin) for value in result.extended_endpoints_uv]
    cv2.line(panel, closed[0], closed[1], (0, 255, 0), 1, cv2.LINE_AA)
    for start, end in zip(closed, exits):
        cv2.line(panel, start, end, (0, 255, 0), 1, cv2.LINE_AA)
    for exit_point, endpoint in zip(exits, endpoints):
        cv2.line(panel, endpoint, exit_point, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(panel, exit_point, 2, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, endpoint, 2, (255, 255, 0), -1, cv2.LINE_AA)
    dashed(panel, endpoints[0], endpoints[1], (255, 0, 255))
    return panel


def quadrilateral_panel(source, result, origin):
    panel = source.copy()
    closed = [point(value, origin) for value in result.closed_intersections_uv]
    exits = [point(value, origin) for value in result.outer_exits_uv]
    endpoints = [point(value, origin) for value in result.extended_endpoints_uv]
    cv2.line(panel, closed[0], closed[1], (0, 255, 0), 1, cv2.LINE_AA)
    for start, exit_point, endpoint in zip(closed, exits, endpoints):
        cv2.line(panel, start, exit_point, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(panel, exit_point, endpoint, (0, 255, 255), 1, cv2.LINE_AA)
    dashed(panel, endpoints[0], endpoints[1], (255, 0, 255))
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
    result = estimate_c_shape_extent(CShapeDensityInput(
        f"{FRAME_ID}:{COMPONENT_LABEL}", component.mask, field, tuple(origin)))
    if result is None:
        raise RuntimeError("No finite-extent completion")

    source = square_source(image, component)
    panels = (source, distance_panel(component, result, origin),
              construction_panel(component, field, result, origin),
              quadrilateral_panel(source, result, origin))
    sheet = np.zeros((HEADER + PANEL, PANEL * 4, 3), np.uint8)
    for index, panel in enumerate(panels):
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = cv2.resize(
            panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    labels = ("isolated source", "distance transform + visible spine",
              "mask exits + outward half-width", "2-D exterior completion")
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 8, HEADER - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1,
                    cv2.LINE_AA)
    cv2.putText(sheet,
                f"frame={FRAME_ID} label={COMPONENT_LABEL} bbox={width}x{height} "
                f"half-width={result.half_width_px:.2f}px "
                f"full-width={result.full_width_px:.2f}px",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1,
                cv2.LINE_AA)
    cv2.putText(sheet,
                f"arm-extents={result.arm_extents_px[0]:.2f}px,"
                f"{result.arm_extents_px[1]:.2f}px  green=fit yellow=mask exit "
                "cyan=exterior endpoint magenta=estimated line",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1,
                cv2.LINE_AA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"frame-{FRAME_ID}-label-{COMPONENT_LABEL}"
    cv2.imwrite(str(OUTPUT / f"{stem}.png"), sheet)
    (OUTPUT / f"{stem}.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
