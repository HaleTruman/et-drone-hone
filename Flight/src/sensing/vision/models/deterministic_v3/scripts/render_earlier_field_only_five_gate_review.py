"""Render five field-only samples directly from the earlier production core."""

from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.legacy_inverse_density import heat_layer
from sensing.vision.models.deterministic_v3.src.legacy_inverse_density_production import (
    _field_for_prepared_mask,
    _prepared_mask,
    component_radius_bucket,
    load_lut,
)


RUN = Path("Flight/logs/runs/run-20260801T031401Z/vision_frames")
OUTPUT = Path(
    "Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
    "earlier-field-only-five-gate-review.png")
SAMPLES = (
    ("compact 35 px", "00106765", 1),
    ("small 59 px", "00106765", 3),
    ("medium 82 px", "00106630", 3),
    ("large boundary 91 px", "00106668", 1),
    ("large rotated 185 px", "00106821", 1),
)
PANEL = 280
ROW_HEIGHT = 255
HEADER = 54
IMAGE = 190


def place(canvas, image, column, row, interpolation):
    x = column * PANEL + (PANEL - IMAGE) // 2
    y = row * ROW_HEIGHT + HEADER
    scale = min(IMAGE / image.shape[1], IMAGE / image.shape[0])
    size = (max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))))
    resized = cv2.resize(image, size, interpolation=interpolation)
    left = x + (IMAGE - size[0]) // 2
    top = y + (IMAGE - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized


def main():
    lut = load_lut()
    canvas = np.zeros((len(SAMPLES) * ROW_HEIGHT, PANEL * 3, 3), np.uint8)
    paths = sorted(RUN.glob("frame-*"))
    for row, (sample_name, frame_id, target_label) in enumerate(SAMPLES):
        path = next(path for path in paths if frame_id in path.name)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask, gated = _prepared_mask(image, lut)
        if gated:
            raise RuntimeError(f"Unexpected gated sample: {path}")
        field = _field_for_prepared_mask(mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if target_label >= count:
            raise RuntimeError(f"Missing label {target_label}: {path}")
        x, y, width, height, area = (
            int(value) for value in stats[target_label])
        component = labels[y:y + height, x:x + width] == target_label
        side = max(width, height)
        offset_x = (side - width) // 2
        offset_y = (side - height) // 2
        square_mask = np.zeros((side, side), np.uint8)
        square_field = np.zeros((side, side), np.float64)
        square_mask[offset_y:offset_y + height,
                    offset_x:offset_x + width] = component
        local_field = field[y:y + height, x:x + width]
        destination = square_field[offset_y:offset_y + height,
                                   offset_x:offset_x + width]
        destination[component] = local_field[component]
        heat = heat_layer(square_field, square_mask != 0)
        mask_bgr = cv2.cvtColor(square_mask * 255, cv2.COLOR_GRAY2BGR)
        source = image[y:y + height, x:x + width]
        bucket, _, density_radius, ridge_radius = component_radius_bucket(side)

        title = (f"{sample_name}  label={target_label} bbox={width}x{height} "
                 f"area={area}  {bucket} density-r={density_radius} "
                 f"ridge-r={ridge_radius}")
        cv2.putText(canvas, title, (8, row * ROW_HEIGHT + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 240, 240), 1,
                    cv2.LINE_AA)
        for column, label in enumerate((
                "source crop", "prepared component mask",
                "earlier final inverse-density field")):
            cv2.putText(canvas, label,
                        (column * PANEL + 8, row * ROW_HEIGHT + 39),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
                        cv2.LINE_AA)
        place(canvas, source, 0, row, cv2.INTER_AREA)
        place(canvas, mask_bgr, 1, row, cv2.INTER_NEAREST)
        place(canvas, heat, 2, row, cv2.INTER_NEAREST)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), canvas):
        raise OSError(f"Unable to write {OUTPUT}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
