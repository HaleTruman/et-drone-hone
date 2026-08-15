"""Archived: render gamma sweeps with calibrated radii fixed."""

from pathlib import Path

import cv2
import numpy as np

from Vision.deterministic_v3.src_unused.legacy_inverse_density import heat_layer
from Vision.deterministic_v3.src_unused.legacy_inverse_density_production import (
    RELATIVE_CAP,
    _box_sum,
    _coordinate_grids,
    _density_denominator,
    _prepared_mask,
    component_radius_bucket,
    load_lut,
)


RUN = (Path(__file__).resolve().parents[4] / "Viewer" / "logs" / "flight" / "runs" / "run-20260801T031401Z" / "vision_frames")
OUTPUT = Path(
    "Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
    "density-gamma-sweep-fixed-radii")
SAMPLES = (
    ("compact-35px", "00106765", 1),
    ("small-59px", "00106765", 3),
    ("medium-82px", "00106630", 3),
    ("large-boundary-91px", "00106668", 1),
    ("large-rotated-185px", "00106821", 1),
)
INVERSE_GAMMAS = (1.20, 1.65, 2.10)
RIDGE_GAMMAS = (3.00, 4.15, 5.50)
TILE = 300
HEADER = 76
IMAGE = 250


def density_field(mask, density_radius, ridge_radius,
                  inverse_gamma, ridge_gamma):
    inside = mask != 0
    if not np.any(inside):
        return np.zeros(mask.shape, np.float64)
    foreground = inside.astype(np.float32)
    density = _box_sum(foreground, density_radius, cv2.CV_32F)
    density /= _density_denominator(*mask.shape, density_radius)
    density[~inside] = 0
    normalized = np.zeros_like(density, np.float32)
    mean_density = float(density[inside].mean())
    normalized[inside] = np.clip(
        density[inside] / (mean_density * RELATIVE_CAP), 0, 1)
    inverse = np.zeros_like(density, np.float32)
    inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / inverse_gamma)

    yy, xx = _coordinate_grids(*mask.shape)
    moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
    sums = _box_sum(moments, ridge_radius)
    mass, weighted_x, weighted_y = cv2.split(sums)
    centroid_x = np.divide(
        weighted_x, mass, out=np.zeros_like(mass), where=mass > 0)
    centroid_y = np.divide(
        weighted_y, mass, out=np.zeros_like(mass), where=mass > 0)
    distance = cv2.magnitude(xx - centroid_x, yy - centroid_y)
    ridge = np.power(
        1.0 - np.clip(distance / ridge_radius, 0, 1), ridge_gamma)
    ridge[(~inside) | (mass <= 0)] = 0
    return inverse * ridge


def component_data(image, lut, target_label):
    mask, gated = _prepared_mask(image, lut)
    if gated:
        raise RuntimeError("Unexpected gated sample")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if target_label >= count:
        raise RuntimeError(f"Missing label {target_label}")
    x, y, width, height, area = (
        int(value) for value in stats[target_label])
    component = labels[y:y + height, x:x + width] == target_label
    side = max(width, height)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    square_mask = np.zeros((side, side), np.uint8)
    square_mask[offset_y:offset_y + height,
                offset_x:offset_x + width] = component
    return (mask, square_mask, (x, y, width, height, area),
            (offset_x, offset_y))


def sample_field(mask, square_mask, stats, offsets, profile,
                 inverse_gamma, ridge_gamma):
    bucket, _, density_radius, ridge_radius = profile
    if bucket != "large":
        return density_field(
            square_mask, density_radius, ridge_radius,
            inverse_gamma, ridge_gamma)
    full = density_field(
        mask, density_radius, ridge_radius, inverse_gamma, ridge_gamma)
    x, y, width, height, _ = stats
    offset_x, offset_y = offsets
    component = square_mask[offset_y:offset_y + height,
                            offset_x:offset_x + width] != 0
    square = np.zeros(square_mask.shape, np.float64)
    crop = full[y:y + height, x:x + width]
    destination = square[offset_y:offset_y + height,
                         offset_x:offset_x + width]
    destination[component] = crop[component]
    return square


def render_sample(sample_name, image, mask, square_mask, stats, offsets):
    _, _, width, height, area = stats
    profile = component_radius_bucket(max(width, height))
    bucket, _, density_radius, ridge_radius = profile
    sheet = np.zeros((HEADER + TILE * 3, TILE * 3, 3), np.uint8)
    title = (f"{sample_name}  bbox={width}x{height} area={area}  {bucket}  "
             f"fixed density-r={density_radius} ridge-r={ridge_radius}  "
             f"relative-cap={RELATIVE_CAP:g}")
    cv2.putText(sheet, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "columns: inverse gamma 1.20 / 1.65 / 2.10    rows: ridge gamma 3.00 / 4.15 / 5.50",
        (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(sheet, "baseline is center tile", (8, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1,
                cv2.LINE_AA)
    for row, ridge_gamma in enumerate(RIDGE_GAMMAS):
        for column, inverse_gamma in enumerate(INVERSE_GAMMAS):
            field = sample_field(
                mask, square_mask, stats, offsets, profile,
                inverse_gamma, ridge_gamma)
            heat = heat_layer(field, square_mask != 0)
            resized = cv2.resize(
                heat, (IMAGE, IMAGE), interpolation=cv2.INTER_NEAREST)
            left = column * TILE + (TILE - IMAGE) // 2
            top = HEADER + row * TILE + 38
            sheet[top:top + IMAGE, left:left + IMAGE] = resized
            label = f"inv-gamma={inverse_gamma:.2f}  ridge-gamma={ridge_gamma:.2f}"
            color = ((0, 255, 255) if
                     (inverse_gamma == 1.65 and ridge_gamma == 4.15)
                     else (205, 205, 205))
            cv2.putText(sheet, label, (column * TILE + 16,
                                      HEADER + row * TILE + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1,
                        cv2.LINE_AA)
            if color == (0, 255, 255):
                cv2.rectangle(sheet, (left - 3, top - 3),
                              (left + IMAGE + 2, top + IMAGE + 2), color, 1)
    return sheet


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    paths = sorted(RUN.glob("frame-*"))
    for index, (sample_name, frame_id, target_label) in enumerate(SAMPLES, 1):
        path = next(path for path in paths if frame_id in path.name)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        data = component_data(image, lut, target_label)
        sheet = render_sample(sample_name, image, *data)
        output = OUTPUT / f"{index:02d}-{sample_name}.png"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"Unable to write {output}")
        print(output)


if __name__ == "__main__":
    main()
