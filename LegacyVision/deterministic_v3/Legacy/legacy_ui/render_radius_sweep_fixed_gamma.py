"""Archived: render radius sweeps at the selected gamma baseline."""

from pathlib import Path

import cv2
import numpy as np

from LegacyVision.deterministic_v3.Legacy.legacy_ui.render_density_gamma_sweep_fixed_radii import (
    RUN,
    SAMPLES,
    component_data,
    density_field,
)
from LegacyVision.deterministic_v3.Legacy.legacy_ui.render_density_gamma_sweep_overlaps import (
    evenly_spaced,
    overlap_candidates,
)
from LegacyVision.deterministic_v3.src_unused.legacy_inverse_density import heat_layer
from LegacyVision.deterministic_v3.src_unused.legacy_inverse_density_production import (
    INVERSE_GAMMA,
    RELATIVE_CAP,
    RIDGE_GAMMA,
    component_radius_bucket,
    load_lut,
)


OUTPUT = Path(
    "Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
    "radius-sweep-fixed-gamma-2p10-3p00")
RADIUS_SWEEPS = {
    "compact": ((1, 2, 3), (1, 2, 3)),
    "small": ((3, 4, 5), (2, 3, 4)),
    "medium": ((5, 7, 9), (4, 6, 8)),
    "large": ((16, 20, 24), (12, 16, 20)),
}
TILE = 300
HEADER = 76
IMAGE = 250


def radius_field(mask, square_mask, stats, offsets, bucket,
                 density_radius, ridge_radius):
    if bucket != "large":
        return density_field(
            square_mask, density_radius, ridge_radius,
            INVERSE_GAMMA, RIDGE_GAMMA)
    full = density_field(
        mask, density_radius, ridge_radius, INVERSE_GAMMA, RIDGE_GAMMA)
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


def render_sample(name, mask, square_mask, stats, offsets):
    _, _, width, height, area = stats
    bucket, _, baseline_density, baseline_ridge = component_radius_bucket(
        max(width, height))
    density_radii, ridge_radii = RADIUS_SWEEPS[bucket]
    sheet = np.zeros((HEADER + TILE * 3, TILE * 3, 3), np.uint8)
    title = (
        f"{name}  bbox={width}x{height} area={area}  {bucket}  "
        f"fixed inv-gamma={INVERSE_GAMMA:.2f} ridge-gamma={RIDGE_GAMMA:.2f} "
        f"cap={RELATIVE_CAP:g}")
    cv2.putText(sheet, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.41, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"columns: density radius {density_radii}    rows: ridge radius {ridge_radii}",
        (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 185, 185), 1,
        cv2.LINE_AA)
    cv2.putText(sheet, "current radius baseline is center tile", (8, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1,
                cv2.LINE_AA)

    for row, ridge_radius in enumerate(ridge_radii):
        for column, density_radius in enumerate(density_radii):
            field = radius_field(
                mask, square_mask, stats, offsets, bucket,
                density_radius, ridge_radius)
            heat = heat_layer(field, square_mask != 0)
            resized = cv2.resize(
                heat, (IMAGE, IMAGE), interpolation=cv2.INTER_NEAREST)
            left = column * TILE + (TILE - IMAGE) // 2
            top = HEADER + row * TILE + 38
            sheet[top:top + IMAGE, left:left + IMAGE] = resized
            baseline = (density_radius == baseline_density and
                        ridge_radius == baseline_ridge)
            color = (0, 255, 255) if baseline else (205, 205, 205)
            label = f"density-r={density_radius}  ridge-r={ridge_radius}"
            cv2.putText(sheet, label, (column * TILE + 16,
                                      HEADER + row * TILE + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1,
                        cv2.LINE_AA)
            if baseline:
                cv2.rectangle(sheet, (left - 3, top - 3),
                              (left + IMAGE + 2, top + IMAGE + 2), color, 1)
    return sheet


def regular_samples(paths):
    samples = []
    for name, frame_id, label in SAMPLES:
        path = next(path for path in paths if frame_id in path.name)
        samples.append((name, path, label))
    return samples


def merged_samples(paths, lut):
    selected = evenly_spaced(overlap_candidates(paths, lut), 5)
    return [
        (f"overlap-{holes}holes-{max(width, height)}px", path, label)
        for _, path, label, holes, width, height, _ in selected
    ]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    paths = sorted(RUN.glob("frame-*"))
    samples = regular_samples(paths) + merged_samples(paths, lut)
    prepared = []
    for name, path, label in samples:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        data = component_data(image, lut, label)
        dimension = max(data[2][2], data[2][3])
        prepared.append((dimension, name, path, data))

    for index, (dimension, name, path, data) in enumerate(
            sorted(prepared, reverse=True), 1):
        sheet = render_sample(name, *data)
        frame_id = path.name.split("-")[1]
        output = OUTPUT / f"{index:02d}-{dimension:03d}px-{name}-{frame_id}.png"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"Unable to write {output}")
        print(output)


if __name__ == "__main__":
    main()
