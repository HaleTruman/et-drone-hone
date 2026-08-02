"""Archived: render fixed-radius gamma sweeps for merged masks."""

from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.ui.legacy.render_density_gamma_sweep_fixed_radii import (
    RUN,
    component_data,
    render_sample,
)
from sensing.vision.models.deterministic_v3.src.legacy_inverse_density_production import (
    _prepared_mask,
    load_lut,
)


OUTPUT = Path(
    "Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
    "density-gamma-sweep-fixed-radii-overlaps")
SAMPLE_COUNT = 5
MIN_HOLE_AREA = 12.0


def significant_holes(component):
    contours, hierarchy = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(
        item[3] >= 0 and cv2.contourArea(contour) >= MIN_HOLE_AREA
        for contour, item in zip(contours, hierarchy[0]))


def overlap_candidates(paths, lut):
    candidates = []
    for frame_index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask, gated = _prepared_mask(image, lut)
        if gated:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        for label in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[label])
            component = labels[y:y + height, x:x + width] == label
            holes = significant_holes(component)
            if holes >= 2:
                candidates.append(
                    (frame_index, path, label, holes, width, height, area))
    return candidates


def evenly_spaced(candidates, count):
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [candidates[index] for index in indices]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    paths = sorted(RUN.glob("frame-*"))
    candidates = overlap_candidates(paths, lut)
    selected = evenly_spaced(candidates, SAMPLE_COUNT)
    if not selected:
        raise RuntimeError("No merged multi-opening components found")

    for index, (_, path, label, holes, width, height, area) in enumerate(
            selected, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        data = component_data(image, lut, label)
        frame_id = path.name.split("-")[1]
        name = f"overlap-{holes}holes-{max(width, height)}px-frame-{frame_id}"
        sheet = render_sample(name, image, *data)
        output = OUTPUT / f"{index:02d}-{name}.png"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"Unable to write {output}")
        print(f"{output}  label={label} bbox={width}x{height} area={area}")


if __name__ == "__main__":
    main()
