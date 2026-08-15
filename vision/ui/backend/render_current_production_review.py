"""Render exact current production fields and P90 fits by mask topology."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from Vision.deterministic_v3.src_unused.inverse_density_quadrilateral_production import (
    INVERSE_GAMMA,
    P90_PERCENTILE,
    RELATIVE_CAP,
    RIDGE_GAMMA,
    fit_p90_quadrilateral,
    inverse_density_field,
    iter_component_inputs,
    load_lut,
    prepare_mask,
)
from Vision.deterministic_v3.src_unused.legacy_inverse_density import heat_layer


LEGACYVISION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = LEGACYVISION_ROOT.parent
RUN = PROJECT_ROOT / "Viewer" / "logs" / "flight" / "runs" / \
    "run-20260801T031401Z" / "vision_frames"
OUTPUT = Path(
    LEGACYVISION_ROOT / "deterministic_v3" / "Legacy" /
    "production_samples" / "current-production-by-size-and-topology-v3")
MIN_HOLE_AREA = 12.0
SAMPLES_PER_GROUP = 8
PANEL = 250
HEADER = 62
CURATED_OPEN_SHAPES = (
    ("00106708", 586),
    ("00106887", 349),
    ("00106753", 221),
    ("00106682", 110),
    ("00106784", 39),
    ("00106774", 30),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    frame_index: int
    component: object
    holes: int
    solidity: float
    touches_frame: bool

    @property
    def dimension(self):
        return max(self.component.bbox[2:])


def topology(component_mask):
    mask = component_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return 0, 1.0
    hierarchy = hierarchy[0]
    outer = [index for index, item in enumerate(hierarchy) if item[3] < 0]
    outer_index = max(outer, key=lambda index: cv2.contourArea(contours[index]))
    outer_contour = contours[outer_index]
    hull_area = cv2.contourArea(cv2.convexHull(outer_contour))
    area = cv2.countNonZero(mask)
    solidity = float(area / hull_area) if hull_area > 0 else 1.0
    holes = sum(
        item[3] >= 0 and cv2.contourArea(contour) >= MIN_HOLE_AREA
        for contour, item in zip(contours, hierarchy))
    return holes, solidity


def scan(paths, lut):
    regular, overlap, open_shapes = [], [], []
    for frame_index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask, gated, _ = prepare_mask(image, lut)
        if gated:
            continue
        image_height, image_width = mask.shape
        for component in iter_component_inputs(mask):
            holes, solidity = topology(component.mask)
            x, y, width, height = component.bbox
            touches = (x == 0 or y == 0 or x + width == image_width or
                       y + height == image_height)
            extent = component.area / max(width * height, 1)
            candidate = Candidate(
                path, frame_index, component, holes, solidity, touches)
            if holes >= 2:
                overlap.append(candidate)
            elif holes == 1:
                regular.append(candidate)
            elif (min(width, height) >= 20 and extent >= 0.32 and
                  (touches or solidity <= 0.78)):
                open_shapes.append(candidate)
    return regular, overlap, open_shapes


def dimension_spread(candidates, count):
    candidates = sorted(candidates, key=lambda item: item.dimension)
    if len(candidates) <= count:
        return list(reversed(candidates))
    targets = np.quantile(
        [item.dimension for item in candidates], np.linspace(0, 1, count))
    selected = []
    used_paths = set()
    for target in targets:
        available = [item for item in candidates if item.path not in used_paths]
        if not available:
            available = candidates
        choice = min(
            available,
            key=lambda item: (abs(item.dimension - target), item.frame_index))
        selected.append(choice)
        used_paths.add(choice.path)
    unique = {}
    for item in selected:
        unique[(item.path, item.component.label)] = item
    return sorted(unique.values(), key=lambda item: item.dimension, reverse=True)


def curated_open_shapes(candidates):
    selected = []
    for frame_id, dimension in CURATED_OPEN_SHAPES:
        matches = [
            item for item in candidates
            if frame_id in item.path.name and item.dimension == dimension
        ]
        if matches:
            selected.append(matches[0])
    return sorted(selected, key=lambda item: item.dimension, reverse=True)


def square_source(image, bbox, offset, side):
    x, y, width, height = bbox
    offset_x, offset_y = offset
    square = np.zeros((side, side, 3), np.uint8)
    square[offset_y:offset_y + height, offset_x:offset_x + width] = (
        image[y:y + height, x:x + width])
    return square


def fit_panel(component, field, fit):
    panel = heat_layer(field, component.mask != 0)
    if fit is None:
        return panel
    evidence = ((component.mask != 0) & (field >= fit.threshold))
    panel[evidence] = (245, 245, 245)
    scale = PANEL / component.mask.shape[0]
    corners = np.rint(fit.corners * scale).astype(np.int32).reshape(-1, 1, 2)
    panel = cv2.resize(panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    cv2.polylines(panel, [corners], True, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def render(candidate, group, lut):
    image = cv2.imread(str(candidate.path), cv2.IMREAD_COLOR)
    component = candidate.component
    profile = component.profile
    field = inverse_density_field(
        component.mask, profile.density_radius, profile.ridge_radius)
    fit = fit_p90_quadrilateral(component.mask, field)
    side = component.mask.shape[0]
    source = square_source(image, component.bbox, component.offset, side)
    mask = cv2.cvtColor(component.mask * 255, cv2.COLOR_GRAY2BGR)
    heat = heat_layer(field, component.mask != 0)
    panels = [source, mask, heat, fit_panel(component, field, fit)]

    sheet = np.zeros((HEADER + PANEL, PANEL * 4, 3), np.uint8)
    for index, panel in enumerate(panels):
        if panel.shape[:2] != (PANEL, PANEL):
            panel = cv2.resize(
                panel, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        sheet[HEADER:, index * PANEL:(index + 1) * PANEL] = panel
    labels = ("source", "prepared mask", "production density", "P90 + quad")
    for index, label in enumerate(labels):
        cv2.putText(sheet, label, (index * PANEL + 7, HEADER - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 205, 205), 1,
                    cv2.LINE_AA)
    fit_status = "fit" if fit is not None else "no-fit"
    title = (
        f"{group}  {candidate.dimension}px  frame={candidate.path.name.split('-')[1]} "
        f"label={component.label} bbox={component.bbox[2]}x{component.bbox[3]} "
        f"area={component.area} holes={candidate.holes} solidity={candidate.solidity:.2f} "
        f"{profile.name} dR={profile.density_radius} rR={profile.ridge_radius} {fit_status}")
    cv2.putText(sheet, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"inv-gamma={INVERSE_GAMMA:.2f} ridge-gamma={RIDGE_GAMMA:.2f} "
        f"cap={RELATIVE_CAP:g} percentile=P{P90_PERCENTILE:g}",
        (7, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
        cv2.LINE_AA)
    return sheet


def write_group(name, candidates, lut):
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, candidate in enumerate(candidates, 1):
        sheet = render(candidate, name, lut)
        output = folder / f"{index:02d}-{candidate.dimension:03d}px.png"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"Unable to write {output}")
        rendered.append(sheet)
        print(output)
    if rendered:
        contact = np.vstack(rendered)
        output = OUTPUT / f"{name}-ordered-by-pixel-size.png"
        if not cv2.imwrite(str(output), contact):
            raise OSError(f"Unable to write {output}")
        print(output)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    paths = sorted(RUN.glob("frame-*"))
    regular, overlap, open_shapes = scan(paths, lut)
    groups = {
        "regular": dimension_spread(regular, SAMPLES_PER_GROUP),
        "overlap-multi-hole": dimension_spread(overlap, SAMPLES_PER_GROUP),
        "c-shape-or-cutoff": curated_open_shapes(open_shapes),
    }
    for name, candidates in groups.items():
        write_group(name, candidates, lut)


if __name__ == "__main__":
    main()
