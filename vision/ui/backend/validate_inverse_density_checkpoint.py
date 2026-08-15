"""Time the production detector and render five fixed validation samples."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from Vision.deterministic_v3.src_unused.inverse_density_quadrilateral_production import (
    detect_density_quadrilaterals,
    fit_p90_quadrilateral,
    inverse_density_field,
    iter_component_inputs,
    load_lut,
    prepare_mask,
)


EXTENSIONS = {".png", ".jpg", ".jpeg"}
SAMPLES = (
    ("compact 35 px", "00106765", 1),
    ("small 59 px", "00106765", 3),
    ("medium 82 px", "00106630", 3),
    ("large boundary 91 px", "00106668", 1),
    ("large rotated 185 px", "00106821", 1),
)
PANEL = 240
HEADER = 46
IMAGE_SIZE = 185


def frame_paths(run):
    folder = run / "vision_frames" if (run / "vision_frames").is_dir() else run
    return sorted(path for path in folder.iterdir()
                  if path.suffix.lower() in EXTENSIONS)


def _place_fitted(canvas, image, x, y, width, height, interpolation):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))))
    resized = cv2.resize(image, size, interpolation=interpolation)
    left = x + (width - size[0]) // 2
    top = y + (height - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized


def _local_panel(values, mask):
    maximum = float(values[mask != 0].max()) if np.any(mask) else 0.0
    normalized = np.zeros(mask.shape, np.uint8)
    if maximum > 0:
        normalized[mask != 0] = np.clip(
            values[mask != 0] * 255 / maximum, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heat[mask == 0] = 0
    return heat


def _render_sample(image, component, field, fit):
    row = np.zeros((PANEL, PANEL * 4, 3), np.uint8)
    x, y, width, height = component.bbox
    source = image[y:y + height, x:x + width]
    mask_bgr = cv2.cvtColor(component.mask * 255, cv2.COLOR_GRAY2BGR)
    heat = _local_panel(field, component.mask)
    evidence = heat.copy()
    support = ((component.mask != 0) & (field >= fit.threshold))
    evidence[support] = (255, 255, 255)

    for column, panel in enumerate((source, mask_bgr, heat)):
        _place_fitted(
            row, panel, column * PANEL + 10, HEADER, IMAGE_SIZE,
            PANEL - HEADER - 5,
            cv2.INTER_AREA if column == 0 else cv2.INTER_NEAREST)

    scale = IMAGE_SIZE / component.mask.shape[0]
    resized = cv2.resize(
        evidence, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    left, top = PANEL * 3 + 10, HEADER
    row[top:top + IMAGE_SIZE, left:left + IMAGE_SIZE] = resized
    corners = np.rint(fit.corners * scale + np.array((left, top))).astype(np.int32)
    cv2.polylines(row, [corners.reshape(-1, 1, 2)], True,
                  (0, 255, 255), 2, cv2.LINE_AA)

    profile = component.profile
    title = (f"{component.label=}  bbox={width}x{height} area={component.area}  "
             f"density-r={profile.density_radius} ridge-r={profile.ridge_radius}  "
             f"P90 points={fit.evidence_points}")
    cv2.putText(row, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (240, 240, 240), 1, cv2.LINE_AA)
    for column, label in enumerate((
            "source crop", "prepared square mask", "final field",
            "P90 evidence + quadrilateral")):
        cv2.putText(row, label, (column * PANEL + 10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1,
                    cv2.LINE_AA)
    return row


def render_samples(frames, lut, output):
    selected = []
    for sample_name, frame_id, label in SAMPLES:
        path = next((path for path in frames if frame_id in path.name), None)
        if path is None:
            raise ValueError(f"Missing sample frame {frame_id}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask, gated, _ = prepare_mask(image, lut)
        if gated:
            raise ValueError(f"Sample frame unexpectedly gated: {path}")
        component = next((item for item in iter_component_inputs(mask)
                          if item.label == label), None)
        if component is None:
            raise ValueError(f"Missing label {label} in {path}")
        profile = component.profile
        field = inverse_density_field(
            component.mask, profile.density_radius, profile.ridge_radius)
        fit = fit_p90_quadrilateral(component.mask, field)
        if fit is None:
            raise ValueError(f"P90 fit failed for label {label} in {path}")
        row = _render_sample(image, component, field, fit)
        cv2.putText(row, sample_name, (650, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 255), 1, cv2.LINE_AA)
        selected.append(row)
    report = np.vstack(selected)
    path = output / "five-gate-checkpoint.png"
    if not cv2.imwrite(str(path), report):
        raise OSError(f"Unable to write {path}")
    return path


def benchmark(frames, lut):
    compute_ms = []
    total_quadrilaterals = gated_frames = retained_components = 0
    wall_start = perf_counter()
    for path in frames:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read {path}")
        compute_start = perf_counter()
        batch = detect_density_quadrilaterals(image, lut)
        compute_ms.append((perf_counter() - compute_start) * 1000)
        gated_frames += int(batch.gated)
        retained_components += batch.retained_components
        total_quadrilaterals += len(batch.quadrilaterals)
    wall_seconds = perf_counter() - wall_start
    values = np.asarray(compute_ms)
    return {
        "frames": len(frames), "gated_frames": gated_frames,
        "retained_components": retained_components,
        "quadrilaterals": total_quadrilaterals,
        "compute_mean_ms_per_frame": float(values.mean()),
        "compute_median_ms_per_frame": float(np.median(values)),
        "compute_p95_ms_per_frame": float(np.percentile(values, 95)),
        "compute_max_ms_per_frame": float(values.max()),
        "wall_seconds": wall_seconds,
        "wall_ms_per_frame": wall_seconds * 1000 / len(frames),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = frame_paths(args.run)
    if not frames:
        parser.error(f"No frames found in {args.run}")
    lut = load_lut()
    metrics = benchmark(frames, lut)
    report = render_samples(frames, lut, args.output)
    (args.output / "checkpoint-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
