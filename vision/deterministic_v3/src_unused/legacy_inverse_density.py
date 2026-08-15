"""Full-resolution calibrated legacy inverse-density review pipeline."""

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT.parent
RUNS = MODELS_ROOT / "detection_v2" / "runs"
LOG_RUNS = ROOT.parents[5] / "Viewer" / "logs" / "flight" / "runs"
REVIEWS = ROOT / "legacy_review_runs"
MANIFEST = ROOT / "ui" / "frontend" / "data" / "legacy-runs-manifest.json"
LUT_PATH = ROOT / "assets" / "color_lut_v1.npz"
EXTENSIONS = {".png", ".jpg", ".jpeg"}

SMALL_COMPONENT_AREA = 100
CLOSE_KERNEL_PX = 5
DENSITY_RADIUS_PX = 12
RELATIVE_CAP = 2.0
INVERSE_GAMMA = 1.65
RIDGE_RADIUS_PX = 10
RIDGE_GAMMA = 4.15

LAYERS = (
    ("base_mask", "1 · LUT base mask",
     "legacy_inverse_density.py::image_to_mask"),
    ("rejected_small_components", "2 · Rejected small components",
     "legacy_inverse_density.py::compute_preprocessing [area < 100 px]"),
    ("size_filtered_mask", "3 · Size-filtered mask",
     "legacy_inverse_density.py::compute_preprocessing [connectivity 8]"),
    ("first_closed_mask", "4 · First 5×5 close",
     "legacy_inverse_density.py::compute_preprocessing [close pass 1]"),
    ("second_closed_mask", "5 · Second 5×5 close",
     "legacy_inverse_density.py::compute_preprocessing [close pass 2]"),
    ("raw_density", "6 · Raw square-neighborhood density",
     "legacy_inverse_density.py::compute_density_fields [radius 12]"),
    ("normalized_density", "7 · Mean-relative density",
     "legacy_inverse_density.py::compute_density_fields [cap 2.0]"),
    ("base_inverse_density", "8 · Reciprocal-gamma inverse density",
     "legacy_inverse_density.py::compute_density_fields [gamma 1/1.65]"),
    ("ridge_multiplier", "9 · Ridge-centroid multiplier",
     "legacy_inverse_density.py::compute_density_fields [radius 10, gamma 4.15]"),
    ("final_inverse_density", "10 · Calibrated final inverse density",
     "legacy_inverse_density.py::compute_density_fields [base × ridge]"),
)


@dataclass(frozen=True, slots=True)
class PreprocessingStages:
    base_mask: np.ndarray
    rejected_small_pixels: np.ndarray
    size_filtered_mask: np.ndarray
    first_closed_mask: np.ndarray
    second_closed_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class DensityFields:
    raw_density: np.ndarray
    normalized_density: np.ndarray
    base_inverse_density: np.ndarray
    ridge_multiplier: np.ndarray
    final_inverse_density: np.ndarray
    foreground_mean_density: float


def frame_paths(folder):
    return sorted(path for path in folder.iterdir()
                  if path.suffix.lower() in EXTENSIONS)


def load_lut(path=LUT_PATH):
    with np.load(path) as archive:
        lut = archive["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"Unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut


def image_to_mask(image, lut):
    bgr = image.astype(np.uint32)
    key = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return (lut[key] != 0).astype(np.uint8) * 255


def compute_preprocessing(image, lut):
    base = image_to_mask(image, lut)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        base, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < SMALL_COMPONENT_AREA
    small[0] = False
    rejected = small[labels]
    filtered = base.copy()
    filtered[rejected] = 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    first = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    second = cv2.morphologyEx(first, cv2.MORPH_CLOSE, kernel)
    return PreprocessingStages(base, rejected, filtered, first, second)


def integral_image(values):
    return np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def square_bounds(shape, radius):
    height, width = shape
    yy, xx = np.indices(shape)
    y0 = np.maximum(0, yy - radius)
    y1 = np.minimum(height - 1, yy + radius)
    x0 = np.maximum(0, xx - radius)
    x1 = np.minimum(width - 1, xx + radius)
    return yy, xx, y0, y1, x0, x1


def integral_rect(integral, y0, y1, x0, x1):
    return (integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1]
            - integral[y1 + 1, x0] + integral[y0, x0])


def compute_density_fields(
        mask, density_radius_px=DENSITY_RADIUS_PX,
        ridge_radius_px=RIDGE_RADIUS_PX):
    foreground = (mask > 0).astype(np.float32)
    inside = foreground > 0
    _, _, y0, y1, x0, x1 = square_bounds(
        foreground.shape, density_radius_px)
    possible = (y1 - y0 + 1) * (x1 - x0 + 1)
    density = integral_rect(
        integral_image(foreground), y0, y1, x0, x1) / possible
    density[~inside] = 0
    mean_density = float(density[inside].mean()) if np.any(inside) else 0.0
    normalized = np.zeros_like(density, np.float32)
    if mean_density > 0:
        normalized[inside] = np.clip(
            density[inside] / (mean_density * RELATIVE_CAP), 0, 1)
    base_inverse = np.zeros_like(density, np.float32)
    base_inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / INVERSE_GAMMA)

    yy, xx, ry0, ry1, rx0, rx1 = square_bounds(
        foreground.shape, ridge_radius_px)
    mass = integral_rect(
        integral_image(base_inverse), ry0, ry1, rx0, rx1)
    weighted_x = integral_rect(
        integral_image(base_inverse * xx), ry0, ry1, rx0, rx1)
    weighted_y = integral_rect(
        integral_image(base_inverse * yy), ry0, ry1, rx0, rx1)
    centroid_x = np.divide(
        weighted_x, mass, out=np.zeros_like(mass), where=mass > 0)
    centroid_y = np.divide(
        weighted_y, mass, out=np.zeros_like(mass), where=mass > 0)
    delta = np.hypot(xx - centroid_x, yy - centroid_y)
    ridge = np.power(
        1.0 - np.clip(delta / ridge_radius_px, 0, 1), RIDGE_GAMMA)
    ridge[(~inside) | (mass <= 0)] = 0
    final = base_inverse * ridge
    return DensityFields(
        density.astype(np.float32), normalized, base_inverse, ridge, final,
        mean_density)


def mask_layer(mask):
    return cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def heat_layer(values, foreground):
    t = np.clip(values, 0, 1).astype(np.float32)
    trough = np.array([37, 99, 235], np.float32)
    peak = np.array([255, 59, 68], np.float32)
    rgb = np.empty((*t.shape, 3), np.float32)
    rgb[:] = (5, 6, 7)
    rgb[foreground] = trough + (peak - trough) * t[foreground, None]
    return cv2.cvtColor(np.round(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_layers(image, lut):
    stages = compute_preprocessing(image, lut)
    fields = compute_density_fields(stages.second_closed_mask)
    foreground = stages.second_closed_mask > 0
    rejected = np.zeros_like(image)
    rejected[stages.rejected_small_pixels] = (128, 128, 128)
    layers = {
        "base_mask": mask_layer(stages.base_mask),
        "rejected_small_components": rejected,
        "size_filtered_mask": mask_layer(stages.size_filtered_mask),
        "first_closed_mask": mask_layer(stages.first_closed_mask),
        "second_closed_mask": mask_layer(stages.second_closed_mask),
        "raw_density": heat_layer(fields.raw_density, foreground),
        "normalized_density": heat_layer(fields.normalized_density, foreground),
        "base_inverse_density": heat_layer(
            fields.base_inverse_density, foreground),
        "ridge_multiplier": heat_layer(fields.ridge_multiplier, foreground),
        "final_inverse_density": heat_layer(
            fields.final_inverse_density, foreground),
    }
    return layers, stages, fields


def latest_logged_run(runs, logs_root=LOG_RUNS):
    selected = list(runs)
    if not logs_root.is_dir():
        return selected
    candidates = [
        run for run in logs_root.iterdir()
        if run.is_dir() and (run / "vision_frames").is_dir()
        and frame_paths(run / "vision_frames")
    ]
    latest = max(candidates, key=lambda run: run.name, default=None)
    if latest is not None and latest.name not in {run.name for run in selected}:
        selected.append(latest)
    return selected


def selected_runs(runs_root, run_names=None, all_runs=False):
    if not runs_root.is_dir():
        raise ValueError(f"Runs root does not exist: {runs_root}")
    if all_runs:
        runs = sorted(run for run in runs_root.iterdir()
                      if run.is_dir() and (run / "vision_frames").is_dir())
        if not runs:
            raise ValueError(f"No runs with vision_frames found in {runs_root}")
        return runs
    runs = []
    for name in dict.fromkeys(run_names or ()):
        run = runs_root / name
        if not (run / "vision_frames").is_dir():
            raise ValueError(f"Run has no vision_frames directory: {run}")
        runs.append(run)
    return runs


def process_run(run, output_root, lut, start=0, limit=None):
    frames = frame_paths(run / "vision_frames")
    if not frames:
        raise ValueError(f"Run has no supported vision frames: {run}")
    if start >= len(frames):
        raise ValueError(
            f"Start index {start} is outside {run.name} ({len(frames)} frames)")
    selected = frames[start:start + limit if limit is not None else None]
    output = output_root / run.name / "layers"
    for layer_id, _, _ in LAYERS:
        (output / layer_id).mkdir(parents=True)
    for frame in selected:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read {frame}")
        layers, _, _ = render_layers(image, lut)
        for layer_id, layer in layers.items():
            destination = output / layer_id / f"{frame.stem}.png"
            if not cv2.imwrite(str(destination), layer):
                raise ValueError(f"Unable to write {destination}")
    return len(selected)


def static_url(path, static_root=MODELS_ROOT):
    try:
        relative = path.resolve().relative_to(static_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Output must be inside {static_root}: {path}") from exc
    return f"/{quote(relative.as_posix(), safe='/')}"


def write_manifest(run_names, output_root=REVIEWS, manifest=MANIFEST,
                   static_root=MODELS_ROOT):
    runs = []
    for run_name in run_names:
        output = output_root / run_name / "layers"
        files = {
            layer_id: {path.stem: path for path in frame_paths(output / layer_id)}
            for layer_id, _, _ in LAYERS
        }
        frame_ids = sorted(next(iter(files.values())))
        frames = [{
            "id": frame_id,
            "layers": {
                layer_id: (f"{static_url(paths[frame_id], static_root)}"
                           f"?v={paths[frame_id].stat().st_mtime_ns}")
                for layer_id, paths in files.items()
            },
        } for frame_id in frame_ids]
        runs.append({"id": run_name, "name": run_name, "frames": frames})
    payload = {
        "profile": {
            "density_radius_px": DENSITY_RADIUS_PX,
            "normalization": "mean-relative",
            "relative_cap": RELATIVE_CAP,
            "inverse_gamma": INVERSE_GAMMA,
            "inverse_exponent": 1.0 / INVERSE_GAMMA,
            "ridge_enabled": True,
            "ridge_radius_px": RIDGE_RADIUS_PX,
            "ridge_gamma": RIDGE_GAMMA,
            "layer_confidence": False,
            "point_centroid": False,
            "resolution": "native full frame",
        },
        "layers": [{"id": item[0], "label": item[1], "reference": item[2]}
                   for item in LAYERS],
        "runs": runs,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(f"{manifest.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest)


def generate_review(runs, output_root, manifest, lut, start=0, limit=None,
                    static_root=MODELS_ROOT):
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}-", dir=output_root.parent))
    counts = {}
    try:
        for run in runs:
            counts[run.name] = process_run(run, staging, lut, start, limit)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
        write_manifest(
            [run.name for run in runs], output_root, manifest, static_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return counts


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", action="append", dest="run_names")
    selection.add_argument("--all", action="store_true", dest="all_runs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--output-root", type=Path, default=REVIEWS)
    parser.add_argument("--lut", type=Path, default=LUT_PATH)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start < 0:
        parser.error("--start must be zero or greater")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    try:
        runs = selected_runs(args.runs_root, args.run_names, args.all_runs)
        runs = latest_logged_run(runs)
        counts = generate_review(
            runs, args.output_root, MANIFEST, load_lut(args.lut),
            args.start, args.limit)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Rendered {sum(counts.values())} frames from {len(counts)} runs "
          f"into {args.output_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
