import colorsys
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reports.void_geometry import render_geometry
from instance_tracking import InstanceTracker, publish_tracked
from schema import EllipseEstimate, VoidDetectionGeometry, geometry_source
RUNS = ROOT / "runs"
REVIEWS = ROOT / "review_runs"
GEOMETRY = ROOT / "geometry-analysis/data"
MANIFEST = ROOT / "frame-viewer/data/runs-manifest.json"
LUT_PATH = ROOT / "assets/color_lut_v1.npz"
EXTENSIONS = {".png", ".jpg", ".jpeg"}
def frame_paths(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
def load_lut():
    lut = np.load(LUT_PATH)["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"Unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut
def image_to_mask(image, lut):
    bgr = image.astype(np.uint32)
    key = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return (lut[key] != 0).astype(np.uint8) * 255
def next_ellipse_color(rng, used):
    while True:
        rgb = colorsys.hsv_to_rgb(rng.random(), 0.9, 1.0)
        candidate = tuple(round(value * 255) for value in reversed(rgb))
        if candidate not in used:
            used.add(candidate)
            return candidate
def fit_contour(component):
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) >= 5:
        center, size, angle = cv2.fitEllipse(contour)
        center = tuple(round(value) for value in center)
        axes = tuple(max(1, round(value / 2)) for value in size)
    else:
        x, y, width, height = cv2.boundingRect(contour)
        center = (x + width // 2, y + height // 2)
        axes, angle = (max(1, width // 2), max(1, height // 2)), 0
    return center, axes, angle
def geometry_record(ellipse, color, features):
    return VoidDetectionGeometry(
        geometry_source("void_detection", ellipse), ellipse,
        tuple(map(int, color)), features.outer_corners, features.inner_corners,
        features.outer_corner_circles, features.inner_corner_circles,
        features.connection_circles)
GEOMETRY_FACTORIES = {"void_geometry.json": geometry_record}
def draw_ellipses(image, contours, hierarchy, context):
    if hierarchy is None:
        return [], {name: [] for name in GEOMETRY_FACTORIES}
    rng = random.Random(f"ellipses:{context['frame_id']}")
    used = {(0, 0, 0), (255, 255, 255), (128, 128, 128),
            (0, 255, 0), (255, 0, 255)}
    blank = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blank[:] = 0
    ellipses = []
    for index, node in enumerate(hierarchy[0]):
        depth = 0
        parent = int(node[3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        if depth % 2 == 0:
            continue
        component = blank.copy()
        cv2.drawContours(component, contours, index, 255, cv2.FILLED)
        area = cv2.countNonZero(component)
        fitted = fit_contour(component) if area >= 75 else None
        if fitted:
            color = next_ellipse_color(rng, used)
            ellipses.append((area, *fitted, color, int(node[3])))
    estimates, points = [], []
    ordered = sorted(ellipses, key=lambda item: item[0], reverse=True)
    for area, center, axes, angle, color, parent in ordered:
        cv2.ellipse(image, center, axes, angle, 0, 360, color, cv2.FILLED)
        points.append((parent, center, axes, color))
        estimates.append(EllipseEstimate(
            ellipse_id=context["id_start"] + len(estimates),
            run_id=context["run_id"], frame_id=context["frame_id"],
            frame_index=context["frame_index"], frame_count=context["frame_count"],
            frame_size_px=(image.shape[1], image.shape[0]),
            center_px=tuple(map(float, center)),
            semi_axes_px=tuple(map(float, axes)),
            angle_degrees=float(angle), source_area_px=area,
        ))
    features = render_geometry(image, contours, points)
    records = {
        name: [factory(estimate, item[4], feature)
               for estimate, item, feature in zip(estimates, ordered, features)]
        for name, factory in GEOMETRY_FACTORIES.items()
    }
    return estimates, records
def render_frame(source, destination, context, lut):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read {source}")
    binary = image_to_mask(image, lut)
    mask = binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 100
    small[0] = False
    small_pixels = small[labels]
    morph_input = binary.copy()
    morph_input[small_pixels] = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(morph_input, cv2.MORPH_CLOSE, close_kernel)
    added = cv2.subtract(closed, morph_input)
    contour_mask = closed.copy()
    contour_mask[small_pixels] = 0
    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output[added > 0] = (0, 255, 0)
    output[small_pixels] = (128, 128, 128)
    tree, hierarchy = cv2.findContours(
        contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tree = [cv2.approxPolyDP(contour, 1, True) for contour in tree]
    cv2.drawContours(output, tree, -1, (255, 0, 255), 1)
    output[small_pixels] = (128, 128, 128)
    estimates, geometry = draw_ellipses(output, tree, hierarchy, context)
    if not cv2.imwrite(str(destination), output):
        raise ValueError(f"Unable to write {destination}")
    return estimates, geometry

def process_run(run, lut):
    source = run / "vision_frames"
    output = REVIEWS / run.name / "mask_frames"
    geometry_dir = GEOMETRY / run.name
    shutil.rmtree(output.parent, ignore_errors=True)
    shutil.rmtree(geometry_dir, ignore_errors=True)
    output.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    frames, estimates, tracked = frame_paths(source), [], []
    tracker = InstanceTracker()
    geometry = {name: [] for name in GEOMETRY_FACTORIES}
    for index, frame in enumerate(frames):
        context = {"id_start": len(estimates), "run_id": run.name,
                   "frame_id": frame.stem, "frame_index": index,
                   "frame_count": len(frames)}
        found, records = render_frame(frame, output / frame.name, context, lut)
        estimates.extend(found)
        tracked.extend(tracker.update_geometry(records["void_geometry.json"]))
        for name, items in records.items():
            geometry[name].extend(items)
    metadata = output.parent / "void_estimates.json"
    metadata.write_text(json.dumps([asdict(item) for item in estimates], indent=2)
                        + "\n")
    for name, items in geometry.items():
        path = geometry_dir / name
        path.write_text(
            json.dumps([asdict(item) for item in items], indent=2) + "\n")
    publish_tracked(tracked, output.parent)
    return output

def write_manifest(outputs):
    runs = []
    for output in outputs:
        frames = [f"/review_runs/{output.parent.name}/mask_frames/{frame.name}"
                  f"?v={frame.stat().st_mtime_ns}"
                  for frame in frame_paths(output)]
        metadata = output.parent / "void_estimates.json"
        geometry = {
            name: f"/geometry-analysis/data/{output.parent.name}/{name}"
                  f"?v={(GEOMETRY / output.parent.name / name).stat().st_mtime_ns}"
            for name in GEOMETRY_FACTORIES
        }
        runs.append({"id": output.parent.name, "name": output.parent.name,
                     "frames": frames,
                     "void_estimates": f"/review_runs/{output.parent.name}/"
                     f"void_estimates.json?v={metadata.stat().st_mtime_ns}",
                     "tracked_detections": f"/review_runs/{output.parent.name}/"
                     f"tracked_void_detections.json?v={(output.parent / 'tracked_void_detections.json').stat().st_mtime_ns}",
                     "geometry": geometry})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"runs": runs}, indent=2) + "\n")

def main():
    runs = sorted(run for run in RUNS.iterdir()
                  if run.is_dir() and (run / "vision_frames").is_dir())
    lut = load_lut()
    outputs = [process_run(run, lut) for run in runs]
    write_manifest(outputs)
    total = sum(len(frame_paths(out)) for out in outputs)
    print(f"Processed {len(runs)} runs and {total} frames.")
if __name__ == "__main__":
    main()
