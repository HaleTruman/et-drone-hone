#!/usr/bin/env python3
"""Build tight pixel-only bboxes from enabled color-mask layers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from aigp_vision.models.deterministic.tools.build_vision_memory import (
    APP_DIR,
    DEFAULT_CONFIG,
    DEFAULT_REVIEW,
    DEFAULT_RULES,
    REPO_ROOT,
    apply_layer_metadata,
    atomic_write_json,
    build_class_membership,
    layer_enabled_map,
    layer_metadata,
    read_json,
    read_frame_color_ids,
    repo_path_from_manifest,
    selected_precompute_manifest,
    stable_hash,
)


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "mask_bboxes"
DEFAULT_MIN_PIXELS = 50
DEFAULT_MAX_BBOXES_PER_FRAME = 1000
DEFAULT_FIT_TIGHTNESS = 100
DEFAULT_QUAD_FIT_ENABLED = True
DEFAULT_QUAD_FIT_MODE = "rotated-rect"
DEFAULT_TARGET_ASPECT = 1.0
DEFAULT_ASPECT_TOLERANCE = 0.6
DEFAULT_QUAD_THICKNESS_PX = 8
DEFAULT_EDGE_COVERAGE_MIN = 0.5
DEFAULT_CORNER_MIN_PIXELS = 2
DEFAULT_VOID_OVERLAP_MAX_RATIO = 0.75
DEFAULT_VOID_OVERLAP_MIN_PIXELS = 25
DEFAULT_FOV_CLIP_ENABLED = False
DEFAULT_FOV_CLIP_MARGIN_PX = 6
DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS = 12
DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO = 0.01
DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH = True
DEFAULT_FOV_CLIP_WARN_ONLY = True
QUAD_FIT_MODES = {"axis-bbox", "rotated-rect", "free-quad"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def repo_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/" + resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def clean_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or "").strip())[:160]


def bbox_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build bbox artifacts without an active run key")
    return output_root / clean


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bool_setting(value, default: bool) -> bool:
    if value in {None, ""}:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def numeric_setting(raw: dict, key: str, default: float, low: float, high: float, integer: bool = False):
    value = raw.get(key, default)
    if value in {None, ""}:
        value = default
    value = clamp(float(value), low, high)
    return int(round(value)) if integer else value


def normalized_fov_clip_settings(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool_setting(source.get("enabled"), DEFAULT_FOV_CLIP_ENABLED),
        "marginPx": numeric_setting(source, "marginPx", DEFAULT_FOV_CLIP_MARGIN_PX, 0, 64, integer=True),
        "minContactPixels": numeric_setting(
            source,
            "minContactPixels",
            DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS,
            1,
            5000,
            integer=True,
        ),
        "minContactRatio": numeric_setting(
            source,
            "minContactRatio",
            DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO,
            0.0,
            1.0,
        ),
        "requireBboxTouch": bool_setting(source.get("requireBboxTouch"), DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH),
        "warnOnly": bool_setting(source.get("warnOnly"), DEFAULT_FOV_CLIP_WARN_ONLY),
    }


def normalized_bbox_settings(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    raw_min_pixels = source.get("minPixels", DEFAULT_MIN_PIXELS)
    raw_fit_tightness = source.get("fitTightness", DEFAULT_FIT_TIGHTNESS)
    if raw_min_pixels in {None, ""}:
        raw_min_pixels = DEFAULT_MIN_PIXELS
    if raw_fit_tightness in {None, ""}:
        raw_fit_tightness = DEFAULT_FIT_TIGHTNESS
    min_pixels = int(round(float(raw_min_pixels)))
    fit_tightness = int(round(float(raw_fit_tightness)))
    min_pixels = max(1, min(50000, min_pixels))
    fit_tightness = int(clamp(fit_tightness, 0, 100))
    merge_radius_px = int(round((100 - fit_tightness) / 10))
    quad_fit_mode = str(source.get("quadFitMode") or DEFAULT_QUAD_FIT_MODE)
    if quad_fit_mode not in QUAD_FIT_MODES:
        quad_fit_mode = DEFAULT_QUAD_FIT_MODE
    return {
        "minPixels": min_pixels,
        "maxBboxesPerFrame": numeric_setting(source, "maxBboxesPerFrame", DEFAULT_MAX_BBOXES_PER_FRAME, 1, 5000, integer=True),
        "fitTightness": fit_tightness,
        "mergeRadiusPx": merge_radius_px,
        "connectivity": 8,
        "quadFitEnabled": bool_setting(source.get("quadFitEnabled"), DEFAULT_QUAD_FIT_ENABLED),
        "quadFitMode": quad_fit_mode,
        "targetAspect": numeric_setting(source, "targetAspect", DEFAULT_TARGET_ASPECT, 0.1, 10.0),
        "aspectTolerance": numeric_setting(source, "aspectTolerance", DEFAULT_ASPECT_TOLERANCE, 0.0, 2.0),
        "quadThicknessPx": numeric_setting(source, "quadThicknessPx", DEFAULT_QUAD_THICKNESS_PX, 1, 80, integer=True),
        "edgeCoverageMin": numeric_setting(source, "edgeCoverageMin", DEFAULT_EDGE_COVERAGE_MIN, 0.0, 1.0),
        "cornerMinPixels": numeric_setting(source, "cornerMinPixels", DEFAULT_CORNER_MIN_PIXELS, 0, 10000, integer=True),
        "voidOverlapMaxRatio": numeric_setting(
            source,
            "voidOverlapMaxRatio",
            DEFAULT_VOID_OVERLAP_MAX_RATIO,
            0.0,
            1.0,
        ),
        "voidOverlapMinPixels": numeric_setting(
            source,
            "voidOverlapMinPixels",
            DEFAULT_VOID_OVERLAP_MIN_PIXELS,
            1,
            50000,
            integer=True,
        ),
        "fovClip": normalized_fov_clip_settings(source.get("fovClip") if isinstance(source.get("fovClip"), dict) else None),
    }


def bbox_settings_from_review(review: dict) -> dict:
    raw = review.get("bboxFlow") if isinstance(review.get("bboxFlow"), dict) else {}
    return normalized_bbox_settings(raw)


def bbox_signature_settings(settings: dict) -> dict:
    clean = dict(normalized_bbox_settings(settings))
    fov_clip = clean.get("fovClip") if isinstance(clean.get("fovClip"), dict) else {}
    if not fov_clip.get("enabled"):
        clean.pop("fovClip", None)
    return clean


def concise_layer_summaries(summaries: list[dict]) -> list[dict]:
    return [
        {
            "prefix": item.get("prefix"),
            "displayName": item.get("displayName"),
            "enabled": bool(item.get("enabled")),
            "colorIdCount": item.get("colorIdCount"),
            "sourceColorIdCount": item.get("sourceColorIdCount"),
            "ruleSource": item.get("ruleSource"),
        }
        for item in summaries
    ]


def build_mask_context(config: dict, rules: dict, review: dict, manifest: dict) -> dict:
    base_classes = list(config.get("classes") or [])
    if not base_classes:
        raise ValueError("alpha class config has no classes")
    metadata = layer_metadata(review)
    classes = apply_layer_metadata(base_classes, metadata)
    enabled_by_prefix = layer_enabled_map(review)
    membership, summaries = build_class_membership(classes, rules, manifest, enabled_by_prefix)
    signature_layers = []
    for item in summaries:
        ranges = item.get("colorIdRanges") if item.get("enabled") else []
        signature_layers.append(
            {
                "prefix": item.get("prefix"),
                "enabled": bool(item.get("enabled")),
                "colorIdCount": item.get("colorIdCount"),
                "rangeHash": stable_hash({"ranges": ranges}),
            }
        )
    return {
        "classes": classes,
        "membership": membership,
        "summaries": summaries,
        "signature": stable_hash(
            {
                "kind": "0720-enabled-color-mask",
                "runKey": manifest.get("runKey"),
                "colorTableCount": manifest.get("colorTable", {}).get("count"),
                "layers": signature_layers,
            }
        ),
        "enabledPrefixes": [
            str(item.get("prefix"))
            for item in summaries
            if item.get("enabled") and int(item.get("colorIdCount") or 0) > 0
        ],
    }


def bbox_source_signature(manifest: dict, manifest_path: Path, settings: dict, mask_signature: str) -> str:
    return stable_hash(
        {
            "kind": "0720-color-mask-pixel-bboxes",
            "runKey": manifest.get("runKey"),
            "precomputeManifest": repo_url(manifest_path),
            "frameCount": int(manifest.get("frameCount") or len(manifest.get("frames") or [])),
            "width": int(manifest.get("width") or 640),
            "height": int(manifest.get("height") or 360),
            "settings": bbox_signature_settings(settings),
            "maskSignature": mask_signature,
        }
    )


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def points_to_json(points: np.ndarray) -> list[list[float]]:
    return [[clean_float(point[0]), clean_float(point[1])] for point in np.asarray(points, dtype=np.float32)]


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("quad fit requires four points")
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.array(
        [
            pts[int(np.argmin(sums))],
            pts[int(np.argmax(diffs))],
            pts[int(np.argmax(sums))],
            pts[int(np.argmin(diffs))],
        ],
        dtype=np.float32,
    )
    if np.unique(np.round(ordered, 3), axis=0).shape[0] == 4 and abs(polygon_area(ordered)) > 0.1:
        return ordered

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0).astype(np.float32)
    if ordered[1][1] > ordered[-1][1]:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    x_values = pts[:, 0]
    y_values = pts[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def quad_side_lengths(points: np.ndarray) -> list[float]:
    pts = np.asarray(points, dtype=np.float32)
    return [float(np.linalg.norm(pts[(index + 1) % 4] - pts[index])) for index in range(4)]


def quad_dimensions(points: np.ndarray) -> tuple[float, float, float]:
    lengths = quad_side_lengths(points)
    width = (lengths[0] + lengths[2]) / 2.0
    height = (lengths[1] + lengths[3]) / 2.0
    aspect = width / max(height, 1e-6)
    return width, height, aspect


def aspect_in_range(aspect: float, target: float, tolerance: float) -> bool:
    low = max(0.01, target * max(0.0, 1.0 - tolerance))
    high = target * (1.0 + tolerance)
    return low <= aspect <= high


def layer_counts_for_mask(layer_hits: np.ndarray, classes: list[dict], pixel_mask: np.ndarray, bounds=None) -> list[dict]:
    if bounds:
        x0, y0, x1, y1 = bounds
        hits = layer_hits[:, y0:y1, x0:x1]
    else:
        hits = layer_hits
    layer_counts = []
    for class_index, item in enumerate(classes):
        count = int(np.count_nonzero(hits[class_index] & pixel_mask))
        if count:
            layer_counts.append(
                {
                    "prefix": str(item.get("prefix")),
                    "displayName": item.get("displayName"),
                    "pixelCount": count,
                }
        )
    return layer_counts


def fov_clip_for_component(
    xs: np.ndarray,
    ys: np.ndarray,
    bbox_px: list[int],
    pixel_count: int,
    width: int,
    height: int,
    settings: dict,
) -> dict | None:
    fov_settings = settings.get("fovClip") if isinstance(settings.get("fovClip"), dict) else {}
    if not fov_settings.get("enabled"):
        return None
    margin = int(fov_settings["marginPx"])
    min_pixels = int(fov_settings["minContactPixels"])
    min_ratio = float(fov_settings["minContactRatio"])
    require_touch = bool(fov_settings["requireBboxTouch"])
    x0, y0, x1, y1 = [int(value) for value in bbox_px]
    bbox_touches = {
        "left": x0 <= margin,
        "right": x1 >= width - 1 - margin,
        "top": y0 <= margin,
        "bottom": y1 >= height - 1 - margin,
    }
    contact_masks = {
        "left": xs <= margin,
        "right": xs >= width - 1 - margin,
        "top": ys <= margin,
        "bottom": ys >= height - 1 - margin,
    }
    contact_pixels = {side: int(np.count_nonzero(mask)) for side, mask in contact_masks.items()}
    contact_ratio = {
        side: float(contact_pixels[side]) / float(max(1, pixel_count))
        for side in contact_pixels
    }
    side_severity = {}
    side_trust = {}
    clipped_sides = []
    near_sides = []
    for side in ("left", "right", "top", "bottom"):
        pixel_score = clamp(float(contact_pixels[side]) / float(max(1, min_pixels)), 0.0, 1.0)
        ratio_score = 1.0 if min_ratio <= 0 else clamp(float(contact_ratio[side]) / min_ratio, 0.0, 1.0)
        touch_score = 1.0 if bbox_touches[side] else 0.0
        active = bbox_touches[side] or contact_pixels[side] > 0
        severity = (0.45 * touch_score + 0.35 * pixel_score + 0.20 * ratio_score) if active else 0.0
        touch_gate = bbox_touches[side] or not require_touch
        is_clipped = touch_gate and contact_pixels[side] >= min_pixels and contact_ratio[side] >= min_ratio
        if is_clipped:
            clipped_sides.append(side)
        elif active:
            near_sides.append(side)
        side_severity[side] = clean_float(severity, 4)
        side_trust[side] = clean_float(1.0 - severity, 4)
    status = "clipped" if clipped_sides else "near-edge" if near_sides else "clear"
    severity = max(side_severity.values()) if side_severity else 0.0
    return {
        "enabled": True,
        "status": status,
        "severity": clean_float(severity, 4),
        "sides": clipped_sides,
        "nearSides": near_sides,
        "marginPx": margin,
        "bboxTouches": bbox_touches,
        "contactPixels": contact_pixels,
        "contactRatio": {side: clean_float(value, 6) for side, value in contact_ratio.items()},
        "sideSeverity": side_severity,
        "sideTrust": side_trust,
        "requireBboxTouch": require_touch,
        "warnOnly": bool(fov_settings["warnOnly"]),
    }


def empty_fov_clip_summary(enabled: bool) -> dict:
    return {
        "enabled": bool(enabled),
        "clippedBboxCount": 0,
        "nearEdgeBboxCount": 0,
        "sideCounts": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "maxSeverity": 0.0,
        "meanSeverity": 0.0,
    }


def collect_fov_clip_summary(summary: dict, bboxes: list[dict]) -> list[float]:
    severities = []
    for bbox in bboxes:
        fov_clip = bbox.get("fovClip") if isinstance(bbox.get("fovClip"), dict) else None
        if not fov_clip:
            continue
        severity = float(fov_clip.get("severity") or 0.0)
        severities.append(severity)
        if fov_clip.get("status") == "clipped":
            summary["clippedBboxCount"] += 1
        elif fov_clip.get("status") == "near-edge":
            summary["nearEdgeBboxCount"] += 1
        for side in fov_clip.get("sides") or []:
            if side in summary["sideCounts"]:
                summary["sideCounts"][side] += 1
    return severities


def axis_bbox_points(bbox_px: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in bbox_px]
    return order_quad_points(np.array([[x0, y0], [x1 + 1.0, y0], [x1 + 1.0, y1 + 1.0], [x0, y1 + 1.0]], dtype=np.float32))


def rotated_rect_points(cv2, component_pixels: np.ndarray, bbox_px: list[int]) -> tuple[np.ndarray, str | None]:
    ys, xs = np.nonzero(component_pixels)
    if xs.size < 4:
        return axis_bbox_points(bbox_px), "rotated-rect-fallback-axis"
    points = np.column_stack([xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5]).reshape(-1, 1, 2)
    box = cv2.boxPoints(cv2.minAreaRect(points))
    if polygon_area(box) <= 0.1:
        return axis_bbox_points(bbox_px), "rotated-rect-fallback-axis"
    return order_quad_points(box), None


def free_quad_points(cv2, component_pixels: np.ndarray, bbox_px: list[int]) -> tuple[np.ndarray, str | None]:
    source = component_pixels.astype(np.uint8) * 255
    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        points, _ = rotated_rect_points(cv2, component_pixels, bbox_px)
        return points, "free-quad-fallback-rotated-rect"
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 0:
        points, _ = rotated_rect_points(cv2, component_pixels, bbox_px)
        return points, "free-quad-fallback-rotated-rect"
    for epsilon_ratio in np.linspace(0.01, 0.12, 12):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2).astype(np.float32) + 0.5
            if polygon_area(points) > 0.1:
                return order_quad_points(points), None
    points, _ = rotated_rect_points(cv2, component_pixels, bbox_px)
    return points, "free-quad-fallback-rotated-rect"


def fit_candidate_points(cv2, component_pixels: np.ndarray, bbox_px: list[int], settings: dict) -> tuple[np.ndarray, str | None]:
    mode = str(settings.get("quadFitMode") or DEFAULT_QUAD_FIT_MODE)
    if mode == "axis-bbox":
        return axis_bbox_points(bbox_px), None
    if mode == "free-quad":
        return free_quad_points(cv2, component_pixels, bbox_px)
    return rotated_rect_points(cv2, component_pixels, bbox_px)


def roi_bounds_for_points(points: np.ndarray, width: int, height: int, extra: int) -> tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=np.float32)
    x0 = max(0, int(math.floor(float(pts[:, 0].min()) - extra)))
    y0 = max(0, int(math.floor(float(pts[:, 1].min()) - extra)))
    x1 = min(width, int(math.ceil(float(pts[:, 0].max()) + extra)))
    y1 = min(height, int(math.ceil(float(pts[:, 1].max()) + extra)))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def filled_polygon_mask(cv2, shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def edge_band_polygon(p0: np.ndarray, p1: np.ndarray, thickness: int) -> np.ndarray:
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        radius = max(1.0, float(thickness) / 2.0)
        return np.array(
            [[p0[0] - radius, p0[1] - radius], [p0[0] + radius, p0[1] - radius], [p0[0] + radius, p0[1] + radius], [p0[0] - radius, p0[1] + radius]],
            dtype=np.float32,
        )
    normal = np.array([-vector[1] / length, vector[0] / length], dtype=np.float32) * (float(thickness) / 2.0)
    return np.array([p0 + normal, p1 + normal, p1 - normal, p0 - normal], dtype=np.float32)


def edge_support_for_band(
    cv2,
    component_roi: np.ndarray,
    shape: tuple[int, int],
    p0: np.ndarray,
    p1: np.ndarray,
    thickness: int,
) -> tuple[dict, np.ndarray]:
    band = filled_polygon_mask(cv2, shape, edge_band_polygon(p0, p1, thickness))
    supported = component_roi & band
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return {"coverage": 0.0, "supportPixels": int(np.count_nonzero(supported)), "lengthPx": 0.0}, band
    ys, xs = np.nonzero(supported)
    if xs.size == 0:
        return {"coverage": 0.0, "supportPixels": 0, "lengthPx": clean_float(length)}, band
    coords = np.column_stack([xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5])
    projection = ((coords - p0) @ vector) / max(length * length, 1e-6)
    projection = np.clip(projection, 0.0, 1.0)
    bin_count = max(1, int(math.ceil(length)))
    bins = np.unique(np.clip(np.floor(projection * bin_count).astype(np.int32), 0, bin_count - 1))
    return {
        "coverage": clean_float(float(len(bins)) / float(bin_count)),
        "supportPixels": int(xs.size),
        "lengthPx": clean_float(length),
    }, band


def corner_support_masks(cv2, shape: tuple[int, int], points: np.ndarray, thickness: int) -> list[np.ndarray]:
    radius = max(1, int(math.ceil(float(thickness) * 1.25)))
    masks = []
    for point in points:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (int(round(float(point[0]))), int(round(float(point[1])))), radius, 1, -1)
        masks.append(mask.astype(bool))
    return masks


def connected_ring_support(cv2, component_roi: np.ndarray, edge_masks: list[np.ndarray], corner_masks: list[np.ndarray]) -> dict:
    if not edge_masks:
        return {"edgeTouches": [False] * 4, "cornerTouches": [False] * 4, "pixelCount": 0}
    edge_union = np.zeros(component_roi.shape, dtype=bool)
    for mask in edge_masks:
        edge_union |= mask
    ring = (component_roi & edge_union).astype(np.uint8)
    label_count, labels, _, _ = cv2.connectedComponentsWithStats(ring, 8)
    best = {"edgeTouches": [False] * 4, "cornerTouches": [False] * 4, "pixelCount": 0}
    best_rank = (-1, -1)
    for label in range(1, int(label_count)):
        label_mask = labels == label
        pixel_count = int(np.count_nonzero(label_mask))
        edge_touches = [bool(np.any(label_mask & mask)) for mask in edge_masks]
        corner_touches = [bool(np.any(label_mask & mask)) for mask in corner_masks]
        rank = (sum(edge_touches) + sum(corner_touches), pixel_count)
        if rank > best_rank:
            best_rank = rank
            best = {"edgeTouches": edge_touches, "cornerTouches": corner_touches, "pixelCount": pixel_count}
    return best


def overlap_children_for_void(
    cv2,
    void_pixels_roi: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    bounds: tuple[int, int, int, int],
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> list[dict]:
    label_count, labels, _, _ = cv2.connectedComponentsWithStats(void_pixels_roi.astype(np.uint8), 8)
    x_origin, y_origin, _, _ = bounds
    children = []
    for label in range(1, int(label_count)):
        child_mask = labels == label
        pixel_count = int(np.count_nonzero(child_mask))
        if pixel_count < int(settings["voidOverlapMinPixels"]):
            continue
        ys, xs = np.nonzero(child_mask)
        if xs.size == 0:
            continue
        full_xs = xs + x_origin
        full_ys = ys + y_origin
        x0 = int(full_xs.min())
        x1 = int(full_xs.max())
        y0 = int(full_ys.min())
        y1 = int(full_ys.max())
        centroid_x = float(full_xs.mean())
        centroid_y = float(full_ys.mean())
        child = {
            "overlapId": "",
            "parentBboxId": "",
            "frameOrdinal": frame_ordinal,
            "pixelCount": pixel_count,
            "bboxPx": [x0, y0, x1, y1],
            "bboxUv": [x0 / width, y0 / height, (x1 + 1) / width, (y1 + 1) / height],
            "centroidPx": [clean_float(centroid_x), clean_float(centroid_y)],
            "centroidUv": [clean_float(centroid_x / width, 6), clean_float(centroid_y / height, 6)],
            "layerCounts": layer_counts_for_mask(layer_hits, classes, child_mask, bounds),
        }
        if xs.size >= 4:
            points = np.column_stack([full_xs.astype(np.float32) + 0.5, full_ys.astype(np.float32) + 0.5]).reshape(-1, 1, 2)
            try:
                child["quadPointsPx"] = points_to_json(order_quad_points(cv2.boxPoints(cv2.minAreaRect(points))))
            except Exception:
                pass
        children.append(child)
    children.sort(key=lambda item: (-int(item["pixelCount"]), item["bboxPx"][1], item["bboxPx"][0]))
    return children


def quad_fit_for_component(
    cv2,
    component_pixels: np.ndarray,
    selected: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    bbox_px: list[int],
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> tuple[dict, list[dict]]:
    if not settings.get("quadFitEnabled"):
        return {"enabled": False, "accepted": False, "status": "disabled", "reasonCodes": ["quad-fit-disabled"]}, []

    reason_codes = []
    points, fallback_reason = fit_candidate_points(cv2, component_pixels, bbox_px, settings)
    if fallback_reason:
        reason_codes.append(fallback_reason)
    points = order_quad_points(points)
    if polygon_area(points) <= 0.1:
        return {"enabled": True, "accepted": False, "status": "rejected", "reasonCodes": ["degenerate-quad"]}, []

    thickness = int(settings["quadThicknessPx"])
    bounds = roi_bounds_for_points(points, width, height, extra=max(2, thickness * 2))
    x0, y0, x1, y1 = bounds
    shape = (y1 - y0, x1 - x0)
    relative_points = points - np.array([x0, y0], dtype=np.float32)
    component_roi = component_pixels[y0:y1, x0:x1]
    selected_roi = selected[y0:y1, x0:x1]

    outer_mask = filled_polygon_mask(cv2, shape, relative_points)
    edge_metrics = []
    edge_masks = []
    for edge_index in range(4):
        metric, band = edge_support_for_band(
            cv2,
            component_roi,
            shape,
            relative_points[edge_index],
            relative_points[(edge_index + 1) % 4],
            thickness,
        )
        metric["edge"] = ["top", "right", "bottom", "left"][edge_index]
        edge_metrics.append(metric)
        edge_masks.append(band)
    edge_union = np.zeros(shape, dtype=bool)
    for mask in edge_masks:
        edge_union |= mask

    corner_masks = corner_support_masks(cv2, shape, relative_points, thickness)
    corner_counts = [int(np.count_nonzero(component_roi & mask)) for mask in corner_masks]
    connected = connected_ring_support(cv2, component_roi, edge_masks, corner_masks)
    void_mask = outer_mask & ~edge_union
    void_pixels = selected_roi & void_mask
    void_area = int(np.count_nonzero(void_mask))
    void_count = int(np.count_nonzero(void_pixels))
    void_ratio = float(void_count) / float(max(1, void_area))
    has_void_overlap = (
        void_count >= int(settings["voidOverlapMinPixels"])
        and void_ratio >= float(settings["voidOverlapMaxRatio"])
    )
    overlap_children = overlap_children_for_void(cv2, void_pixels, layer_hits, classes, bounds, frame_ordinal, width, height, settings) if has_void_overlap else []

    _, _, aspect = quad_dimensions(points)
    aspect_ok = aspect_in_range(float(aspect), float(settings["targetAspect"]), float(settings["aspectTolerance"]))
    coverage_ok = all(float(item["coverage"]) >= float(settings["edgeCoverageMin"]) for item in edge_metrics)
    corners_ok = all(count >= int(settings["cornerMinPixels"]) for count in corner_counts)
    connected_ok = all(connected["edgeTouches"]) and all(connected["cornerTouches"])
    if not aspect_ok:
        reason_codes.append("aspect-out-of-range")
    if not coverage_ok:
        reason_codes.append("edge-coverage-low")
    if not corners_ok:
        reason_codes.append("corner-support-low")
    if not connected_ok:
        reason_codes.append("ring-disconnected")
    if has_void_overlap:
        reason_codes.append("void-overlap")

    accepted = aspect_ok and coverage_ok and corners_ok and connected_ok
    edge_mean = float(np.mean([float(item["coverage"]) for item in edge_metrics])) if edge_metrics else 0.0
    aspect_score = 1.0 if aspect_ok else max(0.0, 1.0 - abs(float(aspect) - float(settings["targetAspect"])) / max(0.01, float(settings["targetAspect"])))
    corner_score = sum(count >= int(settings["cornerMinPixels"]) for count in corner_counts) / 4.0
    connected_score = (sum(connected["edgeTouches"]) + sum(connected["cornerTouches"])) / 8.0
    score = (edge_mean * 0.45) + (aspect_score * 0.2) + (corner_score * 0.2) + (connected_score * 0.15)

    quad_fit = {
        "enabled": True,
        "accepted": bool(accepted),
        "status": "accepted" if accepted else "rejected",
        "mode": settings["quadFitMode"],
        "reasonCodes": reason_codes,
        "pointsPx": points_to_json(points),
        "pointsUv": [[clean_float(point[0] / width, 6), clean_float(point[1] / height, 6)] for point in points],
        "areaPx": clean_float(polygon_area(points)),
        "aspectRatio": clean_float(float(aspect)),
        "edgeCoverage": edge_metrics,
        "cornerPixelCounts": corner_counts,
        "connectedEdges": connected["edgeTouches"],
        "connectedCorners": connected["cornerTouches"],
        "connectedRingPixels": int(connected["pixelCount"]),
        "voidAreaPx": void_area,
        "voidMaskedPixelCount": void_count,
        "voidFillRatio": clean_float(void_ratio, 6),
        "hasVoidOverlap": bool(has_void_overlap),
        "overlapChildCount": len(overlap_children),
        "score": clean_float(score, 5),
    }
    return quad_fit, overlap_children


def component_bboxes(
    selected: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> list[dict]:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for bbox connected components: {error}") from error

    source_mask = selected.astype(np.uint8)
    working_mask = source_mask
    radius = int(settings["mergeRadiusPx"])
    if radius > 0:
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        working_mask = cv2.morphologyEx(source_mask, cv2.MORPH_CLOSE, kernel)

    label_count, labels, _, _ = cv2.connectedComponentsWithStats(working_mask, int(settings["connectivity"]))
    bboxes: list[dict] = []
    for label in range(1, int(label_count)):
        component_pixels = selected & (labels == label)
        pixel_count = int(np.count_nonzero(component_pixels))
        if pixel_count < int(settings["minPixels"]):
            continue
        ys, xs = np.nonzero(component_pixels)
        if xs.size == 0:
            continue
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        layer_counts = layer_counts_for_mask(layer_hits, classes, component_pixels)
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        bbox_px = [x0, y0, x1, y1]
        fov_clip = fov_clip_for_component(xs, ys, bbox_px, pixel_count, width, height, settings)
        quad_fit, overlap_children = quad_fit_for_component(
            cv2,
            component_pixels,
            selected,
            layer_hits,
            classes,
            bbox_px,
            frame_ordinal,
            width,
            height,
            settings,
        )
        bboxes.append(
            {
                "bboxId": "",
                "frameOrdinal": frame_ordinal,
                "pixelCount": pixel_count,
                "bboxPx": bbox_px,
                "bboxUv": [x0 / width, y0 / height, (x1 + 1) / width, (y1 + 1) / height],
                "sizePx": [x1 - x0 + 1, y1 - y0 + 1],
                "centroidPx": [clean_float(centroid_x), clean_float(centroid_y)],
                "centroidUv": [clean_float(centroid_x / width, 6), clean_float(centroid_y / height, 6)],
                "layerCounts": layer_counts,
                **({"fovClip": fov_clip} if fov_clip else {}),
                "quadFit": quad_fit,
                "overlapChildren": overlap_children,
            }
        )

    bboxes.sort(key=lambda item: (-int(item["pixelCount"]), item["bboxPx"][1], item["bboxPx"][0]))
    bboxes = bboxes[: int(settings["maxBboxesPerFrame"])]
    for index, item in enumerate(bboxes, start=1):
        item["bboxId"] = f"bbox-{frame_ordinal:06d}-{index:03d}"
        children = item.get("overlapChildren") if isinstance(item.get("overlapChildren"), list) else []
        for child_index, child in enumerate(children, start=1):
            child["parentBboxId"] = item["bboxId"]
            child["overlapId"] = f"{item['bboxId']}-overlap-{child_index:03d}"
    return bboxes


def build_mask_bboxes(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    config = read_json(args.config)
    rules = read_json(args.rules)
    review = read_json(args.review) if args.review.exists() else {"annotations": [], "decisions": []}
    manifest_path = args.precompute_manifest or selected_precompute_manifest()
    manifest = read_json(manifest_path)
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frames_in = list(manifest.get("frames") or [])
    frame_count = int(manifest.get("frameCount") or len(frames_in))
    if not frames_in:
        raise ValueError("bbox build requires frame entries in the precompute manifest")

    raw_settings = {
        "minPixels": args.min_pixels,
        "maxBboxesPerFrame": getattr(args, "max_bboxes_per_frame", DEFAULT_MAX_BBOXES_PER_FRAME),
        "fitTightness": args.fit_tightness,
        "quadFitEnabled": getattr(args, "quad_fit_enabled", DEFAULT_QUAD_FIT_ENABLED),
        "quadFitMode": getattr(args, "quad_fit_mode", DEFAULT_QUAD_FIT_MODE),
        "targetAspect": getattr(args, "target_aspect", DEFAULT_TARGET_ASPECT),
        "aspectTolerance": getattr(args, "aspect_tolerance", DEFAULT_ASPECT_TOLERANCE),
        "quadThicknessPx": getattr(args, "quad_thickness_px", DEFAULT_QUAD_THICKNESS_PX),
        "edgeCoverageMin": getattr(args, "edge_coverage_min", DEFAULT_EDGE_COVERAGE_MIN),
        "cornerMinPixels": getattr(args, "corner_min_pixels", DEFAULT_CORNER_MIN_PIXELS),
        "voidOverlapMaxRatio": getattr(args, "void_overlap_max_ratio", DEFAULT_VOID_OVERLAP_MAX_RATIO),
        "voidOverlapMinPixels": getattr(args, "void_overlap_min_pixels", DEFAULT_VOID_OVERLAP_MIN_PIXELS),
        "fovClip": getattr(
            args,
            "fov_clip",
            {
                "enabled": getattr(args, "fov_clip_enabled", DEFAULT_FOV_CLIP_ENABLED),
                "marginPx": getattr(args, "fov_clip_margin_px", DEFAULT_FOV_CLIP_MARGIN_PX),
                "minContactPixels": getattr(args, "fov_clip_min_contact_pixels", DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS),
                "minContactRatio": getattr(args, "fov_clip_min_contact_ratio", DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO),
                "requireBboxTouch": getattr(args, "fov_clip_require_bbox_touch", DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH),
                "warnOnly": getattr(args, "fov_clip_warn_only", DEFAULT_FOV_CLIP_WARN_ONLY),
            },
        ),
    }
    settings = normalized_bbox_settings(raw_settings)
    mask_context = build_mask_context(config, rules, review, manifest)
    source_signature = bbox_source_signature(manifest, manifest_path, settings, mask_context["signature"])

    start = max(0, int(args.start_frame or 0))
    end = frame_count
    if args.max_frames is not None:
        end = min(end, start + max(0, int(args.max_frames)))
    selected_frames = list(enumerate(frames_in[start:end], start=start))
    if not selected_frames:
        raise ValueError("selected bbox frame range is empty")

    output_root = bbox_run_root(run_key, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frames_out = []
    total_bboxes = 0
    total_quad_accepted = 0
    total_quad_rejected = 0
    total_void_overlaps = 0
    total_overlap_children = 0
    total_selected_pixels = 0
    max_bboxes = 0
    fov_clip_summary = empty_fov_clip_summary(bool(settings.get("fovClip", {}).get("enabled")))
    fov_clip_severities: list[float] = []
    for local_index, (frame_ordinal, frame_entry) in enumerate(selected_frames):
        color_ids = read_frame_color_ids(frame_entry, width, height)
        layer_hits = mask_context["membership"][:, color_ids]
        selected = np.any(layer_hits, axis=0)
        selected_pixel_count = int(np.count_nonzero(selected))
        bboxes = component_bboxes(selected, layer_hits, mask_context["classes"], frame_ordinal, width, height, settings)
        overlap_bboxes = []
        frame_quad_accepted = 0
        frame_quad_rejected = 0
        frame_void_overlaps = 0
        for bbox in bboxes:
            quad_fit = bbox.get("quadFit") if isinstance(bbox.get("quadFit"), dict) else {}
            if quad_fit.get("accepted"):
                frame_quad_accepted += 1
            elif quad_fit.get("enabled"):
                frame_quad_rejected += 1
            if quad_fit.get("hasVoidOverlap"):
                frame_void_overlaps += 1
            children = bbox.get("overlapChildren") if isinstance(bbox.get("overlapChildren"), list) else []
            overlap_bboxes.extend(children)
        if fov_clip_summary["enabled"]:
            fov_clip_severities.extend(collect_fov_clip_summary(fov_clip_summary, bboxes))
        total_bboxes += len(bboxes)
        total_quad_accepted += frame_quad_accepted
        total_quad_rejected += frame_quad_rejected
        total_void_overlaps += frame_void_overlaps
        total_overlap_children += len(overlap_bboxes)
        max_bboxes = max(max_bboxes, len(bboxes))
        total_selected_pixels += selected_pixel_count
        frames_out.append(
            {
                "frameOrdinal": frame_ordinal,
                "frameId": frame_entry.get("frameId") or frame_entry.get("rawIndex") or frame_entry.get("index"),
                "path": frame_entry.get("path") or frame_entry.get("sourcePath"),
                "cachePath": frame_entry.get("cachePath"),
                "selectedPixelCount": selected_pixel_count,
                "bboxCount": len(bboxes),
                "bboxes": bboxes,
                "quadAcceptedCount": frame_quad_accepted,
                "quadRejectedCount": frame_quad_rejected,
                "voidOverlapCount": frame_void_overlaps,
                "overlapBboxes": overlap_bboxes,
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-mask-bboxes",
                    "index": local_index + 1,
                    "total": len(selected_frames),
                    "frameOrdinal": frame_ordinal,
                    "selectedPixels": selected_pixel_count,
                    "bboxCount": len(bboxes),
                    "quadAcceptedCount": frame_quad_accepted,
                    "voidOverlapCount": frame_void_overlaps,
                }
            )

    max_bbox_frame_ordinals = []
    for frame in frames_out:
        is_max_frame = max_bboxes > 0 and int(frame.get("bboxCount") or 0) == max_bboxes
        frame["isMaxBboxCountFrame"] = is_max_frame
        if is_max_frame:
            max_bbox_frame_ordinals.append(int(frame.get("frameOrdinal") or 0))
    if fov_clip_summary["enabled"]:
        fov_clip_summary["maxSeverity"] = clean_float(max(fov_clip_severities) if fov_clip_severities else 0.0, 4)
        fov_clip_summary["meanSeverity"] = clean_float(float(np.mean(fov_clip_severities)) if fov_clip_severities else 0.0, 4)

    summary = {
        "bboxCount": total_bboxes,
        "maxBboxesPerFrame": max_bboxes,
        "observedMaxBboxesPerFrame": max_bboxes,
        "maxBboxFrameCount": len(max_bbox_frame_ordinals),
        "maxBboxFrameOrdinals": max_bbox_frame_ordinals[:100],
        "bboxFrameCap": int(settings["maxBboxesPerFrame"]),
        "selectedPixelCount": total_selected_pixels,
        "meanBboxesPerFrame": total_bboxes / len(frames_out) if frames_out else 0,
        "meanSelectedPixelsPerFrame": total_selected_pixels / len(frames_out) if frames_out else 0,
        "quadAcceptedCount": total_quad_accepted,
        "quadRejectedCount": total_quad_rejected,
        "voidOverlapCount": total_void_overlaps,
        "overlapChildCount": total_overlap_children,
    }
    if fov_clip_summary["enabled"]:
        summary["fovClip"] = fov_clip_summary

    output = {
        "version": 1,
        "app": "vision_passthrough_review",
        "kind": "color-mask-pixel-bboxes",
        "createdAt": utc_now(),
        "runKey": run_key,
        "precomputeManifest": repo_url(manifest_path),
        "sourceSignature": source_signature,
        "complete": start == 0 and len(frames_out) >= frame_count,
        "frameStart": start,
        "frameCount": len(frames_out),
        "sourceFrameCount": frame_count,
        "image": {"width": width, "height": height},
        "settings": settings,
        "input": {
            "source": "enabled-color-layers",
            "maskSignature": mask_context["signature"],
            "enabledPrefixes": mask_context["enabledPrefixes"],
            "layerSummaries": concise_layer_summaries(mask_context["summaries"]),
        },
        **({"fovClipSummary": fov_clip_summary} if fov_clip_summary["enabled"] else {}),
        "summary": summary,
        "frames": frames_out,
    }
    atomic_write_json(output_root / "bbox_manifest.json", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-bboxes-per-frame", type=int, default=DEFAULT_MAX_BBOXES_PER_FRAME)
    parser.add_argument("--fit-tightness", type=int, default=DEFAULT_FIT_TIGHTNESS)
    parser.add_argument("--quad-fit-enabled", action=argparse.BooleanOptionalAction, default=DEFAULT_QUAD_FIT_ENABLED)
    parser.add_argument("--quad-fit-mode", choices=sorted(QUAD_FIT_MODES), default=DEFAULT_QUAD_FIT_MODE)
    parser.add_argument("--target-aspect", type=float, default=DEFAULT_TARGET_ASPECT)
    parser.add_argument("--aspect-tolerance", type=float, default=DEFAULT_ASPECT_TOLERANCE)
    parser.add_argument("--quad-thickness-px", type=int, default=DEFAULT_QUAD_THICKNESS_PX)
    parser.add_argument("--edge-coverage-min", type=float, default=DEFAULT_EDGE_COVERAGE_MIN)
    parser.add_argument("--corner-min-pixels", type=int, default=DEFAULT_CORNER_MIN_PIXELS)
    parser.add_argument("--void-overlap-max-ratio", type=float, default=DEFAULT_VOID_OVERLAP_MAX_RATIO)
    parser.add_argument("--void-overlap-min-pixels", type=int, default=DEFAULT_VOID_OVERLAP_MIN_PIXELS)
    parser.add_argument("--fov-clip-enabled", action=argparse.BooleanOptionalAction, default=DEFAULT_FOV_CLIP_ENABLED)
    parser.add_argument("--fov-clip-margin-px", type=int, default=DEFAULT_FOV_CLIP_MARGIN_PX)
    parser.add_argument("--fov-clip-min-contact-pixels", type=int, default=DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS)
    parser.add_argument("--fov-clip-min-contact-ratio", type=float, default=DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO)
    parser.add_argument("--fov-clip-require-bbox-touch", action=argparse.BooleanOptionalAction, default=DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH)
    parser.add_argument("--fov-clip-warn-only", action=argparse.BooleanOptionalAction, default=DEFAULT_FOV_CLIP_WARN_ONLY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_mask_bboxes(args)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": repo_url(bbox_run_root(result["runKey"], args.output_root) / "bbox_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
