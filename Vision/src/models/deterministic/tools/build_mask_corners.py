#!/usr/bin/env python3
"""Build bbox-local hull and internal-void corner evidence from enabled color-mask layers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from models.deterministic.tools.build_mask_bboxes import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_BBOX_ROOT,
    bbox_run_root,
    bbox_settings_from_review,
    bbox_source_signature,
    build_mask_context,
    concise_layer_summaries,
)
from models.deterministic.tools.build_vision_memory import (
    APP_DIR,
    DEFAULT_CONFIG,
    DEFAULT_REVIEW,
    DEFAULT_RULES,
    REPO_ROOT,
    atomic_write_json,
    read_frame_color_ids,
    read_json,
    selected_precompute_manifest,
    stable_hash,
)


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "mask_corners"
DEFAULT_RADIUS_PX = 8
DEFAULT_MIN_ANGLE_DEG = 45
DEFAULT_MAX_ANGLE_DEG = 135
DEFAULT_MIN_SUPPORT_PIXELS = 4
DEFAULT_CONTOUR_EPSILON_PX = 2.0
DEFAULT_MIN_DISTANCE_PX = 8
DEFAULT_MAX_HULL_CORNERS_PER_BBOX = 64
DEFAULT_MAX_VOID_CORNERS_PER_VOID = 24
DEFAULT_VOID_MIN_AREA_PX = 20
ANGLE_BIN_COUNT = 72
CORNER_TYPE_ORDER = ("white", "black", "orange", "green")
CORNER_TYPE_META = {
    "white": {
        "category": "hull",
        "position": "inside",
        "baseGroup": "hull",
        "maxKey": "maxCornersPerBbox",
        "maxDefault": DEFAULT_MAX_HULL_CORNERS_PER_BBOX,
        "legacyMinKey": "minMaskInsidePoints",
        "legacyMaxKey": "maxMaskInsidePoints",
    },
    "black": {
        "category": "hull",
        "position": "outside",
        "baseGroup": "hull",
        "maxKey": "maxCornersPerBbox",
        "maxDefault": DEFAULT_MAX_HULL_CORNERS_PER_BBOX,
        "legacyMinKey": "minMaskOutsidePoints",
        "legacyMaxKey": "maxMaskOutsidePoints",
    },
    "orange": {
        "category": "void",
        "position": "inside",
        "baseGroup": "void",
        "maxKey": "maxCornersPerVoid",
        "maxDefault": DEFAULT_MAX_VOID_CORNERS_PER_VOID,
        "legacyMinKey": "minMaskInsidePoints",
        "legacyMaxKey": "maxMaskInsidePoints",
    },
    "green": {
        "category": "void",
        "position": "outside",
        "baseGroup": "void",
        "maxKey": "maxCornersPerVoid",
        "maxDefault": DEFAULT_MAX_VOID_CORNERS_PER_VOID,
        "legacyMinKey": "minMaskOutsidePoints",
        "legacyMaxKey": "maxMaskOutsidePoints",
    },
}


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


def corner_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build corner artifacts without an active run key")
    return output_root / clean


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def numeric_setting(raw: dict, key: str, default: float, low: float, high: float, integer: bool = False):
    value = raw.get(key, default)
    if value in {None, ""}:
        value = default
    value = clamp(float(value), low, high)
    return int(round(value)) if integer else value


def numeric_setting_any(raw: dict, keys: list[str], default: float, low: float, high: float, integer: bool = False):
    for key in keys:
        if key in raw and raw.get(key) not in {None, ""}:
            return numeric_setting(raw, key, default, low, high, integer)
    return int(round(default)) if integer else default


def normalized_angle_settings(source: dict, group_key: str, max_key: str, max_default: int) -> dict:
    group = source.get(group_key) if isinstance(source.get(group_key), dict) else {}
    merged = {**source, **group}
    prefix = f"{group_key[0].upper()}{group_key[1:]}"
    min_angle = numeric_setting_any(
        merged,
        ["minAngleDeg", f"{group_key}MinAngleDeg", f"{prefix}MinAngleDeg"],
        DEFAULT_MIN_ANGLE_DEG,
        1,
        179,
        integer=True,
    )
    max_angle = numeric_setting_any(
        merged,
        ["maxAngleDeg", f"{group_key}MaxAngleDeg", f"{prefix}MaxAngleDeg"],
        DEFAULT_MAX_ANGLE_DEG,
        1,
        179,
        integer=True,
    )
    if max_angle < min_angle:
        min_angle, max_angle = max_angle, min_angle
    settings = {
        "radiusPx": numeric_setting_any(
            merged,
            ["radiusPx", f"{group_key}RadiusPx", f"{prefix}RadiusPx"],
            DEFAULT_RADIUS_PX,
            2,
            64,
            integer=True,
        ),
        "minAngleDeg": min_angle,
        "maxAngleDeg": max_angle,
        "minSupportPixels": numeric_setting_any(
            merged,
            ["minSupportPixels", f"{group_key}MinSupportPixels", f"{prefix}MinSupportPixels"],
            DEFAULT_MIN_SUPPORT_PIXELS,
            1,
            5000,
            integer=True,
        ),
        "contourEpsilonPx": numeric_setting_any(
            merged,
            ["contourEpsilonPx", f"{group_key}ContourEpsilonPx", f"{prefix}ContourEpsilonPx"],
            DEFAULT_CONTOUR_EPSILON_PX,
            0.25,
            24,
        ),
        "minDistancePx": numeric_setting_any(
            merged,
            ["minDistancePx", f"{group_key}MinDistancePx", f"{prefix}MinDistancePx"],
            DEFAULT_MIN_DISTANCE_PX,
            1,
            80,
            integer=True,
        ),
        max_key: numeric_setting_any(
            merged,
            [max_key, f"{group_key}{max_key[0].upper()}{max_key[1:]}", f"{prefix}{max_key[0].upper()}{max_key[1:]}"],
            max_default,
            1,
            5000,
            integer=True,
        ),
    }
    max_inside = numeric_setting_any(
        merged,
        ["maxMaskInsidePoints", f"{group_key}MaxMaskInsidePoints", f"{prefix}MaxMaskInsidePoints"],
        int(settings[max_key]),
        0,
        5000,
        integer=True,
    )
    min_inside = numeric_setting_any(
        merged,
        ["minMaskInsidePoints", f"{group_key}MinMaskInsidePoints", f"{prefix}MinMaskInsidePoints"],
        0,
        0,
        5000,
        integer=True,
    )
    max_outside = numeric_setting_any(
        merged,
        ["maxMaskOutsidePoints", f"{group_key}MaxMaskOutsidePoints", f"{prefix}MaxMaskOutsidePoints"],
        int(settings[max_key]),
        0,
        5000,
        integer=True,
    )
    min_outside = numeric_setting_any(
        merged,
        ["minMaskOutsidePoints", f"{group_key}MinMaskOutsidePoints", f"{prefix}MinMaskOutsidePoints"],
        0,
        0,
        5000,
        integer=True,
    )
    settings["minMaskInsidePoints"] = min_inside
    settings["maxMaskInsidePoints"] = max(min_inside, max_inside)
    settings["minMaskOutsidePoints"] = min_outside
    settings["maxMaskOutsidePoints"] = max(min_outside, max_outside)
    return settings


def normalized_type_settings(source: dict, type_key: str, base_settings: dict) -> dict:
    types = source.get("types") if isinstance(source.get("types"), dict) else {}
    explicit = types.get(type_key) if isinstance(types.get(type_key), dict) else {}
    direct = source.get(type_key) if isinstance(source.get(type_key), dict) else {}
    raw = {**direct, **explicit}
    meta = CORNER_TYPE_META[type_key]
    merged = {**base_settings, **raw}
    min_angle = numeric_setting_any(
        merged,
        ["minAngleDeg"],
        float(base_settings.get("minAngleDeg", DEFAULT_MIN_ANGLE_DEG)),
        1,
        179,
        integer=True,
    )
    max_angle = numeric_setting_any(
        merged,
        ["maxAngleDeg"],
        float(base_settings.get("maxAngleDeg", DEFAULT_MAX_ANGLE_DEG)),
        1,
        179,
        integer=True,
    )
    if max_angle < min_angle:
        min_angle, max_angle = max_angle, min_angle
    legacy_max = int(base_settings.get(meta["legacyMaxKey"], base_settings.get(meta["maxKey"], meta["maxDefault"])) or 0)
    legacy_min = int(base_settings.get(meta["legacyMinKey"], 0) or 0)
    max_points = numeric_setting_any(
        merged,
        ["maxPoints", meta["legacyMaxKey"]],
        legacy_max,
        0,
        5000,
        integer=True,
    )
    min_points = numeric_setting_any(
        merged,
        ["minPoints", meta["legacyMinKey"]],
        legacy_min,
        0,
        5000,
        integer=True,
    )
    settings = {
        "category": meta["category"],
        "targetPosition": meta["position"],
        "radiusPx": numeric_setting_any(
            merged,
            ["radiusPx"],
            float(base_settings.get("radiusPx", DEFAULT_RADIUS_PX)),
            2,
            64,
            integer=True,
        ),
        "minAngleDeg": min_angle,
        "maxAngleDeg": max_angle,
        "minSupportPixels": numeric_setting_any(
            merged,
            ["minSupportPixels"],
            float(base_settings.get("minSupportPixels", DEFAULT_MIN_SUPPORT_PIXELS)),
            1,
            5000,
            integer=True,
        ),
        "contourEpsilonPx": numeric_setting_any(
            merged,
            ["contourEpsilonPx"],
            float(base_settings.get("contourEpsilonPx", DEFAULT_CONTOUR_EPSILON_PX)),
            0.25,
            24,
        ),
        "minDistancePx": numeric_setting_any(
            merged,
            ["minDistancePx"],
            float(base_settings.get("minDistancePx", DEFAULT_MIN_DISTANCE_PX)),
            1,
            80,
            integer=True,
        ),
        "minPoints": min_points,
        "maxPoints": max(min_points, max_points),
    }
    if meta["category"] == "void":
        settings["minAreaPx"] = numeric_setting_any(
            merged,
            ["minAreaPx"],
            float(base_settings.get("minAreaPx", DEFAULT_VOID_MIN_AREA_PX)),
            1,
            100000,
            integer=True,
        )
    return settings


def normalized_corner_settings(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    void_settings = normalized_angle_settings(source, "void", "maxCornersPerVoid", DEFAULT_MAX_VOID_CORNERS_PER_VOID)
    void_group = source.get("void") if isinstance(source.get("void"), dict) else {}
    void_merged = {**source, **void_group}
    void_settings["minAreaPx"] = numeric_setting_any(
        void_merged,
        ["minAreaPx", "voidMinAreaPx", "VoidMinAreaPx"],
        DEFAULT_VOID_MIN_AREA_PX,
        1,
        100000,
        integer=True,
    )
    hull_settings = normalized_angle_settings(source, "hull", "maxCornersPerBbox", DEFAULT_MAX_HULL_CORNERS_PER_BBOX)
    settings = {
        "hull": hull_settings,
        "void": void_settings,
    }
    settings["types"] = {
        key: normalized_type_settings(source, key, settings[CORNER_TYPE_META[key]["baseGroup"]])
        for key in CORNER_TYPE_ORDER
    }
    return settings


def corner_settings_from_review(review: dict) -> dict:
    raw = review.get("cornerFlow") if isinstance(review.get("cornerFlow"), dict) else {}
    return normalized_corner_settings(raw)


def corner_source_signature(
    manifest: dict,
    manifest_path: Path,
    settings: dict,
    mask_signature: str,
    bbox_signature: str | None = None,
) -> str:
    return stable_hash(
        {
            "kind": "0720-color-mask-bbox-corners",
            "runKey": manifest.get("runKey"),
            "precomputeManifest": repo_url(manifest_path),
            "frameCount": int(manifest.get("frameCount") or len(manifest.get("frames") or [])),
            "width": int(manifest.get("width") or 640),
            "height": int(manifest.get("height") or 360),
            "settings": normalized_corner_settings(settings),
            "maskSignature": mask_signature,
            "bboxSignature": bbox_signature,
        }
    )


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def largest_circular_run(flags: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(flags, dtype=bool)
    count = int(values.size)
    if count <= 0 or not np.any(values):
        return 0.0, 0.0, 0
    if np.all(values):
        return 360.0, 0.0, count
    doubled = np.concatenate([values, values])
    best_start = 0
    best_length = 0
    start = 0
    length = 0
    for index, value in enumerate(doubled):
        if value:
            if length == 0:
                start = index
            length += 1
            if length > count:
                length = count
                start = index - count + 1
            if length > best_length:
                best_length = length
                best_start = start
        else:
            length = 0
    bin_degrees = 360.0 / count
    center_bin = (best_start + (best_length - 1) / 2.0) % count
    return best_length * bin_degrees, (center_bin * bin_degrees) % 360.0, best_length


def layer_counts_for_disk(layer_hits: np.ndarray, classes: list[dict], disk_mask: np.ndarray, bounds: tuple[int, int, int, int]) -> list[dict]:
    x0, y0, x1, y1 = bounds
    hits = layer_hits[:, y0:y1, x0:x1]
    counts = []
    for class_index, item in enumerate(classes):
        count = int(np.count_nonzero(hits[class_index] & disk_mask))
        if count:
            counts.append(
                {
                    "prefix": str(item.get("prefix")),
                    "displayName": item.get("displayName"),
                    "pixelCount": count,
                }
            )
    return counts


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> tuple[float | None, float | None]:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return None, None
    u1 = v1 / n1
    u2 = v2 / n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    bisector = u1 + u2
    if float(np.linalg.norm(bisector)) <= 1e-6:
        bisector = np.array([-u1[1], u1[0]], dtype=np.float32)
    direction = (math.degrees(math.atan2(float(bisector[1]), float(bisector[0]))) + 360.0) % 360.0
    return angle, direction


def sample_minor_wedge(
    selected: np.ndarray,
    px: float,
    py: float,
    direction_deg: float,
    angle_deg: float,
    radius: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    mask_count = 0
    void_count = 0
    ray_count = max(3, min(13, int(round(angle_deg / 12.0)) + 1))
    distances = range(1, max(2, int(radius)) + 1)
    for offset in np.linspace(-angle_deg / 2.0, angle_deg / 2.0, ray_count):
        theta = math.radians(direction_deg + float(offset))
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        for distance in distances:
            x = int(round(px + cos_theta * distance))
            y = int(round(py + sin_theta * distance))
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if selected[y, x]:
                mask_count += 1
            else:
                void_count += 1
    return mask_count, void_count


def mask_angle_position(mask_samples: int, void_samples: int, min_support: int, angle_valid: bool) -> str:
    if not angle_valid:
        return "ambiguous"
    if mask_samples > void_samples and mask_samples >= min_support:
        return "inside"
    if void_samples > mask_samples and void_samples >= min_support:
        return "outside"
    return "ambiguous"


def legacy_wedge_type(position: str) -> str:
    if position == "inside":
        return "mask-wedge"
    if position == "outside":
        return "void-wedge"
    return "ambiguous"


def corner_mask_position(corner: dict) -> str:
    position = str(corner.get("maskAnglePosition") or "")
    if position in {"inside", "outside", "ambiguous"}:
        return position
    corner_type = str(corner.get("type") or "")
    if corner_type == "mask-wedge":
        return "inside"
    if corner_type == "void-wedge":
        return "outside"
    return "ambiguous"


def classify_corner(
    selected: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    point: tuple[float, float],
    frame_ordinal: int,
    contour_index: int,
    contour_role: str,
    source: str,
    component_labels: np.ndarray,
    settings: dict,
    width: int,
    height: int,
    contour_angle_deg: float | None = None,
    contour_direction_deg: float | None = None,
) -> dict | None:
    px, py = float(point[0]), float(point[1])
    radius = int(settings["radiusPx"])
    x0 = max(0, int(math.floor(px - radius)))
    y0 = max(0, int(math.floor(py - radius)))
    x1 = min(width, int(math.ceil(px + radius + 1)))
    y1 = min(height, int(math.ceil(py + radius + 1)))
    if x1 <= x0 or y1 <= y0:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) + 0.5 - px
    dy = yy.astype(np.float32) + 0.5 - py
    distances = np.sqrt((dx * dx) + (dy * dy))
    disk = (distances <= radius) & (distances >= 0.5)
    if not np.any(disk):
        return None

    local_mask = selected[y0:y1, x0:x1]
    mask_disk = disk & local_mask
    void_disk = disk & ~local_mask
    mask_support = int(np.count_nonzero(mask_disk))
    void_support = int(np.count_nonzero(void_disk))
    if mask_support < int(settings["minSupportPixels"]) and void_support < int(settings["minSupportPixels"]):
        return None

    if contour_angle_deg is not None and contour_direction_deg is not None:
        small_angle = float(contour_angle_deg)
        direction = float(contour_direction_deg)
        wedge_mask, wedge_void = sample_minor_wedge(selected, px, py, direction, small_angle, radius, width, height)
        flipped_direction = (direction + 180.0) % 360.0
        flipped_mask, flipped_void = sample_minor_wedge(selected, px, py, flipped_direction, small_angle, radius, width, height)
        min_angle = float(settings["minAngleDeg"])
        max_angle = float(settings["maxAngleDeg"])
        angle_valid = min_angle <= small_angle <= max_angle
        min_support = int(settings["minSupportPixels"])
        position = mask_angle_position(wedge_mask, wedge_void, min_support, angle_valid)
        corner_type = legacy_wedge_type(position)
        if position == "inside":
            mask_angle = small_angle
            void_angle = 360.0 - small_angle
            support = wedge_mask
        elif position == "outside":
            mask_angle = 360.0 - small_angle
            void_angle = small_angle
            support = wedge_void
        else:
            if wedge_mask >= wedge_void:
                mask_angle = small_angle
                void_angle = 360.0 - small_angle
                support = wedge_mask
            else:
                mask_angle = 360.0 - small_angle
                void_angle = small_angle
                support = wedge_void
        angle_score = max(0.0, 1.0 - abs(small_angle - 90.0) / 90.0)
        support_score = min(1.0, float(support) / max(1.0, float(radius * radius)))
        type_bonus = 0.25 if corner_type != "ambiguous" else 0.0
        nearest_x = int(round(px))
        nearest_y = int(round(py))
        nearest_x = max(0, min(width - 1, nearest_x))
        nearest_y = max(0, min(height - 1, nearest_y))
        return {
            "cornerId": "",
            "frameOrdinal": frame_ordinal,
            "type": corner_type,
            "maskAnglePosition": position,
            "pointPx": [clean_float(px), clean_float(py)],
            "pointUv": [clean_float(px / width, 6), clean_float(py / height, 6)],
            "angleDeg": clean_float(small_angle),
            "directionDeg": clean_float(direction),
            "radiusPx": radius,
            "maskAngleDeg": clean_float(mask_angle),
            "voidAngleDeg": clean_float(void_angle),
            "maskSupportPixels": mask_support,
            "voidSupportPixels": void_support,
            "wedgeMaskSamples": wedge_mask,
            "wedgeVoidSamples": wedge_void,
            "measuredWedge": {
                "directionDeg": clean_float(direction),
                "angleDeg": clean_float(small_angle),
                "maskSamples": wedge_mask,
                "voidSamples": wedge_void,
            },
            "flippedWedge": {
                "directionDeg": clean_float(flipped_direction),
                "angleDeg": clean_float(small_angle),
                "maskSamples": flipped_mask,
                "voidSamples": flipped_void,
            },
            "score": clean_float((angle_score * 0.7) + (support_score * 0.3) + type_bonus, 5),
            "source": source,
            "contourIndex": contour_index,
            "contourRole": contour_role,
            "componentLabel": int(component_labels[nearest_y, nearest_x]),
            "layerCounts": layer_counts_for_disk(layer_hits, classes, disk & local_mask, (x0, y0, x1, y1)),
        }

    angles = (np.degrees(np.arctan2(dy[disk], dx[disk])) + 360.0) % 360.0
    bins = np.floor(angles / (360.0 / ANGLE_BIN_COUNT)).astype(np.int32)
    bins = np.clip(bins, 0, ANGLE_BIN_COUNT - 1)
    disk_mask_values = local_mask[disk]
    mask_bins = np.zeros(ANGLE_BIN_COUNT, dtype=bool)
    void_bins = np.zeros(ANGLE_BIN_COUNT, dtype=bool)
    if np.any(disk_mask_values):
        mask_bins[np.unique(bins[disk_mask_values])] = True
    if np.any(~disk_mask_values):
        void_bins[np.unique(bins[~disk_mask_values])] = True

    mask_angle, mask_direction, _ = largest_circular_run(mask_bins)
    void_angle, void_direction, _ = largest_circular_run(void_bins)
    min_angle = float(settings["minAngleDeg"])
    max_angle = float(settings["maxAngleDeg"])
    mask_valid = min_angle <= mask_angle <= max_angle and mask_support >= int(settings["minSupportPixels"])
    void_valid = min_angle <= void_angle <= max_angle and void_support >= int(settings["minSupportPixels"])

    if mask_valid and not void_valid:
        corner_type = "mask-wedge"
        position = "inside"
        angle = mask_angle
        direction = mask_direction
        support = mask_support
    elif void_valid and not mask_valid:
        corner_type = "void-wedge"
        position = "outside"
        angle = void_angle
        direction = void_direction
        support = void_support
    elif mask_valid and void_valid:
        mask_error = abs(mask_angle - 90.0)
        void_error = abs(void_angle - 90.0)
        corner_type = "mask-wedge" if mask_error <= void_error else "void-wedge"
        position = "inside" if corner_type == "mask-wedge" else "outside"
        angle = mask_angle if corner_type == "mask-wedge" else void_angle
        direction = mask_direction if corner_type == "mask-wedge" else void_direction
        support = mask_support if corner_type == "mask-wedge" else void_support
    else:
        corner_type = "ambiguous"
        position = "ambiguous"
        if abs(mask_angle - 90.0) <= abs(void_angle - 90.0):
            angle = mask_angle
            direction = mask_direction
            support = mask_support
        else:
            angle = void_angle
            direction = void_direction
            support = void_support

    angle_score = max(0.0, 1.0 - abs(float(angle) - 90.0) / 90.0)
    support_score = min(1.0, float(support) / max(1.0, float(radius * radius)))
    type_bonus = 0.25 if corner_type != "ambiguous" else 0.0
    nearest_x = int(round(px))
    nearest_y = int(round(py))
    nearest_x = max(0, min(width - 1, nearest_x))
    nearest_y = max(0, min(height - 1, nearest_y))
    return {
        "cornerId": "",
        "frameOrdinal": frame_ordinal,
        "type": corner_type,
        "maskAnglePosition": position,
        "pointPx": [clean_float(px), clean_float(py)],
        "pointUv": [clean_float(px / width, 6), clean_float(py / height, 6)],
        "angleDeg": clean_float(angle),
        "directionDeg": clean_float(direction),
        "radiusPx": radius,
        "maskAngleDeg": clean_float(mask_angle),
        "voidAngleDeg": clean_float(void_angle),
        "maskSupportPixels": mask_support,
        "voidSupportPixels": void_support,
        "score": clean_float((angle_score * 0.7) + (support_score * 0.3) + type_bonus, 5),
        "source": source,
        "contourIndex": contour_index,
        "contourRole": contour_role,
        "componentLabel": int(component_labels[nearest_y, nearest_x]),
        "layerCounts": layer_counts_for_disk(layer_hits, classes, disk & local_mask, (x0, y0, x1, y1)),
    }


def contour_role(hierarchy, index: int) -> str:
    if hierarchy is None or len(hierarchy) == 0:
        return "external"
    parent = int(hierarchy[0][index][3])
    return "void-boundary" if parent >= 0 else "external"


def raw_corner_candidates(
    cv2,
    selected: np.ndarray,
    settings: dict,
    offset: tuple[int, int] = (0, 0),
    retrieval_mode=None,
    role_override: str | None = None,
) -> list[dict]:
    source = selected.astype(np.uint8) * 255
    mode = retrieval_mode if retrieval_mode is not None else cv2.RETR_TREE
    contours, hierarchy = cv2.findContours(source, mode, cv2.CHAIN_APPROX_NONE)
    candidates = []
    offset_x, offset_y = offset
    for contour_index, contour in enumerate(contours):
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, float(settings["contourEpsilonPx"]), True)
        role = role_override or contour_role(hierarchy, contour_index)
        points = approx.reshape(-1, 2).astype(np.float32) + 0.5
        if points.shape[0] < 3:
            continue
        for index, point in enumerate(points):
            angle, direction = angle_between_vectors(points[(index - 1) % len(points)] - point, points[(index + 1) % len(points)] - point)
            candidates.append(
                {
                    "point": (float(point[0] + offset_x), float(point[1] + offset_y)),
                    "contourIndex": contour_index,
                    "contourRole": role,
                    "source": "approx-poly",
                    "angleDeg": angle,
                    "directionDeg": direction,
                }
            )
    return candidates


def merge_corners(corners: list[dict], settings: dict, id_prefix: str, max_key: str) -> list[dict]:
    min_distance = float(settings["minDistancePx"])
    if "maxPoints" in settings and settings.get("maxPoints") is not None:
        max_corners = max(0, int(settings.get("maxPoints") or 0))
    else:
        max_corners = int(settings.get(max_key) or settings.get("maxCornersPerFrame") or DEFAULT_MAX_HULL_CORNERS_PER_BBOX)
    target_position = str(settings.get("targetPosition") or "")
    order = {"mask-wedge": 0, "void-wedge": 1, "ambiguous": 2}
    sorted_corners = sorted(
        [
            corner
            for corner in corners
            if target_position not in {"inside", "outside"} or corner_mask_position(corner) == target_position
        ],
        key=lambda item: (
            order.get(str(item.get("type")), 3),
            -float(item.get("score") or 0),
            item.get("pointPx", [0, 0])[1],
            item.get("pointPx", [0, 0])[0],
        ),
    )
    kept = []
    for corner in sorted_corners:
        x, y = [float(value) for value in corner.get("pointPx", [0, 0])[:2]]
        if any(math.hypot(x - float(other["pointPx"][0]), y - float(other["pointPx"][1])) < min_distance for other in kept):
            continue
        kept.append(corner)
        if len(kept) >= 5000:
            break
    kept = apply_position_point_limits(kept, settings)
    kept = kept[:max_corners]
    for index, corner in enumerate(kept, start=1):
        corner["cornerId"] = f"{id_prefix}-corner-{index:03d}"
    return kept


def apply_position_point_limits(corners: list[dict], settings: dict) -> list[dict]:
    target_position = str(settings.get("targetPosition") or "")
    if target_position in {"inside", "outside"}:
        items = [corner for corner in corners if corner_mask_position(corner) == target_position]
        min_points = int(settings.get("minPoints", 0) or 0)
        max_points = int(settings.get("maxPoints", len(items)) if settings.get("maxPoints") is not None else len(items))
        capped = items[: max(0, max_points)]
        return capped if len(capped) >= min_points else []

    allowed = set()
    for position, min_key, max_key in (
        ("inside", "minMaskInsidePoints", "maxMaskInsidePoints"),
        ("outside", "minMaskOutsidePoints", "maxMaskOutsidePoints"),
    ):
        items = [corner for corner in corners if corner_mask_position(corner) == position]
        min_points = int(settings.get(min_key, 0) or 0)
        max_points = int(settings.get(max_key, len(items)) if settings.get(max_key) is not None else len(items))
        max_points = max(0, max_points)
        capped = items[:max_points]
        if len(capped) >= min_points:
            allowed.update(id(corner) for corner in capped)
    allowed.update(id(corner) for corner in corners if corner_mask_position(corner) == "ambiguous")
    return [corner for corner in corners if id(corner) in allowed]


def bbox_bounds(bbox: dict, width: int, height: int, margin: int = 0) -> tuple[int, int, int, int] | None:
    values = bbox.get("bboxPx") if isinstance(bbox, dict) else None
    if not isinstance(values, list) or len(values) != 4:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in values]
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width - 1, x1 + margin)
    y1 = min(height - 1, y1 + margin)
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def bbox_frame_map(bbox_manifest: dict) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    for frame in bbox_manifest.get("frames") or []:
        try:
            mapping[int(frame.get("frameOrdinal"))] = frame
        except Exception:
            continue
    return mapping


def load_valid_bbox_manifest(
    run_key: str,
    manifest: dict,
    manifest_path: Path,
    mask_signature: str,
    review: dict,
    bbox_output_root: Path,
) -> tuple[dict, Path, str]:
    bbox_manifest_path = bbox_run_root(run_key, bbox_output_root) / "bbox_manifest.json"
    if not bbox_manifest_path.exists():
        raise ValueError("bbox manifest is required before corners can be built; build bboxes first")
    bbox_manifest = read_json(bbox_manifest_path)
    expected_precompute = repo_url(manifest_path)
    expected_bbox_signature = bbox_source_signature(
        manifest,
        manifest_path,
        bbox_settings_from_review(review),
        mask_signature,
    )
    if str(bbox_manifest.get("runKey") or "") != run_key:
        raise ValueError("bbox manifest run changed; rebuild bboxes first")
    if str(bbox_manifest.get("precomputeManifest") or "") != expected_precompute:
        raise ValueError("bbox manifest precompute changed; rebuild bboxes first")
    if str(bbox_manifest.get("sourceSignature") or "") != expected_bbox_signature:
        raise ValueError("bbox manifest is stale; rebuild bboxes first")
    if not bbox_manifest.get("complete"):
        raise ValueError("bbox manifest is partial; rebuild bboxes first")
    source_frames = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    if int(bbox_manifest.get("frameCount") or 0) < source_frames:
        raise ValueError("bbox manifest does not cover the full run; rebuild bboxes first")
    return bbox_manifest, bbox_manifest_path, str(bbox_manifest.get("sourceSignature") or expected_bbox_signature)


def component_mask_for_bbox(selected: np.ndarray, component_labels: np.ndarray, bbox: dict, width: int, height: int) -> tuple[np.ndarray, int]:
    bounds = bbox_bounds(bbox, width, height)
    empty = np.zeros_like(selected, dtype=bool)
    if bounds is None:
        return empty, 0
    x0, y0, x1, y1 = bounds
    roi_labels = component_labels[y0 : y1 + 1, x0 : x1 + 1]
    positive = roi_labels[roi_labels > 0]
    if positive.size == 0:
        return empty, 0
    labels, counts = np.unique(positive, return_counts=True)
    target = int(bbox.get("pixelCount") or 0)
    if target > 0:
        best_index = min(range(len(labels)), key=lambda index: (abs(int(counts[index]) - target), -int(counts[index])))
    else:
        best_index = int(np.argmax(counts))
    label = int(labels[best_index])
    return component_labels == label, label


def detect_hull_corners_for_bbox_type(
    cv2,
    component_mask: np.ndarray,
    component_labels: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    bbox: dict,
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
    type_key: str,
) -> list[dict]:
    bounds = bbox_bounds(bbox, width, height, int(settings["radiusPx"]))
    if bounds is None:
        return []
    x0, y0, x1, y1 = bounds
    crop = component_mask[y0 : y1 + 1, x0 : x1 + 1]
    if not np.any(crop):
        return []
    corners = []
    for candidate in raw_corner_candidates(
        cv2,
        crop,
        settings,
        offset=(x0, y0),
        retrieval_mode=cv2.RETR_EXTERNAL,
        role_override="bbox-hull",
    ):
        corner = classify_corner(
            component_mask,
            layer_hits,
            classes,
            candidate["point"],
            frame_ordinal,
            int(candidate["contourIndex"]),
            str(candidate["contourRole"]),
            str(candidate["source"]),
            component_labels,
            settings,
            width,
            height,
            candidate.get("angleDeg"),
            candidate.get("directionDeg"),
        )
        if not corner:
            continue
        target_position = str(settings.get("targetPosition") or "")
        if target_position in {"inside", "outside"} and corner_mask_position(corner) != target_position:
            continue
        corner["category"] = "hull"
        corner["cornerType"] = type_key
        corner["bboxId"] = bbox.get("bboxId")
        corners.append(corner)
    return merge_corners(corners, settings, f"{bbox.get('bboxId')}-hull-{type_key}", "maxCornersPerBbox")


def detect_hull_corners_for_bbox(
    cv2,
    component_mask: np.ndarray,
    component_labels: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    bbox: dict,
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> list[dict]:
    type_settings = settings.get("types") if isinstance(settings.get("types"), dict) else {}
    if type_settings:
        corners = []
        for type_key in ("white", "black"):
            corners.extend(
                detect_hull_corners_for_bbox_type(
                    cv2,
                    component_mask,
                    component_labels,
                    layer_hits,
                    classes,
                    bbox,
                    frame_ordinal,
                    width,
                    height,
                    type_settings[type_key],
                    type_key,
                )
            )
        return corners

    legacy = dict(settings.get("hull") or settings)
    legacy["targetPosition"] = ""
    return detect_hull_corners_for_bbox_type(
        cv2,
        component_mask,
        component_labels,
        layer_hits,
        classes,
        bbox,
        frame_ordinal,
        width,
        height,
        legacy,
        "legacy",
    )


def internal_void_components(cv2, component_mask: np.ndarray, bbox: dict, width: int, height: int, settings: dict) -> list[dict]:
    bounds = bbox_bounds(bbox, width, height)
    if bounds is None:
        return []
    x0, y0, x1, y1 = bounds
    mask_roi = component_mask[y0 : y1 + 1, x0 : x1 + 1]
    if mask_roi.size == 0:
        return []
    empty = ~mask_roi
    if not np.any(empty):
        return []
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(empty.astype(np.uint8), 8)
    border_labels = set()
    if labels.shape[0] > 0:
        border_labels.update(int(value) for value in labels[0, :] if int(value) > 0)
        border_labels.update(int(value) for value in labels[-1, :] if int(value) > 0)
    if labels.shape[1] > 0:
        border_labels.update(int(value) for value in labels[:, 0] if int(value) > 0)
        border_labels.update(int(value) for value in labels[:, -1] if int(value) > 0)

    min_area = int(settings["minAreaPx"])
    voids = []
    for label in range(1, int(label_count)):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        lx = int(stats[label, cv2.CC_STAT_LEFT])
        ly = int(stats[label, cv2.CC_STAT_TOP])
        lw = int(stats[label, cv2.CC_STAT_WIDTH])
        lh = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[label][0])
        cy = float(centroids[label][1])
        voids.append(
            {
                "label": label,
                "mask": labels == label,
                "areaPx": area,
                "bboxPx": [x0 + lx, y0 + ly, x0 + lx + lw - 1, y0 + ly + lh - 1],
                "bboxUv": [
                    clean_float((x0 + lx) / width, 6),
                    clean_float((y0 + ly) / height, 6),
                    clean_float((x0 + lx + lw) / width, 6),
                    clean_float((y0 + ly + lh) / height, 6),
                ],
                "centerPx": [clean_float(x0 + cx), clean_float(y0 + cy)],
                "centerUv": [clean_float((x0 + cx) / width, 6), clean_float((y0 + cy) / height, 6)],
                "offset": (x0, y0),
            }
        )
    voids.sort(key=lambda item: (-int(item["areaPx"]), item["bboxPx"][1], item["bboxPx"][0]))
    return voids


def detect_voids_for_bbox_type(
    cv2,
    selected: np.ndarray,
    component_mask: np.ndarray,
    component_labels: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    bbox: dict,
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
    type_key: str,
) -> list[dict]:
    voids_out = []
    bbox_id = str(bbox.get("bboxId") or f"bbox-{frame_ordinal:06d}")
    bounds = bbox_bounds(bbox, width, height)
    if bounds is None:
        return []
    bx0, by0, bx1, by1 = bounds
    selected_roi = selected[by0 : by1 + 1, bx0 : bx1 + 1]
    for index, void in enumerate(internal_void_components(cv2, component_mask, bbox, width, height, settings), start=1):
        void_id = f"{bbox_id}-{type_key}-void-{index:03d}"
        corners = []
        offset = void["offset"]
        for candidate in raw_corner_candidates(
            cv2,
            void["mask"],
            settings,
            offset=offset,
            retrieval_mode=cv2.RETR_EXTERNAL,
            role_override="internal-void",
        ):
            corner = classify_corner(
                component_mask,
                layer_hits,
                classes,
                candidate["point"],
                frame_ordinal,
                int(candidate["contourIndex"]),
                str(candidate["contourRole"]),
                str(candidate["source"]),
                component_labels,
                settings,
                width,
                height,
                candidate.get("angleDeg"),
                candidate.get("directionDeg"),
            )
            if not corner:
                continue
            target_position = str(settings.get("targetPosition") or "")
            if target_position in {"inside", "outside"} and corner_mask_position(corner) != target_position:
                continue
            corner["category"] = "void"
            corner["cornerType"] = type_key
            corner["bboxId"] = bbox_id
            corner["voidId"] = void_id
            corner["isInternalVoid"] = True
            corners.append(corner)
        merged = merge_corners(corners, settings, void_id, "maxCornersPerVoid")
        masked_pixels = int(np.count_nonzero(selected_roi & void["mask"]))
        voids_out.append(
            {
                "voidId": void_id,
                "bboxId": bbox_id,
                "isInternalVoid": True,
                "areaPx": int(void["areaPx"]),
                "bboxPx": void["bboxPx"],
                "bboxUv": void["bboxUv"],
                "centerPx": void["centerPx"],
                "centerUv": void["centerUv"],
                "selectedPixelCount": masked_pixels,
                "cornerCount": len(merged),
                "maskInsideCount": sum(1 for item in merged if corner_mask_position(item) == "inside"),
                "maskOutsideCount": sum(1 for item in merged if corner_mask_position(item) == "outside"),
                "maskWedgeCount": sum(1 for item in merged if corner_mask_position(item) == "inside"),
                "voidWedgeCount": sum(1 for item in merged if corner_mask_position(item) == "outside"),
                "ambiguousCount": sum(1 for item in merged if corner_mask_position(item) == "ambiguous"),
                "voidCorners": merged,
            }
        )
    return voids_out


def recompute_void_corner_counts(void_entry: dict) -> None:
    corners = list(void_entry.get("voidCorners") or [])
    void_entry["cornerCount"] = len(corners)
    void_entry["maskInsideCount"] = sum(1 for item in corners if corner_mask_position(item) == "inside")
    void_entry["maskOutsideCount"] = sum(1 for item in corners if corner_mask_position(item) == "outside")
    void_entry["maskWedgeCount"] = void_entry["maskInsideCount"]
    void_entry["voidWedgeCount"] = void_entry["maskOutsideCount"]
    void_entry["ambiguousCount"] = sum(1 for item in corners if corner_mask_position(item) == "ambiguous")


def combine_void_type_entries(void_entries: list[dict], bbox_id: str) -> list[dict]:
    grouped: dict[tuple[int, int, int, int], dict] = {}
    for entry in void_entries:
        bbox_values = entry.get("bboxPx") if isinstance(entry.get("bboxPx"), list) else None
        if not bbox_values or len(bbox_values) != 4:
            continue
        key = tuple(int(round(float(value))) for value in bbox_values)
        target = grouped.get(key)
        if target is None:
            target = {**entry, "voidCorners": []}
            grouped[key] = target
        target["voidCorners"].extend(list(entry.get("voidCorners") or []))
        target["selectedPixelCount"] = max(int(target.get("selectedPixelCount") or 0), int(entry.get("selectedPixelCount") or 0))
        target["areaPx"] = max(int(target.get("areaPx") or 0), int(entry.get("areaPx") or 0))

    merged = sorted(grouped.values(), key=lambda item: (-int(item.get("areaPx") or 0), item.get("bboxPx", [0, 0])[1], item.get("bboxPx", [0, 0])[0]))
    for index, entry in enumerate(merged, start=1):
        void_id = f"{bbox_id}-void-{index:03d}"
        entry["voidId"] = void_id
        for corner_index, corner in enumerate(entry.get("voidCorners") or [], start=1):
            corner["voidId"] = void_id
            corner["cornerId"] = f"{void_id}-{corner.get('cornerType', 'corner')}-{corner_index:03d}"
        recompute_void_corner_counts(entry)
    return merged


def detect_voids_for_bbox(
    cv2,
    selected: np.ndarray,
    component_mask: np.ndarray,
    component_labels: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    bbox: dict,
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> list[dict]:
    type_settings = settings.get("types") if isinstance(settings.get("types"), dict) else {}
    bbox_id = str(bbox.get("bboxId") or f"bbox-{frame_ordinal:06d}")
    if type_settings:
        void_entries = []
        for type_key in ("orange", "green"):
            void_entries.extend(
                detect_voids_for_bbox_type(
                    cv2,
                    selected,
                    component_mask,
                    component_labels,
                    layer_hits,
                    classes,
                    bbox,
                    frame_ordinal,
                    width,
                    height,
                    type_settings[type_key],
                    type_key,
                )
            )
        return combine_void_type_entries(void_entries, bbox_id)

    legacy = dict(settings.get("void") or settings)
    legacy["targetPosition"] = ""
    return detect_voids_for_bbox_type(
        cv2,
        selected,
        component_mask,
        component_labels,
        layer_hits,
        classes,
        bbox,
        frame_ordinal,
        width,
        height,
        legacy,
        "legacy",
    )


def detect_bbox_corner_group(
    cv2,
    selected: np.ndarray,
    layer_hits: np.ndarray,
    classes: list[dict],
    component_labels: np.ndarray,
    bbox: dict,
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> dict:
    component_mask, component_label = component_mask_for_bbox(selected, component_labels, bbox, width, height)
    hull_corners = detect_hull_corners_for_bbox(
        cv2,
        component_mask,
        component_labels,
        layer_hits,
        classes,
        bbox,
        frame_ordinal,
        width,
        height,
        settings,
    )
    voids = detect_voids_for_bbox(
        cv2,
        selected,
        component_mask,
        component_labels,
        layer_hits,
        classes,
        bbox,
        frame_ordinal,
        width,
        height,
        settings,
    )
    void_corner_count = sum(int(item.get("cornerCount") or 0) for item in voids)
    hull_mask_inside_count = sum(1 for item in hull_corners if corner_mask_position(item) == "inside")
    hull_mask_outside_count = sum(1 for item in hull_corners if corner_mask_position(item) == "outside")
    void_mask_inside_count = sum(int(item.get("maskInsideCount") or item.get("maskWedgeCount") or 0) for item in voids)
    void_mask_outside_count = sum(int(item.get("maskOutsideCount") or item.get("voidWedgeCount") or 0) for item in voids)
    ambiguous_count = sum(1 for item in hull_corners if corner_mask_position(item) == "ambiguous") + sum(int(item.get("ambiguousCount") or 0) for item in voids)
    white_corner_count = sum(1 for item in hull_corners if str(item.get("cornerType") or "") == "white" or (not item.get("cornerType") and corner_mask_position(item) == "inside"))
    black_corner_count = sum(1 for item in hull_corners if str(item.get("cornerType") or "") == "black" or (not item.get("cornerType") and corner_mask_position(item) == "outside"))
    orange_corner_count = sum(1 for void in voids for item in list(void.get("voidCorners") or []) if str(item.get("cornerType") or "") == "orange" or (not item.get("cornerType") and corner_mask_position(item) == "inside"))
    green_corner_count = sum(1 for void in voids for item in list(void.get("voidCorners") or []) if str(item.get("cornerType") or "") == "green" or (not item.get("cornerType") and corner_mask_position(item) == "outside"))
    return {
        "bboxId": bbox.get("bboxId"),
        "frameOrdinal": frame_ordinal,
        "bboxPx": bbox.get("bboxPx"),
        "bboxUv": bbox.get("bboxUv"),
        "pixelCount": bbox.get("pixelCount"),
        "componentLabel": component_label,
        "hullCornerCount": len(hull_corners),
        "hullMaskInsideCount": hull_mask_inside_count,
        "hullMaskOutsideCount": hull_mask_outside_count,
        "hullMaskWedgeCount": hull_mask_inside_count,
        "hullVoidWedgeCount": hull_mask_outside_count,
        "maskWedgeCount": hull_mask_inside_count + void_mask_inside_count,
        "voidCount": len(voids),
        "voidCornerCount": void_corner_count,
        "voidMaskInsideCount": void_mask_inside_count,
        "voidMaskOutsideCount": void_mask_outside_count,
        "voidMaskWedgeCount": void_mask_inside_count,
        "voidVoidWedgeCount": void_mask_outside_count,
        "voidWedgeCount": void_mask_outside_count,
        "whiteCornerCount": white_corner_count,
        "blackCornerCount": black_corner_count,
        "orangeCornerCount": orange_corner_count,
        "greenCornerCount": green_corner_count,
        "ambiguousCount": ambiguous_count,
        "hullCorners": hull_corners,
        "voids": voids,
    }


def corner_counts_for_bbox_groups(groups: list[dict]) -> dict:
    hull_corner_count = sum(int(item.get("hullCornerCount") or 0) for item in groups)
    void_count = sum(int(item.get("voidCount") or 0) for item in groups)
    void_corner_count = sum(int(item.get("voidCornerCount") or 0) for item in groups)
    ambiguous_count = sum(int(item.get("ambiguousCount") or 0) for item in groups)
    mask_wedge_count = sum(int(item.get("maskWedgeCount") or 0) for item in groups)
    void_wedge_count = sum(int(item.get("voidWedgeCount") or 0) for item in groups)
    hull_mask_inside_count = sum(int(item.get("hullMaskInsideCount") or item.get("hullMaskWedgeCount") or 0) for item in groups)
    hull_mask_outside_count = sum(int(item.get("hullMaskOutsideCount") or item.get("hullVoidWedgeCount") or 0) for item in groups)
    void_mask_inside_count = sum(int(item.get("voidMaskInsideCount") or item.get("voidMaskWedgeCount") or 0) for item in groups)
    void_mask_outside_count = sum(int(item.get("voidMaskOutsideCount") or item.get("voidVoidWedgeCount") or 0) for item in groups)
    white_corner_count = sum(int(item.get("whiteCornerCount") or 0) for item in groups)
    black_corner_count = sum(int(item.get("blackCornerCount") or 0) for item in groups)
    orange_corner_count = sum(int(item.get("orangeCornerCount") or 0) for item in groups)
    green_corner_count = sum(int(item.get("greenCornerCount") or 0) for item in groups)
    return {
        "cornerCount": hull_corner_count + void_corner_count,
        "hullCornerCount": hull_corner_count,
        "voidCount": void_count,
        "voidCornerCount": void_corner_count,
        "maskWedgeCount": mask_wedge_count,
        "voidWedgeCount": void_wedge_count,
        "hullMaskInsideCount": hull_mask_inside_count,
        "hullMaskOutsideCount": hull_mask_outside_count,
        "voidMaskInsideCount": void_mask_inside_count,
        "voidMaskOutsideCount": void_mask_outside_count,
        "hullMaskWedgeCount": hull_mask_inside_count,
        "hullVoidWedgeCount": hull_mask_outside_count,
        "voidMaskWedgeCount": void_mask_inside_count,
        "voidVoidWedgeCount": void_mask_outside_count,
        "whiteCornerCount": white_corner_count,
        "blackCornerCount": black_corner_count,
        "orangeCornerCount": orange_corner_count,
        "greenCornerCount": green_corner_count,
        "ambiguousCount": ambiguous_count,
        "bboxRegionCount": len(groups),
    }


def build_mask_corners(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for mask corner detection: {error}") from error

    try:
        raw_settings = args.corner_settings
    except AttributeError:
        raw_settings = None
    if raw_settings is None:
        legacy_radius = getattr(args, "radius_px", None)
        legacy_min_angle = getattr(args, "min_angle_deg", None)
        legacy_max_angle = getattr(args, "max_angle_deg", None)
        legacy_min_support = getattr(args, "min_support_pixels", None)
        legacy_epsilon = getattr(args, "contour_epsilon_px", None)
        legacy_distance = getattr(args, "min_distance_px", None)
        def arg_value(name: str, fallback):
            value = getattr(args, name, None)
            return fallback if value is None else value

        raw_settings = {
            "hull": {
                "radiusPx": arg_value("hull_radius_px", legacy_radius if legacy_radius is not None else DEFAULT_RADIUS_PX),
                "minAngleDeg": arg_value("hull_min_angle_deg", legacy_min_angle if legacy_min_angle is not None else DEFAULT_MIN_ANGLE_DEG),
                "maxAngleDeg": arg_value("hull_max_angle_deg", legacy_max_angle if legacy_max_angle is not None else DEFAULT_MAX_ANGLE_DEG),
                "minSupportPixels": arg_value("hull_min_support_pixels", legacy_min_support if legacy_min_support is not None else DEFAULT_MIN_SUPPORT_PIXELS),
                "contourEpsilonPx": arg_value("hull_contour_epsilon_px", legacy_epsilon if legacy_epsilon is not None else DEFAULT_CONTOUR_EPSILON_PX),
                "minDistancePx": arg_value("hull_min_distance_px", legacy_distance if legacy_distance is not None else DEFAULT_MIN_DISTANCE_PX),
                "maxCornersPerBbox": arg_value("hull_max_corners_per_bbox", DEFAULT_MAX_HULL_CORNERS_PER_BBOX),
                "minMaskInsidePoints": arg_value("hull_white_min_points", 0),
                "maxMaskInsidePoints": arg_value("hull_white_max_points", arg_value("hull_max_corners_per_bbox", DEFAULT_MAX_HULL_CORNERS_PER_BBOX)),
                "minMaskOutsidePoints": arg_value("hull_black_min_points", 0),
                "maxMaskOutsidePoints": arg_value("hull_black_max_points", arg_value("hull_max_corners_per_bbox", DEFAULT_MAX_HULL_CORNERS_PER_BBOX)),
            },
            "void": {
                "minAreaPx": arg_value("void_min_area_px", DEFAULT_VOID_MIN_AREA_PX),
                "radiusPx": arg_value("void_radius_px", legacy_radius if legacy_radius is not None else DEFAULT_RADIUS_PX),
                "minAngleDeg": arg_value("void_min_angle_deg", legacy_min_angle if legacy_min_angle is not None else DEFAULT_MIN_ANGLE_DEG),
                "maxAngleDeg": arg_value("void_max_angle_deg", legacy_max_angle if legacy_max_angle is not None else DEFAULT_MAX_ANGLE_DEG),
                "minSupportPixels": arg_value("void_min_support_pixels", legacy_min_support if legacy_min_support is not None else DEFAULT_MIN_SUPPORT_PIXELS),
                "contourEpsilonPx": arg_value("void_contour_epsilon_px", legacy_epsilon if legacy_epsilon is not None else DEFAULT_CONTOUR_EPSILON_PX),
                "minDistancePx": arg_value("void_min_distance_px", legacy_distance if legacy_distance is not None else DEFAULT_MIN_DISTANCE_PX),
                "maxCornersPerVoid": arg_value("void_max_corners_per_void", DEFAULT_MAX_VOID_CORNERS_PER_VOID),
                "minMaskInsidePoints": arg_value("void_orange_min_points", 0),
                "maxMaskInsidePoints": arg_value("void_orange_max_points", arg_value("void_max_corners_per_void", DEFAULT_MAX_VOID_CORNERS_PER_VOID)),
                "minMaskOutsidePoints": arg_value("void_green_min_points", 0),
                "maxMaskOutsidePoints": arg_value("void_green_max_points", arg_value("void_max_corners_per_void", DEFAULT_MAX_VOID_CORNERS_PER_VOID)),
            },
        }
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
        raise ValueError("corner build requires frame entries in the precompute manifest")

    settings = normalized_corner_settings(raw_settings)
    mask_context = build_mask_context(config, rules, review, manifest)
    bbox_output_root = getattr(args, "bbox_output_root", DEFAULT_BBOX_ROOT)
    bbox_manifest, bbox_manifest_path, bbox_signature = load_valid_bbox_manifest(
        run_key,
        manifest,
        manifest_path,
        mask_context["signature"],
        review,
        bbox_output_root,
    )
    bbox_frames = bbox_frame_map(bbox_manifest)
    source_signature = corner_source_signature(manifest, manifest_path, settings, mask_context["signature"], bbox_signature)

    start = max(0, int(args.start_frame or 0))
    end = frame_count
    if args.max_frames is not None:
        end = min(end, start + max(0, int(args.max_frames)))
    selected_frames = list(enumerate(frames_in[start:end], start=start))
    if not selected_frames:
        raise ValueError("selected corner frame range is empty")

    output_root = corner_run_root(run_key, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frames_out = []
    totals = {
        "cornerCount": 0,
        "hullCornerCount": 0,
        "voidCount": 0,
        "voidCornerCount": 0,
        "maskWedgeCount": 0,
        "voidWedgeCount": 0,
        "hullMaskInsideCount": 0,
        "hullMaskOutsideCount": 0,
        "voidMaskInsideCount": 0,
        "voidMaskOutsideCount": 0,
        "hullMaskWedgeCount": 0,
        "hullVoidWedgeCount": 0,
        "voidMaskWedgeCount": 0,
        "voidVoidWedgeCount": 0,
        "whiteCornerCount": 0,
        "blackCornerCount": 0,
        "orangeCornerCount": 0,
        "greenCornerCount": 0,
        "ambiguousCount": 0,
        "bboxRegionCount": 0,
    }
    total_selected_pixels = 0
    max_corners = 0
    max_bboxes = 0
    max_voids = 0
    for local_index, (frame_ordinal, frame_entry) in enumerate(selected_frames):
        bbox_frame = bbox_frames.get(frame_ordinal)
        if not bbox_frame:
            raise ValueError(f"bbox manifest has no frame entry for frame {frame_ordinal}")
        bboxes = list(bbox_frame.get("bboxes") or [])
        color_ids = read_frame_color_ids(frame_entry, width, height)
        layer_hits = mask_context["membership"][:, color_ids]
        selected = np.any(layer_hits, axis=0)
        selected_pixel_count = int(np.count_nonzero(selected))
        _, component_labels, _, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
        bbox_corner_groups = [
            detect_bbox_corner_group(
                cv2,
                selected,
                layer_hits,
                mask_context["classes"],
                component_labels,
                bbox,
                frame_ordinal,
                width,
                height,
                settings,
            )
            for bbox in bboxes
        ]
        counts = corner_counts_for_bbox_groups(bbox_corner_groups)
        for key, value in counts.items():
            totals[key] += value
        max_corners = max(max_corners, int(counts["cornerCount"]))
        max_bboxes = max(max_bboxes, len(bbox_corner_groups))
        max_voids = max(max_voids, int(counts["voidCount"]))
        total_selected_pixels += selected_pixel_count
        frames_out.append(
            {
                "frameOrdinal": frame_ordinal,
                "frameId": frame_entry.get("frameId") or frame_entry.get("rawIndex") or frame_entry.get("index"),
                "path": frame_entry.get("path") or frame_entry.get("sourcePath"),
                "cachePath": frame_entry.get("cachePath"),
                "selectedPixelCount": selected_pixel_count,
                **counts,
                "bboxCorners": bbox_corner_groups,
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-mask-corners",
                    "index": local_index + 1,
                    "total": len(selected_frames),
                    "frameOrdinal": frame_ordinal,
                    "selectedPixels": selected_pixel_count,
                    **counts,
                }
            )

    output = {
        "version": 2,
        "app": "vision_passthrough_review",
        "kind": "color-mask-bbox-corners",
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
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": bbox_signature,
        },
        "summary": {
            **totals,
            "maxCornersPerFrame": max_corners,
            "maxBboxesPerFrame": max_bboxes,
            "maxVoidsPerFrame": max_voids,
            "selectedPixelCount": total_selected_pixels,
            "meanCornersPerFrame": totals["cornerCount"] / len(frames_out) if frames_out else 0,
            "meanSelectedPixelsPerFrame": total_selected_pixels / len(frames_out) if frames_out else 0,
        },
        "frames": frames_out,
    }
    atomic_write_json(output_root / "corners_manifest.json", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bbox-output-root", type=Path, default=DEFAULT_BBOX_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--radius-px", type=int, default=None)
    parser.add_argument("--min-angle-deg", type=int, default=None)
    parser.add_argument("--max-angle-deg", type=int, default=None)
    parser.add_argument("--min-support-pixels", type=int, default=None)
    parser.add_argument("--contour-epsilon-px", type=float, default=None)
    parser.add_argument("--min-distance-px", type=int, default=None)
    parser.add_argument("--hull-radius-px", type=int, default=None)
    parser.add_argument("--hull-min-angle-deg", type=int, default=None)
    parser.add_argument("--hull-max-angle-deg", type=int, default=None)
    parser.add_argument("--hull-min-support-pixels", type=int, default=None)
    parser.add_argument("--hull-contour-epsilon-px", type=float, default=None)
    parser.add_argument("--hull-min-distance-px", type=int, default=None)
    parser.add_argument("--hull-max-corners-per-bbox", type=int, default=None)
    parser.add_argument("--hull-white-min-points", type=int, default=None)
    parser.add_argument("--hull-white-max-points", type=int, default=None)
    parser.add_argument("--hull-black-min-points", type=int, default=None)
    parser.add_argument("--hull-black-max-points", type=int, default=None)
    parser.add_argument("--void-min-area-px", type=int, default=None)
    parser.add_argument("--void-radius-px", type=int, default=None)
    parser.add_argument("--void-min-angle-deg", type=int, default=None)
    parser.add_argument("--void-max-angle-deg", type=int, default=None)
    parser.add_argument("--void-min-support-pixels", type=int, default=None)
    parser.add_argument("--void-contour-epsilon-px", type=float, default=None)
    parser.add_argument("--void-min-distance-px", type=int, default=None)
    parser.add_argument("--void-max-corners-per-void", type=int, default=None)
    parser.add_argument("--void-orange-min-points", type=int, default=None)
    parser.add_argument("--void-orange-max-points", type=int, default=None)
    parser.add_argument("--void-green-min-points", type=int, default=None)
    parser.add_argument("--void-green-max-points", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_mask_corners(args)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": repo_url(corner_run_root(result["runKey"], args.output_root) / "corners_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
