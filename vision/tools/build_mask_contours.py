#!/usr/bin/env python3
"""Build bbox-local contour hierarchy from decoded color-mask layers."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Callable

import numpy as np

from build_mask_bboxes import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_BBOX_ROOT,
    bbox_settings_from_review,
    bbox_source_signature,
    build_mask_context,
    concise_layer_summaries,
)
from build_mask_corners import (
    bbox_bounds,
    bbox_frame_map,
    component_mask_for_bbox,
    load_valid_bbox_manifest,
)
from build_vision_memory import (
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


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "mask_contours"
DEFAULT_MASK_SOURCE = "enabled-layers"
DEFAULT_MIN_OUTER_AREA_PX = 10
DEFAULT_MIN_VOID_AREA_PX = 20
DEFAULT_MAX_VOIDS_PER_BBOX = 12
DEFAULT_SIMPLIFY_EPSILON_PX = 1.5
DEFAULT_CLOSE_RADIUS_PX = 0
DEFAULT_OPEN_RADIUS_PX = 0
DEFAULT_NOTCH_PROXIMITY_PX = 4
DEFAULT_INCLUDE_SMALL_CONTOURS = False
MASK_SOURCES = {"enabled-layers", "002-only"}


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


def contour_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build contour artifacts without an active run key")
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


def normalized_contour_settings(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    mask_source = str(source.get("maskSource") or DEFAULT_MASK_SOURCE)
    if mask_source not in MASK_SOURCES:
        mask_source = DEFAULT_MASK_SOURCE
    return {
        "maskSource": mask_source,
        "minOuterAreaPx": numeric_setting(source, "minOuterAreaPx", DEFAULT_MIN_OUTER_AREA_PX, 1, 500000, integer=True),
        "minVoidAreaPx": numeric_setting(source, "minVoidAreaPx", DEFAULT_MIN_VOID_AREA_PX, 1, 500000, integer=True),
        "maxVoidsPerBbox": numeric_setting(source, "maxVoidsPerBbox", DEFAULT_MAX_VOIDS_PER_BBOX, 0, 200, integer=True),
        "simplifyEpsilonPx": numeric_setting(source, "simplifyEpsilonPx", DEFAULT_SIMPLIFY_EPSILON_PX, 0.0, 32.0),
        "closeRadiusPx": numeric_setting(source, "closeRadiusPx", DEFAULT_CLOSE_RADIUS_PX, 0, 32, integer=True),
        "openRadiusPx": numeric_setting(source, "openRadiusPx", DEFAULT_OPEN_RADIUS_PX, 0, 32, integer=True),
        "notchProximityPx": numeric_setting(source, "notchProximityPx", DEFAULT_NOTCH_PROXIMITY_PX, 0, 128, integer=True),
        "includeSmallContours": bool_setting(source.get("includeSmallContours"), DEFAULT_INCLUDE_SMALL_CONTOURS),
    }


def contour_settings_from_review(review: dict) -> dict:
    raw = review.get("contourHierarchy") if isinstance(review.get("contourHierarchy"), dict) else {}
    return normalized_contour_settings(raw)


def contour_source_signature(
    manifest: dict,
    manifest_path: Path,
    settings: dict,
    mask_signature: str,
    bbox_signature: str | None = None,
) -> str:
    return stable_hash(
        {
            "kind": "0720-color-mask-contour-hierarchy-v2-raw-points",
            "runKey": manifest.get("runKey"),
            "precomputeManifest": repo_url(manifest_path),
            "frameCount": int(manifest.get("frameCount") or len(manifest.get("frames") or [])),
            "width": int(manifest.get("width") or 640),
            "height": int(manifest.get("height") or 360),
            "settings": normalized_contour_settings(settings),
            "maskSignature": mask_signature,
            "bboxSignature": bbox_signature,
        }
    )


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def point_px_list(points: np.ndarray, offset_x: int, offset_y: int) -> list[list[float]]:
    raw = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [[clean_float(x + offset_x), clean_float(y + offset_y)] for x, y in raw]


def point_uv_list(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[clean_float(x / width, 6), clean_float(y / height, 6)] for x, y in points]


def uv_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> list[float]:
    return [
        clean_float(x0 / width, 6),
        clean_float(y0 / height, 6),
        clean_float((x1 + 1) / width, 6),
        clean_float((y1 + 1) / height, 6),
    ]


def contour_depth(rows, index: int) -> int:
    depth = 0
    parent = int(rows[index][3])
    guard = 0
    while parent >= 0 and guard < len(rows):
        depth += 1
        parent = int(rows[parent][3])
        guard += 1
    return depth


def role_for_depth(depth: int) -> str:
    if depth == 0:
        return "outer"
    if depth % 2 == 1:
        return "void"
    return "nested-island"


def apply_morphology(cv2, mask: np.ndarray, settings: dict) -> np.ndarray:
    working = mask.astype(np.uint8, copy=True)
    close_radius = int(settings["closeRadiusPx"])
    open_radius = int(settings["openRadiusPx"])
    if close_radius > 0:
        size = close_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel)
    if open_radius > 0:
        size = open_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        working = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel)
    return working.astype(bool)


def contour_feature(
    cv2,
    contour: np.ndarray,
    contour_index: int,
    role: str,
    depth: int,
    parent_index: int,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
    settings: dict,
    bbox_id: str,
    frame_ordinal: int,
) -> dict | None:
    raw_points = contour.reshape(-1, 2)
    if raw_points.shape[0] < 3:
        return None
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = float(settings["simplifyEpsilonPx"])
    simplified = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
    sx, sy, sw, sh = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = offset_x + moments["m10"] / moments["m00"]
        cy = offset_y + moments["m01"] / moments["m00"]
    else:
        cx = offset_x + sx + sw * 0.5
        cy = offset_y + sy + sh * 0.5
    points_px = point_px_list(simplified, offset_x, offset_y)
    raw_points_px = point_px_list(raw_points, offset_x, offset_y)
    px0 = int(offset_x + sx)
    py0 = int(offset_y + sy)
    px1 = int(offset_x + sx + sw - 1)
    py1 = int(offset_y + sy + sh - 1)
    return {
        "contourId": f"{bbox_id}-contour-{contour_index + 1:03d}",
        "bboxId": bbox_id,
        "frameOrdinal": frame_ordinal,
        "role": role,
        "depth": int(depth),
        "parentContourIndex": int(parent_index),
        "parentContourId": None,
        "rawPointCount": int(raw_points.shape[0]),
        "simplifiedPointCount": int(simplified.reshape(-1, 2).shape[0]),
        "areaPx": clean_float(area, 3),
        "perimeterPx": clean_float(perimeter, 3),
        "bboxPx": [px0, py0, px1, py1],
        "bboxUv": uv_bbox(px0, py0, px1, py1, width, height),
        "centerPx": [clean_float(cx), clean_float(cy)],
        "centerUv": [clean_float(cx / width, 6), clean_float(cy / height, 6)],
        "rawPointsPx": raw_points_px,
        "pointsPx": points_px,
        "pointsUv": point_uv_list(points_px, width, height),
    }


def contour_passes_filter(feature: dict, settings: dict) -> bool:
    role = str(feature.get("role") or "")
    area = float(feature.get("areaPx") or 0)
    if role == "outer":
        return area >= float(settings["minOuterAreaPx"])
    return area >= float(settings["minVoidAreaPx"])


def bbox_edge_distance(feature: dict, bbox_values: list[int | float]) -> float:
    fx0, fy0, fx1, fy1 = [float(value) for value in feature.get("bboxPx") or [0, 0, 0, 0]]
    bx0, by0, bx1, by1 = [float(value) for value in bbox_values]
    return min(abs(fx0 - bx0), abs(fy0 - by0), abs(bx1 - fx1), abs(by1 - fy1))


def contour_edge_distance(cv2, outer_contour: np.ndarray | None, contour: np.ndarray) -> float | None:
    if outer_contour is None:
        return None
    points = contour.reshape(-1, 2)
    if points.size == 0:
        return None
    distances = [abs(float(cv2.pointPolygonTest(outer_contour, (float(x), float(y)), True))) for x, y in points]
    return min(distances) if distances else None


def contour_hierarchy_for_bbox(
    cv2,
    selected: np.ndarray,
    source_mask: np.ndarray,
    component_labels: np.ndarray,
    bbox: dict,
    frame_ordinal: int,
    width: int,
    height: int,
    settings: dict,
) -> dict:
    bbox_id = str(bbox.get("bboxId") or f"bbox-{frame_ordinal:06d}")
    component_mask, component_label = component_mask_for_bbox(selected, component_labels, bbox, width, height)
    analysis_mask = component_mask & source_mask
    bounds = bbox_bounds(bbox, width, height)
    empty = {
        "bboxId": bbox_id,
        "frameOrdinal": frame_ordinal,
        "bboxPx": bbox.get("bboxPx"),
        "bboxUv": bbox.get("bboxUv"),
        "pixelCount": bbox.get("pixelCount"),
        "componentLabel": component_label,
        "maskSource": settings["maskSource"],
        "selectedPixelCount": 0,
        "contourCount": 0,
        "outer": None,
        "additionalOuters": [],
        "voids": [],
        "nestedIslands": [],
        "notchCandidates": [],
        "overlapCandidates": [],
        "rejectedContours": [],
        "metrics": {"outerCount": 0, "voidCount": 0, "nestedIslandCount": 0},
    }
    if bounds is None or not np.any(analysis_mask):
        return empty
    x0, y0, x1, y1 = bounds
    roi = apply_morphology(cv2, analysis_mask[y0 : y1 + 1, x0 : x1 + 1], settings)
    selected_pixel_count = int(np.count_nonzero(roi))
    if selected_pixel_count <= 0:
        return empty

    contours, hierarchy = cv2.findContours((roi.astype(np.uint8) * 255), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        empty["selectedPixelCount"] = selected_pixel_count
        return empty
    rows = hierarchy[0]
    features_by_index: dict[int, dict] = {}
    rejected = []
    for index, contour in enumerate(contours):
        depth = contour_depth(rows, index)
        parent = int(rows[index][3])
        role = role_for_depth(depth)
        feature = contour_feature(cv2, contour, index, role, depth, parent, x0, y0, width, height, settings, bbox_id, frame_ordinal)
        if not feature:
            continue
        if contour_passes_filter(feature, settings) or bool(settings["includeSmallContours"]):
            features_by_index[index] = feature
        else:
            feature["rejectedReason"] = "below-area-threshold"
            rejected.append(feature)

    for index, feature in features_by_index.items():
        parent = int(feature.get("parentContourIndex", -1))
        if parent >= 0 and parent in features_by_index:
            feature["parentContourId"] = features_by_index[parent].get("contourId")
    roots = sorted(
        [feature for feature in features_by_index.values() if feature.get("role") == "outer"],
        key=lambda item: (-float(item.get("areaPx") or 0), item.get("bboxPx", [0, 0])[1], item.get("bboxPx", [0, 0])[0]),
    )
    outer = roots[0] if roots else None
    additional_outers = roots[1:]
    voids = sorted(
        [feature for feature in features_by_index.values() if feature.get("role") == "void"],
        key=lambda item: (-float(item.get("areaPx") or 0), item.get("bboxPx", [0, 0])[1], item.get("bboxPx", [0, 0])[0]),
    )
    if int(settings["maxVoidsPerBbox"]) > 0:
        voids = voids[: int(settings["maxVoidsPerBbox"])]
    else:
        voids = []
    nested = sorted(
        [feature for feature in features_by_index.values() if feature.get("role") == "nested-island"],
        key=lambda item: (-float(item.get("areaPx") or 0), item.get("bboxPx", [0, 0])[1], item.get("bboxPx", [0, 0])[0]),
    )

    outer_contour = None
    if outer:
        outer_index = int(str(outer.get("contourId", "0")).rsplit("-", 1)[-1]) - 1
        if 0 <= outer_index < len(contours):
            outer_contour = contours[outer_index]
    proximity = float(settings["notchProximityPx"])
    notch_candidates = []
    for void in voids:
        contour_index = int(str(void.get("contourId", "0")).rsplit("-", 1)[-1]) - 1
        contour_distance = contour_edge_distance(cv2, outer_contour, contours[contour_index]) if 0 <= contour_index < len(contours) else None
        edge_distance = bbox_edge_distance(void, bbox.get("bboxPx") or [x0, y0, x1, y1])
        reasons = []
        if contour_distance is not None and contour_distance <= proximity:
            reasons.append("near-outer-boundary")
        if edge_distance <= proximity:
            reasons.append("near-bbox-edge")
        if reasons:
            notch_candidates.append({**void, "role": "notch-candidate", "reasons": reasons, "outerDistancePx": clean_float(contour_distance or 0), "bboxEdgeDistancePx": clean_float(edge_distance)})

    overlap_candidates = []
    if len(voids) > 1:
        overlap_candidates.append({"type": "multi-void", "voidCount": len(voids), "bboxId": bbox_id})
    if nested:
        overlap_candidates.append({"type": "nested-island", "nestedIslandCount": len(nested), "bboxId": bbox_id})
    if notch_candidates:
        overlap_candidates.append({"type": "boundary-notch", "notchCount": len(notch_candidates), "bboxId": bbox_id})

    return {
        **empty,
        "selectedPixelCount": selected_pixel_count,
        "contourCount": len(features_by_index),
        "outer": outer,
        "additionalOuters": additional_outers,
        "voids": voids,
        "nestedIslands": nested,
        "notchCandidates": notch_candidates,
        "overlapCandidates": overlap_candidates,
        "rejectedContours": rejected if bool(settings["includeSmallContours"]) else [],
        "metrics": {
            "outerCount": len(roots),
            "voidCount": len(voids),
            "nestedIslandCount": len(nested),
            "notchCandidateCount": len(notch_candidates),
            "overlapCandidateCount": len(overlap_candidates),
        },
    }


def source_mask_for_settings(layer_hits: np.ndarray, classes: list[dict], settings: dict) -> np.ndarray:
    if settings["maskSource"] == "002-only":
        index = next((idx for idx, item in enumerate(classes) if str(item.get("prefix")) == "002"), -1)
        if index < 0:
            return np.zeros(layer_hits.shape[1:], dtype=bool)
        return layer_hits[index].astype(bool)
    return np.any(layer_hits, axis=0)


def contour_counts_for_objects(objects: list[dict]) -> dict:
    return {
        "objectCount": len(objects),
        "contourCount": sum(int(item.get("contourCount") or 0) for item in objects),
        "outerCount": sum(int(item.get("metrics", {}).get("outerCount") or 0) for item in objects),
        "voidCount": sum(int(item.get("metrics", {}).get("voidCount") or 0) for item in objects),
        "nestedIslandCount": sum(int(item.get("metrics", {}).get("nestedIslandCount") or 0) for item in objects),
        "notchCandidateCount": sum(int(item.get("metrics", {}).get("notchCandidateCount") or 0) for item in objects),
        "overlapCandidateCount": sum(int(item.get("metrics", {}).get("overlapCandidateCount") or 0) for item in objects),
    }


def build_mask_contours(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for mask contour hierarchy: {error}") from error

    raw_settings = getattr(args, "contour_settings", None)
    settings = normalized_contour_settings(raw_settings)
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
        raise ValueError("contour build requires frame entries in the precompute manifest")

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
    source_signature = contour_source_signature(manifest, manifest_path, settings, mask_context["signature"], bbox_signature)

    start = max(0, int(args.start_frame or 0))
    end = frame_count
    if args.max_frames is not None:
        end = min(end, start + max(0, int(args.max_frames)))
    selected_frames = list(enumerate(frames_in[start:end], start=start))
    if not selected_frames:
        raise ValueError("selected contour frame range is empty")

    output_root = contour_run_root(run_key, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frames_out = []
    totals = {
        "objectCount": 0,
        "contourCount": 0,
        "outerCount": 0,
        "voidCount": 0,
        "nestedIslandCount": 0,
        "notchCandidateCount": 0,
        "overlapCandidateCount": 0,
    }
    total_selected_pixels = 0
    max_voids = 0
    for local_index, (frame_ordinal, frame_entry) in enumerate(selected_frames):
        bbox_frame = bbox_frames.get(frame_ordinal)
        if not bbox_frame:
            raise ValueError(f"bbox manifest has no frame entry for frame {frame_ordinal}")
        bboxes = list(bbox_frame.get("bboxes") or [])
        color_ids = read_frame_color_ids(frame_entry, width, height)
        layer_hits = mask_context["membership"][:, color_ids]
        enabled_selected = np.any(layer_hits, axis=0)
        source_mask = source_mask_for_settings(layer_hits, mask_context["classes"], settings)
        _, component_labels, _, _ = cv2.connectedComponentsWithStats(enabled_selected.astype(np.uint8), 8)
        objects = [
            contour_hierarchy_for_bbox(
                cv2,
                enabled_selected,
                source_mask,
                component_labels,
                bbox,
                frame_ordinal,
                width,
                height,
                settings,
            )
            for bbox in bboxes
        ]
        counts = contour_counts_for_objects(objects)
        for key, value in counts.items():
            totals[key] += value
        max_voids = max(max_voids, counts["voidCount"])
        selected_pixel_count = int(np.count_nonzero(source_mask))
        total_selected_pixels += selected_pixel_count
        frames_out.append(
            {
                "frameOrdinal": frame_ordinal,
                "frameId": frame_entry.get("frameId") or frame_entry.get("rawIndex") or frame_entry.get("index"),
                "path": frame_entry.get("path") or frame_entry.get("sourcePath"),
                "cachePath": frame_entry.get("cachePath"),
                "selectedPixelCount": selected_pixel_count,
                **counts,
                "objects": objects,
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-mask-contours",
                    "index": local_index + 1,
                    "total": len(selected_frames),
                    "frameOrdinal": frame_ordinal,
                    "selectedPixels": selected_pixel_count,
                    **counts,
                }
            )

    output = {
        "version": 1,
        "app": "vision_passthrough_review",
        "kind": "color-mask-contour-hierarchy",
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
            "source": settings["maskSource"],
            "maskSignature": mask_context["signature"],
            "enabledPrefixes": mask_context["enabledPrefixes"],
            "layerSummaries": concise_layer_summaries(mask_context["summaries"]),
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": bbox_signature,
        },
        "summary": {
            **totals,
            "maxVoidsPerFrame": max_voids,
            "selectedPixelCount": total_selected_pixels,
            "meanContoursPerFrame": totals["contourCount"] / len(frames_out) if frames_out else 0,
            "meanSelectedPixelsPerFrame": total_selected_pixels / len(frames_out) if frames_out else 0,
        },
        "frames": frames_out,
    }
    atomic_write_json(output_root / "contours_manifest.json", output)
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
    parser.add_argument("--mask-source", default=DEFAULT_MASK_SOURCE, choices=sorted(MASK_SOURCES))
    parser.add_argument("--min-outer-area-px", type=int, default=DEFAULT_MIN_OUTER_AREA_PX)
    parser.add_argument("--min-void-area-px", type=int, default=DEFAULT_MIN_VOID_AREA_PX)
    parser.add_argument("--max-voids-per-bbox", type=int, default=DEFAULT_MAX_VOIDS_PER_BBOX)
    parser.add_argument("--simplify-epsilon-px", type=float, default=DEFAULT_SIMPLIFY_EPSILON_PX)
    parser.add_argument("--close-radius-px", type=int, default=DEFAULT_CLOSE_RADIUS_PX)
    parser.add_argument("--open-radius-px", type=int, default=DEFAULT_OPEN_RADIUS_PX)
    parser.add_argument("--notch-proximity-px", type=int, default=DEFAULT_NOTCH_PROXIMITY_PX)
    parser.add_argument("--include-small-contours", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.contour_settings = {
        "maskSource": args.mask_source,
        "minOuterAreaPx": args.min_outer_area_px,
        "minVoidAreaPx": args.min_void_area_px,
        "maxVoidsPerBbox": args.max_voids_per_bbox,
        "simplifyEpsilonPx": args.simplify_epsilon_px,
        "closeRadiusPx": args.close_radius_px,
        "openRadiusPx": args.open_radius_px,
        "notchProximityPx": args.notch_proximity_px,
        "includeSmallContours": args.include_small_contours,
    }
    result = build_mask_contours(args)
    print(
        f"wrote {result['frameCount']} contour frames, "
        f"{result['summary']['contourCount']} contours, "
        f"{result['summary']['voidCount']} voids"
    )


if __name__ == "__main__":
    main()
