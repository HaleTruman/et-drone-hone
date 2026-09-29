#!/usr/bin/env python3
"""Build fresh 0721 vision memory from decoded HNL frame/color-layer assets.

This only resolves the read-only HNL color layers onto the selected precompute
run and records per-frame pixel counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
APP_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = APP_DIR / "assets" / "config" / "alpha_classes.json"
DEFAULT_RULES = APP_DIR / "assets" / "rules" / "alpha_class_rules.json"
DEFAULT_REVIEW = APP_DIR / "assets" / "review" / "review_state.json"
DEFAULT_OUTPUT = APP_DIR / "assets" / "memory" / "vision_memory.json"
DEFAULT_PRECOMPUTE_ROOT = APP_DIR / "assets" / "precomputed"
DEFAULT_FRAME_RUNS_ROOT = APP_DIR / "assets" / "frame_runs"
DEFAULT_HNL_LAYER_SETS = APP_DIR / "assets" / "color_layers" / "layer_sets.json"
HNL_ACTIVE_RUN_PATH = DEFAULT_FRAME_RUNS_ROOT / "active_run.json"
HNL_DECODED_MANIFEST_NAME = "hnl_decoded_manifest.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_json_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def repo_path_from_manifest(value: str) -> Path:
    text = str(value or "")
    if not text:
        raise ValueError("empty manifest path")
    path = Path(text)
    if path.is_absolute() and str(path).startswith(str(REPO_ROOT)):
        return path
    if text.startswith("/"):
        return REPO_ROOT / text.lstrip("/")
    return REPO_ROOT / text


def clean_child_name(value) -> str | None:
    name = str(value or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    return name


def natural_key(value: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def latest_precompute_manifest(root: Path = DEFAULT_PRECOMPUTE_ROOT) -> Path:
    manifests = sorted(root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"No precompute manifest found in {root}")
    return manifests[0]


def active_frame_run_name() -> str | None:
    data = read_json_optional(HNL_ACTIVE_RUN_PATH)
    if not data:
        return None
    name = clean_child_name(data.get("runName"))
    if not name:
        return None
    root = DEFAULT_FRAME_RUNS_ROOT.resolve()
    run_dir = (DEFAULT_FRAME_RUNS_ROOT / name).resolve()
    return name if run_dir.parent == root and run_dir.is_dir() else None


def precompute_manifest_for_decoded(decoded_manifest: dict | None, root: Path = DEFAULT_PRECOMPUTE_ROOT) -> Path | None:
    if not decoded_manifest:
        return None
    source_hash = decoded_manifest.get("sourceHash")
    decoded_path = decoded_manifest.get("manifestPath")
    if not source_hash or not decoded_path:
        return None
    manifests = sorted(root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in manifests:
        manifest = read_json_optional(path)
        if manifest and manifest.get("sourceHash") == source_hash and manifest.get("sourceDecodedManifest") == decoded_path:
            return path
    return None


def selected_precompute_manifest(root: Path = DEFAULT_PRECOMPUTE_ROOT) -> Path:
    active_name = active_frame_run_name()
    if active_name:
        decoded = read_json_optional(DEFAULT_FRAME_RUNS_ROOT / active_name / HNL_DECODED_MANIFEST_NAME)
        matched = precompute_manifest_for_decoded(decoded, root)
        if matched:
            return matched
        raise FileNotFoundError(f"Active HNL frame run has no matching precompute manifest: {active_name}")
    return latest_precompute_manifest(root)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def expand_ranges(ranges: list) -> np.ndarray:
    values: list[int] = []
    for item in ranges:
        if not isinstance(item, list) or len(item) != 2:
            continue
        start = int(item[0])
        end = int(item[1])
        if end < start:
            start, end = end, start
        if end < 0:
            continue
        values.extend(range(max(0, start), end + 1))
    if not values:
        return np.zeros(0, dtype=np.uint32)
    return np.unique(np.asarray(values, dtype=np.uint32))


def compress_ids(values: np.ndarray | list[int]) -> list[list[int]]:
    arr = np.unique(np.asarray(values, dtype=np.uint32))
    if arr.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(arr[0])
    previous = start
    for raw in arr[1:]:
        value = int(raw)
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def read_precompute_array(manifest: dict, group: str, name: str, dtype: str) -> np.ndarray:
    record = manifest[group]
    info = record["arrays"][name]
    path = repo_path_from_manifest(record["path"])
    return np.fromfile(path, dtype=dtype, count=int(info["length"]), offset=int(info.get("offset", 0)))


def normalize_layer_name(value: str) -> str:
    text = str(value or "").lower().replace("suport", "support")
    return "".join(ch for ch in text if ch.isalnum())


def alpha_layer_set() -> tuple[list[dict], dict[str, dict]]:
    if not DEFAULT_HNL_LAYER_SETS.exists():
        return [], {}
    data = read_json(DEFAULT_HNL_LAYER_SETS)
    sets = data.get("sets") if isinstance(data.get("sets"), list) else []
    selected = next((item for item in sets if item.get("name") == "Alpha Group Maximus Pro Ultra Plus 23"), None)
    if selected is None and sets:
        selected = sets[0]
    layers = list(selected.get("layers") or []) if isinstance(selected, dict) else []
    by_name = {normalize_layer_name(layer.get("name")): layer for layer in layers if isinstance(layer, dict)}
    return layers, by_name


def matched_layer_for_class(item: dict, layers: list[dict], by_name: dict[str, dict]) -> dict | None:
    for key in ("sourceLayerName", "label", "displayName"):
        name = normalize_layer_name(item.get(key))
        if name in by_name:
            return by_name[name]
    try:
        index = int(str(item.get("prefix") or "0")) - 1
    except ValueError:
        index = -1
    if 0 <= index < len(layers) and isinstance(layers[index], dict):
        return layers[index]
    return None


def rgb_keys_from_matched_values(layer: dict | None) -> np.ndarray:
    matched = layer.get("matchedColorValues") if isinstance(layer, dict) else None
    if not isinstance(matched, dict):
        return np.zeros(0, dtype=np.uint32)
    srgb = matched.get("srgb")
    if isinstance(srgb, list) and len(srgb) >= 3:
        arr = np.asarray(srgb, dtype=np.uint32)
        arr = arr[: (arr.size // 3) * 3].reshape(-1, 3)
        keys = (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]
        return np.unique(keys.astype(np.uint32))
    hex_values = matched.get("rgbHex")
    if isinstance(hex_values, list):
        values: list[int] = []
        for raw in hex_values:
            text = str(raw or "").strip().lstrip("#")
            if len(text) != 6:
                continue
            try:
                values.append(int(text, 16))
            except ValueError:
                continue
        if values:
            return np.unique(np.asarray(values, dtype=np.uint32))
    return np.zeros(0, dtype=np.uint32)


def ids_from_matched_rgb(layer: dict | None, rgb_keys: np.ndarray) -> np.ndarray:
    matched_keys = rgb_keys_from_matched_values(layer)
    if matched_keys.size == 0 or rgb_keys.size == 0:
        return np.zeros(0, dtype=np.uint32)
    return np.nonzero(np.isin(rgb_keys, matched_keys))[0].astype(np.uint32)


def range_mask(values: np.ndarray, low: float, high: float, low_inclusive: bool = True, high_inclusive: bool = True) -> np.ndarray:
    arr = values.astype(np.float32)
    finite = np.isfinite(arr)
    if low <= high:
        left = arr >= low if low_inclusive else arr > low
        right = arr <= high if high_inclusive else arr < high
        return finite & left & right
    left = arr >= low if low_inclusive else arr > low
    right = arr <= high if high_inclusive else arr < high
    return finite & (left | right)


def region_mask(region: dict, hue: np.ndarray, lightness: np.ndarray, kind_mask: np.ndarray | None = None) -> np.ndarray:
    mask = np.ones(hue.shape, dtype=np.bool_) if kind_mask is None else kind_mask.copy()
    if "hueMin" in region and "hueMax" in region:
        mask &= range_mask(
            hue,
            float(region.get("hueMin", 0.0)),
            float(region.get("hueMax", 1.0)),
            bool(region.get("hueMinInclusive", True)),
            bool(region.get("hueMaxInclusive", True)),
        )
    if "lightMin" in region and "lightMax" in region:
        mask &= range_mask(
            lightness,
            float(region.get("lightMin", 0.0)),
            float(region.get("lightMax", 1.0)),
            bool(region.get("lightMinInclusive", True)),
            bool(region.get("lightMaxInclusive", True)),
        )
    return mask


def ids_from_hnl_rule(item: dict, color_table: dict[str, np.ndarray]) -> np.ndarray:
    rule = item.get("sourceHnlRule") if isinstance(item.get("sourceHnlRule"), dict) else {}
    if not rule:
        return np.zeros(0, dtype=np.uint32)
    hue = color_table["hue"]
    lightness = color_table["lightness"]
    kind = color_table["kind"]
    selected = np.zeros(hue.shape, dtype=np.bool_)

    chromatic = kind == 2
    if all(key in rule for key in ("hueMin", "hueMax", "lightMin", "lightMax")):
        selected |= region_mask(rule, hue, lightness, chromatic)
    for region in rule.get("allowedRegions") or []:
        if isinstance(region, dict):
            selected |= region_mask(region, hue, lightness, chromatic)
    for cutout in rule.get("cutouts") or []:
        if isinstance(cutout, dict):
            selected &= ~region_mask(cutout, hue, lightness, chromatic)

    neutral = kind == 0
    for region in rule.get("neutralAllowedRegions") or []:
        if isinstance(region, dict):
            selected |= region_mask(region, hue, lightness, neutral)
    for cutout in rule.get("neutralCutouts") or []:
        if isinstance(cutout, dict):
            selected &= ~region_mask(cutout, hue, lightness, neutral)

    near_neutral = kind == 1
    for key in ("nearNeutralAllowedRegions", "nearNeutralStrokeRegions", "darkNeutralAllowedRegions", "darkNeutralStrokeRegions"):
        for region in rule.get(key) or []:
            if isinstance(region, dict):
                selected |= region_mask(region, hue, lightness, near_neutral)
    for key in ("nearNeutralCutouts", "nearNeutralStrokeCutouts", "darkNeutralCutouts", "darkNeutralStrokeCutouts"):
        for cutout in rule.get(key) or []:
            if isinstance(cutout, dict):
                selected &= ~region_mask(cutout, hue, lightness, near_neutral)
    return np.nonzero(selected)[0].astype(np.uint32)


def stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def class_rule_ranges(rules: dict, prefix: str) -> list:
    class_rules = rules.get("classRules") if isinstance(rules.get("classRules"), dict) else {}
    rule = class_rules.get(prefix) if isinstance(class_rules.get(prefix), dict) else {}
    return rule.get("colorIdRanges") if isinstance(rule.get("colorIdRanges"), list) else []


def layer_enabled_map(review: dict) -> dict[str, bool]:
    visualization = review.get("visualization") if isinstance(review.get("visualization"), dict) else {}
    raw = visualization.get("layerEnabledByPrefix") if isinstance(visualization.get("layerEnabledByPrefix"), dict) else {}
    return {str(prefix): value is not False for prefix, value in raw.items()}


def layer_metadata(review: dict) -> dict:
    metadata = review.get("layerMetadata") if isinstance(review.get("layerMetadata"), dict) else {}
    confidence = metadata.get("layerConfidenceByPrefix") if isinstance(metadata.get("layerConfidenceByPrefix"), dict) else {}
    group_ids = metadata.get("layerGroupIdsByPrefix") if isinstance(metadata.get("layerGroupIdsByPrefix"), dict) else {}
    group_slots = metadata.get("layerGroupSlotsByPrefix") if isinstance(metadata.get("layerGroupSlotsByPrefix"), dict) else {}
    clean_confidence: dict[str, float] = {}
    clean_groups: dict[str, list[str]] = {}
    for prefix, value in confidence.items():
        try:
            clean_confidence[str(prefix)] = clamp(float(value), 0.0, 1.0)
        except (TypeError, ValueError):
            continue
    allowed_groups = {f"group-{index}" for index in range(1, 6)}
    for source in (group_ids, group_slots):
        for prefix, values in source.items():
            if not isinstance(values, list):
                continue
            seen = clean_groups.get(str(prefix), [])
            for value in values:
                group = str(value)
                if group in allowed_groups and group not in seen:
                    seen.append(group)
            clean_groups[str(prefix)] = seen[:3]
    return {"confidenceByPrefix": clean_confidence, "groupIdsByPrefix": clean_groups}


def apply_layer_metadata(classes: list[dict], metadata: dict) -> list[dict]:
    confidence_by_prefix = metadata.get("confidenceByPrefix", {})
    groups_by_prefix = metadata.get("groupIdsByPrefix", {})
    effective: list[dict] = []
    for item in classes:
        prefix = str(item.get("prefix"))
        preset_confidence = clamp(float(item.get("confidence", 0.0) or 0.0), 0.0, 1.0)
        confidence = confidence_by_prefix.get(prefix, preset_confidence)
        clean = dict(item)
        clean["presetConfidence"] = preset_confidence
        clean["confidence"] = clamp(float(confidence), 0.0, 1.0)
        clean["groupIds"] = list(groups_by_prefix.get(prefix, []))
        effective.append(clean)
    return effective


def source_precompute_run_key(rules: dict, item: dict) -> str:
    source = rules.get("source") if isinstance(rules.get("source"), dict) else {}
    if source.get("precomputeRunKey"):
        return str(source.get("precomputeRunKey"))
    color_rule = item.get("sourceColorRule") if isinstance(item.get("sourceColorRule"), dict) else {}
    return str(color_rule.get("precomputeRunKey") or "")


def build_class_membership(classes: list[dict], rules: dict, manifest: dict, enabled_by_prefix: dict[str, bool]) -> tuple[np.ndarray, list[dict]]:
    color_count = int(manifest["colorTable"]["count"])
    run_key = str(manifest.get("runKey") or "")
    rgb_keys = read_precompute_array(manifest, "colorTable", "rgbKey", "<u4")
    color_table = {
        "hue": read_precompute_array(manifest, "colorTable", "hue", "<f4"),
        "lightness": read_precompute_array(manifest, "colorTable", "lightness", "<f4"),
        "kind": read_precompute_array(manifest, "colorTable", "kind", "u1"),
    }
    layers, layers_by_name = alpha_layer_set()
    membership = np.zeros((len(classes), color_count), dtype=np.bool_)
    summaries = []
    for class_index, item in enumerate(classes):
        prefix = str(item["prefix"])
        source_run_key = source_precompute_run_key(rules, item)
        rule_source = "none"
        ids = np.zeros(0, dtype=np.uint32)
        if source_run_key and source_run_key == run_key:
            ids = expand_ranges(class_rule_ranges(rules, prefix))
            rule_source = "current-run-color-id-ranges"
        if ids.size == 0:
            ids = ids_from_matched_rgb(matched_layer_for_class(item, layers, layers_by_name), rgb_keys)
            if ids.size:
                rule_source = "matched-rgb-values"
        if ids.size == 0:
            ids = ids_from_hnl_rule(item, color_table)
            if ids.size:
                rule_source = "source-hnl-rule"

        ids = ids[ids < color_count]
        ranges = compress_ids(ids)
        enabled = enabled_by_prefix.get(prefix, True)
        if enabled and ids.size:
            membership[class_index, ids] = True
        summaries.append(
            {
                "prefix": prefix,
                "displayName": item.get("displayName"),
                "enabled": bool(enabled),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "presetConfidence": float(item.get("presetConfidence", item.get("confidence", 0.0)) or 0.0),
                "groupIds": list(item.get("groupIds") or []),
                "colorIdCount": int(ids.size) if enabled else 0,
                "sourceColorIdCount": int(ids.size),
                "ruleSource": rule_source,
                "sourcePrecomputeRunKey": source_run_key or None,
                "activePrecomputeRunKey": run_key,
                "colorIdRanges": ranges if enabled else [],
                "sourceColorIdRanges": ranges,
            }
        )
    return membership, sorted(summaries, key=lambda item: item["prefix"])


def read_frame_color_ids(frame_entry: dict, width: int, height: int) -> np.ndarray:
    path = repo_path_from_manifest(frame_entry["cachePath"])
    info = frame_entry["arrays"]["colorIds"]
    ids = np.fromfile(path, dtype="<u4", count=int(info["length"]), offset=int(info.get("offset", 0)))
    expected = width * height
    if ids.size != expected:
        raise ValueError(f"{path} colorIds length {ids.size} != {expected}")
    return ids.reshape((height, width))


def frame_layer_counts(ids: np.ndarray, membership: np.ndarray, classes: list[dict]) -> tuple[list[dict], int]:
    hits = membership[:, ids]
    layer_counts = []
    for class_index, item in enumerate(classes):
        count = int(np.count_nonzero(hits[class_index]))
        layer_counts.append(
            {
                "prefix": str(item.get("prefix")),
                "displayName": item.get("displayName"),
                "pixelCount": count,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "enabled": bool(count or membership[class_index].any()),
            }
        )
    classified = int(np.count_nonzero(np.any(hits, axis=0)))
    return layer_counts, classified


def build_memory(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    config = read_json(args.config)
    rules = read_json(args.rules)
    review = read_json(args.review) if args.review.exists() else {"annotations": [], "decisions": []}
    manifest_path = args.precompute_manifest or selected_precompute_manifest()
    manifest = read_json(manifest_path)

    base_classes = list(config.get("classes") or [])
    if not base_classes:
        raise ValueError("alpha class config has no classes")
    metadata = layer_metadata(review)
    classes = apply_layer_metadata(base_classes, metadata)
    enabled_by_prefix = layer_enabled_map(review)
    membership, class_rule_summary = build_class_membership(classes, rules, manifest, enabled_by_prefix)

    frames_in = list(manifest.get("frames") or [])
    start = max(0, int(args.start_frame))
    end = len(frames_in) if args.max_frames is None else min(len(frames_in), start + max(0, int(args.max_frames)))
    selected = list(enumerate(frames_in[start:end], start=start))
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)

    frames_out: list[dict] = []
    total_classified_pixels = 0
    total_layer_counts = {str(item.get("prefix")): 0 for item in classes}
    for local_index, (frame_ordinal, frame_entry) in enumerate(selected):
        ids = read_frame_color_ids(frame_entry, width, height)
        layer_counts, classified = frame_layer_counts(ids, membership, classes)
        for item in layer_counts:
            total_layer_counts[str(item["prefix"])] = total_layer_counts.get(str(item["prefix"]), 0) + int(item["pixelCount"])
        total_classified_pixels += classified
        frames_out.append(
            {
                "frameOrdinal": frame_ordinal,
                "frameId": frame_entry.get("frameId"),
                "path": frame_entry.get("path"),
                "sourcePath": frame_entry.get("path"),
                "cachePath": frame_entry.get("cachePath"),
                "classifiedPixelCount": classified,
                "layerCounts": layer_counts,
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "decoding-frames",
                    "index": local_index + 1,
                    "total": len(selected),
                    "frameOrdinal": frame_ordinal,
                    "classifiedPixels": classified,
                }
            )

    layer_totals = [
        {
            "prefix": str(item.get("prefix")),
            "displayName": item.get("displayName"),
            "pixelCount": int(total_layer_counts.get(str(item.get("prefix")), 0)),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "enabled": enabled_by_prefix.get(str(item.get("prefix")), True),
            "groupIds": list(item.get("groupIds") or []),
        }
        for item in classes
    ]
    rule_hash_payload = {
        "classRules": rules.get("classRules", {}),
        "layerEnabledByPrefix": enabled_by_prefix,
        "layerConfidenceByPrefix": metadata["confidenceByPrefix"],
        "layerGroupIdsByPrefix": metadata["groupIdsByPrefix"],
        "runKey": manifest.get("runKey"),
        "analysisMode": "fresh-layer-decode",
    }
    result = {
        "version": 1,
        "app": "vision_passthrough_review",
        "analysisMode": "fresh-layer-decode",
        "status": "ready",
        "createdAt": utc_now(),
        "source": {
            "runKey": manifest.get("runKey"),
            "precomputeManifest": "/" + manifest_path.relative_to(REPO_ROOT).as_posix(),
            "frameCount": int(manifest.get("frameCount") or len(frames_in)),
            "processedFrameCount": len(frames_out),
            "image": {"width": width, "height": height},
            "frameStartOrdinal": start,
        },
        "classes": classes,
        "classRules": {
            "ruleModel": "decoded-active-run-color-ids",
            "hash": stable_hash(rule_hash_payload),
            "summaries": class_rule_summary,
            "layerEnabledByPrefix": enabled_by_prefix,
            "layerConfidenceByPrefix": metadata["confidenceByPrefix"],
            "layerGroupIdsByPrefix": metadata["groupIdsByPrefix"],
        },
        "frames": frames_out,
        "layerTotals": layer_totals,
        "summary": {
            "processedFrameCount": len(frames_out),
            "decodedLayerCount": len(classes),
            "enabledLayerCount": sum(1 for item in class_rule_summary if item["enabled"]),
            "classifiedPixelCount": int(total_classified_pixels),
            "enabledClassColorIdCount": sum(item["colorIdCount"] for item in class_rule_summary),
            "sourceClassColorIdCount": sum(item["sourceColorIdCount"] for item in class_rule_summary),
        },
    }
    if args.output:
        atomic_write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_memory(args)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()) if args.output else None,
                "frames": result["source"]["processedFrameCount"],
                "layers": result["summary"]["decodedLayerCount"],
                "ruleHash": result["classRules"]["hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
