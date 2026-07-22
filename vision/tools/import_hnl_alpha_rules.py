#!/usr/bin/env python3
"""Import Alpha HNL layer rules as independent exact color-ID ranges."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
APP_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_EXPORT = Path("/Users/trumanhale/Downloads/gate_color_hnl_layer_sets-10.json")
DEFAULT_CONFIG = APP_DIR / "assets" / "config" / "alpha_classes.json"
DEFAULT_RULES = APP_DIR / "assets" / "rules" / "alpha_class_rules.json"
DEFAULT_PRECOMPUTE_ROOT = REPO_ROOT / "visualize" / "gate_color_hnl" / "assets" / "precomputed"
ALPHA_SET_NAME = "Alpha Group Maximus Pro Ultra Plus 23"
HSV_NEUTRAL_S = 0.02
DARK_NEUTRAL_LIGHTNESS_MAX = 0.5
DARK_NEUTRAL_S = 0.5
PREFIX_TO_SOURCE_LAYER_ID = {
    "001": 6,
    "002": 10,
    "003": 12,
    "004": 13,
    "005": 15,
    "006": 16,
    "007": 23,
    "008": 24,
}
HNL_FIELDS = [
    "lightMinInclusive",
    "lightMaxInclusive",
    "hueMinInclusive",
    "hueMaxInclusive",
    "lightMin",
    "lightMax",
    "hueMin",
    "hueMax",
    "cutouts",
    "allowedRegions",
    "strokeRegions",
    "strokeCutouts",
    "neutralAllowedRegions",
    "neutralCutouts",
    "nearNeutralAllowedRegions",
    "nearNeutralCutouts",
    "nearNeutralStrokeRegions",
    "nearNeutralStrokeCutouts",
    "darkNeutralAllowedRegions",
    "darkNeutralCutouts",
    "darkNeutralStrokeRegions",
    "darkNeutralStrokeCutouts",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


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


def latest_precompute_manifest(root: Path = DEFAULT_PRECOMPUTE_ROOT) -> Path:
    manifests = sorted(root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"No precompute manifest found in {root}")
    return manifests[0]


def repo_path_from_manifest(value: str) -> Path:
    text = str(value or "")
    if text.startswith("/"):
        return REPO_ROOT / text.lstrip("/")
    return REPO_ROOT / text


def clamp01(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


def normalize_range(a, b) -> tuple[float, float]:
    low = clamp01(a)
    high = clamp01(1.0 if b is None else b)
    return (min(low, high), max(low, high))


def normalize_hnl_region(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    light_min, light_max = normalize_range(source.get("lightMin"), source.get("lightMax", 1.0))
    return {
        "lightMin": light_min,
        "lightMax": light_max,
        "hueMin": clamp01(source.get("hueMin")),
        "hueMax": clamp01(source.get("hueMax", 1.0)),
        "lightMinInclusive": source.get("lightMinInclusive") is not False,
        "lightMaxInclusive": source.get("lightMaxInclusive") is not False,
        "hueMinInclusive": source.get("hueMinInclusive") is not False,
        "hueMaxInclusive": source.get("hueMaxInclusive") is not False,
    }


def normalize_neutral_region(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    light_min, light_max = normalize_range(source.get("lightMin"), source.get("lightMax", 1.0))
    return {
        "lightMin": light_min,
        "lightMax": light_max,
        "lightMinInclusive": source.get("lightMinInclusive") is not False,
        "lightMaxInclusive": source.get("lightMaxInclusive") is not False,
    }


def normalize_stroke(raw: dict | None) -> dict | None:
    source = raw if isinstance(raw, dict) else {}
    points = []
    for point in source.get("points") or []:
        if not isinstance(point, dict):
            continue
        points.append(
            {
                "hue": clamp01(point.get("hue", point.get("h"))),
                "light": clamp01(point.get("light", point.get("l"))),
            }
        )
    if not points:
        return None
    return {
        "points": points,
        "radiusHue": max(1e-6, min(1.0, float(source.get("radiusHue") or 0.003))),
        "radiusLight": max(1e-6, min(1.0, float(source.get("radiusLight") or 0.003))),
    }


def normalize_hnl_layer(raw: dict) -> dict:
    layer = normalize_hnl_region(raw)
    layer.update(
        {
            "id": raw.get("id"),
            "name": str(raw.get("name") or ""),
            "enabled": raw.get("enabled") is not False,
            "cutouts": [normalize_hnl_region(item) for item in raw.get("cutouts") or [] if isinstance(item, dict)],
            "allowedRegions": [normalize_hnl_region(item) for item in raw.get("allowedRegions") or [] if isinstance(item, dict)],
            "strokeRegions": [item for item in (normalize_stroke(raw_item) for raw_item in raw.get("strokeRegions") or []) if item],
            "strokeCutouts": [item for item in (normalize_stroke(raw_item) for raw_item in raw.get("strokeCutouts") or []) if item],
            "neutralAllowedRegions": [normalize_neutral_region(item) for item in raw.get("neutralAllowedRegions") or [] if isinstance(item, dict)],
            "neutralCutouts": [normalize_neutral_region(item) for item in raw.get("neutralCutouts") or [] if isinstance(item, dict)],
            "nearNeutralAllowedRegions": [normalize_neutral_region(item) for item in raw.get("nearNeutralAllowedRegions") or [] if isinstance(item, dict)],
            "nearNeutralCutouts": [normalize_neutral_region(item) for item in raw.get("nearNeutralCutouts") or [] if isinstance(item, dict)],
            "nearNeutralStrokeRegions": [item for item in (normalize_stroke(raw_item) for raw_item in raw.get("nearNeutralStrokeRegions") or []) if item],
            "nearNeutralStrokeCutouts": [item for item in (normalize_stroke(raw_item) for raw_item in raw.get("nearNeutralStrokeCutouts") or []) if item],
            "darkNeutralAllowedRegions": [normalize_hnl_region(item) for item in raw.get("darkNeutralAllowedRegions") or [] if isinstance(item, dict)],
            "darkNeutralCutouts": [normalize_hnl_region(item) for item in raw.get("darkNeutralCutouts") or [] if isinstance(item, dict)],
            "darkNeutralStrokeRegions": [item for item in (normalize_stroke(raw_item) for raw_item in raw.get("darkNeutralStrokeRegions") or []) if item],
            "darkNeutralStrokeCutouts": [item for item in (normalize_stroke(raw_item) for raw_item in raw.get("darkNeutralStrokeCutouts") or []) if item],
        }
    )
    return layer


def light_mask(lightness: np.ndarray, region: dict) -> np.ndarray:
    if region.get("lightMinInclusive", True):
        lower = lightness >= float(region["lightMin"])
    else:
        lower = lightness > float(region["lightMin"])
    if region.get("lightMaxInclusive", True):
        upper = lightness <= float(region["lightMax"])
    else:
        upper = lightness < float(region["lightMax"])
    return lower & upper


def hue_mask(hue: np.ndarray, region: dict) -> np.ndarray:
    h = np.mod(hue, 1.0)
    h[~np.isfinite(hue)] = np.nan
    low = float(region["hueMin"])
    high = float(region["hueMax"])
    if region.get("hueMinInclusive", True):
        lower = h >= low
    else:
        lower = h > low
    if region.get("hueMaxInclusive", True):
        upper = h <= high
    else:
        upper = h < high
    return lower & upper if low <= high else lower | upper


def hnl_region_mask(layer: dict, lightness: np.ndarray, hue: np.ndarray) -> np.ndarray:
    return np.isfinite(hue) & light_mask(lightness, layer) & hue_mask(hue, layer)


def neutral_region_mask(region: dict, lightness: np.ndarray) -> np.ndarray:
    return light_mask(lightness, region)


def hue_distance01(a: np.ndarray, b: float) -> np.ndarray:
    diff = np.abs(np.mod(a, 1.0) - (b % 1.0))
    return np.minimum(diff, 1.0 - diff)


def stroke_mask(strokes: list[dict], lightness: np.ndarray, hue: np.ndarray) -> np.ndarray:
    if not strokes:
        return np.zeros(lightness.shape, dtype=bool)
    result = np.zeros(lightness.shape, dtype=bool)
    finite = np.isfinite(hue)
    for stroke in strokes:
        points = stroke.get("points") or []
        rh = max(1e-6, float(stroke.get("radiusHue") or 0.003))
        rl = max(1e-6, float(stroke.get("radiusLight") or 0.003))
        if len(points) == 1:
            point = points[0]
            dist = (hue_distance01(hue, float(point["hue"])) / rh) ** 2 + ((lightness - float(point["light"])) / rl) ** 2
            result |= finite & (dist <= 1.0)
            continue
        for first, second in zip(points, points[1:]):
            result |= segment_stroke_mask(hue, lightness, first, second, rh, rl)
    return result


def segment_stroke_mask(hue: np.ndarray, lightness: np.ndarray, a: dict, b: dict, radius_hue: float, radius_light: float) -> np.ndarray:
    h0 = clamp01(a["hue"])
    h1 = clamp01(b["hue"])
    if abs(h1 - h0) > 0.5:
        h1 += 1 if h1 < h0 else -1
    l0 = clamp01(a["light"])
    l1 = clamp01(b["light"])
    dx = (h1 - h0) / radius_hue
    dy = (l1 - l0) / radius_light
    length_sq = dx * dx + dy * dy
    best = np.full(hue.shape, np.inf, dtype=np.float32)
    finite = np.isfinite(hue)
    for offset in (-1.0, 0.0, 1.0):
        hx = (np.mod(hue, 1.0) + offset - h0) / radius_hue
        ly = (lightness - l0) / radius_light
        t = np.clip((hx * dx + ly * dy) / length_sq, 0.0, 1.0) if length_sq > 0 else 0.0
        px = dx * t
        py = dy * t
        dist = (hx - px) ** 2 + (ly - py) ** 2
        best = np.minimum(best, dist)
    return finite & (best <= 1.0)


def any_hnl_region_mask(regions: list[dict], lightness: np.ndarray, hue: np.ndarray) -> np.ndarray:
    result = np.zeros(lightness.shape, dtype=bool)
    for region in regions:
        result |= hnl_region_mask(region, lightness, hue)
    return result


def any_neutral_region_mask(regions: list[dict], lightness: np.ndarray) -> np.ndarray:
    result = np.zeros(lightness.shape, dtype=bool)
    for region in regions:
        result |= neutral_region_mask(region, lightness)
    return result


def default_hnl_region(layer: dict) -> bool:
    epsilon = 1e-9
    return (
        abs(float(layer.get("lightMin", 0.0)) - 0.0) <= epsilon
        and abs(float(layer.get("lightMax", 1.0)) - 1.0) <= epsilon
        and abs(float(layer.get("hueMin", 0.0)) - 0.0) <= epsilon
        and abs(float(layer.get("hueMax", 1.0)) - 1.0) <= epsilon
        and layer.get("lightMinInclusive", True)
        and layer.get("lightMaxInclusive", True)
        and layer.get("hueMinInclusive", True)
        and layer.get("hueMaxInclusive", True)
    )


def dark_neutral_accepts(lightness: np.ndarray, saturation: np.ndarray, hue: np.ndarray) -> np.ndarray:
    threshold = np.where(lightness < DARK_NEUTRAL_LIGHTNESS_MAX, DARK_NEUTRAL_S, HSV_NEUTRAL_S)
    return np.isfinite(hue) & (saturation < threshold)


def match_layer(layer: dict, lightness: np.ndarray, hue: np.ndarray, saturation: np.ndarray, kind: np.ndarray) -> np.ndarray:
    if not layer.get("enabled", True):
        return np.zeros(lightness.shape, dtype=bool)
    result = np.zeros(lightness.shape, dtype=bool)
    dark_allowed = layer["darkNeutralAllowedRegions"]
    dark_strokes = layer["darkNeutralStrokeRegions"]
    dark_matches = np.zeros(lightness.shape, dtype=bool)
    if dark_allowed or dark_strokes:
        dark_matches = dark_neutral_accepts(lightness, saturation, hue)
        dark_matches &= any_hnl_region_mask(dark_allowed, lightness, hue) | stroke_mask(dark_strokes, lightness, hue)
        dark_cutouts = any_hnl_region_mask(layer["darkNeutralCutouts"], lightness, hue) | stroke_mask(layer["darkNeutralStrokeCutouts"], lightness, hue)
        base_cutouts = any_hnl_region_mask(layer["cutouts"], lightness, hue) | stroke_mask(layer["strokeCutouts"], lightness, hue)
        result |= dark_matches & ~(dark_cutouts | base_cutouts)

    undecided = ~dark_matches
    near_scope = undecided & np.isfinite(hue) & (kind == 1)
    if layer["nearNeutralAllowedRegions"] or layer["nearNeutralStrokeRegions"]:
        near_matches = near_scope & (
            any_neutral_region_mask(layer["nearNeutralAllowedRegions"], lightness)
            | stroke_mask(layer["nearNeutralStrokeRegions"], lightness, hue)
        )
        near_cutouts = any_neutral_region_mask(layer["nearNeutralCutouts"], lightness) | stroke_mask(layer["nearNeutralStrokeCutouts"], lightness, hue)
        result |= near_matches & ~near_cutouts

    neutral_scope = undecided & ~np.isfinite(hue)
    if layer["neutralAllowedRegions"]:
        neutral_matches = neutral_scope & any_neutral_region_mask(layer["neutralAllowedRegions"], lightness)
        result |= neutral_matches & ~any_neutral_region_mask(layer["neutralCutouts"], lightness)

    chromatic_scope = undecided & np.isfinite(hue) & (kind != 1)
    has_special = (
        bool(layer["neutralAllowedRegions"])
        or bool(layer["nearNeutralAllowedRegions"])
        or bool(layer["nearNeutralStrokeRegions"])
        or bool(layer["darkNeutralAllowedRegions"])
        or bool(layer["darkNeutralStrokeRegions"])
        or bool(layer["strokeRegions"])
    )
    base_regions = layer["allowedRegions"] if layer["allowedRegions"] else ([] if has_special and default_hnl_region(layer) else [layer])
    base_matches = chromatic_scope & (any_hnl_region_mask(base_regions, lightness, hue) | stroke_mask(layer["strokeRegions"], lightness, hue))
    base_cutouts = any_hnl_region_mask(layer["cutouts"], lightness, hue) | stroke_mask(layer["strokeCutouts"], lightness, hue)
    result |= base_matches & ~base_cutouts
    return result


def compress_ids(values: np.ndarray) -> list[list[int]]:
    arr = np.unique(np.asarray(values, dtype=np.uint32))
    if arr.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(arr[0])
    previous = start
    for value in arr[1:]:
        clean = int(value)
        if clean == previous + 1:
            previous = clean
            continue
        ranges.append([start, previous])
        start = previous = clean
    ranges.append([start, previous])
    return ranges


def load_color_table(manifest: dict) -> dict:
    table = manifest["colorTable"]
    path = repo_path_from_manifest(table["path"])
    arrays = table["arrays"]
    return {
        "rgb": np.fromfile(path, dtype=np.uint8, count=int(arrays["rgb"]["length"]), offset=int(arrays["rgb"].get("offset", 0))),
        "hue": np.fromfile(path, dtype="<f4", count=int(arrays["hue"]["length"]), offset=int(arrays["hue"].get("offset", 0))),
        "saturation": np.fromfile(path, dtype="<f4", count=int(arrays["saturation"]["length"]), offset=int(arrays["saturation"].get("offset", 0))),
        "lightness": np.fromfile(path, dtype="<f4", count=int(arrays["lightness"]["length"]), offset=int(arrays["lightness"].get("offset", 0))),
        "kind": np.fromfile(path, dtype=np.uint8, count=int(arrays["kind"]["length"]), offset=int(arrays["kind"].get("offset", 0))),
    }


def find_alpha_set(document: dict) -> tuple[dict, dict]:
    for layer_set in document.get("sets") or []:
        if str(layer_set.get("name")) == ALPHA_SET_NAME or str(layer_set.get("id")) == "set-redgate-clean-gamut":
            for group in layer_set.get("groups") or []:
                if str(group.get("name")) == ALPHA_SET_NAME:
                    return layer_set, group
    raise ValueError(f"Could not find {ALPHA_SET_NAME!r} in layer export")


def source_rule_for_layer(layer: dict) -> dict:
    return {key: layer.get(key, [] if key.endswith(("Regions", "Cutouts")) or key == "cutouts" else None) for key in HNL_FIELDS}


def sample_rgb(rgb: np.ndarray, ids: np.ndarray, limit: int = 24) -> list[list[int]]:
    samples = []
    for color_id in ids[:limit]:
        offset = int(color_id) * 3
        samples.append([int(rgb[offset]), int(rgb[offset + 1]), int(rgb[offset + 2])])
    return samples


def kind_counts(mask: np.ndarray, hue: np.ndarray, kind: np.ndarray) -> dict:
    neutral = mask & (~np.isfinite(hue) | (kind == 0))
    near = mask & np.isfinite(hue) & (kind == 1)
    chromatic = mask & np.isfinite(hue) & (kind != 0) & (kind != 1)
    return {
        "chromatic": int(np.count_nonzero(chromatic)),
        "nearNeutral": int(np.count_nonzero(near)),
        "neutral": int(np.count_nonzero(neutral)),
    }


def update_documents(args: argparse.Namespace) -> dict:
    config = read_json(args.config)
    rules = read_json(args.rules) if args.rules.exists() else {}
    layer_document = read_json(args.layer_export)
    manifest_path = args.precompute_manifest or latest_precompute_manifest()
    manifest = read_json(manifest_path)
    table = load_color_table(manifest)
    layer_set, group = find_alpha_set(layer_document)
    source_layers = {int(layer["id"]): normalize_hnl_layer(layer) for layer in layer_set.get("layers") or [] if layer.get("id") is not None}
    class_rules = rules.setdefault("classRules", {})
    summaries = []
    now = utc_now()
    for item in config.get("classes") or []:
        prefix = str(item.get("prefix"))
        source_layer_id = PREFIX_TO_SOURCE_LAYER_ID.get(prefix)
        if source_layer_id not in source_layers:
            raise ValueError(f"No source HNL layer found for prefix {prefix}")
        source_layer = source_layers[source_layer_id]
        mask = match_layer(source_layer, table["lightness"], table["hue"], table["saturation"], table["kind"])
        ids = np.flatnonzero(mask).astype(np.uint32)
        ranges = compress_ids(ids)
        samples = sample_rgb(table["rgb"], ids)
        counts = kind_counts(mask, table["hue"], table["kind"])
        item["sourceLayerId"] = source_layer_id
        item["sourceLayerName"] = source_layer["name"]
        item["sourceLayerSetId"] = str(layer_set.get("id"))
        item["sourceLayerSetName"] = str(layer_set.get("name"))
        item["sourceGroupId"] = str(group.get("id"))
        item["sourceGroupName"] = str(group.get("name"))
        item["sourceHnlRule"] = source_rule_for_layer(source_layer)
        item["sourceColorRule"] = {
            "model": "gate_color_hnl.hnl-v1",
            "precomputeRunKey": manifest.get("runKey"),
            "precomputeManifest": "/" + manifest_path.relative_to(REPO_ROOT).as_posix(),
            "colorIdCount": int(ids.size),
            "colorIdRangeCount": len(ranges),
            "kindCounts": counts,
            "sampleRgb": samples,
            "importedAt": now,
        }
        class_rules[prefix] = {
            "source": "gate_color_hnl.hnl-v1",
            "sourceLayerId": source_layer_id,
            "sourceLayerName": source_layer["name"],
            "colorIdRanges": ranges,
            "sampleRgb": samples,
            "colorIdCount": int(ids.size),
            "kindCounts": counts,
        }
        summaries.append({"prefix": prefix, "sourceLayerId": source_layer_id, "colorIdCount": int(ids.size), "colorIdRangeCount": len(ranges)})

    config["ruleModel"] = "independent-exact-color-ids-from-hnl"
    config["source"] = {
        "app": "gate_color_hnl",
        "layerExport": str(args.layer_export),
        "layerSetId": str(layer_set.get("id")),
        "layerSetName": str(layer_set.get("name")),
        "groupId": str(group.get("id")),
        "groupName": str(group.get("name")),
        "precomputeRunKey": manifest.get("runKey"),
        "precomputeManifest": "/" + manifest_path.relative_to(REPO_ROOT).as_posix(),
        "conversion": {
            "hue": "HSV hue normalized 0..1",
            "lightness": "Oklab L normalized 0..1",
            "saturation": "HSV saturation",
            "neutralKind": "non-finite hue or kind code 0",
            "nearNeutralKind": f"HSV saturation < {HSV_NEUTRAL_S}",
            "darkNearNeutralPreview": f"lightness < {DARK_NEUTRAL_LIGHTNESS_MAX} uses saturation < {DARK_NEUTRAL_S}",
        },
        "importedAt": now,
    }
    rules.update(
        {
            "version": 1,
            "app": "alpha_pixel_review",
            "ruleModel": "independent-exact-color-ids",
            "name": "Alpha Group Maximus Pro Ultra Plus 23 exact color rules",
            "updatedAt": now,
            "source": config["source"],
            "classRules": class_rules,
        }
    )
    atomic_write_json(args.config, config)
    atomic_write_json(args.rules, rules)
    return {"ok": True, "config": str(args.config), "rules": str(args.rules), "summaries": summaries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    result = update_documents(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
