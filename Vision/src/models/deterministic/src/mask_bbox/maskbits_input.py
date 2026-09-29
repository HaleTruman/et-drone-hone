"""Decode deterministic RGB maskbits into bbox-ready layer masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
TOOLS_DIR = APP_DIR / "tools"



from models.deterministic.tools.build_vision_memory import apply_layer_metadata, layer_enabled_map, layer_metadata, stable_hash  # noqa: E402


DEFAULT_MASK_MANIFEST = APP_DIR / "src" / "color_masks" / "output" / "mask_manifest.json"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/" + resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def repo_path(value: str | Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty repo path")
    path = Path(text)
    if path.is_absolute() and path.exists():
        return path
    if text.startswith("/0721Vision/"):
        return APP_DIR / text.removeprefix("/0721Vision/")
    if text.startswith("/src/") or text.startswith("/assets/"):
        return APP_DIR / text.lstrip("/")
    if text.startswith("src/") or text.startswith("assets/"):
        return APP_DIR / text
    if path.is_absolute():
        return path
    return APP_DIR / path


def path_key(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if "/0721Vision/" in text:
        text = text.split("/0721Vision/", 1)[1]
    elif "/assets/" in text:
        text = "assets/" + text.split("/assets/", 1)[1]
    elif "/src/" in text:
        text = "src/" + text.split("/src/", 1)[1]
    text = text.lstrip("/")
    if text.startswith("0721Vision/"):
        text = text.removeprefix("0721Vision/")
    return text


def frame_source_values(frame: dict) -> list[str]:
    values = []
    for key in ("source", "sourcePath", "path", "relativePath", "filename"):
        value = frame.get(key)
        if value:
            values.append(str(value))
    return values


def lut_layer_metadata(mask_manifest: dict) -> dict[str, dict]:
    lut = mask_manifest.get("lut") if isinstance(mask_manifest.get("lut"), dict) else {}
    metadata = lut.get("metadata") if isinstance(lut.get("metadata"), dict) else {}
    layers = metadata.get("layers") if isinstance(metadata.get("layers"), list) else []
    return {str(item.get("prefix")): item for item in layers if isinstance(item, dict)}


def mask_manifest_layers(mask_manifest: dict) -> list[dict]:
    layers = mask_manifest.get("layers") if isinstance(mask_manifest.get("layers"), list) else []
    clean = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        bit = int(layer.get("bit", index))
        if bit < 0 or bit > 7:
            raise ValueError(f"mask layer bit must be in range 0..7, got {bit}")
        prefix = str(layer.get("prefix") or f"bit-{bit}")
        clean.append({"prefix": prefix, "bit": bit, "name": str(layer.get("name") or prefix)})
    return sorted(clean, key=lambda item: int(item["bit"]))


def build_maskbits_context(config: dict, review: dict, mask_manifest: dict, mask_manifest_path: Path) -> dict:
    classes = list(config.get("classes") or [])
    if not classes:
        raise ValueError("alpha class config has no classes")
    classes = apply_layer_metadata(classes, layer_metadata(review))
    enabled_by_prefix = layer_enabled_map(review)
    manifest_layers = mask_manifest_layers(mask_manifest)
    bits_by_prefix = {str(item["prefix"]): int(item["bit"]) for item in manifest_layers}
    lut_layers = lut_layer_metadata(mask_manifest)
    layer_pixel_totals = {}
    summary = mask_manifest.get("summary") if isinstance(mask_manifest.get("summary"), dict) else {}
    raw_counts = summary.get("layerPixelCounts") if isinstance(summary.get("layerPixelCounts"), dict) else {}
    for prefix, count in raw_counts.items():
        layer_pixel_totals[str(prefix)] = int(count or 0)

    class_bits: list[int | None] = []
    summaries: list[dict] = []
    signature_layers: list[dict] = []
    for item in classes:
        prefix = str(item.get("prefix") or "")
        bit = bits_by_prefix.get(prefix)
        lut_item = lut_layers.get(prefix, {})
        rgb_key_count = int(lut_item.get("rgbKeyCount") or 0)
        source_pixel_count = int(layer_pixel_totals.get(prefix, 0))
        enabled = bool(enabled_by_prefix.get(prefix, True)) and bit is not None
        class_bits.append(bit if enabled else None)
        summaries.append(
            {
                "prefix": prefix,
                "displayName": item.get("displayName"),
                "enabled": enabled,
                "colorIdCount": rgb_key_count if enabled else 0,
                "sourceColorIdCount": rgb_key_count,
                "rgbKeyCount": rgb_key_count,
                "sourcePixelCount": source_pixel_count,
                "bit": bit,
                "ruleSource": "rgb-lut-maskbits" if bit is not None else "missing-maskbits-layer",
            }
        )
        signature_layers.append(
            {
                "prefix": prefix,
                "enabled": enabled,
                "bit": bit,
                "rgbKeyCount": rgb_key_count,
                "sourcePixelCount": source_pixel_count,
            }
        )

    enabled_prefixes = [
        str(item["prefix"])
        for item in summaries
        if item.get("enabled") and item.get("bit") is not None
    ]
    if not enabled_prefixes:
        raise ValueError("maskbits manifest has no enabled layer prefixes")

    lut = mask_manifest.get("lut") if isinstance(mask_manifest.get("lut"), dict) else {}
    return {
        "classes": classes,
        "summaries": sorted(summaries, key=lambda item: str(item["prefix"])),
        "classBits": class_bits,
        "enabledPrefixes": enabled_prefixes,
        "manifestLayers": manifest_layers,
        "lutSha256": str(lut.get("sha256") or ""),
        "maskManifestSha256": sha256_file(mask_manifest_path),
        "signature": stable_hash(
            {
                "kind": "0721-rgb-lut-maskbits",
                "maskManifest": repo_url(mask_manifest_path),
                "maskManifestSha256": sha256_file(mask_manifest_path),
                "lutSha256": str(lut.get("sha256") or ""),
                "frameCount": int(mask_manifest.get("frameCount") or len(mask_manifest.get("frames") or [])),
                "width": int(mask_manifest.get("width") or 0),
                "height": int(mask_manifest.get("height") or 0),
                "layers": signature_layers,
            }
        ),
    }


def build_mask_frame_lookup(mask_manifest: dict) -> dict:
    frames = mask_manifest.get("frames") if isinstance(mask_manifest.get("frames"), list) else []
    by_source: dict[str, dict] = {}
    by_index: dict[int, dict] = {}
    by_id: dict[str, dict] = {}
    for ordinal, entry in enumerate(frames):
        if not isinstance(entry, dict):
            continue
        for value in frame_source_values(entry):
            key = path_key(value)
            if key:
                by_source[key] = entry
        if entry.get("index") is not None:
            by_index[int(entry.get("index"))] = entry
        if entry.get("frameOrdinal") is not None:
            by_index[int(entry.get("frameOrdinal"))] = entry
        if entry.get("frameId") is not None:
            by_id[str(entry.get("frameId"))] = entry
        by_index.setdefault(ordinal, entry)
    return {"bySource": by_source, "byIndex": by_index, "byId": by_id, "frames": frames}


def mask_entry_for_frame(lookup: dict, frame_entry: dict, frame_ordinal: int) -> dict:
    for value in frame_source_values(frame_entry):
        key = path_key(value)
        if key and key in lookup["bySource"]:
            return lookup["bySource"][key]
    for key in ("frameId", "rawIndex", "index"):
        value = frame_entry.get(key)
        if value is not None and str(value) in lookup["byId"]:
            return lookup["byId"][str(value)]
    if frame_ordinal in lookup["byIndex"]:
        return lookup["byIndex"][frame_ordinal]
    raise ValueError(f"maskbits manifest has no entry for frame {frame_ordinal}")


def load_mask_bits(mask_entry: dict, width: int, height: int) -> np.ndarray:
    ref = mask_entry.get("maskBits") or mask_entry.get("path")
    if not ref:
        raise ValueError("maskbits frame entry is missing maskBits")
    path = repo_path(str(ref))
    expected = int(width) * int(height)
    values = np.fromfile(path, dtype=np.uint8)
    if values.size != expected:
        raise ValueError(f"{path} length {values.size} != expected {expected}")
    return values.reshape((int(height), int(width)))


def layer_hits_from_mask_bits(mask_bits: np.ndarray, context: dict) -> np.ndarray:
    class_bits = list(context.get("classBits") or [])
    layer_hits = np.zeros((len(class_bits), mask_bits.shape[0], mask_bits.shape[1]), dtype=np.bool_)
    for class_index, bit in enumerate(class_bits):
        if bit is None:
            continue
        layer_hits[class_index] = (mask_bits & np.uint8(1 << int(bit))) != 0
    return layer_hits
