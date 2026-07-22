#!/usr/bin/env python3
"""Build the standalone dense RGB mask LUT artifact.

This script reads existing JSON artifacts as source material, but it does not
import or execute any current vision-pipeline code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYER_SETS = APP_ROOT / "assets" / "color_layers" / "layer_sets.json"
DEFAULT_OUTPUT = APP_ROOT / "src" / "color_masks" / "artifacts" / "color_lut_v1.npz"
DEFAULT_LAYERS = ("001", "002", "003", "007")
LUT_SCHEMA = "rgb-mask-lut.v1"
DECODER_CONTRACT = "PIL.Image.open(path).convert('RGB')"

BIT_COUNTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def parse_layer_prefixes(value: str) -> list[str]:
    prefixes = [item.strip() for item in value.split(",") if item.strip()]
    clean: list[str] = []
    for prefix in prefixes:
        try:
            clean.append(f"{int(prefix):03d}")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid layer prefix: {prefix}") from exc
    if len(clean) > 8:
        raise argparse.ArgumentTypeError("uint8 mask bitfields support at most 8 layers")
    return clean


def active_layer_set(layer_sets: dict) -> dict:
    sets = layer_sets.get("sets")
    if not isinstance(sets, list):
        raise ValueError("layer_sets JSON is missing a sets list")
    active_id = layer_sets.get("activeSetId")
    for item in sets:
        if isinstance(item, dict) and item.get("id") == active_id:
            return item
    for item in sets:
        if isinstance(item, dict):
            return item
    raise ValueError("no usable layer set found")


def layer_prefix(layer: dict) -> str:
    return f"{int(layer.get('id')):03d}"


def rgb_keys_from_srgb(values: list) -> np.ndarray:
    if len(values) < 3:
        return np.zeros(0, dtype=np.uint32)
    arr = np.asarray(values, dtype=np.uint32)
    arr = arr[: (arr.size // 3) * 3].reshape(-1, 3)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if np.any(arr > 255):
        raise ValueError("srgb values must be uint8-sized")
    keys = (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]
    return np.unique(keys.astype(np.uint32))


def rgb_keys_from_hex(values: list) -> np.ndarray:
    keys: list[int] = []
    for raw in values:
        text = str(raw or "").strip().lstrip("#")
        if len(text) != 6:
            continue
        keys.append(int(text, 16))
    if not keys:
        return np.zeros(0, dtype=np.uint32)
    return np.unique(np.asarray(keys, dtype=np.uint32))


def rgb_keys_from_matched_values(layer: dict) -> np.ndarray:
    matched = layer.get("matchedColorValues")
    if not isinstance(matched, dict):
        return np.zeros(0, dtype=np.uint32)
    srgb = matched.get("srgb")
    if isinstance(srgb, list) and srgb:
        return rgb_keys_from_srgb(srgb)
    rgb_hex = matched.get("rgbHex")
    if isinstance(rgb_hex, list) and rgb_hex:
        return rgb_keys_from_hex(rgb_hex)
    return np.zeros(0, dtype=np.uint32)


def metadata_source_from_layer(layer: dict) -> dict:
    matched = layer.get("matchedColorValues")
    if not isinstance(matched, dict):
        return {}
    source = matched.get("source")
    return dict(source) if isinstance(source, dict) else {}


def build_lut(layer_sets_path: Path, prefixes: list[str]) -> tuple[np.ndarray, dict]:
    layer_sets = read_json(layer_sets_path)
    layer_set = active_layer_set(layer_sets)
    layers = layer_set.get("layers")
    if not isinstance(layers, list):
        raise ValueError("active layer set is missing layers")

    layers_by_prefix = {layer_prefix(layer): layer for layer in layers if isinstance(layer, dict) and layer.get("id") is not None}
    lut = np.zeros(1 << 24, dtype=np.uint8)
    layer_metadata: list[dict] = []

    for bit, prefix in enumerate(prefixes):
        layer = layers_by_prefix.get(prefix)
        if layer is None:
            raise ValueError(f"requested layer {prefix} was not found in active layer set")
        keys = rgb_keys_from_matched_values(layer)
        if keys.size == 0:
            raise ValueError(f"requested layer {prefix} has no matched RGB values")
        lut[keys] |= np.uint8(1 << bit)
        layer_metadata.append(
            {
                "prefix": prefix,
                "bit": bit,
                "name": str(layer.get("name") or prefix),
                "sourceLayerId": int(layer.get("id")),
                "rgbKeyCount": int(keys.size),
                "source": metadata_source_from_layer(layer),
            }
        )

    active_values = lut[lut != 0]
    metadata = {
        "schema": LUT_SCHEMA,
        "generatedAt": utc_now(),
        "decoder": DECODER_CONTRACT,
        "rgbKeyFormat": "0xRRGGBB packed from decoded uint8 RGB",
        "bitfieldDtype": "uint8",
        "layerSet": {
            "path": str(layer_sets_path),
            "sha256": sha256_file(layer_sets_path),
            "activeSetId": layer_sets.get("activeSetId"),
            "activeSetName": layer_set.get("name"),
        },
        "layers": layer_metadata,
        "summary": {
            "lutEntries": int(lut.size),
            "selectedRgbKeyCount": int(active_values.size),
            "overlapRgbKeyCount": int(np.count_nonzero(BIT_COUNTS[active_values] > 1)),
        },
    }
    return lut, metadata


def atomic_write_lut(path: Path, lut: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, lut=lut, metadata_json=np.array(metadata_json))
    os.replace(temp_path, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense RGB mask LUT artifact.")
    parser.add_argument("--layer-sets", type=Path, default=DEFAULT_LAYER_SETS, help="Path to layer_sets.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output color_lut_v1.npz path.")
    parser.add_argument(
        "--layers",
        type=parse_layer_prefixes,
        default=list(DEFAULT_LAYERS),
        help="Comma-separated layer prefixes to encode, in bit order.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and summarize without writing the LUT.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lut, metadata = build_lut(args.layer_sets.resolve(), list(args.layers))
        if not args.dry_run:
            atomic_write_lut(args.output.resolve(), lut, metadata)
    except Exception as exc:
        print(f"build_color_lut.py: error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": None if args.dry_run else str(args.output.resolve()),
                "layers": [{"prefix": item["prefix"], "bit": item["bit"], "rgbKeyCount": item["rgbKeyCount"]} for item in metadata["layers"]],
                "summary": metadata["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
