#!/usr/bin/env python3
"""Apply a dense RGB mask LUT to source frames.

This runtime is intentionally independent from the existing vision review
pipeline. It accepts decoded-frame manifests or plain frame folders, decodes
images with PIL RGB bytes, and writes one uint8 mask bitfield per frame.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LUT = Path(__file__).resolve().parent / "artifacts" / "color_lut_v1.npz"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "input_frames"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_MANIFEST_NAME = "mask_manifest.json"
JPEG_SUFFIXES = {".jpg", ".jpeg"}
RUN_SCHEMA = "color-mask-run.v1"
DECODER_CONTRACT = "PIL.Image.open(path).convert('RGB')"

BIT_COUNTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


@dataclass(frozen=True)
class FrameRef:
    index: int
    frame_id: str
    path: Path
    source_record: dict


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def atomic_write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("wb") as handle:
        np.ascontiguousarray(array).tofile(handle)
    os.replace(temp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(raw: np.ndarray | str | bytes) -> dict:
    if isinstance(raw, np.ndarray):
        value = raw.item() if raw.shape == () else raw.tolist()
    else:
        value = raw
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("LUT metadata_json must be a JSON string")
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError("LUT metadata_json must decode to an object")
    return metadata


def load_lut(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"LUT artifact not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "lut" not in payload:
            raise ValueError(f"{path} is missing the 'lut' array")
        lut = payload["lut"].astype(np.uint8, copy=False)
        metadata = parse_metadata(payload["metadata_json"]) if "metadata_json" in payload else {}
    if lut.dtype != np.uint8:
        raise ValueError("LUT must have dtype uint8")
    if lut.shape != (1 << 24,):
        raise ValueError(f"LUT must have shape {(1 << 24,)}, got {lut.shape}")
    return lut, metadata


def layers_from_metadata(metadata: dict) -> list[dict]:
    raw_layers = metadata.get("layers") if isinstance(metadata, dict) else None
    if not isinstance(raw_layers, list) or not raw_layers:
        return [{"prefix": f"bit-{bit}", "name": f"bit {bit}", "bit": bit} for bit in range(8)]
    layers: list[dict] = []
    for item in raw_layers:
        if not isinstance(item, dict):
            continue
        bit = int(item.get("bit", len(layers)))
        if bit < 0 or bit > 7:
            raise ValueError(f"layer bit must be in uint8 range 0..7, got {bit}")
        layers.append(
            {
                "prefix": str(item.get("prefix") or f"bit-{bit}"),
                "name": str(item.get("name") or item.get("prefix") or f"bit {bit}"),
                "bit": bit,
            }
        )
    return sorted(layers, key=lambda item: int(item["bit"]))


def repo_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/" + resolved.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_frame_path(raw_path: str, manifest_path: Path | None, frames_dir: Path | None, repo_root: Path) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError("frame record is missing a path")

    path = Path(text)
    if path.is_absolute() and path.exists():
        return path
    if text.startswith("/0721Vision/"):
        mapped = repo_root / text.removeprefix("/0721Vision/")
        if mapped.exists():
            return mapped
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if frames_dir is not None:
        candidates.append(frames_dir / path)
    if manifest_path is not None:
        candidates.append(manifest_path.parent / path)
    candidates.append(repo_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def frame_refs_from_manifest(manifest_path: Path, frames_dir: Path | None, repo_root: Path) -> list[FrameRef]:
    manifest = read_json(manifest_path)
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{manifest_path} does not contain a frames list")
    refs: list[FrameRef] = []
    for ordinal, item in enumerate(frames):
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path") or item.get("source") or item.get("relativePath") or item.get("filename")
        path = resolve_frame_path(str(raw_path or ""), manifest_path, frames_dir, repo_root)
        index = int(item.get("index", ordinal))
        frame_id = str(item.get("frameId", index))
        refs.append(FrameRef(index=index, frame_id=frame_id, path=path, source_record=dict(item)))
    return refs


def frame_refs_from_dir(frames_dir: Path) -> list[FrameRef]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"frame directory not found: {frames_dir}")
    paths = sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES)
    return [
        FrameRef(index=ordinal, frame_id=path.stem, path=path, source_record={"filename": path.name})
        for ordinal, path in enumerate(paths)
    ]


def discover_frame_refs(manifest_path: Path | None, frames_dir: Path | None, repo_root: Path) -> list[FrameRef]:
    if manifest_path is not None:
        return frame_refs_from_manifest(manifest_path, frames_dir, repo_root)
    if frames_dir is None:
        raise ValueError("either --manifest or --frames-dir is required")
    return frame_refs_from_dir(frames_dir)


def decode_rgb_keys(frame_path: Path) -> tuple[np.ndarray, int, int]:
    with Image.open(frame_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"{frame_path} did not decode to RGB")
    keys = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )
    height, width = keys.shape
    return keys, width, height


def apply_lut_to_rgb_keys(lut: np.ndarray, rgb_keys: np.ndarray) -> np.ndarray:
    return lut[rgb_keys].astype(np.uint8, copy=False)


def layer_pixel_counts(mask_bits: np.ndarray, layers: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer in layers:
        bit = int(layer["bit"])
        counts[str(layer["prefix"])] = int(np.count_nonzero(mask_bits & np.uint8(1 << bit)))
    return counts


def mask_stats(mask_bits: np.ndarray, layers: list[dict]) -> dict:
    selected = mask_bits != 0
    return {
        "selectedPixels": int(np.count_nonzero(selected)),
        "unselectedPixels": int(mask_bits.size - np.count_nonzero(selected)),
        "overlapPixels": int(np.count_nonzero(BIT_COUNTS[mask_bits] > 1)),
        "layerPixelCounts": layer_pixel_counts(mask_bits, layers),
    }


def output_name_for_frame(frame: FrameRef) -> str:
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(frame.frame_id))
    if not safe_id:
        safe_id = f"{frame.index:06d}"
    return f"frame_{frame.index:06d}_{safe_id}.maskbits.u8.bin"


def process_frame(frame: FrameRef, lut: np.ndarray, layers: list[dict], masks_dir: Path) -> tuple[dict, dict]:
    if not frame.path.exists():
        raise FileNotFoundError(f"frame not found: {frame.path}")
    rgb_keys, width, height = decode_rgb_keys(frame.path)
    mask_bits = apply_lut_to_rgb_keys(lut, rgb_keys)
    output_name = output_name_for_frame(frame)
    output_path = masks_dir / output_name
    atomic_write_array(output_path, mask_bits)
    stats = mask_stats(mask_bits, layers)
    entry = {
        "index": int(frame.index),
        "frameId": str(frame.frame_id),
        "source": repo_display_path(frame.path),
        "maskBits": repo_display_path(output_path),
        "width": int(width),
        "height": int(height),
        "dtype": "uint8",
        "layout": "row-major",
        "sha1": hashlib.sha1(np.ascontiguousarray(mask_bits).tobytes()).hexdigest(),
        **stats,
    }
    return entry, stats


def empty_summary(layers: list[dict]) -> dict:
    return {
        "framesProcessed": 0,
        "selectedPixels": 0,
        "unselectedPixels": 0,
        "overlapPixels": 0,
        "layerPixelCounts": {str(layer["prefix"]): 0 for layer in layers},
    }


def merge_summary(total: dict, stats: dict) -> None:
    total["framesProcessed"] += 1
    total["selectedPixels"] += int(stats["selectedPixels"])
    total["unselectedPixels"] += int(stats["unselectedPixels"])
    total["overlapPixels"] += int(stats["overlapPixels"])
    for prefix, count in stats["layerPixelCounts"].items():
        total["layerPixelCounts"][str(prefix)] = total["layerPixelCounts"].get(str(prefix), 0) + int(count)


def build_manifest(lut_path: Path, lut_metadata: dict, layers: list[dict], entries: list[dict], summary: dict) -> dict:
    widths = {entry["width"] for entry in entries}
    heights = {entry["height"] for entry in entries}
    return {
        "schema": RUN_SCHEMA,
        "generatedAt": utc_now(),
        "decoder": DECODER_CONTRACT,
        "lut": {
            "path": repo_display_path(lut_path),
            "sha256": sha256_file(lut_path),
            "metadata": lut_metadata,
        },
        "layers": layers,
        "frameCount": len(entries),
        "width": next(iter(widths)) if len(widths) == 1 else None,
        "height": next(iter(heights)) if len(heights) == 1 else None,
        "summary": summary,
        "frames": entries,
    }


def process_frames(
    frames: Iterable[FrameRef],
    lut_path: Path,
    lut: np.ndarray,
    lut_metadata: dict,
    output_dir: Path,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> dict:
    layers = layers_from_metadata(lut_metadata)
    masks_dir = output_dir / "masks"
    entries: list[dict] = []
    summary = empty_summary(layers)
    for frame in frames:
        entry, stats = process_frame(frame, lut, layers, masks_dir)
        entries.append(entry)
        merge_summary(summary, stats)
    manifest = build_manifest(lut_path, lut_metadata, layers, entries, summary)
    atomic_write_json(output_dir / manifest_name, manifest)
    return manifest


def process_frames_watch(
    manifest_path: Path | None,
    frames_dir: Path | None,
    repo_root: Path,
    lut_path: Path,
    lut: np.ndarray,
    lut_metadata: dict,
    output_dir: Path,
    manifest_name: str,
    poll_seconds: float,
    max_frames: int | None,
) -> dict:
    layers = layers_from_metadata(lut_metadata)
    masks_dir = output_dir / "masks"
    entries: list[dict] = []
    summary = empty_summary(layers)
    seen: set[str] = set()
    try:
        while True:
            refs = discover_frame_refs(manifest_path, frames_dir, repo_root)
            new_refs = [ref for ref in refs if str(ref.path.resolve()) not in seen]
            if max_frames is not None:
                remaining = max(0, max_frames - len(entries))
                new_refs = new_refs[:remaining]
            for frame in new_refs:
                entry, stats = process_frame(frame, lut, layers, masks_dir)
                entries.append(entry)
                merge_summary(summary, stats)
                seen.add(str(frame.path.resolve()))
            manifest = build_manifest(lut_path, lut_metadata, layers, entries, summary)
            atomic_write_json(output_dir / manifest_name, manifest)
            if max_frames is not None and len(entries) >= max_frames:
                return manifest
            time.sleep(max(0.1, poll_seconds))
    except KeyboardInterrupt:
        manifest = build_manifest(lut_path, lut_metadata, layers, entries, summary)
        atomic_write_json(output_dir / manifest_name, manifest)
        return manifest


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply deterministic RGB LUT masks to JPEG frames.")
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT, help="Path to color_lut_v1.npz.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder of .jpg/.jpeg frames.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON manifest with a frames list.")
    parser.add_argument("--repo-root", type=Path, default=APP_ROOT, help="Root used to resolve /0721Vision/... manifest paths.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for mask files and manifest.")
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME, help="Output manifest filename.")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="Process at most this many frames.")
    parser.add_argument("--watch", action="store_true", help="Poll the input source and process new frames until interrupted.")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Watch-mode polling interval.")
    parser.add_argument("--quiet", action="store_true", help="Only print errors.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lut_path = args.lut.resolve()
    output_dir = args.output.resolve()
    frames_dir = args.frames_dir.resolve() if args.frames_dir is not None else None
    manifest_path = args.manifest.resolve() if args.manifest is not None else None
    repo_root = args.repo_root.resolve()

    try:
        lut, lut_metadata = load_lut(lut_path)
        if args.watch:
            manifest = process_frames_watch(
                manifest_path,
                frames_dir,
                repo_root,
                lut_path,
                lut,
                lut_metadata,
                output_dir,
                args.manifest_name,
                args.poll_seconds,
                args.max_frames,
            )
        else:
            refs = discover_frame_refs(manifest_path, frames_dir, repo_root)
            if args.max_frames is not None:
                refs = refs[: args.max_frames]
            manifest = process_frames(refs, lut_path, lut, lut_metadata, output_dir, args.manifest_name)
    except Exception as exc:
        print(f"mask_frames.py: error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        summary = manifest["summary"]
        print(
            json.dumps(
                {
                    "manifest": str(output_dir / args.manifest_name),
                    "framesProcessed": summary["framesProcessed"],
                    "selectedPixels": summary["selectedPixels"],
                    "overlapPixels": summary["overlapPixels"],
                    "layerPixelCounts": summary["layerPixelCounts"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
