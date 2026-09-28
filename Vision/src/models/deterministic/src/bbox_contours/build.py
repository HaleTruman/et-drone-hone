#!/usr/bin/env python3
"""Build bbox-local contour hierarchy from maskbits bboxes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Callable

import numpy as np

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
TOOLS_DIR = APP_DIR / "tools"
SRC_DIR = APP_DIR / "src"

from models.deterministic.tools.build_mask_bboxes import concise_layer_summaries  # noqa: E402
from models.deterministic.tools.build_mask_contours import (  # noqa: E402
    DEFAULT_INCLUDE_SMALL_CONTOURS,
    DEFAULT_MASK_SOURCE,
    DEFAULT_MAX_VOIDS_PER_BBOX,
    DEFAULT_MIN_OUTER_AREA_PX,
    DEFAULT_MIN_VOID_AREA_PX,
    contour_counts_for_objects,
    contour_hierarchy_for_bbox,
    normalized_contour_settings,
    source_mask_for_settings,
)
from models.deterministic.tools.build_vision_memory import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REVIEW,
    atomic_write_json,
    read_json,
    selected_precompute_manifest,
    stable_hash,
)
from models.deterministic.tools.build_mask_corners import bbox_frame_map  # noqa: E402
from models.deterministic.src.mask_bbox.build_from_maskbits import (  # noqa: E402
    DEFAULT_MASK_MANIFEST,
    DEFAULT_OUTPUT_ROOT as DEFAULT_MASKBITS_BBOX_ROOT,
    maskbits_bbox_run_root,
)
from models.deterministic.src.mask_bbox.maskbits_input import (  # noqa: E402
    build_mask_frame_lookup,
    build_maskbits_context,
    layer_hits_from_mask_bits,
    load_mask_bits,
    mask_entry_for_frame,
    repo_url,
)


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "bbox_contours"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or "").strip())[:160]


def bbox_contour_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build bbox contour artifacts without an active run key")
    return output_root / clean


def bbox_contour_source_signature(
    precompute_manifest: dict,
    precompute_manifest_path: Path,
    mask_manifest: dict,
    mask_manifest_path: Path,
    bbox_manifest: dict,
    settings: dict,
    maskbits_signature: str,
) -> str:
    return stable_hash(
        {
            "kind": "0721-maskbits-bbox-local-contours",
            "runKey": precompute_manifest.get("runKey"),
            "precomputeManifest": repo_url(precompute_manifest_path),
            "maskManifest": repo_url(mask_manifest_path),
            "bboxManifest": bbox_manifest.get("sourceSignature"),
            "frameCount": int(precompute_manifest.get("frameCount") or len(precompute_manifest.get("frames") or [])),
            "maskFrameCount": int(mask_manifest.get("frameCount") or len(mask_manifest.get("frames") or [])),
            "width": int(precompute_manifest.get("width") or mask_manifest.get("width") or 640),
            "height": int(precompute_manifest.get("height") or mask_manifest.get("height") or 360),
            "settings": normalized_contour_settings(settings),
            "maskbitsSignature": maskbits_signature,
        }
    )


def contour_settings_from_args(args: argparse.Namespace) -> dict:
    raw = getattr(args, "contour_settings", None)
    if raw is not None:
        return normalized_contour_settings(raw)
    return normalized_contour_settings(
        {
            "maskSource": getattr(args, "mask_source", DEFAULT_MASK_SOURCE),
            "minOuterAreaPx": getattr(args, "min_outer_area_px", DEFAULT_MIN_OUTER_AREA_PX),
            "minVoidAreaPx": getattr(args, "min_void_area_px", DEFAULT_MIN_VOID_AREA_PX),
            "maxVoidsPerBbox": getattr(args, "max_voids_per_bbox", DEFAULT_MAX_VOIDS_PER_BBOX),
            "simplifyEpsilonPx": getattr(args, "simplify_epsilon_px", 1.5),
            "closeRadiusPx": getattr(args, "close_radius_px", 0),
            "openRadiusPx": getattr(args, "open_radius_px", 0),
            "notchProximityPx": getattr(args, "notch_proximity_px", 4),
            "includeSmallContours": getattr(args, "include_small_contours", DEFAULT_INCLUDE_SMALL_CONTOURS),
        }
    )


def load_bbox_manifest(run_key: str, root: Path) -> tuple[dict, Path]:
    path = maskbits_bbox_run_root(run_key, root) / "bbox_manifest.json"
    if not path.exists():
        raise ValueError("maskbits bbox manifest is required before bbox contours can be built")
    return read_json(path), path


def build_bbox_contour_frame(
    cv2,
    frame_entry: dict,
    frame_ordinal: int,
    bbox_frame: dict,
    mask_lookup: dict,
    mask_context: dict,
    width: int,
    height: int,
    settings: dict,
) -> dict:
    bboxes = list(bbox_frame.get("bboxes") or [])
    mask_entry = mask_entry_for_frame(mask_lookup, frame_entry, frame_ordinal)
    mask_bits = load_mask_bits(mask_entry, width, height)
    layer_hits = layer_hits_from_mask_bits(mask_bits, mask_context)
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
    selected_pixel_count = int(np.count_nonzero(source_mask))
    return {
        "frameOrdinal": frame_ordinal,
        "frameId": frame_entry.get("frameId") or frame_entry.get("rawIndex") or frame_entry.get("index"),
        "path": frame_entry.get("path") or frame_entry.get("sourcePath"),
        "maskBits": mask_entry.get("maskBits") or mask_entry.get("path"),
        "selectedPixelCount": selected_pixel_count,
        **counts,
        "objects": objects,
    }


def build_bbox_contours(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for bbox contour hierarchy: {error}") from error

    config = read_json(args.config)
    review = read_json(args.review) if args.review.exists() else {"annotations": [], "decisions": []}
    precompute_manifest_path = args.precompute_manifest or selected_precompute_manifest()
    precompute_manifest = read_json(precompute_manifest_path)
    mask_manifest_path = args.mask_manifest
    mask_manifest = read_json(mask_manifest_path)
    run_key = str(precompute_manifest.get("runKey") or "")
    width = int(precompute_manifest.get("width") or mask_manifest.get("width") or 640)
    height = int(precompute_manifest.get("height") or mask_manifest.get("height") or 360)
    frames_in = list(precompute_manifest.get("frames") or [])
    frame_count = int(precompute_manifest.get("frameCount") or len(frames_in))
    if not frames_in:
        raise ValueError("bbox contour build requires frame entries in the precompute manifest")
    if int(mask_manifest.get("width") or width) != width or int(mask_manifest.get("height") or height) != height:
        raise ValueError("maskbits manifest dimensions do not match the active precompute manifest")

    settings = contour_settings_from_args(args)
    mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
    bbox_manifest, bbox_manifest_path = load_bbox_manifest(run_key, args.bbox_output_root)
    bbox_frames = bbox_frame_map(bbox_manifest)
    source_signature = bbox_contour_source_signature(
        precompute_manifest,
        precompute_manifest_path,
        mask_manifest,
        mask_manifest_path,
        bbox_manifest,
        settings,
        mask_context["signature"],
    )
    lookup = build_mask_frame_lookup(mask_manifest)

    start = max(0, int(args.start_frame or 0))
    end = frame_count
    if args.max_frames is not None:
        end = min(end, start + max(0, int(args.max_frames)))
    selected_frames = list(enumerate(frames_in[start:end], start=start))
    if not selected_frames:
        raise ValueError("selected bbox contour frame range is empty")

    output_root = bbox_contour_run_root(run_key, args.output_root)
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
            raise ValueError(f"maskbits bbox manifest has no frame entry for frame {frame_ordinal}")
        frame_out = build_bbox_contour_frame(
            cv2,
            frame_entry,
            frame_ordinal,
            bbox_frame,
            lookup,
            mask_context,
            width,
            height,
            settings,
        )
        counts = contour_counts_for_objects(list(frame_out.get("objects") or []))
        for key, value in counts.items():
            totals[key] += value
        max_voids = max(max_voids, counts["voidCount"])
        selected_pixel_count = int(frame_out.get("selectedPixelCount") or 0)
        total_selected_pixels += selected_pixel_count
        frames_out.append(frame_out)
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-bbox-contours",
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
        "kind": "bbox-local-contour-hierarchy",
        "createdAt": utc_now(),
        "runKey": run_key,
        "precomputeManifest": repo_url(precompute_manifest_path),
        "maskManifest": repo_url(mask_manifest_path),
        "bboxManifest": repo_url(bbox_manifest_path),
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
            "bboxSourceSignature": bbox_manifest.get("sourceSignature"),
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
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    parser.add_argument("--mask-manifest", type=Path, default=DEFAULT_MASK_MANIFEST)
    parser.add_argument("--bbox-output-root", type=Path, default=DEFAULT_MASKBITS_BBOX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--mask-source", choices=["enabled-layers", "002-only"], default=DEFAULT_MASK_SOURCE)
    parser.add_argument("--min-outer-area-px", type=int, default=DEFAULT_MIN_OUTER_AREA_PX)
    parser.add_argument("--min-void-area-px", type=int, default=DEFAULT_MIN_VOID_AREA_PX)
    parser.add_argument("--max-voids-per-bbox", type=int, default=DEFAULT_MAX_VOIDS_PER_BBOX)
    parser.add_argument("--simplify-epsilon-px", type=float, default=1.5)
    parser.add_argument("--close-radius-px", type=int, default=0)
    parser.add_argument("--open-radius-px", type=int, default=0)
    parser.add_argument("--notch-proximity-px", type=int, default=4)
    parser.add_argument("--include-small-contours", action="store_true", default=DEFAULT_INCLUDE_SMALL_CONTOURS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bbox_contours(args)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": repo_url(bbox_contour_run_root(result["runKey"], args.output_root) / "contours_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
