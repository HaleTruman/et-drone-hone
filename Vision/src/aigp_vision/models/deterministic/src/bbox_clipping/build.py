#!/usr/bin/env python3
"""Build bbox-local FOV clipping diagnostics from maskbits bboxes."""

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

from aigp_vision.models.deterministic.tools.build_mask_bboxes import (  # noqa: E402
    clean_float,
    collect_fov_clip_summary,
    empty_fov_clip_summary,
    fov_clip_for_component,
    normalized_fov_clip_settings,
)
from aigp_vision.models.deterministic.tools.build_mask_corners import bbox_frame_map, component_mask_for_bbox  # noqa: E402
from aigp_vision.models.deterministic.tools.build_vision_memory import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REVIEW,
    atomic_write_json,
    read_json,
    selected_precompute_manifest,
    stable_hash,
)
from aigp_vision.models.deterministic.src.mask_bbox.build_from_maskbits import (  # noqa: E402
    DEFAULT_MASK_MANIFEST,
    DEFAULT_OUTPUT_ROOT as DEFAULT_MASKBITS_BBOX_ROOT,
    maskbits_bbox_run_root,
)
from aigp_vision.models.deterministic.src.mask_bbox.maskbits_input import (  # noqa: E402
    build_mask_frame_lookup,
    build_maskbits_context,
    layer_hits_from_mask_bits,
    load_mask_bits,
    mask_entry_for_frame,
    repo_url,
)


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "bbox_clipping"
DEFAULT_FOV_CLIP_ENABLED = True
DEFAULT_FOV_CLIP_MARGIN_PX = 22
DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS = 306
DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO = 0.055
DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH = True
DEFAULT_FOV_CLIP_WARN_ONLY = True


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or "").strip())[:160]


def clipping_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build clipping artifacts without an active run key")
    return output_root / clean


def clipping_source_signature(
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
            "kind": "0721-maskbits-bbox-local-clipping",
            "runKey": precompute_manifest.get("runKey"),
            "precomputeManifest": repo_url(precompute_manifest_path),
            "maskManifest": repo_url(mask_manifest_path),
            "bboxManifest": bbox_manifest.get("sourceSignature"),
            "frameCount": int(precompute_manifest.get("frameCount") or len(precompute_manifest.get("frames") or [])),
            "maskFrameCount": int(mask_manifest.get("frameCount") or len(mask_manifest.get("frames") or [])),
            "width": int(precompute_manifest.get("width") or mask_manifest.get("width") or 640),
            "height": int(precompute_manifest.get("height") or mask_manifest.get("height") or 360),
            "settings": normalized_fov_clip_settings(settings),
            "maskbitsSignature": maskbits_signature,
        }
    )


def clipping_settings_from_args(args: argparse.Namespace) -> dict:
    raw = getattr(
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
    )
    return normalized_fov_clip_settings(raw)


def load_bbox_manifest(run_key: str, root: Path) -> tuple[dict, Path]:
    path = maskbits_bbox_run_root(run_key, root) / "bbox_manifest.json"
    if not path.exists():
        raise ValueError("maskbits bbox manifest is required before clipping can be built")
    return read_json(path), path


def build_bbox_clipping_frame(
    cv2,
    frame_entry: dict,
    frame_ordinal: int,
    bbox_frame: dict,
    mask_lookup: dict,
    mask_context: dict,
    width: int,
    height: int,
    settings: dict,
) -> tuple[dict, dict]:
    """Build bbox-local clipping diagnostics for one frame."""
    bboxes = list(bbox_frame.get("bboxes") or [])
    mask_entry = mask_entry_for_frame(mask_lookup, frame_entry, frame_ordinal)
    mask_bits = load_mask_bits(mask_entry, width, height)
    layer_hits = layer_hits_from_mask_bits(mask_bits, mask_context)
    selected = np.any(layer_hits, axis=0)
    _, component_labels, _, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
    objects = []
    for bbox in bboxes:
        component_mask, component_label = component_mask_for_bbox(selected, component_labels, bbox, width, height)
        ys, xs = np.nonzero(component_mask)
        pixel_count = int(xs.size)
        fov_clip = None
        if pixel_count:
            fov_clip = fov_clip_for_component(xs, ys, bbox.get("bboxPx") or [0, 0, 0, 0], pixel_count, width, height, {"fovClip": settings})
        objects.append(
            {
                "bboxId": bbox.get("bboxId"),
                "frameOrdinal": frame_ordinal,
                "bboxPx": bbox.get("bboxPx"),
                "bboxUv": bbox.get("bboxUv"),
                "pixelCount": pixel_count,
                "bboxPixelCount": bbox.get("pixelCount"),
                "componentLabel": component_label,
                "fovClip": fov_clip,
            }
        )
    clipped = sum(1 for item in objects if (item.get("fovClip") or {}).get("status") == "clipped")
    near = sum(1 for item in objects if (item.get("fovClip") or {}).get("status") == "near-edge")
    frame_out = {
        "frameOrdinal": frame_ordinal,
        "frameId": frame_entry.get("frameId") or frame_entry.get("rawIndex") or frame_entry.get("index"),
        "path": frame_entry.get("path") or frame_entry.get("sourcePath"),
        "maskBits": mask_entry.get("maskBits") or mask_entry.get("path"),
        "bboxCount": len(objects),
        "clippedCount": clipped,
        "nearEdgeCount": near,
        "objects": objects,
    }
    return frame_out, {"objectCount": len(objects), "clippedCount": clipped, "nearEdgeCount": near}


def build_bbox_clipping(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for bbox clipping diagnostics: {error}") from error

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
        raise ValueError("clipping build requires frame entries in the precompute manifest")
    if int(mask_manifest.get("width") or width) != width or int(mask_manifest.get("height") or height) != height:
        raise ValueError("maskbits manifest dimensions do not match the active precompute manifest")

    settings = clipping_settings_from_args(args)
    mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
    bbox_manifest, bbox_manifest_path = load_bbox_manifest(run_key, args.bbox_output_root)
    bbox_frames = bbox_frame_map(bbox_manifest)
    source_signature = clipping_source_signature(
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
        raise ValueError("selected clipping frame range is empty")

    output_root = clipping_run_root(run_key, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frames_out = []
    summary = empty_fov_clip_summary(bool(settings.get("enabled")))
    severities: list[float] = []
    total_objects = 0
    for local_index, (frame_ordinal, frame_entry) in enumerate(selected_frames):
        bbox_frame = bbox_frames.get(frame_ordinal)
        if not bbox_frame:
            raise ValueError(f"maskbits bbox manifest has no frame entry for frame {frame_ordinal}")
        frame_out, frame_stats = build_bbox_clipping_frame(
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
        objects = list(frame_out.get("objects") or [])
        severities.extend(collect_fov_clip_summary(summary, objects))
        clipped = int(frame_stats["clippedCount"])
        near = int(frame_stats["nearEdgeCount"])
        total_objects += len(objects)
        frames_out.append(frame_out)
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-bbox-clipping",
                    "index": local_index + 1,
                    "total": len(selected_frames),
                    "frameOrdinal": frame_ordinal,
                    "bboxCount": len(objects),
                    "clippedCount": clipped,
                    "nearEdgeCount": near,
                }
            )

    if summary["enabled"]:
        summary["maxSeverity"] = clean_float(max(severities) if severities else 0.0, 4)
        summary["meanSeverity"] = clean_float(float(np.mean(severities)) if severities else 0.0, 4)

    output = {
        "version": 1,
        "app": "vision_passthrough_review",
        "kind": "bbox-local-clipping",
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
            "source": "rgb-lut-maskbits-bboxes",
            "maskSignature": mask_context["signature"],
            "bboxSourceSignature": bbox_manifest.get("sourceSignature"),
            "enabledPrefixes": mask_context["enabledPrefixes"],
        },
        "summary": {
            **summary,
            "objectCount": total_objects,
            "meanObjectsPerFrame": total_objects / len(frames_out) if frames_out else 0,
        },
        "frames": frames_out,
    }
    atomic_write_json(output_root / "clipping_manifest.json", output)
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
    parser.add_argument("--fov-clip-enabled", action=argparse.BooleanOptionalAction, default=DEFAULT_FOV_CLIP_ENABLED)
    parser.add_argument("--fov-clip-margin-px", type=int, default=DEFAULT_FOV_CLIP_MARGIN_PX)
    parser.add_argument("--fov-clip-min-contact-pixels", type=int, default=DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS)
    parser.add_argument("--fov-clip-min-contact-ratio", type=float, default=DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO)
    parser.add_argument("--fov-clip-require-bbox-touch", action=argparse.BooleanOptionalAction, default=DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH)
    parser.add_argument("--fov-clip-warn-only", action=argparse.BooleanOptionalAction, default=DEFAULT_FOV_CLIP_WARN_ONLY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bbox_clipping(args)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": repo_url(clipping_run_root(result["runKey"], args.output_root) / "clipping_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
