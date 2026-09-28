#!/usr/bin/env python3
"""Build bbox manifests from deterministic RGB maskbits artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
TOOLS_DIR = APP_DIR / "tools"
SRC_DIR = APP_DIR / "src"

from models.deterministic.tools.build_mask_bboxes import (  # noqa: E402
    DEFAULT_ASPECT_TOLERANCE,
    DEFAULT_CORNER_MIN_PIXELS,
    DEFAULT_EDGE_COVERAGE_MIN,
    DEFAULT_FIT_TIGHTNESS,
    DEFAULT_FOV_CLIP_ENABLED,
    DEFAULT_FOV_CLIP_MARGIN_PX,
    DEFAULT_FOV_CLIP_MIN_CONTACT_PIXELS,
    DEFAULT_FOV_CLIP_MIN_CONTACT_RATIO,
    DEFAULT_FOV_CLIP_REQUIRE_BBOX_TOUCH,
    DEFAULT_FOV_CLIP_WARN_ONLY,
    DEFAULT_MAX_BBOXES_PER_FRAME,
    DEFAULT_MIN_PIXELS,
    DEFAULT_OUTPUT_ROOT as DEFAULT_BASELINE_BBOX_ROOT,
    DEFAULT_QUAD_FIT_ENABLED,
    DEFAULT_QUAD_FIT_MODE,
    DEFAULT_QUAD_THICKNESS_PX,
    DEFAULT_TARGET_ASPECT,
    DEFAULT_VOID_OVERLAP_MAX_RATIO,
    DEFAULT_VOID_OVERLAP_MIN_PIXELS,
    QUAD_FIT_MODES,
    bbox_run_root,
    bbox_signature_settings,
    clean_float,
    collect_fov_clip_summary,
    component_bboxes,
    concise_layer_summaries,
    empty_fov_clip_summary,
    normalized_bbox_settings,
    utc_now,
)
from models.deterministic.tools.build_vision_memory import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REVIEW,
    DEFAULT_RULES,
    atomic_write_json,
    read_json,
    read_json_optional,
    selected_precompute_manifest,
    stable_hash,
)
from models.deterministic.src.mask_bbox.maskbits_input import (  # noqa: E402
    DEFAULT_MASK_MANIFEST,
    build_mask_frame_lookup,
    build_maskbits_context,
    layer_hits_from_mask_bits,
    load_mask_bits,
    mask_entry_for_frame,
    repo_url,
)


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "mask_bboxes_maskbits"


def maskbits_bbox_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return bbox_run_root(run_key, output_root)


def maskbits_bbox_source_signature(
    precompute_manifest: dict,
    precompute_manifest_path: Path,
    mask_manifest: dict,
    mask_manifest_path: Path,
    settings: dict,
    maskbits_signature: str,
) -> str:
    lut = mask_manifest.get("lut") if isinstance(mask_manifest.get("lut"), dict) else {}
    return stable_hash(
        {
            "kind": "0721-rgb-lut-maskbits-pixel-bboxes",
            "runKey": precompute_manifest.get("runKey"),
            "precomputeManifest": repo_url(precompute_manifest_path),
            "maskManifest": repo_url(mask_manifest_path),
            "maskFrameCount": int(mask_manifest.get("frameCount") or len(mask_manifest.get("frames") or [])),
            "frameCount": int(precompute_manifest.get("frameCount") or len(precompute_manifest.get("frames") or [])),
            "width": int(precompute_manifest.get("width") or 640),
            "height": int(precompute_manifest.get("height") or 360),
            "settings": bbox_signature_settings(settings),
            "lutSha256": str(lut.get("sha256") or ""),
            "maskbitsSignature": maskbits_signature,
        }
    )


def bbox_settings_from_args(args: argparse.Namespace) -> dict:
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
    return normalized_bbox_settings(raw_settings)


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = [int(value) for value in a]
    bx0, by0, bx1, by1 = [int(value) for value in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    intersection = iw * ih
    area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
    area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
    union = max(1, area_a + area_b - intersection)
    return float(intersection) / float(union)


def compare_frame_output(frame: dict, baseline_frame: dict | None) -> dict:
    if not baseline_frame:
        return {
            "available": False,
            "selectedPixelDelta": None,
            "bboxCountDelta": None,
            "maxCoordinateDeltaPx": None,
            "minIou": None,
        }
    bboxes = list(frame.get("bboxes") or [])
    baseline_bboxes = list(baseline_frame.get("bboxes") or [])
    max_delta = 0
    min_iou = 1.0 if bboxes and baseline_bboxes else (1.0 if not bboxes and not baseline_bboxes else 0.0)
    for bbox, baseline_bbox in zip(bboxes, baseline_bboxes):
        bbox_px = [int(value) for value in bbox.get("bboxPx") or [0, 0, 0, 0]]
        baseline_px = [int(value) for value in baseline_bbox.get("bboxPx") or [0, 0, 0, 0]]
        max_delta = max(max_delta, max(abs(left - right) for left, right in zip(bbox_px, baseline_px)))
        min_iou = min(min_iou, bbox_iou(bbox_px, baseline_px))
    return {
        "available": True,
        "baselineSelectedPixelCount": int(baseline_frame.get("selectedPixelCount") or 0),
        "selectedPixelDelta": int(frame.get("selectedPixelCount") or 0) - int(baseline_frame.get("selectedPixelCount") or 0),
        "baselineBboxCount": int(baseline_frame.get("bboxCount") or len(baseline_bboxes)),
        "bboxCountDelta": int(frame.get("bboxCount") or len(bboxes)) - int(baseline_frame.get("bboxCount") or len(baseline_bboxes)),
        "maxCoordinateDeltaPx": int(max_delta),
        "minIou": clean_float(min_iou, 6),
    }


def attach_comparison(frames_out: list[dict], baseline_manifest: dict | None) -> dict:
    if not baseline_manifest:
        return {"available": False, "reason": "baseline bbox manifest not found"}
    baseline_by_frame = {
        int(frame.get("frameOrdinal")): frame
        for frame in baseline_manifest.get("frames", [])
        if isinstance(frame, dict) and frame.get("frameOrdinal") is not None
    }
    selected_mismatches = 0
    bbox_count_mismatches = 0
    geometry_mismatches = 0
    max_coordinate_delta = 0
    min_iou = 1.0
    first_diffs = []
    for frame in frames_out:
        comparison = compare_frame_output(frame, baseline_by_frame.get(int(frame.get("frameOrdinal") or 0)))
        frame["comparison"] = comparison
        if not comparison.get("available"):
            continue
        selected_delta = int(comparison.get("selectedPixelDelta") or 0)
        bbox_delta = int(comparison.get("bboxCountDelta") or 0)
        coordinate_delta = int(comparison.get("maxCoordinateDeltaPx") or 0)
        iou = float(comparison.get("minIou") if comparison.get("minIou") is not None else 0.0)
        if selected_delta:
            selected_mismatches += 1
        if bbox_delta:
            bbox_count_mismatches += 1
        if coordinate_delta:
            geometry_mismatches += 1
        max_coordinate_delta = max(max_coordinate_delta, coordinate_delta)
        min_iou = min(min_iou, iou)
        if (selected_delta or bbox_delta or coordinate_delta) and len(first_diffs) < 20:
            first_diffs.append(
                {
                    "frameOrdinal": frame.get("frameOrdinal"),
                    "selectedPixelDelta": selected_delta,
                    "bboxCountDelta": bbox_delta,
                    "maxCoordinateDeltaPx": coordinate_delta,
                    "minIou": clean_float(iou, 6),
                }
            )
    return {
        "available": True,
        "baselineManifest": baseline_manifest.get("precomputeManifest"),
        "framesCompared": len(frames_out),
        "selectedPixelMismatches": selected_mismatches,
        "bboxCountMismatches": bbox_count_mismatches,
        "geometryMismatches": geometry_mismatches,
        "maxCoordinateDeltaPx": int(max_coordinate_delta),
        "minIou": clean_float(min_iou, 6),
        "firstDifferences": first_diffs,
        "passed": selected_mismatches == 0 and bbox_count_mismatches == 0 and geometry_mismatches == 0,
    }


def build_maskbits_bbox_frame(
    frame_entry: dict,
    frame_ordinal: int,
    mask_lookup: dict,
    mask_context: dict,
    width: int,
    height: int,
    settings: dict,
) -> tuple[dict, dict]:
    """Build the maskbits bbox record for one frame."""
    mask_entry = mask_entry_for_frame(mask_lookup, frame_entry, frame_ordinal)
    mask_bits = load_mask_bits(mask_entry, width, height)
    layer_hits = layer_hits_from_mask_bits(mask_bits, mask_context)
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
        overlap_bboxes.extend(list(bbox.get("overlapChildren") or []))
    frame_out = {
        "frameOrdinal": frame_ordinal,
        "frameId": frame_entry.get("frameId") or frame_entry.get("rawIndex") or frame_entry.get("index"),
        "path": frame_entry.get("path") or frame_entry.get("sourcePath"),
        "cachePath": frame_entry.get("cachePath"),
        "maskBits": mask_entry.get("maskBits") or mask_entry.get("path"),
        "selectedPixelCount": selected_pixel_count,
        "bboxCount": len(bboxes),
        "bboxes": bboxes,
        "quadAcceptedCount": frame_quad_accepted,
        "quadRejectedCount": frame_quad_rejected,
        "voidOverlapCount": frame_void_overlaps,
        "overlapBboxes": overlap_bboxes,
    }
    stats = {
        "bboxCount": len(bboxes),
        "quadAcceptedCount": frame_quad_accepted,
        "quadRejectedCount": frame_quad_rejected,
        "voidOverlapCount": frame_void_overlaps,
        "overlapChildCount": len(overlap_bboxes),
        "selectedPixelCount": selected_pixel_count,
    }
    return frame_out, stats


def build_maskbits_bboxes(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
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
        raise ValueError("maskbits bbox build requires frame entries in the precompute manifest")
    if int(mask_manifest.get("width") or width) != width or int(mask_manifest.get("height") or height) != height:
        raise ValueError("maskbits manifest dimensions do not match the active precompute manifest")
    mask_frame_count = int(mask_manifest.get("frameCount") or len(mask_manifest.get("frames") or []))
    if mask_frame_count < frame_count:
        raise ValueError(f"maskbits manifest has {mask_frame_count} frames, expected at least {frame_count}")

    settings = bbox_settings_from_args(args)
    mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
    source_signature = maskbits_bbox_source_signature(
        precompute_manifest,
        precompute_manifest_path,
        mask_manifest,
        mask_manifest_path,
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
        raise ValueError("selected maskbits bbox frame range is empty")

    output_root = maskbits_bbox_run_root(run_key, args.output_root)
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
        frame_out, frame_stats = build_maskbits_bbox_frame(
            frame_entry,
            frame_ordinal,
            lookup,
            mask_context,
            width,
            height,
            settings,
        )
        bboxes = list(frame_out.get("bboxes") or [])
        selected_pixel_count = int(frame_stats["selectedPixelCount"])
        frame_quad_accepted = int(frame_stats["quadAcceptedCount"])
        frame_quad_rejected = int(frame_stats["quadRejectedCount"])
        frame_void_overlaps = int(frame_stats["voidOverlapCount"])
        overlap_bboxes = list(frame_out.get("overlapBboxes") or [])
        if fov_clip_summary["enabled"]:
            fov_clip_severities.extend(collect_fov_clip_summary(fov_clip_summary, bboxes))
        total_bboxes += len(bboxes)
        total_quad_accepted += frame_quad_accepted
        total_quad_rejected += frame_quad_rejected
        total_void_overlaps += frame_void_overlaps
        total_overlap_children += len(overlap_bboxes)
        max_bboxes = max(max_bboxes, len(bboxes))
        total_selected_pixels += selected_pixel_count
        frames_out.append(frame_out)
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-maskbits-bboxes",
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

    baseline_path = args.compare_manifest
    if baseline_path is None:
        baseline_path = bbox_run_root(run_key, DEFAULT_BASELINE_BBOX_ROOT) / "bbox_manifest.json"
    baseline_manifest = read_json_optional(baseline_path)
    comparison = attach_comparison(frames_out, baseline_manifest)

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
        "comparison": comparison,
    }
    if fov_clip_summary["enabled"]:
        summary["fovClip"] = fov_clip_summary

    output = {
        "version": 1,
        "app": "vision_passthrough_review",
        "kind": "color-mask-pixel-bboxes",
        "createdAt": utc_now(),
        "runKey": run_key,
        "precomputeManifest": repo_url(precompute_manifest_path),
        "maskManifest": repo_url(mask_manifest_path),
        "sourceSignature": source_signature,
        "complete": start == 0 and len(frames_out) >= frame_count,
        "frameStart": start,
        "frameCount": len(frames_out),
        "sourceFrameCount": frame_count,
        "image": {"width": width, "height": height},
        "settings": settings,
        "input": {
            "source": "rgb-lut-maskbits",
            "maskManifest": repo_url(mask_manifest_path),
            "maskSignature": mask_context["signature"],
            "maskManifestSha256": mask_context["maskManifestSha256"],
            "lutSha256": mask_context["lutSha256"],
            "enabledPrefixes": mask_context["enabledPrefixes"],
            "layerSummaries": concise_layer_summaries(mask_context["summaries"]),
        },
        **({"fovClipSummary": fov_clip_summary} if fov_clip_summary["enabled"] else {}),
        "comparison": comparison,
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
    parser.add_argument("--mask-manifest", type=Path, default=DEFAULT_MASK_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--compare-manifest", type=Path, default=None)
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
    result = build_maskbits_bboxes(args)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": repo_url(maskbits_bbox_run_root(result["runKey"], args.output_root) / "bbox_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "comparison": result.get("comparison"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
