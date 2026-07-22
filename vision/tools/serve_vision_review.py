#!/usr/bin/env python3
"""Serve the 0721 vision review UI with local persistence APIs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse


SCRIPT_PATH = Path(__file__).resolve()
APP_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
TOOLS_DIR = SCRIPT_PATH.parent
SRC_DIR = APP_DIR / "src"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_vision_memory import (
    DEFAULT_FRAME_RUNS_ROOT,
    DEFAULT_HNL_LAYER_SETS,
    HNL_DECODED_MANIFEST_NAME,
    active_frame_run_name,
    build_memory,
    precompute_manifest_for_decoded,
    read_json,
    read_json_optional,
    selected_precompute_manifest,
    stable_hash,
)
from build_mask_bboxes import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_BBOX_ROOT,
    bbox_signature_settings,
    bbox_run_root,
    bbox_settings_from_review,
    bbox_source_signature,
    build_mask_bboxes,
    build_mask_context,
    normalized_bbox_settings,
    normalized_fov_clip_settings,
)
from mask_bbox.build_from_maskbits import (
    DEFAULT_MASK_MANIFEST,
    DEFAULT_OUTPUT_ROOT as DEFAULT_MASKBITS_BBOX_ROOT,
    build_maskbits_bboxes,
    maskbits_bbox_run_root,
    maskbits_bbox_source_signature,
)
from mask_bbox.maskbits_input import build_mask_frame_lookup, build_maskbits_context
from bbox_clipping.build import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_BBOX_CLIPPING_ROOT,
    build_bbox_clipping,
    clipping_run_root,
    clipping_source_signature,
)
from bbox_contours.build import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_BBOX_CONTOUR_ROOT,
    build_bbox_contour_frame,
    bbox_contour_run_root,
    bbox_contour_source_signature,
    build_bbox_contours,
)
from build_mask_corners import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_CORNER_ROOT,
    bbox_frame_map,
    build_mask_corners,
    corner_run_root,
    corner_settings_from_review,
    corner_source_signature,
    normalized_corner_settings,
)
from build_mask_contours import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_CONTOUR_ROOT,
    build_mask_contours,
    contour_run_root,
    contour_settings_from_review,
    contour_source_signature,
    normalized_contour_settings,
)
from build_square_pose import (
    CAMERA as SQUARE_POSE_CAMERA,
    DEFAULT_OUTPUT_ROOT as DEFAULT_SQUARE_POSE_ROOT,
    build_square_pose,
    normalized_square_pose_settings,
    square_pose_run_root,
    square_pose_settings_from_review,
    square_pose_source_signature,
)
from build_instance_provenance import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_INSTANCE_ROOT,
    build_instance_provenance,
    instance_run_root,
    instance_signature_settings,
    instance_settings_from_review,
    instance_source_signature,
    normalized_instance_settings,
)
from pose_estimation.pose_fit import (
    CAMERA as POSE_ESTIMATION_CAMERA,
    DEFAULT_OUTPUT_ROOT as DEFAULT_POSE_ESTIMATION_ROOT,
    build_pose_estimation,
    normalized_pose_settings,
    pose_run_root,
    pose_settings_from_review,
    pose_source_signature,
)
from instance_tracking.instance_mapping import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_INSTANCE_TRACKING_ROOT,
    build_instance_tracking,
    instance_mapping_run_root,
    instance_tracking_settings_from_review,
    instance_tracking_source_signature,
    normalized_instance_tracking_settings,
)
try:
    from pipeline import PipelineConfig, VisionPipeline, latest_pipeline_status

    PIPELINE_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - keeps the review UI available if the pipeline import fails
    PipelineConfig = None
    VisionPipeline = None
    latest_pipeline_status = None
    PIPELINE_IMPORT_ERROR = error


API_PREFIX = "/api/vision-review"
PIPELINE_API_PREFIX = "/api/pipeline"
MAX_BODY_BYTES = 200 * 1024 * 1024
CONFIG_PATH = APP_DIR / "assets" / "config" / "alpha_classes.json"
RULES_PATH = APP_DIR / "assets" / "rules" / "alpha_class_rules.json"
REVIEW_PATH = APP_DIR / "assets" / "review" / "review_state.json"
MEMORY_PATH = APP_DIR / "assets" / "memory" / "vision_memory.json"
JOBS_LOCK = threading.Lock()
REBUILD_JOB: dict | None = None
BBOX_BUILD_JOB: dict | None = None
MASKBITS_BBOX_BUILD_JOB: dict | None = None
BBOX_CLIPPING_BUILD_JOB: dict | None = None
BBOX_CONTOUR_BUILD_JOB: dict | None = None
CORNER_BUILD_JOB: dict | None = None
CONTOUR_BUILD_JOB: dict | None = None
SQUARE_POSE_BUILD_JOB: dict | None = None
INSTANCE_BUILD_JOB: dict | None = None
POSE_ESTIMATION_BUILD_JOB: dict | None = None
INSTANCE_TRACKING_BUILD_JOB: dict | None = None
PIPELINE_LOCK = threading.Lock()
PIPELINE_RUNNER = None
PIPELINE_THREAD: threading.Thread | None = None
PLAYBACK_LOCK = threading.Lock()
PLAYBACK_FEEDER = None
PLAYBACK_THREAD: threading.Thread | None = None
PLAYBACK_TARGET_HZ = 30.0
PLAYBACK_PERIOD_S = 1.0 / PLAYBACK_TARGET_HZ
PLAYBACK_ROOT = APP_DIR / "assets" / "pipeline_feeds"
JPEG_SUFFIXES = {".jpg", ".jpeg"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def read_json_or_default(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return read_json(path)


def repo_url(path: Path) -> str:
    return "/" + path.resolve().relative_to(REPO_ROOT).as_posix()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/" + resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def legacy_asset_key(value: str | None) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if "/assets/" in text:
        return "assets/" + text.split("/assets/", 1)[1].lstrip("/")
    if "/0721Vision/" in text:
        return text.split("/0721Vision/", 1)[1].lstrip("/")
    return text.lstrip("/")


def same_legacy_asset(value: str | None, expected: str | None) -> bool:
    left = str(value or "")
    right = str(expected or "")
    if left == right:
        return True
    left_key = legacy_asset_key(left)
    right_key = legacy_asset_key(right)
    return bool(left_key and right_key and left_key == right_key)


def source_signature_for_artifact(
    artifact: dict | None,
    expected_precompute: str,
    current_signature: str | None,
    signature_for_precompute,
) -> str | None:
    if not artifact:
        return current_signature
    actual_signature = str(artifact.get("sourceSignature") or "")
    if actual_signature and actual_signature == str(current_signature or ""):
        return actual_signature
    artifact_precompute = str(artifact.get("precomputeManifest") or "")
    if actual_signature and same_legacy_asset(artifact_precompute, expected_precompute):
        try:
            candidates = signature_for_precompute(artifact_precompute)
            if isinstance(candidates, str):
                candidates = [candidates]
            if actual_signature in candidates:
                return actual_signature
        except Exception:
            pass
    return current_signature


def source_shape_payload(manifest: dict) -> dict:
    return {
        "runKey": manifest.get("runKey"),
        "frameCount": int(manifest.get("frameCount") or len(manifest.get("frames") or [])),
        "width": int(manifest.get("width") or 640),
        "height": int(manifest.get("height") or 360),
    }


def bbox_signature_for_precompute(manifest: dict, precompute_manifest: str, settings: dict, mask_signature: str) -> str:
    return stable_hash(
        {
            "kind": "0720-color-mask-pixel-bboxes",
            **source_shape_payload(manifest),
            "precomputeManifest": precompute_manifest,
            "settings": bbox_signature_settings(settings),
            "maskSignature": mask_signature,
        }
    )


def contour_signature_for_precompute(
    manifest: dict,
    precompute_manifest: str,
    settings: dict,
    mask_signature: str,
    bbox_signature: str | None,
) -> str:
    return stable_hash(
        {
            "kind": "0720-color-mask-contour-hierarchy-v2-raw-points",
            **source_shape_payload(manifest),
            "precomputeManifest": precompute_manifest,
            "settings": normalized_contour_settings(settings),
            "maskSignature": mask_signature,
            "bboxSignature": bbox_signature,
        }
    )


def corner_signature_for_precompute(
    manifest: dict,
    precompute_manifest: str,
    settings: dict,
    mask_signature: str,
    bbox_signature: str | None,
) -> str:
    return stable_hash(
        {
            "kind": "0720-color-mask-bbox-corners",
            **source_shape_payload(manifest),
            "precomputeManifest": precompute_manifest,
            "settings": normalized_corner_settings(settings),
            "maskSignature": mask_signature,
            "bboxSignature": bbox_signature,
        }
    )


def square_pose_signature_for_precompute(
    manifest: dict,
    precompute_manifest: str,
    settings: dict,
    contour_signature: str | None,
    bbox_signature: str | None,
    corner_signature: str | None,
) -> str:
    shape = source_shape_payload(manifest)
    return stable_hash(
        {
            "kind": "0720-square-pose-2p7m-from-contours-and-corners-v2",
            "runKey": shape["runKey"],
            "precomputeManifest": precompute_manifest,
            "frameCount": shape["frameCount"],
            "width": int(manifest.get("width") or SQUARE_POSE_CAMERA["widthPx"]),
            "height": int(manifest.get("height") or SQUARE_POSE_CAMERA["heightPx"]),
            "camera": SQUARE_POSE_CAMERA,
            "settings": normalized_square_pose_settings(settings),
            "contourSourceSignature": contour_signature,
            "bboxSourceSignature": bbox_signature,
            "cornerSourceSignature": corner_signature,
        }
    )


def instance_signature_for_precompute(
    manifest: dict,
    precompute_manifest: str,
    settings: dict,
    bbox_signature: str | None,
    square_pose_signature: str | None,
) -> str:
    return stable_hash(
        {
            "kind": "0720-instance-provenance-association-v1",
            **source_shape_payload(manifest),
            "precomputeManifest": precompute_manifest,
            "settings": instance_signature_settings(settings),
            "bboxSourceSignature": bbox_signature,
            "squarePoseSourceSignature": square_pose_signature,
        }
    )


def instance_signature_candidates_for_precompute(
    manifest: dict,
    precompute_manifest: str,
    settings: dict,
    bbox_signature: str | None,
    square_pose_signature: str | None,
    semantic_square_fit_signature: str | None,
) -> list[str]:
    current = instance_signature_for_precompute(
        manifest,
        precompute_manifest,
        settings,
        bbox_signature,
        square_pose_signature,
    )
    legacy = stable_hash(
        {
            "kind": "0720-instance-provenance-association-v1",
            **source_shape_payload(manifest),
            "precomputeManifest": precompute_manifest,
            "settings": instance_signature_settings(settings),
            "bboxSourceSignature": bbox_signature,
            "squarePoseSourceSignature": square_pose_signature,
            "semanticSquareFitSourceSignature": semantic_square_fit_signature,
        }
    )
    return [current, legacy]


def natural_key(value: str) -> list:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def default_rules() -> dict:
    config = read_json(CONFIG_PATH)
    return {
        "version": 1,
        "app": "vision_passthrough_review",
        "ruleModel": "exact-color-ids",
        "name": "Alpha Group Maximus Pro Ultra Plus 23 exact color rules",
        "updatedAt": None,
        "settings": dict(config.get("defaults") or {}),
        "classRules": {
            str(item["prefix"]): {"colorIdRanges": [], "sampleRgb": []}
            for item in config.get("classes", [])
        },
    }


def default_review() -> dict:
    return {"version": 1, "app": "vision_passthrough_review", "updatedAt": None, "annotations": [], "decisions": []}


def default_memory() -> dict:
    return {
        "version": 1,
        "app": "vision_passthrough_review",
        "analysisMode": "fresh-layer-decode",
        "status": "empty",
        "createdAt": None,
        "source": None,
        "classes": [],
        "classRules": {},
        "frames": [],
        "layerTotals": [],
        "summary": {},
    }


def validate_rules(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("rules payload must be a JSON object")
    config = read_json(CONFIG_PATH)
    data.setdefault("version", 1)
    data.setdefault("app", "vision_passthrough_review")
    data.setdefault("ruleModel", "exact-color-ids")
    data.setdefault("name", "Alpha Group Maximus Pro Ultra Plus 23 exact color rules")
    settings = data.setdefault("settings", dict(config.get("defaults") or {}))
    if not isinstance(settings, dict):
        raise ValueError("rules settings must be an object")
    rules = data.setdefault("classRules", {})
    if not isinstance(rules, dict):
        raise ValueError("classRules must be an object")
    for item in config.get("classes", []):
        prefix = str(item["prefix"])
        rule = rules.setdefault(prefix, {"colorIdRanges": [], "sampleRgb": []})
        if not isinstance(rule, dict):
            raise ValueError(f"classRules.{prefix} must be an object")
        ranges = rule.setdefault("colorIdRanges", [])
        if not isinstance(ranges, list):
            raise ValueError(f"classRules.{prefix}.colorIdRanges must be an array")
        for range_item in ranges:
            if not isinstance(range_item, list) or len(range_item) != 2:
                raise ValueError(f"classRules.{prefix}.colorIdRanges entries must be [start,end]")
            int(range_item[0])
            int(range_item[1])
        rule.setdefault("sampleRgb", [])
    data["updatedAt"] = utc_now()
    return data


def validate_review(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("review payload must be a JSON object")
    data.setdefault("version", 1)
    data.setdefault("app", "vision_passthrough_review")
    if not isinstance(data.setdefault("annotations", []), list):
        raise ValueError("review annotations must be an array")
    if not isinstance(data.setdefault("decisions", []), list):
        raise ValueError("review decisions must be an array")
    data["updatedAt"] = utc_now()
    return data


def dependency_status() -> dict:
    try:
        import numpy as np

    except Exception as error:
        return {"ok": False, "analysisMode": "fresh-layer-decode", "error": str(error)}
    bbox = {"ok": True}
    try:
        import cv2

        bbox["opencv"] = cv2.__version__
    except Exception as error:
        bbox = {"ok": False, "error": str(error)}
    corners = {"ok": True}
    try:
        import cv2

        corners["opencv"] = cv2.__version__
    except Exception as error:
        corners = {"ok": False, "error": str(error)}
    contours = {"ok": True}
    try:
        import cv2

        contours["opencv"] = cv2.__version__
    except Exception as error:
        contours = {"ok": False, "error": str(error)}
    square_pose = {"ok": True}
    try:
        import cv2

        square_pose["opencv"] = cv2.__version__
    except Exception as error:
        square_pose = {"ok": False, "error": str(error)}
    return {
        "ok": True,
        "analysisMode": "fresh-layer-decode",
        "pythonDeps": {"numpy": np.__version__},
        "maskBboxes": bbox,
        "bboxClipping": bbox,
        "bboxContours": contours,
        "maskCorners": corners,
        "maskContours": contours,
        "squarePose": square_pose,
    }


def bbox_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = bbox_settings_from_review(review)
    root = bbox_run_root(run_key, DEFAULT_BBOX_ROOT)
    bbox_manifest_path = root / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    expected_precompute = repo_url(manifest_path)
    expected_signature = None
    mask_context = None
    signature_error = None
    try:
        mask_context = build_mask_context(config, rules, review, manifest)
        expected_signature = bbox_source_signature(manifest, manifest_path, settings, mask_context["signature"])
    except Exception as error:
        signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"mask signature error: {signature_error}")
    accepted_signature = source_signature_for_artifact(
        bbox_manifest,
        expected_precompute,
        expected_signature,
        lambda precompute_ref: bbox_signature_for_precompute(manifest, precompute_ref, settings, mask_context["signature"]),
    ) if mask_context else expected_signature
    if bbox_manifest:
        image = bbox_manifest.get("image") if isinstance(bbox_manifest.get("image"), dict) else {}
        if str(bbox_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(bbox_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if accepted_signature and str(bbox_manifest.get("sourceSignature") or "") != accepted_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(bbox_manifest.get("frameCount") or 0) < frame_count or not bbox_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(bbox_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(bbox_manifest_path) if bbox_manifest_path.exists() else None,
        "expectedSourceSignature": accepted_signature or expected_signature,
        "settings": settings,
        "input": bbox_manifest.get("input") if bbox_manifest else {
            "source": "enabled-color-layers",
            "maskSignature": mask_context["signature"] if mask_context else None,
            "enabledPrefixes": mask_context["enabledPrefixes"] if mask_context else [],
        },
        "manifest": {
            "createdAt": bbox_manifest.get("createdAt") if bbox_manifest else None,
            "kind": bbox_manifest.get("kind") if bbox_manifest else None,
            "frameStart": bbox_manifest.get("frameStart") if bbox_manifest else None,
            "frameCount": bbox_manifest.get("frameCount") if bbox_manifest else None,
            "sourceFrameCount": bbox_manifest.get("sourceFrameCount") if bbox_manifest else None,
            "complete": bbox_manifest.get("complete") if bbox_manifest else None,
            "settings": bbox_manifest.get("settings") if bbox_manifest else settings,
            "summary": bbox_manifest.get("summary") if bbox_manifest else {},
        },
        "dependency": dependency_status().get("maskBboxes", {"ok": False}),
    }


def serialize_bbox_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "bbox": bbox_status(manifest, manifest_path, review, config, rules),
        "job": current_bbox_job(),
    }


def maskbits_bbox_status(manifest: dict, manifest_path: Path, review: dict, config: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = bbox_settings_from_review(review)
    root = maskbits_bbox_run_root(run_key, DEFAULT_MASKBITS_BBOX_ROOT)
    bbox_manifest_path = root / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    mask_manifest_path = DEFAULT_MASK_MANIFEST
    mask_manifest = read_json_optional(mask_manifest_path)
    expected_signature = None
    mask_context = None
    signature_error = None
    if not mask_manifest:
        signature_error = f"mask manifest not found: {repo_url(mask_manifest_path)}"
    else:
        try:
            mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
            expected_signature = maskbits_bbox_source_signature(
                manifest,
                manifest_path,
                mask_manifest,
                mask_manifest_path,
                settings,
                mask_context["signature"],
            )
        except Exception as error:
            signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"maskbits signature error: {signature_error}")
    if bbox_manifest:
        image = bbox_manifest.get("image") if isinstance(bbox_manifest.get("image"), dict) else {}
        if str(bbox_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(bbox_manifest.get("precomputeManifest") or ""), repo_url(manifest_path)):
            stale_reasons.append("precompute changed")
        if not same_legacy_asset(str(bbox_manifest.get("maskManifest") or ""), repo_url(mask_manifest_path)):
            stale_reasons.append("mask manifest changed")
        if expected_signature and str(bbox_manifest.get("sourceSignature") or "") != expected_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(bbox_manifest.get("frameCount") or 0) < frame_count or not bbox_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(bbox_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(bbox_manifest_path) if bbox_manifest_path.exists() else None,
        "expectedSourceSignature": expected_signature,
        "settings": settings,
        "input": bbox_manifest.get("input") if bbox_manifest else {
            "source": "rgb-lut-maskbits",
            "maskManifest": repo_url(mask_manifest_path),
            "maskSignature": mask_context["signature"] if mask_context else None,
            "enabledPrefixes": mask_context["enabledPrefixes"] if mask_context else [],
        },
        "manifest": {
            "createdAt": bbox_manifest.get("createdAt") if bbox_manifest else None,
            "kind": bbox_manifest.get("kind") if bbox_manifest else None,
            "frameStart": bbox_manifest.get("frameStart") if bbox_manifest else None,
            "frameCount": bbox_manifest.get("frameCount") if bbox_manifest else None,
            "sourceFrameCount": bbox_manifest.get("sourceFrameCount") if bbox_manifest else None,
            "complete": bbox_manifest.get("complete") if bbox_manifest else None,
            "settings": bbox_manifest.get("settings") if bbox_manifest else settings,
            "summary": bbox_manifest.get("summary") if bbox_manifest else {},
            "comparison": bbox_manifest.get("comparison") if bbox_manifest else None,
        },
        "dependency": dependency_status().get("maskBboxes", {"ok": False}),
    }


def serialize_maskbits_bbox_status() -> dict:
    config = read_json(CONFIG_PATH)
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "maskbitsBbox": maskbits_bbox_status(manifest, manifest_path, review, config),
        "job": current_maskbits_bbox_job(),
    }


def bbox_clipping_settings_from_review(review: dict) -> dict:
    bbox_settings = bbox_settings_from_review(review)
    return normalized_fov_clip_settings(bbox_settings.get("fovClip") if isinstance(bbox_settings.get("fovClip"), dict) else None)


def bbox_clipping_status(manifest: dict, manifest_path: Path, review: dict, config: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = bbox_clipping_settings_from_review(review)
    root = clipping_run_root(run_key, DEFAULT_BBOX_CLIPPING_ROOT)
    clipping_manifest_path = root / "clipping_manifest.json"
    clipping_manifest = read_json_optional(clipping_manifest_path)
    mask_manifest_path = DEFAULT_MASK_MANIFEST
    mask_manifest = read_json_optional(mask_manifest_path)
    bbox_manifest_path = maskbits_bbox_run_root(run_key, DEFAULT_MASKBITS_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    bbox_info = maskbits_bbox_status(manifest, manifest_path, review, config)
    expected_signature = None
    mask_context = None
    signature_error = None
    if not mask_manifest:
        signature_error = f"mask manifest not found: {repo_url(mask_manifest_path)}"
    elif not bbox_manifest:
        signature_error = "maskbits bbox manifest missing"
    else:
        try:
            mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
            expected_signature = clipping_source_signature(
                manifest,
                manifest_path,
                mask_manifest,
                mask_manifest_path,
                bbox_manifest,
                settings,
                mask_context["signature"],
            )
        except Exception as error:
            signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"clipping signature error: {signature_error}")
    if not bbox_info.get("available"):
        stale_reasons.append("maskbits bbox missing")
    if bbox_info.get("stale"):
        stale_reasons.extend([f"maskbits bbox {item}" for item in bbox_info.get("staleReasons") or []])
    if clipping_manifest:
        image = clipping_manifest.get("image") if isinstance(clipping_manifest.get("image"), dict) else {}
        if str(clipping_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(clipping_manifest.get("precomputeManifest") or ""), repo_url(manifest_path)):
            stale_reasons.append("precompute changed")
        if not same_legacy_asset(str(clipping_manifest.get("maskManifest") or ""), repo_url(mask_manifest_path)):
            stale_reasons.append("mask manifest changed")
        if not same_legacy_asset(str(clipping_manifest.get("bboxManifest") or ""), repo_url(bbox_manifest_path)):
            stale_reasons.append("bbox manifest changed")
        if expected_signature and str(clipping_manifest.get("sourceSignature") or "") != expected_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(clipping_manifest.get("frameCount") or 0) < frame_count or not clipping_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(clipping_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(clipping_manifest_path) if clipping_manifest_path.exists() else None,
        "expectedSourceSignature": expected_signature,
        "settings": settings,
        "input": clipping_manifest.get("input") if clipping_manifest else {
            "source": "rgb-lut-maskbits-bboxes",
            "maskManifest": repo_url(mask_manifest_path),
            "bboxManifest": repo_url(bbox_manifest_path),
            "maskSignature": mask_context["signature"] if mask_context else None,
            "bboxSourceSignature": bbox_manifest.get("sourceSignature") if bbox_manifest else None,
            "enabledPrefixes": mask_context["enabledPrefixes"] if mask_context else [],
        },
        "manifest": {
            "createdAt": clipping_manifest.get("createdAt") if clipping_manifest else None,
            "kind": clipping_manifest.get("kind") if clipping_manifest else None,
            "frameStart": clipping_manifest.get("frameStart") if clipping_manifest else None,
            "frameCount": clipping_manifest.get("frameCount") if clipping_manifest else None,
            "sourceFrameCount": clipping_manifest.get("sourceFrameCount") if clipping_manifest else None,
            "complete": clipping_manifest.get("complete") if clipping_manifest else None,
            "settings": clipping_manifest.get("settings") if clipping_manifest else settings,
            "summary": clipping_manifest.get("summary") if clipping_manifest else {},
        },
        "bbox": bbox_info,
        "dependency": dependency_status().get("bboxClipping", {"ok": False}),
    }


def serialize_bbox_clipping_status() -> dict:
    config = read_json(CONFIG_PATH)
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "bboxClipping": bbox_clipping_status(manifest, manifest_path, review, config),
        "job": current_bbox_clipping_job(),
    }


def bbox_contours_status(manifest: dict, manifest_path: Path, review: dict, config: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = contour_settings_from_review(review)
    root = bbox_contour_run_root(run_key, DEFAULT_BBOX_CONTOUR_ROOT)
    contours_manifest_path = root / "contours_manifest.json"
    contours_manifest = read_json_optional(contours_manifest_path)
    mask_manifest_path = DEFAULT_MASK_MANIFEST
    mask_manifest = read_json_optional(mask_manifest_path)
    bbox_manifest_path = maskbits_bbox_run_root(run_key, DEFAULT_MASKBITS_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    bbox_info = maskbits_bbox_status(manifest, manifest_path, review, config)
    expected_signature = None
    mask_context = None
    signature_error = None
    if not mask_manifest:
        signature_error = f"mask manifest not found: {repo_url(mask_manifest_path)}"
    elif not bbox_manifest:
        signature_error = "maskbits bbox manifest missing"
    else:
        try:
            mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
            expected_signature = bbox_contour_source_signature(
                manifest,
                manifest_path,
                mask_manifest,
                mask_manifest_path,
                bbox_manifest,
                settings,
                mask_context["signature"],
            )
        except Exception as error:
            signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"bbox contour signature error: {signature_error}")
    if not bbox_info.get("available"):
        stale_reasons.append("maskbits bbox missing")
    if bbox_info.get("stale"):
        stale_reasons.extend([f"maskbits bbox {item}" for item in bbox_info.get("staleReasons") or []])
    if contours_manifest:
        image = contours_manifest.get("image") if isinstance(contours_manifest.get("image"), dict) else {}
        if str(contours_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(contours_manifest.get("precomputeManifest") or ""), repo_url(manifest_path)):
            stale_reasons.append("precompute changed")
        if not same_legacy_asset(str(contours_manifest.get("maskManifest") or ""), repo_url(mask_manifest_path)):
            stale_reasons.append("mask manifest changed")
        if not same_legacy_asset(str(contours_manifest.get("bboxManifest") or ""), repo_url(bbox_manifest_path)):
            stale_reasons.append("bbox manifest changed")
        if expected_signature and str(contours_manifest.get("sourceSignature") or "") != expected_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(contours_manifest.get("frameCount") or 0) < frame_count or not contours_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(contours_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(contours_manifest_path) if contours_manifest_path.exists() else None,
        "expectedSourceSignature": expected_signature,
        "settings": settings,
        "input": contours_manifest.get("input") if contours_manifest else {
            "source": settings["maskSource"],
            "maskManifest": repo_url(mask_manifest_path),
            "bboxManifest": repo_url(bbox_manifest_path),
            "maskSignature": mask_context["signature"] if mask_context else None,
            "bboxSourceSignature": bbox_manifest.get("sourceSignature") if bbox_manifest else None,
            "enabledPrefixes": mask_context["enabledPrefixes"] if mask_context else [],
        },
        "manifest": {
            "createdAt": contours_manifest.get("createdAt") if contours_manifest else None,
            "kind": contours_manifest.get("kind") if contours_manifest else None,
            "frameStart": contours_manifest.get("frameStart") if contours_manifest else None,
            "frameCount": contours_manifest.get("frameCount") if contours_manifest else None,
            "sourceFrameCount": contours_manifest.get("sourceFrameCount") if contours_manifest else None,
            "complete": contours_manifest.get("complete") if contours_manifest else None,
            "settings": contours_manifest.get("settings") if contours_manifest else settings,
            "summary": contours_manifest.get("summary") if contours_manifest else {},
        },
        "bbox": bbox_info,
        "dependency": dependency_status().get("bboxContours", {"ok": False}),
    }


def serialize_bbox_contours_status() -> dict:
    config = read_json(CONFIG_PATH)
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "bboxContours": bbox_contours_status(manifest, manifest_path, review, config),
        "job": current_bbox_contour_job(),
    }


def serialize_bbox_contour_frame(payload: dict) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for live bbox contour inference: {error}") from error

    started = time.perf_counter()
    config = read_json(CONFIG_PATH)
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    frames = list(manifest.get("frames") or [])
    frame_count = int(manifest.get("frameCount") or len(frames))
    if not frames:
        raise ValueError("active precompute manifest has no frames")

    frame_ordinal = int(payload.get("frameOrdinal", payload.get("frameIndex", 0)) or 0)
    if frame_ordinal < 0 or frame_ordinal >= len(frames):
        raise ValueError(f"frameOrdinal {frame_ordinal} is outside 0..{len(frames) - 1}")
    frame_entry = frames[frame_ordinal]
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    raw_settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else None
    settings = normalized_contour_settings(raw_settings or contour_settings_from_review(review))

    bbox_info = maskbits_bbox_status(manifest, manifest_path, review, config)
    if not bbox_info.get("available"):
        raise ValueError("maskbits bbox input unavailable")
    if bbox_info.get("stale"):
        reasons = ", ".join(str(item) for item in bbox_info.get("staleReasons") or []) or "unknown stale reason"
        raise ValueError(f"maskbits bbox stale: {reasons}")

    mask_manifest_path = DEFAULT_MASK_MANIFEST
    mask_manifest = read_json_optional(mask_manifest_path)
    if not mask_manifest:
        raise ValueError(f"maskbits manifest not found: {repo_url(mask_manifest_path)}")
    if int(mask_manifest.get("width") or width) != width or int(mask_manifest.get("height") or height) != height:
        raise ValueError("maskbits manifest dimensions do not match the active precompute manifest")

    run_key = str(manifest.get("runKey") or "")
    bbox_manifest_path = maskbits_bbox_run_root(run_key, DEFAULT_MASKBITS_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    if not bbox_manifest:
        raise ValueError("maskbits bbox manifest missing")
    bbox_frames = bbox_frame_map(bbox_manifest)
    bbox_frame = bbox_frames.get(frame_ordinal)
    if not bbox_frame:
        raise ValueError(f"maskbits bbox manifest has no frame entry for frame {frame_ordinal}")

    mask_context = build_maskbits_context(config, review, mask_manifest, mask_manifest_path)
    mask_lookup = build_mask_frame_lookup(mask_manifest)
    frame_out = build_bbox_contour_frame(
        cv2,
        frame_entry,
        frame_ordinal,
        bbox_frame,
        mask_lookup,
        mask_context,
        width,
        height,
        settings,
    )
    counts = {
        "objectCount": int(frame_out.get("objectCount") or 0),
        "contourCount": int(frame_out.get("contourCount") or 0),
        "outerCount": int(frame_out.get("outerCount") or 0),
        "voidCount": int(frame_out.get("voidCount") or 0),
        "nestedIslandCount": int(frame_out.get("nestedIslandCount") or 0),
        "notchCandidateCount": int(frame_out.get("notchCandidateCount") or 0),
        "overlapCandidateCount": int(frame_out.get("overlapCandidateCount") or 0),
        "selectedPixelCount": int(frame_out.get("selectedPixelCount") or 0),
    }
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": frame_count,
        "frameOrdinal": frame_ordinal,
        "image": {"width": width, "height": height},
        "settings": settings,
        "timingMs": round((time.perf_counter() - started) * 1000, 3),
        "counts": counts,
        "frame": frame_out,
        "input": {
            "source": settings["maskSource"],
            "maskManifest": repo_url(mask_manifest_path),
            "bboxManifest": repo_url(bbox_manifest_path),
            "maskSignature": mask_context["signature"],
            "bboxSourceSignature": bbox_manifest.get("sourceSignature"),
            "enabledPrefixes": mask_context["enabledPrefixes"],
        },
    }


def corner_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = corner_settings_from_review(review)
    root = corner_run_root(run_key, DEFAULT_CORNER_ROOT)
    corners_manifest_path = root / "corners_manifest.json"
    corners_manifest = read_json_optional(corners_manifest_path)
    bbox_manifest_path = bbox_run_root(run_key, DEFAULT_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    expected_precompute = repo_url(manifest_path)
    expected_signature = None
    expected_bbox_signature = None
    mask_context = None
    signature_error = None
    bbox_stale_reasons: list[str] = []
    try:
        mask_context = build_mask_context(config, rules, review, manifest)
        expected_bbox_signature = bbox_source_signature(
            manifest,
            manifest_path,
            bbox_settings_from_review(review),
            mask_context["signature"],
        )
        expected_signature = corner_source_signature(
            manifest,
            manifest_path,
            settings,
            mask_context["signature"],
            expected_bbox_signature,
        )
    except Exception as error:
        signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"mask signature error: {signature_error}")
    accepted_bbox_signature = source_signature_for_artifact(
        bbox_manifest,
        expected_precompute,
        expected_bbox_signature,
        lambda precompute_ref: bbox_signature_for_precompute(
            manifest,
            precompute_ref,
            bbox_settings_from_review(review),
            mask_context["signature"],
        ),
    ) if mask_context else expected_bbox_signature
    artifact_bbox_signature = (
        corners_manifest.get("input", {}).get("bboxSourceSignature")
        if corners_manifest and isinstance(corners_manifest.get("input"), dict)
        else None
    )
    accepted_signature = source_signature_for_artifact(
        corners_manifest,
        expected_precompute,
        expected_signature,
        lambda precompute_ref: [
            corner_signature_for_precompute(
                manifest,
                precompute_ref,
                settings,
                mask_context["signature"],
                accepted_bbox_signature,
            ),
            corner_signature_for_precompute(
                manifest,
                precompute_ref,
                settings,
                mask_context["signature"],
                artifact_bbox_signature,
            ),
        ],
    ) if mask_context else expected_signature
    corner_signature_ok = bool(
        corners_manifest
        and accepted_signature
        and str(corners_manifest.get("sourceSignature") or "") == str(accepted_signature)
    )
    if not bbox_manifest:
        bbox_stale_reasons.append("bbox missing")
    else:
        bbox_image = bbox_manifest.get("image") if isinstance(bbox_manifest.get("image"), dict) else {}
        if str(bbox_manifest.get("runKey") or "") != run_key:
            bbox_stale_reasons.append("bbox run changed")
        if not same_legacy_asset(str(bbox_manifest.get("precomputeManifest") or ""), expected_precompute):
            bbox_stale_reasons.append("bbox precompute changed")
        if accepted_bbox_signature and str(bbox_manifest.get("sourceSignature") or "") != accepted_bbox_signature:
            if not corner_signature_ok:
                bbox_stale_reasons.append("bbox source signature changed")
        if int(bbox_image.get("width") or 0) != width or int(bbox_image.get("height") or 0) != height:
            bbox_stale_reasons.append("bbox image size changed")
        if int(bbox_manifest.get("frameCount") or 0) < frame_count or not bbox_manifest.get("complete"):
            bbox_stale_reasons.append("bbox partial build")
    stale_reasons.extend(bbox_stale_reasons)
    if corners_manifest:
        image = corners_manifest.get("image") if isinstance(corners_manifest.get("image"), dict) else {}
        if str(corners_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(corners_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if accepted_signature and str(corners_manifest.get("sourceSignature") or "") != accepted_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(corners_manifest.get("frameCount") or 0) < frame_count or not corners_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(corners_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(corners_manifest_path) if corners_manifest_path.exists() else None,
        "expectedSourceSignature": accepted_signature or expected_signature,
        "settings": settings,
        "input": corners_manifest.get("input") if corners_manifest else {
            "source": "enabled-color-layers",
            "maskSignature": mask_context["signature"] if mask_context else None,
            "enabledPrefixes": mask_context["enabledPrefixes"] if mask_context else [],
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": accepted_bbox_signature,
        },
        "manifest": {
            "createdAt": corners_manifest.get("createdAt") if corners_manifest else None,
            "kind": corners_manifest.get("kind") if corners_manifest else None,
            "frameStart": corners_manifest.get("frameStart") if corners_manifest else None,
            "frameCount": corners_manifest.get("frameCount") if corners_manifest else None,
            "sourceFrameCount": corners_manifest.get("sourceFrameCount") if corners_manifest else None,
            "complete": corners_manifest.get("complete") if corners_manifest else None,
            "settings": corners_manifest.get("settings") if corners_manifest else settings,
            "summary": corners_manifest.get("summary") if corners_manifest else {},
        },
        "bbox": {
            "available": bool(bbox_manifest),
            "stale": bool(bbox_stale_reasons),
            "staleReasons": bbox_stale_reasons,
            "manifestUrl": repo_url(bbox_manifest_path) if bbox_manifest_path.exists() else None,
            "expectedSourceSignature": accepted_bbox_signature or expected_bbox_signature,
        },
        "dependency": dependency_status().get("maskCorners", {"ok": False}),
    }


def serialize_corner_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "corners": corner_status(manifest, manifest_path, review, config, rules),
        "job": current_corner_job(),
    }


def contour_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = contour_settings_from_review(review)
    root = contour_run_root(run_key, DEFAULT_CONTOUR_ROOT)
    contours_manifest_path = root / "contours_manifest.json"
    contours_manifest = read_json_optional(contours_manifest_path)
    bbox_manifest_path = bbox_run_root(run_key, DEFAULT_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    expected_precompute = repo_url(manifest_path)
    expected_signature = None
    expected_bbox_signature = None
    mask_context = None
    signature_error = None
    bbox_stale_reasons: list[str] = []
    try:
        mask_context = build_mask_context(config, rules, review, manifest)
        expected_bbox_signature = bbox_source_signature(
            manifest,
            manifest_path,
            bbox_settings_from_review(review),
            mask_context["signature"],
        )
        expected_signature = contour_source_signature(
            manifest,
            manifest_path,
            settings,
            mask_context["signature"],
            expected_bbox_signature,
        )
    except Exception as error:
        signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"mask signature error: {signature_error}")
    accepted_bbox_signature = source_signature_for_artifact(
        bbox_manifest,
        expected_precompute,
        expected_bbox_signature,
        lambda precompute_ref: bbox_signature_for_precompute(
            manifest,
            precompute_ref,
            bbox_settings_from_review(review),
            mask_context["signature"],
        ),
    ) if mask_context else expected_bbox_signature
    accepted_signature = source_signature_for_artifact(
        contours_manifest,
        expected_precompute,
        expected_signature,
        lambda precompute_ref: contour_signature_for_precompute(
            manifest,
            precompute_ref,
            settings,
            mask_context["signature"],
            accepted_bbox_signature,
        ),
    ) if mask_context else expected_signature
    if not bbox_manifest:
        bbox_stale_reasons.append("bbox missing")
    else:
        bbox_image = bbox_manifest.get("image") if isinstance(bbox_manifest.get("image"), dict) else {}
        if str(bbox_manifest.get("runKey") or "") != run_key:
            bbox_stale_reasons.append("bbox run changed")
        if not same_legacy_asset(str(bbox_manifest.get("precomputeManifest") or ""), expected_precompute):
            bbox_stale_reasons.append("bbox precompute changed")
        if accepted_bbox_signature and str(bbox_manifest.get("sourceSignature") or "") != accepted_bbox_signature:
            bbox_stale_reasons.append("bbox source signature changed")
        if int(bbox_image.get("width") or 0) != width or int(bbox_image.get("height") or 0) != height:
            bbox_stale_reasons.append("bbox image size changed")
        if int(bbox_manifest.get("frameCount") or 0) < frame_count or not bbox_manifest.get("complete"):
            bbox_stale_reasons.append("bbox partial build")
    stale_reasons.extend(bbox_stale_reasons)
    if contours_manifest:
        image = contours_manifest.get("image") if isinstance(contours_manifest.get("image"), dict) else {}
        if str(contours_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(contours_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if accepted_signature and str(contours_manifest.get("sourceSignature") or "") != accepted_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(contours_manifest.get("frameCount") or 0) < frame_count or not contours_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(contours_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(contours_manifest_path) if contours_manifest_path.exists() else None,
        "expectedSourceSignature": accepted_signature or expected_signature,
        "settings": settings,
        "input": contours_manifest.get("input") if contours_manifest else {
            "source": settings["maskSource"],
            "maskSignature": mask_context["signature"] if mask_context else None,
            "enabledPrefixes": mask_context["enabledPrefixes"] if mask_context else [],
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": accepted_bbox_signature,
        },
        "manifest": {
            "createdAt": contours_manifest.get("createdAt") if contours_manifest else None,
            "kind": contours_manifest.get("kind") if contours_manifest else None,
            "frameStart": contours_manifest.get("frameStart") if contours_manifest else None,
            "frameCount": contours_manifest.get("frameCount") if contours_manifest else None,
            "sourceFrameCount": contours_manifest.get("sourceFrameCount") if contours_manifest else None,
            "complete": contours_manifest.get("complete") if contours_manifest else None,
            "settings": contours_manifest.get("settings") if contours_manifest else settings,
            "summary": contours_manifest.get("summary") if contours_manifest else {},
        },
        "bbox": {
            "available": bool(bbox_manifest),
            "stale": bool(bbox_stale_reasons),
            "staleReasons": bbox_stale_reasons,
            "manifestUrl": repo_url(bbox_manifest_path) if bbox_manifest_path.exists() else None,
            "expectedSourceSignature": accepted_bbox_signature or expected_bbox_signature,
        },
        "dependency": dependency_status().get("maskContours", {"ok": False}),
    }


def serialize_contour_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "contours": contour_status(manifest, manifest_path, review, config, rules),
        "job": current_contour_job(),
    }


def square_pose_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or SQUARE_POSE_CAMERA["widthPx"])
    height = int(manifest.get("height") or SQUARE_POSE_CAMERA["heightPx"])
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = square_pose_settings_from_review(review)
    root = square_pose_run_root(run_key, DEFAULT_SQUARE_POSE_ROOT)
    square_pose_manifest_path = root / "square_pose_manifest.json"
    square_pose_manifest = read_json_optional(square_pose_manifest_path)
    bbox_status_payload = bbox_status(manifest, manifest_path, review, config, rules)
    contour_status_payload = contour_status(manifest, manifest_path, review, config, rules)
    corner_status_payload = corner_status(manifest, manifest_path, review, config, rules)
    bbox_manifest_path = bbox_run_root(run_key, DEFAULT_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    contour_manifest_path = contour_run_root(run_key, DEFAULT_CONTOUR_ROOT) / "contours_manifest.json"
    contour_manifest = read_json_optional(contour_manifest_path)
    corner_manifest_path = corner_run_root(run_key, DEFAULT_CORNER_ROOT) / "corners_manifest.json"
    corner_manifest = read_json_optional(corner_manifest_path)
    expected_precompute = repo_url(manifest_path)
    expected_signature = None
    signature_error = None
    try:
        expected_signature = square_pose_source_signature(
            manifest,
            manifest_path,
            settings,
            str(contour_manifest.get("sourceSignature") or "") if contour_manifest else None,
            str(bbox_manifest.get("sourceSignature") or "") if bbox_manifest and settings.get("clipPoseGuardEnabled") else None,
            str(corner_manifest.get("sourceSignature") or "") if corner_manifest and settings.get("cornerAwareFitEnabled") else None,
        )
    except Exception as error:
        signature_error = str(error)

    accepted_contour_signature = str(contour_manifest.get("sourceSignature") or "") if contour_manifest else None
    accepted_bbox_signature = str(bbox_manifest.get("sourceSignature") or "") if bbox_manifest and settings.get("clipPoseGuardEnabled") else None
    accepted_corner_signature = str(corner_manifest.get("sourceSignature") or "") if corner_manifest and settings.get("cornerAwareFitEnabled") else None
    accepted_signature = source_signature_for_artifact(
        square_pose_manifest,
        expected_precompute,
        expected_signature,
        lambda precompute_ref: square_pose_signature_for_precompute(
            manifest,
            precompute_ref,
            settings,
            accepted_contour_signature,
            accepted_bbox_signature,
            accepted_corner_signature,
        ),
    )

    contour_stale_reasons: list[str] = []
    if not contour_status_payload.get("available"):
        contour_stale_reasons.append("contours missing")
    if contour_status_payload.get("stale"):
        contour_stale_reasons.extend([f"contours {item}" for item in contour_status_payload.get("staleReasons") or []])
    bbox_stale_reasons: list[str] = []
    if settings.get("clipPoseGuardEnabled"):
        if not bbox_status_payload.get("available"):
            bbox_stale_reasons.append("bboxes missing")
        if bbox_status_payload.get("stale"):
            bbox_stale_reasons.extend([f"bboxes {item}" for item in bbox_status_payload.get("staleReasons") or []])
    corner_stale_reasons: list[str] = []
    if settings.get("cornerAwareFitEnabled"):
        if not corner_status_payload.get("available"):
            corner_stale_reasons.append("corners missing")
        if corner_status_payload.get("stale"):
            corner_stale_reasons.extend([f"corners {item}" for item in corner_status_payload.get("staleReasons") or []])

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"source signature error: {signature_error}")
    stale_reasons.extend(contour_stale_reasons)
    stale_reasons.extend(bbox_stale_reasons)
    stale_reasons.extend(corner_stale_reasons)
    if square_pose_manifest:
        image = square_pose_manifest.get("image") if isinstance(square_pose_manifest.get("image"), dict) else {}
        if str(square_pose_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(square_pose_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if accepted_signature and str(square_pose_manifest.get("sourceSignature") or "") != accepted_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(square_pose_manifest.get("frameCount") or 0) < frame_count or not square_pose_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(square_pose_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(square_pose_manifest_path) if square_pose_manifest_path.exists() else None,
        "expectedSourceSignature": accepted_signature or expected_signature,
        "settings": settings,
        "input": square_pose_manifest.get("input") if square_pose_manifest else {
            "contourManifest": repo_url(contour_manifest_path),
            "contourSourceSignature": contour_manifest.get("sourceSignature") if contour_manifest else None,
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": bbox_manifest.get("sourceSignature") if bbox_manifest else None,
            "cornerManifest": repo_url(corner_manifest_path),
            "cornerSourceSignature": corner_manifest.get("sourceSignature") if corner_manifest else None,
        },
        "manifest": {
            "createdAt": square_pose_manifest.get("createdAt") if square_pose_manifest else None,
            "kind": square_pose_manifest.get("kind") if square_pose_manifest else None,
            "frameStart": square_pose_manifest.get("frameStart") if square_pose_manifest else None,
            "frameCount": square_pose_manifest.get("frameCount") if square_pose_manifest else None,
            "sourceFrameCount": square_pose_manifest.get("sourceFrameCount") if square_pose_manifest else None,
            "complete": square_pose_manifest.get("complete") if square_pose_manifest else None,
            "camera": square_pose_manifest.get("camera") if square_pose_manifest else SQUARE_POSE_CAMERA,
            "square": square_pose_manifest.get("square") if square_pose_manifest else {
                "widthM": settings["squareSizeM"],
                "heightM": settings["squareSizeM"],
            },
            "settings": square_pose_manifest.get("settings") if square_pose_manifest else settings,
            "summary": square_pose_manifest.get("summary") if square_pose_manifest else {},
        },
        "bboxes": {
            "available": bool(bbox_status_payload.get("available")),
            "stale": bool(bbox_status_payload.get("stale")),
            "staleReasons": bbox_status_payload.get("staleReasons") or [],
            "manifestUrl": bbox_status_payload.get("manifestUrl"),
            "expectedSourceSignature": bbox_status_payload.get("expectedSourceSignature"),
        },
        "contours": {
            "available": bool(contour_status_payload.get("available")),
            "stale": bool(contour_status_payload.get("stale")),
            "staleReasons": contour_status_payload.get("staleReasons") or [],
            "manifestUrl": contour_status_payload.get("manifestUrl"),
            "expectedSourceSignature": contour_status_payload.get("expectedSourceSignature"),
        },
        "corners": {
            "available": bool(corner_status_payload.get("available")),
            "stale": bool(corner_status_payload.get("stale")),
            "staleReasons": corner_status_payload.get("staleReasons") or [],
            "manifestUrl": corner_status_payload.get("manifestUrl"),
            "expectedSourceSignature": corner_status_payload.get("expectedSourceSignature"),
        },
        "dependency": dependency_status().get("squarePose", {"ok": False}),
    }


def serialize_square_pose_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "squarePose": square_pose_status(manifest, manifest_path, review, config, rules),
        "job": current_square_pose_job(),
    }


def pose_estimation_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or POSE_ESTIMATION_CAMERA["widthPx"])
    height = int(manifest.get("height") or POSE_ESTIMATION_CAMERA["heightPx"])
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = pose_settings_from_review(review)
    root = pose_run_root(run_key, DEFAULT_POSE_ESTIMATION_ROOT)
    pose_manifest_path = root / "3d_pose_fit.json"
    pose_manifest = read_json_optional(pose_manifest_path)
    contour_status_payload = bbox_contours_status(manifest, manifest_path, review, config)
    clipping_status_payload = bbox_clipping_status(manifest, manifest_path, review, config)
    contour_manifest_path = bbox_contour_run_root(run_key, DEFAULT_BBOX_CONTOUR_ROOT) / "contours_manifest.json"
    contour_manifest = read_json_optional(contour_manifest_path)
    clipping_manifest_path = clipping_run_root(run_key, DEFAULT_BBOX_CLIPPING_ROOT) / "clipping_manifest.json"
    clipping_manifest = read_json_optional(clipping_manifest_path)
    expected_precompute = repo_url(manifest_path)
    expected_signature = None
    signature_error = None
    try:
        expected_signature = pose_source_signature(
            manifest,
            manifest_path,
            settings,
            str(contour_manifest.get("sourceSignature") or "") if contour_manifest else None,
            str(clipping_manifest.get("sourceSignature") or "") if clipping_manifest else None,
        )
    except Exception as error:
        signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"source signature error: {signature_error}")
    if not contour_status_payload.get("available"):
        stale_reasons.append("bbox contours missing")
    if not clipping_status_payload.get("available"):
        stale_reasons.append("bbox clipping missing")
    if pose_manifest:
        image = pose_manifest.get("image") if isinstance(pose_manifest.get("image"), dict) else {}
        if str(pose_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(pose_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if expected_signature and str(pose_manifest.get("sourceSignature") or "") != expected_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(pose_manifest.get("frameCount") or 0) < frame_count or not pose_manifest.get("complete"):
            stale_reasons.append("partial build")

    contour_dependency = contour_status_payload.get("dependency") or {}
    dependency = dependency_status().get("squarePose", {"ok": True})
    return {
        "available": bool(pose_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(pose_manifest_path) if pose_manifest_path.exists() else None,
        "expectedSourceSignature": expected_signature,
        "settings": settings,
        "input": pose_manifest.get("input") if pose_manifest else {
            "contourManifest": repo_url(contour_manifest_path),
            "contourSourceSignature": contour_manifest.get("sourceSignature") if contour_manifest else None,
            "clippingManifest": repo_url(clipping_manifest_path),
            "clippingSourceSignature": clipping_manifest.get("sourceSignature") if clipping_manifest else None,
            "cornerManifest": None,
            "cornerSourceSignature": None,
        },
        "manifest": {
            "createdAt": pose_manifest.get("createdAt") if pose_manifest else None,
            "kind": pose_manifest.get("kind") if pose_manifest else None,
            "frameStart": pose_manifest.get("frameStart") if pose_manifest else None,
            "frameCount": pose_manifest.get("frameCount") if pose_manifest else None,
            "sourceFrameCount": pose_manifest.get("sourceFrameCount") if pose_manifest else None,
            "complete": pose_manifest.get("complete") if pose_manifest else None,
            "camera": pose_manifest.get("camera") if pose_manifest else POSE_ESTIMATION_CAMERA,
            "square": pose_manifest.get("square") if pose_manifest else {
                "widthM": settings["squareSizeM"],
                "heightM": settings["squareSizeM"],
            },
            "settings": pose_manifest.get("settings") if pose_manifest else settings,
            "summary": pose_manifest.get("summary") if pose_manifest else {},
        },
        "contours": {
            "available": bool(contour_status_payload.get("available")),
            "stale": False,
            "staleReasons": contour_status_payload.get("staleReasons") or [],
            "manifestUrl": contour_status_payload.get("manifestUrl"),
            "expectedSourceSignature": contour_status_payload.get("expectedSourceSignature"),
        },
        "bboxContours": {
            "available": bool(contour_status_payload.get("available")),
            "stale": bool(contour_status_payload.get("stale")),
            "staleReasons": contour_status_payload.get("staleReasons") or [],
            "manifestUrl": contour_status_payload.get("manifestUrl"),
            "expectedSourceSignature": contour_status_payload.get("expectedSourceSignature"),
        },
        "bboxes": {
            "available": bool(clipping_status_payload.get("available")),
            "stale": False,
            "staleReasons": clipping_status_payload.get("staleReasons") or [],
            "manifestUrl": clipping_status_payload.get("manifestUrl"),
            "expectedSourceSignature": clipping_status_payload.get("expectedSourceSignature"),
        },
        "bboxClipping": {
            "available": bool(clipping_status_payload.get("available")),
            "stale": bool(clipping_status_payload.get("stale")),
            "staleReasons": clipping_status_payload.get("staleReasons") or [],
            "manifestUrl": clipping_status_payload.get("manifestUrl"),
            "expectedSourceSignature": clipping_status_payload.get("expectedSourceSignature"),
        },
        "corners": {"available": True, "stale": False, "staleReasons": [], "manifestUrl": None, "expectedSourceSignature": None},
        "dependency": dependency if dependency.get("ok") is False else contour_dependency or {"ok": True},
    }


def serialize_pose_estimation_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    status = pose_estimation_status(manifest, manifest_path, review, config, rules)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "poseEstimation": status,
        "squarePose": status,
        "job": current_pose_estimation_job(),
    }


def instance_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = instance_settings_from_review(review)
    pose_source = str(settings.get("poseSource") or "squarePose")
    root = instance_run_root(run_key, DEFAULT_INSTANCE_ROOT)
    instance_manifest_path = root / "instance_manifest.json"
    instance_manifest = read_json_optional(instance_manifest_path)
    bbox_status_payload = bbox_status(manifest, manifest_path, review, config, rules)
    square_status_payload = square_pose_status(manifest, manifest_path, review, config, rules)
    bbox_manifest_path = bbox_run_root(run_key, DEFAULT_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    square_pose_manifest_path = square_pose_run_root(run_key, DEFAULT_SQUARE_POSE_ROOT) / "square_pose_manifest.json"
    square_pose_manifest = read_json_optional(square_pose_manifest_path)
    expected_precompute = repo_url(manifest_path)
    bbox_signature = str(bbox_manifest.get("sourceSignature") or bbox_status_payload.get("expectedSourceSignature") or "") if bbox_manifest else None
    square_pose_signature = None
    if square_pose_manifest and square_status_payload.get("available") and not square_status_payload.get("stale"):
        square_pose_signature = str(square_pose_manifest.get("sourceSignature") or square_status_payload.get("expectedSourceSignature") or "")
    expected_signature = None
    signature_error = None
    try:
        expected_signature = instance_source_signature(
            manifest,
            manifest_path,
            settings,
            bbox_signature,
            square_pose_signature if pose_source == "squarePose" else None,
        )
    except Exception as error:
        signature_error = str(error)
    accepted_signature = source_signature_for_artifact(
        instance_manifest,
        expected_precompute,
        expected_signature,
        lambda precompute_ref: instance_signature_candidates_for_precompute(
            manifest,
            precompute_ref,
            settings,
            bbox_signature,
            square_pose_signature if pose_source == "squarePose" else None,
            (
                instance_manifest.get("input", {}).get("semanticSquareFitSourceSignature")
                if isinstance(instance_manifest.get("input"), dict)
                else None
            ),
        ),
    )

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"source signature error: {signature_error}")
    if not bbox_status_payload.get("available"):
        stale_reasons.append("bboxes missing")
    if bbox_status_payload.get("stale"):
        stale_reasons.extend([f"bboxes {item}" for item in bbox_status_payload.get("staleReasons") or []])
    if pose_source == "squarePose" and square_status_payload.get("available") and square_status_payload.get("stale"):
        stale_reasons.extend([f"square pose {item}" for item in square_status_payload.get("staleReasons") or []])
    if instance_manifest:
        image = instance_manifest.get("image") if isinstance(instance_manifest.get("image"), dict) else {}
        if str(instance_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(instance_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if accepted_signature and str(instance_manifest.get("sourceSignature") or "") != accepted_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(instance_manifest.get("frameCount") or 0) < frame_count or not instance_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(instance_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(instance_manifest_path) if instance_manifest_path.exists() else None,
        "expectedSourceSignature": accepted_signature or expected_signature,
        "settings": settings,
        "input": instance_manifest.get("input") if instance_manifest else {
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": bbox_signature,
            "squarePoseManifest": repo_url(square_pose_manifest_path) if square_pose_manifest_path.exists() else None,
            "squarePoseSourceSignature": square_pose_signature,
            "activePoseSource": pose_source,
            "activePoseSourceSignature": square_pose_signature if pose_source == "squarePose" else None,
            "poseSignalsAvailable": bool(square_pose_signature if pose_source == "squarePose" else None),
        },
        "manifest": {
            "createdAt": instance_manifest.get("createdAt") if instance_manifest else None,
            "kind": instance_manifest.get("kind") if instance_manifest else None,
            "frameStart": instance_manifest.get("frameStart") if instance_manifest else None,
            "frameCount": instance_manifest.get("frameCount") if instance_manifest else None,
            "sourceFrameCount": instance_manifest.get("sourceFrameCount") if instance_manifest else None,
            "complete": instance_manifest.get("complete") if instance_manifest else None,
            "settings": instance_manifest.get("settings") if instance_manifest else settings,
            "summary": instance_manifest.get("summary") if instance_manifest else {},
        },
        "bboxes": {
            "available": bool(bbox_status_payload.get("available")),
            "stale": bool(bbox_status_payload.get("stale")),
            "staleReasons": bbox_status_payload.get("staleReasons") or [],
            "manifestUrl": bbox_status_payload.get("manifestUrl"),
            "expectedSourceSignature": bbox_status_payload.get("expectedSourceSignature"),
        },
        "squarePose": {
            "available": bool(square_status_payload.get("available")),
            "stale": bool(square_status_payload.get("stale")),
            "staleReasons": square_status_payload.get("staleReasons") or [],
            "manifestUrl": square_status_payload.get("manifestUrl"),
            "expectedSourceSignature": square_status_payload.get("expectedSourceSignature"),
        },
        "dependency": {"ok": True},
    }


def serialize_instance_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "instances": instance_status(manifest, manifest_path, review, config, rules),
        "job": current_instance_job(),
    }


def instance_tracking_status(manifest: dict, manifest_path: Path, review: dict, config: dict, rules: dict) -> dict:
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    settings = instance_tracking_settings_from_review(review)
    root = instance_mapping_run_root(run_key, DEFAULT_INSTANCE_TRACKING_ROOT)
    instance_manifest_path = root / "instance_mapping.json"
    instance_manifest = read_json_optional(instance_manifest_path)
    bbox_status_payload = maskbits_bbox_status(manifest, manifest_path, review, config)
    pose_status_payload = pose_estimation_status(manifest, manifest_path, review, config, rules)
    bbox_manifest_path = maskbits_bbox_run_root(run_key, DEFAULT_MASKBITS_BBOX_ROOT) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    pose_manifest_path = pose_run_root(run_key, DEFAULT_POSE_ESTIMATION_ROOT) / "3d_pose_fit.json"
    pose_manifest = read_json_optional(pose_manifest_path)
    expected_precompute = repo_url(manifest_path)
    bbox_signature = str(bbox_manifest.get("sourceSignature") or bbox_status_payload.get("expectedSourceSignature") or "") if bbox_manifest else None
    pose_signature = None
    if pose_manifest and pose_status_payload.get("available") and not pose_status_payload.get("stale"):
        pose_signature = str(pose_manifest.get("sourceSignature") or pose_status_payload.get("expectedSourceSignature") or "")
    expected_signature = None
    signature_error = None
    try:
        expected_signature = instance_tracking_source_signature(
            manifest,
            manifest_path,
            settings,
            bbox_signature,
            pose_signature if settings.get("poseSource") == "poseFit" else None,
        )
    except Exception as error:
        signature_error = str(error)

    stale_reasons: list[str] = []
    if signature_error:
        stale_reasons.append(f"source signature error: {signature_error}")
    if not bbox_status_payload.get("available"):
        stale_reasons.append("maskbits bboxes missing")
    if bbox_status_payload.get("stale"):
        stale_reasons.extend([f"maskbits bboxes {item}" for item in bbox_status_payload.get("staleReasons") or []])
    if settings.get("poseSource") == "poseFit":
        if not pose_status_payload.get("available"):
            stale_reasons.append("pose estimation missing")
        if pose_status_payload.get("stale"):
            stale_reasons.extend([f"pose estimation {item}" for item in pose_status_payload.get("staleReasons") or []])
    if instance_manifest:
        image = instance_manifest.get("image") if isinstance(instance_manifest.get("image"), dict) else {}
        if str(instance_manifest.get("runKey") or "") != run_key:
            stale_reasons.append("run changed")
        if not same_legacy_asset(str(instance_manifest.get("precomputeManifest") or ""), expected_precompute):
            stale_reasons.append("precompute changed")
        if expected_signature and str(instance_manifest.get("sourceSignature") or "") != expected_signature:
            stale_reasons.append("source signature changed")
        if int(image.get("width") or 0) != width or int(image.get("height") or 0) != height:
            stale_reasons.append("image size changed")
        if int(instance_manifest.get("frameCount") or 0) < frame_count or not instance_manifest.get("complete"):
            stale_reasons.append("partial build")

    return {
        "available": bool(instance_manifest),
        "stale": bool(stale_reasons),
        "staleReasons": stale_reasons,
        "runKey": run_key,
        "root": repo_url(root),
        "manifestUrl": repo_url(instance_manifest_path) if instance_manifest_path.exists() else None,
        "expectedSourceSignature": expected_signature,
        "settings": settings,
        "input": instance_manifest.get("input") if instance_manifest else {
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": bbox_signature,
            "poseManifest": repo_url(pose_manifest_path) if pose_manifest_path.exists() else None,
            "poseSourceSignature": pose_signature,
            "activePoseSource": settings.get("poseSource"),
            "activePoseSourceSignature": pose_signature if settings.get("poseSource") == "poseFit" else None,
            "poseSignalsAvailable": bool(pose_signature if settings.get("poseSource") == "poseFit" else None),
        },
        "manifest": {
            "createdAt": instance_manifest.get("createdAt") if instance_manifest else None,
            "kind": instance_manifest.get("kind") if instance_manifest else None,
            "frameStart": instance_manifest.get("frameStart") if instance_manifest else None,
            "frameCount": instance_manifest.get("frameCount") if instance_manifest else None,
            "sourceFrameCount": instance_manifest.get("sourceFrameCount") if instance_manifest else None,
            "complete": instance_manifest.get("complete") if instance_manifest else None,
            "settings": instance_manifest.get("settings") if instance_manifest else settings,
            "summary": instance_manifest.get("summary") if instance_manifest else {},
        },
        "bboxes": {
            "available": bool(bbox_status_payload.get("available")),
            "stale": bool(bbox_status_payload.get("stale")),
            "staleReasons": bbox_status_payload.get("staleReasons") or [],
            "manifestUrl": bbox_status_payload.get("manifestUrl"),
            "expectedSourceSignature": bbox_status_payload.get("expectedSourceSignature"),
        },
        "poseEstimation": {
            "available": bool(pose_status_payload.get("available")),
            "stale": bool(pose_status_payload.get("stale")),
            "staleReasons": pose_status_payload.get("staleReasons") or [],
            "manifestUrl": pose_status_payload.get("manifestUrl"),
            "expectedSourceSignature": pose_status_payload.get("expectedSourceSignature"),
        },
        "squarePose": {
            "available": bool(pose_status_payload.get("available")),
            "stale": bool(pose_status_payload.get("stale")),
            "staleReasons": pose_status_payload.get("staleReasons") or [],
            "manifestUrl": pose_status_payload.get("manifestUrl"),
            "expectedSourceSignature": pose_status_payload.get("expectedSourceSignature"),
        },
        "dependency": {"ok": True},
    }


def serialize_instance_tracking_status() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    review = read_json_or_default(REVIEW_PATH, default_review())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    status = instance_tracking_status(manifest, manifest_path, review, config, rules)
    return {
        "ok": True,
        "runKey": manifest.get("runKey"),
        "precomputeManifest": repo_url(manifest_path),
        "frameCount": manifest.get("frameCount"),
        "image": {"width": manifest.get("width"), "height": manifest.get("height")},
        "instanceTracking": status,
        "instances": status,
        "job": current_instance_tracking_job(),
    }


def coerce_optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def coerce_optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def coerce_optional_bool(value, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def serialize_read_only_assets() -> dict:
    active_name = active_frame_run_name()
    runs = []
    if DEFAULT_FRAME_RUNS_ROOT.exists():
        for run_dir in sorted((item for item in DEFAULT_FRAME_RUNS_ROOT.iterdir() if item.is_dir()), key=lambda item: natural_key(item.name)):
            decoded_path = run_dir / HNL_DECODED_MANIFEST_NAME
            decoded = read_json_optional(decoded_path)
            precompute_path = precompute_manifest_for_decoded(decoded)
            precompute = read_json_optional(precompute_path) if precompute_path else None
            runs.append(
                {
                    "name": run_dir.name,
                    "active": run_dir.name == active_name,
                    "decoded": bool(decoded),
                    "decodedManifestPath": repo_url(decoded_path) if decoded_path.exists() else None,
                    "frameCount": decoded.get("frameCount") if decoded else None,
                    "width": decoded.get("width") if decoded else None,
                    "height": decoded.get("height") if decoded else None,
                    "precompute": {
                        "ready": bool(precompute),
                        "manifestPath": repo_url(precompute_path) if precompute_path else None,
                        "runKey": precompute.get("runKey") if precompute else None,
                        "builtAt": precompute.get("builtAt") if precompute else None,
                    },
                }
            )
    return {
        "frameRunsRoot": repo_url(DEFAULT_FRAME_RUNS_ROOT),
        "colorLayersRoot": repo_url(DEFAULT_HNL_LAYER_SETS.parent),
        "layerSetsPath": repo_url(DEFAULT_HNL_LAYER_SETS),
        "activeRunName": active_name,
        "runs": runs,
    }


def serialize_config() -> dict:
    config = read_json(CONFIG_PATH)
    rules = read_json_or_default(RULES_PATH, default_rules())
    memory = read_json_or_default(MEMORY_PATH, default_memory())
    manifest_path = selected_precompute_manifest()
    manifest = read_json(manifest_path)
    return {
        "ok": True,
        "app": "vision_passthrough_review",
        "config": config,
        "readOnlyAssets": serialize_read_only_assets(),
        "dependencyStatus": dependency_status(),
        "rulesUpdatedAt": rules.get("updatedAt"),
        "memorySummary": memory.get("summary", {}),
        "bboxes": serialize_bbox_status(),
        "maskbitsBboxes": serialize_maskbits_bbox_status(),
        "bboxClipping": serialize_bbox_clipping_status(),
        "bboxContours": serialize_bbox_contours_status(),
        "contours": serialize_contour_status(),
        "corners": serialize_corner_status(),
        "poseEstimation": serialize_pose_estimation_status(),
        "instanceTracking": serialize_instance_tracking_status(),
        "squarePose": serialize_square_pose_status(),
        "instances": serialize_instance_status(),
        "precompute": {
            "manifestPath": "/" + manifest_path.relative_to(REPO_ROOT).as_posix(),
            "runKey": manifest.get("runKey"),
            "frameCount": manifest.get("frameCount"),
            "width": manifest.get("width"),
            "height": manifest.get("height"),
            "colorTable": manifest.get("colorTable"),
            "frames": manifest.get("frames", []),
        },
    }


def _path_from_payload(value) -> Path | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    if text.startswith("/0721Vision/"):
        path = REPO_ROOT / text.lstrip("/")
    elif text.startswith(("/src/", "/assets/", "/vendor/")):
        path = APP_DIR / text.lstrip("/")
    else:
        path = Path(text).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def jpeg_frames_in_dir(source_dir: Path) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    return sorted(
        (item for item in source_dir.iterdir() if item.is_file() and item.suffix.lower() in JPEG_SUFFIXES),
        key=lambda item: natural_key(item.name),
    )


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def source_folder_payload(
    source_dir: Path,
    *,
    name: str,
    label: str,
    kind: str,
    active: bool = False,
    decoded_manifest_path: Path | None = None,
    precompute_path: Path | None = None,
    precompute: dict | None = None,
) -> dict:
    frames = jpeg_frames_in_dir(source_dir)
    first_frame = frames[0] if frames else None
    last_frame = frames[-1] if frames else None
    width = None
    height = None
    dimensions_error = None
    if first_frame is not None:
        try:
            width, height = jpeg_dimensions(first_frame)
        except Exception as error:
            dimensions_error = str(error)
    dimensions_ok = width == 640 and height == 360
    return {
        "id": display_path(source_dir) if source_dir.exists() else str(source_dir),
        "name": name,
        "label": label,
        "kind": kind,
        "active": bool(active),
        "ok": bool(frames) and dimensions_ok and not dimensions_error,
        "sourceDir": display_path(source_dir) if source_dir.exists() else str(source_dir),
        "absolutePath": str(source_dir),
        "frameCount": len(frames),
        "firstFrame": display_path(first_frame) if first_frame else None,
        "lastFrame": display_path(last_frame) if last_frame else None,
        "width": width,
        "height": height,
        "dimensionsOk": dimensions_ok,
        "dimensionsError": dimensions_error,
        "decodedManifestPath": display_path(decoded_manifest_path) if decoded_manifest_path and decoded_manifest_path.exists() else None,
        "precompute": {
            "manifestPath": display_path(precompute_path) if precompute_path and precompute_path.exists() else None,
            "frameCount": precompute.get("frameCount") if precompute else None,
            "runKey": precompute.get("runKey") if precompute else None,
        },
    }


def discover_pipeline_sources() -> dict:
    active_name = active_frame_run_name()
    sources: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()

    def add_source(source_dir: Path, **metadata) -> None:
        try:
            key = str(source_dir.resolve())
        except FileNotFoundError:
            key = str(source_dir)
        if key in seen:
            return
        seen.add(key)
        payload = source_folder_payload(source_dir, **metadata)
        (sources if payload["ok"] else rejected).append(payload)

    color_input_dir = SRC_DIR / "color_masks" / "input_frames"
    add_source(
        color_input_dir,
        name="color_masks/input_frames",
        label="color_masks/input_frames",
        kind="color-mask-input",
        active=False,
    )

    if DEFAULT_FRAME_RUNS_ROOT.exists():
        for run_dir in sorted((item for item in DEFAULT_FRAME_RUNS_ROOT.iterdir() if item.is_dir()), key=lambda item: natural_key(item.name)):
            frames_dir = run_dir / "vision_frames"
            decoded_path = run_dir / HNL_DECODED_MANIFEST_NAME
            decoded = read_json_optional(decoded_path)
            precompute_path = precompute_manifest_for_decoded(decoded)
            precompute = read_json_optional(precompute_path) if precompute_path else None
            run_name = run_dir.name
            frame_count = len(jpeg_frames_in_dir(frames_dir))
            label = f"{run_name} / vision_frames"
            if active_name == run_name:
                label = f"{label} | active"
            if frame_count:
                label = f"{label} | {frame_count} frames"
            add_source(
                frames_dir,
                name=run_name,
                label=label,
                kind="frame-run",
                active=active_name == run_name,
                decoded_manifest_path=decoded_path,
                precompute_path=precompute_path,
                precompute=precompute,
            )

    sources.sort(key=lambda item: (not item.get("active"), natural_key(str(item.get("name") or ""))))
    rejected.sort(key=lambda item: natural_key(str(item.get("name") or "")))
    return {
        "ok": True,
        "schema": "0721vision-ui-sources.v1",
        "expected": {"width": 640, "height": 360, "suffixes": sorted(JPEG_SUFFIXES)},
        "sources": sources,
        "rejected": rejected,
    }


def selected_source_dir_from_payload(payload: dict) -> Path | None:
    return _path_from_payload(payload.get("sourceDir") or payload.get("sourceId"))


def playback_target_hz(payload: dict) -> float:
    hz = coerce_optional_float(payload.get("targetHz"))
    if hz is None:
        return PLAYBACK_TARGET_HZ
    return max(1.0, min(240.0, hz))


def playback_readiness_payload(payload: dict) -> dict:
    source_dir = selected_source_dir_from_payload(payload)
    target_hz = playback_target_hz(payload)
    max_frames = coerce_optional_int(payload.get("maxFrames"))
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict] = []

    def add_check(name: str, ok: bool, detail: str, *, level: str = "blocker") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if ok:
            return
        if level == "warning":
            warnings.append(detail)
        else:
            blockers.append(detail)

    add_check(
        "pipeline import",
        PIPELINE_IMPORT_ERROR is None and PipelineConfig is not None and VisionPipeline is not None,
        "pipeline module is available" if PIPELINE_IMPORT_ERROR is None else f"pipeline unavailable: {PIPELINE_IMPORT_ERROR}",
    )
    add_check("config", CONFIG_PATH.exists(), f"config: {repo_url(CONFIG_PATH)}")
    add_check("review", REVIEW_PATH.exists(), f"review state: {repo_url(REVIEW_PATH)}", level="warning")
    lut_path = PipelineConfig.lut_path if PipelineConfig is not None else SRC_DIR / "color_masks" / "artifacts" / "color_lut_v1.npz"
    add_check("color LUT", lut_path.exists(), f"LUT: {repo_url(lut_path)}")

    frames: list[Path] = []
    bad_frames: list[dict] = []
    width = None
    height = None
    if source_dir is None:
        add_check("source", False, "sourceDir is required")
    else:
        add_check("source exists", source_dir.exists() and source_dir.is_dir(), f"source directory: {source_dir}")
        frames = jpeg_frames_in_dir(source_dir)
        add_check("jpeg frames", bool(frames), f"{len(frames)} JPEG frames found")
        inspect_frames = frames[:max_frames] if max_frames is not None else frames
        for index, frame_path in enumerate(inspect_frames):
            try:
                frame_width, frame_height = jpeg_dimensions(frame_path)
            except Exception as error:
                bad_frames.append({"index": index, "path": display_path(frame_path), "error": str(error)})
                continue
            if width is None:
                width, height = frame_width, frame_height
            if frame_width != 640 or frame_height != 360:
                bad_frames.append(
                    {
                        "index": index,
                        "path": display_path(frame_path),
                        "width": frame_width,
                        "height": frame_height,
                        "error": "expected 640x360",
                    }
                )
        add_check(
            "frame dimensions",
            bool(frames) and not bad_frames,
            "all inspected JPEG frames are 640x360" if frames and not bad_frames else f"{len(bad_frames)} frame dimension/decode issues",
        )

    with PIPELINE_LOCK:
        pipeline_busy = _active_pipeline_locked()
    with PLAYBACK_LOCK:
        playback_busy = bool(PLAYBACK_THREAD and PLAYBACK_THREAD.is_alive() and PLAYBACK_FEEDER)
    add_check("pipeline idle", not pipeline_busy, "pipeline is idle" if not pipeline_busy else "pipeline is already running")
    add_check("playback idle", not playback_busy, "playback feeder is idle" if not playback_busy else "playback feeder is already running")

    if target_hz != PLAYBACK_TARGET_HZ:
        warnings.append(f"targetHz normalized to {target_hz:g}; production review target is 30Hz")

    effective_frame_count = len(frames[:max_frames]) if max_frames is not None else len(frames)
    return {
        "ok": not blockers,
        "schema": "0721vision-ui-readiness.v1",
        "sourceDir": display_path(source_dir) if source_dir and source_dir.exists() else str(source_dir or ""),
        "frameCount": effective_frame_count,
        "sourceFrameCount": len(frames),
        "targetHz": target_hz,
        "periodMs": round(1000.0 / target_hz, 3),
        "expected": {"width": 640, "height": 360},
        "observed": {"width": width, "height": height},
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "badFrames": bad_frames[:12],
    }


def _active_pipeline_locked() -> bool:
    return bool(PIPELINE_THREAD and PIPELINE_THREAD.is_alive() and PIPELINE_RUNNER and PIPELINE_RUNNER.status_snapshot().get("state") in {"created", "starting", "running", "stopping"})


def pipeline_status_payload() -> dict:
    if PIPELINE_IMPORT_ERROR is not None:
        return {"ok": False, "error": f"pipeline unavailable: {PIPELINE_IMPORT_ERROR}", "playback": current_playback_status()}
    with PIPELINE_LOCK:
        runner = PIPELINE_RUNNER
        thread = PIPELINE_THREAD
    if runner is not None:
        status = runner.status_snapshot()
        status["threadAlive"] = bool(thread and thread.is_alive())
        return {"ok": True, "pipeline": status, "playback": current_playback_status()}
    status = latest_pipeline_status() if latest_pipeline_status else {"state": "idle", "manifestUrls": {}}
    return {"ok": True, "pipeline": status, "playback": current_playback_status()}


def start_pipeline_job(payload: dict) -> dict:
    global PIPELINE_RUNNER, PIPELINE_THREAD
    if PIPELINE_IMPORT_ERROR is not None or PipelineConfig is None or VisionPipeline is None:
        return {"ok": False, "error": f"pipeline unavailable: {PIPELINE_IMPORT_ERROR}"}
    mode = str(payload.get("mode") or "watch").lower()
    if mode not in {"batch", "watch", "live"}:
        return {"ok": False, "error": "pipeline mode must be batch, watch, or live"}
    source_dir = _path_from_payload(payload.get("sourceDir"))
    live_point = payload.get("livePoint")
    if source_dir is None and not live_point:
        return {"ok": False, "error": "sourceDir or livePoint is required"}
    no_debug = coerce_optional_bool(payload.get("noDebug"), False)
    debug_artifacts = coerce_optional_bool(payload.get("debugArtifacts"), False) and not no_debug
    aggregate_debug_manifests = coerce_optional_bool(payload.get("aggregateDebugManifests"), False) and not no_debug
    with PIPELINE_LOCK:
        if _active_pipeline_locked():
            return {"ok": False, "conflict": True, "error": "pipeline already running", "pipeline": PIPELINE_RUNNER.status_snapshot()}
        config = PipelineConfig(
            mode=mode,
            source_dir=source_dir,
            live_point=str(live_point) if live_point else None,
            output_root=_path_from_payload(payload.get("outputRoot")) or APP_DIR / "assets" / "pipeline_runs",
            artifact_root=_path_from_payload(payload.get("artifactRoot")) or APP_DIR / "assets",
            poll_interval_s=max(0.001, coerce_optional_float(payload.get("pollIntervalSeconds")) or PipelineConfig.poll_interval_s),
            stable_frame_s=max(0.0, coerce_optional_float(payload.get("stableFrameSeconds")) or PipelineConfig.stable_frame_s),
            max_frames=coerce_optional_int(payload.get("maxFrames")),
            parallel_stateless=payload.get("parallelStateless") is not False,
            debug_artifacts=debug_artifacts,
            aggregate_debug_manifests=aggregate_debug_manifests,
        )
        runner = VisionPipeline(config)
        PIPELINE_RUNNER = runner

        def run_pipeline() -> None:
            try:
                runner.run()
            except Exception:
                traceback.print_exc()

        PIPELINE_THREAD = threading.Thread(target=run_pipeline, daemon=True, name=f"0721vision-pipeline-{runner.run_id}")
        PIPELINE_THREAD.start()
        return {"ok": True, "conflict": False, "pipeline": runner.status_snapshot()}


def stop_pipeline_job() -> dict:
    with PIPELINE_LOCK:
        runner = PIPELINE_RUNNER
    if runner is None:
        return {"ok": True, "pipeline": {"state": "idle", "manifestUrls": {}}}
    runner.stop()
    return {"ok": True, "pipeline": runner.status_snapshot()}


def playback_frame_name(index: int, source_path: Path) -> str:
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in source_path.stem)
    if not safe_stem:
        safe_stem = f"{index:06d}"
    suffix = source_path.suffix.lower() if source_path.suffix.lower() in JPEG_SUFFIXES else ".jpg"
    return f"frame_{index:06d}_{safe_stem}{suffix}"


def atomic_copy_frame(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=str(target_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source_path, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class PlaybackFeeder:
    def __init__(self, source_dir: Path, frames: list[Path], target_hz: float):
        self.playback_id = f"playback-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        self.source_dir = source_dir
        self.frames = list(frames)
        self.target_hz = target_hz
        self.period_s = 1.0 / target_hz
        self.feed_root = PLAYBACK_ROOT / self.playback_id
        self.frames_dir = self.feed_root / "frames"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.status = {
            "schema": "0721vision-ui-playback.v1",
            "playbackId": self.playback_id,
            "state": "queued",
            "targetHz": self.target_hz,
            "periodMs": round(self.period_s * 1000.0, 3),
            "sourceDir": display_path(self.source_dir),
            "feedDir": display_path(self.frames_dir),
            "sourceFrameCount": len(self.frames),
            "fedCount": 0,
            "latestFedFrame": None,
            "latestFedPath": None,
            "latestCompletedFrame": None,
            "completedCount": 0,
            "behindByFrames": 0,
            "deadlineMissed": False,
            "missedDeadlineCount": 0,
            "lastMissedFrame": None,
            "maxBehindByFrames": 0,
            "feederDriftMs": 0.0,
            "maxFeederDriftMs": 0.0,
            "pipelineRunId": None,
            "pipelineState": "idle",
            "pipelineDrained": False,
            "drainTimedOut": False,
            "startedAt": None,
            "updatedAt": utc_now(),
            "completedAt": None,
            "error": None,
        }

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.status))

    def _set_status(self, **updates) -> None:
        with self.lock:
            self.status.update(updates)
            self.status["updatedAt"] = utc_now()

    def stop(self) -> None:
        self.stop_event.set()
        self._set_status(state="stopping")

    def _pipeline_status(self) -> dict:
        with PIPELINE_LOCK:
            runner = PIPELINE_RUNNER
        if runner is not None:
            return runner.status_snapshot()
        return latest_pipeline_status() if latest_pipeline_status else {"state": "idle", "manifestUrls": {}}

    def _update_pipeline_progress(self) -> dict:
        pipeline = self._pipeline_status()
        completed_count = int(pipeline.get("frameCountCompleted") or 0)
        latest_completed = pipeline.get("latestCompletedFrame")
        with self.lock:
            fed_count = int(self.status.get("fedCount") or 0)
            behind = max(0, fed_count - completed_count)
            self.status.update(
                {
                    "pipelineRunId": pipeline.get("runId") or pipeline.get("runKey"),
                    "pipelineState": pipeline.get("state") or "idle",
                    "completedCount": completed_count,
                    "latestCompletedFrame": latest_completed,
                    "behindByFrames": behind,
                    "maxBehindByFrames": max(int(self.status.get("maxBehindByFrames") or 0), behind),
                    "deadlineMissed": bool(self.status.get("deadlineMissed")),
                    "updatedAt": utc_now(),
                }
            )
        return pipeline

    def _record_deadline_if_needed(self, expected_completed_count: int, missed_frame: int) -> None:
        pipeline = self._pipeline_status()
        completed_count = int(pipeline.get("frameCountCompleted") or 0)
        missed = completed_count < expected_completed_count
        with self.lock:
            fed_count = int(self.status.get("fedCount") or 0)
            behind = max(0, fed_count - completed_count)
            updates = {
                "pipelineRunId": pipeline.get("runId") or pipeline.get("runKey"),
                "pipelineState": pipeline.get("state") or "idle",
                "completedCount": completed_count,
                "latestCompletedFrame": pipeline.get("latestCompletedFrame"),
                "behindByFrames": behind,
                "maxBehindByFrames": max(int(self.status.get("maxBehindByFrames") or 0), behind),
                "updatedAt": utc_now(),
            }
            if missed:
                updates.update(
                    {
                        "deadlineMissed": True,
                        "missedDeadlineCount": int(self.status.get("missedDeadlineCount") or 0) + 1,
                        "lastMissedFrame": missed_frame,
                    }
                )
            self.status.update(updates)

    def run(self) -> None:
        try:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            start_payload = {
                "mode": "watch",
                "sourceDir": str(self.frames_dir),
                "maxFrames": len(self.frames),
                "pollIntervalSeconds": min(0.01, self.period_s / 4.0),
                "stableFrameSeconds": min(0.005, self.period_s / 4.0),
                "parallelStateless": True,
                "noDebug": True,
            }
            start_result = start_pipeline_job(start_payload)
            if not start_result.get("ok"):
                self._set_status(state="error", error=start_result.get("error") or "pipeline start failed", completedAt=utc_now())
                return
            pipeline = start_result.get("pipeline") or {}
            self._set_status(
                state="running",
                startedAt=utc_now(),
                pipelineRunId=pipeline.get("runId") or pipeline.get("runKey"),
                pipelineState=pipeline.get("state") or "starting",
            )
            start_time = time.perf_counter()
            for index, source_path in enumerate(self.frames):
                target_time = start_time + index * self.period_s
                remaining = target_time - time.perf_counter()
                if remaining > 0 and self.stop_event.wait(remaining):
                    break
                if self.stop_event.is_set():
                    break
                if index > 0:
                    self._record_deadline_if_needed(index, index - 1)
                drift_ms = max(0.0, (time.perf_counter() - target_time) * 1000.0)
                target_path = self.frames_dir / playback_frame_name(index, source_path)
                atomic_copy_frame(source_path, target_path)
                with self.lock:
                    self.status.update(
                        {
                            "fedCount": index + 1,
                            "latestFedFrame": index,
                            "latestFedPath": display_path(target_path),
                            "feederDriftMs": round(drift_ms, 3),
                            "maxFeederDriftMs": round(max(float(self.status.get("maxFeederDriftMs") or 0.0), drift_ms), 3),
                            "updatedAt": utc_now(),
                        }
                    )
                self._update_pipeline_progress()

            if self.stop_event.is_set():
                stop_pipeline_job()
                self._update_pipeline_progress()
                self._set_status(state="stopped", completedAt=utc_now())
                return

            self._set_status(state="draining")
            drain_deadline = time.monotonic() + max(5.0, len(self.frames) * self.period_s * 2.0)
            while not self.stop_event.is_set() and time.monotonic() < drain_deadline:
                pipeline = self._update_pipeline_progress()
                if int(pipeline.get("frameCountCompleted") or 0) >= len(self.frames):
                    break
                time.sleep(min(0.05, self.period_s))
            pipeline = self._update_pipeline_progress()
            drained = int(pipeline.get("frameCountCompleted") or 0) >= len(self.frames)
            stop_pipeline_job()
            self._update_pipeline_progress()
            previous_status = self.snapshot()
            self._set_status(
                state="complete" if not self.stop_event.is_set() else "stopped",
                pipelineDrained=drained,
                drainTimedOut=not drained,
                deadlineMissed=bool(previous_status.get("deadlineMissed")) or not drained,
                missedDeadlineCount=int(previous_status.get("missedDeadlineCount") or 0) + (0 if drained else 1),
                lastMissedFrame=previous_status.get("lastMissedFrame") if drained else max(0, len(self.frames) - 1),
                completedAt=utc_now(),
            )
        except Exception as error:
            traceback.print_exc()
            stop_pipeline_job()
            self._set_status(state="error", error=str(error), completedAt=utc_now(), deadlineMissed=True)


def current_playback_status() -> dict:
    with PLAYBACK_LOCK:
        feeder = PLAYBACK_FEEDER
        thread = PLAYBACK_THREAD
    if feeder is None:
        return {
            "schema": "0721vision-ui-playback.v1",
            "state": "idle",
            "targetHz": PLAYBACK_TARGET_HZ,
            "periodMs": round(PLAYBACK_PERIOD_S * 1000.0, 3),
        }
    status = feeder.snapshot()
    status["threadAlive"] = bool(thread and thread.is_alive())
    return status


def start_playback_job(payload: dict) -> dict:
    global PLAYBACK_FEEDER, PLAYBACK_THREAD
    readiness = playback_readiness_payload(payload)
    if not readiness.get("ok"):
        return {"ok": False, "conflict": False, "readiness": readiness, "error": "readiness failed"}
    source_dir = selected_source_dir_from_payload(payload)
    if source_dir is None:
        return {"ok": False, "conflict": False, "readiness": readiness, "error": "sourceDir is required"}
    frames = jpeg_frames_in_dir(source_dir)
    max_frames = coerce_optional_int(payload.get("maxFrames"))
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        return {"ok": False, "conflict": False, "readiness": readiness, "error": "source directory has no JPEG frames"}
    feeder = PlaybackFeeder(source_dir, frames, playback_target_hz(payload))
    with PLAYBACK_LOCK:
        active = bool(PLAYBACK_THREAD and PLAYBACK_THREAD.is_alive() and PLAYBACK_FEEDER)
        if active:
            return {"ok": False, "conflict": True, "playback": PLAYBACK_FEEDER.snapshot(), "error": "playback already running"}
        PLAYBACK_FEEDER = feeder
        PLAYBACK_THREAD = threading.Thread(target=feeder.run, daemon=True, name=f"0721vision-playback-{feeder.playback_id}")
        PLAYBACK_THREAD.start()
    return {"ok": True, "conflict": False, "readiness": readiness, "playback": feeder.snapshot()}


def stop_playback_job() -> dict:
    with PLAYBACK_LOCK:
        feeder = PLAYBACK_FEEDER
    if feeder is not None:
        feeder.stop()
    pipeline = stop_pipeline_job()
    return {"ok": True, "playback": current_playback_status(), "pipeline": pipeline.get("pipeline")}


def set_job_updates(**updates) -> None:
    global REBUILD_JOB
    with JOBS_LOCK:
        if REBUILD_JOB is None:
            return
        REBUILD_JOB.update(updates)
        REBUILD_JOB["updatedAt"] = utc_now()


def run_rebuild_job(job_id: str, settings: dict) -> None:
    try:
        set_job_updates(status="running", progress={"phase": "starting", "index": 0, "total": None})

        def progress(payload: dict) -> None:
            set_job_updates(progress=payload)

        args = SimpleNamespace(
            config=CONFIG_PATH,
            rules=RULES_PATH,
            review=REVIEW_PATH,
            output=MEMORY_PATH,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=settings.get("maxFrames"),
        )
        if args.max_frames is not None:
            args.max_frames = int(args.max_frames)
        result = build_memory(args, progress_callback=progress)
        set_job_updates(
            status="complete",
            progress={"phase": "complete", "index": result["source"]["processedFrameCount"], "total": result["source"]["processedFrameCount"]},
            result={
                "output": "/" + MEMORY_PATH.relative_to(REPO_ROOT).as_posix(),
                "summary": result.get("summary", {}),
                "ruleHash": result.get("classRules", {}).get("hash"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_job() -> dict | None:
    with JOBS_LOCK:
        if REBUILD_JOB and REBUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(REBUILD_JOB))
    return None


def current_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(REBUILD_JOB)) if REBUILD_JOB else None


def set_bbox_job_updates(**updates) -> None:
    global BBOX_BUILD_JOB
    with JOBS_LOCK:
        if BBOX_BUILD_JOB is None:
            return
        BBOX_BUILD_JOB.update(updates)
        BBOX_BUILD_JOB["updatedAt"] = utc_now()


def bbox_build_dependency_error() -> str | None:
    try:
        import cv2  # noqa: F401

        return None
    except Exception as error:
        return str(error)


def run_bbox_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = bbox_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build mask bboxes: {missing}")
        clean_settings = normalized_bbox_settings(settings)
        set_bbox_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_bbox_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            config=CONFIG_PATH,
            rules=RULES_PATH,
            review=REVIEW_PATH,
            output_root=DEFAULT_BBOX_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            min_pixels=clean_settings["minPixels"],
            max_bboxes_per_frame=clean_settings["maxBboxesPerFrame"],
            fit_tightness=clean_settings["fitTightness"],
            quad_fit_enabled=clean_settings["quadFitEnabled"],
            quad_fit_mode=clean_settings["quadFitMode"],
            target_aspect=clean_settings["targetAspect"],
            aspect_tolerance=clean_settings["aspectTolerance"],
            quad_thickness_px=clean_settings["quadThicknessPx"],
            edge_coverage_min=clean_settings["edgeCoverageMin"],
            corner_min_pixels=clean_settings["cornerMinPixels"],
            void_overlap_max_ratio=clean_settings["voidOverlapMaxRatio"],
            void_overlap_min_pixels=clean_settings["voidOverlapMinPixels"],
            fov_clip=clean_settings["fovClip"],
        )
        result = build_mask_bboxes(args, progress_callback=progress)
        set_bbox_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(bbox_run_root(str(result.get("runKey") or ""), DEFAULT_BBOX_ROOT) / "bbox_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_bbox_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_bbox_job() -> dict | None:
    with JOBS_LOCK:
        if BBOX_BUILD_JOB and BBOX_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(BBOX_BUILD_JOB))
    return None


def current_bbox_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(BBOX_BUILD_JOB)) if BBOX_BUILD_JOB else None


def start_bbox_build_job(settings: dict) -> dict:
    global BBOX_BUILD_JOB
    existing = active_bbox_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_bbox_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        BBOX_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(BBOX_BUILD_JOB))
    thread = threading.Thread(target=run_bbox_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_corner_job_updates(**updates) -> None:
    global CORNER_BUILD_JOB
    with JOBS_LOCK:
        if CORNER_BUILD_JOB is None:
            return
        CORNER_BUILD_JOB.update(updates)
        CORNER_BUILD_JOB["updatedAt"] = utc_now()


def corner_build_dependency_error() -> str | None:
    try:
        import cv2  # noqa: F401

        return None
    except Exception as error:
        return str(error)


def run_corner_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = corner_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build mask corners: {missing}")
        clean_settings = normalized_corner_settings(settings)
        set_corner_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_corner_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            config=CONFIG_PATH,
            rules=RULES_PATH,
            review=REVIEW_PATH,
            output_root=DEFAULT_CORNER_ROOT,
            bbox_output_root=DEFAULT_BBOX_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            corner_settings=clean_settings,
        )
        result = build_mask_corners(args, progress_callback=progress)
        set_corner_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(corner_run_root(str(result.get("runKey") or ""), DEFAULT_CORNER_ROOT) / "corners_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_corner_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_corner_job() -> dict | None:
    with JOBS_LOCK:
        if CORNER_BUILD_JOB and CORNER_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(CORNER_BUILD_JOB))
    return None


def current_corner_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(CORNER_BUILD_JOB)) if CORNER_BUILD_JOB else None


def start_corner_build_job(settings: dict) -> dict:
    global CORNER_BUILD_JOB
    existing = active_corner_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_corner_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        CORNER_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(CORNER_BUILD_JOB))
    thread = threading.Thread(target=run_corner_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_maskbits_bbox_job_updates(**updates) -> None:
    global MASKBITS_BBOX_BUILD_JOB
    with JOBS_LOCK:
        if MASKBITS_BBOX_BUILD_JOB is None:
            return
        MASKBITS_BBOX_BUILD_JOB.update(updates)
        MASKBITS_BBOX_BUILD_JOB["updatedAt"] = utc_now()


def run_maskbits_bbox_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = bbox_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build maskbits bboxes: {missing}")
        clean_settings = normalized_bbox_settings(settings)
        set_maskbits_bbox_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_maskbits_bbox_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            config=CONFIG_PATH,
            rules=RULES_PATH,
            review=REVIEW_PATH,
            output_root=DEFAULT_MASKBITS_BBOX_ROOT,
            precompute_manifest=None,
            mask_manifest=DEFAULT_MASK_MANIFEST,
            compare_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            min_pixels=clean_settings["minPixels"],
            max_bboxes_per_frame=clean_settings["maxBboxesPerFrame"],
            fit_tightness=clean_settings["fitTightness"],
            quad_fit_enabled=clean_settings["quadFitEnabled"],
            quad_fit_mode=clean_settings["quadFitMode"],
            target_aspect=clean_settings["targetAspect"],
            aspect_tolerance=clean_settings["aspectTolerance"],
            quad_thickness_px=clean_settings["quadThicknessPx"],
            edge_coverage_min=clean_settings["edgeCoverageMin"],
            corner_min_pixels=clean_settings["cornerMinPixels"],
            void_overlap_max_ratio=clean_settings["voidOverlapMaxRatio"],
            void_overlap_min_pixels=clean_settings["voidOverlapMinPixels"],
            fov_clip=clean_settings["fovClip"],
        )
        result = build_maskbits_bboxes(args, progress_callback=progress)
        set_maskbits_bbox_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(maskbits_bbox_run_root(str(result.get("runKey") or ""), DEFAULT_MASKBITS_BBOX_ROOT) / "bbox_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "comparison": result.get("comparison"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_maskbits_bbox_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_maskbits_bbox_job() -> dict | None:
    with JOBS_LOCK:
        if MASKBITS_BBOX_BUILD_JOB and MASKBITS_BBOX_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(MASKBITS_BBOX_BUILD_JOB))
    return None


def current_maskbits_bbox_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(MASKBITS_BBOX_BUILD_JOB)) if MASKBITS_BBOX_BUILD_JOB else None


def start_maskbits_bbox_build_job(settings: dict) -> dict:
    global MASKBITS_BBOX_BUILD_JOB
    existing = active_maskbits_bbox_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_bbox_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        MASKBITS_BBOX_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(MASKBITS_BBOX_BUILD_JOB))
    thread = threading.Thread(target=run_maskbits_bbox_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_bbox_clipping_job_updates(**updates) -> None:
    global BBOX_CLIPPING_BUILD_JOB
    with JOBS_LOCK:
        if BBOX_CLIPPING_BUILD_JOB is None:
            return
        BBOX_CLIPPING_BUILD_JOB.update(updates)
        BBOX_CLIPPING_BUILD_JOB["updatedAt"] = utc_now()


def bbox_clipping_dependency_error() -> str | None:
    try:
        import cv2  # noqa: F401

        return None
    except Exception as error:
        return str(error)


def run_bbox_clipping_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = bbox_clipping_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build bbox clipping: {missing}")
        clean_settings = normalized_fov_clip_settings(settings)
        set_bbox_clipping_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_bbox_clipping_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            config=CONFIG_PATH,
            review=REVIEW_PATH,
            precompute_manifest=None,
            mask_manifest=DEFAULT_MASK_MANIFEST,
            bbox_output_root=DEFAULT_MASKBITS_BBOX_ROOT,
            output_root=DEFAULT_BBOX_CLIPPING_ROOT,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            fov_clip=clean_settings,
        )
        result = build_bbox_clipping(args, progress_callback=progress)
        set_bbox_clipping_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(clipping_run_root(str(result.get("runKey") or ""), DEFAULT_BBOX_CLIPPING_ROOT) / "clipping_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_bbox_clipping_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_bbox_clipping_job() -> dict | None:
    with JOBS_LOCK:
        if BBOX_CLIPPING_BUILD_JOB and BBOX_CLIPPING_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(BBOX_CLIPPING_BUILD_JOB))
    return None


def current_bbox_clipping_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(BBOX_CLIPPING_BUILD_JOB)) if BBOX_CLIPPING_BUILD_JOB else None


def start_bbox_clipping_build_job(settings: dict) -> dict:
    global BBOX_CLIPPING_BUILD_JOB
    existing = active_bbox_clipping_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_fov_clip_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        BBOX_CLIPPING_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(BBOX_CLIPPING_BUILD_JOB))
    thread = threading.Thread(target=run_bbox_clipping_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_bbox_contour_job_updates(**updates) -> None:
    global BBOX_CONTOUR_BUILD_JOB
    with JOBS_LOCK:
        if BBOX_CONTOUR_BUILD_JOB is None:
            return
        BBOX_CONTOUR_BUILD_JOB.update(updates)
        BBOX_CONTOUR_BUILD_JOB["updatedAt"] = utc_now()


def run_bbox_contour_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = contour_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build bbox contours: {missing}")
        clean_settings = normalized_contour_settings(settings)
        set_bbox_contour_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_bbox_contour_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            config=CONFIG_PATH,
            review=REVIEW_PATH,
            precompute_manifest=None,
            mask_manifest=DEFAULT_MASK_MANIFEST,
            bbox_output_root=DEFAULT_MASKBITS_BBOX_ROOT,
            output_root=DEFAULT_BBOX_CONTOUR_ROOT,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            contour_settings=clean_settings,
        )
        result = build_bbox_contours(args, progress_callback=progress)
        set_bbox_contour_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(bbox_contour_run_root(str(result.get("runKey") or ""), DEFAULT_BBOX_CONTOUR_ROOT) / "contours_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_bbox_contour_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_bbox_contour_job() -> dict | None:
    with JOBS_LOCK:
        if BBOX_CONTOUR_BUILD_JOB and BBOX_CONTOUR_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(BBOX_CONTOUR_BUILD_JOB))
    return None


def current_bbox_contour_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(BBOX_CONTOUR_BUILD_JOB)) if BBOX_CONTOUR_BUILD_JOB else None


def start_bbox_contour_build_job(settings: dict) -> dict:
    global BBOX_CONTOUR_BUILD_JOB
    existing = active_bbox_contour_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_contour_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        BBOX_CONTOUR_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(BBOX_CONTOUR_BUILD_JOB))
    thread = threading.Thread(target=run_bbox_contour_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_contour_job_updates(**updates) -> None:
    global CONTOUR_BUILD_JOB
    with JOBS_LOCK:
        if CONTOUR_BUILD_JOB is None:
            return
        CONTOUR_BUILD_JOB.update(updates)
        CONTOUR_BUILD_JOB["updatedAt"] = utc_now()


def contour_build_dependency_error() -> str | None:
    try:
        import cv2  # noqa: F401

        return None
    except Exception as error:
        return str(error)


def run_contour_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = contour_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build mask contours: {missing}")
        clean_settings = normalized_contour_settings(settings)
        set_contour_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_contour_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            config=CONFIG_PATH,
            rules=RULES_PATH,
            review=REVIEW_PATH,
            output_root=DEFAULT_CONTOUR_ROOT,
            bbox_output_root=DEFAULT_BBOX_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            contour_settings=clean_settings,
        )
        result = build_mask_contours(args, progress_callback=progress)
        set_contour_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(contour_run_root(str(result.get("runKey") or ""), DEFAULT_CONTOUR_ROOT) / "contours_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_contour_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_contour_job() -> dict | None:
    with JOBS_LOCK:
        if CONTOUR_BUILD_JOB and CONTOUR_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(CONTOUR_BUILD_JOB))
    return None


def current_contour_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(CONTOUR_BUILD_JOB)) if CONTOUR_BUILD_JOB else None


def start_contour_build_job(settings: dict) -> dict:
    global CONTOUR_BUILD_JOB
    existing = active_contour_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_contour_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        CONTOUR_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(CONTOUR_BUILD_JOB))
    thread = threading.Thread(target=run_contour_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_square_pose_job_updates(**updates) -> None:
    global SQUARE_POSE_BUILD_JOB
    with JOBS_LOCK:
        if SQUARE_POSE_BUILD_JOB is None:
            return
        SQUARE_POSE_BUILD_JOB.update(updates)
        SQUARE_POSE_BUILD_JOB["updatedAt"] = utc_now()


def square_pose_build_dependency_error() -> str | None:
    try:
        import cv2  # noqa: F401

        return None
    except Exception as error:
        return str(error)


def run_square_pose_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = square_pose_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build square pose: {missing}")
        clean_settings = normalized_square_pose_settings(settings)
        set_square_pose_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_square_pose_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            output_root=DEFAULT_SQUARE_POSE_ROOT,
            contour_output_root=DEFAULT_CONTOUR_ROOT,
            bbox_output_root=DEFAULT_BBOX_ROOT,
            corner_output_root=DEFAULT_CORNER_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            square_pose_settings=clean_settings,
        )
        result = build_square_pose(args, progress_callback=progress)
        set_square_pose_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(square_pose_run_root(str(result.get("runKey") or ""), DEFAULT_SQUARE_POSE_ROOT) / "square_pose_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_square_pose_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_square_pose_job() -> dict | None:
    with JOBS_LOCK:
        if SQUARE_POSE_BUILD_JOB and SQUARE_POSE_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(SQUARE_POSE_BUILD_JOB))
    return None


def current_square_pose_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(SQUARE_POSE_BUILD_JOB)) if SQUARE_POSE_BUILD_JOB else None


def start_square_pose_build_job(settings: dict) -> dict:
    global SQUARE_POSE_BUILD_JOB
    existing = active_square_pose_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_square_pose_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        SQUARE_POSE_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(SQUARE_POSE_BUILD_JOB))
    thread = threading.Thread(target=run_square_pose_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_instance_job_updates(**updates) -> None:
    global INSTANCE_BUILD_JOB
    with JOBS_LOCK:
        if INSTANCE_BUILD_JOB is None:
            return
        INSTANCE_BUILD_JOB.update(updates)
        INSTANCE_BUILD_JOB["updatedAt"] = utc_now()


def run_instance_build_job(job_id: str, settings: dict) -> None:
    try:
        clean_settings = normalized_instance_settings(settings)
        set_instance_job_updates(
            status="running",
            progress={
                "phase": "starting",
                "index": 0,
                "total": None,
                "settings": clean_settings,
            },
        )

        def progress(payload: dict) -> None:
            set_instance_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            output_root=DEFAULT_INSTANCE_ROOT,
            bbox_output_root=DEFAULT_BBOX_ROOT,
            square_pose_output_root=DEFAULT_SQUARE_POSE_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            instance_settings=clean_settings,
        )
        result = build_instance_provenance(args, progress_callback=progress)
        set_instance_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(instance_run_root(str(result.get("runKey") or ""), DEFAULT_INSTANCE_ROOT) / "instance_manifest.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_instance_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_instance_job() -> dict | None:
    with JOBS_LOCK:
        if INSTANCE_BUILD_JOB and INSTANCE_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(INSTANCE_BUILD_JOB))
    return None


def current_instance_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(INSTANCE_BUILD_JOB)) if INSTANCE_BUILD_JOB else None


def start_instance_build_job(settings: dict) -> dict:
    global INSTANCE_BUILD_JOB
    existing = active_instance_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_instance_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        INSTANCE_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(INSTANCE_BUILD_JOB))
    thread = threading.Thread(target=run_instance_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_pose_estimation_job_updates(**updates) -> None:
    global POSE_ESTIMATION_BUILD_JOB
    with JOBS_LOCK:
        if POSE_ESTIMATION_BUILD_JOB is None:
            return
        POSE_ESTIMATION_BUILD_JOB.update(updates)
        POSE_ESTIMATION_BUILD_JOB["updatedAt"] = utc_now()


def pose_estimation_build_dependency_error() -> str | None:
    try:
        import cv2  # noqa: F401

        return None
    except Exception as error:
        return str(error)


def run_pose_estimation_build_job(job_id: str, settings: dict) -> None:
    try:
        missing = pose_estimation_build_dependency_error()
        if missing:
            raise RuntimeError(f"OpenCV is required to build pose estimation: {missing}")
        clean_settings = normalized_pose_settings(settings)
        set_pose_estimation_job_updates(
            status="running",
            progress={"phase": "starting", "index": 0, "total": None, "settings": clean_settings},
        )

        def progress(payload: dict) -> None:
            set_pose_estimation_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            output_root=DEFAULT_POSE_ESTIMATION_ROOT,
            contour_output_root=DEFAULT_BBOX_CONTOUR_ROOT,
            clipping_output_root=DEFAULT_BBOX_CLIPPING_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            pose_settings=clean_settings,
        )
        result = build_pose_estimation(args, progress_callback=progress)
        set_pose_estimation_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(pose_run_root(str(result.get("runKey") or ""), DEFAULT_POSE_ESTIMATION_ROOT) / "3d_pose_fit.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_pose_estimation_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_pose_estimation_job() -> dict | None:
    with JOBS_LOCK:
        if POSE_ESTIMATION_BUILD_JOB and POSE_ESTIMATION_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(POSE_ESTIMATION_BUILD_JOB))
    return None


def current_pose_estimation_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(POSE_ESTIMATION_BUILD_JOB)) if POSE_ESTIMATION_BUILD_JOB else None


def start_pose_estimation_build_job(settings: dict) -> dict:
    global POSE_ESTIMATION_BUILD_JOB
    existing = active_pose_estimation_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_pose_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        POSE_ESTIMATION_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(POSE_ESTIMATION_BUILD_JOB))
    thread = threading.Thread(target=run_pose_estimation_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


def set_instance_tracking_job_updates(**updates) -> None:
    global INSTANCE_TRACKING_BUILD_JOB
    with JOBS_LOCK:
        if INSTANCE_TRACKING_BUILD_JOB is None:
            return
        INSTANCE_TRACKING_BUILD_JOB.update(updates)
        INSTANCE_TRACKING_BUILD_JOB["updatedAt"] = utc_now()


def run_instance_tracking_build_job(job_id: str, settings: dict) -> None:
    try:
        clean_settings = normalized_instance_tracking_settings(settings)
        set_instance_tracking_job_updates(
            status="running",
            progress={"phase": "starting", "index": 0, "total": None, "settings": clean_settings},
        )

        def progress(payload: dict) -> None:
            set_instance_tracking_job_updates(progress={**payload, "settings": clean_settings})

        args = SimpleNamespace(
            output_root=DEFAULT_INSTANCE_TRACKING_ROOT,
            bbox_output_root=DEFAULT_MASKBITS_BBOX_ROOT,
            pose_output_root=DEFAULT_POSE_ESTIMATION_ROOT,
            precompute_manifest=None,
            start_frame=int(settings.get("startFrame", 0) or 0),
            max_frames=coerce_optional_int(settings.get("maxFrames")),
            instance_settings=clean_settings,
        )
        result = build_instance_tracking(args, progress_callback=progress)
        set_instance_tracking_job_updates(
            status="complete",
            progress={
                "phase": "complete",
                "index": result.get("frameCount"),
                "total": result.get("sourceFrameCount"),
                "settings": clean_settings,
            },
            result={
                "manifest": repo_url(instance_mapping_run_root(str(result.get("runKey") or ""), DEFAULT_INSTANCE_TRACKING_ROOT) / "instance_mapping.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
                "createdAt": result.get("createdAt"),
            },
            error=None,
            traceback=None,
        )
    except Exception as error:
        set_instance_tracking_job_updates(status="error", error=str(error), traceback=traceback.format_exc())


def active_instance_tracking_job() -> dict | None:
    with JOBS_LOCK:
        if INSTANCE_TRACKING_BUILD_JOB and INSTANCE_TRACKING_BUILD_JOB.get("status") in {"queued", "running"}:
            return json.loads(json.dumps(INSTANCE_TRACKING_BUILD_JOB))
    return None


def current_instance_tracking_job() -> dict | None:
    with JOBS_LOCK:
        return json.loads(json.dumps(INSTANCE_TRACKING_BUILD_JOB)) if INSTANCE_TRACKING_BUILD_JOB else None


def start_instance_tracking_build_job(settings: dict) -> dict:
    global INSTANCE_TRACKING_BUILD_JOB
    existing = active_instance_tracking_job()
    if existing:
        return {"conflict": True, "job": existing}
    clean_settings = normalized_instance_tracking_settings(settings)
    now = utc_now()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        INSTANCE_TRACKING_BUILD_JOB = {
            "ok": True,
            "jobId": job_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "settings": clean_settings,
            "progress": {"phase": "queued", "index": 0, "total": None, "settings": clean_settings},
            "result": None,
            "error": None,
            "traceback": None,
        }
        payload = json.loads(json.dumps(INSTANCE_TRACKING_BUILD_JOB))
    thread = threading.Thread(target=run_instance_tracking_build_job, args=(job_id, clean_settings), daemon=True)
    thread.start()
    return {"conflict": False, "job": payload}


class VisionReviewHandler(SimpleHTTPRequestHandler):
    server_version = "VisionReviewHTTP/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def api_endpoint(self) -> str | None:
        path = urlparse(self.path).path.rstrip("/")
        if path == API_PREFIX:
            return ""
        prefix = f"{API_PREFIX}/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return None

    def pipeline_endpoint(self) -> str | None:
        path = urlparse(self.path).path.rstrip("/")
        if path == PIPELINE_API_PREFIX:
            return "status"
        prefix = f"{PIPELINE_API_PREFIX}/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return None

    def send_json(self, status: HTTPStatus, payload) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json(status, {"ok": False, "error": message})

    def read_json_body(self, fallback: str = "{}"):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "invalid content length")
            return None
        if length > MAX_BODY_BYTES:
            self.send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload too large")
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or fallback)
        except json.JSONDecodeError as error:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return None

    def do_GET(self) -> None:
        pipeline_endpoint = self.pipeline_endpoint()
        if pipeline_endpoint is not None:
            try:
                if pipeline_endpoint in {"", "status"}:
                    self.send_json(HTTPStatus.OK, pipeline_status_payload())
                    return
                if pipeline_endpoint == "sources":
                    self.send_json(HTTPStatus.OK, discover_pipeline_sources())
                    return
                if pipeline_endpoint == "runs":
                    runs_root = APP_DIR / "assets" / "pipeline_runs"
                    runs = []
                    if runs_root.exists():
                        for run_dir in sorted((item for item in runs_root.iterdir() if item.is_dir()), key=lambda item: item.name, reverse=True):
                            status_path = run_dir / "status.json"
                            manifest_path = run_dir / "run_manifest.json"
                            status = read_json_optional(status_path)
                            runs.append(
                                {
                                    "runId": run_dir.name,
                                    "statusUrl": repo_url(status_path) if status_path.exists() else None,
                                    "manifestUrl": repo_url(manifest_path) if manifest_path.exists() else None,
                                    "state": status.get("state") if status else None,
                                    "frameCountCompleted": status.get("frameCountCompleted") if status else None,
                                    "updatedAt": status.get("updatedAt") if status else None,
                                }
                            )
                    self.send_json(HTTPStatus.OK, {"ok": True, "runs": runs})
                    return
                self.send_error_json(HTTPStatus.NOT_FOUND, "unknown pipeline endpoint")
            except Exception as error:
                self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        endpoint = self.api_endpoint()
        if endpoint is None:
            return super().do_GET()
        try:
            if endpoint in {"", "config"}:
                self.send_json(HTTPStatus.OK, serialize_config())
                return
            if endpoint == "runs":
                self.send_json(HTTPStatus.OK, {"ok": True, "readOnlyAssets": serialize_read_only_assets()})
                return
            if endpoint == "rules":
                self.send_json(HTTPStatus.OK, read_json_or_default(RULES_PATH, default_rules()))
                return
            if endpoint == "review":
                self.send_json(HTTPStatus.OK, read_json_or_default(REVIEW_PATH, default_review()))
                return
            if endpoint == "memory":
                self.send_json(HTTPStatus.OK, read_json_or_default(MEMORY_PATH, default_memory()))
                return
            if endpoint == "bboxes":
                self.send_json(HTTPStatus.OK, serialize_bbox_status())
                return
            if endpoint == "maskbits-bboxes":
                self.send_json(HTTPStatus.OK, serialize_maskbits_bbox_status())
                return
            if endpoint == "bbox-clipping":
                self.send_json(HTTPStatus.OK, serialize_bbox_clipping_status())
                return
            if endpoint == "bbox-contours":
                self.send_json(HTTPStatus.OK, serialize_bbox_contours_status())
                return
            if endpoint == "contours":
                self.send_json(HTTPStatus.OK, serialize_contour_status())
                return
            if endpoint == "corners":
                self.send_json(HTTPStatus.OK, serialize_corner_status())
                return
            if endpoint == "pose-estimation":
                self.send_json(HTTPStatus.OK, serialize_pose_estimation_status())
                return
            if endpoint == "instance-tracking":
                self.send_json(HTTPStatus.OK, serialize_instance_tracking_status())
                return
            if endpoint == "square-pose":
                self.send_json(HTTPStatus.OK, serialize_square_pose_status())
                return
            if endpoint == "instances":
                self.send_json(HTTPStatus.OK, serialize_instance_status())
                return
            if endpoint == "rebuild/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_job()})
                return
            if endpoint == "bboxes/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_bbox_job(), "bboxes": serialize_bbox_status()})
                return
            if endpoint == "maskbits-bboxes/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_maskbits_bbox_job(), "maskbitsBboxes": serialize_maskbits_bbox_status()})
                return
            if endpoint == "bbox-clipping/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_bbox_clipping_job(), "bboxClipping": serialize_bbox_clipping_status()})
                return
            if endpoint == "bbox-contours/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_bbox_contour_job(), "bboxContours": serialize_bbox_contours_status()})
                return
            if endpoint == "contours/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_contour_job(), "contours": serialize_contour_status()})
                return
            if endpoint == "corners/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_corner_job(), "corners": serialize_corner_status()})
                return
            if endpoint == "pose-estimation/build/status":
                payload = serialize_pose_estimation_status()
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_pose_estimation_job(), "poseEstimation": payload, "squarePose": payload})
                return
            if endpoint == "instance-tracking/build/status":
                payload = serialize_instance_tracking_status()
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_instance_tracking_job(), "instanceTracking": payload, "instances": payload})
                return
            if endpoint == "square-pose/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_square_pose_job(), "squarePose": serialize_square_pose_status()})
                return
            if endpoint == "instances/build/status":
                self.send_json(HTTPStatus.OK, {"ok": True, "job": current_instance_job(), "instances": serialize_instance_status()})
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown vision review endpoint")
        except Exception as error:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_PUT(self) -> None:
        endpoint = self.api_endpoint()
        if endpoint is None:
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown API endpoint")
            return
        data = self.read_json_body("{}")
        if data is None:
            return
        try:
            if endpoint == "rules":
                clean = validate_rules(data)
                atomic_write_json(RULES_PATH, clean)
                self.send_json(HTTPStatus.OK, {"ok": True, "updatedAt": clean["updatedAt"]})
                return
            if endpoint == "review":
                clean = validate_review(data)
                atomic_write_json(REVIEW_PATH, clean)
                self.send_json(HTTPStatus.OK, {"ok": True, "updatedAt": clean["updatedAt"]})
                return
            if endpoint == "memory":
                if not isinstance(data, dict):
                    raise ValueError("memory payload must be an object")
                atomic_write_json(MEMORY_PATH, data)
                self.send_json(HTTPStatus.OK, {"ok": True})
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown vision review endpoint")
        except ValueError as error:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_POST(self) -> None:
        global REBUILD_JOB
        pipeline_endpoint = self.pipeline_endpoint()
        if pipeline_endpoint is not None:
            data = self.read_json_body("{}")
            if data is None:
                return
            if not isinstance(data, dict):
                self.send_error_json(HTTPStatus.BAD_REQUEST, "payload must be an object")
                return
            if pipeline_endpoint == "readiness":
                self.send_json(HTTPStatus.OK, playback_readiness_payload(data))
                return
            if pipeline_endpoint == "playback/start":
                result = start_playback_job(data)
                if result.get("conflict"):
                    self.send_json(HTTPStatus.CONFLICT, result)
                    return
                self.send_json(HTTPStatus.ACCEPTED if result.get("ok") else HTTPStatus.BAD_REQUEST, result)
                return
            if pipeline_endpoint == "playback/stop":
                self.send_json(HTTPStatus.OK, stop_playback_job())
                return
            if pipeline_endpoint == "start":
                result = start_pipeline_job(data)
                if result.get("conflict"):
                    self.send_json(HTTPStatus.CONFLICT, result)
                    return
                self.send_json(HTTPStatus.ACCEPTED if result.get("ok") else HTTPStatus.BAD_REQUEST, result)
                return
            if pipeline_endpoint == "stop":
                self.send_json(HTTPStatus.OK, stop_pipeline_job())
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown pipeline endpoint")
            return
        endpoint = self.api_endpoint()
        data = self.read_json_body("{}")
        if data is None:
            return
        if not isinstance(data, dict):
            self.send_error_json(HTTPStatus.BAD_REQUEST, "payload must be an object")
            return
        if endpoint == "bbox-contours/frame":
            try:
                self.send_json(HTTPStatus.OK, serialize_bbox_contour_frame(data))
            except ValueError as error:
                self.send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:
                self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        if endpoint == "bboxes/build":
            result = start_bbox_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "BBox build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "maskbits-bboxes/build":
            result = start_maskbits_bbox_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Maskbits bbox build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "bbox-clipping/build":
            result = start_bbox_clipping_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "BBox clipping build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "bbox-contours/build":
            result = start_bbox_contour_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "BBox contour build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "contours/build":
            result = start_contour_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Contour build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "corners/build":
            result = start_corner_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Corner build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "pose-estimation/build":
            result = start_pose_estimation_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Pose estimation build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "instance-tracking/build":
            result = start_instance_tracking_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Instance tracking build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "square-pose/build":
            result = start_square_pose_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Square pose build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint == "instances/build":
            result = start_instance_build_job(data)
            if result["conflict"]:
                self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Instance provenance build already running", "job": result["job"]})
                return
            self.send_json(HTTPStatus.ACCEPTED, result["job"])
            return
        if endpoint != "rebuild":
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown vision review endpoint")
            return
        existing = active_job()
        if existing:
            self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "rebuild already running", "job": existing})
            return
        now = utc_now()
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            REBUILD_JOB = {
                "ok": True,
                "jobId": job_id,
                "status": "queued",
                "createdAt": now,
                "updatedAt": now,
                "settings": data,
                "progress": {"phase": "queued", "index": 0, "total": None},
                "result": None,
                "error": None,
                "traceback": None,
            }
            payload = json.loads(json.dumps(REBUILD_JOB))
        thread = threading.Thread(target=run_rebuild_job, args=(job_id, data), daemon=True)
        thread.start()
        self.send_json(HTTPStatus.ACCEPTED, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    return parser.parse_args()


def serve(host: str = "127.0.0.1", port: int = 8772) -> None:
    server = ThreadingHTTPServer((host, port), VisionReviewHandler)
    app_path = APP_DIR.relative_to(REPO_ROOT).as_posix()
    print(f"Serving 0721 vision review UI at http://{host}:{port}/{app_path}/index.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
