#!/usr/bin/env python3
"""Frame-by-frame 0721Vision inference pipeline."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
TOOLS_DIR = APP_DIR / "tools"
SRC_DIR = APP_DIR / "src"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bbox_clipping.build import build_bbox_clipping_frame, clipping_source_signature  # noqa: E402
from bbox_contours.build import bbox_contour_source_signature, build_bbox_contour_frame  # noqa: E402
from build_mask_bboxes import (  # noqa: E402
    bbox_settings_from_review,
    clean_float,
    collect_fov_clip_summary,
    concise_layer_summaries,
    empty_fov_clip_summary,
    normalized_fov_clip_settings,
)
from build_mask_contours import contour_counts_for_objects, normalized_contour_settings  # noqa: E402
from build_vision_memory import stable_hash  # noqa: E402
from color_masks.mask_frames import (  # noqa: E402
    DEFAULT_LUT,
    DEFAULT_MANIFEST_NAME,
    FrameRef,
    build_manifest as build_mask_manifest,
    empty_summary as empty_mask_summary,
    frame_refs_from_manifest,
    layers_from_metadata,
    load_lut,
    merge_summary as merge_mask_summary,
    process_frame as process_mask_frame,
)
from instance_tracking.instance_mapping import (  # noqa: E402
    InstanceTrackingRuntime,
    instance_tracking_settings_from_review,
    instance_tracking_source_signature,
)
from mask_bbox.build_from_maskbits import build_maskbits_bbox_frame, maskbits_bbox_source_signature  # noqa: E402
from mask_bbox.maskbits_input import build_mask_frame_lookup, build_maskbits_context  # noqa: E402
from pose_estimation.pose_fit import CAMERA, build_pose_frame, pose_settings_from_review, pose_source_signature  # noqa: E402


JPEG_SUFFIXES = {".jpg", ".jpeg"}
PIPELINE_SCHEMA = "0721vision-pipeline-run.v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def utc_run_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{uuid.uuid4().hex[:8]}"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def repo_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/" + resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def clean_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or "").strip())[:160]


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(path: Path) -> tuple:
    text = path.name
    parts = []
    current = ""
    numeric = False
    for char in text:
        is_digit = char.isdigit()
        if current and is_digit != numeric:
            parts.append(int(current) if numeric else current.lower())
            current = char
        else:
            current += char
        numeric = is_digit
    if current:
        parts.append(int(current) if numeric else current.lower())
    return tuple(parts)


@dataclass
class PipelineConfig:
    mode: str = "batch"
    source_dir: Path | None = None
    live_point: str | None = None
    run_id: str | None = None
    output_root: Path = APP_DIR / "assets" / "pipeline_runs"
    artifact_root: Path = APP_DIR / "assets"
    config_path: Path = APP_DIR / "assets" / "config" / "alpha_classes.json"
    review_path: Path = APP_DIR / "assets" / "review" / "review_state.json"
    lut_path: Path = DEFAULT_LUT
    poll_interval_s: float = 0.05
    stable_frame_s: float = 0.05
    max_frames: int | None = None
    expected_width: int = 640
    expected_height: int = 360
    parallel_stateless: bool = True
    debug_artifacts: bool = False
    aggregate_debug_manifests: bool = False

    def source_path(self) -> Path | None:
        def normalize(value: str | Path) -> Path:
            text = str(value).strip()
            if text.startswith("/0721Vision/"):
                return REPO_ROOT / text.lstrip("/")
            if text.startswith(("/src/", "/assets/", "/vendor/")):
                return APP_DIR / text.lstrip("/")
            path = Path(text).expanduser()
            if not path.is_absolute():
                path = APP_DIR / path
            return path

        if self.source_dir is not None:
            return normalize(self.source_dir)
        if self.live_point:
            value = str(self.live_point).strip()
            if value.startswith(("http://", "https://")):
                return None
            return normalize(value)
        return None


@dataclass
class SourceCandidate:
    path: Path
    frame_id: str
    source_index: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FramePacket:
    index: int
    frame_id: str
    source_path: Path
    original_path: Path
    sha1: str
    accepted_at: str
    original_metadata: dict = field(default_factory=dict)

    def manifest_entry(self) -> dict:
        stat = self.source_path.stat()
        return {
            "frameOrdinal": self.index,
            "frameId": self.frame_id,
            "index": self.index,
            "path": repo_url(self.source_path),
            "sourcePath": repo_url(self.source_path),
            "originalPath": str(self.original_path),
            "sha1": self.sha1,
            "mtimeNs": int(stat.st_mtime_ns),
            "jpegSize": int(stat.st_size),
            "acceptedAt": self.accepted_at,
        }


class SourceAdapter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.source_path = config.source_path()
        self.seen_versions: set[tuple[str, int, int]] = set()
        self.pending: dict[str, tuple[int, int, float]] = {}

    def iter_candidates(self, stop_event: threading.Event) -> Iterator[SourceCandidate]:
        if str(self.config.live_point or "").startswith(("http://", "https://")):
            raise ValueError("HTTP live points are reserved for a future live adapter; use a server-visible directory for v1")
        source_path = self.source_path
        if source_path is None:
            raise ValueError("source directory or live point is required")
        if self.config.mode == "batch":
            yield from self._iter_batch(source_path)
            return
        yield from self._iter_watch(source_path, stop_event)

    def _iter_batch(self, source_path: Path) -> Iterator[SourceCandidate]:
        if source_path.is_file():
            for ref in frame_refs_from_manifest(source_path, None, REPO_ROOT):
                yield SourceCandidate(path=ref.path, frame_id=str(ref.frame_id), source_index=ref.index)
            return
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"source directory does not exist: {source_path}")
        for index, path in enumerate(sorted((item for item in source_path.iterdir() if item.suffix.lower() in JPEG_SUFFIXES), key=natural_key)):
            yield SourceCandidate(path=path, frame_id=path.stem, source_index=index)

    def _iter_watch(self, source_path: Path, stop_event: threading.Event) -> Iterator[SourceCandidate]:
        if source_path.is_file():
            yield from self._iter_watch_manifest(source_path, stop_event)
            return
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"source directory does not exist: {source_path}")
        while not stop_event.is_set():
            yielded = False
            for path in sorted((item for item in source_path.iterdir() if item.suffix.lower() in JPEG_SUFFIXES), key=natural_key):
                candidate = self._stable_candidate(path)
                if candidate is None:
                    continue
                yielded = True
                yield candidate
            if not yielded:
                stop_event.wait(self.config.poll_interval_s)

    def _iter_watch_manifest(self, manifest_path: Path, stop_event: threading.Event) -> Iterator[SourceCandidate]:
        while not stop_event.is_set():
            yielded = False
            try:
                refs = frame_refs_from_manifest(manifest_path, None, REPO_ROOT)
            except Exception:
                refs = []
            for ref in refs:
                candidate = self._stable_candidate(ref.path, frame_id=str(ref.frame_id), source_index=ref.index)
                if candidate is None:
                    continue
                yielded = True
                yield candidate
            if not yielded:
                stop_event.wait(self.config.poll_interval_s)

    def _stable_candidate(self, path: Path, frame_id: str | None = None, source_index: int | None = None) -> SourceCandidate | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        version = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        if version in self.seen_versions:
            return None
        now = time.monotonic()
        pending_key = str(path.resolve())
        previous = self.pending.get(pending_key)
        if previous != (int(stat.st_size), int(stat.st_mtime_ns), previous[2] if previous else now):
            if previous is None or previous[0] != int(stat.st_size) or previous[1] != int(stat.st_mtime_ns):
                self.pending[pending_key] = (int(stat.st_size), int(stat.st_mtime_ns), now)
                return None
        first_seen = self.pending.get(pending_key, (int(stat.st_size), int(stat.st_mtime_ns), now))[2]
        if now - first_seen < self.config.stable_frame_s:
            return None
        self.pending.pop(pending_key, None)
        self.seen_versions.add(version)
        return SourceCandidate(path=path, frame_id=frame_id or path.stem, source_index=source_index)


class VisionPipeline:
    def __init__(self, config: PipelineConfig):
        mode = str(config.mode or "batch").lower()
        if mode not in {"batch", "watch", "live"}:
            raise ValueError("mode must be batch, watch, or live")
        self.config = config
        self.config.mode = "watch" if mode == "live" else mode
        self.requested_mode = mode
        self.run_id = config.run_id or utc_run_id()
        self.run_root = config.output_root / self.run_id
        self.source_frames_dir = self.run_root / "source_frames"
        self.mask_output_dir = self.run_root / "color_masks"
        self.mask_manifest_path = self.mask_output_dir / DEFAULT_MANIFEST_NAME
        self.precompute_manifest_path = self.run_root / "source_manifest.json"
        self.run_manifest_path = self.run_root / "run_manifest.json"
        self.status_path = self.run_root / "status.json"
        self.active_status_path = config.output_root / "active_status.json"
        self.final_frames_dir = self.run_root / "frames"
        self.latest_frame_path = self.run_root / "latest.json"
        self.debug_root = self.run_root / "debug"
        self.bbox_root = config.artifact_root / "mask_bboxes_maskbits" / self.run_id
        self.clipping_root = config.artifact_root / "bbox_clipping" / self.run_id
        self.contour_root = config.artifact_root / "bbox_contours" / self.run_id
        self.pose_root = config.artifact_root / "pose_estimation" / self.run_id
        self.instance_root = config.artifact_root / "instance_tracking" / self.run_id
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state = "created"
        self.latest_error: str | None = None
        self.created_at = utc_now()
        self.created_perf = time.perf_counter()
        self.prepare_started_at: str | None = None
        self.prepare_started_perf: float | None = None
        self.prepared_at: str | None = None
        self.prepared_perf: float | None = None
        self.started_at: str | None = None
        self.started_perf: float | None = None
        self.updated_at: str | None = None
        self.completed_at: str | None = None
        self.completed_perf: float | None = None
        self.first_frame_accepted_at: str | None = None
        self.first_frame_accepted_perf: float | None = None
        self.first_frame_completed_at: str | None = None
        self.first_frame_completed_perf: float | None = None
        self.first_output_published_at: str | None = None
        self.first_output_published_perf: float | None = None
        self.stage_latency_ms: dict[str, float | None] = {}
        self.frame_latency_ms: list[float] = []
        self.output_bytes_written = 0
        self.latest_frame_output_url: str | None = None
        self.frames: list[dict] = []
        self.mask_entries: list[dict] = []
        self.mask_summary: dict | None = None
        self.bbox_frames: list[dict] = []
        self.clipping_frames: list[dict] = []
        self.contour_frames: list[dict] = []
        self.pose_frames: list[dict] = []
        self.instance_runtime: InstanceTrackingRuntime | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.width: int | None = None
        self.height: int | None = None
        self._prepared = False

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            if self.state in {"created", "starting", "running"}:
                self.state = "stopping"
                self.updated_at = utc_now()

    def run(self) -> dict:
        self._prepare()
        with self.lock:
            self.state = "running"
            self.started_at = utc_now()
            self.started_perf = time.perf_counter()
            self.updated_at = self.started_at
        self._publish_status()
        adapter = SourceAdapter(self.config)
        try:
            for candidate in adapter.iter_candidates(self.stop_event):
                if self.stop_event.is_set():
                    break
                if self.config.max_frames is not None and len(self.frames) >= self.config.max_frames:
                    break
                self._process_candidate(candidate)
            with self.lock:
                self.state = "stopped" if self.stop_event.is_set() else "complete"
                self.completed_at = utc_now()
                self.completed_perf = time.perf_counter()
                self.updated_at = self.completed_at
            if self.config.aggregate_debug_manifests:
                self._publish_all(complete=True)
            else:
                self._publish_run_manifest(complete=True)
                self._publish_status()
        except Exception as error:
            with self.lock:
                self.state = "error"
                self.latest_error = str(error)
                self.updated_at = utc_now()
            self._publish_status()
            raise
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.executor = None
        return self.status_snapshot()

    def status_snapshot(self) -> dict:
        with self.lock:
            frame_count = len(self.frames)
            complete_count = len(self.instance_runtime.frames_out) if self.instance_runtime else 0
            latest_completed = complete_count - 1 if complete_count else None
            latest_accepted = frame_count - 1 if frame_count else None
            elapsed = self._elapsed_timing_snapshot()
            return {
                "schema": PIPELINE_SCHEMA,
                "runId": self.run_id,
                "runKey": self.run_id,
                "mode": self.requested_mode,
                "effectiveMode": self.config.mode,
                "outputMode": self._output_mode(),
                "debugArtifacts": bool(self.config.debug_artifacts),
                "aggregateDebugManifests": bool(self.config.aggregate_debug_manifests),
                "parallelStateless": bool(self.config.parallel_stateless),
                "source": str(self.config.source_path() or self.config.live_point or ""),
                "state": self.state,
                "createdAt": self.created_at,
                "prepareStartedAt": self.prepare_started_at,
                "preparedAt": self.prepared_at,
                "startedAt": self.started_at,
                "updatedAt": self.updated_at,
                "completedAt": self.completed_at,
                "firstFrameAcceptedAt": self.first_frame_accepted_at,
                "firstFrameCompletedAt": self.first_frame_completed_at,
                "firstOutputPublishedAt": self.first_output_published_at,
                "frameCountAccepted": frame_count,
                "frameCountCompleted": complete_count,
                "latestAcceptedFrame": latest_accepted,
                "latestCompletedFrame": latest_completed,
                "behindByFrames": max(0, frame_count - complete_count),
                "stageLatencyMs": dict(self.stage_latency_ms),
                "meanFrameLatencyMs": clean_float(sum(self.frame_latency_ms) / len(self.frame_latency_ms), 3) if self.frame_latency_ms else None,
                "minFrameLatencyMs": clean_float(min(self.frame_latency_ms), 3) if self.frame_latency_ms else None,
                "maxFrameLatencyMs": clean_float(max(self.frame_latency_ms), 3) if self.frame_latency_ms else None,
                "effectiveHz": clean_float(1000.0 / (sum(self.frame_latency_ms) / len(self.frame_latency_ms)), 3) if self.frame_latency_ms else None,
                "timingMs": elapsed,
                "outputBytesWritten": int(self.output_bytes_written),
                "latestError": self.latest_error,
                "manifestUrls": self._manifest_urls(),
                "outputUrls": self._output_urls(),
            }

    def _prepare(self) -> None:
        if self._prepared:
            return
        self.prepare_started_at = utc_now()
        self.prepare_started_perf = time.perf_counter()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.source_frames_dir.mkdir(parents=True, exist_ok=True)
        self.mask_output_dir.mkdir(parents=True, exist_ok=True)
        self.final_frames_dir.mkdir(parents=True, exist_ok=True)
        if self.config.debug_artifacts:
            self.debug_root.mkdir(parents=True, exist_ok=True)
        if self.config.aggregate_debug_manifests:
            self.bbox_root.mkdir(parents=True, exist_ok=True)
            self.clipping_root.mkdir(parents=True, exist_ok=True)
            self.contour_root.mkdir(parents=True, exist_ok=True)
            self.pose_root.mkdir(parents=True, exist_ok=True)
            self.instance_root.mkdir(parents=True, exist_ok=True)
        self.config_data = read_json(self.config.config_path)
        self.review_data = read_json(self.config.review_path) if self.config.review_path.exists() else {"annotations": [], "decisions": []}
        self.lut, self.lut_metadata = load_lut(self.config.lut_path)
        self.layers = layers_from_metadata(self.lut_metadata)
        self.mask_summary = empty_mask_summary(self.layers)
        self.bbox_settings = bbox_settings_from_review(self.review_data)
        self.clipping_settings = normalized_fov_clip_settings((self.bbox_settings.get("fovClip") or {}) if isinstance(self.bbox_settings, dict) else None)
        self.contour_settings = normalized_contour_settings(self.review_data.get("contourHierarchy") if isinstance(self.review_data.get("contourHierarchy"), dict) else None)
        self.pose_settings = pose_settings_from_review(self.review_data)
        self.instance_settings = instance_tracking_settings_from_review(self.review_data)
        if self.instance_settings.get("poseSource") == "squarePose":
            self.instance_settings["poseSource"] = "poseFit"
        self.instance_runtime = InstanceTrackingRuntime(self.instance_settings)
        import cv2  # type: ignore

        self.cv2 = cv2
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="0721vision-stage") if self.config.parallel_stateless else None
        with self.lock:
            self.state = "starting"
            self.prepared_at = utc_now()
            self.prepared_perf = time.perf_counter()
            self.updated_at = utc_now()
        self._prepared = True

    def _process_candidate(self, candidate: SourceCandidate) -> None:
        frame_start = time.perf_counter()
        packet = self._accept_frame(candidate)
        frame_entry = packet.manifest_entry()
        self.frames.append(frame_entry)
        if self.config.debug_artifacts or self.config.aggregate_debug_manifests:
            self._write_precompute_manifest(complete=False)

        mask_entry, mask_stats = self._timed("colorMaskbits", lambda: process_mask_frame(
            FrameRef(index=packet.index, frame_id=packet.frame_id, path=packet.source_path, source_record=frame_entry),
            self.lut,
            self.layers,
            self.mask_output_dir / "masks",
        ))
        self.mask_entries.append(mask_entry)
        merge_mask_summary(self.mask_summary, mask_stats)
        if self.config.debug_artifacts or self.config.aggregate_debug_manifests:
            mask_manifest = self._write_mask_manifest(complete=False)
        else:
            mask_manifest = self._write_mask_manifest_for_entries([mask_entry], self._mask_summary_for_stats(mask_stats), complete=False)
        mask_lookup = build_mask_frame_lookup(mask_manifest)
        mask_context = build_maskbits_context(self.config_data, self.review_data, mask_manifest, self.mask_manifest_path)

        bbox_frame, _ = self._timed("maskBboxes", lambda: build_maskbits_bbox_frame(
            frame_entry,
            packet.index,
            mask_lookup,
            mask_context,
            self.width or self.config.expected_width,
            self.height or self.config.expected_height,
            self.bbox_settings,
        ))
        self.bbox_frames.append(bbox_frame)

        if self.executor is not None:
            clipping_future = self.executor.submit(
                lambda: self._timed("bboxClipping", lambda: build_bbox_clipping_frame(
                    self.cv2,
                    frame_entry,
                    packet.index,
                    bbox_frame,
                    mask_lookup,
                    mask_context,
                    self.width or self.config.expected_width,
                    self.height or self.config.expected_height,
                    self.clipping_settings,
                ))
            )
            contour_future = self.executor.submit(
                lambda: self._timed("bboxContours", lambda: build_bbox_contour_frame(
                    self.cv2,
                    frame_entry,
                    packet.index,
                    bbox_frame,
                    mask_lookup,
                    mask_context,
                    self.width or self.config.expected_width,
                    self.height or self.config.expected_height,
                    self.contour_settings,
                ))
            )
            clipping_frame, _ = clipping_future.result()
            contour_frame = contour_future.result()
        else:
            clipping_frame, _ = self._timed("bboxClipping", lambda: build_bbox_clipping_frame(
                self.cv2,
                frame_entry,
                packet.index,
                bbox_frame,
                mask_lookup,
                mask_context,
                self.width or self.config.expected_width,
                self.height or self.config.expected_height,
                self.clipping_settings,
            ))
            contour_frame = self._timed("bboxContours", lambda: build_bbox_contour_frame(
                self.cv2,
                frame_entry,
                packet.index,
                bbox_frame,
                mask_lookup,
                mask_context,
                self.width or self.config.expected_width,
                self.height or self.config.expected_height,
                self.contour_settings,
            ))
        self.clipping_frames.append(clipping_frame)
        self.contour_frames.append(contour_frame)

        pose_frame, _ = self._timed("poseEstimation", lambda: build_pose_frame(
            self.cv2,
            frame_entry,
            packet.index,
            contour_frame,
            clipping_frame,
            self.pose_settings,
        ))
        self.pose_frames.append(pose_frame)

        instance_frame = self._timed("instanceTracking", lambda: self.instance_runtime.process_frame(bbox_frame, pose_frame))
        frame_latency = (time.perf_counter() - frame_start) * 1000.0
        self.frame_latency_ms.append(frame_latency)
        with self.lock:
            if self.first_frame_completed_at is None:
                self.first_frame_completed_at = utc_now()
                self.first_frame_completed_perf = time.perf_counter()
        self._publish_frame_output(packet, frame_entry, instance_frame, frame_latency)
        if self.config.debug_artifacts:
            self._publish_frame_debug(mask_entry, bbox_frame, clipping_frame, contour_frame, pose_frame, instance_frame)
        if not self.config.debug_artifacts and not self.config.aggregate_debug_manifests:
            self._cleanup_frame_work(packet, mask_entry)
        if self.config.aggregate_debug_manifests:
            self._publish_all(complete=False)
        else:
            self._publish_status()

    def _accept_frame(self, candidate: SourceCandidate) -> FramePacket:
        index = len(self.frames)
        safe_id = clean_component(candidate.frame_id or candidate.path.stem) or f"{index:06d}"
        suffix = candidate.path.suffix.lower() if candidate.path.suffix.lower() in JPEG_SUFFIXES else ".jpg"
        target = self.source_frames_dir / f"frame_{index:06d}_{safe_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.path, target)
        digest = sha1_file(target)
        accepted_at = utc_now()
        with self.lock:
            if self.first_frame_accepted_at is None:
                self.first_frame_accepted_at = accepted_at
                self.first_frame_accepted_perf = time.perf_counter()
        return FramePacket(
            index=index,
            frame_id=str(candidate.frame_id or index),
            source_path=target,
            original_path=candidate.path,
            sha1=digest,
            accepted_at=accepted_at,
            original_metadata=candidate.metadata,
        )

    def _timed(self, stage: str, fn):
        started = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - started) * 1000.0
        with self.lock:
            self.stage_latency_ms[stage] = clean_float(elapsed, 3)
            self.updated_at = utc_now()
        return result

    def _output_mode(self) -> str:
        if self.config.aggregate_debug_manifests:
            return "aggregate-debug"
        if self.config.debug_artifacts:
            return "per-frame-debug"
        return "production"

    def _elapsed_ms(self, start: float | None, end: float | None = None) -> float | None:
        if start is None:
            return None
        return clean_float(((end if end is not None else time.perf_counter()) - start) * 1000.0, 3)

    def _elapsed_timing_snapshot(self) -> dict:
        now = time.perf_counter()
        return {
            "prepareMs": self._elapsed_ms(self.prepare_started_perf, self.prepared_perf),
            "startupToRunningMs": self._elapsed_ms(self.created_perf, self.started_perf),
            "startupToFirstFrameAcceptedMs": self._elapsed_ms(self.created_perf, self.first_frame_accepted_perf),
            "startupToFirstFrameCompletedMs": self._elapsed_ms(self.created_perf, self.first_frame_completed_perf),
            "startupToFirstOutputMs": self._elapsed_ms(self.created_perf, self.first_output_published_perf),
            "runWallMs": self._elapsed_ms(self.started_perf, self.completed_perf or now),
        }

    def _mask_summary_for_stats(self, stats: dict) -> dict:
        summary = empty_mask_summary(self.layers)
        merge_mask_summary(summary, stats)
        return summary

    def _precompute_manifest(self, complete: bool) -> dict:
        width = self.width or self.config.expected_width
        height = self.height or self.config.expected_height
        return {
            "version": 1,
            "app": "vision_passthrough_review",
            "schema": "0721vision-source-frames.v1",
            "kind": "pipeline-source-frames",
            "runKey": self.run_id,
            "runId": self.run_id,
            "builtAt": utc_now(),
            "complete": complete,
            "mode": self.requested_mode,
            "source": str(self.config.source_path() or self.config.live_point or ""),
            "frameCount": len(self.frames),
            "width": width,
            "height": height,
            "frames": self.frames,
        }

    def _write_precompute_manifest(self, complete: bool) -> dict:
        manifest = self._precompute_manifest(complete)
        atomic_write_json(self.precompute_manifest_path, manifest)
        return manifest

    def _write_mask_manifest_for_entries(self, entries: list[dict], summary: dict, complete: bool) -> dict:
        manifest = build_mask_manifest(self.config.lut_path, self.lut_metadata, self.layers, entries, summary)
        manifest.update(
            {
                "app": "vision_passthrough_review",
                "kind": "rgb-lut-maskbits",
                "runKey": self.run_id,
                "runId": self.run_id,
                "complete": complete,
                "sourceManifest": repo_url(self.precompute_manifest_path),
                "source": str(self.config.source_path() or self.config.live_point or ""),
            }
        )
        atomic_write_json(self.mask_manifest_path, manifest)
        if manifest.get("width") and manifest.get("height"):
            self._assert_dimensions(int(manifest["width"]), int(manifest["height"]))
        return manifest

    def _write_mask_manifest(self, complete: bool) -> dict:
        return self._write_mask_manifest_for_entries(
            self.mask_entries,
            self.mask_summary or empty_mask_summary(self.layers),
            complete,
        )

    def _assert_dimensions(self, width: int, height: int) -> None:
        if width != self.config.expected_width or height != self.config.expected_height:
            raise ValueError(f"pipeline v1 expects {self.config.expected_width}x{self.config.expected_height} frames, got {width}x{height}")
        self.width = width
        self.height = height

    def _bbox_manifest(self, complete: bool) -> dict:
        precompute = self._precompute_manifest(complete)
        mask_manifest = build_mask_manifest(self.config.lut_path, self.lut_metadata, self.layers, self.mask_entries, self.mask_summary or empty_mask_summary(self.layers))
        mask_manifest.update({"runKey": self.run_id, "runId": self.run_id, "complete": complete})
        mask_context = build_maskbits_context(self.config_data, self.review_data, mask_manifest, self.mask_manifest_path) if self.mask_manifest_path.exists() else {}
        source_signature = maskbits_bbox_source_signature(precompute, self.precompute_manifest_path, mask_manifest, self.mask_manifest_path, self.bbox_settings, str(mask_context.get("signature") or ""))
        max_bboxes = max((int(frame.get("bboxCount") or 0) for frame in self.bbox_frames), default=0)
        max_bbox_frame_ordinals = []
        for frame in self.bbox_frames:
            is_max = max_bboxes > 0 and int(frame.get("bboxCount") or 0) == max_bboxes
            frame["isMaxBboxCountFrame"] = is_max
            if is_max:
                max_bbox_frame_ordinals.append(int(frame.get("frameOrdinal") or 0))
        fov_clip_summary = empty_fov_clip_summary(bool(self.bbox_settings.get("fovClip", {}).get("enabled")))
        severities = []
        for frame in self.bbox_frames:
            severities.extend(collect_fov_clip_summary(fov_clip_summary, list(frame.get("bboxes") or [])))
        if fov_clip_summary["enabled"]:
            fov_clip_summary["maxSeverity"] = clean_float(max(severities) if severities else 0.0, 4)
            fov_clip_summary["meanSeverity"] = clean_float(sum(severities) / len(severities) if severities else 0.0, 4)
        total_bboxes = sum(int(frame.get("bboxCount") or 0) for frame in self.bbox_frames)
        total_selected = sum(int(frame.get("selectedPixelCount") or 0) for frame in self.bbox_frames)
        summary = {
            "bboxCount": total_bboxes,
            "maxBboxesPerFrame": max_bboxes,
            "observedMaxBboxesPerFrame": max_bboxes,
            "maxBboxFrameCount": len(max_bbox_frame_ordinals),
            "maxBboxFrameOrdinals": max_bbox_frame_ordinals[:100],
            "bboxFrameCap": int(self.bbox_settings["maxBboxesPerFrame"]),
            "selectedPixelCount": total_selected,
            "meanBboxesPerFrame": total_bboxes / len(self.bbox_frames) if self.bbox_frames else 0,
            "meanSelectedPixelsPerFrame": total_selected / len(self.bbox_frames) if self.bbox_frames else 0,
            "quadAcceptedCount": sum(int(frame.get("quadAcceptedCount") or 0) for frame in self.bbox_frames),
            "quadRejectedCount": sum(int(frame.get("quadRejectedCount") or 0) for frame in self.bbox_frames),
            "voidOverlapCount": sum(int(frame.get("voidOverlapCount") or 0) for frame in self.bbox_frames),
            "overlapChildCount": sum(len(frame.get("overlapBboxes") or []) for frame in self.bbox_frames),
            "comparison": {"available": False, "reason": "pipeline-live-output"},
        }
        if fov_clip_summary["enabled"]:
            summary["fovClip"] = fov_clip_summary
        return {
            "version": 1,
            "app": "vision_passthrough_review",
            "kind": "color-mask-pixel-bboxes",
            "createdAt": utc_now(),
            "runKey": self.run_id,
            "precomputeManifest": repo_url(self.precompute_manifest_path),
            "maskManifest": repo_url(self.mask_manifest_path),
            "sourceSignature": source_signature,
            "complete": complete,
            "frameStart": 0,
            "frameCount": len(self.bbox_frames),
            "sourceFrameCount": len(self.frames),
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "settings": self.bbox_settings,
            "input": {
                "source": "rgb-lut-maskbits",
                "maskManifest": repo_url(self.mask_manifest_path),
                "maskSignature": mask_context.get("signature"),
                "maskManifestSha256": mask_context.get("maskManifestSha256"),
                "lutSha256": mask_context.get("lutSha256"),
                "enabledPrefixes": mask_context.get("enabledPrefixes", []),
                "layerSummaries": concise_layer_summaries(mask_context.get("summaries", [])) if mask_context else [],
            },
            **({"fovClipSummary": fov_clip_summary} if fov_clip_summary["enabled"] else {}),
            "comparison": {"available": False, "reason": "pipeline-live-output"},
            "summary": summary,
            "frames": self.bbox_frames,
        }

    def _clipping_manifest(self, complete: bool) -> dict:
        precompute = self._precompute_manifest(complete)
        mask_manifest = read_json(self.mask_manifest_path) if self.mask_manifest_path.exists() else {}
        bbox_manifest = self._bbox_manifest(complete)
        mask_context = build_maskbits_context(self.config_data, self.review_data, mask_manifest, self.mask_manifest_path) if mask_manifest else {}
        source_signature = clipping_source_signature(precompute, self.precompute_manifest_path, mask_manifest, self.mask_manifest_path, bbox_manifest, self.clipping_settings, str(mask_context.get("signature") or ""))
        summary = empty_fov_clip_summary(bool(self.clipping_settings.get("enabled")))
        severities = []
        total_objects = 0
        for frame in self.clipping_frames:
            objects = list(frame.get("objects") or [])
            severities.extend(collect_fov_clip_summary(summary, objects))
            total_objects += len(objects)
        if summary["enabled"]:
            summary["maxSeverity"] = clean_float(max(severities) if severities else 0.0, 4)
            summary["meanSeverity"] = clean_float(sum(severities) / len(severities) if severities else 0.0, 4)
        return {
            "version": 1,
            "app": "vision_passthrough_review",
            "kind": "bbox-local-clipping",
            "createdAt": utc_now(),
            "runKey": self.run_id,
            "precomputeManifest": repo_url(self.precompute_manifest_path),
            "maskManifest": repo_url(self.mask_manifest_path),
            "bboxManifest": repo_url(self.bbox_root / "bbox_manifest.json"),
            "sourceSignature": source_signature,
            "complete": complete,
            "frameStart": 0,
            "frameCount": len(self.clipping_frames),
            "sourceFrameCount": len(self.frames),
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "settings": self.clipping_settings,
            "input": {
                "source": "rgb-lut-maskbits-bboxes",
                "maskSignature": mask_context.get("signature"),
                "bboxSourceSignature": bbox_manifest.get("sourceSignature"),
                "enabledPrefixes": mask_context.get("enabledPrefixes", []),
            },
            "summary": {
                **summary,
                "objectCount": total_objects,
                "meanObjectsPerFrame": total_objects / len(self.clipping_frames) if self.clipping_frames else 0,
            },
            "frames": self.clipping_frames,
        }

    def _contour_manifest(self, complete: bool) -> dict:
        precompute = self._precompute_manifest(complete)
        mask_manifest = read_json(self.mask_manifest_path) if self.mask_manifest_path.exists() else {}
        bbox_manifest = self._bbox_manifest(complete)
        mask_context = build_maskbits_context(self.config_data, self.review_data, mask_manifest, self.mask_manifest_path) if mask_manifest else {}
        source_signature = bbox_contour_source_signature(precompute, self.precompute_manifest_path, mask_manifest, self.mask_manifest_path, bbox_manifest, self.contour_settings, str(mask_context.get("signature") or ""))
        totals = {
            "objectCount": 0,
            "contourCount": 0,
            "outerCount": 0,
            "voidCount": 0,
            "nestedIslandCount": 0,
            "notchCandidateCount": 0,
            "overlapCandidateCount": 0,
        }
        total_selected = 0
        max_voids = 0
        for frame in self.contour_frames:
            counts = contour_counts_for_objects(list(frame.get("objects") or []))
            for key, value in counts.items():
                totals[key] += value
            max_voids = max(max_voids, counts["voidCount"])
            total_selected += int(frame.get("selectedPixelCount") or 0)
        return {
            "version": 1,
            "app": "vision_passthrough_review",
            "kind": "bbox-local-contour-hierarchy",
            "createdAt": utc_now(),
            "runKey": self.run_id,
            "precomputeManifest": repo_url(self.precompute_manifest_path),
            "maskManifest": repo_url(self.mask_manifest_path),
            "bboxManifest": repo_url(self.bbox_root / "bbox_manifest.json"),
            "sourceSignature": source_signature,
            "complete": complete,
            "frameStart": 0,
            "frameCount": len(self.contour_frames),
            "sourceFrameCount": len(self.frames),
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "settings": self.contour_settings,
            "input": {
                "source": self.contour_settings["maskSource"],
                "maskSignature": mask_context.get("signature"),
                "enabledPrefixes": mask_context.get("enabledPrefixes", []),
                "layerSummaries": concise_layer_summaries(mask_context.get("summaries", [])) if mask_context else [],
                "bboxManifest": repo_url(self.bbox_root / "bbox_manifest.json"),
                "bboxSourceSignature": bbox_manifest.get("sourceSignature"),
            },
            "summary": {
                **totals,
                "maxVoidsPerFrame": max_voids,
                "selectedPixelCount": total_selected,
                "meanContoursPerFrame": totals["contourCount"] / len(self.contour_frames) if self.contour_frames else 0,
                "meanSelectedPixelsPerFrame": total_selected / len(self.contour_frames) if self.contour_frames else 0,
            },
            "frames": self.contour_frames,
        }

    def _pose_manifest(self, complete: bool) -> dict:
        precompute = self._precompute_manifest(complete)
        contour_manifest = self._contour_manifest(complete)
        clipping_manifest = self._clipping_manifest(complete)
        source_signature = pose_source_signature(precompute, self.precompute_manifest_path, self.pose_settings, contour_manifest.get("sourceSignature"), clipping_manifest.get("sourceSignature"))
        totals = {
            "objectCount": 0,
            "acceptedCount": 0,
            "rejectedCount": 0,
            "poseCount": 0,
            "usablePoseCount": 0,
            "invalidPoseCount": 0,
            "clippedIgnoredCount": 0,
        }
        errors = []
        depths = []
        for frame in self.pose_frames:
            for key in totals:
                totals[key] += int(frame.get(key) or 0)
            for obj in frame.get("objects") or []:
                pose = obj.get("bestPose") if isinstance(obj.get("bestPose"), dict) else None
                if pose:
                    errors.append(float(pose.get("reprojectionErrorPx") or 0.0))
                    depths.append(float(pose.get("depthM") or 0.0))
        summary = {
            **totals,
            "ambiguousOverlapCount": 0,
            "multiGateEvidenceCount": 0,
            "meanReprojectionErrorPx": clean_float(sum(errors) / len(errors), 4) if errors else None,
            "maxReprojectionErrorPx": clean_float(max(errors), 4) if errors else None,
            "meanDepthM": clean_float(sum(depths) / len(depths), 6) if depths else None,
            "minDepthM": clean_float(min(depths), 6) if depths else None,
            "maxDepthM": clean_float(max(depths), 6) if depths else None,
        }
        return {
            "version": 1,
            "app": "vision_passthrough_review",
            "kind": "3d-pose-fit-from-bbox-contours-v1",
            "createdAt": utc_now(),
            "runKey": self.run_id,
            "precomputeManifest": repo_url(self.precompute_manifest_path),
            "sourceSignature": source_signature,
            "complete": complete,
            "frameStart": 0,
            "frameCount": len(self.pose_frames),
            "sourceFrameCount": len(self.frames),
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "camera": CAMERA,
            "square": {"widthM": clean_float(self.pose_settings["squareSizeM"], 6), "heightM": clean_float(self.pose_settings["squareSizeM"], 6)},
            "settings": self.pose_settings,
            "input": {
                "contourManifest": repo_url(self.contour_root / "contours_manifest.json"),
                "contourSourceSignature": contour_manifest.get("sourceSignature"),
                "clippingManifest": repo_url(self.clipping_root / "clipping_manifest.json"),
                "clippingSourceSignature": clipping_manifest.get("sourceSignature"),
                "cornerAwareAvailable": False,
                "cornerSourceSignature": None,
            },
            "summary": summary,
            "frames": self.pose_frames,
        }

    def _instance_manifest(self, complete: bool) -> dict:
        precompute = self._precompute_manifest(complete)
        bbox_manifest = self._bbox_manifest(complete)
        pose_manifest = self._pose_manifest(complete)
        pose_signature = pose_manifest.get("sourceSignature") if self.instance_settings.get("poseSource") == "poseFit" else None
        source_signature = instance_tracking_source_signature(precompute, self.precompute_manifest_path, self.instance_settings, bbox_manifest.get("sourceSignature"), pose_signature)
        pose_signals_available = bool(pose_signature)
        return {
            "version": 1,
            "app": "vision_passthrough_review",
            "kind": "instance-mapping-pose-estimation-v1",
            "createdAt": utc_now(),
            "runKey": self.run_id,
            "precomputeManifest": repo_url(self.precompute_manifest_path),
            "sourceSignature": source_signature,
            "complete": complete,
            "frameStart": 0,
            "frameCount": len(self.instance_runtime.frames_out) if self.instance_runtime else 0,
            "sourceFrameCount": len(self.frames),
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "settings": self.instance_settings,
            "input": {
                "bboxManifest": repo_url(self.bbox_root / "bbox_manifest.json"),
                "bboxSourceSignature": bbox_manifest.get("sourceSignature"),
                "poseManifest": repo_url(self.pose_root / "3d_pose_fit.json"),
                "poseSourceSignature": pose_manifest.get("sourceSignature"),
                "activePoseSource": self.instance_settings.get("poseSource"),
                "activePoseSourceSignature": pose_signature,
                "poseSignalsAvailable": pose_signals_available,
            },
            "summary": self.instance_runtime.summary(pose_signals_available) if self.instance_runtime else {},
            "instances": self.instance_runtime.instances() if self.instance_runtime else [],
            "frames": self.instance_runtime.frames_out if self.instance_runtime else [],
        }

    def _frame_output_path(self, frame_ordinal: int) -> Path:
        return self.final_frames_dir / f"frame_{int(frame_ordinal):06d}.json"

    def _compact_instances_for_output(self, instance_frame: dict) -> list[dict]:
        observations = []
        for observation in instance_frame.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            item = dict(observation)
            if not self.config.debug_artifacts:
                item.pop("candidates", None)
            observations.append(item)
        return observations

    def _publish_frame_output(self, packet: FramePacket, frame_entry: dict, instance_frame: dict, frame_latency_ms: float) -> None:
        frame_ordinal = int(instance_frame.get("frameOrdinal") if instance_frame.get("frameOrdinal") is not None else packet.index)
        output_path = self._frame_output_path(frame_ordinal)
        instances = self._compact_instances_for_output(instance_frame)
        pose_signals_available = bool(self.instance_settings.get("poseSource") == "poseFit")
        published_at = utc_now()
        payload = {
            "schema": "0721vision-instance-frame.v1",
            "kind": "instance-frame-mapping-v1",
            "runId": self.run_id,
            "runKey": self.run_id,
            "frameOrdinal": frame_ordinal,
            "frameId": instance_frame.get("frameId") or packet.frame_id,
            "publishedAt": published_at,
            "outputPath": repo_url(output_path),
            "latestPath": repo_url(self.latest_frame_path),
            "sourceFrame": {
                "path": repo_url(packet.original_path),
                "workingPath": (frame_entry.get("path") or frame_entry.get("sourcePath")) if self.config.debug_artifacts or self.config.aggregate_debug_manifests else None,
                "originalPath": frame_entry.get("originalPath"),
                "sha1": packet.sha1,
                "acceptedAt": packet.accepted_at,
            },
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "timingMs": {
                "frameTotal": clean_float(frame_latency_ms, 3),
                "stages": dict(self.stage_latency_ms),
            },
            "instanceCount": len(instances),
            "observationCount": int(instance_frame.get("observationCount") or len(instances)),
            "instances": instances,
            "tracker": {
                "summary": self.instance_runtime.summary(pose_signals_available) if self.instance_runtime else {},
                "activeInstanceCount": len(self.instance_runtime.active_tracks) if self.instance_runtime else 0,
            },
        }
        atomic_write_json(output_path, payload)
        atomic_write_json(self.latest_frame_path, payload)
        with self.lock:
            self.latest_frame_output_url = repo_url(output_path)
            self.output_bytes_written += int(output_path.stat().st_size) + int(self.latest_frame_path.stat().st_size)
            if self.first_output_published_at is None:
                self.first_output_published_at = published_at
                self.first_output_published_perf = time.perf_counter()

    def _publish_frame_debug(
        self,
        mask_entry: dict,
        bbox_frame: dict,
        clipping_frame: dict,
        contour_frame: dict,
        pose_frame: dict,
        instance_frame: dict,
    ) -> None:
        frame_ordinal = int(instance_frame.get("frameOrdinal") or bbox_frame.get("frameOrdinal") or 0)
        frame_name = f"frame_{frame_ordinal:06d}.json"
        debug_payloads = {
            "color_masks": {"kind": "debug-color-maskbits-frame", "frame": mask_entry},
            "mask_bboxes": {"kind": "debug-mask-bboxes-frame", "frame": bbox_frame},
            "bbox_clipping": {"kind": "debug-bbox-clipping-frame", "frame": clipping_frame},
            "bbox_contours": {"kind": "debug-bbox-contours-frame", "frame": contour_frame},
            "pose_estimation": {"kind": "debug-pose-estimation-frame", "frame": pose_frame},
            "instance_tracking": {"kind": "debug-instance-tracking-frame", "frame": instance_frame},
        }
        for stage, payload in debug_payloads.items():
            path = self.debug_root / stage / "frames" / frame_name
            payload.update({"schema": "0721vision-debug-frame.v1", "runId": self.run_id, "runKey": self.run_id, "frameOrdinal": frame_ordinal, "createdAt": utc_now()})
            atomic_write_json(path, payload)
            with self.lock:
                self.output_bytes_written += int(path.stat().st_size)

    def _path_from_repo_ref(self, value: str | None) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.startswith("/0721Vision/"):
            return REPO_ROOT / text.lstrip("/")
        if text.startswith(("/assets/", "/src/")):
            return APP_DIR / text.lstrip("/")
        path = Path(text)
        if path.is_absolute():
            return path
        if text.startswith(("assets/", "src/")):
            return APP_DIR / text
        return APP_DIR / path

    def _cleanup_frame_work(self, packet: FramePacket, mask_entry: dict) -> None:
        paths = [
            packet.source_path,
            self._path_from_repo_ref(str(mask_entry.get("maskBits") or mask_entry.get("path") or "")),
            self.mask_manifest_path,
        ]
        for path in paths:
            if not path:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for directory in (self.mask_output_dir / "masks", self.mask_output_dir, self.source_frames_dir):
            try:
                directory.rmdir()
            except OSError:
                pass

    def _publish_run_manifest(self, complete: bool) -> None:
        payload = {
            "schema": PIPELINE_SCHEMA,
            "runId": self.run_id,
            "runKey": self.run_id,
            "generatedAt": utc_now(),
            "complete": complete,
            "outputMode": self._output_mode(),
            "debugArtifacts": bool(self.config.debug_artifacts),
            "aggregateDebugManifests": bool(self.config.aggregate_debug_manifests),
            "source": str(self.config.source_path() or self.config.live_point or ""),
            "frameCount": len(self.frames),
            "frameCountCompleted": len(self.instance_runtime.frames_out) if self.instance_runtime else 0,
            "image": {"width": self.width or self.config.expected_width, "height": self.height or self.config.expected_height},
            "outputUrls": self._output_urls(),
            "timingMs": self._elapsed_timing_snapshot(),
            "summary": {
                "instances": self.instance_runtime.summary(bool(self.instance_settings.get("poseSource") == "poseFit")) if self.instance_runtime else {},
                "meanFrameLatencyMs": clean_float(sum(self.frame_latency_ms) / len(self.frame_latency_ms), 3) if self.frame_latency_ms else None,
                "minFrameLatencyMs": clean_float(min(self.frame_latency_ms), 3) if self.frame_latency_ms else None,
                "maxFrameLatencyMs": clean_float(max(self.frame_latency_ms), 3) if self.frame_latency_ms else None,
            },
        }
        atomic_write_json(self.run_manifest_path, payload)
        with self.lock:
            self.output_bytes_written += int(self.run_manifest_path.stat().st_size)

    def _output_urls(self) -> dict:
        urls = {
            "runManifest": repo_url(self.run_manifest_path),
            "status": repo_url(self.status_path),
            "latestFrame": repo_url(self.latest_frame_path),
            "framesRoot": repo_url(self.final_frames_dir),
        }
        if self.latest_frame_output_url:
            urls["latestFrameOutput"] = self.latest_frame_output_url
        if self.config.debug_artifacts:
            urls["debugRoot"] = repo_url(self.debug_root)
        return urls

    def _manifest_urls(self) -> dict:
        urls = {
            "runManifest": repo_url(self.run_manifest_path),
            "status": repo_url(self.status_path),
        }
        if self.config.debug_artifacts or self.config.aggregate_debug_manifests:
            urls.update(
                {
                    "sourceManifest": repo_url(self.precompute_manifest_path),
                    "maskManifest": repo_url(self.mask_manifest_path),
                }
            )
        if self.config.aggregate_debug_manifests:
            urls.update(
                {
                    "bboxManifest": repo_url(self.bbox_root / "bbox_manifest.json"),
                    "clippingManifest": repo_url(self.clipping_root / "clipping_manifest.json"),
                    "contourManifest": repo_url(self.contour_root / "contours_manifest.json"),
                    "poseManifest": repo_url(self.pose_root / "3d_pose_fit.json"),
                    "instanceManifest": repo_url(self.instance_root / "instance_mapping.json"),
                }
            )
        return urls

    def _publish_all(self, complete: bool) -> None:
        self._write_precompute_manifest(complete)
        if not self.mask_entries:
            self._publish_status()
            return
        mask_manifest = self._write_mask_manifest(complete)
        bbox_manifest = self._bbox_manifest(complete)
        atomic_write_json(self.bbox_root / "bbox_manifest.json", bbox_manifest)
        clipping_manifest = self._clipping_manifest(complete)
        atomic_write_json(self.clipping_root / "clipping_manifest.json", clipping_manifest)
        contour_manifest = self._contour_manifest(complete)
        atomic_write_json(self.contour_root / "contours_manifest.json", contour_manifest)
        pose_manifest = self._pose_manifest(complete)
        atomic_write_json(self.pose_root / "3d_pose_fit.json", pose_manifest)
        instance_manifest = self._instance_manifest(complete)
        atomic_write_json(self.instance_root / "instance_mapping.json", instance_manifest)
        atomic_write_json(
            self.run_manifest_path,
            {
                "schema": PIPELINE_SCHEMA,
                "runId": self.run_id,
                "runKey": self.run_id,
                "generatedAt": utc_now(),
                "complete": complete,
                "sourceManifest": repo_url(self.precompute_manifest_path),
                "maskManifest": repo_url(self.mask_manifest_path),
                "stageManifests": self._manifest_urls(),
                "frameCount": len(self.frames),
                "image": {"width": mask_manifest.get("width"), "height": mask_manifest.get("height")},
                "summary": {
                    "bboxes": bbox_manifest.get("summary", {}),
                    "clipping": clipping_manifest.get("summary", {}),
                    "contours": contour_manifest.get("summary", {}),
                    "pose": pose_manifest.get("summary", {}),
                    "instances": instance_manifest.get("summary", {}),
                },
            },
        )
        self._publish_status()

    def _publish_status(self) -> None:
        payload = self.status_snapshot()
        atomic_write_json(self.status_path, payload)
        atomic_write_json(self.active_status_path, payload)


def latest_pipeline_status(output_root: Path = APP_DIR / "assets" / "pipeline_runs") -> dict:
    path = output_root / "active_status.json"
    if not path.exists():
        return {"schema": PIPELINE_SCHEMA, "state": "idle", "manifestUrls": {}}
    return read_json(path)
