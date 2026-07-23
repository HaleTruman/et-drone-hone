#!/usr/bin/env python3
"""Orchestrator for the deterministic vision pipeline."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

try:  # pragma: no cover - exercised by package imports
    from .src.bboxing import BBoxer
    from .src.clipping import ClippingAnalyzer
    from .src.color_masking import ColorMasker, write_mask_binary
    from .src.contouring import Contourer
    from .src.instance_tracking import InstanceTracker
    from .src.pose_estimation import PoseEstimator
    from .src.schema import (
        PipelineFrameResult,
        PipelinePreset,
        RunManifest,
        RunStatus,
        SourceFrame,
        StageDebugEnvelope,
        sha256_file,
        utc_now,
    )
    from .src.vision_results import VisionResultsBuilder
except ImportError:  # pragma: no cover - exercised by script-mode imports
    from src.bboxing import BBoxer
    from src.clipping import ClippingAnalyzer
    from src.color_masking import ColorMasker, write_mask_binary
    from src.contouring import Contourer
    from src.instance_tracking import InstanceTracker
    from src.pose_estimation import PoseEstimator
    from src.schema import (
        PipelineFrameResult,
        PipelinePreset,
        RunManifest,
        RunStatus,
        SourceFrame,
        StageDebugEnvelope,
        sha256_file,
        utc_now,
    )
    from src.vision_results import VisionResultsBuilder


VISION_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET_PATH = VISION_ROOT / "assets" / "pipeline_presets.json"
JPEG_SUFFIXES = {".jpg", ".jpeg"}
STAGES = (
    "color_masking",
    "bboxing",
    "clipping",
    "contouring",
    "pose_estimation",
    "instance_tracking",
    "vision_results",
)


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


def natural_key(path: Path) -> tuple:
    parts: list[object] = []
    current = ""
    numeric = False
    for char in path.name:
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


def jpeg_paths(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    return sorted(
        [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES],
        key=natural_key,
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


@dataclass
class PipelineOptions:
    mode: str
    output_root: Path
    source_dir: Path | None = None
    single_frame: Path | None = None
    run_id: str | None = None
    debug: bool = False
    max_frames: int | None = None
    preset_path: Path = DEFAULT_PRESET_PATH
    poll_interval_s: float = 0.1
    stable_frame_s: float = 0.1


class PipelineRunner:
    def __init__(self, options: PipelineOptions):
        self.options = options
        self.preset_path = options.preset_path.resolve()
        self.preset = PipelinePreset.from_path(self.preset_path)
        self.run_id = options.run_id or utc_run_id()
        self.output_root = options.output_root.resolve()
        self.run_root = self.output_root / self.run_id
        self.frames_dir = self.run_root / "frames"
        self.debug_root = self.run_root / "debug"
        self.status_path = self.run_root / "status.json"
        self.latest_path = self.run_root / "latest.json"
        self.manifest_path = self.run_root / "run_manifest.json"
        self.frame_index: list[dict] = []
        self.timings: dict[str, list[float]] = {stage: [] for stage in STAGES}
        self.started_at = utc_now()

        self.color_masker = ColorMasker(self.preset, VISION_ROOT)
        self.bboxer = BBoxer(self.preset)
        self.clipping = ClippingAnalyzer(self.preset)
        self.contouring = Contourer(self.preset)
        self.pose = PoseEstimator(self.preset)
        self.tracker = InstanceTracker(self.preset)
        self.results_builder = VisionResultsBuilder(self.preset)

    def _status(self, status: str, *, error: str | None = None, completed_at: str | None = None) -> RunStatus:
        errors = [error] if error else []
        latest = self.frame_index[-1]["final_json"] if self.frame_index else None
        return RunStatus(
            schema="vision-run-status.v1",
            run_id=self.run_id,
            status=status,
            run_root=str(self.run_root),
            started_at=self.started_at,
            updated_at=utc_now(),
            frame_count=len(self.frame_index),
            latest_frame=latest,
            completed_at=completed_at,
            errors=errors,
        )

    def _write_status(self, status: str, *, error: str | None = None, completed_at: str | None = None) -> None:
        atomic_write_json(self.status_path, self._status(status, error=error, completed_at=completed_at).to_dict())

    def _manifest(self, completed_at: str | None = None) -> RunManifest:
        timing_summary = {
            stage: {
                "count": len(values),
                "total": round(sum(values), 4),
                "mean": round(sum(values) / len(values), 4) if values else 0.0,
                "max": round(max(values), 4) if values else 0.0,
            }
            for stage, values in self.timings.items()
        }
        return RunManifest(
            schema="vision-run-manifest.v1",
            run_id=self.run_id,
            mode=self.options.mode,
            run_root=str(self.run_root),
            source_dir=str(self.options.source_dir.resolve()) if self.options.source_dir else None,
            single_frame=str(self.options.single_frame.resolve()) if self.options.single_frame else None,
            output_root=str(self.output_root),
            created_at=self.started_at,
            completed_at=completed_at,
            debug=bool(self.options.debug),
            preset_path=str(self.preset_path),
            preset_sha256=sha256_file(self.preset_path),
            preset=self.preset.to_dict(),
            stages=list(STAGES),
            frame_count=len(self.frame_index),
            frame_index=list(self.frame_index),
            timings_ms=timing_summary,
            output_paths={
                "run_manifest": str(self.manifest_path),
                "status": str(self.status_path),
                "latest": str(self.latest_path),
                "frames": str(self.frames_dir),
                "debug": str(self.debug_root) if self.options.debug else "",
            },
        )

    def _write_manifest(self, completed_at: str | None = None) -> None:
        atomic_write_json(self.manifest_path, self._manifest(completed_at=completed_at).to_dict())

    def _timed(self, stage: str, fn: Callable[[], object]) -> tuple[object, float]:
        start = time.perf_counter()
        result = fn()
        elapsed = round((time.perf_counter() - start) * 1000.0, 4)
        self.timings[stage].append(elapsed)
        if hasattr(result, "timing_ms"):
            result.timing_ms[stage] = elapsed
        return result, elapsed

    def _make_source_frame(self, source_path: Path, frame_ordinal: int) -> SourceFrame:
        start = time.perf_counter()
        width, height = _image_size(source_path)
        digest = sha256_file(source_path)
        elapsed = round((time.perf_counter() - start) * 1000.0, 4)
        return SourceFrame(
            run_id=self.run_id,
            frame_ordinal=frame_ordinal,
            frame_id=f"frame_{frame_ordinal:06d}",
            source_path=str(source_path.resolve()),
            image_width=width,
            image_height=height,
            created_at=utc_now(),
            timing_ms={"source": elapsed},
            source_sha256=digest,
        )

    def _write_debug(self, stage: str, frame_ordinal: int, frame_id: str, payload: dict, timing_ms: float, artifacts: dict[str, str] | None = None) -> str:
        debug_path = self.debug_root / stage / "frames" / f"frame_{frame_ordinal:06d}.json"
        envelope = StageDebugEnvelope(
            schema="vision-stage-debug.v1",
            run_id=self.run_id,
            frame_ordinal=frame_ordinal,
            frame_id=frame_id,
            stage=stage,
            created_at=utc_now(),
            timing_ms=timing_ms,
            payload=payload,
            artifacts=artifacts or {},
        )
        atomic_write_json(debug_path, envelope.to_dict())
        return str(debug_path)

    def process_frame(self, source_path: Path, frame_ordinal: int) -> PipelineFrameResult:
        source_path = source_path.resolve()
        source = self._make_source_frame(source_path, frame_ordinal)
        color_frame, color_ms = self._timed("color_masking", lambda: self.color_masker.process(source))
        bbox_frame, bbox_ms = self._timed("bboxing", lambda: self.bboxer.process(color_frame))
        clipping_frame, clipping_ms = self._timed("clipping", lambda: self.clipping.process(bbox_frame, color_frame))
        contour_frame, contour_ms = self._timed("contouring", lambda: self.contouring.process(bbox_frame, color_frame))
        pose_frame, pose_ms = self._timed("pose_estimation", lambda: self.pose.process(contour_frame, clipping_frame))
        instance_frame, instance_ms = self._timed("instance_tracking", lambda: self.tracker.process(bbox_frame, pose_frame))
        vision_results, results_ms = self._timed("vision_results", lambda: self.results_builder.process(instance_frame))

        stage_timings = {
            "source": source.timing_ms["source"],
            "color_masking": color_ms,
            "bboxing": bbox_ms,
            "clipping": clipping_ms,
            "contouring": contour_ms,
            "pose_estimation": pose_ms,
            "instance_tracking": instance_ms,
            "vision_results": results_ms,
        }
        debug_artifacts: dict[str, str] = {}
        if self.options.debug:
            mask_path = self.debug_root / "color_masking" / "masks" / f"frame_{frame_ordinal:06d}.bin"
            write_mask_binary(mask_path, color_frame)
            debug_artifacts["color_masking_mask"] = str(mask_path)
            debug_artifacts["color_masking"] = self._write_debug(
                "color_masking",
                frame_ordinal,
                source.frame_id,
                color_frame.to_dict(),
                color_ms,
                {"mask_bin": str(mask_path)},
            )
            debug_artifacts["bboxing"] = self._write_debug("bboxing", frame_ordinal, source.frame_id, bbox_frame.to_dict(), bbox_ms)
            debug_artifacts["clipping"] = self._write_debug("clipping", frame_ordinal, source.frame_id, clipping_frame.to_dict(), clipping_ms)
            debug_artifacts["contouring"] = self._write_debug("contouring", frame_ordinal, source.frame_id, contour_frame.to_dict(), contour_ms)
            debug_artifacts["pose_estimation"] = self._write_debug("pose_estimation", frame_ordinal, source.frame_id, pose_frame.to_dict(), pose_ms)
            debug_artifacts["instance_tracking"] = self._write_debug("instance_tracking", frame_ordinal, source.frame_id, instance_frame.to_dict(), instance_ms)

        result = PipelineFrameResult(
            run_id=self.run_id,
            frame_ordinal=frame_ordinal,
            frame_id=source.frame_id,
            source_path=source.source_path,
            image_width=source.image_width,
            image_height=source.image_height,
            created_at=utc_now(),
            timing_ms=stage_timings,
            vision_results=vision_results,
            source=source.to_dict(),
            color_masking=color_frame.to_dict(),
            bboxing=bbox_frame.to_dict(),
            clipping=clipping_frame.to_dict(),
            contouring=contour_frame.to_dict(),
            pose_estimation=pose_frame.to_dict(),
            instance_tracking=instance_frame.to_dict(),
            artifacts={"debug": debug_artifacts} if self.options.debug else {},
            stage_timings_ms=stage_timings,
        )
        final_path = self.frames_dir / f"frame_{frame_ordinal:06d}.json"
        atomic_write_json(final_path, result.to_dict())
        atomic_write_json(self.latest_path, result.to_dict())
        self.frame_index.append(
            {
                "frame_ordinal": frame_ordinal,
                "frame_id": source.frame_id,
                "source_path": source.source_path,
                "source_sha256": source.source_sha256,
                "final_json": str(final_path),
                "debug_artifacts": debug_artifacts,
                "stage_timings_ms": stage_timings,
            }
        )
        self._write_status("running")
        self._write_manifest()
        return result

    def _batch_sources(self) -> Iterable[Path]:
        if self.options.single_frame is not None:
            yield self.options.single_frame
            return
        if self.options.source_dir is None:
            raise ValueError("batch mode requires source_dir or single_frame")
        yield from jpeg_paths(self.options.source_dir)

    def run_batch(self) -> RunManifest:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._write_status("running")
        try:
            for ordinal, source_path in enumerate(self._batch_sources()):
                if self.options.max_frames is not None and ordinal >= self.options.max_frames:
                    break
                self.process_frame(source_path, ordinal)
            completed_at = utc_now()
            self._write_status("complete", completed_at=completed_at)
            self._write_manifest(completed_at=completed_at)
            return self._manifest(completed_at=completed_at)
        except Exception as exc:
            self._write_status("failed", error=str(exc))
            self._write_manifest()
            raise

    def run_watch(self) -> RunManifest:
        if self.options.source_dir is None:
            raise ValueError("watch mode requires source_dir")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._write_status("running")
        seen: set[Path] = set()
        frame_ordinal = 0
        try:
            while True:
                for path in jpeg_paths(self.options.source_dir):
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    if time.time() - resolved.stat().st_mtime < self.options.stable_frame_s:
                        continue
                    self.process_frame(resolved, frame_ordinal)
                    seen.add(resolved)
                    frame_ordinal += 1
                    if self.options.max_frames is not None and frame_ordinal >= self.options.max_frames:
                        completed_at = utc_now()
                        self._write_status("complete", completed_at=completed_at)
                        self._write_manifest(completed_at=completed_at)
                        return self._manifest(completed_at=completed_at)
                time.sleep(float(self.options.poll_interval_s))
        except KeyboardInterrupt:
            completed_at = utc_now()
            self._write_status("stopped", completed_at=completed_at)
            self._write_manifest(completed_at=completed_at)
            return self._manifest(completed_at=completed_at)
        except Exception as exc:
            self._write_status("failed", error=str(exc))
            self._write_manifest()
            raise


def run_pipeline(options: PipelineOptions) -> RunManifest:
    runner = PipelineRunner(options)
    if options.mode == "watch":
        return runner.run_watch()
    return runner.run_batch()
