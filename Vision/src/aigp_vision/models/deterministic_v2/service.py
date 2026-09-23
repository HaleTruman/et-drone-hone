from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from aigp_vision.schema import VisionGateObservation, VisionObservation
from aigp_vision.models.deterministic_v2.config import DeterministicVisionV2Config
from aigp_vision.models.deterministic_v2.src.pipeline import PipelineOptions, ProjectionPipeline
from aigp_vision.schema import VisionFrame


class DeterministicVisionV2Backend:
    """Flight-facing adapter around the copied projection pipeline."""

    def __init__(self, config: DeterministicVisionV2Config | None = None) -> None:
        self.config = config or DeterministicVisionV2Config()
        self.run_id = f"flight-deterministic-v2-{int(time.time() * 1000)}"
        self.run_root = self.config.scratch_root / self.run_id
        self.source_frames_dir = self.run_root / "source_frames"
        self.output_root = self.run_root / "projection"
        self.frame_count = 0
        self._pipeline: ProjectionPipeline | None = None

    def process_vision_frame(self, frame: VisionFrame) -> VisionObservation:
        return self.process_frame(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            jpeg_bytes=frame.jpeg_bytes,
        )

    def process_frame(self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes) -> VisionObservation:
        self._prepare()
        pipeline = self._require_pipeline()
        frame_ordinal = self.frame_count
        self.frame_count += 1
        source_path = self._write_source_frame(frame_ordinal, frame_id, jpeg_bytes)

        processed = pipeline.process_frame(source_path, frame_ordinal)
        if (
            int(processed.meta.image_width or 0) != int(self.config.expected_width)
            or int(processed.meta.image_height or 0) != int(self.config.expected_height)
        ):
            raise ValueError(
                f"deterministic v2 vision expects {self.config.expected_width}x{self.config.expected_height} frames, "
                f"got {processed.meta.image_width}x{processed.meta.image_height}"
            )

        pipeline.processed_frames.append(processed)
        frame_ordinals = [frame.meta.frame_ordinal for frame in pipeline.processed_frames]
        frame_ids = {frame.meta.frame_ordinal: frame.meta.frame_id for frame in pipeline.processed_frames}
        profile_tracks = pipeline._profile_tracks(pipeline.processed_frames)
        gates_by_frame, global_payload = pipeline.global_mapper.process_run(profile_tracks, frame_ordinals, frame_ids)
        gates = [
            self._gate_from_result(gate)
            for gate in gates_by_frame.get(frame_ordinal, [])
        ]
        trace = {
            "backend": "deterministic_0721_v2",
            "runId": self.run_id,
            "frameOrdinal": frame_ordinal,
            "sourceFrame": {
                "path": str(source_path),
                "sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
            },
            "stageTimingsMs": dict(processed.stage_timings_ms),
            "profileTimingsMs": dict(processed.profile_timings_ms),
            "stageCounts": {
                "profiles": len(processed.profile_payloads),
                "gates": len(gates),
            },
            "globalMapping": global_payload,
        }

        if not self.config.keep_scratch:
            self._cleanup_frame_work(source_path)
        return VisionObservation(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            gates=gates,
            source="deterministic_0721_v2",
            trace=trace,
        )

    def shutdown(self) -> None:
        if not self.config.keep_scratch:
            shutil.rmtree(self.run_root, ignore_errors=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "deterministic_0721_v2",
            "run_id": self.run_id,
            "loaded": self._pipeline is not None,
            "frame_count": self.frame_count,
            "scratch_root": str(self.config.scratch_root),
            "keep_scratch": self.config.keep_scratch,
            "debug": self.config.debug,
        }

    def _prepare(self) -> None:
        if self._pipeline is not None:
            return
        self.source_frames_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._pipeline = ProjectionPipeline(
            PipelineOptions(
                mode="stream",
                output_root=self.output_root,
                source_dir=self.source_frames_dir,
                run_id=self.run_id,
                debug=bool(self.config.debug),
                preset_path=Path(self.config.preset_path),
            )
        )
        self._pipeline.run_root.mkdir(parents=True, exist_ok=True)
        if self.config.debug:
            self._pipeline.debug_root.mkdir(parents=True, exist_ok=True)

    def _require_pipeline(self) -> ProjectionPipeline:
        if self._pipeline is None:
            raise RuntimeError("deterministic v2 pipeline has not been prepared")
        return self._pipeline

    def _write_source_frame(self, frame_ordinal: int, frame_id: int, jpeg_bytes: bytes) -> Path:
        path = self.source_frames_dir / f"frame_{frame_ordinal:06d}_{int(frame_id)}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(jpeg_bytes)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return path

    @staticmethod
    def _gate_from_result(result: Any) -> VisionGateObservation:
        payload = result.to_controller_payload()
        if hasattr(result, "trace"):
            payload["trace"] = getattr(result, "trace")
        return VisionGateObservation.from_payload(payload)

    def _cleanup_frame_work(self, source_path: Path) -> None:
        try:
            source_path.unlink()
        except FileNotFoundError:
            pass
