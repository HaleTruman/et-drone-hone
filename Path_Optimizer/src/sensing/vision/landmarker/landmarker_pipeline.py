"""Batch/live-style Regressor JSON to controller landmark output pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .landmark_output import append_jsonl, build_controller_payload, write_controller_frame
from .landmarker import Landmarker, load_landmarker_state, save_landmarker_state
from .regressor_ingress import read_regressor_frames


@dataclass(frozen=True)
class LandmarkerPipelineConfig:
    input_jsonl: Path | None = None
    input_dir: Path | None = None
    output_dir: Path = Path("logs/vision/landmarker_controller_json")
    state_path: Path = Path("logs/vision/landmarker_state.json")
    top_k: int = 5
    max_frames: int = 0


@dataclass(frozen=True)
class LandmarkerPipelineStats:
    frames_processed: int
    final_landmark_count: int
    elapsed_seconds: float
    output_dir: Path
    state_path: Path
    jsonl_path: Path


def run_landmarker_pipeline(config: LandmarkerPipelineConfig) -> LandmarkerPipelineStats:
    frames = read_regressor_frames(
        input_jsonl=config.input_jsonl,
        input_dir=config.input_dir,
        max_frames=int(config.max_frames),
    )
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "controller_frames.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    state_path = Path(config.state_path).expanduser().resolve()
    landmarker = Landmarker(load_landmarker_state(state_path))
    started = time.perf_counter()

    for frame in frames:
        update_result = landmarker.update_frame(frame)
        payload = build_controller_payload(
            landmarker.state,
            run=frame.get("run") or {},
            output_dir=output_dir,
            top_k=int(config.top_k),
            current_gates=update_result["current_gates"],
        )
        write_controller_frame(output_dir, payload)
        append_jsonl(jsonl_path, payload)

    save_landmarker_state(state_path, landmarker.state)
    return LandmarkerPipelineStats(
        frames_processed=len(frames),
        final_landmark_count=len(landmarker.state["landmarks"]["gates"]),
        elapsed_seconds=time.perf_counter() - started,
        output_dir=output_dir,
        state_path=state_path,
        jsonl_path=jsonl_path,
    )


__all__ = ["LandmarkerPipelineConfig", "LandmarkerPipelineStats", "run_landmarker_pipeline"]
