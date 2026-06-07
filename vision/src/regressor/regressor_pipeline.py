"""Directory-level raw-logit to Surveyer JSON pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .cnn_ingress import iter_raw_logit_files, read_raw_logits_file
from .logit_inference import DEFAULT_REGRESSOR_CHECKPOINT, LogitRegressor
from .regression_output import append_jsonl, write_frame_json


@dataclass(frozen=True)
class RegressorPipelineConfig:
    input_dir: Path
    output_dir: Path = Path("vision/output/regressor_json")
    checkpoint: Path = DEFAULT_REGRESSOR_CHECKPOINT
    gate_threshold: float = 0.50
    confidence_threshold: float = 0.50
    min_component_area: int = 3
    max_candidates: int = 32
    max_frames: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class RegressorPipelineStats:
    frames_processed: int
    gates_emitted: int
    elapsed_seconds: float
    output_dir: Path
    jsonl_path: Path


def run_regressor_pipeline(config: RegressorPipelineConfig) -> RegressorPipelineStats:
    input_files = iter_raw_logit_files(config.input_dir)
    if int(config.max_frames) > 0:
        input_files = input_files[: int(config.max_frames)]
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "regressor_frames.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    runner = LogitRegressor(
        Path(config.checkpoint),
        device=config.device,
        gate_threshold=float(config.gate_threshold),
        confidence_threshold=float(config.confidence_threshold),
        min_component_area=int(config.min_component_area),
        max_candidates=int(config.max_candidates),
    )
    started = time.perf_counter()
    gates_emitted = 0
    for input_file in input_files:
        raw_frame = read_raw_logits_file(input_file)
        frame = runner.run_frame(raw_frame)
        _output_path, payload = write_frame_json(output_dir, frame)
        append_jsonl(jsonl_path, payload)
        gates_emitted += len(frame.gates)

    return RegressorPipelineStats(
        frames_processed=len(input_files),
        gates_emitted=int(gates_emitted),
        elapsed_seconds=time.perf_counter() - started,
        output_dir=output_dir,
        jsonl_path=jsonl_path,
    )


__all__ = ["RegressorPipelineConfig", "RegressorPipelineStats", "run_regressor_pipeline"]
