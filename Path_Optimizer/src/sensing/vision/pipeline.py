from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sensing.vision.cnn.cnn_pipeline import (
    DEFAULT_CHECKPOINT as DEFAULT_CNN_CHECKPOINT,
    PipelineConfig as CnnPipelineConfig,
    PipelineStats as CnnPipelineStats,
    run_pipeline as run_cnn_pipeline,
)
from sensing.vision.io.udp_protocol import DEFAULT_HOST, DEFAULT_PORT
from sensing.vision.landmarker.landmarker_pipeline import (
    LandmarkerPipelineConfig,
    LandmarkerPipelineStats,
    run_landmarker_pipeline,
)
from sensing.vision.regressor.regressor_pipeline import (
    RegressorPipelineConfig,
    RegressorPipelineStats,
    run_regressor_pipeline,
)
from sensing.vision.regressor.logit_inference import DEFAULT_REGRESSOR_CHECKPOINT


DEFAULT_OUTPUT_ROOT = Path("logs/vision")


@dataclass(slots=True)
class VisionPipelineConfig:
    checkpoint: Path = DEFAULT_CNN_CHECKPOINT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    bind_host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_s: float | None = None
    max_frames: int = 0
    device: str = "auto"
    run_regressor: bool = False
    run_landmarker: bool = False
    regressor_checkpoint: Path = DEFAULT_REGRESSOR_CHECKPOINT
    top_k: int = 5
    ready_event: Any | None = None
    review_sink: Any | None = None


@dataclass(slots=True)
class VisionPipelineStats:
    output_root: Path
    cnn: CnnPipelineStats
    regressor: RegressorPipelineStats | None = None
    landmarker: LandmarkerPipelineStats | None = None


def run_regressor_stage(config: RegressorPipelineConfig) -> RegressorPipelineStats:
    return run_regressor_pipeline(config)


def run_landmarker_stage(config: LandmarkerPipelineConfig) -> LandmarkerPipelineStats:
    return run_landmarker_pipeline(config)


def run_live_pipeline(config: VisionPipelineConfig) -> VisionPipelineStats:
    output_root = Path(config.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    review_sink = config.review_sink
    start_review = getattr(review_sink, "start_run", None) if review_sink is not None else None
    if callable(start_review):
        start_review(
            {
                "run_mode": "live",
                "bind_host": config.bind_host,
                "port": config.port,
                "output_root": str(output_root),
                "run_regressor": config.run_regressor,
                "run_landmarker": config.run_landmarker,
            }
        )

    logits_dir = output_root / "lightmask_logits"
    cnn_stats = run_cnn_pipeline(
        CnnPipelineConfig(
            checkpoint=config.checkpoint,
            output_dir=logits_dir,
            bind_host=config.bind_host,
            port=config.port,
            timeout_s=config.timeout_s,
            max_frames=config.max_frames,
            device=config.device,
            ready_event=config.ready_event,
            frame_callback=review_sink,
        )
    )

    regressor_stats: RegressorPipelineStats | None = None
    landmarker_stats: LandmarkerPipelineStats | None = None
    should_run_regressor = config.run_regressor or config.run_landmarker

    if should_run_regressor:
        regressor_dir = output_root / "regressor_json"
        regressor_stats = run_regressor_stage(
            RegressorPipelineConfig(
                input_dir=logits_dir,
                output_dir=regressor_dir,
                checkpoint=config.regressor_checkpoint,
                device=config.device,
                max_frames=config.max_frames,
            )
        )

    if config.run_landmarker:
        regressor_dir = output_root / "regressor_json"
        landmarker_stats = run_landmarker_stage(
            LandmarkerPipelineConfig(
                input_jsonl=regressor_dir / "regressor_frames.jsonl",
                input_dir=regressor_dir,
                output_dir=output_root / "landmarker_controller_json",
                state_path=output_root / "landmarker_state.json",
                top_k=config.top_k,
                max_frames=config.max_frames,
            )
        )
        record_landmarker = getattr(review_sink, "record_landmarker_outputs", None) if review_sink is not None else None
        if callable(record_landmarker):
            record_landmarker(controller_jsonl=landmarker_stats.jsonl_path)

    stats = VisionPipelineStats(
        output_root=output_root.resolve(),
        cnn=cnn_stats,
        regressor=regressor_stats,
        landmarker=landmarker_stats,
    )
    finish_review = getattr(review_sink, "finish_run", None) if review_sink is not None else None
    if callable(finish_review):
        finish_review(stats)
    return stats
