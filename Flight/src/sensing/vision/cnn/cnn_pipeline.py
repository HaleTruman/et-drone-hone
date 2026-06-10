"""Minimal live-like UDP ingress to raw lightmask logits pipeline."""

from __future__ import annotations

import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rgb_inference import DEFAULT_CHECKPOINT, LightmaskInference
from .logits_output import write_raw_logits
from .rgb_normalizer import jpeg_bytes_to_tensor
from sensing.vision.io.udp_ingress import IngressConfig, iter_jpeg_frames_from_socket, open_ingress_socket
from sensing.vision.io.udp_protocol import DEFAULT_HOST, DEFAULT_PORT


@dataclass(frozen=True)
class PipelineConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    output_dir: Path = Path("logs/vision/lightmask_logits")
    bind_host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_s: float | None = None
    max_frames: int = 0
    device: str = "auto"
    ready_event: Any | None = None
    frame_callback: Any | None = None


@dataclass(frozen=True)
class PipelineStats:
    frames_processed: int
    elapsed_seconds: float
    output_dir: Path


def run_pipeline(config: PipelineConfig) -> PipelineStats:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    ingress_config = IngressConfig(
        bind_host=config.bind_host,
        port=config.port,
        timeout_s=config.timeout_s,
        max_frames=config.max_frames,
    )

    started = time.perf_counter()
    with closing(open_ingress_socket(ingress_config)) as sock:
        if config.ready_event is not None:
            config.ready_event.set()
        runner = LightmaskInference(Path(config.checkpoint), device=config.device)
        frames_processed = 0
        for frame in iter_jpeg_frames_from_socket(sock, ingress_config):
            tensor = jpeg_bytes_to_tensor(frame.jpeg_bytes)
            result = runner.run_frame(tensor)
            logits_path = write_raw_logits(
                output_dir=output_dir,
                frame_id=frame.frame_id,
                sim_time_ns=frame.sim_time_ns,
                mask_logits=result.mask_logits,
                depth_logits=result.depth_logits,
            )
            if config.frame_callback is not None:
                record = getattr(config.frame_callback, "record_cnn_frame", None)
                if callable(record):
                    record(frame=frame, logits_path=logits_path)
            frames_processed += 1
    return PipelineStats(
        frames_processed=frames_processed,
        elapsed_seconds=time.perf_counter() - started,
        output_dir=output_dir.resolve(),
    )
