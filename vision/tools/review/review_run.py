"""End-to-end extractor review run orchestration."""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vision.src.cnn.cnn_pipeline import PipelineConfig, run_pipeline
from vision.src.cnn.rgb_inference import DEFAULT_CHECKPOINT
from vision.src.io.udp_protocol import DEFAULT_CHUNK_PAYLOAD_BYTES, DEFAULT_HOST, sorted_jpeg_paths
from vision.src.landmarker.landmarker_pipeline import LandmarkerPipelineConfig, run_landmarker_pipeline
from vision.src.regressor import DEFAULT_REGRESSOR_CHECKPOINT, RegressorPipelineConfig, run_regressor_pipeline
from vision.tools.review.review_render import ReviewRenderConfig, ReviewRenderStats, render_two_pane_review
from vision.tools.udp_spoof.udp_shim import ShimConfig, ShimStats, send_sample_frames


@dataclass(frozen=True)
class ExtractorReviewRunConfig:
    frames_dir: Path
    output_root: Path
    checkpoint: Path = DEFAULT_CHECKPOINT
    regressor_checkpoint: Path = DEFAULT_REGRESSOR_CHECKPOINT
    top_k: int = 5
    fps: float = 30.0
    retry_fps: float = 10.0
    max_frames: int = 0
    device: str = "auto"
    host: str = DEFAULT_HOST
    chunk_payload_bytes: int = DEFAULT_CHUNK_PAYLOAD_BYTES
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class CnnUdpAttemptStats:
    fps: float
    frames_expected: int
    frames_sent: int
    frames_processed: int
    raw_logit_files: int
    packets_sent: int
    bytes_sent: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ExtractorReviewRunStats:
    frames_expected: int
    cnn_attempts: list[CnnUdpAttemptStats]
    regressor_frames_processed: int
    regressor_gates_emitted: int
    landmarker_frames_processed: int
    landmarker_final_landmarks: int
    review_frames_rendered: int
    elapsed_seconds: float
    output_root: Path
    summary_path: Path
    review_mp4: Path


def _free_udp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((str(host), 0))
        return int(sock.getsockname()[1])


def _reset_generated_outputs(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("lightmask_logits", "regressor_json", "landmarker_controller_json", "review"):
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("landmarker_state.json", "run_summary.json"):
        path = output_root / name
        if path.exists():
            path.unlink()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _run_cnn_udp_attempt(
    *,
    config: ExtractorReviewRunConfig,
    logits_dir: Path,
    frames_expected: int,
    fps: float,
) -> CnnUdpAttemptStats:
    _reset_dir(logits_dir)
    port = _free_udp_port(config.host)
    ready = threading.Event()
    errors: list[BaseException] = []
    stats_holder: dict[str, Any] = {}

    def _run_cnn() -> None:
        try:
            stats_holder["stats"] = run_pipeline(
                PipelineConfig(
                    checkpoint=Path(config.checkpoint),
                    output_dir=logits_dir,
                    bind_host=config.host,
                    port=port,
                    timeout_s=float(config.timeout_seconds),
                    max_frames=int(frames_expected),
                    device=str(config.device),
                    ready_event=ready,
                )
            )
        except BaseException as exc:  # pragma: no cover - forwarded to caller
            errors.append(exc)

    started = time.perf_counter()
    thread = threading.Thread(target=_run_cnn, daemon=True)
    thread.start()
    if not ready.wait(timeout=10.0):
        raise TimeoutError("CNN UDP ingress did not become ready.")

    shim_stats: ShimStats = send_sample_frames(
        ShimConfig(
            frames_dir=Path(config.frames_dir),
            host=config.host,
            port=port,
            fps=float(fps),
            chunk_payload_bytes=int(config.chunk_payload_bytes),
            max_frames=int(frames_expected),
        )
    )
    join_timeout = max(120.0, (float(frames_expected) / max(1.0, float(fps))) * 3.0 + float(config.timeout_seconds) + 60.0)
    thread.join(timeout=join_timeout)
    if thread.is_alive():
        raise TimeoutError(f"CNN UDP ingress did not finish within {join_timeout:.1f}s.")
    if errors:
        raise errors[0]

    pipeline_stats = stats_holder["stats"]
    return CnnUdpAttemptStats(
        fps=float(fps),
        frames_expected=int(frames_expected),
        frames_sent=int(shim_stats.frames_sent),
        frames_processed=int(pipeline_stats.frames_processed),
        raw_logit_files=len(sorted(logits_dir.glob("frame_*.bin"))),
        packets_sent=int(shim_stats.packets_sent),
        bytes_sent=int(shim_stats.bytes_sent),
        elapsed_seconds=time.perf_counter() - started,
    )


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run_extractor_review_run(config: ExtractorReviewRunConfig) -> ExtractorReviewRunStats:
    started = time.perf_counter()
    frames = sorted_jpeg_paths(Path(config.frames_dir).expanduser())
    if int(config.max_frames) > 0:
        frames = frames[: int(config.max_frames)]
    if not frames:
        raise FileNotFoundError(f"No JPEG frames found in {config.frames_dir}")
    frames_expected = len(frames)

    output_root = Path(config.output_root).expanduser().resolve()
    _reset_generated_outputs(output_root)
    logits_dir = output_root / "lightmask_logits"
    regressor_dir = output_root / "regressor_json"
    landmarker_dir = output_root / "landmarker_controller_json"
    review_dir = output_root / "review"
    state_path = output_root / "landmarker_state.json"

    attempts: list[CnnUdpAttemptStats] = []
    fps_values = [float(config.fps)]
    if float(config.retry_fps) > 0.0 and abs(float(config.retry_fps) - float(config.fps)) > 1.0e-6:
        fps_values.append(float(config.retry_fps))

    for fps in fps_values:
        attempt = _run_cnn_udp_attempt(config=config, logits_dir=logits_dir, frames_expected=frames_expected, fps=fps)
        attempts.append(attempt)
        if attempt.frames_processed == frames_expected and attempt.raw_logit_files == frames_expected:
            break
    else:
        raise RuntimeError(
            f"CNN stage did not produce all frames after {len(attempts)} attempt(s): "
            f"expected={frames_expected} last_processed={attempts[-1].frames_processed} "
            f"last_files={attempts[-1].raw_logit_files}."
        )

    regressor_stats = run_regressor_pipeline(
        RegressorPipelineConfig(
            input_dir=logits_dir,
            output_dir=regressor_dir,
            checkpoint=Path(config.regressor_checkpoint),
            device=str(config.device),
        )
    )
    regressor_rows = _count_jsonl_rows(regressor_stats.jsonl_path)
    if regressor_stats.frames_processed != frames_expected or regressor_rows != frames_expected:
        raise RuntimeError(
            f"Regressor stage frame mismatch: expected={frames_expected} "
            f"processed={regressor_stats.frames_processed} jsonl_rows={regressor_rows}."
        )

    landmarker_stats = run_landmarker_pipeline(
        LandmarkerPipelineConfig(
            input_jsonl=regressor_stats.jsonl_path,
            output_dir=landmarker_dir,
            state_path=state_path,
            top_k=int(config.top_k),
        )
    )
    controller_rows = _count_jsonl_rows(landmarker_stats.jsonl_path)
    if landmarker_stats.frames_processed != frames_expected or controller_rows != frames_expected:
        raise RuntimeError(
            f"Landmarker stage frame mismatch: expected={frames_expected} "
            f"processed={landmarker_stats.frames_processed} jsonl_rows={controller_rows}."
        )

    review_mp4 = review_dir / "extractor_regressor_landmarker_two_pane.mp4"
    render_stats: ReviewRenderStats = render_two_pane_review(
        ReviewRenderConfig(
            frames_dir=Path(config.frames_dir),
            regressor_jsonl=regressor_stats.jsonl_path,
            controller_jsonl=landmarker_stats.jsonl_path,
            output_mp4=review_mp4,
            fps=float(config.fps),
            max_frames=int(frames_expected),
        )
    )
    if render_stats.frames_rendered != frames_expected:
        raise RuntimeError(
            f"Review render frame mismatch: expected={frames_expected} rendered={render_stats.frames_rendered}."
        )

    elapsed = time.perf_counter() - started
    summary_path = output_root / "run_summary.json"
    stats = ExtractorReviewRunStats(
        frames_expected=frames_expected,
        cnn_attempts=attempts,
        regressor_frames_processed=int(regressor_stats.frames_processed),
        regressor_gates_emitted=int(regressor_stats.gates_emitted),
        landmarker_frames_processed=int(landmarker_stats.frames_processed),
        landmarker_final_landmarks=int(landmarker_stats.final_landmark_count),
        review_frames_rendered=int(render_stats.frames_rendered),
        elapsed_seconds=elapsed,
        output_root=output_root,
        summary_path=summary_path,
        review_mp4=render_stats.output_mp4,
    )
    _write_summary(
        summary_path,
        {
            "config": {
                "frames_dir": str(Path(config.frames_dir).expanduser().resolve()),
                "output_root": str(output_root),
                "checkpoint": str(Path(config.checkpoint).expanduser().resolve()),
                "regressor_checkpoint": str(Path(config.regressor_checkpoint).expanduser().resolve()),
                "top_k": int(config.top_k),
                "fps": float(config.fps),
                "retry_fps": float(config.retry_fps),
                "max_frames": int(config.max_frames),
                "device": str(config.device),
            },
            "stats": {
                **{
                    key: value
                    for key, value in asdict(stats).items()
                    if key not in {"cnn_attempts", "output_root", "summary_path", "review_mp4"}
                },
                "cnn_attempts": [asdict(attempt) for attempt in attempts],
                "output_root": str(output_root),
                "summary_path": str(summary_path),
                "review_mp4": str(render_stats.output_mp4),
                "regressor_jsonl": str(regressor_stats.jsonl_path),
                "controller_jsonl": str(landmarker_stats.jsonl_path),
                "landmarker_state": str(state_path.resolve()),
            },
        },
    )
    return stats


__all__ = [
    "ExtractorReviewRunConfig",
    "ExtractorReviewRunStats",
    "run_extractor_review_run",
]
