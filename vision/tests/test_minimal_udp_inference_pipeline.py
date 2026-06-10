from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest
import torch

from vision.src.cnn.logits_output import (
    DEPTH_CHANNELS,
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    MASK_CHANNELS,
    OUTPUT_HEADER,
    OUTPUT_MAGIC,
    OUTPUT_VERSION,
    PAYLOAD_KIND_RAW_FLOAT32_LOGITS,
    parse_output_header,
    serialize_raw_logits,
)
from vision.src.cnn.cnn_pipeline import PipelineConfig, run_pipeline
from vision.src.cnn.rgb_inference import DEFAULT_CHECKPOINT
from vision.src.cnn.rgb_normalizer import jpeg_bytes_to_tensor
from vision.src.io.udp_ingress import FrameAccumulator
from vision.src.io.udp_protocol import chunk_jpeg, frame_id_from_path, sorted_jpeg_paths
from vision.tools.udp_spoof.udp_shim import ShimConfig, send_sample_frames


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FRAMES_DIR = (
    REPO_ROOT
    / "vision/tools/sample_runs/universe_75m75g50d_nearest_front_facing_75gates_fullrun_retry720_20260605"
)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_udp_chunks_reassemble_jpeg_bytes() -> None:
    jpeg_bytes = b"\xff\xd8" + (b"abc123" * 500) + b"\xff\xd9"
    packets = chunk_jpeg(frame_id=42, jpeg_bytes=jpeg_bytes, sim_time_ns=123456789, chunk_payload_bytes=257)
    accumulator = FrameAccumulator()
    frame = None
    for packet in reversed(packets):
        frame = accumulator.add_packet(packet) or frame

    assert frame is not None
    assert frame.frame_id == 42
    assert frame.sim_time_ns == 123456789
    assert frame.jpeg_bytes == jpeg_bytes


def test_rgb_normalizer_decodes_sample_jpeg() -> None:
    samples = sorted_jpeg_paths(SAMPLE_FRAMES_DIR)[:1]
    if not samples:
        pytest.skip(f"sample frame missing: {SAMPLE_FRAMES_DIR}")
    sample = samples[0]

    tensor = jpeg_bytes_to_tensor(sample.read_bytes())

    assert tuple(tensor.shape) == (1, 3, 360, 640)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_raw_logit_output_header_and_payload() -> None:
    mask_logits = torch.zeros((MASK_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=torch.float32)
    depth_logits = torch.ones((DEPTH_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=torch.float32)

    data = serialize_raw_logits(frame_id=7, sim_time_ns=99, mask_logits=mask_logits, depth_logits=depth_logits)
    header = parse_output_header(data)

    assert OUTPUT_HEADER.size == 32
    assert header.magic == OUTPUT_MAGIC
    assert header.version == OUTPUT_VERSION
    assert header.payload_kind == PAYLOAD_KIND_RAW_FLOAT32_LOGITS
    assert header.frame_id == 7
    assert header.sim_time_ns == 99
    assert header.heatmap_width == HEATMAP_WIDTH
    assert header.heatmap_height == HEATMAP_HEIGHT
    assert header.mask_channels == MASK_CHANNELS
    assert header.depth_channels == DEPTH_CHANNELS
    assert header.payload_size == (MASK_CHANNELS + DEPTH_CHANNELS) * HEATMAP_HEIGHT * HEATMAP_WIDTH * 4
    assert len(data) == OUTPUT_HEADER.size + header.payload_size


def test_end_to_end_udp_sample_frames_to_raw_logits(tmp_path: Path) -> None:
    samples = sorted_jpeg_paths(SAMPLE_FRAMES_DIR)[:2]
    if len(samples) < 2:
        pytest.skip(f"sample frames missing: {SAMPLE_FRAMES_DIR}")
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint missing: {DEFAULT_CHECKPOINT}")

    port = _free_udp_port()
    ready = threading.Event()
    errors: list[BaseException] = []
    stats_holder = {}

    def _run_pipeline() -> None:
        try:
            stats_holder["stats"] = run_pipeline(
                PipelineConfig(
                    checkpoint=DEFAULT_CHECKPOINT,
                    output_dir=tmp_path,
                    bind_host="127.0.0.1",
                    port=port,
                    timeout_s=10.0,
                    max_frames=2,
                    device="cpu",
                    ready_event=ready,
                )
            )
        except BaseException as exc:  # pragma: no cover - forwarded to main thread
            errors.append(exc)

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    assert ready.wait(timeout=5.0)

    shim_stats = send_sample_frames(
        ShimConfig(
            frames_dir=SAMPLE_FRAMES_DIR,
            host="127.0.0.1",
            port=port,
            fps=0.0,
            max_frames=2,
        )
    )
    thread.join(timeout=30.0)

    assert not thread.is_alive()
    assert not errors
    assert shim_stats.frames_sent == 2
    assert stats_holder["stats"].frames_processed == 2

    output_files = sorted(tmp_path.glob("frame_*.bin"))
    assert len(output_files) == 2
    expected_frame_ids = [frame_id_from_path(path) for path in samples]
    for expected_frame_id, output_file in zip(expected_frame_ids, output_files):
        payload = output_file.read_bytes()
        header = parse_output_header(payload)
        assert header.frame_id == expected_frame_id
        assert header.payload_kind == PAYLOAD_KIND_RAW_FLOAT32_LOGITS
        assert header.payload_size == (MASK_CHANNELS + DEPTH_CHANNELS) * HEATMAP_HEIGHT * HEATMAP_WIDTH * 4
        assert len(payload) == OUTPUT_HEADER.size + header.payload_size
