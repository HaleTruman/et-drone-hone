"""Raw output byte contract for downstream inference consumers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import torch


OUTPUT_MAGIC = b"GNGO"
OUTPUT_VERSION = 1
PAYLOAD_KIND_RAW_FLOAT32_LOGITS = 1
OUTPUT_HEADER = struct.Struct("<4sHHIQHHHHI")
HEATMAP_WIDTH = 160
HEATMAP_HEIGHT = 90
MASK_CHANNELS = 2
DEPTH_CHANNELS = 1


@dataclass(frozen=True)
class OutputHeader:
    magic: bytes
    version: int
    payload_kind: int
    frame_id: int
    sim_time_ns: int
    heatmap_width: int
    heatmap_height: int
    mask_channels: int
    depth_channels: int
    payload_size: int


def _tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def serialize_raw_logits(frame_id: int, sim_time_ns: int, mask_logits: torch.Tensor, depth_logits: torch.Tensor) -> bytes:
    if tuple(mask_logits.shape) != (MASK_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH):
        raise ValueError(f"Expected mask logits shape {(MASK_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH)}, got {tuple(mask_logits.shape)}.")
    if tuple(depth_logits.shape) != (DEPTH_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH):
        raise ValueError(
            f"Expected depth logits shape {(DEPTH_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH)}, got {tuple(depth_logits.shape)}."
        )
    payload = _tensor_payload_bytes(mask_logits) + _tensor_payload_bytes(depth_logits)
    header = OUTPUT_HEADER.pack(
        OUTPUT_MAGIC,
        OUTPUT_VERSION,
        PAYLOAD_KIND_RAW_FLOAT32_LOGITS,
        int(frame_id),
        int(sim_time_ns),
        HEATMAP_WIDTH,
        HEATMAP_HEIGHT,
        MASK_CHANNELS,
        DEPTH_CHANNELS,
        len(payload),
    )
    return header + payload


def parse_output_header(data: bytes) -> OutputHeader:
    if len(data) < OUTPUT_HEADER.size:
        raise ValueError(f"Output is shorter than {OUTPUT_HEADER.size} byte header.")
    values = OUTPUT_HEADER.unpack_from(data)
    return OutputHeader(*values)


def write_raw_logits(
    output_dir: Path,
    frame_id: int,
    sim_time_ns: int,
    mask_logits: torch.Tensor,
    depth_logits: torch.Tensor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{int(frame_id):06d}.bin"
    output_path.write_bytes(serialize_raw_logits(frame_id, sim_time_ns, mask_logits, depth_logits))
    return output_path
