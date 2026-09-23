"""Ingress helpers for CNN raw-logit egress files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aigp_vision.models.cnn.logits_output import (
    DEPTH_CHANNELS,
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    MASK_CHANNELS,
    OUTPUT_HEADER,
    OUTPUT_MAGIC,
    OUTPUT_VERSION,
    PAYLOAD_KIND_RAW_FLOAT32_LOGITS,
    parse_output_header,
)


@dataclass(frozen=True)
class RawLogitsFrame:
    frame_id: int
    sim_time_ns: int
    mask_logits: torch.Tensor
    depth_logits: torch.Tensor
    source_path: Path | None = None


def parse_raw_logits(data: bytes, *, source_path: Path | None = None) -> RawLogitsFrame:
    header = parse_output_header(data)
    if header.magic != OUTPUT_MAGIC:
        raise ValueError(f"Unsupported logits magic {header.magic!r}; expected {OUTPUT_MAGIC!r}.")
    if header.version != OUTPUT_VERSION:
        raise ValueError(f"Unsupported logits version {header.version}; expected {OUTPUT_VERSION}.")
    if header.payload_kind != PAYLOAD_KIND_RAW_FLOAT32_LOGITS:
        raise ValueError(f"Unsupported logits payload kind {header.payload_kind}.")
    expected_payload_size = (MASK_CHANNELS + DEPTH_CHANNELS) * HEATMAP_HEIGHT * HEATMAP_WIDTH * 4
    if header.payload_size != expected_payload_size:
        raise ValueError(f"Unexpected logits payload size {header.payload_size}; expected {expected_payload_size}.")
    if header.heatmap_width != HEATMAP_WIDTH or header.heatmap_height != HEATMAP_HEIGHT:
        raise ValueError(
            "Unexpected logits heatmap shape "
            f"{header.heatmap_width}x{header.heatmap_height}; expected {HEATMAP_WIDTH}x{HEATMAP_HEIGHT}."
        )
    if header.mask_channels != MASK_CHANNELS or header.depth_channels != DEPTH_CHANNELS:
        raise ValueError(
            "Unexpected logits channel counts "
            f"mask={header.mask_channels} depth={header.depth_channels}; expected mask={MASK_CHANNELS} depth={DEPTH_CHANNELS}."
        )
    expected_total_size = OUTPUT_HEADER.size + expected_payload_size
    if len(data) != expected_total_size:
        raise ValueError(f"Unexpected logits byte length {len(data)}; expected {expected_total_size}.")

    values = np.frombuffer(data, dtype="<f4", offset=OUTPUT_HEADER.size).copy()
    mask_value_count = MASK_CHANNELS * HEATMAP_HEIGHT * HEATMAP_WIDTH
    mask_logits = torch.from_numpy(values[:mask_value_count]).reshape(MASK_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH)
    depth_logits = torch.from_numpy(values[mask_value_count:]).reshape(DEPTH_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH)
    return RawLogitsFrame(
        frame_id=int(header.frame_id),
        sim_time_ns=int(header.sim_time_ns),
        mask_logits=mask_logits.to(dtype=torch.float32).contiguous(),
        depth_logits=depth_logits.to(dtype=torch.float32).contiguous(),
        source_path=source_path,
    )


def read_raw_logits_file(path: Path) -> RawLogitsFrame:
    resolved = Path(path).expanduser().resolve()
    return parse_raw_logits(resolved.read_bytes(), source_path=resolved)


def iter_raw_logit_files(input_dir: Path) -> list[Path]:
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Logits input directory not found: {directory}")
    return sorted(directory.glob("frame_*.bin"))


__all__ = ["RawLogitsFrame", "iter_raw_logit_files", "parse_raw_logits", "read_raw_logits_file"]
