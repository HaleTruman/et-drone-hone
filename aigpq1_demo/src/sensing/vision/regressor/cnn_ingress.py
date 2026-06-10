"""In-memory CNN raw-logit frame container for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RawLogitsFrame:
    frame_id: int
    sim_time_ns: int
    mask_logits: torch.Tensor
    depth_logits: torch.Tensor
    source_path: Path | None = None


__all__ = ["RawLogitsFrame"]
