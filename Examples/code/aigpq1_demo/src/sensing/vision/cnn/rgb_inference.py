"""One-frame lightmask inference over already-normalized RGB tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from Examples.aigpq1_demo.src.sensing.vision.cnn.lightmask_model import MASK_CHANNELS, SCHEMA_VERSION, build_model, load_compatible_state_dict


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "cnn_last.pt"


@dataclass(frozen=True)
class InferenceResult:
    mask_logits: torch.Tensor
    depth_logits: torch.Tensor


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LightmaskInference:
    def __init__(self, checkpoint_path: Path, device: str | torch.device = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.device = select_device(str(device)) if not isinstance(device, torch.device) else device
        self.model, self.config = self._load_model(self.checkpoint_path)

    def _load_model(self, checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema: {config.get('schema_version')} expected {SCHEMA_VERSION}."
            )
        if not bool(config.get("depth_head_enabled", False)):
            raise ValueError("Checkpoint does not have the required depth head enabled.")

        model = build_model(
            backbone=str(config.get("backbone", "mobilenet_v3_small")),
            output_channels=len(MASK_CHANNELS),
            weights="none",
            decoder_channels=int(config.get("decoder_channels", 32)),
            decoder_blocks=int(config.get("decoder_blocks", 2)),
            dropout=float(config.get("dropout", 0.0)),
            enable_depth_head=True,
        )
        load_compatible_state_dict(model, checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, config

    def run_frame(self, frame_tensor: torch.Tensor) -> InferenceResult:
        if frame_tensor.ndim != 4 or frame_tensor.shape[0] != 1 or frame_tensor.shape[1] != 3:
            raise ValueError(f"Expected normalized tensor shape [1,3,H,W], got {tuple(frame_tensor.shape)}.")
        with torch.no_grad():
            output = self.model(frame_tensor.to(self.device))
        if "depth_logits" not in output:
            raise ValueError("Model output is missing depth_logits.")
        return InferenceResult(
            mask_logits=output["mask_logits"].detach().cpu()[0].to(dtype=torch.float32).contiguous(),
            depth_logits=output["depth_logits"].detach().cpu()[0].to(dtype=torch.float32).contiguous(),
        )
