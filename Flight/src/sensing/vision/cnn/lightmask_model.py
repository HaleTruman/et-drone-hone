"""Standalone runtime model for the gate/non-gate lightmask CNN."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
OUTPUT_STRIDE = 4
HEATMAP_WIDTH = IMAGE_WIDTH // OUTPUT_STRIDE
HEATMAP_HEIGHT = IMAGE_HEIGHT // OUTPUT_STRIDE

SCHEMA_VERSION = "gate_non_gate_lightmask_v1"
MASK_CHANNELS: tuple[str, ...] = ("gate_mask", "obstacle_mask")
DEPTH_CHANNELS: tuple[str, ...] = ("z_layer",)
BACKBONES: tuple[str, ...] = ("mobilenet_v3_small",)
WEIGHTS: tuple[str, ...] = ("none", "default")


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class MobileNetV3SmallLightMask(nn.Module):
    """Tiny stride-4 mask/depth model with MobileNetV3-Small features."""

    def __init__(
        self,
        output_channels: int = len(MASK_CHANNELS),
        weights: str = "none",
        decoder_channels: int = 32,
        decoder_blocks: int = 2,
        dropout: float = 0.0,
        enable_depth_head: bool = False,
    ) -> None:
        super().__init__()
        if weights not in WEIGHTS:
            raise ValueError(f"Unsupported weights option: {weights}")
        weight_enum = MobileNet_V3_Small_Weights.DEFAULT if weights == "default" else None
        backbone = mobilenet_v3_small(weights=weight_enum).features
        self.low_features = nn.Sequential(*list(backbone.children())[:2])
        self.deep_features = nn.Sequential(*list(backbone.children())[2:9])
        channels = max(8, int(decoder_channels))
        self.low_projection = ConvNormAct(16, channels)
        self.deep_projection = ConvNormAct(48, channels)
        self.decoder = nn.Sequential(*[DepthwiseSeparableConv(channels) for _ in range(max(1, int(decoder_blocks)))])
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.mask_head = nn.Conv2d(channels, int(output_channels), kernel_size=1)
        self.enable_depth_head = bool(enable_depth_head)
        if self.enable_depth_head:
            self.depth_head = nn.Conv2d(channels, len(DEPTH_CHANNELS), kernel_size=1)

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.low_features.parameters()
        yield from self.deep_features.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.low_projection.parameters()
        yield from self.deep_projection.parameters()
        yield from self.decoder.parameters()
        yield from self.mask_head.parameters()
        if self.enable_depth_head:
            yield from self.depth_head.parameters()

    def freeze_backbone_batch_norm(self) -> None:
        for module in (*self.low_features.modules(), *self.deep_features.modules()):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        low = self.low_features(images)
        deep = self.deep_features(low)
        low_features = self.low_projection(low)
        deep_features = torch.nn.functional.interpolate(
            self.deep_projection(deep),
            size=low_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder(low_features + deep_features)
        decoded = self.dropout(decoded)
        mask_logits = self.mask_head(decoded)
        if mask_logits.shape[-2:] != (HEATMAP_HEIGHT, HEATMAP_WIDTH):
            mask_logits = torch.nn.functional.interpolate(
                mask_logits,
                size=(HEATMAP_HEIGHT, HEATMAP_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
        outputs = {"mask_logits": mask_logits}
        if self.enable_depth_head:
            depth_logits = self.depth_head(decoded)
            if depth_logits.shape[-2:] != (HEATMAP_HEIGHT, HEATMAP_WIDTH):
                depth_logits = torch.nn.functional.interpolate(
                    depth_logits,
                    size=(HEATMAP_HEIGHT, HEATMAP_WIDTH),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs["depth_logits"] = depth_logits
        return outputs


def build_model(
    backbone: str = "mobilenet_v3_small",
    output_channels: int = len(MASK_CHANNELS),
    weights: str = "none",
    decoder_channels: int = 32,
    decoder_blocks: int = 2,
    dropout: float = 0.0,
    enable_depth_head: bool = False,
) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return MobileNetV3SmallLightMask(
        output_channels=output_channels,
        weights=weights,
        decoder_channels=decoder_channels,
        decoder_blocks=decoder_blocks,
        dropout=dropout,
        enable_depth_head=enable_depth_head,
    )


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, Any]) -> None:
    model.load_state_dict(state_dict)


__all__ = [
    "BACKBONES",
    "DEPTH_CHANNELS",
    "HEATMAP_HEIGHT",
    "HEATMAP_WIDTH",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "MASK_CHANNELS",
    "OUTPUT_STRIDE",
    "SCHEMA_VERSION",
    "WEIGHTS",
    "ConvNormAct",
    "DepthwiseSeparableConv",
    "MobileNetV3SmallLightMask",
    "build_model",
    "load_compatible_state_dict",
]
