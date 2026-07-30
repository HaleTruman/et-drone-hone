"""Tiny multi-head regressor model used by extraction runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=int(stride),
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class GatePositionMultiHeadRegressorLite(nn.Module):
    def __init__(self, input_channels: int, scalar_feature_count: int, mask_geometry_feature_count: int = 0) -> None:
        super().__init__()
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(int(input_channels), 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            DepthwiseSeparableBlock(16, 16),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 32, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(int(scalar_feature_count), 32),
            nn.Hardswish(inplace=True),
            nn.Linear(32, 32),
            nn.Hardswish(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.Hardswish(inplace=True),
            nn.Linear(64, 64),
            nn.Hardswish(inplace=True),
        )
        self.center_head = nn.Linear(64, 2)
        self.depth_head = nn.Linear(64, 1)
        self.residual_head = nn.Linear(64, 3)
        self.orientation_crop_encoder = nn.Sequential(
            nn.Conv2d(int(input_channels), 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            DepthwiseSeparableBlock(16, 16),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 32, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.orientation_scalar_encoder = nn.Sequential(
            nn.Linear(max(1, int(mask_geometry_feature_count)), 32),
            nn.Hardswish(inplace=True),
            nn.Linear(32, 32),
            nn.Hardswish(inplace=True),
        )
        self.orientation_shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.Hardswish(inplace=True),
            nn.Linear(64, 32),
            nn.Hardswish(inplace=True),
        )
        self.orientation_axis_head = nn.Linear(32, 3)

    def forward(
        self,
        crops: torch.Tensor,
        scalars: torch.Tensor,
        mask_geometry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        crop_features = self.crop_encoder(crops)
        scalar_features = self.scalar_encoder(scalars)
        shared = self.shared(torch.cat([crop_features, scalar_features], dim=1))
        if mask_geometry is None:
            geometry_width = self.orientation_scalar_encoder[0].in_features
            mask_geometry = torch.zeros((crops.shape[0], geometry_width), dtype=crops.dtype, device=crops.device)
        orientation_crop_features = self.orientation_crop_encoder(crops)
        orientation_scalar_features = self.orientation_scalar_encoder(mask_geometry)
        orientation_shared = self.orientation_shared(torch.cat([orientation_crop_features, orientation_scalar_features], dim=1))
        return {
            "center_norm": self.center_head(shared),
            "z_norm": self.depth_head(shared),
            "residual_norm": self.residual_head(shared),
            "orientation_axis_raw": self.orientation_axis_head(orientation_shared),
        }


def build_model(input_channels: int, scalar_feature_count: int, mask_geometry_feature_count: int = 0) -> nn.Module:
    return GatePositionMultiHeadRegressorLite(
        input_channels=input_channels,
        scalar_feature_count=scalar_feature_count,
        mask_geometry_feature_count=mask_geometry_feature_count,
    )


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, Any]) -> Any:
    return model.load_state_dict(state_dict, strict=True)


__all__ = ["GatePositionMultiHeadRegressorLite", "build_model", "load_compatible_state_dict"]
