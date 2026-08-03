"""Loss aggregation for torchvision's multi-gate detector."""

from __future__ import annotations

import torch


def gate_pose_loss(
    losses: dict[str, torch.Tensor],
    *_: object,
    **__: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not losses:
        raise ValueError("The detector returned no training losses.")
    total = sum(losses.values())
    return total, {name: value.detach() for name, value in losses.items()}
