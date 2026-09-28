"""Minimal logits-to-gate-pose regressor runtime."""

from .logit_inference import DEFAULT_REGRESSOR_CHECKPOINT, LogitRegressor

__all__ = [
    "DEFAULT_REGRESSOR_CHECKPOINT",
    "LogitRegressor",
]
