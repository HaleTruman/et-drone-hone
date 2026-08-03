"""Minimal logits-to-gate-pose regressor runtime for the demo."""

from .logit_inference import DEFAULT_REGRESSOR_CHECKPOINT, LogitRegressor

__all__ = [
    "DEFAULT_REGRESSOR_CHECKPOINT",
    "LogitRegressor",
]
