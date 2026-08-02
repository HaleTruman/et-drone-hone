"""Compatibility exports for the package-owned gate pose regressor."""

from .gate_regression import (
    REGRESSOR_NAME,
    GatePoseRegressor,
    PostPnPGateRegressor,
)

__all__ = (
    "REGRESSOR_NAME",
    "GatePoseRegressor",
    "PostPnPGateRegressor",
)
