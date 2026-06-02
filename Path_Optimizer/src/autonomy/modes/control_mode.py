"""Low-level control mode selection."""

from __future__ import annotations

from enum import Enum


class ControlMode(str, Enum):
    """Control interfaces that can be implemented by the control stack."""

    ATTITUDE = "ATTITUDE"
    RATE = "RATE"


class ControlModeManager:
    """Track which low-level control interface should receive commands."""

    def __init__(self, initial_mode: ControlMode = ControlMode.ATTITUDE):
        self.control_mode = initial_mode

    def set_mode(self, mode: ControlMode) -> ControlMode:
        self.control_mode = mode
        return self.control_mode

