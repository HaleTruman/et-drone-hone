"""Low-level control mode selection."""

from enum import Enum


class ControlMode(str, Enum):
    """Control interfaces that can be implemented by the control stack."""

    RATE_DIRECT = "RATE_DIRECT"
    ATTITUDE = "ATTITUDE"
    POSITION_HOLD = "POSITION_HOLD"
    VELOCITY = "VELOCITY"
    WAYPOINT_FOLLOW = "WAYPOINT_FOLLOW"
    TRAJECTORY_TRACK = "TRAJECTORY_TRACK"
    MPCC_TRACKER = "MPCC_TRACKER"
    VISION_GATE = "VISION_GATE"
    RATE = "RATE_DIRECT"


class ControlModeManager:
    """Track which low-level control interface should receive commands."""

    def __init__(self, initial_mode: ControlMode = ControlMode.ATTITUDE):
        self.control_mode = initial_mode

    def set_mode(self, mode: ControlMode) -> ControlMode:
        self.control_mode = mode
        return self.control_mode
