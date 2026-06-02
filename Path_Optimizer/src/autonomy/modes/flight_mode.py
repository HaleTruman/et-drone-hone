"""Flight intent selection for the control stack."""

from __future__ import annotations

from enum import Enum


class FlightMode(str, Enum):
    """High-level flight behaviors that controllers can support."""

    HOVER = "HOVER"
    PATH_FOLLOWING = "PATH_FOLLOWING"


class FlightModeManager:
    """Track the requested high-level flight behavior.

    Flight modes are intentionally independent of system lifecycle modes. For
    example, the system can be ARMED while the selected flight mode is HOVER.
    """

    def __init__(self, initial_mode: FlightMode = FlightMode.HOVER):
        self.flight_mode = initial_mode

    def set_mode(self, mode: FlightMode) -> FlightMode:
        self.flight_mode = mode
        return self.flight_mode

