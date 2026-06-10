"""Race intent modes."""

from dataclasses import dataclass
from enum import Enum

from .system_mode import SystemMode


class RaceMode(str, Enum):
    HOLD = "HOLD"
    RACE = "RACE"


class RaceModeManager:
    def __init__(self, initial_mode: RaceMode = RaceMode.HOLD):
        self.race_mode = initial_mode

    def set_mode(self, mode: RaceMode) -> RaceMode:
        self.race_mode = mode
        return self.race_mode


@dataclass(frozen=True)
class ModeState:
    system: SystemMode
    race: RaceMode = RaceMode.HOLD
