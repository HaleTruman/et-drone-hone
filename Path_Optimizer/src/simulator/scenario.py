"""Small JSON scenario loader for flight-only simulations."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.modes import ControlMode, FlightMode, ModeSelection, SystemMode, validate_modes


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    initial_state: np.ndarray
    initial_modes: ModeSelection
    events: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 7
    environment: dict[str, float] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


def load_scenario(path: str | Path) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario root must be an object")
    duration_s = float(data.get("duration_s", 0.0))
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    events = list(data.get("events", []))
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("events must contain objects")
    times = [float(event["at_s"]) for event in events]
    if times != sorted(times):
        raise ValueError("events must be ordered by at_s")
    if times and (times[0] < 0.0 or times[-1] > duration_s):
        raise ValueError("event timestamps must stay within scenario duration")
    modes = _modes(data.get("initial_modes", {"system": "IDLE"}))
    validate_modes(modes, flight_only=True)
    return Scenario(
        name=str(data.get("name", Path(path).stem)),
        duration_s=duration_s,
        initial_state=_state(data.get("initial_state", {})),
        initial_modes=modes,
        events=events,
        seed=int(data.get("seed", 7)),
        environment={key: float(value) for key, value in data.get("environment", {}).items()},
        telemetry=dict(data.get("telemetry", {})),
    )


def modes_from_event(event: dict[str, Any]) -> ModeSelection:
    return _modes(event)


def _modes(data: dict[str, Any]) -> ModeSelection:
    return ModeSelection(
        system=SystemMode(data["system"]),
        flight=FlightMode(data["flight"]) if data.get("flight") else None,
        control=ControlMode(data["control"]) if data.get("control") else None,
    )


def _state(data: dict[str, Any]) -> np.ndarray:
    state = np.zeros(13, dtype=float)
    state[0:3] = data.get("position_ned_m", [0.0, 0.0, 0.0])
    state[3:6] = data.get("velocity_ned_mps", [0.0, 0.0, 0.0])
    state[6:10] = data.get("quaternion", [1.0, 0.0, 0.0, 0.0])
    state[10:13] = data.get("body_rates_rps", [0.0, 0.0, 0.0])
    norm = np.linalg.norm(state[6:10])
    if norm == 0.0:
        raise ValueError("initial quaternion cannot be zero")
    state[6:10] /= norm
    return state
