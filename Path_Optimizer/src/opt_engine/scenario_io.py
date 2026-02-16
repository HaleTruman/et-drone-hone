from __future__ import annotations

import json
from pathlib import Path

from opt_engine.types import Scenario, Waypoint


def list_scenarios(scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.exists():
        return []
    return sorted([p for p in scenarios_dir.glob("*.json") if p.is_file()])


def load_scenario(path: Path) -> Scenario:
    data = json.loads(path.read_text(encoding="utf-8"))

    name = str(data.get("name") or path.stem)
    frame = str(data.get("frame") or "internal")
    units = str(data.get("units") or "m")
    waypoints_raw = data.get("waypoints")
    if not isinstance(waypoints_raw, list):
        raise ValueError("Scenario JSON must contain a list field: waypoints")

    waypoints: list[Waypoint] = []
    for i, item in enumerate(waypoints_raw):
        if not isinstance(item, dict):
            raise ValueError(f"waypoints[{i}] must be an object with x/y/z")
        try:
            x = float(item["x"])
            y = float(item["y"])
            z = float(item["z"])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Invalid waypoint at index {i}: expected numeric x/y/z") from e
        waypoints.append(Waypoint(x=x, y=y, z=z))

    scenario = Scenario(name=name, frame=frame, units=units, waypoints=tuple(waypoints))
    _validate_scenario(scenario)
    return scenario


def _validate_scenario(scenario: Scenario) -> None:
    if len(scenario.waypoints) < 2:
        raise ValueError("Scenario must contain at least 2 waypoints")
    if scenario.units not in ("m", "cm"):
        raise ValueError('Scenario units must be "m" or "cm"')
    if scenario.frame not in ("internal", "unreal"):
        raise ValueError('Scenario frame must be "internal" or "unreal"')

    for i in range(len(scenario.waypoints) - 1):
        a = scenario.waypoints[i]
        b = scenario.waypoints[i + 1]
        if a.x == b.x and a.y == b.y and a.z == b.z:
            raise ValueError(f"Consecutive duplicate waypoint at indices {i} and {i+1}")

