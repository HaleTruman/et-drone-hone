from __future__ import annotations

import json
from pathlib import Path

from autonomy.opt_engine.types import Scenario, Target, Vec3, Waypoint


def list_scenarios(scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.exists():
        return []
    return sorted([p for p in scenarios_dir.glob("*.json") if p.is_file()])


def discover_scenario_files(root_dir: Path) -> list[Path]:
    course_dir = root_dir / "course_model"
    course_files = list_scenarios(course_dir)
    if course_files:
        return course_files

    legacy_dir = root_dir / "data" / "scenarios"
    return list_scenarios(legacy_dir)


def load_scenario(path: Path) -> Scenario:
    data = json.loads(path.read_text(encoding="utf-8"))

    name = str(data.get("name") or path.stem)
    frame = str(data.get("frame") or "internal")
    units = str(data.get("units") or "m")

    if "targets" in data:
        scenario = _load_targets_scenario(data, name=name, frame=frame, units=units)
    else:
        scenario = _load_waypoints_scenario(data, name=name, frame=frame, units=units)

    _validate_scenario(scenario)
    return scenario


def _load_waypoints_scenario(data: dict, *, name: str, frame: str, units: str) -> Scenario:
    waypoints_raw = data.get("waypoints")
    if not isinstance(waypoints_raw, list):
        raise ValueError("Scenario JSON must contain a list field: waypoints")

    waypoints: list[Waypoint] = []
    for i, item in enumerate(waypoints_raw):
        if not isinstance(item, dict):
            raise ValueError(f"waypoints[{i}] must be an object with x/y/z")
        waypoints.append(_parse_waypoint(item, context=f"waypoints[{i}]"))

    return Scenario(name=name, frame=frame, units=units, waypoints=tuple(waypoints))


def _load_targets_scenario(data: dict, *, name: str, frame: str, units: str) -> Scenario:
    targets_raw = data.get("targets")
    if not isinstance(targets_raw, list):
        raise ValueError("Scenario JSON must contain a list field: targets")

    level = data.get("level")
    mesh = data.get("mesh")
    generated_at = data.get("generated_at")

    course_targets: list[Target] = []
    waypoints: list[Waypoint] = []
    origin_target: Target | None = None

    for i, item in enumerate(targets_raw):
        if not isinstance(item, dict):
            raise ValueError(f"targets[{i}] must be an object")

        actor_label = str(item.get("actor_label") or "")
        actor_path = str(item.get("actor_path") or "")

        pos_key = f"position_{units}"
        position_raw = item.get(pos_key) or item.get("position_cm") or item.get("position_m") or item.get("position")
        if not isinstance(position_raw, dict):
            raise ValueError(f"targets[{i}].{pos_key} must be an object with x/y/z")
        position = _parse_waypoint(position_raw, context=f"targets[{i}].{pos_key}")

        axis_x = _parse_vec3(item.get("axis_x"), context=f"targets[{i}].axis_x")
        axis_y = _parse_vec3(item.get("axis_y"), context=f"targets[{i}].axis_y")
        axis_z = _parse_vec3(item.get("axis_z"), context=f"targets[{i}].axis_z")

        tgt = Target(
            actor_label=actor_label,
            actor_path=actor_path,
            position=position,
            axis_x=axis_x,
            axis_y=axis_y,
            axis_z=axis_z,
        )

        if _is_origin_target_label(actor_label):
            # Origin is treated as metadata and excluded from the course waypoint list.
            if origin_target is None:
                origin_target = tgt
            continue

        course_targets.append(tgt)
        waypoints.append(position)

    return Scenario(
        name=name,
        frame=frame,
        units=units,
        waypoints=tuple(waypoints),
        targets=tuple(course_targets),
        origin_target=origin_target,
        level=str(level) if level is not None else None,
        mesh=str(mesh) if mesh is not None else None,
        generated_at=str(generated_at) if generated_at is not None else None,
    )


def _is_origin_target_label(actor_label: str) -> bool:
    s = (actor_label or "").strip().lower()
    return s.endswith("origin") or "_origin" in s


def _parse_waypoint(obj: dict, *, context: str) -> Waypoint:
    if not isinstance(obj, dict):
        raise ValueError(f"{context} must be an object with x/y/z")
    try:
        x = float(obj["x"])
        y = float(obj["y"])
        z = float(obj["z"])
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid {context}: expected numeric x/y/z") from e
    return Waypoint(x=x, y=y, z=z)


def _parse_vec3(obj: object, *, context: str) -> Vec3:
    if not isinstance(obj, dict):
        raise ValueError(f"{context} must be an object with x/y/z")
    try:
        x = float(obj["x"])
        y = float(obj["y"])
        z = float(obj["z"])
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid {context}: expected numeric x/y/z") from e
    if not (_is_finite(x) and _is_finite(y) and _is_finite(z)):
        raise ValueError(f"Invalid {context}: axis values must be finite numbers")
    return Vec3(x=x, y=y, z=z)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


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
