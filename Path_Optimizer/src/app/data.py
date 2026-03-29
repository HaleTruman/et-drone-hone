from __future__ import annotations

import json
import math
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class CoursePoint:
    label: str
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class Course:
    name: str
    frame: str
    units: str
    targets: tuple[CoursePoint, ...]
    origin: CoursePoint | None = None
    level: str | None = None
    generated_at: str | None = None

    @property
    def path_length_m(self) -> float:
        total = 0.0
        for index in range(1, len(self.targets)):
            previous = self.targets[index - 1]
            current = self.targets[index]
            total += math.dist(
                (previous.x_m, previous.y_m, previous.z_m),
                (current.x_m, current.y_m, current.z_m),
            )
        return total

    @property
    def bounds_m(self) -> dict[str, float]:
        xs = [point.x_m for point in self.targets]
        ys = [point.y_m for point in self.targets]
        zs = [point.z_m for point in self.targets]
        return {
            "width": (max(xs) - min(xs)) if xs else 0.0,
            "depth": (max(ys) - min(ys)) if ys else 0.0,
            "height": (max(zs) - min(zs)) if zs else 0.0,
        }


@dataclass(frozen=True)
class PlannerTrajectory:
    t_s: tuple[float, ...]
    pos_m: tuple[CoursePoint, ...]


@dataclass(frozen=True)
class ReferenceTrajectorySnapshot:
    course_path: str
    drone_position: CoursePoint
    trajectory: PlannerTrajectory


@dataclass(frozen=True)
class PlanningSession:
    course_path: str
    loop_status: str
    drone_position: CoursePoint
    visible_gates: tuple[CoursePoint, ...]
    initial_guess: PlannerTrajectory
    planner_horizon: int
    planner_tol: float


def discover_course_files(root_dir: str) -> list[str]:
    course_dir = root_dir + "/course_model"
    if os.path.isdir(course_dir):
        files = sorted(
            os.path.join(course_dir, name)
            for name in os.listdir(course_dir)
            if name.endswith(".json") and os.path.isfile(os.path.join(course_dir, name))
        )
        if files:
            return files

    legacy_dir = root_dir + "/data/scenarios"
    if os.path.isdir(legacy_dir):
        return sorted(
            os.path.join(legacy_dir, name)
            for name in os.listdir(legacy_dir)
            if name.endswith(".json") and os.path.isfile(os.path.join(legacy_dir, name))
        )

    return []


def load_reference_trajectory(path: str) -> ReferenceTrajectorySnapshot:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    drone_position_xyz = data["drone_state"]["position"]
    trajectory_positions = data["reference_trajectory"]["pos"]
    trajectory_times = data["reference_trajectory"].get("t", [])

    return ReferenceTrajectorySnapshot(
        course_path=os.path.abspath(str(data["course_path"])),
        drone_position=_point_from_xyz("Drone", drone_position_xyz),
        trajectory=PlannerTrajectory(
            t_s=tuple(float(t_value) for t_value in trajectory_times),
            pos_m=tuple(
                _point_from_xyz(f"Reference {index + 1}", xyz)
                for index, xyz in enumerate(trajectory_positions)
            ),
        ),
    )


def load_planning_session(path: str) -> PlanningSession:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    drone_position_xyz = data["drone_state"]["position"]
    visible_gate_xyz = data.get("visible_gates", [])
    guess_positions = data["initial_guess"]["pos"]
    guess_times = data["initial_guess"].get("t", [])

    return PlanningSession(
        course_path=os.path.abspath(str(data["course_path"])),
        loop_status=str(data.get("loop_status") or "unknown"),
        drone_position=_point_from_xyz("Drone", drone_position_xyz),
        visible_gates=tuple(
            _point_from_xyz(f"Visible Gate {index + 1}", xyz)
            for index, xyz in enumerate(visible_gate_xyz)
        ),
        initial_guess=PlannerTrajectory(
            t_s=tuple(float(t_value) for t_value in guess_times),
            pos_m=tuple(
                _point_from_xyz(f"Guess {index + 1}", xyz)
                for index, xyz in enumerate(guess_positions)
            ),
        ),
        planner_horizon=int(data["planner"]["horizon"]),
        planner_tol=float(data["planner"]["tol"]),
    )


def load_course(path: str) -> Course:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    units = str(data.get("units") or "m")
    scale_to_m = 0.01 if units == "cm" else 1.0

    if "targets" in data:
        targets, origin = _load_targets(targets_data=data["targets"], units=units, scale_to_m=scale_to_m)
    else:
        targets, origin = _load_waypoints(data.get("waypoints", []), scale_to_m=scale_to_m), None

    if len(targets) < 2:
        raise ValueError("Course must contain at least two targets or waypoints.")

    return Course(
        name=str(data.get("name") or os.path.splitext(os.path.basename(path))[0]),
        frame=str(data.get("frame") or "internal"),
        units=units,
        targets=tuple(targets),
        origin=origin,
        level=str(data["level"]) if data.get("level") else None,
        generated_at=str(data["generated_at"]) if data.get("generated_at") else None,
    )


def _load_targets(*, targets_data: list[dict], units: str, scale_to_m: float) -> tuple[list[CoursePoint], CoursePoint | None]:
    if not isinstance(targets_data, list):
        raise ValueError("targets must be a list")

    course_targets: list[CoursePoint] = []
    origin: CoursePoint | None = None
    position_key = f"position_{units}"

    for index, item in enumerate(targets_data):
        if not isinstance(item, dict):
            raise ValueError(f"targets[{index}] must be an object")

        label = str(item.get("actor_label") or f"Target {index + 1}")
        position_raw = item.get(position_key) or item.get("position_cm") or item.get("position_m") or item.get("position")
        point = _make_point(label=label, raw_position=position_raw, scale_to_m=scale_to_m, context=f"targets[{index}]")
        if _is_origin(label):
            origin = point
        else:
            course_targets.append(point)

    return course_targets, origin


def _load_waypoints(waypoints_data: list[dict], scale_to_m: float) -> list[CoursePoint]:
    if not isinstance(waypoints_data, list):
        raise ValueError("waypoints must be a list")

    return [
        _make_point(
            label=f"WP {index + 1}",
            raw_position=item,
            scale_to_m=scale_to_m,
            context=f"waypoints[{index}]",
        )
        for index, item in enumerate(waypoints_data)
    ]


def _make_point(*, label: str, raw_position: object, scale_to_m: float, context: str) -> CoursePoint:
    if not isinstance(raw_position, dict):
        raise ValueError(f"{context} is missing a valid position")
    try:
        x_value = float(raw_position["x"]) * scale_to_m
        y_value = float(raw_position["y"]) * scale_to_m
        z_value = float(raw_position["z"]) * scale_to_m
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{context} must contain numeric x/y/z values") from exc
    return CoursePoint(label=label, x_m=x_value, y_m=y_value, z_m=z_value)


def _is_origin(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized.endswith("origin") or "_origin" in normalized


def _point_from_xyz(label: str, xyz: object) -> CoursePoint:
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise ValueError(f"{label} must be a 3-element coordinate")
    return CoursePoint(
        label=label,
        x_m=float(xyz[0]),
        y_m=float(xyz[1]),
        z_m=float(xyz[2]),
    )
