"""Obstacle volume loading and clearance checks."""

import json
import math
from pathlib import Path

from src.dataset_generation.config import OBSTACLE_MAP_JSONL


_OBSTACLES = None


def dataset_generation_root():
    return Path(__file__).resolve().parent


def obstacle_map_path():
    configured = Path(OBSTACLE_MAP_JSONL)
    if configured.is_absolute():
        return configured
    return dataset_generation_root() / configured


def load_obstacles():
    global _OBSTACLES
    if _OBSTACLES is not None:
        return _OBSTACLES

    path = obstacle_map_path()
    if not path.exists():
        _OBSTACLES = []
        return _OBSTACLES

    obstacles = []
    with path.open("r", encoding="utf-8-sig") as obstacle_file:
        for line in obstacle_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "world_location_m" in record:
                location = record.get("world_location_m", {})
                location_scale = 100.0
            else:
                location = record.get("world_location_cm", {})
                location_scale = 1.0
            if "size_m" in record:
                size = record.get("size_m", {})
                size_scale = 100.0
            else:
                size = record.get("size_cm", {})
                size_scale = 1.0
            rotation = record.get("world_rotation_deg", {})
            obstacles.append(
                {
                    "name": record.get("component_name", ""),
                    "kind": record.get("kind", ""),
                    "center": (
                        float(location.get("x", 0.0)) * location_scale,
                        float(location.get("y", 0.0)) * location_scale,
                        float(location.get("z", 0.0)) * location_scale,
                    ),
                    "half_size": (
                        float(size.get("x", 0.0)) * size_scale * 0.5,
                        float(size.get("y", 0.0)) * size_scale * 0.5,
                        float(size.get("z", 0.0)) * size_scale * 0.5,
                    ),
                    "yaw_deg": float(rotation.get("yaw", 0.0)),
                }
            )

    _OBSTACLES = obstacles
    return _OBSTACLES


def vector_xyz(point):
    return (float(point.x), float(point.y), float(point.z))


def point_to_obstacle_local(point, obstacle):
    x, y, z = vector_xyz(point)
    cx, cy, cz = obstacle["center"]
    dx = x - cx
    dy = y - cy
    dz = z - cz

    yaw = math.radians(-obstacle["yaw_deg"])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        dx * cos_yaw - dy * sin_yaw,
        dx * sin_yaw + dy * cos_yaw,
        dz,
    )


def distance_from_point_to_obstacle(point, obstacle):
    local_x, local_y, local_z = point_to_obstacle_local(point, obstacle)
    half_x, half_y, half_z = obstacle["half_size"]
    outside_x = max(abs(local_x) - half_x, 0.0)
    outside_y = max(abs(local_y) - half_y, 0.0)
    outside_z = max(abs(local_z) - half_z, 0.0)
    return math.sqrt(outside_x * outside_x + outside_y * outside_y + outside_z * outside_z)


def nearest_obstacle(point):
    closest = None
    for obstacle in load_obstacles():
        clearance = distance_from_point_to_obstacle(point, obstacle)
        if closest is None or clearance < closest["distance_cm"]:
            closest = {
                "obstacle": obstacle,
                "distance_cm": clearance,
            }
    return closest


def point_clears_obstacles(point, clearance_cm):
    closest = nearest_obstacle(point)
    if closest is None:
        return True
    return closest["distance_cm"] >= clearance_cm


def all_points_clear_obstacles(points, clearance_cm):
    return all(point_clears_obstacles(point, clearance_cm) for point in points)
