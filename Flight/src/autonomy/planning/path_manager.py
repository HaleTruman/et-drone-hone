from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from core.schemas import Vec3
from sensing.gates import GateMap


class PathManager:
    def __init__(self, spline_generator: Callable[[np.ndarray], object] | None = None):
        self.spline_generator = spline_generator
        self._waypoints = np.empty((0, 3))
        self._segment_lengths = np.empty((0,))
        self._cumulative_lengths = np.array([0.0], dtype=float)

    def update_from_gate_map(self, gate_map: GateMap, limit: int = 3) -> np.ndarray:
        return self.set_waypoints(
            [gate.position_relative_ned_m or gate.position_local_ned_m for gate in gate_map.get_next_gates(limit)],
        )

    def set_waypoints(self, waypoints: Iterable[Vec3]) -> np.ndarray:
        self._waypoints = _waypoint_array(waypoints)
        self._refresh_lengths()
        return self.get_waypoints()

    def generate_spline(self) -> object:
        if self.spline_generator is None:
            raise NotImplementedError("Inject the chosen spline generator.")
        return self.spline_generator(self.get_waypoints())

    def get_waypoints(self) -> np.ndarray:
        return self._waypoints.copy()

    def project(self, position_local_ned_m: Vec3) -> dict[str, Any]:
        """Project a position onto the managed polyline path."""
        self._require_path()
        position = _vec3(position_local_ned_m, "position_local_ned_m")

        best_distance_sq = float("inf")
        best_segment_index = 0
        best_fraction = 0.0
        best_point = self._waypoints[0]

        for index, segment in enumerate(np.diff(self._waypoints, axis=0)):
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-12:
                fraction = 0.0
            else:
                fraction = float(np.clip(np.dot(position - self._waypoints[index], segment) / length_sq, 0.0, 1.0))
            point = self._waypoints[index] + fraction * segment
            distance_sq = float(np.dot(position - point, position - point))
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_segment_index = index
                best_fraction = fraction
                best_point = point

        along_track_m = float(
            self._cumulative_lengths[best_segment_index]
            + best_fraction * self._segment_lengths[best_segment_index]
        )
        tangent = self._segment_tangent(best_segment_index)
        return {
            "position_local_ned_m": tuple(float(value) for value in best_point),
            "tangent_local_ned": tuple(float(value) for value in tangent),
            "along_track_m": along_track_m,
            "cross_track_error_m": float(np.sqrt(best_distance_sq)),
            "segment_index": best_segment_index,
            "segment_fraction": best_fraction,
        }

    def carrot_point(self, position_local_ned_m: Vec3, lookahead_m: float) -> dict[str, Any]:
        """Return a path point lookahead_m ahead of the current path projection."""
        projection = self.project(position_local_ned_m)
        target_distance_m = min(
            float(projection["along_track_m"]) + max(0.0, float(lookahead_m)),
            float(self._cumulative_lengths[-1]),
        )
        point, tangent, segment_index = self._sample_at_distance(target_distance_m)
        return {
            "position_local_ned_m": tuple(float(value) for value in point),
            "tangent_local_ned": tuple(float(value) for value in tangent),
            "along_track_m": target_distance_m,
            "cross_track_error_m": projection["cross_track_error_m"],
            "projection_position_local_ned_m": projection["position_local_ned_m"],
            "projection_along_track_m": projection["along_track_m"],
            "segment_index": segment_index,
        }

    def _sample_at_distance(self, distance_m: float) -> tuple[np.ndarray, np.ndarray, int]:
        self._require_path()
        distance = float(np.clip(distance_m, 0.0, self._cumulative_lengths[-1]))
        index = int(np.searchsorted(self._cumulative_lengths, distance, side="right") - 1)
        index = min(max(index, 0), len(self._segment_lengths) - 1)
        length = float(self._segment_lengths[index])
        fraction = 0.0 if length <= 1e-12 else (distance - self._cumulative_lengths[index]) / length
        point = self._waypoints[index] + fraction * (self._waypoints[index + 1] - self._waypoints[index])
        return point, self._segment_tangent(index), index

    def _segment_tangent(self, index: int) -> np.ndarray:
        segment = self._waypoints[index + 1] - self._waypoints[index]
        norm = float(np.linalg.norm(segment))
        if norm <= 1e-12:
            return np.array((1.0, 0.0, 0.0), dtype=float)
        return segment / norm

    def _refresh_lengths(self) -> None:
        if len(self._waypoints) < 2:
            self._segment_lengths = np.empty((0,), dtype=float)
            self._cumulative_lengths = np.array([0.0], dtype=float)
            return
        self._segment_lengths = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        self._cumulative_lengths = np.concatenate(([0.0], np.cumsum(self._segment_lengths)))

    def _require_path(self) -> None:
        if len(self._waypoints) < 2 or float(self._cumulative_lengths[-1]) <= 1e-12:
            raise ValueError("PathManager requires at least two distinct waypoints.")


def _waypoint_array(waypoints: Iterable[Vec3]) -> np.ndarray:
    array = np.asarray(tuple(waypoints), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("waypoints must be a sequence of three-value local-NED positions")
    return array


def _vec3(value: Vec3, name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array
