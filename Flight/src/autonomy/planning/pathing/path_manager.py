from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from core.schemas import Vec3
from mapping.gates import GateRecord


@dataclass(frozen=True)
class PlannedPath:
    points_relative_ned_m: list[list[float]]
    anchors_relative_ned_m: list[list[float]]
    gate_ids: list[str]
    spacing_m: float
    computation_ms: float
    source: str = "path_manager"

    def to_log_dict(self, *, origin_local_ned_m: Any | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["origin_local_ned_m"] = (
            None if origin_local_ned_m is None else [float(value) for value in origin_local_ned_m]
        )
        return payload


class PathManager:
    def __init__(
        self,
        spline_generator: Callable[[np.ndarray], object] | None = None,
        *,
        spacing_m: float = 0.75,
        gate_axis_offset_m: float = 1.5,
        max_points: int = 240,
        max_gates: int = 8,
    ):
        self.spline_generator = spline_generator
        self.spacing_m = max(0.1, float(spacing_m))
        self.gate_axis_offset_m = max(0.0, float(gate_axis_offset_m))
        self.max_points = max(2, int(max_points))
        self.max_gates = max(1, int(max_gates))
        self._waypoints = np.empty((0, 3))
        self._segment_lengths = np.empty((0,))
        self._cumulative_lengths = np.array([0.0], dtype=float)

    def update_from_gate_map(self, gates: Iterable[GateRecord], limit: int = 3) -> np.ndarray:
        return self.set_waypoints(
            [gate.position_local_ned_m for gate in list(gates)[:limit]],
        )

    def plan_from_gate_map(self, gates: Iterable[GateRecord]) -> PlannedPath:
        started = perf_counter()
        planned_gates = self._planning_gates(gates)
        anchors = self._anchors_for_gates(planned_gates)
        points = self._sample_spline(anchors)
        self.set_waypoints(points)
        return PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=anchors.astype(float).tolist(),
            gate_ids=[gate.gate_id for gate in planned_gates],
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
        )

    def _planning_gates(self, gates: Iterable[GateRecord]) -> list[GateRecord]:
        return sorted(
            [
                gate
                for gate in gates
                if not gate.crossed
            ],
            key=lambda gate: (gate.sequence is None, gate.sequence, gate.gate_id),
        )[: self.max_gates]

    def set_waypoints(self, waypoints: Iterable[Vec3]) -> np.ndarray:
        self._waypoints = _waypoint_array(waypoints)
        self._refresh_lengths()
        return self.get_waypoints()

    def build_test_path(
        self,
        *,
        length_m: float = 30.0,
        width_m: float = 8.0,
        height_m: float = 1.0,
        point_count: int = 31,
    ) -> np.ndarray:
        length = float(length_m)
        width = float(width_m)
        height = float(height_m)
        count = int(point_count)
        if length <= 0.0:
            raise ValueError("length_m must be positive")
        if count < 2:
            raise ValueError("point_count must be at least 2")

        north = np.linspace(0.0, length, count)
        east = 0.5 * width * np.sin(2.0 * np.pi * north / (0.5 * length))
        down = -1.5 * height + 0.5 * height * np.sin(2.0 * np.pi * north / length)
        return self.set_waypoints(zip(north, east, down))

    def build_straight_line(
        self,
        *,
        length_m: float = 10,
        point_count: int = 30,
        up_down_angle_deg: float = 0.0,
        left_right_angle_deg: float = 0.0,
    ) -> np.ndarray:
        """Build a straight local-NED path from the origin.

        Angles are measured from forward/north. Positive up_down_angle_deg points
        upward, and positive left_right_angle_deg points right/east.
        """
        length = float(length_m)
        count = int(point_count)
        up_down_rad = np.deg2rad(float(up_down_angle_deg))
        left_right_rad = np.deg2rad(float(left_right_angle_deg))
        if length <= 0.0:
            raise ValueError("length_m must be positive")
        if count < 2:
            raise ValueError("point_count must be at least 2")

        distance = np.linspace(0.0, length, count)
        horizontal_scale = np.cos(up_down_rad)
        north = distance * horizontal_scale * np.cos(left_right_rad)
        east = distance * horizontal_scale * np.sin(left_right_rad)
        down = -distance * np.sin(up_down_rad)

        return self.set_waypoints(zip(north, east, down))

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

    def _anchors_for_gates(self, gates: list[GateRecord]) -> np.ndarray:
        anchors = [np.zeros(3, dtype=float)]
        for gate in gates:
            anchors.append(np.asarray(gate.position_local_ned_m, dtype=float))
        return self._dedupe_points(np.asarray(anchors, dtype=float))

    def _sample_spline(self, anchors: np.ndarray) -> list[list[float]]:
        if len(anchors) < 3:
            return self._sample_polyline(anchors)

        samples = [anchors[0]]
        for index in range(len(anchors) - 1):
            p0 = anchors[max(index - 1, 0)]
            p1 = anchors[index]
            p2 = anchors[index + 1]
            p3 = anchors[min(index + 2, len(anchors) - 1)]
            distance = float(np.linalg.norm(p2 - p1))
            steps = max(1, int(np.ceil(distance / self.spacing_m)))
            for step in range(1, steps + 1):
                t = step / steps
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
                )
                samples.append(point)
                if len(samples) >= self.max_points:
                    return np.asarray(samples, dtype=float).tolist()
        return np.asarray(samples, dtype=float).tolist()

    def _sample_polyline(self, anchors: np.ndarray) -> list[list[float]]:
        if len(anchors) <= 1:
            return anchors.astype(float).tolist()
        samples = [anchors[0]]
        for start, end in zip(anchors[:-1], anchors[1:]):
            delta = end - start
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                continue
            steps = max(1, int(np.ceil(distance / self.spacing_m)))
            for step in range(1, steps + 1):
                samples.append(start + delta * (step / steps))
                if len(samples) >= self.max_points:
                    return np.asarray(samples, dtype=float).tolist()
        return np.asarray(samples, dtype=float).tolist()

    @staticmethod
    def _dedupe_points(points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return points
        deduped = [points[0]]
        for point in points[1:]:
            if float(np.linalg.norm(point - deduped[-1])) > 1e-6:
                deduped.append(point)
        return np.asarray(deduped, dtype=float)


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

