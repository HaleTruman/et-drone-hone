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
    gate_center_errors_m: dict[str, float]
    gate_center_tolerance_m: float
    spline_corner_tightness: float
    spacing_m: float
    computation_ms: float
    source: str = "path_manager"

    def to_log_dict(self, *, origin_local_ned_m: Any | None = None) -> dict[str, Any]:
        payload = asdict(self)
        local_points = [list(point) for point in self.points_relative_ned_m]
        payload["points_local_ned_m"] = local_points
        payload["origin_local_ned_m"] = None
        if origin_local_ned_m is not None:
            origin = [float(value) for value in origin_local_ned_m]
            payload["origin_local_ned_m"] = origin
            payload["points_relative_ned_m"] = [
                [float(point[index]) - origin[index] for index in range(3)]
                for point in local_points
            ]
        return payload


@dataclass
class _ObservedGateTrack:
    track_id: str
    gate: GateRecord
    observation_count: int
    last_observed_cycle: int | None


class PathManager:
    def __init__(
        self,
        spline_generator: Callable[[np.ndarray], object] | None = None,
        *,
        spacing_m: float = 0.75,
        gate_axis_offset_m: float = 1.5,
        max_points: int = 240,
        max_gates: int = 8,
        exclusion_distance_m: float = 0.0,
        max_gate_distance_m: float | None = None,
        passed_gate_distance_m: float = 2.0,
        gate_center_tolerance_m: float = 0.5,
        spline_corner_tightness: float = 0.5,
        planning_mode: str = "gate_map",
        gate_association_distance_m: float = 6.0,
        gate_min_observations: int = 1,
    ):
        self.spline_generator = spline_generator
        self.spacing_m = max(0.1, float(spacing_m))
        self.gate_axis_offset_m = max(0.0, float(gate_axis_offset_m))
        self.max_points = max(2, int(max_points))
        self.max_gates = max(1, int(max_gates))
        self.exclusion_distance_m = max(0.0, float(exclusion_distance_m))
        self.max_gate_distance_m = (
            None
            if max_gate_distance_m is None
            else max(0.0, float(max_gate_distance_m))
        )
        self.passed_gate_distance_m = max(0.0, float(passed_gate_distance_m))
        self.gate_center_tolerance_m = max(0.0, float(gate_center_tolerance_m))
        self.spline_corner_tightness = float(np.clip(float(spline_corner_tightness), 0.0, 1.0))
        self.planning_mode = _normalize_planning_mode(planning_mode)
        self.gate_association_distance_m = max(0.0, float(gate_association_distance_m))
        self.gate_min_observations = max(1, int(gate_min_observations))
        self._waypoints = np.empty((0, 3))
        self._segment_lengths = np.empty((0,))
        self._cumulative_lengths = np.array([0.0], dtype=float)
        self._observed_gate_tracks: dict[str, _ObservedGateTrack] = {}
        self._observed_gate_track_sequence = 0
        self.primary_gate_position_local_ned_m: tuple[float, float, float] | None = None
        self.secondary_gate_position_local_ned_m: tuple[float, float, float] | None = None
        self.primary_gate_id: str | None = None
        self.secondary_gate_id: str | None = None

    def update_from_gate_map(self, gates: Iterable[GateRecord], limit: int = 3) -> np.ndarray:
        return self.set_waypoints(
            [gate.position_local_ned_m for gate in list(gates)[:limit]],
        )
 
    def plan_from_gate_map(
        self,
        gates: Iterable[GateRecord],
        *,
        position_local_ned_m: Vec3 | None = None,
        activate: bool = True,
    ) -> PlannedPath:
        started = perf_counter()
        start_position = None if position_local_ned_m is None else _vec3(position_local_ned_m, "position_local_ned_m")
        planned_gates = self._planning_gates(gates, position_local_ned_m=position_local_ned_m)
        anchors = self._anchors_for_gates(planned_gates, start_position_local_ned_m=start_position)
        points = self._sample_spline(anchors)
        points = self._constrain_gate_centers(points, planned_gates)
        gate_center_errors = self._gate_center_errors(points, planned_gates)
        if activate and len(points) >= 2:
            self.set_waypoints(points)
        return PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=anchors.astype(float).tolist(),
            gate_ids=[gate.gate_id for gate in planned_gates],
            gate_center_errors_m=gate_center_errors,
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="gate_map",
        )

    def plan_from_observed_gates(
        self,
        gates: Iterable[GateRecord],
        *,
        position_local_ned_m: Vec3,
        activate: bool = True,
    ) -> PlannedPath:
        started = perf_counter()
        position = _vec3(position_local_ned_m, "position_local_ned_m")
        visible_track_ids = self._update_observed_gate_tracks(gates)
        planned_gates = self._planning_observed_gates(visible_track_ids, position)
        self._set_primary_secondary_gates(planned_gates)

        anchors = self._anchors_for_gates(planned_gates, start_position_local_ned_m=position)
        points = self._sample_spline(anchors)
        points = self._constrain_gate_centers(points, planned_gates)
        gate_center_errors = self._gate_center_errors(points, planned_gates)
        if activate and len(points) >= 2:
            self.set_waypoints(points)
        return PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=anchors.astype(float).tolist(),
            gate_ids=[gate.gate_id for gate in planned_gates],
            gate_center_errors_m=gate_center_errors,
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="observed_next_two",
        )

    def _planning_gates(
        self,
        gates: Iterable[GateRecord],
        *,
        position_local_ned_m: Vec3 | None = None,
    ) -> list[GateRecord]:
        position = None if position_local_ned_m is None else _vec3(position_local_ned_m, "position_local_ned_m")

        def distance_m(gate: GateRecord) -> float:
            if position is None:
                return 0.0
            return float(np.linalg.norm(np.asarray(gate.position_local_ned_m, dtype=float) - position))

        return sorted(
            [
                gate
                for gate in gates
                if not gate.crossed
                and (
                    position is None
                    or (
                        distance_m(gate) > self.passed_gate_distance_m
                        and distance_m(gate) >= self.exclusion_distance_m
                        and (
                            self.max_gate_distance_m is None
                            or distance_m(gate) <= self.max_gate_distance_m
                        )
                    )
                )
            ],
            key=lambda gate: (
                distance_m(gate),
                gate.sequence is None,
                gate.sequence,
                gate.gate_id,
            ),
        )[: self.max_gates]

    def _planning_observed_gates(self, visible_track_ids: set[str], position: np.ndarray) -> list[GateRecord]:
        def distance_m(gate: GateRecord) -> float:
            return float(np.linalg.norm(np.asarray(gate.position_local_ned_m, dtype=float) - position))

        return sorted(
            [
                self._observed_gate_tracks[track_id].gate
                for track_id in visible_track_ids
                if self._observed_gate_tracks[track_id].observation_count >= self.gate_min_observations
                and distance_m(self._observed_gate_tracks[track_id].gate) > self.passed_gate_distance_m
                and distance_m(self._observed_gate_tracks[track_id].gate) >= self.exclusion_distance_m
                and (
                    self.max_gate_distance_m is None
                    or distance_m(self._observed_gate_tracks[track_id].gate) <= self.max_gate_distance_m
                )
            ],
            key=lambda gate: (
                distance_m(gate),
                gate.sequence is None,
                gate.sequence,
                gate.gate_id,
            ),
        )[: self.max_gates]

    def _update_observed_gate_tracks(self, gates: Iterable[GateRecord]) -> set[str]:
        visible_track_ids: set[str] = set()
        for gate in gates:
            track_id = self._matching_observed_gate_track_id(gate, excluded_track_ids=visible_track_ids)
            if track_id is None:
                track_id = self._new_observed_gate_track_id(gate)
                self._observed_gate_tracks[track_id] = _ObservedGateTrack(
                    track_id=track_id,
                    gate=gate,
                    observation_count=1,
                    last_observed_cycle=gate.last_observed_cycle,
                )
            else:
                track = self._observed_gate_tracks[track_id]
                track.gate = gate
                track.observation_count += 1
                track.last_observed_cycle = gate.last_observed_cycle
            visible_track_ids.add(track_id)
        return visible_track_ids

    def _matching_observed_gate_track_id(
        self,
        gate: GateRecord,
        *,
        excluded_track_ids: set[str],
    ) -> str | None:
        if gate.gate_id in self._observed_gate_tracks and gate.gate_id not in excluded_track_ids:
            track = self._observed_gate_tracks[gate.gate_id]
            distance = float(
                np.linalg.norm(
                    np.asarray(track.gate.position_local_ned_m, dtype=float)
                    - np.asarray(gate.position_local_ned_m, dtype=float)
                )
            )
            if distance <= self.gate_association_distance_m:
                return gate.gate_id

        gate_position = np.asarray(gate.position_local_ned_m, dtype=float)
        closest_id: str | None = None
        closest_distance = self.gate_association_distance_m
        for track_id, track in self._observed_gate_tracks.items():
            if track_id in excluded_track_ids:
                continue
            distance = float(np.linalg.norm(np.asarray(track.gate.position_local_ned_m, dtype=float) - gate_position))
            if distance <= closest_distance:
                closest_id = track_id
                closest_distance = distance
        return closest_id

    def _new_observed_gate_track_id(self, gate: GateRecord) -> str:
        if gate.gate_id and gate.gate_id not in self._observed_gate_tracks:
            return gate.gate_id
        self._observed_gate_track_sequence += 1
        return f"observed_gate_{self._observed_gate_track_sequence}"

    def _set_primary_secondary_gates(self, gates: list[GateRecord]) -> None:
        primary = gates[0] if len(gates) >= 1 else None
        secondary = gates[1] if len(gates) >= 2 else None
        self.primary_gate_position_local_ned_m = None if primary is None else primary.position_local_ned_m
        self.secondary_gate_position_local_ned_m = None if secondary is None else secondary.position_local_ned_m
        self.primary_gate_id = None if primary is None else primary.gate_id
        self.secondary_gate_id = None if secondary is None else secondary.gate_id

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
        activate: bool = True,
    ) -> PlannedPath:
        started = perf_counter()
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
        waypoints = np.column_stack((north, east, down))
        waypoints = waypoints - waypoints[0]
        if activate:
            points = self.set_waypoints(waypoints).astype(float).tolist()
        else:
            points = _waypoint_array(waypoints).astype(float).tolist()
        return PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=points,
            gate_ids=[],
            gate_center_errors_m={},
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="test_path",
        )

    def build_straight_line(
        self,
        *,
        length_m: float = 10,
        point_count: int = 30,
        up_down_angle_deg: float = 0.0,
        left_right_angle_deg: float = 0.0,
        activate: bool = True,
    ) -> PlannedPath:
        """Build a straight local-NED path from the origin.

        Angles are measured from forward/north. Positive up_down_angle_deg points
        upward, and positive left_right_angle_deg points right/east.
        """
        started = perf_counter()
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

        waypoints = np.column_stack((north, east, down))
        if activate:
            points = self.set_waypoints(waypoints).astype(float).tolist()
        else:
            points = _waypoint_array(waypoints).astype(float).tolist()
        return PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=points,
            gate_ids=[],
            gate_center_errors_m={},
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="straight_line",
        )

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

    def _anchors_for_gates(
        self,
        gates: list[GateRecord],
        *,
        start_position_local_ned_m: np.ndarray | None = None,
    ) -> np.ndarray:
        start_position = (
            np.zeros(3, dtype=float)
            if start_position_local_ned_m is None
            else np.asarray(start_position_local_ned_m, dtype=float)
        )
        anchors = [start_position]
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
                line_point = p1 + (p2 - p1) * t
                point = (1.0 - self.spline_corner_tightness) * point + self.spline_corner_tightness * line_point
                samples.append(point)
                if len(samples) >= self.max_points:
                    return np.asarray(samples, dtype=float).tolist()
        return np.asarray(samples, dtype=float).tolist()

    def _constrain_gate_centers(self, points: list[list[float]], gates: list[GateRecord]) -> list[list[float]]:
        if not gates:
            return points

        constrained = _waypoint_array(points).tolist()
        for gate in gates:
            center = np.asarray(gate.position_local_ned_m, dtype=float)
            if self._minimum_distance(constrained, center) <= self.gate_center_tolerance_m:
                continue

            insert_at = self._nearest_path_index(constrained, center)
            constrained.insert(insert_at, center.astype(float).tolist())

        if len(constrained) <= self.max_points:
            return constrained

        required = {
            tuple(float(value) for value in gate.position_local_ned_m)
            for gate in gates
        }
        return self._trim_preserving_required_points(constrained, required)

    def _gate_center_errors(self, points: list[list[float]], gates: list[GateRecord]) -> dict[str, float]:
        return {
            gate.gate_id: self._minimum_distance(points, np.asarray(gate.position_local_ned_m, dtype=float))
            for gate in gates
        }

    @staticmethod
    def _minimum_distance(points: list[list[float]], target: np.ndarray) -> float:
        if not points:
            return float("inf")
        return float(np.min(np.linalg.norm(_waypoint_array(points) - target, axis=1)))

    @staticmethod
    def _nearest_path_index(points: list[list[float]], target: np.ndarray) -> int:
        if not points:
            return 0
        distances = np.linalg.norm(_waypoint_array(points) - target, axis=1)
        return int(np.argmin(distances)) + 1

    def _trim_preserving_required_points(
        self,
        points: list[list[float]],
        required: set[tuple[float, float, float]],
    ) -> list[list[float]]:
        keep_indices = {
            index
            for index, point in enumerate(points)
            if tuple(float(value) for value in point) in required
        }
        keep_indices.add(0)
        keep_indices.add(len(points) - 1)

        removable = [index for index in range(len(points)) if index not in keep_indices]
        overflow = len(points) - self.max_points
        remove_indices = set(removable[-overflow:]) if overflow > 0 else set()
        return [point for index, point in enumerate(points) if index not in remove_indices]

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


def _normalize_planning_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"gate_map", "observed_next_two"}:
        return mode
    raise ValueError("planning_mode must be 'gate_map' or 'observed_next_two'")
