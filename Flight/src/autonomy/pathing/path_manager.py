import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from core.schema import Vec3, VehicleState
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
    adaptive_spline_tightness: bool = False
    distant_spline_corner_tightness: float | None = None
    min_spline_corner_tightness: float | None = None
    max_spline_corner_tightness: float | None = None
    gentle_turn_angle_deg: float | None = None
    sharp_turn_angle_deg: float | None = None
    short_segment_reference_m: float | None = None
    long_segment_reference_m: float | None = None

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


@dataclass(frozen=True)
class PathProjection:
    closest_point_local_ned_m: Vec3
    tangent_local_ned: Vec3
    along_track_m: float
    cross_track_error_m: float
    segment_index: int
    segment_fraction: float

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        adaptive_spline_tightness: bool = True,
        distant_spline_corner_tightness: float = 0.10,
        min_spline_corner_tightness: float = 0.15,
        max_spline_corner_tightness: float = 0.9,
        gentle_turn_angle_deg: float = 20.0,
        sharp_turn_angle_deg: float = 80.0,
        short_segment_reference_m: float = 12.0,
        long_segment_reference_m: float = 25.0,
        planning_mode: str = "gate_map",
        path_tail_length_m: float = 10.0,
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
        self.adaptive_spline_tightness = bool(adaptive_spline_tightness)
        self.distant_spline_corner_tightness = float(np.clip(float(distant_spline_corner_tightness), 0.0, 1.0))
        self.min_spline_corner_tightness = float(np.clip(float(min_spline_corner_tightness), 0.0, 1.0))
        self.max_spline_corner_tightness = float(np.clip(float(max_spline_corner_tightness), 0.0, 1.0))
        if self.distant_spline_corner_tightness > self.max_spline_corner_tightness:
            raise ValueError("distant_spline_corner_tightness cannot exceed max_spline_corner_tightness")
        if self.min_spline_corner_tightness > self.max_spline_corner_tightness:
            raise ValueError("min_spline_corner_tightness cannot exceed max_spline_corner_tightness")
        self.gentle_turn_angle_rad = math.radians(max(0.0, float(gentle_turn_angle_deg)))
        self.sharp_turn_angle_rad = math.radians(max(0.0, float(sharp_turn_angle_deg)))
        if self.gentle_turn_angle_rad > self.sharp_turn_angle_rad:
            raise ValueError("gentle_turn_angle_deg cannot exceed sharp_turn_angle_deg")
        self.short_segment_reference_m = max(1e-6, float(short_segment_reference_m))
        self.long_segment_reference_m = max(self.short_segment_reference_m, float(long_segment_reference_m))
        self.planning_mode = _normalize_planning_mode(planning_mode)
        self._path_tail_length_m = max(0.0, float(path_tail_length_m))
        self._last_terminal_gate_tangent: np.ndarray | None = None
        self._waypoints = np.empty((0, 3))
        self._segment_lengths = np.empty((0,))
        self._cumulative_lengths = np.array([0.0], dtype=float)
        self.test_path: PlannedPath | None = None

    def update_from_gate_map(self, gates: Iterable[GateRecord], limit: int = 3) -> np.ndarray:
        return self.set_waypoints(
            [gate.position_local_ned_m for gate in list(gates)[:limit]],
        )
 
    def plan_from_gate_map(
        self,
        gates: Iterable[GateRecord],
        *,
        position_local_ned_m: Vec3 | None = None,
    ) -> PlannedPath:
        started = perf_counter()
        start_position = None if position_local_ned_m is None else _vec3(position_local_ned_m, "position_local_ned_m")
        planned_gates = self._planning_gates(gates, position_local_ned_m=position_local_ned_m)
        anchors = self._anchors_for_gates(planned_gates, start_position_local_ned_m=start_position)
        points = self._sample_spline(anchors)
        points = self._constrain_gate_centers(points, planned_gates)
        points = self._extend_path(
            _waypoint_array(points),
            tail_tangent=self._terminal_gate_tangent(anchors),
        ).tolist()
        gate_center_errors = self._gate_center_errors(points, planned_gates)
        if len(points) >= 2:
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
            **self._planned_path_spline_metadata(),
        )

    def plan_from_gate_centers(
        self,
        gates: Iterable[GateRecord],
        *,
        position_local_ned_m: Vec3,
    ) -> PlannedPath:
        started = perf_counter()
        position = _vec3(position_local_ned_m, "position_local_ned_m")
        gate_centers = [
            (
                gate,
                _vec3(gate.position_local_ned_m, f"gates[{index}].position_local_ned_m"),
            )
            for index, gate in enumerate(gates)
        ]
        gate_centers.sort(key=lambda item: float(np.linalg.norm(item[1] - position)))
        planned_gates = [gate for gate, _ in gate_centers]
        centers = [center for _, center in gate_centers]

        anchors = self._dedupe_points(np.asarray([position, *centers], dtype=float))
        points = self._extend_path(
            _waypoint_array(self._sample_spline(anchors)),
            tail_tangent=self._terminal_gate_tangent(anchors),
        ).tolist()
        if len(points) >= 2:
            self.set_waypoints(points)
        elif len(points) < 2 and len(self._waypoints) >= 2:
            points = self.get_waypoints().astype(float).tolist()

        gate_ids = [
            str(gate.gate_id or f"gate_{index + 1}")
            for index, gate in enumerate(planned_gates)
        ]
        gate_center_errors = {
            gate_id: self._minimum_distance(points, center)
            for gate_id, center in zip(gate_ids, centers)
        }
        return PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=anchors.astype(float).tolist(),
            gate_ids=gate_ids,
            gate_center_errors_m=gate_center_errors,
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="center_targets",
            **self._planned_path_spline_metadata(),
        )

    def plan(
        self,
        *,
        gates: Iterable[GateRecord],
        vehicle_state: VehicleState,
    ) -> PlannedPath:
        position_local_ned_m = vehicle_state.position_local_ned_m

        if self.planning_mode == "test_path":
            if self.test_path is None:
                raise ValueError("test_path planning mode requires PathManager.test_path.")
            self.set_waypoints(self.test_path.points_relative_ned_m)
            return self.test_path

        if self.planning_mode == "center_targets":
            return self.plan_from_gate_centers(
                gates,
                position_local_ned_m=position_local_ned_m,
            )

        return self.plan_from_gate_map(
            gates,
            position_local_ned_m=position_local_ned_m,
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
        points = self.set_waypoints(waypoints).astype(float).tolist()
        self.test_path = PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=points,
            gate_ids=[],
            gate_center_errors_m={},
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="test_path",
            **self._planned_path_spline_metadata(),
        )
        return self.test_path

    def build_straight_line(
        self,
        *,
        length_m: float = 10,
        point_count: int = 30,
        up_down_angle_deg: float = 0.0,
        left_right_angle_deg: float = 0.0,
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
        points = self.set_waypoints(waypoints).astype(float).tolist()
        self.test_path = PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=points,
            gate_ids=[],
            gate_center_errors_m={},
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="straight_line",
            **self._planned_path_spline_metadata(),
        )
        return self.test_path

    def build_circular_path(
        self,
        *,
        radius_m: float = 8.0,
        point_count: int = 65,
        clockwise: bool = False,
    ) -> PlannedPath:
        """Build a horizontal circular local-NED path that starts and ends at the origin.

        The circle is offset east by ``radius_m`` so the first waypoint is
        ``(0, 0, 0)`` and the final waypoint returns to ``(0, 0, 0)``. The
        initial tangent points north when ``clockwise`` is false and south when
        ``clockwise`` is true.
        """
        started = perf_counter()
        radius = float(radius_m)
        count = int(point_count)
        if radius <= 0.0:
            raise ValueError("radius_m must be positive")
        if count < 4:
            raise ValueError("point_count must be at least 4")

        angle = np.linspace(0.0, 2.0 * np.pi, count)
        direction = -1.0 if clockwise else 1.0
        north = radius * np.sin(direction * angle)
        east = radius * (1.0 - np.cos(direction * angle))
        down = np.zeros_like(angle)

        waypoints = np.column_stack((north, east, down))
        points = self.set_waypoints(waypoints).astype(float).tolist()
        self.test_path = PlannedPath(
            points_relative_ned_m=points,
            anchors_relative_ned_m=points,
            gate_ids=[],
            gate_center_errors_m={},
            gate_center_tolerance_m=float(self.gate_center_tolerance_m),
            spline_corner_tightness=float(self.spline_corner_tightness),
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
            source="circular_path",
            **self._planned_path_spline_metadata(),
        )
        return self.test_path

    def generate_spline(self) -> object:
        if self.spline_generator is None:
            raise NotImplementedError("Inject the chosen spline generator.")
        return self.spline_generator(self.get_waypoints())

    def get_waypoints(self) -> np.ndarray:
        return self._waypoints.copy()

    def project(self, position_local_ned_m: Vec3) -> PathProjection:
        """Project a query position onto the currently managed path.

        ``position_local_ned_m`` is the local-NED point to compare against the
        path. In normal flight-control usage this is the drone position from
        ``VehicleState.position_local_ned_m`` at the time ``project()`` is
        called, but the method accepts any local-NED query point.

        The path is treated as a piecewise-linear polyline between the current
        waypoint samples. Each segment is checked and the closest point on the
        full path is returned. Segment fractions are clamped to the segment
        endpoints, so query positions before the first waypoint or beyond the
        final waypoint project to the nearest endpoint.

        Returns:
            PathProjection: Typed projection result containing:
            - ``closest_point_local_ned_m``: closest point on the path.
            - ``tangent_local_ned``: unit tangent of the selected path segment.
            - ``along_track_m``: distance from path start to the projected point.
            - ``cross_track_error_m``: Euclidean distance from the query position to the path.
            - ``segment_index``: index of the selected segment start waypoint.
            - ``segment_fraction``: normalized position on that segment in ``[0, 1]``.

        Raises:
            ValueError: If no valid path has been set.
        """
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
        return PathProjection(
            closest_point_local_ned_m=tuple(float(value) for value in best_point),
            tangent_local_ned=tuple(float(value) for value in tangent),
            along_track_m=along_track_m,
            cross_track_error_m=float(np.sqrt(best_distance_sq)),
            segment_index=best_segment_index,
            segment_fraction=best_fraction,
        )

    def carrot_point(
        self,
        position_local_ned_m: Vec3,
        lookahead_m: float,
        speed_lookahead_m: float | None = None,
    ) -> dict[str, Any]:
        """Return a preview point ahead of the query position's path projection.

        ``position_local_ned_m`` is the local-NED query point used to find the
        current nearest point on the path. In normal flight-control usage this
        is the drone position from ``VehicleState.position_local_ned_m`` at the
        time ``carrot_point()`` is called.

        The method first calls ``project()`` to find the query point's current
        along-track distance. It then samples a target point ``lookahead_m``
        meters farther along the path, clamped to the final waypoint if the
        requested preview goes past the end of the path. ``speed_lookahead_m``
        controls the farther preview window used for reporting
        ``max_curvature_ahead``; when omitted, it defaults to ``lookahead_m``.

        Returns a dictionary containing:
        - ``position_local_ned_m``: preview/carrot point on the path.
        - ``tangent_local_ned``: unit tangent at the preview point.
        - ``along_track_m``: distance from path start to the preview point.
        - ``cross_track_error_m``: distance from the query position to the path.
        - ``curvature``: curvature estimate at the preview segment.
        - ``max_curvature_ahead``: max curvature between the projection and speed preview point.
        - ``speed_lookahead_m``: speed-preview distance used for curvature planning.
        - ``speed_preview_along_track_m``: along-track end of the speed preview window.
        - ``projection_closest_point_local_ned_m``: nearest point found by ``project()``.
        - ``projection_along_track_m``: along-track distance of that nearest point.
        - ``segment_index``: segment index used for the preview point.

        Raises:
            ValueError: If no valid path has been set.
        """
        projection = self.project(position_local_ned_m)
        projection_distance_m = float(projection.along_track_m)
        target_distance_m = min(
            projection_distance_m + max(0.0, float(lookahead_m)),
            float(self._cumulative_lengths[-1]),
        )
        speed_preview_distance_m = min(
            projection_distance_m
            + max(0.0, float(lookahead_m if speed_lookahead_m is None else speed_lookahead_m)),
            float(self._cumulative_lengths[-1]),
        )
        point, tangent, segment_index = self._sample_at_distance(target_distance_m)
        curvature = self._curvature_at_segment(segment_index)
        max_curvature_ahead = self._max_curvature_between(projection_distance_m, speed_preview_distance_m)
        return {
            "position_local_ned_m": tuple(float(value) for value in point),
            "tangent_local_ned": tuple(float(value) for value in tangent),
            "along_track_m": target_distance_m,
            "cross_track_error_m": projection.cross_track_error_m,
            "curvature": float(curvature),
            "max_curvature_ahead": float(max_curvature_ahead),
            "speed_lookahead_m": float(
                lookahead_m if speed_lookahead_m is None else speed_lookahead_m
            ),
            "speed_preview_along_track_m": speed_preview_distance_m,
            "projection_closest_point_local_ned_m": projection.closest_point_local_ned_m,
            "projection_along_track_m": projection.along_track_m,
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

    def _curvature_at_segment(self, index: int) -> float:
        self._require_path()
        segment_index = min(max(int(index), 0), len(self._segment_lengths) - 1)
        return max(
            self._curvature_at_vertex(segment_index),
            self._curvature_at_vertex(segment_index + 1),
        )

    def _max_curvature_between(self, start_distance_m: float, end_distance_m: float) -> float:
        self._require_path()
        start = float(np.clip(start_distance_m, 0.0, self._cumulative_lengths[-1]))
        end = float(np.clip(end_distance_m, 0.0, self._cumulative_lengths[-1]))
        if end < start:
            start, end = end, start

        start_index = int(np.searchsorted(self._cumulative_lengths, start, side="right") - 1)
        end_index = int(np.searchsorted(self._cumulative_lengths, end, side="right") - 1)
        start_index = min(max(start_index, 0), len(self._segment_lengths) - 1)
        end_index = min(max(end_index, 0), len(self._segment_lengths) - 1)

        max_curvature = 0.0
        for vertex_index in range(start_index, end_index + 2):
            max_curvature = max(max_curvature, self._curvature_at_vertex(vertex_index))
        return float(max_curvature)

    def _curvature_at_vertex(self, index: int) -> float:
        if index <= 0 or index >= len(self._waypoints) - 1:
            return 0.0

        before = self._waypoints[index] - self._waypoints[index - 1]
        after = self._waypoints[index + 1] - self._waypoints[index]
        before_length = float(np.linalg.norm(before))
        after_length = float(np.linalg.norm(after))
        if before_length <= 1e-12 or after_length <= 1e-12:
            return 0.0

        before_tangent = before / before_length
        after_tangent = after / after_length
        turn_angle_rad = math.acos(float(np.clip(np.dot(before_tangent, after_tangent), -1.0, 1.0)))
        arc_length_m = 0.5 * (before_length + after_length)
        if arc_length_m <= 1e-12:
            return 0.0
        return float(turn_angle_rad / arc_length_m)

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
        turn_tightness = self._anchor_turn_tightness(anchors)
        tangents = self._anchor_tangents(anchors, turn_tightness)
        for index in range(len(anchors) - 1):
            p1 = anchors[index]
            p2 = anchors[index + 1]
            m1 = tangents[index]
            m2 = tangents[index + 1]
            distance = float(np.linalg.norm(p2 - p1))
            steps = max(1, int(np.ceil(distance / self.spacing_m)))
            for step in range(1, steps + 1):
                t = step / steps
                t2 = t * t
                t3 = t2 * t
                point = (
                    (2.0 * t3 - 3.0 * t2 + 1.0) * p1
                    + (t3 - 2.0 * t2 + t) * m1
                    + (-2.0 * t3 + 3.0 * t2) * p2
                    + (t3 - t2) * m2
                )
                samples.append(point)
                if len(samples) >= self.max_points:
                    return np.asarray(samples, dtype=float).tolist()
        return np.asarray(samples, dtype=float).tolist()

    def _anchor_tangents(self, anchors: np.ndarray, anchor_tightness: np.ndarray) -> np.ndarray:
        tangents = np.zeros_like(anchors, dtype=float)
        for index in range(len(anchors)):
            if index == 0:
                raw_tangent = anchors[1] - anchors[0]
            elif index == len(anchors) - 1:
                raw_tangent = anchors[-1] - anchors[-2]
            else:
                before = anchors[index] - anchors[index - 1]
                after = anchors[index + 1] - anchors[index]
                before_length = float(np.linalg.norm(before))
                after_length = float(np.linalg.norm(after))
                if before_length <= 1e-12 or after_length <= 1e-12:
                    continue
                raw_tangent = before / before_length + after / after_length
                raw_norm = float(np.linalg.norm(raw_tangent))
                if raw_norm <= 1e-12:
                    continue
                local_distance = min(before_length, after_length)
                raw_tangent = raw_tangent / raw_norm * local_distance

            tangent_norm = float(np.linalg.norm(raw_tangent))
            if tangent_norm <= 1e-12:
                continue

            tightness = float(np.clip(anchor_tightness[index], 0.0, 1.0))
            tangent_scale = 0.75 - 0.55 * tightness
            tangents[index] = raw_tangent * tangent_scale
        return tangents

    def _planned_path_spline_metadata(self) -> dict[str, Any]:
        return {
            "adaptive_spline_tightness": bool(self.adaptive_spline_tightness),
            "distant_spline_corner_tightness": float(self.distant_spline_corner_tightness),
            "min_spline_corner_tightness": float(self.min_spline_corner_tightness),
            "max_spline_corner_tightness": float(self.max_spline_corner_tightness),
            "gentle_turn_angle_deg": float(math.degrees(self.gentle_turn_angle_rad)),
            "sharp_turn_angle_deg": float(math.degrees(self.sharp_turn_angle_rad)),
            "short_segment_reference_m": float(self.short_segment_reference_m),
            "long_segment_reference_m": float(self.long_segment_reference_m),
        }

    def _anchor_turn_tightness(self, anchors: np.ndarray) -> np.ndarray:
        tightness = np.full(len(anchors), self.spline_corner_tightness, dtype=float)
        if not self.adaptive_spline_tightness or len(anchors) < 3:
            return tightness

        angle_span = self.sharp_turn_angle_rad - self.gentle_turn_angle_rad
        for index in range(1, len(anchors) - 1):
            before = anchors[index] - anchors[index - 1]
            after = anchors[index + 1] - anchors[index]
            before_length = float(np.linalg.norm(before))
            after_length = float(np.linalg.norm(after))
            if before_length <= 1e-12 or after_length <= 1e-12:
                continue

            before_tangent = before / before_length
            after_tangent = after / after_length
            turn_angle = math.acos(float(np.clip(np.dot(before_tangent, after_tangent), -1.0, 1.0)))
            if angle_span <= 1e-12:
                angle_weight = 1.0 if turn_angle >= self.sharp_turn_angle_rad else 0.0
            else:
                angle_weight = float(np.clip(
                    (turn_angle - self.gentle_turn_angle_rad) / angle_span,
                    0.0,
                    1.0,
                ))
            angle_weight = angle_weight * angle_weight * (3.0 - 2.0 * angle_weight)

            local_spacing_m = min(before_length, after_length)
            if self.long_segment_reference_m <= self.short_segment_reference_m:
                distance_weight = 1.0 if local_spacing_m <= self.short_segment_reference_m else 0.0
            else:
                distance_weight = float(np.clip(
                    (self.long_segment_reference_m - local_spacing_m)
                    / (self.long_segment_reference_m - self.short_segment_reference_m),
                    0.0,
                    1.0,
                ))
            distance_weight = distance_weight * distance_weight * (3.0 - 2.0 * distance_weight)

            close_turn_weight = distance_weight * (0.55 + 0.45 * angle_weight)
            far_sharp_weight = 0.25 * angle_weight * (1.0 - distance_weight)
            weight = float(np.clip(close_turn_weight + far_sharp_weight, 0.0, 1.0))
            lower_tightness = (
                self.distant_spline_corner_tightness
                + distance_weight
                * (self.min_spline_corner_tightness - self.distant_spline_corner_tightness)
            )
            tightness[index] = (
                lower_tightness
                + weight * (self.max_spline_corner_tightness - lower_tightness)
            )

        return tightness

    def _segment_spline_tightness(self, anchor_tightness: np.ndarray, segment_index: int) -> float:
        if not self.adaptive_spline_tightness:
            return float(self.spline_corner_tightness)
        start_tightness = anchor_tightness[segment_index]
        end_tightness = anchor_tightness[segment_index + 1]
        return float(max(start_tightness, end_tightness))

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

    def _terminal_gate_tangent(self, anchors: np.ndarray) -> np.ndarray | None:
        if len(anchors) >= 3:
            tangent = anchors[-1] - anchors[-2]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1e-12:
                self._last_terminal_gate_tangent = tangent / tangent_norm
                return self._last_terminal_gate_tangent.copy()
        if self._last_terminal_gate_tangent is not None:
            return self._last_terminal_gate_tangent.copy()
        return None

    def _extend_path(self, points: np.ndarray, *, tail_tangent: np.ndarray | None = None) -> np.ndarray:
        if len(points) < 2 or self._path_tail_length_m <= 1e-9:
            return points.astype(float)

        tangent = tail_tangent if tail_tangent is not None else points[-1] - points[-2]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            return points.astype(float)
        tangent = tangent / tangent_norm

        remaining_slots = self.max_points - len(points)
        if remaining_slots <= 0:
            return points.astype(float)

        steps = min(remaining_slots, max(1, int(np.ceil(self._path_tail_length_m / self.spacing_m))))
        distances = np.linspace(self._path_tail_length_m / steps, self._path_tail_length_m, steps)
        tail = points[-1] + distances[:, np.newaxis] * tangent
        return np.vstack((points, tail)).astype(float)

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
    if mode in {"gate_map", "center_targets", "test_path"}:
        return mode
    raise ValueError("planning_mode must be 'gate_map', 'center_targets', or 'test_path'")
