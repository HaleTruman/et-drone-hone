import math
from collections.abc import Iterable
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


@dataclass(frozen=True)
class PathCarrot:
    position_local_ned_m: Vec3
    tangent_local_ned: Vec3
    along_track_m: float
    cross_track_error_m: float
    cross_track_error_local_ned_m: Vec3
    curvature: float
    max_curvature_ahead: float
    speed_lookahead_m: float
    speed_preview_along_track_m: float
    projection_closest_point_local_ned_m: Vec3
    projection_tangent_local_ned: Vec3
    projection_along_track_m: float
    projection_segment_index: int
    projection_segment_fraction: float
    segment_index: int

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class PathManager:
    def __init__(
        self,
        *,
        spacing_m: float = 0.75,
        max_points: int = 240,
        spline_corner_tightness: float = 0.5,
        adaptive_spline_tightness: bool = True,
        distant_spline_corner_tightness: float = 0.10,
        min_spline_corner_tightness: float = 0.15,
        max_spline_corner_tightness: float = 0.9,
        gentle_turn_angle_deg: float = 20.0,
        sharp_turn_angle_deg: float = 80.0,
        short_segment_reference_m: float = 12.0,
        long_segment_reference_m: float = 25.0,
        planning_mode: str = "gate",
        path_update_mode: str = "original",
        path_crossed_gate_persist_distance_m: float = 5.0,
        path_crossed_gate_curvature_preserve_m: float = 3.0,
        path_tail_length_m: float = 10.0,
        path_splice_lookahead_gain_s: float = 0.0,
    ):
        self.spacing_m = max(0.1, float(spacing_m))
        self.max_points = max(2, int(max_points))
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
        self.path_update_mode = _normalize_path_update_mode(path_update_mode)
        self._path_crossed_gate_persist_distance_m = max(
            0.0,
            float(path_crossed_gate_persist_distance_m),
        )
        self._path_crossed_gate_curvature_preserve_m = max(
            0.0,
            float(path_crossed_gate_curvature_preserve_m),
        )
        self._path_tail_length_m = max(0.0, float(path_tail_length_m))
        self._path_splice_lookahead_gain_s = max(0.0, float(path_splice_lookahead_gain_s))
        self._minimum_splice_segment_m = max(1e-3, 0.05 * self.spacing_m)
        self._splice_bridge_min_length_m = max(12.0 * self.spacing_m, 3.0)
        self._last_terminal_gate_tangent: np.ndarray | None = None
        self._waypoints = np.empty((0, 3))
        self._segment_lengths = np.empty((0,))
        self._cumulative_lengths = np.array([0.0], dtype=float)
        self.test_path: PlannedPath | None = None


    def plan(
        self,
        *,
        gates: Iterable[GateRecord],
        vehicle_state: VehicleState,
    ) -> PlannedPath:
        position_local_ned_m = vehicle_state.position_local_ned_m

        if self.planning_mode == "test":
            if self.test_path is None:
                raise ValueError("test planning mode requires PathManager.test_path.")
            self.set_waypoints(self.test_path.points_relative_ned_m)
            return self.test_path

        return self.plan_from_gate_centers(
            gates,
            position_local_ned_m=position_local_ned_m,
            velocity_local_ned_mps=vehicle_state.velocity_local_ned_mps,
        )

    
    def plan_from_gate_centers(
        self,
        gates: Iterable[GateRecord],
        *,
        position_local_ned_m: Vec3,
        velocity_local_ned_mps: Vec3 | None = None,
    ) -> PlannedPath:
        started = perf_counter()
        position = np.asarray(position_local_ned_m, dtype=float)
        gate_records = list(gates)
        start_anchors = self._planning_start_anchors(position, gate_records)
        start_position = start_anchors[0]
        gate_centers = [
            (
                gate,
                np.asarray(gate.position_local_ned_m, dtype=float),
            )
            for gate in gate_records
            if not gate.crossed
        ]
        gate_centers.sort(key=lambda item: float(np.linalg.norm(item[1] - start_position)))
        planned_gates = [gate for gate, _ in gate_centers]
        centers = [center for _, center in gate_centers]

        anchors = self._dedupe_points(np.asarray([*start_anchors, *centers], dtype=float))
        points = self._extend_path(
            _waypoint_array(self._sample_spline(anchors)),
            tail_tangent=self._terminal_gate_tangent(anchors),
        )
        points = self._splice_planned_points(
            points,
            position_local_ned_m=position,
            velocity_local_ned_mps=velocity_local_ned_mps,
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
            computation_ms=(perf_counter() - started) * 1000.0,
            source="gate",
        )


    def _planning_start_anchors(
        self,
        position: np.ndarray,
        gates: list[GateRecord],
    ) -> list[np.ndarray]:
        if self.path_update_mode == "persist_crossed":
            persisted_gate = self._persisted_crossed_gate(position, gates)
            if persisted_gate is not None:
                return self._persisted_crossed_gate_anchors(persisted_gate)
            return [position.astype(float)]

        if self.path_update_mode != "projected":
            return [position.astype(float)]
        if len(self._waypoints) < 2 or float(self._cumulative_lengths[-1]) <= 1e-12:
            return [position.astype(float)]
        return [np.asarray(self.project(position).closest_point_local_ned_m, dtype=float)]

    def _persisted_crossed_gate(
        self,
        position: np.ndarray,
        gates: list[GateRecord],
    ) -> GateRecord | None:
        crossed_gates = [
            gate
            for gate in gates
            if gate.crossed
            and float(np.linalg.norm(np.asarray(gate.position_local_ned_m, dtype=float) - position))
            <= self._path_crossed_gate_persist_distance_m
        ]
        if not crossed_gates:
            return None
        return min(
            crossed_gates,
            key=lambda gate: float(
                np.linalg.norm(np.asarray(gate.position_local_ned_m, dtype=float) - position)
            ),
        )

    def _persisted_crossed_gate_anchors(self, gate: GateRecord) -> list[np.ndarray]:
        gate_position = np.asarray(gate.position_local_ned_m, dtype=float)
        if len(self._waypoints) < 3 or float(self._cumulative_lengths[-1]) <= 1e-12:
            return [gate_position]

        projection = self._nearest_projection_on_points(self._waypoints, gate_position)
        segment_index, segment_fraction, projected_point, _ = projection
        start_distance = float(
            self._cumulative_lengths[segment_index]
            + segment_fraction * self._segment_lengths[segment_index]
        )
        preserve_distance = min(
            self._path_crossed_gate_curvature_preserve_m,
            float(self._cumulative_lengths[-1] - start_distance),
        )
        if preserve_distance <= self._minimum_splice_segment_m:
            return [projected_point]

        mid_distance = start_distance + 0.5 * preserve_distance
        end_distance = start_distance + preserve_distance
        mid_point, _, _ = self._sample_at_distance(mid_distance)
        end_point, _, _ = self._sample_at_distance(end_distance)
        return [
            point
            for point in self._dedupe_points(
                np.asarray([projected_point, mid_point, end_point], dtype=float)
            )
        ]

    def set_waypoints(self, waypoints: Iterable[Vec3]) -> np.ndarray:
        self._waypoints = _waypoint_array(waypoints)
        self._refresh_lengths()
        return self.get_waypoints()

    def build_test_path(
        self,
        *,
        gate_points_local_ned_m: Iterable[Vec3] | None = None,
        length_m: float = 30.0,
        width_m: float = 8.0,
        height_m: float = 1.0,
        point_count: int = 31,
    ) -> PlannedPath:
        started = perf_counter()

        if gate_points_local_ned_m is not None:
            gates = [
                GateRecord(
                    gate_id=f"test-gate-{index + 1:03d}",
                    position_local_ned_m=tuple(float(value) for value in point),
                    quaternion=None,
                    position_confidence=1.0,
                    quaternion_confidence=None,
                    sequence=index,
                    source="test",
                )
                for index, point in enumerate(gate_points_local_ned_m)
            ]
            planned_path = self.plan_from_gate_centers(
                gates,
                position_local_ned_m=(0.0, 0.0, 0.0),
                velocity_local_ned_mps=None,
            )
            self.test_path = PlannedPath(
                points_relative_ned_m=planned_path.points_relative_ned_m,
                anchors_relative_ned_m=planned_path.anchors_relative_ned_m,
                gate_ids=planned_path.gate_ids,
                gate_center_errors_m=planned_path.gate_center_errors_m,
                computation_ms=(perf_counter() - started) * 1000.0,
                source="test",
            )
            self.set_waypoints(self.test_path.points_relative_ned_m)
            return self.test_path

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
            computation_ms=(perf_counter() - started) * 1000.0,
            source="test",
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
            computation_ms=(perf_counter() - started) * 1000.0,
            source="straight_line",
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
            computation_ms=(perf_counter() - started) * 1000.0,
            source="circular_path",
        )
        return self.test_path

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
        position = np.asarray(position_local_ned_m, dtype=float)

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

    def carrot(
        self,
        position_local_ned_m: Vec3,
        lookahead_m: float,
        speed_lookahead_m: float | None = None,
    ) -> PathCarrot:
        """Return a preview point ahead of the query position's path projection.

        ``position_local_ned_m`` is the local-NED query point used to find the
        current nearest point on the path. In normal flight-control usage this
        is the drone position from ``VehicleState.position_local_ned_m`` at the
        time ``carrot()`` is called.

        The method first calls ``project()`` to find the query point's current
        along-track distance. It then samples a target point ``lookahead_m``
        meters farther along the path, clamped to the final waypoint if the
        requested preview goes past the end of the path. ``speed_lookahead_m``
        controls the farther preview window used for reporting
        ``max_curvature_ahead``; when omitted, it defaults to ``lookahead_m``.

        Returns a typed preview containing:
        - ``position_local_ned_m``: preview/carrot point on the path.
        - ``tangent_local_ned``: unit tangent at the preview point.
        - ``along_track_m``: distance from path start to the preview point.
        - ``cross_track_error_m``: distance from the query position to the path.
        - ``cross_track_error_local_ned_m``: perpendicular error vector from the path to the query position.
        - ``curvature``: curvature estimate at the preview segment.
        - ``max_curvature_ahead``: max curvature between the projection and speed preview point.
        - ``speed_lookahead_m``: speed-preview distance used for curvature planning.
        - ``speed_preview_along_track_m``: along-track end of the speed preview window.
        - ``projection_closest_point_local_ned_m``: nearest point found by ``project()``.
        - ``projection_tangent_local_ned``: unit path tangent at the nearest point.
        - ``projection_along_track_m``: along-track distance of that nearest point.
        - ``projection_segment_index``: projected path segment index.
        - ``projection_segment_fraction``: fractional position on the projected segment.
        - ``segment_index``: segment index used for the preview point.

        Raises:
            ValueError: If no valid path has been set.
        """
        projection = self.project(position_local_ned_m)
        position = np.asarray(position_local_ned_m, dtype=float)
        projection_closest = np.asarray(projection.closest_point_local_ned_m, dtype=float)
        projection_tangent = np.asarray(projection.tangent_local_ned, dtype=float)
        position_error = position - projection_closest
        cross_track_error = position_error - projection_tangent * float(
            np.dot(position_error, projection_tangent)
        )
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
        return PathCarrot(
            position_local_ned_m=tuple(float(value) for value in point),
            tangent_local_ned=tuple(float(value) for value in tangent),
            along_track_m=target_distance_m,
            cross_track_error_m=projection.cross_track_error_m,
            cross_track_error_local_ned_m=tuple(float(value) for value in cross_track_error),
            curvature=float(curvature),
            max_curvature_ahead=float(max_curvature_ahead),
            speed_lookahead_m=float(
                lookahead_m if speed_lookahead_m is None else speed_lookahead_m
            ),
            speed_preview_along_track_m=speed_preview_distance_m,
            projection_closest_point_local_ned_m=projection.closest_point_local_ned_m,
            projection_tangent_local_ned=projection.tangent_local_ned,
            projection_along_track_m=projection.along_track_m,
            projection_segment_index=projection.segment_index,
            projection_segment_fraction=projection.segment_fraction,
            segment_index=segment_index,
        )

    def _splice_planned_points(
        self,
        points: np.ndarray,
        *,
        position_local_ned_m: Vec3 | np.ndarray | None,
        velocity_local_ned_mps: Vec3 | None,
    ) -> np.ndarray:
        new_points = _waypoint_array(points)
        if self.path_update_mode in {"original", "projected", "persist_crossed"}:
            return new_points.astype(float)
        if len(new_points) < 2 or position_local_ned_m is None or velocity_local_ned_mps is None:
            return new_points.astype(float)
        if len(self._waypoints) < 2 or float(self._cumulative_lengths[-1]) <= 1e-12:
            return new_points.astype(float)

        position = np.asarray(position_local_ned_m, dtype=float)
        speed_mps = float(np.linalg.norm(np.asarray(velocity_local_ned_mps, dtype=float)))
        splice_distance_m = min(
            self.project(position).along_track_m
            + self._path_splice_lookahead_gain_s * speed_mps,
            float(self._cumulative_lengths[-1]),
        )
        splice_point, splice_tangent, _ = self._sample_at_distance(splice_distance_m)
        splice_curvature = self._curvature_vector_at_distance(splice_distance_m)
        old_prefix = self._active_path_prefix_through(splice_distance_m, splice_point)
        new_suffix = self._planned_suffix_after_splice(
            new_points,
            splice_point,
            splice_tangent,
            splice_curvature,
        )
        spliced = self._dedupe_points(np.vstack((old_prefix, new_suffix)))
        spliced = self._clean_spliced_points(spliced)
        if len(spliced) > self.max_points:
            return spliced[: self.max_points].astype(float)
        return spliced.astype(float)

    def _active_path_prefix_through(self, distance_m: float, point: np.ndarray) -> np.ndarray:
        index = int(np.searchsorted(self._cumulative_lengths, distance_m, side="right") - 1)
        index = min(max(index, 0), len(self._segment_lengths) - 1)
        prefix = [*self._waypoints[: index + 1], point]
        return self._dedupe_points(np.asarray(prefix, dtype=float))

    def _planned_suffix_after_splice(
        self,
        points: np.ndarray,
        splice_point: np.ndarray,
        splice_tangent: np.ndarray,
        splice_curvature: np.ndarray,
    ) -> np.ndarray:
        lengths, cumulative = self._polyline_lengths(points)
        (
            segment_index,
            segment_fraction,
            projected_point,
            projected_tangent,
        ) = self._nearest_projection_on_points(points, splice_point)
        projected_distance = float(
            cumulative[segment_index]
            + segment_fraction * lengths[segment_index]
        )
        turn_angle = self._tangent_turn_angle(
            splice_tangent,
            projected_tangent,
        )
        lateral_join_offset_m = float(np.linalg.norm(projected_point - splice_point))
        blend_distance = max(
            self._splice_bridge_min_length_m,
            2.5 * lateral_join_offset_m,
            8.0 * self.spacing_m * (1.0 + turn_angle / math.pi),
        )
        attach_distance = min(projected_distance + blend_distance, float(cumulative[-1]))
        attach_point, attach_tangent, attach_segment = self._sample_points_at_distance(
            points,
            cumulative,
            attach_distance,
        )
        attach_curvature = self._curvature_vector_on_points(points, attach_segment)

        suffix = self._curvature_bridge(
            splice_point,
            splice_tangent,
            splice_curvature,
            attach_point,
            attach_tangent,
            attach_curvature,
        ).tolist()
        suffix.extend(self._points_after_distance(points, cumulative, attach_distance).tolist())
        return self._dedupe_points(np.asarray(suffix, dtype=float))

    @staticmethod
    def _nearest_projection_on_points(
        points: np.ndarray,
        target: np.ndarray,
    ) -> tuple[int, float, np.ndarray, np.ndarray]:
        best_distance_sq = float("inf")
        best_segment_index = 0
        best_fraction = 0.0
        best_point = points[0]
        best_tangent = np.array((1.0, 0.0, 0.0), dtype=float)
        for index, segment in enumerate(np.diff(points, axis=0)):
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-12:
                fraction = 0.0
                tangent = np.array((1.0, 0.0, 0.0), dtype=float)
            else:
                fraction = float(
                    np.clip(
                        np.dot(target - points[index], segment) / length_sq,
                        0.0,
                        1.0,
                    )
                )
                tangent = segment / math.sqrt(length_sq)
            projected = points[index] + fraction * segment
            distance_sq = float(np.dot(target - projected, target - projected))
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_segment_index = index
                best_fraction = fraction
                best_point = projected
                best_tangent = tangent
        return best_segment_index, best_fraction, best_point, best_tangent

    def _curvature_bridge(
        self,
        start: np.ndarray,
        start_tangent: np.ndarray,
        start_curvature: np.ndarray,
        end: np.ndarray,
        end_tangent: np.ndarray,
        end_curvature: np.ndarray,
    ) -> np.ndarray:
        chord = end - start
        chord_distance = float(np.linalg.norm(chord))
        if chord_distance <= 1e-9:
            return np.asarray([start], dtype=float)

        start_tangent = self._unit_or_default(start_tangent, default=chord)
        end_tangent = self._unit_or_default(end_tangent, default=chord)
        turn_angle = self._tangent_turn_angle(start_tangent, end_tangent)
        curve_length = max(
            chord_distance,
            self._splice_bridge_min_length_m,
            4.0 * self.spacing_m,
            2.0 * self.spacing_m * (1.0 + turn_angle / math.pi),
        )
        start_derivative = start_tangent * curve_length
        end_derivative = end_tangent * curve_length
        start_second = np.asarray(start_curvature, dtype=float) * curve_length * curve_length
        end_second = np.asarray(end_curvature, dtype=float) * curve_length * curve_length

        samples: list[np.ndarray] = []
        steps = max(8, int(np.ceil(curve_length / (0.5 * self.spacing_m))))
        for step in range(0, steps + 1):
            t = step / steps
            point = self._quintic_hermite(
                t,
                start,
                start_derivative,
                start_second,
                end,
                end_derivative,
                end_second,
            )
            samples.append(point)
        return self._resample_polyline(
            self._drop_short_segments(np.asarray(samples, dtype=float)),
            spacing_m=0.5 * self.spacing_m,
            include_end=True,
        )

    @staticmethod
    def _quintic_hermite(
        t: float,
        p0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        p1: np.ndarray,
        v1: np.ndarray,
        a1: np.ndarray,
    ) -> np.ndarray:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
        h10 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
        h20 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
        h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
        h11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
        h21 = 0.5 * t3 - t4 + 0.5 * t5
        return h00 * p0 + h10 * v0 + h20 * a0 + h01 * p1 + h11 * v1 + h21 * a1

    @staticmethod
    def _tangent_turn_angle(first: np.ndarray, second: np.ndarray) -> float:
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= 1e-12 or second_norm <= 1e-12:
            return 0.0
        return math.acos(
            float(
                np.clip(
                    np.dot(first, second) / (first_norm * second_norm),
                    -1.0,
                    1.0,
                )
            )
        )

    @staticmethod
    def _polyline_lengths(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        return lengths, cumulative

    def _sample_points_at_distance(
        self,
        points: np.ndarray,
        cumulative: np.ndarray,
        distance_m: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if len(points) < 2:
            raise ValueError("At least two points are required for path sampling.")
        distance = float(np.clip(distance_m, 0.0, cumulative[-1]))
        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        index = min(max(index, 0), len(points) - 2)
        segment = points[index + 1] - points[index]
        length = float(np.linalg.norm(segment))
        fraction = 0.0 if length <= 1e-12 else (distance - cumulative[index]) / length
        point = points[index] + fraction * segment
        tangent = self._unit_or_default(segment, default=np.array((1.0, 0.0, 0.0), dtype=float))
        return point, tangent, index

    def _points_after_distance(
        self,
        points: np.ndarray,
        cumulative: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        if len(points) < 2:
            return points.astype(float)
        point, _, index = self._sample_points_at_distance(points, cumulative, distance_m)
        tail = [point]
        next_index = index + 1
        if (
            next_index < len(points)
            and float(np.linalg.norm(points[next_index] - point)) <= self._minimum_splice_segment_m
        ):
            next_index += 1
        if next_index < len(points):
            tail.extend(points[next_index:])
        return self._dedupe_points(np.asarray(tail, dtype=float))

    def _curvature_vector_at_distance(self, distance_m: float) -> np.ndarray:
        self._require_path()
        distance = float(np.clip(distance_m, 0.0, self._cumulative_lengths[-1]))
        index = int(np.searchsorted(self._cumulative_lengths, distance, side="right") - 1)
        index = min(max(index, 0), len(self._segment_lengths) - 1)
        return self._curvature_vector_on_points(self._waypoints, index)

    def _curvature_vector_on_points(self, points: np.ndarray, segment_index: int) -> np.ndarray:
        if len(points) < 3:
            return np.zeros(3, dtype=float)
        vertex_index = min(max(int(segment_index) + 1, 1), len(points) - 2)
        before = points[vertex_index] - points[vertex_index - 1]
        after = points[vertex_index + 1] - points[vertex_index]
        before_length = float(np.linalg.norm(before))
        after_length = float(np.linalg.norm(after))
        if before_length <= 1e-12 or after_length <= 1e-12:
            return np.zeros(3, dtype=float)
        before_tangent = before / before_length
        after_tangent = after / after_length
        arc_length = 0.5 * (before_length + after_length)
        if arc_length <= 1e-12:
            return np.zeros(3, dtype=float)
        return (after_tangent - before_tangent) / arc_length

    def _clean_spliced_points(self, points: np.ndarray) -> np.ndarray:
        cleaned = self._drop_short_segments(points)
        if len(cleaned) < 2:
            return cleaned.astype(float)
        return self._resample_polyline(cleaned, spacing_m=self.spacing_m, include_end=True)

    def _drop_short_segments(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return points.astype(float)
        kept = [points[0]]
        for point in points[1:-1]:
            if float(np.linalg.norm(point - kept[-1])) >= self._minimum_splice_segment_m:
                kept.append(point)
        if float(np.linalg.norm(points[-1] - kept[-1])) >= 1e-9:
            kept.append(points[-1])
        return np.asarray(kept, dtype=float)

    def _resample_polyline(
        self,
        points: np.ndarray,
        *,
        spacing_m: float,
        include_end: bool,
    ) -> np.ndarray:
        points = self._drop_short_segments(points)
        if len(points) <= 1:
            return points.astype(float)
        lengths, cumulative = self._polyline_lengths(points)
        total_length = float(cumulative[-1])
        if total_length <= 1e-12:
            return points[:1].astype(float)

        spacing = max(self._minimum_splice_segment_m, float(spacing_m))
        sample_count = max(1, int(np.floor(total_length / spacing)))
        distances = [index * spacing for index in range(sample_count + 1)]
        if (
            include_end
            and (not distances or total_length - distances[-1] > self._minimum_splice_segment_m)
        ):
            distances.append(total_length)
        elif include_end:
            distances[-1] = total_length

        samples = [
            self._sample_points_at_distance(points, cumulative, distance)[0]
            for distance in distances
        ]
        return self._dedupe_points(np.asarray(samples, dtype=float))

    @staticmethod
    def _unit_or_default(vector: np.ndarray, *, default: np.ndarray) -> np.ndarray:
        candidate = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-12:
            candidate = np.asarray(default, dtype=float)
            norm = float(np.linalg.norm(candidate))
        if norm <= 1e-12:
            return np.array((1.0, 0.0, 0.0), dtype=float)
        return candidate / norm

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

    @staticmethod
    def _minimum_distance(points: list[list[float]], target: np.ndarray) -> float:
        if not points:
            return float("inf")
        return float(np.min(np.linalg.norm(_waypoint_array(points) - target, axis=1)))

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


def _normalize_planning_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"gate", "test"}:
        return mode
    raise ValueError("planning_mode must be 'gate' or 'test'")


def _normalize_path_update_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"original", "projected", "persist_crossed", "splice"}:
        return mode
    raise ValueError(
        "path_update_mode must be 'original', 'projected', 'persist_crossed', or 'splice'"
    )
