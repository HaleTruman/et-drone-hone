"""Estimate persistent physical gate landmarks in a fixed local-NED frame.

Upstream ``track_id`` values identify temporary evidence, never physical gates. Qualified
camera rays are globally associated with persistent landmarks, retained in bounded rolling
pools, and solved again from the retained evidence after every update. Published
``VisionGateObservation`` positions are absolute local-NED landmark coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import camera_odometry as codo
from camera_odometry import MAX_TELEMETRY_GAP_S
from schema import VehicleState, VisionGateObservation, VisionObservation, VoidAnalysis


# Camera model and mounting transform.
FX, FY, CX, CY = 320.0, 320.0, 320.0, 180.0  # duplicated from camera_odometry.py
CAMERA_TILT_DEG = 20.0  # duplicated from camera_odometry.py

# Monocular depth prior. It regularizes weak ray geometry but never replaces retained rays.
ANCHOR_PX, ANCHOR_M = 4000.0, 10.0
_OTHER_PX, _OTHER_M = 40.0, 40.0
DEPTH_PRIOR_POWER = math.log(_OTHER_M / ANCHOR_M) / math.log(_OTHER_PX / ANCHOR_PX)
PRIOR_WEIGHT = 0.05

# Centralized calibration surface.
MIN_VOID_OBSERVATION_CONFIDENCE = 0.55
ROLLING_POOL_SIZE = 5
MIN_ROLLING_POOL_SIZE = 2
MIN_GATE_SEPARATION_M = 5.0
SPATIAL_ASSOCIATION_MAX_M = 4.0
RAY_RESIDUAL_MAX_M = 2.5
ASSOCIATION_AMBIGUITY_MARGIN = 0.10  # normalized association-cost margin
ROBUST_LOSS_SCALE_M = 1.5
MIN_INLIER_OBSERVATIONS = 2
GLOBAL_SOLVER_MAX_ITERATIONS = 5
LANDMARK_MAX_MISSED_FRAMES = 120
LANDMARK_MERGE_MAX_M = 2.25
MIN_MERGE_OBSERVATIONS = 2
MAX_ACTIVE_LANDMARKS = ROLLING_POOL_SIZE * (LANDMARK_MAX_MISSED_FRAMES + 1)
POSITION_CONVERGENCE_M = 1e-4
NUMERICAL_EPS = 1e-9

# Retained for compatibility with the module's focused numerical unit tests.
SPREAD_SATURATION_DEG = 15.0
RAY_COUNT_TIERS = ((8, 0.9), (3, 0.6), (1, 0.3))


def _pitch_matrix(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Camera CV (X=right, Y=down, Z=forward) -> body FRD, including the fixed mounting tilt.
_BASE_BODY_FROM_CAM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_BODY_FROM_CAM = _pitch_matrix(CAMERA_TILT_DEG) @ _BASE_BODY_FROM_CAM


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > NUMERICAL_EPS else v


def _camera_ray(center: tuple[float, float]) -> np.ndarray:
    u, v = center
    return _normalize(np.array([(u - CX) / FX, (v - CY) / FY, 1.0]))


def _expected_distance(pixel_count: float) -> float:
    return ANCHOR_M * (max(pixel_count, 1.0) / ANCHOR_PX) ** DEPTH_PRIOR_POWER


def _frame_identity(frame_path: str) -> tuple[int, int]:
    parts = Path(frame_path).stem.split("-")
    return int(parts[1]), int(parts[-1])


def _projector(direction: np.ndarray) -> np.ndarray:
    return np.eye(3) - np.outer(direction, direction)


def _ray_residual(position: np.ndarray, camera_position: np.ndarray,
                  direction: np.ndarray) -> float:
    return float(np.linalg.norm(_projector(direction) @ (position - camera_position)))


def _ray_depth(position: np.ndarray, camera_position: np.ndarray,
               direction: np.ndarray) -> float:
    return float(np.dot(position - camera_position, direction))


def _new_entry() -> dict:
    """Compatibility fixture for existing low-level numerical tests.

    Production estimator state is represented by ``_GlobalLandmark`` and never uses this
    permanent accumulator.
    """
    return {
        "A": np.zeros((3, 3)), "b": np.zeros(3), "direction_sum": np.zeros(3), "min_dot": 1.0,
        "ray_count": 0, "last_pixel_count": None, "position": None, "passed": False,
    }


@dataclass
class _RayObservation:
    source_track_id: str
    source: str
    frame_id: int
    sim_time_ns: int
    camera_position_ned: np.ndarray
    direction_ned: np.ndarray
    confidence: float
    pixel_count: float
    center: tuple[float, float]
    provisional_position_ned: np.ndarray
    residual_m: float = math.inf
    inlier: bool = False


@dataclass
class _Solution:
    position_ned: np.ndarray
    residuals_m: list[float]
    inlier_indices: tuple[int, ...]
    rms_residual_m: float
    confidence: float


@dataclass
class _GlobalLandmark:
    gate_id: str
    observations: list[_RayObservation]
    position_ned: np.ndarray
    residual_rms_m: float = math.inf
    position_confidence: float = 0.0
    missed_frames: int = 0
    passed: bool = False
    source_track_ids: list[str] = field(default_factory=list)
    merged_gate_ids: list[str] = field(default_factory=list)


def _observation_key(observation: _RayObservation) -> tuple:
    return (
        observation.sim_time_ns,
        observation.frame_id,
        observation.source_track_id,
        observation.center[0],
        observation.center[1],
    )


def _linear_sum_assignment(costs: list[list[float]]) -> list[int]:
    """Deterministic rectangular Hungarian assignment (rows <= columns)."""
    if not costs:
        return []
    row_count, column_count = len(costs), len(costs[0])
    if row_count > column_count:
        raise ValueError("Assignment requires at least as many columns as rows.")

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            active_row = matched_row[column0]
            delta, column1 = math.inf, 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = costs[active_row - 1][column - 1] - u[active_row] - v[column]
                if reduced < minimum[column] - NUMERICAL_EPS:
                    minimum[column] = reduced
                    previous_column[column] = column0
                if (minimum[column] < delta - NUMERICAL_EPS
                        or (abs(minimum[column] - delta) <= NUMERICAL_EPS and column < column1)):
                    delta, column1 = minimum[column], column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = previous_column[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return assignment


class Final3DPoseEstimator:
    def __init__(
        self,
        *,
        min_void_observation_confidence: float = MIN_VOID_OBSERVATION_CONFIDENCE,
        rolling_pool_size: int = ROLLING_POOL_SIZE,
        minimum_gate_separation_m: float = MIN_GATE_SEPARATION_M,
        spatial_association_max_m: float = SPATIAL_ASSOCIATION_MAX_M,
        ray_residual_max_m: float = RAY_RESIDUAL_MAX_M,
        association_ambiguity_margin: float = ASSOCIATION_AMBIGUITY_MARGIN,
        robust_loss_scale_m: float = ROBUST_LOSS_SCALE_M,
        min_inlier_observations: int = MIN_INLIER_OBSERVATIONS,
        global_solver_max_iterations: int = GLOBAL_SOLVER_MAX_ITERATIONS,
    ) -> None:
        if rolling_pool_size < MIN_ROLLING_POOL_SIZE:
            raise ValueError(f"rolling_pool_size must be >= {MIN_ROLLING_POOL_SIZE}")
        if not 0.0 <= min_void_observation_confidence <= 1.0:
            raise ValueError("min_void_observation_confidence must be within [0, 1].")
        if min_inlier_observations < 1 or min_inlier_observations > rolling_pool_size:
            raise ValueError("min_inlier_observations must be within the rolling pool.")
        if min(
            minimum_gate_separation_m,
            spatial_association_max_m,
            ray_residual_max_m,
            robust_loss_scale_m,
        ) <= 0:
            raise ValueError("Association and robust-loss distances must be positive.")
        if association_ambiguity_margin < 0 or global_solver_max_iterations < 1:
            raise ValueError("Ambiguity margin must be nonnegative and iterations must be positive.")

        self._min_void_observation_confidence = float(min_void_observation_confidence)
        self._rolling_pool_size = int(rolling_pool_size)
        self._minimum_gate_separation_m = float(minimum_gate_separation_m)
        self._spatial_association_max_m = float(spatial_association_max_m)
        self._ray_residual_max_m = float(ray_residual_max_m)
        self._association_ambiguity_margin = float(association_ambiguity_margin)
        self._robust_loss_scale_m = float(robust_loss_scale_m)
        self._min_inlier_observations = int(min_inlier_observations)
        self._global_solver_max_iterations = int(global_solver_max_iterations)

        self._telemetry_checked = False
        self._telemetry = None
        self._landmarks: dict[str, _GlobalLandmark] = {}
        self._next_gate_number = 1
        self._last_rejections: list[dict] = []
        self._separation_conflicts: dict[str, list[str]] = {}

    def _ensure_telemetry(self, frame_path: str) -> None:
        if self._telemetry_checked:
            return
        self._telemetry_checked = True
        path = codo.find_telemetry_for_frame(frame_path)
        if path is not None:
            self._telemetry = codo.load_telemetry(path)

    def _pose(self, frame_path: str):
        if self._telemetry is None:
            return None
        sample = codo.sample_for_frame(self._telemetry, frame_path)
        if sample is None:
            return None
        quaternion, position = sample
        rotation_world_from_camera = _quat_to_matrix(np.array(quaternion)) @ _BODY_FROM_CAM
        return rotation_world_from_camera, np.array(position, dtype=float)

    def _pose_from_vehicle_state(self, vehicle_state: VehicleState | None, frame_sim_time_ns: int):
        if vehicle_state is None:
            return None
        if abs(frame_sim_time_ns - vehicle_state.sim_time_ns) > MAX_TELEMETRY_GAP_S * 1e9:
            return None
        rotation_world_from_camera = (
            _quat_to_matrix(np.array(vehicle_state.attitude_quaternion)) @ _BODY_FROM_CAM
        )
        return rotation_world_from_camera, np.array(vehicle_state.position_local_ned_m, dtype=float)

    # The following four methods preserve compatibility with existing focused numerical tests.
    # Production landmark estimation below does not use their permanent accumulator.
    def _fold_ray(self, entry: dict, center: tuple[float, float], weight: float,
                  r_wc: np.ndarray, c_t: np.ndarray) -> None:
        direction = _normalize(r_wc @ _camera_ray(center))
        projector = _projector(direction)
        entry["A"] += weight * projector
        entry["b"] += weight * (projector @ c_t)
        if entry["ray_count"] > 0:
            mean_direction = _normalize(entry["direction_sum"])
            entry["min_dot"] = min(entry["min_dot"], float(np.dot(direction, mean_direction)))
        entry["direction_sum"] += direction
        entry["ray_count"] += 1

    def _solve(self, entry: dict, c_t: np.ndarray) -> np.ndarray:
        mean_direction = _normalize(entry["direction_sum"])
        prior = c_t + _expected_distance(entry["last_pixel_count"] or ANCHOR_PX) * mean_direction
        return np.linalg.pinv(entry["A"] + PRIOR_WEIGHT * np.eye(3)) @ (
            entry["b"] + PRIOR_WEIGHT * prior
        )

    def _confidence(self, entry: dict) -> float:
        if entry["ray_count"] <= 0:
            return 0.0
        base = next(value for minimum, value in RAY_COUNT_TIERS
                    if entry["ray_count"] >= minimum)
        if entry["ray_count"] < 2:
            return base * 0.5
        spread_deg = math.degrees(math.acos(max(-1.0, min(1.0, entry["min_dot"]))))
        return max(0.0, min(1.0, base * min(1.0, spread_deg / SPREAD_SATURATION_DEG)))

    def _check_passed(self, entry, r_wc: np.ndarray, c_t: np.ndarray) -> None:
        position = entry["position"] if isinstance(entry, dict) else entry.position_ned
        passed = entry["passed"] if isinstance(entry, dict) else entry.passed
        if passed or position is None:
            return
        x_cam = r_wc.T @ (entry["position"] - c_t)
        if x_cam[2] <= 0:
            entry["passed"] = True

    def _gate_observation(self, track_id: str, entry: dict, r_wc: np.ndarray, c_t: np.ndarray) -> VisionGateObservation:
        # Camera-relative NED: translate to the camera's current position only -- no rotation
        # into the camera's own optical attitude, so axes stay north/east/down.
        position_local_ned = tuple(float(v) for v in (entry["position"]))
        trace = {
            "track_id": track_id,
            "position_ned_m": tuple(float(v) for v in entry["position"]),
            "ray_count": entry["ray_count"],
        }

    def _gate_observation(self, landmark: _GlobalLandmark,
                          camera_position_ned: np.ndarray) -> VisionGateObservation:
        return VisionGateObservation(
            gate_id=landmark.gate_id,
            position_local_ned=(
                float(landmark.position_ned[0]),
                float(landmark.position_ned[1]),
                float(landmark.position_ned[2]),
            ),
            position_confidence=float(landmark.position_confidence),
            trace=self._landmark_trace(landmark, camera_position_ned),
        )

    def _advance(self, analysis: VoidAnalysis, frame_path: str,
                 pose) -> list[VisionGateObservation]:
        self._last_rejections = []
        if pose is None:
            self._last_rejections.append({"reason": "missing_or_stale_camera_pose"})
            return []

        frame_id, sim_time_ns = _frame_identity(frame_path)
        r_wc, c_t = pose
        self._retire_passed_and_stale(r_wc, c_t)

        pixel_count_by_track: dict[str, float] = {}
        for region in analysis.regions:
            for void in region.voids:
                if void.track_id:
                    pixel_count_by_track[void.track_id] = max(
                        float(void.pixel_count),
                        pixel_count_by_track.get(void.track_id, 0.0),
                    )

        observations = []
        for estimate in analysis.final_void_estimates:
            confidence = float(estimate.confidence)
            if (
                not math.isfinite(confidence)
                or confidence < self._min_void_observation_confidence
            ):
                self._last_rejections.append({
                    "reason": "observation_confidence",
                    "frame_id": frame_id,
                    "sim_time_ns": sim_time_ns,
                    "source_track_id": str(estimate.track_id),
                    "confidence": confidence if math.isfinite(confidence) else None,
                })
                continue
            if not estimate.track_id:
                self._last_rejections.append({
                    "reason": "missing_track_id",
                    "frame_id": frame_id,
                    "sim_time_ns": sim_time_ns,
                })
                continue
            observation = self._make_observation(
                estimate, frame_id, sim_time_ns, r_wc, c_t, pixel_count_by_track
            )
            if (not np.all(np.isfinite(observation.direction_ned))
                    or not np.all(np.isfinite(observation.provisional_position_ned))):
                self._record_rejection(observation, "invalid_observation_geometry")
                continue
            observations.append(observation)

        # Stable spatial ordering makes gate creation and assignment independent of producer order.
        observations.sort(key=lambda observation: (
            *(float(value) for value in observation.provisional_position_ned),
            observation.source_track_id,
            observation.center,
        ))
        assignments, dispositions = self._joint_assignment(observations)
        for gate_id, landmark in sorted(self._landmarks.items()):
            additions = [
                observations[index] for index, assigned_gate in assignments.items()
                if assigned_gate == gate_id
            ]
            if additions:
                landmark.observations = self._bounded_pool(landmark.observations + additions)
                landmark.missed_frames = 0
                for observation in additions:
                    self._remember_source(landmark, observation.source_track_id)
                self._apply_solution(
                    landmark, self._solve_pool(landmark.observations, landmark.position_ned)
                )
            else:
                landmark.missed_frames += 1

        for index, observation in enumerate(observations):
            disposition = dispositions.get(index, {"status": "new", "candidates": []})
            status = disposition["status"]
            if status == "assigned":
                continue
            if status != "new":
                self._record_rejection(
                    observation, status, {"association_candidates": disposition["candidates"]}
                )
                continue

            nearest_distance = min(
                (
                    float(np.linalg.norm(
                        observation.provisional_position_ned - landmark.position_ned
                    ))
                    for landmark in self._landmarks.values()
                ),
                default=math.inf,
            )
            if nearest_distance < self._minimum_gate_separation_m:
                self._record_rejection(
                    observation,
                    "minimum_gate_separation",
                    {"nearest_landmark_distance_m": nearest_distance},
                )
                continue
            if len(self._landmarks) >= MAX_ACTIVE_LANDMARKS:
                self._record_rejection(observation, "active_landmark_capacity")
                continue
            landmark = self._create_landmark(observation)

        self._merge_landmarks()
        self._retire_passed_and_stale(r_wc, c_t)
        self._update_separation_conflicts()
        return [
            self._gate_observation(landmark, c_t)
            for _, landmark in sorted(self._landmarks.items())
            if not landmark.passed
            and landmark.gate_id not in self._separation_conflicts
            and sum(observation.inlier for observation in landmark.observations)
            >= self._min_inlier_observations
        ]

    def _observation_trace(self) -> dict:
        return {
            "coordinate_frame": "local_ned",
            "position_semantics": "absolute_landmark",
            "rolling_pool_size": self._rolling_pool_size,
            "active_landmark_count": len(self._landmarks),
            "separation_conflicts": {
                gate_id: list(conflicts)
                for gate_id, conflicts in sorted(self._separation_conflicts.items())
            },
            "rejected_observations": list(self._last_rejections),
        }

    def update(self, analysis: VoidAnalysis, frame_path: str) -> VoidAnalysis:
        self._ensure_telemetry(frame_path)
        gates = self._advance(analysis, frame_path, self._pose(frame_path))
        frame_id, sim_time_ns = _frame_identity(frame_path)
        analysis.vision_observation = VisionObservation(
            frame_id, sim_time_ns, gates, source="vision", trace=self._observation_trace()
        )
        return analysis

    def update_live(self, analysis: VoidAnalysis, frame_path: str,
                    vehicle_state: VehicleState | None) -> VoidAnalysis:
        frame_id, sim_time_ns = _frame_identity(frame_path)
        pose = self._pose_from_vehicle_state(vehicle_state, sim_time_ns)
        gates = self._advance(analysis, frame_path, pose)
        analysis.vision_observation = VisionObservation(
            frame_id, sim_time_ns, gates, source="vision", trace=self._observation_trace()
        )
        return analysis
