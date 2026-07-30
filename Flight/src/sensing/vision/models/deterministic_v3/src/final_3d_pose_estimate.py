"""Estimate persistent physical gate landmarks in a fixed local-NED frame.

Upstream ``track_id`` values identify temporary evidence, never physical gates. A landmark is
never created from a single ray's depth guess: unmatched rays wait in a pending pool until a
mutually consistent cluster of rays (spanning at least two camera positions) triangulates a
common point -- that joint intersection seeds the landmark. Association and inlier membership
are gated by a small fixed-radius beam around each ray (not a range-scaled cone), reflecting
real target size and sensor noise rather than an arbitrary tolerance.

A landmark's position is never frozen. Every active landmark keeps accepting matching rays and
is re-solved every frame over its full retained ray history; published confidence is computed
honestly each frame from that evidence (ray support, residual tightness, and angular diversity)
and rises only as genuinely diverse agreement accumulates -- a narrow single-track ray bundle
yields a position with deliberately low confidence until differently-angled evidence confirms
it. Landmarks that are physically fixed relative to one another mutually reinforce: pairwise
relative offsets are learned continuously, and any currently-observed, evidenced neighbor plus
its learned offset helps stabilize a weaker landmark's estimate. Published
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

# Monocular depth prior. It regularizes weak ray geometry and seeds a landmark's first solve,
# but never gates association against an existing landmark and never seeds a landmark alone.
ANCHOR_PX, ANCHOR_M = 4000.0, 10.0
_OTHER_PX, _OTHER_M = 40.0, 40.0
DEPTH_PRIOR_POWER = math.log(_OTHER_M / ANCHOR_M) / math.log(_OTHER_PX / ANCHOR_PX)
# Dual role: at creation the prior is the founding rays' crude depth guess (where a large weight
# biases the solved depth -- at 0.05 a clean two-ray triangulation once converged 0.45m off);
# for the rest of a landmark's life the prior is its own last solved position, where this weight
# is the estimator's ONLY temporal damping. Calibrated at 0.05 by a motion-attribution A/B on
# real runs: run-wide published motion fell 62% (764m -> 290m) and final accuracy *improved*
# (one gate 0.40m -> 0.04m; another settled for the first time) -- the creation-time bias is a
# transient that accumulating rays correct, while the damping benefit compounds all run. 0.2
# over-damps (strands estimates off-truth). Split into separate creation/ongoing constants if
# creation bias ever resurfaces.
PRIOR_WEIGHT = 0.05

# Centralized calibration surface.
MIN_VOID_OBSERVATION_CONFIDENCE = 0.55
MAX_RETAINED_OBSERVATIONS_PER_LANDMARK = 500  # memory bound only, not a behavioral truncation:
                                              # a single gate lineage produced 247 qualifying
                                              # rays in ~10s of real data, and rays are cached
                                              # for continuous re-optimization by design
MIN_GATE_SEPARATION_M = 5.0  # duplicate-avoidance threshold: spawn-guard and merge-eligibility
# Calibrated on real runs (11Z/38z sweep, 9 iterations): 0.4 captures most genuine near-miss
# rays (largest residual mass sat at 0.25-0.5m) with acceptable inlier loosening; 0.25 starved
# landmarks of returning evidence and fed duplicate fragments, 0.5 loosened discipline enough
# to spawn new fragments.
BEAM_RADIUS_M = 0.4  # fixed (not range-scaled) target-size tolerance around each ray
ASSOCIATION_AMBIGUITY_MARGIN = 0.10  # normalized (residual / BEAM_RADIUS_M) cost margin
ROBUST_LOSS_SCALE_M = 0.15
# 4 (was 2, same sweep): suppresses marginal 2-inlier flicker fragments from ever publishing
# or founding landmarks; drives publish gate, creation minimum, merge eligibility,
# offset-learning and corroboration-anchor eligibility.
MIN_INLIER_OBSERVATIONS = 4
IRLS_MAX_ITERATIONS = 5
LANDMARK_MAX_MISSED_FRAMES = 120  # unobserved this long -> retired (recreatable from pending)
MAX_ACTIVE_LANDMARKS = 5  # concurrently in-play landmarks; a passed landmark frees its slot
PENDING_MAX_AGE_FRAMES = 120  # matches LANDMARK_MAX_MISSED_FRAMES: real parallax on a head-on
                              # approach accumulates slowly (radial motion changes apparent size,
                              # not bearing), so an unpromoted ray needs several seconds of
                              # retained history before a converging cluster can form around it
MAX_PENDING_OBSERVATIONS = 256  # memory bound on the pending pool, evict-oldest-on-insert
POSITION_CONVERGENCE_M = 1e-4
NUMERICAL_EPS = 1e-9
# Angular diversity at which the confidence geometry factor saturates. Purely a confidence
# normalization -- never a hard gate: narrow-spread evidence still creates and publishes a
# landmark, it just carries honestly low confidence until diverse rays confirm it.
SPREAD_SATURATION_DEG = 15.0

SUPPORT_SATURATION_RAYS = 5  # inlier count at which the confidence support factor saturates
# 0.15 (was 0.5, motion-attribution A/B): pairwise offsets are running means between anchors
# that themselves move, so corroboration was a feedback cascade driving 61% of all published
# position motion -- including movement on frames with zero evidence change. 0.15 keeps the
# stabilizing benefit (corroboration-off measurably worsens fragmentation) without the cascade.
CORROBORATION_MAX_WEIGHT = 0.15


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


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def _max_spread_deg(directions: list[np.ndarray]) -> float:
    """Largest pairwise angle among unit direction vectors, vectorized: with hundreds of
    retained rays re-scored every frame, the O(n^2) pure-Python pairwise loop is a real
    per-frame cost, while one Gram-matrix product is microseconds."""
    if len(directions) < 2:
        return 0.0
    stacked = np.stack(directions)
    minimum_dot = float(np.min(stacked @ stacked.T))
    return math.degrees(math.acos(max(-1.0, min(1.0, minimum_dot))))


@dataclass
class _RayObservation:
    source_track_id: str
    source: str
    frame_id: int
    sim_time_ns: int
    sequence: int  # monotonic ingestion-order tiebreaker, replaces pixel-space center
    camera_position_ned: np.ndarray
    direction_ned: np.ndarray
    confidence: float
    pixel_count: float
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
class _Landmark:
    gate_id: str
    observations: list[_RayObservation]
    position_ned: np.ndarray  # published: direct solve, plus corroboration when available
    direct_position_ned: np.ndarray  # ray+prior-only solve; drives offset learning
    residual_rms_m: float = math.inf
    position_confidence: float = 0.0
    passed: bool = False  # sticky
    missed_frames: int = 0
    source_track_ids: list[str] = field(default_factory=list)
    merged_gate_ids: list[str] = field(default_factory=list)
    corroborated_by: list[str] = field(default_factory=list)


@dataclass
class _OffsetEstimate:
    offset_ned: np.ndarray  # running mean of position[high_id] - position[low_id]
    sample_count: int = 0


def _observation_key(observation: _RayObservation) -> tuple:
    return (observation.sim_time_ns, observation.frame_id, observation.source_track_id,
            observation.sequence)


class Final3DPoseEstimator:
    def __init__(
        self,
        *,
        min_void_observation_confidence: float = MIN_VOID_OBSERVATION_CONFIDENCE,
        minimum_gate_separation_m: float = MIN_GATE_SEPARATION_M,
        beam_radius_m: float = BEAM_RADIUS_M,
        association_ambiguity_margin: float = ASSOCIATION_AMBIGUITY_MARGIN,
        robust_loss_scale_m: float = ROBUST_LOSS_SCALE_M,
        min_inlier_observations: int = MIN_INLIER_OBSERVATIONS,
        irls_max_iterations: int = IRLS_MAX_ITERATIONS,
        max_active_landmarks: int = MAX_ACTIVE_LANDMARKS,
        pending_max_age_frames: int = PENDING_MAX_AGE_FRAMES,
        max_pending_observations: int = MAX_PENDING_OBSERVATIONS,
    ) -> None:
        if not 0.0 <= min_void_observation_confidence <= 1.0:
            raise ValueError("min_void_observation_confidence must be within [0, 1].")
        if min_inlier_observations < 1:
            raise ValueError("min_inlier_observations must be positive.")
        if min(minimum_gate_separation_m, beam_radius_m, robust_loss_scale_m) <= 0:
            raise ValueError("Association and robust-loss distances must be positive.")
        if association_ambiguity_margin < 0 or irls_max_iterations < 1:
            raise ValueError("Ambiguity margin must be nonnegative and iterations must be positive.")
        if max_active_landmarks < 1 or pending_max_age_frames < 1 or max_pending_observations < 1:
            raise ValueError("Landmark and pending-pool capacities must be positive.")

        self._min_void_observation_confidence = float(min_void_observation_confidence)
        self._minimum_gate_separation_m = float(minimum_gate_separation_m)
        self._beam_radius_m = float(beam_radius_m)
        self._association_ambiguity_margin = float(association_ambiguity_margin)
        self._robust_loss_scale_m = float(robust_loss_scale_m)
        self._min_inlier_observations = int(min_inlier_observations)
        self._irls_max_iterations = int(irls_max_iterations)
        self._max_active_landmarks = int(max_active_landmarks)
        self._pending_max_age_frames = int(pending_max_age_frames)
        self._max_pending_observations = int(max_pending_observations)

        self._telemetry_checked = False
        self._telemetry = None
        self._landmarks: dict[str, _Landmark] = {}
        self._pending: list[_RayObservation] = []
        self._offsets: dict[tuple[str, str], _OffsetEstimate] = {}
        self._next_gate_number = 1
        self._next_sequence = 0
        self._last_rejections: list[dict] = []

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

    def _recent_pixel_count(self, track_id: str) -> float:
        candidates = [
            observation
            for landmark in self._landmarks.values()
            for observation in landmark.observations
            if observation.source_track_id == track_id
        ]
        if not candidates:
            return ANCHOR_PX
        return max(candidates, key=_observation_key).pixel_count

    def _make_observation(self, estimate, frame_id: int, sim_time_ns: int,
                          r_wc: np.ndarray, c_t: np.ndarray,
                          pixel_count_by_track: dict[str, float]) -> _RayObservation:
        center = (float(estimate.center[0]), float(estimate.center[1]))
        direction = _normalize(r_wc @ _camera_ray(center))
        pixel_count = float(pixel_count_by_track.get(
            estimate.track_id, self._recent_pixel_count(estimate.track_id)
        ))
        provisional = c_t + _expected_distance(pixel_count) * direction
        self._next_sequence += 1
        return _RayObservation(
            source_track_id=str(estimate.track_id),
            source=str(estimate.source),
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            sequence=self._next_sequence,
            camera_position_ned=c_t.copy(),
            direction_ned=direction,
            confidence=float(estimate.confidence),
            pixel_count=pixel_count,
            provisional_position_ned=provisional,
        )

    def _solve_landmark(self, observations: list[_RayObservation], prior: np.ndarray,
                        corroboration: tuple[tuple[float, np.ndarray], ...] = ()) -> _Solution:
        """Robust IRLS position solve over ray-projector rows, a low-weight regularizer toward
        ``prior``, and optional weighted corroboration rows (same linear-system pattern). At
        landmark creation ``prior`` is the average of the two founding rays' monocular depth
        guesses; on later updates it is the landmark's own last solved position, so the
        regularizer encodes "assume the landmark hasn't moved" rather than re-anchoring to a
        fresh noisy guess. Also used, unmodified, as the triangulation test for promoting a
        pending cluster: a group of rays "intersects" well enough to seed a landmark exactly
        when they remain joint inliers of this same solve."""
        if not observations:
            return _Solution(prior.copy(), [], (), math.inf, 0.0)

        active = tuple(range(len(observations)))
        position = prior.copy()

        def solve_weighted(indices: tuple[int, ...], robust_weights: list[float]) -> np.ndarray:
            matrix = PRIOR_WEIGHT * np.eye(3)
            vector = PRIOR_WEIGHT * prior
            for weight, point in corroboration:
                matrix += weight * np.eye(3)
                vector += weight * point
            for index, robust_weight in zip(indices, robust_weights):
                observation = observations[index]
                weight = observation.confidence * robust_weight
                projector = _projector(observation.direction_ned)
                matrix += weight * projector
                vector += weight * (projector @ observation.camera_position_ned)
            return np.linalg.pinv(matrix) @ vector

        def irls(indices: tuple[int, ...], start: np.ndarray) -> np.ndarray:
            current = start
            for _ in range(self._irls_max_iterations):
                residuals = [
                    _ray_residual(current, observations[index].camera_position_ned,
                                  observations[index].direction_ned)
                    for index in indices
                ]
                robust_weights = [
                    1.0 if residual <= self._robust_loss_scale_m
                    else self._robust_loss_scale_m / max(residual, NUMERICAL_EPS)
                    for residual in residuals
                ]
                candidate = solve_weighted(indices, robust_weights)
                if float(np.linalg.norm(candidate - current)) <= POSITION_CONVERGENCE_M:
                    return candidate
                current = candidate
            return current

        # Recompute hard inliers after each robust solve; every call starts from all retained
        # observations, so a previously rejected ray can become an inlier as evidence changes.
        for _ in range(self._irls_max_iterations):
            position = irls(active, position)
            new_active = tuple(
                index for index, observation in enumerate(observations)
                if _ray_depth(position, observation.camera_position_ned,
                              observation.direction_ned) > NUMERICAL_EPS
                and _ray_residual(position, observation.camera_position_ned,
                                  observation.direction_ned) <= self._beam_radius_m
            )
            if not new_active:
                position = prior.copy()
                active = ()
                break
            if new_active == active:
                break
            active = new_active

        residuals = [
            _ray_residual(position, observation.camera_position_ned, observation.direction_ned)
            for observation in observations
        ]
        final_inliers = tuple(
            index for index in active
            if _ray_depth(position, observations[index].camera_position_ned,
                          observations[index].direction_ned) > NUMERICAL_EPS
            and residuals[index] <= self._beam_radius_m
        )
        rms = (
            math.sqrt(sum(residuals[index] ** 2 for index in final_inliers) / len(final_inliers))
            if final_inliers else math.inf
        )
        confidence = self._solution_confidence(observations, final_inliers, rms)
        return _Solution(position, residuals, final_inliers, rms, confidence)

    def _solution_confidence(self, observations: list[_RayObservation],
                             inlier_indices: tuple[int, ...], rms_residual_m: float) -> float:
        if not inlier_indices:
            return 0.0
        mean_observation_confidence = sum(
            observations[index].confidence for index in inlier_indices
        ) / len(inlier_indices)
        support_factor = min(1.0, len(inlier_indices) / SUPPORT_SATURATION_RAYS)
        residual_factor = max(0.0, 1.0 - rms_residual_m / self._beam_radius_m)

        directions = [observations[index].direction_ned for index in inlier_indices]
        if len(directions) < 2:
            geometry_factor = 0.5
        else:
            geometry_factor = min(1.0, _max_spread_deg(directions) / SPREAD_SATURATION_DEG)
        return max(0.0, min(
            1.0,
            mean_observation_confidence * support_factor * residual_factor * geometry_factor,
        ))

    @staticmethod
    def _apply_direct_solution(landmark: _Landmark, solution: _Solution) -> None:
        landmark.direct_position_ned = solution.position_ned
        landmark.residual_rms_m = solution.rms_residual_m
        landmark.position_confidence = solution.confidence
        inlier_set = set(solution.inlier_indices)
        for index, observation in enumerate(landmark.observations):
            observation.residual_m = solution.residuals_m[index]
            observation.inlier = index in inlier_set

    def _cap_history(self, observations: list[_RayObservation]) -> list[_RayObservation]:
        ordered = sorted(observations, key=_observation_key)
        return ordered[-MAX_RETAINED_OBSERVATIONS_PER_LANDMARK:]

    def _check_passed(self, landmark: _Landmark, r_wc: np.ndarray, c_t: np.ndarray) -> None:
        if landmark.passed:
            return
        if float((r_wc.T @ (landmark.position_ned - c_t))[2]) <= 0:
            landmark.passed = True

    @staticmethod
    def _offset_key(first_id: str, second_id: str) -> tuple[str, str]:
        return (first_id, second_id) if first_id < second_id else (second_id, first_id)

    def _update_offsets(self) -> None:
        gate_ids = sorted(self._landmarks)
        for first_index, low_id in enumerate(gate_ids):
            low = self._landmarks[low_id]
            low_inliers = sum(observation.inlier for observation in low.observations)
            if low_inliers < self._min_inlier_observations:
                continue
            for high_id in gate_ids[first_index + 1:]:
                high = self._landmarks[high_id]
                high_inliers = sum(observation.inlier for observation in high.observations)
                if high_inliers < self._min_inlier_observations:
                    continue
                key = (low_id, high_id)
                sample = high.direct_position_ned - low.direct_position_ned
                estimate = self._offsets.get(key)
                if estimate is None:
                    self._offsets[key] = _OffsetEstimate(offset_ned=sample.copy(), sample_count=1)
                else:
                    count = estimate.sample_count + 1
                    estimate.offset_ned = estimate.offset_ned + (sample - estimate.offset_ned) / count
                    estimate.sample_count = count

    def _corroboration_rows(self, landmark: _Landmark) -> tuple[tuple[float, np.ndarray], ...]:
        # Gates are physically fixed relative to one another, so any *currently observed*,
        # evidenced neighbor is a legitimate stabilizing reference -- the only disqualifier is
        # not presently providing visual information (missed this frame). Anchor trust scales
        # with the neighbor's own honestly-computed confidence, never a binary settled flag.
        rows = []
        contributors = []
        for other_id, other in sorted(self._landmarks.items()):
            if other_id == landmark.gate_id or other.missed_frames != 0:
                continue
            if sum(o.inlier for o in other.observations) < self._min_inlier_observations:
                continue
            key = self._offset_key(landmark.gate_id, other_id)
            estimate = self._offsets.get(key)
            if estimate is None:
                continue
            offset_confidence = min(1.0, estimate.sample_count / SUPPORT_SATURATION_RAYS)
            weight = other.position_confidence * offset_confidence * CORROBORATION_MAX_WEIGHT
            if weight <= 0.0:
                continue
            sign = 1.0 if landmark.gate_id > other_id else -1.0
            target = other.position_ned + sign * estimate.offset_ned
            rows.append((weight, target))
            contributors.append(other_id)
        landmark.corroborated_by = contributors
        return tuple(rows)

    def _new_gate_id(self) -> str:
        gate_id = f"global_gate_{self._next_gate_number:03d}"
        self._next_gate_number += 1
        return gate_id

    def _record_rejection(self, observation: _RayObservation | None, reason: str,
                          details: dict | None = None) -> None:
        record: dict[str, object] = {"reason": reason}
        if observation is not None:
            record.update({
                "frame_id": observation.frame_id,
                "sim_time_ns": observation.sim_time_ns,
                "source_track_id": observation.source_track_id,
                "confidence": observation.confidence,
                "provisional_position_ned_m": [
                    float(value) for value in observation.provisional_position_ned
                ],
            })
        if details:
            record.update(details)
        self._last_rejections.append(record)

    def _create_landmark(self, observations: list[_RayObservation],
                         solution: _Solution) -> _Landmark:
        gate_id = self._new_gate_id()
        landmark = _Landmark(
            gate_id=gate_id,
            observations=list(observations),
            position_ned=solution.position_ned.copy(),
            direct_position_ned=solution.position_ned.copy(),
        )
        for observation in observations:
            if observation.source_track_id not in landmark.source_track_ids:
                landmark.source_track_ids.append(observation.source_track_id)
        self._apply_direct_solution(landmark, solution)
        landmark.position_ned = landmark.direct_position_ned
        self._landmarks[gate_id] = landmark
        return landmark

    def _age_out_pending(self, frame_id: int) -> None:
        self._pending = [
            observation for observation in self._pending
            if frame_id - observation.frame_id <= self._pending_max_age_frames
        ]

    def _add_to_pending(self, observation: _RayObservation) -> None:
        if len(self._pending) >= self._max_pending_observations:
            self._pending.remove(min(self._pending, key=_observation_key))
        self._pending.append(observation)

    def _best_pending_cluster(self, observation: _RayObservation
                              ) -> tuple[list[_RayObservation], _Solution] | None:
        # A landmark is seeded from a *converging group* of rays, never a single ray's depth
        # guess: gather every pending ray plausibly aimed at the same physical point as this
        # new one (the same MIN_GATE_SEPARATION_M neighborhood already used for landmark
        # dedup), solve them all together, and seed from the aggregate inlier set. There is
        # deliberately no minimum angular-spread bar here: a narrow, radial-approach bundle
        # still creates a landmark -- it just publishes at honestly-low confidence
        # (geometry_factor) until differently-angled evidence arrives and refines it. The only
        # geometric requirement is a real baseline: inliers must span at least two distinct
        # frames, since rays from a single camera position cannot triangulate anything.
        candidates = [observation] + [
            pending for pending in self._pending
            if float(np.linalg.norm(
                pending.provisional_position_ned - observation.provisional_position_ned
            )) < self._minimum_gate_separation_m
        ]
        if len(candidates) < 2:
            return None
        prior = sum((c.provisional_position_ned for c in candidates), np.zeros(3)) / len(candidates)
        trial = self._solve_landmark(candidates, prior)
        if len(trial.inlier_indices) < self._min_inlier_observations:
            return None
        inliers = [candidates[index] for index in trial.inlier_indices]
        if len({inlier.frame_id for inlier in inliers}) < 2:
            return None
        return inliers, trial

    def _handle_unmatched(self, observation: _RayObservation) -> None:
        cluster = self._best_pending_cluster(observation)
        if cluster is None:
            self._add_to_pending(observation)
            self._record_rejection(
                observation, "joined_pending_pool", {"pending_pool_size": len(self._pending)}
            )
            return
        inliers, trial = cluster
        for member in inliers:
            if member is not observation and member in self._pending:
                self._pending.remove(member)
        consumed = observation in inliers
        if not consumed:
            # The converging cluster came together from other pending rays alone; this
            # observation just triggered the check and stays queued for its own future match.
            self._add_to_pending(observation)

        nearby = min(
            self._landmarks.values(),
            key=lambda landmark: float(np.linalg.norm(trial.position_ned - landmark.position_ned)),
            default=None,
        )
        if nearby is not None and float(np.linalg.norm(
            trial.position_ned - nearby.position_ned
        )) < self._minimum_gate_separation_m:
            nearby.observations = self._cap_history(nearby.observations + inliers)
            nearby.missed_frames = 0
            for source in inliers:
                if source.source_track_id not in nearby.source_track_ids:
                    nearby.source_track_ids.append(source.source_track_id)
            solution = self._solve_landmark(nearby.observations, nearby.direct_position_ned)
            self._apply_direct_solution(nearby, solution)
            nearby.position_ned = nearby.direct_position_ned
            self._record_rejection(
                observation, "routed_to_nearby_landmark",
                {"gate_id": nearby.gate_id, "pending_cluster_size": len(inliers)},
            )
            return

        if len(self._landmarks) >= self._max_active_landmarks:
            for member in inliers:
                if member is not observation:
                    self._pending.append(member)  # don't lose a good cluster to a transient cap
            self._record_rejection(observation, "active_landmark_capacity")
            return

        landmark = self._create_landmark(inliers, trial)
        self._record_rejection(observation, "promoted_to_landmark", {
            "gate_id": landmark.gate_id,
            "pending_cluster_size": len(inliers),
            "triangulation_rms_m": float(trial.rms_residual_m),
        })

    def _merge_landmarks(self) -> None:
        # Physical prior: no two real gates sit within MIN_GATE_SEPARATION_M of each other, so
        # two landmarks inside that radius are fragments of one gate and must consolidate. The
        # joint robust solve decides the merged position -- with the tight beam radius it locks
        # onto the dominant coherent ray bundle and demotes the rest to retained outliers, so
        # consolidation cannot drag the position into a meaningless average. The only veto is a
        # sanity floor: the joint solve must retain at least the stronger side's support
        # (a merge may absorb junk, but must never destroy the better-evidenced estimate).
        for _ in range(self._irls_max_iterations):
            merged = False
            gate_ids = sorted(self._landmarks)
            for first_index, first_id in enumerate(gate_ids):
                first = self._landmarks.get(first_id)
                if first is None:
                    continue
                for second_id in gate_ids[first_index + 1:]:
                    second = self._landmarks.get(second_id)
                    if second is None:
                        continue
                    if (float(np.linalg.norm(first.position_ned - second.position_ned))
                            > self._minimum_gate_separation_m):
                        continue
                    first_inliers = sum(observation.inlier for observation in first.observations)
                    second_inliers = sum(observation.inlier for observation in second.observations)
                    if (first_inliers < self._min_inlier_observations
                            or second_inliers < self._min_inlier_observations):
                        continue

                    survivor, removed = (first, second) if first.gate_id < second.gate_id else (second, first)
                    combined = survivor.observations + removed.observations
                    trial = self._solve_landmark(combined, survivor.direct_position_ned)
                    if len(trial.inlier_indices) < max(first_inliers, second_inliers):
                        continue  # joint solve destroys the stronger side's support -- abandon
                    survivor.observations = combined
                    self._apply_direct_solution(survivor, trial)
                    survivor.position_ned = survivor.direct_position_ned
                    survivor.observations = self._cap_history(survivor.observations)

                    survivor.source_track_ids = (
                        survivor.source_track_ids + [
                            track_id for track_id in removed.source_track_ids
                            if track_id not in survivor.source_track_ids
                        ]
                    )
                    survivor.merged_gate_ids = (
                        survivor.merged_gate_ids + [removed.gate_id] + removed.merged_gate_ids
                    )
                    survivor.missed_frames = min(survivor.missed_frames, removed.missed_frames)
                    for key in [key for key in self._offsets if removed.gate_id in key]:
                        del self._offsets[key]
                    del self._landmarks[removed.gate_id]
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                return

    def _retire_stale(self, r_wc: np.ndarray, c_t: np.ndarray) -> None:
        for gate_id in list(sorted(self._landmarks)):
            landmark = self._landmarks[gate_id]
            self._check_passed(landmark, r_wc, c_t)
            stale = landmark.missed_frames > LANDMARK_MAX_MISSED_FRAMES
            if landmark.passed or stale:
                for key in [key for key in self._offsets if gate_id in key]:
                    del self._offsets[key]
                del self._landmarks[gate_id]

    def _associate(self, observations: list[_RayObservation],
                   positions: dict[str, np.ndarray]) -> tuple[dict[int, str], dict[int, dict]]:
        # A landmark may be reinforced by more than one observation per frame: with a tight,
        # fixed beam radius, two different tracks both passing the same landmark's residual
        # gate in one frame is corroborating evidence, not ambiguity, so there is no cross-
        # observation claim contention -- each observation independently takes its own best,
        # unambiguous match. The only remaining ambiguity check is per-observation: does this
        # one ray plausibly match two *different* landmarks equally well.
        assignments: dict[int, str] = {}
        dispositions: dict[int, dict] = {}
        for index, observation in enumerate(observations):
            row = []
            for gate_id, position in positions.items():
                depth = _ray_depth(position, observation.camera_position_ned, observation.direction_ned)
                if depth <= NUMERICAL_EPS:
                    continue
                residual = _ray_residual(position, observation.camera_position_ned, observation.direction_ned)
                if residual > self._beam_radius_m:
                    continue
                row.append((residual / self._beam_radius_m, residual, gate_id))
            row.sort()

            if not row:
                dispositions[index] = {"status": "new", "candidates": []}
                continue
            trace = [
                {"gate_id": gate_id, "ray_residual_m": float(residual)}
                for _, residual, gate_id in row
            ]
            if len(row) > 1 and row[1][0] - row[0][0] <= self._association_ambiguity_margin:
                dispositions[index] = {"status": "ambiguous", "candidates": trace}
                continue
            gate_id = row[0][2]
            assignments[index] = gate_id
            dispositions[index] = {"status": "assigned", "gate_id": gate_id, "candidates": trace}
        return assignments, dispositions

    def _landmark_trace(self, landmark: _Landmark, camera_position_ned: np.ndarray) -> dict:
        supporting: list[dict] = []
        rejected: list[dict] = []
        for observation in sorted(landmark.observations, key=_observation_key):
            item = {
                "frame_id": observation.frame_id,
                "sim_time_ns": observation.sim_time_ns,
                "source_track_id": observation.source_track_id,
                "source": observation.source,
                "confidence": observation.confidence,
                "provisional_position_ned_m": [
                    float(value) for value in observation.provisional_position_ned
                ],
                "ray_residual_m": float(observation.residual_m),
            }
            (supporting if observation.inlier else rejected).append(item)
        return {
            "coordinate_frame": "local_ned",
            "position_semantics": "absolute_landmark",
            "position_ned_m": [float(value) for value in landmark.position_ned],
            "camera_relative_position_ned_m": [
                float(value) for value in landmark.position_ned - camera_position_ned
            ],
            "inlier_spread_deg": float(_max_spread_deg(
                [observation.direction_ned for observation in landmark.observations
                 if observation.inlier]
            )),
            "source_track_ids": list(landmark.source_track_ids),
            "merged_gate_ids": list(landmark.merged_gate_ids),
            "corroborated_by": list(landmark.corroborated_by),
            "supporting_frames": supporting,
            "rejected_observations": rejected,
            "inlier_count": len(supporting),
            "ray_count": len(supporting),
            "pool_occupancy": len(landmark.observations),
            "rms_residual_m": (
                None if not math.isfinite(landmark.residual_rms_m)
                else float(landmark.residual_rms_m)
            ),
        }

    def _gate_observation(self, landmark: _Landmark, camera_position_ned: np.ndarray) -> VisionGateObservation:
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

    def _advance(self, analysis: VoidAnalysis, frame_path: str, pose) -> list[VisionGateObservation]:
        self._last_rejections = []
        if pose is None:
            self._last_rejections.append({"reason": "missing_or_stale_camera_pose"})
            return []

        frame_id, sim_time_ns = _frame_identity(frame_path)
        r_wc, c_t = pose
        self._retire_stale(r_wc, c_t)
        self._age_out_pending(frame_id)

        pixel_count_by_track: dict[str, float] = {}
        for region in analysis.regions:
            for void in region.voids:
                if void.track_id:
                    pixel_count_by_track[void.track_id] = max(
                        float(void.pixel_count), pixel_count_by_track.get(void.track_id, 0.0),
                    )

        observations: list[_RayObservation] = []
        for estimate in analysis.final_void_estimates:
            confidence = float(estimate.confidence)
            if not math.isfinite(confidence) or confidence < self._min_void_observation_confidence:
                self._last_rejections.append({
                    "reason": "observation_confidence", "frame_id": frame_id, "sim_time_ns": sim_time_ns,
                    "source_track_id": str(estimate.track_id),
                    "confidence": confidence if math.isfinite(confidence) else None,
                })
                continue
            if not estimate.track_id:
                self._last_rejections.append({
                    "reason": "missing_track_id", "frame_id": frame_id, "sim_time_ns": sim_time_ns,
                })
                continue
            observation = self._make_observation(estimate, frame_id, sim_time_ns, r_wc, c_t, pixel_count_by_track)
            if (not np.all(np.isfinite(observation.direction_ned))
                    or not np.all(np.isfinite(observation.provisional_position_ned))):
                self._record_rejection(observation, "invalid_observation_geometry")
                continue
            observations.append(observation)

        # Stable spatial ordering makes gate creation and assignment independent of producer order.
        observations.sort(key=lambda observation: (
            *(float(value) for value in observation.provisional_position_ned),
            observation.source_track_id,
        ))

        positions = {gate_id: landmark.position_ned for gate_id, landmark in self._landmarks.items()}
        assignments, dispositions = self._associate(observations, positions)

        for gate_id, landmark in sorted(self._landmarks.items()):
            additions = [
                observations[index] for index, assigned_gate in assignments.items()
                if assigned_gate == gate_id
            ]
            if additions:
                landmark.missed_frames = 0
                for observation in additions:
                    if observation.source_track_id not in landmark.source_track_ids:
                        landmark.source_track_ids.append(observation.source_track_id)
                landmark.observations = self._cap_history(landmark.observations + additions)
            else:
                landmark.missed_frames += 1

        # Every active landmark re-solves every frame over its full retained history -- a
        # landmark's position is never frozen, it only ever stops changing because the
        # evidence stops moving it.
        for gate_id, landmark in self._landmarks.items():
            solution = self._solve_landmark(landmark.observations, landmark.direct_position_ned)
            self._apply_direct_solution(landmark, solution)

        self._update_offsets()

        for gate_id, landmark in self._landmarks.items():
            corroboration = self._corroboration_rows(landmark)
            if corroboration:
                solution = self._solve_landmark(
                    landmark.observations, landmark.direct_position_ned, corroboration
                )
                landmark.position_ned = solution.position_ned
            else:
                landmark.position_ned = landmark.direct_position_ned

        for index, observation in enumerate(observations):
            disposition = dispositions.get(index, {"status": "new", "candidates": []})
            status = disposition["status"]
            if status == "assigned":
                continue
            if status != "new":
                self._record_rejection(observation, status, {"association_candidates": disposition["candidates"]})
                continue
            self._handle_unmatched(observation)

        self._merge_landmarks()
        self._retire_stale(r_wc, c_t)
        return [
            self._gate_observation(landmark, c_t)
            for _, landmark in sorted(self._landmarks.items())
            if not landmark.passed
            and sum(observation.inlier for observation in landmark.observations) >= self._min_inlier_observations
        ]

    def _observation_trace(self) -> dict:
        return {
            "coordinate_frame": "local_ned",
            "position_semantics": "absolute_landmark",
            "active_landmark_count": len(self._landmarks),
            "pending_pool_size": len(self._pending),
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
