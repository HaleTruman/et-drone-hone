from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from core.coordinates import (
    normalize_quaternion,
    quaternion_from_roll_pitch_yaw_deg,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
    vec3,
)
from core.schemas import VehicleState
from mapping.perception import VisionGateObservation, VisionObservation


@dataclass
class GateRecord:
    gate_id: str
    position_local_ned_m: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    position_confidence: float
    quaternion_confidence: float
    crossed: bool = False
    frozen: bool = False
    sequence: int | None = None
    observation_count: int = 1
    last_observed_cycle: int | None = None
    source: str = "placeholder"
    outer_width_m: float = 2.7
    outer_height_m: float = 2.7
    inner_width_m: float = 1.5
    inner_height_m: float = 1.5
    depth_m: float = 0.26
    average_residual_m: float = 0.0
    last_seen_time_s: float | None = None
    locked: bool = False


@dataclass
class GateLandmark:
    id: int
    pose_position_ned_m: np.ndarray
    pose_quaternion_ned: tuple[float, float, float, float]
    information: np.ndarray
    observation_count: int
    last_seen_time_s: float
    last_observed_cycle: int | None
    average_residual_m: float = 0.0
    sequential_index: int | None = None
    locked: bool = False
    crossed: bool = False
    source_gate_id: str = ""
    position_confidence: float = 0.0
    orientation_confidence: float = 0.0


@dataclass
class CandidateTrack:
    id: int
    estimated_position_ned_m: np.ndarray
    estimated_quaternion_ned: tuple[float, float, float, float]
    observations: list[GateRecord] = field(default_factory=list)
    hit_count: int = 0
    last_seen_time_s: float = 0.0
    last_observed_cycle: int | None = None
    average_residual_m: float = 0.0
    associated_map_id: int | None = None
    source_gate_id: str = ""
    sequential_index: int | None = None


@dataclass(frozen=True)
class _PredictedLandmark:
    landmark_id: int
    position_camera_m: np.ndarray
    visible: bool


class GateMap:
    """Robust static gate landmark map for noisy, unordered vision detections.

    Vision detections are transformed through camera -> body FRD -> local NED,
    associated against predicted confirmed landmarks with hard gates, then fused
    with robust static-landmark updates. Unmatched detections are kept as
    delayed-init candidate tracks until they are seen consistently.
    """

    def __init__(
        self,
        *args: Any,
        camera_tilt_deg: float = 20.0,
        min_candidate_hits: int = 5,
        confirmation_observations: int | None = None,
        association_position_gate_m: float = 0.8,
        association_orientation_gate_deg: float = 25.0,
        update_outlier_position_m: float = 1.2,
        candidate_association_gate_m: float = 1.5,
        candidate_timeout_s: float = 0.8,
        candidate_timeout_cycles: int | None = None,
        landmark_prune_timeout_s: float = 4.0,
        landmark_prune_min_observations: int = 8,
        merge_radius_m: float | None = None,
        min_candidate_position_confidence: float = 0.0,
        min_candidate_orientation_confidence: float = 0.0,
        min_promotion_position_confidence: float = 0.0,
        min_promotion_orientation_confidence: float = 0.0,
        measurement_position_variance_m2: float = 0.35 * 0.35,
        measurement_orientation_variance_rad2: float = np.deg2rad(20.0) ** 2,
        min_landmark_separation_m: float = 1.0,
        lock_observation_count: int = 20,
        lock_average_residual_m: float = 0.25,
        sequential_bias_cost: float = 0.35,
        horizontal_fov_deg: float = 90.0,
        vertical_fov_deg: float = 70.0,
        frustum_margin: float = 1.2,
        **kwargs: Any,
    ):
        self._landmarks: dict[int, GateLandmark] = {}
        self._candidates: dict[int, CandidateTrack] = {}
        self._next_landmark_id = 1
        self._next_candidate_id = 1
        self.next_expected_id: int | None = None
        self.origin_fixed = False

        if confirmation_observations is not None:
            min_candidate_hits = confirmation_observations
        if merge_radius_m is not None:
            association_position_gate_m = merge_radius_m
            candidate_association_gate_m = max(float(merge_radius_m), float(candidate_association_gate_m))

        self.camera_tilt_deg = float(camera_tilt_deg)
        self.min_candidate_hits = max(1, int(min_candidate_hits))
        self.confirmation_observations = self.min_candidate_hits
        self.association_position_gate_m = max(0.0, float(association_position_gate_m))
        self.merge_radius_m = self.association_position_gate_m
        self.association_orientation_gate_rad = np.deg2rad(max(0.0, float(association_orientation_gate_deg)))
        self.update_outlier_position_m = max(0.0, float(update_outlier_position_m))
        self.candidate_association_gate_m = max(0.0, float(candidate_association_gate_m))
        self.candidate_timeout_s = max(0.0, float(candidate_timeout_s))
        self.candidate_timeout_cycles = None if candidate_timeout_cycles is None else max(0, int(candidate_timeout_cycles))
        self.landmark_prune_timeout_s = max(0.0, float(landmark_prune_timeout_s))
        self.landmark_prune_min_observations = max(1, int(landmark_prune_min_observations))
        self.min_candidate_position_confidence = self._clamp_confidence(min_candidate_position_confidence)
        self.min_candidate_orientation_confidence = self._clamp_confidence(min_candidate_orientation_confidence)
        self.min_promotion_position_confidence = self._clamp_confidence(min_promotion_position_confidence)
        self.min_promotion_orientation_confidence = self._clamp_confidence(min_promotion_orientation_confidence)
        self.measurement_position_variance_m2 = max(1e-6, float(measurement_position_variance_m2))
        self.measurement_orientation_variance_rad2 = max(1e-6, float(measurement_orientation_variance_rad2))
        self.min_landmark_separation_m = max(0.0, float(min_landmark_separation_m))
        self.lock_observation_count = max(1, int(lock_observation_count))
        self.lock_average_residual_m = max(0.0, float(lock_average_residual_m))
        self.sequential_bias_cost = max(0.0, float(sequential_bias_cost))
        self.horizontal_fov_rad = np.deg2rad(max(1.0, float(horizontal_fov_deg)))
        self.vertical_fov_rad = np.deg2rad(max(1.0, float(vertical_fov_deg)))
        self.frustum_margin = max(1.0, float(frustum_margin))

    def add_or_update_gate(self, gate: GateRecord, *, allow_new: bool = True) -> GateRecord | None:
        landmark_id = self._landmark_id_from_gate_id(gate.gate_id)
        if landmark_id is None:
            if not allow_new:
                return None
            landmark = self._new_landmark_from_record(gate, self._time_from_cycle(gate.last_observed_cycle))
        else:
            landmark = self._landmarks[landmark_id]
            landmark.pose_position_ned_m = np.asarray(gate.position_local_ned_m, dtype=float)
            landmark.pose_quaternion_ned = tuple(float(value) for value in normalize_quaternion(gate.quaternion))
            landmark.observation_count = max(1, int(gate.observation_count))
            landmark.last_observed_cycle = gate.last_observed_cycle
            landmark.last_seen_time_s = gate.last_seen_time_s or self._time_from_cycle(gate.last_observed_cycle)
            landmark.sequential_index = gate.sequence
            landmark.crossed = bool(gate.crossed)
            landmark.locked = bool(gate.locked or gate.frozen)
            landmark.source_gate_id = gate.gate_id
            landmark.position_confidence = self._clamp_confidence(gate.position_confidence)
            landmark.orientation_confidence = self._clamp_confidence(gate.quaternion_confidence)
        self._landmarks[landmark.id] = landmark
        self._refresh_next_expected()
        return self._record_from_landmark(landmark)

    def clear(self) -> None:
        self._landmarks.clear()
        self._candidates.clear()
        self._next_landmark_id = 1
        self._next_candidate_id = 1
        self.next_expected_id = None
        self.origin_fixed = False

    def get_gate(self, gate_id: str) -> GateRecord | None:
        landmark_id = self._landmark_id_from_gate_id(gate_id)
        if landmark_id is None:
            return None
        return self._record_from_landmark(self._landmarks[landmark_id])

    @property
    def gates(self) -> list[GateRecord]:
        return [
            self._record_from_landmark(landmark)
            for landmark in sorted(
                self._landmarks.values(),
                key=lambda item: (
                    item.sequential_index is None,
                    item.sequential_index if item.sequential_index is not None else item.id,
                    item.id,
                ),
            )
        ]

    @property
    def candidate_gates(self) -> list[GateRecord]:
        return [
            self._record_from_candidate(candidate)
            for candidate in sorted(self._candidates.values(), key=lambda item: item.id)
        ]

    def mark_crossed(self, gate_id: str) -> None:
        landmark_id = self._landmark_id_from_gate_id(gate_id)
        if landmark_id is None:
            raise KeyError(gate_id)
        self._landmarks[landmark_id].crossed = True
        self._refresh_next_expected()

    def mark_passed_near_position(
        self,
        position_local_ned_m: tuple[float, float, float],
        *,
        distance_m: float = 2.0,
    ) -> list[GateRecord]:
        position = np.asarray(position_local_ned_m, dtype=float)
        threshold_m = max(0.0, float(distance_m))
        passed: list[GateRecord] = []
        for landmark in self._landmarks.values():
            if landmark.crossed:
                continue
            if float(np.linalg.norm(landmark.pose_position_ned_m - position)) <= threshold_m:
                landmark.crossed = True
                passed.append(self._record_from_landmark(landmark))
        self._refresh_next_expected()
        return passed

    def update_from_observation(
        self,
        observation: VisionObservation,
        vehicle_state: VehicleState,
        allow_new_gates: bool = True,
    ) -> list[GateRecord]:
        now_s = self._observation_time_s(observation)
        self._cleanup_candidates(now_s, observation.frame_id)
        predictions = self._predict_landmarks(vehicle_state)
        observed_records = [
            self.gate_record_from_observation(
                observed_gate,
                vehicle_state=vehicle_state,
                sequence=index,
                observed_cycle=observation.frame_id,
                observed_time_s=now_s,
            )
            for index, observed_gate in enumerate(observation.gates)
            if self._raw_observation_meets_candidate_confidence(observed_gate)
        ]

        matches, unmatched_indices = self._associate(observed_records, predictions)
        updated: list[GateRecord] = []
        matched_indices: set[int] = set()
        for observation_index, landmark_id in matches:
            matched_indices.add(observation_index)
            record = self._update_landmark(
                self._landmarks[landmark_id],
                observed_records[observation_index],
                now_s=now_s,
            )
            if record is not None:
                updated.append(record)

        if allow_new_gates:
            for index in unmatched_indices:
                if index in matched_indices:
                    continue
                promoted = self._update_candidates(observed_records[index], now_s=now_s)
                if promoted is not None:
                    updated.append(promoted)

        self._prune_landmarks(now_s)
        self._refresh_next_expected()
        return updated

    def gate_record_from_observation(
        self,
        observation: VisionGateObservation,
        *,
        vehicle_state: VehicleState,
        sequence: int | None = None,
        observed_cycle: int | None = None,
        observed_time_s: float | None = None,
    ) -> GateRecord:
        position_body_frd_m = self.camera_to_body_frd(observation.position_camera_m)
        position_local_ned_m = self.body_relative_frd_to_local_ned_m(
            position_body_frd_m,
            vehicle_state=vehicle_state,
        )
        return GateRecord(
            gate_id=observation.gate_id,
            position_local_ned_m=position_local_ned_m,
            quaternion=self.camera_orientation_to_local_ned_quaternion(
                observation.orientation_camera,
                vehicle_state=vehicle_state,
            ),
            position_confidence=float(observation.position_confidence),
            quaternion_confidence=float(observation.orientation_confidence),
            sequence=sequence,
            observation_count=1,
            last_observed_cycle=observed_cycle,
            source="vision",
            last_seen_time_s=observed_time_s,
        )

    def observed_gate_records_for_planning(
        self,
        observation: VisionObservation,
        *,
        vehicle_state: VehicleState,
        min_position_confidence: float = 0.0,
        require_forward_camera_position: bool = True,
    ) -> list[GateRecord]:
        records: list[GateRecord] = []
        min_confidence = self._clamp_confidence(min_position_confidence)
        now_s = self._observation_time_s(observation)
        for sequence, observed_gate in enumerate(observation.gates):
            if float(observed_gate.position_confidence) < min_confidence:
                continue
            if require_forward_camera_position and float(observed_gate.position_camera_m[2]) <= 0.0:
                continue
            records.append(
                self.gate_record_from_observation(
                    observed_gate,
                    vehicle_state=vehicle_state,
                    sequence=sequence,
                    observed_cycle=observation.frame_id,
                    observed_time_s=now_s,
                )
            )
        return records

    def _predict_landmarks(self, vehicle_state: VehicleState) -> dict[int, _PredictedLandmark]:
        predictions: dict[int, _PredictedLandmark] = {}
        for landmark_id, landmark in self._landmarks.items():
            position_camera = self.local_ned_position_to_camera(
                landmark.pose_position_ned_m,
                vehicle_state=vehicle_state,
            )
            predictions[landmark_id] = _PredictedLandmark(
                landmark_id=landmark_id,
                position_camera_m=position_camera,
                visible=self._inside_camera_frustum(position_camera),
            )
        return predictions

    def _associate(
        self,
        observed_records: list[GateRecord],
        predictions: dict[int, _PredictedLandmark],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        costs: list[tuple[float, int, int]] = []
        for observation_index, observed in enumerate(observed_records):
            observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
            for landmark_id, prediction in predictions.items():
                landmark = self._landmarks[landmark_id]
                if not prediction.visible:
                    continue
                position_residual = float(np.linalg.norm(landmark.pose_position_ned_m - observed_position))
                if position_residual > self.association_position_gate_m:
                    continue
                orientation_residual = self._quaternion_angle_rad(
                    landmark.pose_quaternion_ned,
                    observed.quaternion,
                )
                if orientation_residual > self.association_orientation_gate_rad:
                    continue
                position_cost = position_residual / max(self.association_position_gate_m, 1e-6)
                orientation_cost = orientation_residual / max(self.association_orientation_gate_rad, 1e-6)
                cost = position_cost * position_cost + orientation_cost * orientation_cost
                if landmark_id == self.next_expected_id:
                    cost *= self.sequential_bias_cost
                costs.append((cost, observation_index, landmark_id))

        matches: list[tuple[int, int]] = []
        used_observations: set[int] = set()
        used_landmarks: set[int] = set()
        for _, observation_index, landmark_id in sorted(costs, key=lambda item: item[0]):
            if observation_index in used_observations or landmark_id in used_landmarks:
                continue
            used_observations.add(observation_index)
            used_landmarks.add(landmark_id)
            matches.append((observation_index, landmark_id))
        unmatched = [index for index in range(len(observed_records)) if index not in used_observations]
        return matches, unmatched

    def _update_landmark(self, landmark: GateLandmark, observed: GateRecord, *, now_s: float) -> GateRecord | None:
        observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
        residual_m = float(np.linalg.norm(observed_position - landmark.pose_position_ned_m))
        if residual_m > self.update_outlier_position_m:
            return None

        measurement_weight = self._measurement_weight(observed.position_confidence)
        if not landmark.locked:
            prior_weight = max(float(landmark.observation_count), 1.0)
            alpha = measurement_weight / (prior_weight + measurement_weight)
            landmark.pose_position_ned_m = (1.0 - alpha) * landmark.pose_position_ned_m + alpha * observed_position
            landmark.pose_quaternion_ned = self._blend_quaternions(
                landmark.pose_quaternion_ned,
                observed.quaternion,
                self._measurement_weight(observed.quaternion_confidence)
                / (prior_weight + self._measurement_weight(observed.quaternion_confidence)),
            )
            position_info = 1.0 / self.measurement_position_variance_m2
            orientation_info = 1.0 / self.measurement_orientation_variance_rad2
            landmark.information += np.diag([position_info] * 3 + [orientation_info] * 3) * measurement_weight

        landmark.observation_count += 1
        landmark.last_seen_time_s = now_s
        landmark.last_observed_cycle = observed.last_observed_cycle
        landmark.average_residual_m = self._running_average(
            landmark.average_residual_m,
            residual_m,
            landmark.observation_count,
        )
        landmark.position_confidence = max(landmark.position_confidence, observed.position_confidence)
        landmark.orientation_confidence = max(landmark.orientation_confidence, observed.quaternion_confidence)
        if (
            landmark.observation_count >= self.lock_observation_count
            and landmark.average_residual_m <= self.lock_average_residual_m
        ):
            landmark.locked = True
        return self._record_from_landmark(landmark)

    def _update_candidates(self, observed: GateRecord, *, now_s: float) -> GateRecord | None:
        if not self._meets_candidate_confidence(observed):
            return None

        candidate_id = self._matching_candidate_id(observed)
        if candidate_id is None:
            candidate = CandidateTrack(
                id=self._next_candidate_id,
                estimated_position_ned_m=np.asarray(observed.position_local_ned_m, dtype=float),
                estimated_quaternion_ned=observed.quaternion,
                observations=[observed],
                hit_count=1,
                last_seen_time_s=now_s,
                last_observed_cycle=observed.last_observed_cycle,
                source_gate_id=observed.gate_id,
                sequential_index=observed.sequence,
            )
            self._next_candidate_id += 1
            self._candidates[candidate.id] = candidate
            return None

        candidate = self._candidates[candidate_id]
        residual_m = float(
            np.linalg.norm(
                np.asarray(observed.position_local_ned_m, dtype=float)
                - candidate.estimated_position_ned_m
            )
        )
        if residual_m > self.candidate_association_gate_m:
            return None
        candidate.observations.append(observed)
        candidate.observations = candidate.observations[-max(self.min_candidate_hits * 2, 8):]
        candidate.hit_count += 1
        candidate.last_seen_time_s = now_s
        candidate.last_observed_cycle = observed.last_observed_cycle
        alpha = 1.0 / max(float(candidate.hit_count), 1.0)
        candidate.estimated_position_ned_m = (
            (1.0 - alpha) * candidate.estimated_position_ned_m
            + alpha * np.asarray(observed.position_local_ned_m, dtype=float)
        )
        candidate.estimated_quaternion_ned = self._blend_quaternions(
            candidate.estimated_quaternion_ned,
            observed.quaternion,
            alpha,
        )
        candidate.average_residual_m = self._running_average(candidate.average_residual_m, residual_m, candidate.hit_count)
        if observed.sequence is not None and candidate.sequential_index is None:
            candidate.sequential_index = observed.sequence

        if not self._candidate_ready_for_promotion(candidate):
            return None

        landmark = self._promote_candidate(candidate)
        self._candidates.pop(candidate.id, None)
        return self._record_from_landmark(landmark)

    def _candidate_ready_for_promotion(self, candidate: CandidateTrack) -> bool:
        if candidate.hit_count < self.min_candidate_hits:
            return False
        if candidate.average_residual_m > max(self.association_position_gate_m, self.candidate_association_gate_m * 0.75):
            return False
        latest = candidate.observations[-1]
        if not self._meets_promotion_confidence(latest):
            return False
        for landmark in self._landmarks.values():
            distance = float(np.linalg.norm(landmark.pose_position_ned_m - candidate.estimated_position_ned_m))
            if distance < self.min_landmark_separation_m:
                return False
        return True

    def _promote_candidate(self, candidate: CandidateTrack) -> GateLandmark:
        latest = candidate.observations[-1]
        landmark = self._new_landmark_from_record(
            replace(
                latest,
                gate_id=latest.gate_id or f"gate-{self._next_landmark_id:03d}",
                position_local_ned_m=vec3(candidate.estimated_position_ned_m),
                quaternion=candidate.estimated_quaternion_ned,
                observation_count=candidate.hit_count,
                average_residual_m=candidate.average_residual_m,
                sequence=self._next_sequence_index(candidate),
                last_seen_time_s=candidate.last_seen_time_s,
            ),
            candidate.last_seen_time_s,
        )
        self._landmarks[landmark.id] = landmark
        self.origin_fixed = True
        return landmark

    def _new_landmark_from_record(self, record: GateRecord, now_s: float) -> GateLandmark:
        landmark_id = self._next_landmark_id
        self._next_landmark_id += 1
        position_info = 1.0 / self.measurement_position_variance_m2
        orientation_info = 1.0 / self.measurement_orientation_variance_rad2
        return GateLandmark(
            id=landmark_id,
            pose_position_ned_m=np.asarray(record.position_local_ned_m, dtype=float),
            pose_quaternion_ned=tuple(float(value) for value in normalize_quaternion(record.quaternion)),
            information=np.diag([position_info] * 3 + [orientation_info] * 3),
            observation_count=max(1, int(record.observation_count)),
            last_seen_time_s=record.last_seen_time_s or now_s,
            last_observed_cycle=record.last_observed_cycle,
            average_residual_m=float(record.average_residual_m),
            sequential_index=record.sequence,
            locked=bool(record.locked or record.frozen),
            crossed=bool(record.crossed),
            source_gate_id=record.gate_id or f"gate-{landmark_id:03d}",
            position_confidence=self._clamp_confidence(record.position_confidence),
            orientation_confidence=self._clamp_confidence(record.quaternion_confidence),
        )

    def _cleanup_candidates(self, now_s: float, observed_cycle: int | None) -> None:
        keep: dict[int, CandidateTrack] = {}
        for candidate_id, candidate in self._candidates.items():
            stale_by_time = self.candidate_timeout_s > 0.0 and now_s - candidate.last_seen_time_s > self.candidate_timeout_s
            stale_by_cycle = (
                self.candidate_timeout_cycles is not None
                and observed_cycle is not None
                and candidate.last_observed_cycle is not None
                and int(observed_cycle) - int(candidate.last_observed_cycle) > self.candidate_timeout_cycles
            )
            if not stale_by_time and not stale_by_cycle:
                keep[candidate_id] = candidate
        self._candidates = keep

    def _prune_landmarks(self, now_s: float) -> None:
        self._landmarks = {
            landmark_id: landmark
            for landmark_id, landmark in self._landmarks.items()
            if not (
                landmark.observation_count < self.landmark_prune_min_observations
                and now_s - landmark.last_seen_time_s > self.landmark_prune_timeout_s
            )
        }

    def _record_from_landmark(self, landmark: GateLandmark) -> GateRecord:
        gate_id = landmark.source_gate_id or f"gate-{landmark.id:03d}"
        return GateRecord(
            gate_id=gate_id,
            position_local_ned_m=vec3(landmark.pose_position_ned_m),
            quaternion=tuple(float(value) for value in normalize_quaternion(landmark.pose_quaternion_ned)),
            position_confidence=float(landmark.position_confidence),
            quaternion_confidence=float(landmark.orientation_confidence),
            crossed=bool(landmark.crossed),
            frozen=bool(landmark.locked),
            sequence=landmark.sequential_index,
            observation_count=int(landmark.observation_count),
            last_observed_cycle=landmark.last_observed_cycle,
            source="vision_map",
            average_residual_m=float(landmark.average_residual_m),
            last_seen_time_s=float(landmark.last_seen_time_s),
            locked=bool(landmark.locked),
        )

    def _record_from_candidate(self, candidate: CandidateTrack) -> GateRecord:
        latest = candidate.observations[-1] if candidate.observations else None
        return GateRecord(
            gate_id=candidate.source_gate_id or f"candidate-{candidate.id:03d}",
            position_local_ned_m=vec3(candidate.estimated_position_ned_m),
            quaternion=tuple(float(value) for value in normalize_quaternion(candidate.estimated_quaternion_ned)),
            position_confidence=0.0 if latest is None else float(latest.position_confidence),
            quaternion_confidence=0.0 if latest is None else float(latest.quaternion_confidence),
            sequence=candidate.sequential_index,
            observation_count=int(candidate.hit_count),
            last_observed_cycle=candidate.last_observed_cycle,
            source="vision_candidate",
            average_residual_m=float(candidate.average_residual_m),
            last_seen_time_s=float(candidate.last_seen_time_s),
        )

    def camera_to_body_frd(self, vector_camera: tuple[float, float, float]) -> tuple[float, float, float]:
        """Rotate camera optical [right, up, forward] into body FRD [forward, right, down]."""

        return vec3(self.camera_to_body_frd_rotation() @ np.asarray(vector_camera, dtype=float))

    def local_ned_position_to_camera(
        self,
        position_local_ned_m: tuple[float, float, float] | np.ndarray,
        *,
        vehicle_state: VehicleState,
    ) -> np.ndarray:
        rotation_body_to_local = rotation_matrix_from_quaternion(vehicle_state.attitude_quaternion)
        relative_local = np.asarray(position_local_ned_m, dtype=float) - np.asarray(vehicle_state.position_local_ned_m, dtype=float)
        relative_body = rotation_body_to_local.T @ relative_local
        return self.camera_to_body_frd_rotation().T @ relative_body

    def camera_orientation_to_local_ned_quaternion(
        self,
        orientation_camera: tuple[float, float, float] | None,
        *,
        vehicle_state: VehicleState,
    ) -> tuple[float, float, float, float]:
        rotation_gate_to_local = self.camera_orientation_to_local_ned_rotation(
            orientation_camera,
            vehicle_state=vehicle_state,
        )
        quaternion = quaternion_from_rotation_matrix(rotation_gate_to_local)
        return tuple(float(value) for value in normalize_quaternion(quaternion))

    def camera_orientation_to_local_ned_rotation(
        self,
        orientation_camera: tuple[float, float, float] | None,
        *,
        vehicle_state: VehicleState,
    ) -> np.ndarray:
        rotation_body_to_local = rotation_matrix_from_quaternion(vehicle_state.attitude_quaternion)
        rotation_camera_to_body = self.camera_frd_to_body_frd_rotation()
        rotation_gate_to_camera = self.camera_to_gate_frd_rotation(orientation_camera)
        return rotation_body_to_local @ rotation_camera_to_body @ rotation_gate_to_camera

    def camera_to_gate_frd_rotation(self, orientation_camera: tuple[float, float, float] | None) -> np.ndarray:
        """Return gate FRD axes in camera FRD coordinates from camera-relative RPY degrees."""

        if orientation_camera is None:
            return np.eye(3)
        roll_right_deg, pitch_deg, yaw_right_deg = orientation_camera
        return rotation_matrix_from_quaternion(
            quaternion_from_roll_pitch_yaw_deg(
                float(roll_right_deg),
                float(pitch_deg),
                float(yaw_right_deg),
            )
        )

    def body_relative_frd_to_local_ned_m(
        self,
        position_body_frd_m: tuple[float, float, float],
        *,
        vehicle_state: VehicleState,
    ) -> tuple[float, float, float]:
        relative_local_ned = self.body_vector_frd_to_local_ned(position_body_frd_m, vehicle_state=vehicle_state)
        return vec3(np.asarray(vehicle_state.position_local_ned_m, dtype=float) + np.asarray(relative_local_ned))

    def body_vector_frd_to_local_ned(
        self,
        vector_body_frd: tuple[float, float, float],
        *,
        vehicle_state: VehicleState,
    ) -> tuple[float, float, float]:
        rotation_body_to_local_ned = rotation_matrix_from_quaternion(vehicle_state.attitude_quaternion)
        return vec3(rotation_body_to_local_ned @ np.asarray(vector_body_frd, dtype=float))

    def camera_to_body_frd_rotation(self) -> np.ndarray:
        tilt_rad = np.deg2rad(self.camera_tilt_deg)
        cos_tilt = np.cos(tilt_rad)
        sin_tilt = np.sin(tilt_rad)

        camera_optical_to_body_frd = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        camera_pitch_up_in_body_frd = np.array(
            [
                [cos_tilt, 0.0, sin_tilt],
                [0.0, 1.0, 0.0],
                [-sin_tilt, 0.0, cos_tilt],
            ],
            dtype=float,
        )
        return camera_pitch_up_in_body_frd @ camera_optical_to_body_frd

    def camera_frd_to_body_frd_rotation(self) -> np.ndarray:
        """Map camera FRD [forward, right, down] into body FRD."""

        camera_optical_from_camera_frd = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        return self.camera_to_body_frd_rotation() @ camera_optical_from_camera_frd

    def _inside_camera_frustum(self, position_camera_m: np.ndarray) -> bool:
        forward = float(position_camera_m[2])
        if forward <= 0.05:
            return False
        horizontal = abs(float(position_camera_m[0]) / forward)
        vertical = abs(float(position_camera_m[1]) / forward)
        return (
            horizontal <= np.tan(self.horizontal_fov_rad / 2.0) * self.frustum_margin
            and vertical <= np.tan(self.vertical_fov_rad / 2.0) * self.frustum_margin
        )

    def _matching_candidate_id(self, observed: GateRecord) -> int | None:
        observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
        closest_id: int | None = None
        closest_distance = self.candidate_association_gate_m
        for candidate_id, candidate in self._candidates.items():
            distance = float(np.linalg.norm(candidate.estimated_position_ned_m - observed_position))
            if distance <= closest_distance:
                closest_id = candidate_id
                closest_distance = distance
        return closest_id

    def _landmark_id_from_gate_id(self, gate_id: str) -> int | None:
        for landmark_id, landmark in self._landmarks.items():
            if landmark.source_gate_id == gate_id or f"gate-{landmark.id:03d}" == gate_id:
                return landmark_id
        return None

    def _next_sequence_index(self, candidate: CandidateTrack) -> int:
        existing = [
            landmark.sequential_index
            for landmark in self._landmarks.values()
            if landmark.sequential_index is not None
        ]
        if candidate.sequential_index is not None and candidate.sequential_index not in existing:
            return candidate.sequential_index
        return 0 if not existing else max(existing) + 1

    def _refresh_next_expected(self) -> None:
        ordered = sorted(
            (landmark for landmark in self._landmarks.values() if not landmark.crossed),
            key=lambda item: (
                item.sequential_index is None,
                item.sequential_index if item.sequential_index is not None else item.id,
                item.id,
            ),
        )
        self.next_expected_id = None if not ordered else ordered[0].id

    @staticmethod
    def _running_average(current: float, observed: float, count_after_update: int) -> float:
        count = max(1, int(count_after_update))
        return float(current + (observed - current) / count)

    @staticmethod
    def _blend_quaternions(
        current: tuple[float, float, float, float],
        observed: tuple[float, float, float, float],
        alpha: float,
    ) -> tuple[float, float, float, float]:
        current_q = normalize_quaternion(current)
        observed_q = normalize_quaternion(observed)
        if float(np.dot(current_q, observed_q)) < 0.0:
            observed_q = -observed_q
        blended = (1.0 - float(np.clip(alpha, 0.0, 1.0))) * current_q + float(np.clip(alpha, 0.0, 1.0)) * observed_q
        return tuple(float(value) for value in normalize_quaternion(blended))

    @staticmethod
    def _quaternion_angle_rad(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        q1 = normalize_quaternion(first)
        q2 = normalize_quaternion(second)
        dot = abs(float(np.dot(q1, q2)))
        return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _time_from_cycle(cycle: int | None) -> float:
        return 0.0 if cycle is None else float(cycle) / 10.0

    @staticmethod
    def _observation_time_s(observation: VisionObservation) -> float:
        if int(observation.sim_time_ns) > 0:
            return float(observation.sim_time_ns) * 1e-9
        return float(observation.frame_id) / 10.0

    def _measurement_weight(self, confidence: float) -> float:
        return max(0.05, self._clamp_confidence(confidence))

    def _raw_observation_meets_candidate_confidence(self, observation: VisionGateObservation) -> bool:
        return (
            float(observation.position_confidence) >= self.min_candidate_position_confidence
            and float(observation.orientation_confidence) >= self.min_candidate_orientation_confidence
        )

    def _meets_candidate_confidence(self, gate: GateRecord) -> bool:
        return (
            float(gate.position_confidence) >= self.min_candidate_position_confidence
            and float(gate.quaternion_confidence) >= self.min_candidate_orientation_confidence
        )

    def _meets_promotion_confidence(self, gate: GateRecord) -> bool:
        return (
            float(gate.position_confidence) >= self.min_promotion_position_confidence
            and float(gate.quaternion_confidence) >= self.min_promotion_orientation_confidence
        )
