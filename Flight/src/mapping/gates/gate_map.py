from dataclasses import dataclass, replace
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


class GateMap:
    """Gate map with a candidate stage for raw vision observations.

    Raw observed gates are accumulated in ``_candidate_gates`` until they are
    observed enough times to be promoted into ``_gates``. Planner callers that
    read ``gates`` only see confirmed records.
    """

    def __init__(
        self,
        *args: Any,
        camera_tilt_deg: float = 20.0,
        confirmation_observations: int = 3,
        merge_radius_m: float = 1.0,
        candidate_timeout_cycles: int = 30,
        min_candidate_position_confidence: float = 0.85,
        min_candidate_orientation_confidence: float = 0.85,
        min_promotion_position_confidence: float = 0.95,
        min_promotion_orientation_confidence: float = 0.95,
        **kwargs: Any,
    ):
        self._gates: dict[str, GateRecord] = {}
        self._candidate_gates: dict[str, GateRecord] = {}
        self.camera_tilt_deg = float(camera_tilt_deg)
        self.confirmation_observations = max(1, int(confirmation_observations))
        self.merge_radius_m = max(0.0, float(merge_radius_m))
        self.candidate_timeout_cycles = max(0, int(candidate_timeout_cycles))
        self.min_candidate_position_confidence = self._clamp_confidence(min_candidate_position_confidence)
        self.min_candidate_orientation_confidence = self._clamp_confidence(min_candidate_orientation_confidence)
        self.min_promotion_position_confidence = self._clamp_confidence(min_promotion_position_confidence)
        self.min_promotion_orientation_confidence = self._clamp_confidence(min_promotion_orientation_confidence)

    def add_or_update_gate(self, gate: GateRecord, *, allow_new: bool = True) -> GateRecord | None:
        if not allow_new and gate.gate_id not in self._gates:
            return None
        self._gates[gate.gate_id] = gate
        return gate

    def clear(self) -> None:
        self._gates.clear()
        self._candidate_gates.clear()

    def get_gate(self, gate_id: str) -> GateRecord | None:
        return self._gates.get(gate_id)

    @property
    def gates(self) -> list[GateRecord]:
        return list(self._gates.values())

    @property
    def candidate_gates(self) -> list[GateRecord]:
        return list(self._candidate_gates.values())

    def mark_crossed(self, gate_id: str) -> None:
        self._gates[gate_id].crossed = True

    def mark_passed_near_position(
        self,
        position_local_ned_m: tuple[float, float, float],
        *,
        distance_m: float = 2.0,
    ) -> list[GateRecord]:
        position = np.asarray(position_local_ned_m, dtype=float)
        threshold_m = max(0.0, float(distance_m))
        passed: list[GateRecord] = []
        for gate in self._gates.values():
            if gate.crossed:
                continue
            gate_position = np.asarray(gate.position_local_ned_m, dtype=float)
            if float(np.linalg.norm(gate_position - position)) <= threshold_m:
                gate.crossed = True
                passed.append(gate)
        return passed

    def update_from_observation(
        self,
        observation: VisionObservation,
        vehicle_state: VehicleState,
        allow_new_gates: bool = True,
    ) -> list[GateRecord]:
        self._cleanup_candidates(observation.frame_id)
        records: list[GateRecord] = []
        for sequence, observed_gate in enumerate(observation.gates):
            record = self.gate_record_from_observation(
                observed_gate,
                vehicle_state=vehicle_state,
                sequence=sequence,
                observed_cycle=observation.frame_id,
            )
            mapped = self._update_from_candidate(record, allow_new=allow_new_gates)
            if mapped is not None:
                records.append(mapped)
        return records

    def _update_from_candidate(self, observed: GateRecord, *, allow_new: bool = True) -> GateRecord | None:
        if not self._meets_candidate_confidence(observed):
            return None

        confirmed_id = self._matching_gate_id(self._gates, observed)
        if confirmed_id is not None:
            merged = self._merge_records(self._gates[confirmed_id], observed)
            if confirmed_id != merged.gate_id:
                self._gates.pop(confirmed_id)
            self._gates[merged.gate_id] = merged
            return merged

        if not allow_new:
            return None

        candidate_id = self._matching_gate_id(self._candidate_gates, observed)
        if candidate_id is None:
            self._candidate_gates[observed.gate_id] = observed
            return None

        candidate = self._merge_records(self._candidate_gates[candidate_id], observed)
        if candidate_id != candidate.gate_id:
            self._candidate_gates.pop(candidate_id)
        self._candidate_gates[candidate.gate_id] = candidate

        if candidate.observation_count < self.confirmation_observations or not self._meets_promotion_confidence(observed):
            return None

        self._candidate_gates.pop(candidate.gate_id, None)
        self._gates[candidate.gate_id] = candidate
        return candidate

    def _cleanup_candidates(self, observed_cycle: int | None) -> None:
        if observed_cycle is None or self.candidate_timeout_cycles <= 0:
            return
        self._candidate_gates = {
            gate_id: gate
            for gate_id, gate in self._candidate_gates.items()
            if gate.last_observed_cycle is None
            or int(observed_cycle) - int(gate.last_observed_cycle) <= self.candidate_timeout_cycles
        }

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

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

    def _matching_gate_id(self, gates: dict[str, GateRecord], observed: GateRecord) -> str | None:
        if observed.gate_id in gates:
            return observed.gate_id
        if self.merge_radius_m <= 0.0:
            return None

        observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
        closest_id: str | None = None
        closest_distance = self.merge_radius_m
        for gate_id, gate in gates.items():
            distance = float(np.linalg.norm(np.asarray(gate.position_local_ned_m, dtype=float) - observed_position))
            if distance <= closest_distance:
                closest_id = gate_id
                closest_distance = distance
        return closest_id

    def _merge_records(self, current: GateRecord, observed: GateRecord) -> GateRecord:
        current_count = max(0, int(current.observation_count))
        observed_count = max(1, int(observed.observation_count))
        total_count = current_count + observed_count
        position = (
            (
                np.asarray(current.position_local_ned_m, dtype=float) * current_count
                + np.asarray(observed.position_local_ned_m, dtype=float) * observed_count
            )
            / total_count
        )
        use_observed_quaternion = observed.quaternion_confidence >= current.quaternion_confidence
        return replace(
            current,
            gate_id=current.gate_id or observed.gate_id,
            position_local_ned_m=vec3(position),
            quaternion=observed.quaternion if use_observed_quaternion else current.quaternion,
            position_confidence=max(float(current.position_confidence), float(observed.position_confidence)),
            quaternion_confidence=max(float(current.quaternion_confidence), float(observed.quaternion_confidence)),
            sequence=current.sequence if current.sequence is not None else observed.sequence,
            observation_count=total_count,
            last_observed_cycle=observed.last_observed_cycle,
            source=observed.source or current.source,
        )

    def gate_record_from_observation(
        self,
        observation: VisionGateObservation,
        *,
        vehicle_state: VehicleState,
        sequence: int | None = None,
        observed_cycle: int | None = None,
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
                )
            )
        return records

    def camera_to_body_frd(self, vector_camera: tuple[float, float, float]) -> tuple[float, float, float]:
        """Rotate camera optical [right, up, forward] into body FRD [forward, right, down]."""

        return vec3(self.camera_to_body_frd_rotation() @ np.asarray(vector_camera, dtype=float))

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
