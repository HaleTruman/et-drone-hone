from dataclasses import dataclass
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
    """Placeholder gate map storage while mapping is rebuilt from scratch."""

    def __init__(self, *args: Any, camera_tilt_deg: float = 20.0, **kwargs: Any):
        self._gates: dict[str, GateRecord] = {}
        self.camera_tilt_deg = float(camera_tilt_deg)

    def add_or_update_gate(self, gate: GateRecord, *, allow_new: bool = True) -> GateRecord | None:
        if not allow_new and gate.gate_id not in self._gates:
            return None
        self._gates[gate.gate_id] = gate
        return gate

    def clear(self) -> None:
        self._gates.clear()

    def get_gate(self, gate_id: str) -> GateRecord | None:
        return self._gates.get(gate_id)

    @property
    def gates(self) -> list[GateRecord]:
        return list(self._gates.values())

    def mark_crossed(self, gate_id: str) -> None:
        self._gates[gate_id].crossed = True

    def update_from_observation(
        self,
        observation: VisionObservation,
        vehicle_state: VehicleState,
        allow_new_gates: bool = True,
    ) -> list[GateRecord]:
        records: list[GateRecord] = []
        for sequence, observed_gate in enumerate(observation.gates):
            record = self.gate_record_from_observation(
                observed_gate,
                vehicle_state=vehicle_state,
                sequence=sequence,
                observed_cycle=observation.frame_id,
            )
            mapped = self.add_or_update_gate(record, allow_new=allow_new_gates)
            if mapped is not None:
                records.append(mapped)
        return records

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
