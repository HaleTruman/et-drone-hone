from typing import Any

import numpy as np

from .gate_map import GateMap, GateRecord
from .vision_observation import VisionGateObservation, VisionObservation


class GatePoseEstimator:
    def __init__(
        self,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 320.0,
        cy: float = 180.0,
        camera_tilt_deg: float = 20.0,
    ):
        self.intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.camera_tilt_deg = camera_tilt_deg

    def estimate_gate_pose(
        self,
        observation: VisionGateObservation | dict[str, Any],
        *,
        vehicle_position_local_ned_m: np.ndarray | tuple[float, float, float],
        attitude_quaternion: np.ndarray | tuple[float, float, float, float] | None = None,
        rotation_body_to_ned: np.ndarray | None = None,
        sequence: int | None = None,
        observed_cycle: int | None = None,
    ) -> GateRecord:
        gate = (
            observation
            if isinstance(observation, VisionGateObservation)
            else VisionGateObservation.from_payload(observation)
        )
        body_to_ned = (
            np.asarray(rotation_body_to_ned, dtype=float)
            if rotation_body_to_ned is not None
            else self.quaternion_to_rotation(attitude_quaternion)
        )
        camera_to_body = self.camera_optical_to_body_transform()
        position_body = camera_to_body @ np.asarray(gate.position_camera_m, dtype=float)
        position_relative_ned = body_to_ned @ position_body
        position_local = np.asarray(vehicle_position_local_ned_m, dtype=float) + position_relative_ned

        normal_local = None
        if gate.orientation_camera is not None:
            orientation_body = camera_to_body @ np.asarray(gate.orientation_camera, dtype=float)
            normal_local = body_to_ned @ orientation_body

        return GateRecord(
            gate_id=gate.gate_id,
            position_local_ned_m=tuple(float(value) for value in position_local),
            position_relative_ned_m=tuple(float(value) for value in position_relative_ned),
            quaternion=self.normal_to_quaternion(normal_local),
            confidence=float(gate.position_confidence),
            sequence=sequence,
            last_observed_cycle=observed_cycle,
        )

    def update_gate_map_from_observation(
        self,
        observation: VisionObservation,
        *,
        telemetry: Any,
        gate_map: GateMap,
        allow_new_gates: bool = True,
    ) -> list[GateRecord]:
        odometry = getattr(telemetry, "odometry", None)
        vehicle_position = None if odometry is None else odometry.position_local_ned_m
        if vehicle_position is None:
            raise ValueError("Telemetry must include position_local_ned_m to map camera-local gates.")
        records: list[GateRecord] = []
        for sequence, gate in enumerate(observation.gates):
            record = self.estimate_gate_pose(
                gate,
                vehicle_position_local_ned_m=vehicle_position,
                attitude_quaternion=odometry.attitude_quaternion,
                sequence=sequence,
                observed_cycle=observation.frame_id,
            )
            mapped = gate_map.add_or_update_gate(record, allow_new=allow_new_gates)
            if mapped is not None:
                records.append(mapped)
        return records

    def camera_to_body_transform(self) -> np.ndarray:
        return self.camera_optical_to_body_transform()

    def camera_optical_to_body_transform(self) -> np.ndarray:
        """Map camera optical [right, up, forward] into body FRD [forward, right, down]."""

        tilt = np.deg2rad(self.camera_tilt_deg)
        c, s = np.cos(tilt), np.sin(tilt)
        optical_to_body_frd = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        upward_camera_tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return upward_camera_tilt @ optical_to_body_frd

    def body_to_local_ned(self, vector_body: np.ndarray, rotation_body_to_ned: np.ndarray) -> np.ndarray:
        return np.asarray(rotation_body_to_ned, dtype=float) @ np.asarray(vector_body, dtype=float)

    def quaternion_to_rotation(self, quaternion: np.ndarray | tuple[float, float, float, float] | None) -> np.ndarray:
        if quaternion is None:
            return np.eye(3)
        q = np.asarray(quaternion, dtype=float)
        q = q / max(np.linalg.norm(q), 1e-12)
        qw, qx, qy, qz = q
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

    def normal_to_quaternion(self, normal_local_ned: np.ndarray | None) -> tuple[float, float, float, float]:
        if normal_local_ned is None:
            return (1.0, 0.0, 0.0, 0.0)
        x_axis = np.asarray(normal_local_ned, dtype=float)
        norm = np.linalg.norm(x_axis)
        if norm <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        x_axis = x_axis / norm
        z_reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(x_axis, z_reference))) > 0.95:
            z_reference = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_reference, x_axis)
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
        z_axis = np.cross(x_axis, y_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        return self.rotation_to_quaternion(rotation)

    def rotation_to_quaternion(self, rotation: np.ndarray) -> tuple[float, float, float, float]:
        matrix = np.asarray(rotation, dtype=float)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                qw = (matrix[2, 1] - matrix[1, 2]) / scale
                qx = 0.25 * scale
                qy = (matrix[0, 1] + matrix[1, 0]) / scale
                qz = (matrix[0, 2] + matrix[2, 0]) / scale
            elif index == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                qw = (matrix[0, 2] - matrix[2, 0]) / scale
                qx = (matrix[0, 1] + matrix[1, 0]) / scale
                qy = 0.25 * scale
                qz = (matrix[1, 2] + matrix[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                qw = (matrix[1, 0] - matrix[0, 1]) / scale
                qx = (matrix[0, 2] + matrix[2, 0]) / scale
                qy = (matrix[1, 2] + matrix[2, 1]) / scale
                qz = 0.25 * scale
        quat = np.asarray([qw, qx, qy, qz], dtype=float)
        quat = quat / max(np.linalg.norm(quat), 1e-12)
        return tuple(float(value) for value in quat)
