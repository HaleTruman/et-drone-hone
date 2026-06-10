from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sensing.perception import GatePoseEstimator, VisionGateObservation


@dataclass(frozen=True)
class LocalVisionTarget:
    frame_id: int
    sim_time_ns: int
    gate_id: str
    position_camera_m: tuple[float, float, float]
    position_local_ned_m: tuple[float, float, float]
    yaw_rad: float
    position_confidence: float
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0


class TargetMapper:
    """Maps a camera-optical vision target into local NED command space."""

    def __init__(self, camera_tilt_deg: float = 20.0):
        self.pose_estimator = GatePoseEstimator(camera_tilt_deg=camera_tilt_deg)

    def map_gate(
        self,
        gate: VisionGateObservation,
        *,
        frame_id: int,
        sim_time_ns: int,
        telemetry: Any,
    ) -> LocalVisionTarget:
        vehicle_position = getattr(telemetry, "position_local_ned_m", None)
        attitude = getattr(telemetry, "attitude", None)
        if vehicle_position is None:
            raise ValueError("Telemetry must include position_local_ned_m to map a vision target.")
        if attitude is None:
            raise ValueError("Telemetry must include attitude to map a vision target.")

        record = self.pose_estimator.estimate_gate_pose(
            gate,
            vehicle_position_local_ned_m=vehicle_position,
            attitude_quaternion=attitude,
        )
        yaw = self.yaw_to_target(
            current_position_local_ned_m=vehicle_position,
            target_position_local_ned_m=record.position_local_ned_m,
            fallback_attitude_quaternion=attitude,
        )
        return LocalVisionTarget(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            gate_id=gate.gate_id,
            position_camera_m=gate.position_camera_m,
            position_local_ned_m=record.position_local_ned_m,
            yaw_rad=float(yaw),
            position_confidence=float(gate.position_confidence),
            orientation_camera=gate.orientation_camera,
            orientation_confidence=float(gate.orientation_confidence),
        )

    @staticmethod
    def yaw_to_target(
        *,
        current_position_local_ned_m: np.ndarray | tuple[float, float, float],
        target_position_local_ned_m: np.ndarray | tuple[float, float, float],
        fallback_attitude_quaternion: np.ndarray | tuple[float, float, float, float] | None = None,
    ) -> float:
        current = np.asarray(current_position_local_ned_m, dtype=float)
        target = np.asarray(target_position_local_ned_m, dtype=float)
        delta = target - current
        if float(np.linalg.norm(delta[0:2])) <= 1e-9:
            return TargetMapper.quaternion_to_yaw(fallback_attitude_quaternion)
        return float(math.atan2(float(delta[1]), float(delta[0])))

    @staticmethod
    def quaternion_to_yaw(quaternion: np.ndarray | tuple[float, float, float, float] | None) -> float:
        if quaternion is None:
            return 0.0
        q = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm(q))
        if norm <= 1e-12:
            return 0.0
        qw, qx, qy, qz = q / norm
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(math.atan2(float(siny_cosp), float(cosy_cosp)))
