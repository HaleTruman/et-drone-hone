from enum import Enum

import numpy as np

from mapping.gates import GateRecord


class SystemMode(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    RACING = "RACING"
    FINISHED = "FINISHED"
    FAULT = "FAULT"


class SystemModeManager:
    def __init__(self, max_run_s: float = 480.0):
        self.system_mode = SystemMode.IDLE
        self.max_run_s = max_run_s
        self.fault_reason: str | None = None

    def update_mode(self, event: str, run_duration_s: float = 0.0) -> SystemMode:
        if run_duration_s > self.max_run_s:
            return self.handle_fault("run timeout")
        transitions = {
            (SystemMode.IDLE, "arm"): SystemMode.ARMED,
            (SystemMode.ARMED, "start"): SystemMode.RACING,
            (SystemMode.RACING, "finish"): SystemMode.FINISHED,
        }
        self.system_mode = transitions.get((self.system_mode, event), self.system_mode)
        return self.system_mode

    def check_gate_crossing(self, previous_position: np.ndarray, position: np.ndarray, gate: GateRecord) -> bool:
        rotation = self._rotation_matrix(np.asarray(gate.quaternion, dtype=float))
        previous_local = rotation.T @ (np.asarray(previous_position, dtype=float) - gate.position_local_ned_m)
        current_local = rotation.T @ (np.asarray(position, dtype=float) - gate.position_local_ned_m)
        inside = abs(current_local[1]) <= 0.75 and abs(current_local[2]) <= 0.75
        return bool(previous_local[0] < 0.0 <= current_local[0] and inside)

    def is_racing(self) -> bool:
        return self.system_mode == SystemMode.RACING

    def handle_fault(self, reason: str) -> SystemMode:
        self.fault_reason = reason
        self.system_mode = SystemMode.FAULT
        return self.system_mode

    def _rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )
