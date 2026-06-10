from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class GateRecord:
    gate_id: str
    position_local_ned_m: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    confidence: float
    crossed: bool = False
    sequence: int | None = None


class GateMap:
    def __init__(self):
        self._gates: dict[str, GateRecord] = {}

    def add_or_update_gate(self, gate: GateRecord) -> GateRecord:
        current = self._gates.get(gate.gate_id)
        fused = gate if current is None else self.fuse_gate(current, gate)
        self._gates[gate.gate_id] = fused
        return fused

    def get_gate(self, gate_id: str) -> GateRecord | None:
        return self._gates.get(gate_id)

    def get_next_gates(self, n: int) -> list[GateRecord]:
        remaining = [gate for gate in self._gates.values() if not gate.crossed]
        remaining.sort(key=lambda gate: (gate.sequence is None, gate.sequence, gate.gate_id))
        return remaining[:n]

    def get_reference_path(self) -> np.ndarray:
        return np.asarray([gate.position_local_ned_m for gate in self.get_next_gates(len(self._gates))])

    def fuse_gate(self, current: GateRecord, observed: GateRecord) -> GateRecord:
        total = max(current.confidence + observed.confidence, 1e-12)
        old_weight = current.confidence / total
        new_weight = observed.confidence / total
        pos = old_weight * np.asarray(current.position_local_ned_m) + new_weight * np.asarray(
            observed.position_local_ned_m
        )
        quat = old_weight * np.asarray(current.quaternion) + new_weight * np.asarray(observed.quaternion)
        quat /= max(np.linalg.norm(quat), 1e-12)
        return replace(
            current,
            position_local_ned_m=tuple(float(value) for value in pos),
            quaternion=tuple(float(value) for value in quat),
            confidence=min(1.0, total),
            crossed=current.crossed or observed.crossed,
            sequence=current.sequence if current.sequence is not None else observed.sequence,
        )

    def mark_crossed(self, gate_id: str) -> None:
        self._gates[gate_id] = replace(self._gates[gate_id], crossed=True)
