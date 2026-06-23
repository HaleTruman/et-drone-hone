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
    observation_count: int = 1
    last_observed_cycle: int | None = None
    position_relative_ned_m: tuple[float, float, float] | None = None


class GateMap:
    def __init__(self, association_distance_m: float = 6.0, min_observations: int = 1):
        self._gates: dict[str, GateRecord] = {}
        self.association_distance_m = float(association_distance_m)
        self.min_observations = max(1, int(min_observations))

    def add_or_update_gate(self, gate: GateRecord) -> GateRecord:
        gate_id = gate.gate_id
        current = self._gates.get(gate_id)
        if current is None:
            gate_id = self._nearest_gate_id(gate) or gate_id
            current = self._gates.get(gate_id)
        fused = gate if current is None else self.fuse_gate(current, gate)
        self._gates[gate_id] = fused
        return fused

    def get_gate(self, gate_id: str) -> GateRecord | None:
        return self._gates.get(gate_id)

    def get_next_gates(self, n: int) -> list[GateRecord]:
        remaining = [
            gate
            for gate in self._gates.values()
            if not gate.crossed and gate.observation_count >= self.min_observations
        ]
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
            position_relative_ned_m=self._fused_relative_position(current, observed, old_weight, new_weight),
            quaternion=tuple(float(value) for value in quat),
            confidence=min(1.0, total),
            crossed=current.crossed or observed.crossed,
            sequence=current.sequence if current.sequence is not None else observed.sequence,
            observation_count=self._fused_observation_count(current, observed),
            last_observed_cycle=observed.last_observed_cycle
            if observed.last_observed_cycle is not None
            else current.last_observed_cycle,
        )

    def mark_crossed(self, gate_id: str) -> None:
        self._gates[gate_id] = replace(self._gates[gate_id], crossed=True)

    def _nearest_gate_id(self, observed: GateRecord) -> str | None:
        if self.association_distance_m <= 0.0 or not self._gates:
            return None
        observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
        best_gate_id: str | None = None
        best_distance = float("inf")
        for gate_id, current in self._gates.items():
            distance = float(np.linalg.norm(observed_position - np.asarray(current.position_local_ned_m, dtype=float)))
            if distance < best_distance:
                best_gate_id = gate_id
                best_distance = distance
        if best_distance <= self.association_distance_m:
            return best_gate_id
        return None

    def _fused_observation_count(self, current: GateRecord, observed: GateRecord) -> int:
        if observed.last_observed_cycle is not None and observed.last_observed_cycle == current.last_observed_cycle:
            return current.observation_count
        return current.observation_count + max(1, observed.observation_count)

    def _fused_relative_position(
        self,
        current: GateRecord,
        observed: GateRecord,
        old_weight: float,
        new_weight: float,
    ) -> tuple[float, float, float] | None:
        if current.position_relative_ned_m is None:
            return observed.position_relative_ned_m
        if observed.position_relative_ned_m is None:
            return current.position_relative_ned_m
        relative = old_weight * np.asarray(current.position_relative_ned_m) + new_weight * np.asarray(
            observed.position_relative_ned_m
        )
        return tuple(float(value) for value in relative)
