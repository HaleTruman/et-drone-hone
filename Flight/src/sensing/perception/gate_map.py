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
    source: str = "vision"
    width_m: float | None = None
    height_m: float | None = None


class GateMap:
    def __init__(
        self,
        association_distance_m: float = 6.0,
        min_observations: int = 1,
        authoritative_association_distance_m: float = 18.0,
    ):
        self._gates: dict[str, GateRecord] = {}
        self.association_distance_m = float(association_distance_m)
        self.min_observations = max(1, int(min_observations))
        self.authoritative_association_distance_m = float(authoritative_association_distance_m)

    def add_or_update_gate(self, gate: GateRecord, *, allow_new: bool = True) -> GateRecord | None:
        gate_id = gate.gate_id
        current = self._gates.get(gate_id)
        if current is None:
            gate_id = self._nearest_gate_id(gate) or gate_id
            current = self._gates.get(gate_id)
        if current is None and not allow_new:
            return None
        fused = gate if current is None else self.fuse_gate(current, gate)
        self._gates[gate_id] = fused
        return self._merge_nearby_gate(gate_id)

    def has_authoritative_gates(self) -> bool:
        return any(gate.source == "track" for gate in self._gates.values())

    def clear(self) -> None:
        self._gates.clear()

    def get_gate(self, gate_id: str) -> GateRecord | None:
        return self._gates.get(gate_id)

    def get_next_gates(self, n: int) -> list[GateRecord]:
        remaining = [
            gate
            for gate in self._gates.values()
            if not gate.crossed and (gate.source == "track" or gate.observation_count >= self.min_observations)
        ]
        remaining.sort(key=lambda gate: (gate.sequence is None, gate.sequence, gate.gate_id))
        return remaining[:n]

    def get_reference_path(self) -> np.ndarray:
        return np.asarray([gate.position_local_ned_m for gate in self.get_next_gates(len(self._gates))])

    def fuse_gate(self, current: GateRecord, observed: GateRecord) -> GateRecord:
        if current.source == "track" and observed.source != "track":
            return replace(
                current,
                crossed=current.crossed or observed.crossed,
                confidence=max(current.confidence, observed.confidence),
                observation_count=self._fused_observation_count(current, observed),
                last_observed_cycle=observed.last_observed_cycle
                if observed.last_observed_cycle is not None
                else current.last_observed_cycle,
                position_relative_ned_m=observed.position_relative_ned_m,
            )
        if observed.source == "track":
            return replace(
                observed,
                crossed=current.crossed or observed.crossed,
                observation_count=max(current.observation_count, observed.observation_count),
                last_observed_cycle=current.last_observed_cycle,
                position_relative_ned_m=current.position_relative_ned_m,
            )
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
        if not self._gates:
            return None
        observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
        best_gate_id: str | None = None
        best_distance = float("inf")
        best_is_authoritative = False
        for gate_id, current in self._gates.items():
            distance = float(np.linalg.norm(observed_position - np.asarray(current.position_local_ned_m, dtype=float)))
            if distance < best_distance:
                best_gate_id = gate_id
                best_distance = distance
                best_is_authoritative = current.source == "track"
        threshold = (
            self.authoritative_association_distance_m
            if best_is_authoritative
            else self.association_distance_m
        )
        if threshold > 0.0 and best_distance <= threshold:
            return best_gate_id
        return None

    def _merge_nearby_gate(self, gate_id: str) -> GateRecord:
        gate = self._gates[gate_id]
        nearest_id = self._nearest_gate_id_excluding(gate, exclude_gate_id=gate_id)
        if nearest_id is None:
            return gate
        target_id, duplicate_id = self._merge_target_ids(gate_id, nearest_id)
        merged = self.fuse_gate(self._gates[target_id], self._gates[duplicate_id])
        self._gates[target_id] = merged
        del self._gates[duplicate_id]
        return merged

    def _nearest_gate_id_excluding(self, observed: GateRecord, *, exclude_gate_id: str) -> str | None:
        if self.association_distance_m <= 0.0 or len(self._gates) <= 1:
            return None
        observed_position = np.asarray(observed.position_local_ned_m, dtype=float)
        best_gate_id: str | None = None
        best_distance = float("inf")
        for gate_id, current in self._gates.items():
            if gate_id == exclude_gate_id:
                continue
            if current.source == "track" and observed.source == "track":
                continue
            distance = float(np.linalg.norm(observed_position - np.asarray(current.position_local_ned_m, dtype=float)))
            if distance < best_distance:
                best_gate_id = gate_id
                best_distance = distance
        if best_distance <= self.association_distance_m:
            return best_gate_id
        return None

    def _merge_target_ids(self, gate_id: str, other_gate_id: str) -> tuple[str, str]:
        gate = self._gates[gate_id]
        other = self._gates[other_gate_id]
        if gate.source == "track" and other.source != "track":
            return gate_id, other_gate_id
        if other.source == "track" and gate.source != "track":
            return other_gate_id, gate_id
        gate_sequence = float("inf") if gate.sequence is None else gate.sequence
        other_sequence = float("inf") if other.sequence is None else other.sequence
        if gate_sequence != other_sequence:
            return (gate_id, other_gate_id) if gate_sequence < other_sequence else (other_gate_id, gate_id)
        if gate.observation_count != other.observation_count:
            return (gate_id, other_gate_id) if gate.observation_count > other.observation_count else (other_gate_id, gate_id)
        return (gate_id, other_gate_id) if gate_id <= other_gate_id else (other_gate_id, gate_id)

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
