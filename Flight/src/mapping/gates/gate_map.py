from __future__ import annotations

from dataclasses import dataclass, replace
import math

from core.schema import QuatWxyz, Vec3, VisionGateObservation, VisionObservation

TEST_GATE_001_DOWN_OFFSET_M = 0.0


@dataclass
class GateRecord:
    gate_id: str
    position_local_ned_m: Vec3
    quaternion: QuatWxyz | None
    position_confidence: float
    quaternion_confidence: float | None
    crossed: bool = False
    frozen: bool = False
    sequence: int | None = None
    observation_count: int = 1
    last_observed_cycle: int | None = None
    source: str = "vision"
    outer_width_m: float = 2.7
    outer_height_m: float = 2.7
    inner_width_m: float = 1.5
    inner_height_m: float = 1.5
    depth_m: float = 0.26
    average_residual_m: float = 0.0
    last_seen_time_s: float | None = None
    locked: bool = False
    observation_identity: str | None = None


class GateMap:
    """Holds persistent gate records reported by vision."""

    def __init__(
        self,
        merge_distance_m: float = 2.7,
        required_minimum_observation_count: int = 3,
        lock_observation_count: int = 5,
        gate_passed_distance_m: float = 2.0,
        max_observation_distance_m: float | None = None,
        candidate_target_min_observation_count: int = 2,
        candidate_target_min_position_confidence: float = 0.35,
        candidate_target_max_average_residual_m: float = 2.5,
        candidate_target_max_distance_m: float | None = None,
    ) -> None:
        self.merge_distance_m = float(merge_distance_m)
        self.required_minimum_observation_count = max(
            1,
            int(required_minimum_observation_count),
        )
        self.lock_observation_count = max(1, int(lock_observation_count))
        self.gate_passed_distance_m = float(gate_passed_distance_m)
        self.max_observation_distance_m = (
            None if max_observation_distance_m is None else float(max_observation_distance_m)
        )
        self.candidate_target_min_observation_count = max(
            1,
            int(candidate_target_min_observation_count),
        )
        self.candidate_target_min_position_confidence = float(
            candidate_target_min_position_confidence
        )
        self.candidate_target_max_average_residual_m = float(
            candidate_target_max_average_residual_m
        )
        self.candidate_target_max_distance_m = (
            None
            if candidate_target_max_distance_m is None
            else float(candidate_target_max_distance_m)
        )
        self._gates: list[GateRecord] = []
        self._candidates: list[GateRecord] = []
        self._last_crossed_gates: list[GateRecord] = []
        self._target_gate: GateRecord | None = None
        self._next_gate: GateRecord | None = None
        self._candidate_target: GateRecord | None = None
        self._candidate_next: GateRecord | None = None

    @property
    def gates(self) -> list[GateRecord]:
        return [_with_test_position_offset(gate) for gate in self._gates]

    @property
    def candidates(self) -> list[GateRecord]:
        return list(self._candidates)

    @property
    def last_crossed_gates(self) -> list[GateRecord]:
        return list(self._last_crossed_gates)

    @property
    def target_gate(self) -> GateRecord | None:
        return self._target_gate

    @property
    def next_gate(self) -> GateRecord | None:
        return self._next_gate

    @property
    def candidate_target(self) -> GateRecord | None:
        return self._candidate_target

    @property
    def candidate_next(self) -> GateRecord | None:
        return self._candidate_next

    def clear(self) -> None:
        self._gates.clear()
        self._candidates.clear()
        self._last_crossed_gates.clear()
        self._target_gate = None
        self._next_gate = None
        self._candidate_target = None
        self._candidate_next = None

    def uncrossed_gates(self) -> list[GateRecord]:
        return [gate for gate in self.gates if not gate.crossed]

    def update(
        self,
        observation: VisionObservation | None = None,
        observer_position_local_ned_m: Vec3 | None = None,
    ) -> list[GateRecord]:
        if observation is not None:
            observed_time_s = _observation_time_s(observation)
            for gate in observation.gates:
                position = tuple(float(value) for value in gate.position_local_ned)
                if not self._within_observation_distance(position, observer_position_local_ned_m):
                    continue
                identity = _observation_identity(gate, observation)
                gate_index = self._target_record_index_for_position(position)
                if gate_index is None:
                    gate_index = self._identity_record_index(self._gates, identity)
                if gate_index is None:
                    gate_index = self._nearest_record_index(self._gates, position)
                if gate_index is not None:
                    self._merge_record(
                        gate_index,
                        gate,
                        observation,
                        observed_time_s,
                        self._gates,
                        identity=identity,
                    )
                    continue

                candidate_index = self._identity_record_index(self._candidates, identity)
                identity_jump = False
                if candidate_index is None:
                    candidate_index = self._nearest_record_index(self._candidates, position)
                else:
                    identity_jump = (
                        _distance_m(position, self._candidates[candidate_index].position_local_ned_m)
                        > self.merge_distance_m
                    )
                if candidate_index is None:
                    self._candidates.append(_record_from_observation(
                        gate, observation, observed_time_s, identity=identity))
                    candidate_index = len(self._candidates) - 1
                else:
                    self._merge_record(
                        candidate_index,
                        gate,
                        observation,
                        observed_time_s,
                        self._candidates,
                        identity=identity,
                        replace_position=identity_jump,
                    )

                if (
                    self._candidates[candidate_index].observation_count
                    >= self.required_minimum_observation_count
                ):
                    self._gates.append(self._candidates.pop(candidate_index))

        self._sort_and_rename_gates()
        self._last_crossed_gates = self._crossed_gates_at(observer_position_local_ned_m)
        self._update_targets(observer_position_local_ned_m)
        return self.gates

    def _within_observation_distance(
        self,
        position: Vec3,
        observer_position_local_ned_m: Vec3 | None,
    ) -> bool:
        if self.max_observation_distance_m is None:
            return True
        origin = (
            (0.0, 0.0, 0.0)
            if observer_position_local_ned_m is None
            else observer_position_local_ned_m
        )
        return _distance_m(position, origin) <= self.max_observation_distance_m

    def _crossed_gates_at(self, position_local_ned_m: Vec3 | None) -> list[GateRecord]:
        if position_local_ned_m is None:
            return []

        crossed_now: list[GateRecord] = []
        for index, record in enumerate(self._gates):
            visible_record = _with_test_position_offset(record)
            if (
                not record.crossed
                and _distance_m(position_local_ned_m, visible_record.position_local_ned_m)
                <= self.gate_passed_distance_m
            ):
                self._gates[index] = replace(record, crossed=True)
                crossed_now.append(_with_test_position_offset(self._gates[index]))
        return crossed_now

    def _nearest_record_index(self, records: list[GateRecord], position: Vec3) -> int | None:
        nearest_index = None
        nearest_distance = self.merge_distance_m
        for index, record in enumerate(records):
            distance = _distance_m(position, record.position_local_ned_m)
            if distance <= nearest_distance:
                nearest_index = index
                nearest_distance = distance
        return nearest_index

    def _target_record_index_for_position(self, position: Vec3) -> int | None:
        target = self._target_gate
        if target is None:
            return None
        for index, record in enumerate(self._gates):
            visible_record = _with_test_position_offset(record)
            if record.crossed:
                continue
            same_gate_id = bool(target.gate_id) and visible_record.gate_id == target.gate_id
            same_sequence = (
                target.sequence is not None
                and visible_record.sequence == target.sequence
            )
            if not same_gate_id and not same_sequence:
                continue
            if _distance_m(position, visible_record.position_local_ned_m) <= self.merge_distance_m:
                return index
        return None

    @staticmethod
    def _identity_record_index(records: list[GateRecord], identity: str | None) -> int | None:
        if identity is None:
            return None
        for index, record in enumerate(records):
            if record.observation_identity == identity:
                return index
        return None

    def _merge_record(
        self,
        index: int,
        gate: VisionGateObservation,
        observation: VisionObservation,
        observed_time_s: float,
        records: list[GateRecord],
        *,
        identity: str | None,
        replace_position: bool = False,
    ) -> None:
        record = records[index]
        position = tuple(float(value) for value in gate.position_local_ned)
        residual_m = _distance_m(position, record.position_local_ned_m)
        next_count = record.observation_count + 1
        locked = record.locked or next_count >= self.lock_observation_count
        position_local_ned_m = (
            record.position_local_ned_m
            if record.locked
            else position
            if replace_position
            else _average_vec3(record.position_local_ned_m, position)
        )
        quaternion = (
            record.quaternion
            if record.locked
            else _average_quaternion(record.quaternion, gate.orientation_local_ned_quat)
        )
        records[index] = GateRecord(
            gate_id=record.gate_id,
            position_local_ned_m=position_local_ned_m,
            quaternion=quaternion,
            position_confidence=(record.position_confidence + float(gate.position_confidence)) / 2.0,
            quaternion_confidence=_average_optional_float(
                record.quaternion_confidence,
                gate.orientation_confidence,
            ),
            crossed=record.crossed,
            frozen=record.frozen,
            sequence=record.sequence,
            observation_count=next_count,
            last_observed_cycle=observation.frame_id,
            source=observation.source,
            outer_width_m=record.outer_width_m,
            outer_height_m=record.outer_height_m,
            inner_width_m=record.inner_width_m,
            inner_height_m=record.inner_height_m,
            depth_m=record.depth_m,
            average_residual_m=(record.average_residual_m + residual_m) / 2.0,
            last_seen_time_s=observed_time_s,
            locked=locked,
            observation_identity=identity or record.observation_identity,
        )

    def _sort_and_rename_gates(self) -> None:
        self._gates.sort(
            key=lambda record: _distance_m(record.position_local_ned_m, (0.0, 0.0, 0.0))
        )
        for index, record in enumerate(self._gates, start=1):
            record.gate_id = f"gate-{index:03d}"
            record.sequence = index - 1

    def _update_targets(self, observer_position_local_ned_m: Vec3 | None) -> None:
        observer_position = (
            (0.0, 0.0, 0.0)
            if observer_position_local_ned_m is None
            else observer_position_local_ned_m
        )
        target_pair = _closest_uncrossed_pair(self.gates, observer_position)
        self._target_gate = target_pair[0]
        self._next_gate = target_pair[1]

        candidate_pair = _closest_uncrossed_pair(
            self._target_promotable_candidates(observer_position),
            observer_position,
        )
        self._candidate_target = candidate_pair[0]
        self._candidate_next = candidate_pair[1]

    def _target_promotable_candidates(
        self,
        observer_position_local_ned_m: Vec3,
    ) -> list[GateRecord]:
        records: list[GateRecord] = []
        for record in self._candidates:
            distance_m = _distance_m(
                observer_position_local_ned_m,
                record.position_local_ned_m,
            )
            if record.crossed or distance_m <= self.gate_passed_distance_m:
                continue
            if (
                self.candidate_target_max_distance_m is not None
                and distance_m > self.candidate_target_max_distance_m
            ):
                continue
            if record.observation_count < self.candidate_target_min_observation_count:
                continue
            if record.position_confidence < self.candidate_target_min_position_confidence:
                continue
            if record.average_residual_m > self.candidate_target_max_average_residual_m:
                continue
            records.append(record)
        return records


def _record_from_observation(
    gate: VisionGateObservation,
    observation: VisionObservation,
    observed_time_s: float,
    *,
    identity: str | None,
) -> GateRecord:
    return GateRecord(
        gate_id="",
        position_local_ned_m=tuple(float(value) for value in gate.position_local_ned),
        quaternion=_normalize_quaternion(gate.orientation_local_ned_quat),
        position_confidence=float(gate.position_confidence),
        quaternion_confidence=gate.orientation_confidence,
        observation_count=1,
        last_observed_cycle=observation.frame_id,
        source=observation.source,
        last_seen_time_s=observed_time_s,
        observation_identity=identity,
    )


def _observation_identity(
    gate: VisionGateObservation,
    observation: VisionObservation,
) -> str | None:
    if gate.gate_id:
        return f"{observation.source}:{gate.gate_id}"
    track_id = gate.trace.get("track_id") if isinstance(gate.trace, dict) else None
    if track_id:
        return f"{observation.source}:track:{track_id}"
    return None


def _with_test_position_offset(record: GateRecord) -> GateRecord:
    if record.gate_id != "gate-001":
        return record
    position = (
        float(record.position_local_ned_m[0]) + TEST_GATE_001_DOWN_OFFSET_M,
        float(record.position_local_ned_m[1]),
        float(record.position_local_ned_m[2]),
    )
    return replace(record, position_local_ned_m=position)


def _average_vec3(a: Vec3, b: Vec3) -> Vec3:
    return tuple((float(a[axis]) + float(b[axis])) / 2.0 for axis in range(3))


def _average_quaternion(a: QuatWxyz | None, b: QuatWxyz | None) -> QuatWxyz | None:
    normalized_b = _normalize_quaternion(b)
    if normalized_b is None:
        return a
    normalized_a = _normalize_quaternion(a)
    if normalized_a is None:
        return normalized_b
    if sum(normalized_a[axis] * normalized_b[axis] for axis in range(4)) < 0.0:
        normalized_b = tuple(-value for value in normalized_b)
    return _normalize_quaternion(
        tuple((normalized_a[axis] + normalized_b[axis]) / 2.0 for axis in range(4))
    )


def _average_optional_float(a: float | None, b: float | None) -> float | None:
    if a is None:
        return None if b is None else float(b)
    if b is None:
        return float(a)
    return (float(a) + float(b)) / 2.0


def _distance_m(a: Vec3, b: Vec3) -> float:
    return math.sqrt(
        sum(
            (float(a[axis]) - float(b[axis])) * (float(a[axis]) - float(b[axis]))
            for axis in range(3)
        )
    )


def _closest_uncrossed_pair(
    records: list[GateRecord],
    observer_position_local_ned_m: Vec3,
) -> tuple[GateRecord | None, GateRecord | None]:
    sorted_records = sorted(
        (record for record in records if not record.crossed),
        key=lambda record: _distance_m(
            observer_position_local_ned_m,
            record.position_local_ned_m,
        ),
    )
    target = sorted_records[0] if len(sorted_records) >= 1 else None
    next_gate = sorted_records[1] if len(sorted_records) >= 2 else None
    return target, next_gate


def _normalize_quaternion(quaternion: QuatWxyz | None) -> QuatWxyz | None:
    if quaternion is None:
        return None
    values = tuple(float(value) for value in quaternion)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return None
    return tuple(value / norm for value in values)


def _observation_time_s(observation: VisionObservation) -> float:
    if int(observation.sim_time_ns) > 0:
        return float(observation.sim_time_ns) * 1e-9
    return float(observation.frame_id) / 10.0
