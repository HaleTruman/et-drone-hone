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


class GateMap:
    """Holds persistent gate records reported by vision."""

    def __init__(
        self,
        merge_distance_m: float = 2.7,
        required_minimum_observation_count: int = 3,
        lock_observation_count: int = 5,
        gate_passed_distance_m: float = 2.0,
        max_observation_distance_m: float | None = None,
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
        self._gates: list[GateRecord] = []
        self._candidates: list[GateRecord] = []

    @property
    def gates(self) -> list[GateRecord]:
        return [_with_test_position_offset(gate) for gate in self._gates]

    def clear(self) -> None:
        self._gates.clear()
        self._candidates.clear()

    def uncrossed_gates(self) -> list[GateRecord]:
        return [gate for gate in self.gates if not gate.crossed]

    def update(
        self,
        observation: VisionObservation,
        observer_position_local_ned_m: Vec3 | None = None,
    ) -> list[GateRecord]:
        observed_time_s = _observation_time_s(observation)
        for gate in observation.gates:
            position = tuple(float(value) for value in gate.position_local_ned)
            if not self._within_observation_distance(position, observer_position_local_ned_m):
                continue
            gate_index = self._nearest_record_index(self._gates, position)
            if gate_index is not None:
                self._merge_record(gate_index, gate, observation, observed_time_s, self._gates)
                continue

            candidate_index = self._nearest_record_index(self._candidates, position)
            if candidate_index is None:
                self._candidates.append(_record_from_observation(gate, observation, observed_time_s))
                candidate_index = len(self._candidates) - 1
            else:
                self._merge_record(
                    candidate_index,
                    gate,
                    observation,
                    observed_time_s,
                    self._candidates,
                )

            if (
                self._candidates[candidate_index].observation_count
                >= self.required_minimum_observation_count
            ):
                self._gates.append(self._candidates.pop(candidate_index))

        self._sort_and_rename_gates()
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

    def update_crossed_gates(self, position_local_ned_m: Vec3) -> list[GateRecord]:
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

    def _merge_record(
        self,
        index: int,
        gate: VisionGateObservation,
        observation: VisionObservation,
        observed_time_s: float,
        records: list[GateRecord],
    ) -> None:
        record = records[index]
        position = tuple(float(value) for value in gate.position_local_ned)
        residual_m = _distance_m(position, record.position_local_ned_m)
        next_count = record.observation_count + 1
        locked = record.locked or next_count >= self.lock_observation_count
        position_local_ned_m = (
            record.position_local_ned_m
            if record.locked
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
        )

    def _sort_and_rename_gates(self) -> None:
        self._gates.sort(
            key=lambda record: _distance_m(record.position_local_ned_m, (0.0, 0.0, 0.0))
        )
        for index, record in enumerate(self._gates, start=1):
            record.gate_id = f"gate-{index:03d}"
            record.sequence = index - 1


def _record_from_observation(
    gate: VisionGateObservation,
    observation: VisionObservation,
    observed_time_s: float,
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
    )


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
