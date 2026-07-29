from __future__ import annotations

from dataclasses import dataclass

from core.schema import QuatWxyz, Vec3, VisionObservation


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
    """Holds the latest gate records reported by vision."""

    def __init__(self) -> None:
        self._gates: list[GateRecord] = []

    @property
    def gates(self) -> list[GateRecord]:
        return list(self._gates)

    def clear(self) -> None:
        self._gates.clear()

    def update(self, observation: VisionObservation) -> list[GateRecord]:
        self._gates = [
            GateRecord(
                gate_id=gate.gate_id,
                position_local_ned_m=tuple(float(value) for value in gate.position_local_ned),
                quaternion=gate.orientation_local_ned_quat,
                position_confidence=float(gate.position_confidence),
                quaternion_confidence=gate.orientation_confidence,
                sequence=index,
                observation_count=1,
                last_observed_cycle=observation.frame_id,
                source=observation.source,
                last_seen_time_s=_observation_time_s(observation),
            )
            for index, gate in enumerate(observation.gates)
        ]
        return self.gates


def _observation_time_s(observation: VisionObservation) -> float:
    if int(observation.sim_time_ns) > 0:
        return float(observation.sim_time_ns) * 1e-9
    return float(observation.frame_id) / 10.0
