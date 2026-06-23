from dataclasses import dataclass
from typing import Any

import numpy as np

from .gate_map import GateRecord


@dataclass(frozen=True)
class TrackedGateTarget:
    gate_id: str
    position_local_ned_m: tuple[float, float, float]
    confidence: float
    updated_s: float
    frame_id: int | None = None
    position_relative_ned_m: tuple[float, float, float] | None = None


class GateTargetTracker:
    """Hold the last usable mapped gate target across short vision gaps."""

    def __init__(self, hold_s: float = 0.75):
        self.hold_s = float(hold_s)
        self._target: TrackedGateTarget | None = None

    def update(self, gate: GateRecord, *, now_s: float, frame_id: int | None = None) -> TrackedGateTarget:
        self._target = TrackedGateTarget(
            gate_id=gate.gate_id,
            position_local_ned_m=tuple(float(value) for value in gate.position_local_ned_m),
            confidence=float(gate.confidence),
            updated_s=float(now_s),
            frame_id=frame_id,
            position_relative_ned_m=None
            if gate.position_relative_ned_m is None
            else tuple(float(value) for value in gate.position_relative_ned_m),
        )
        return self._target

    def latest(self, *, now_s: float) -> TrackedGateTarget | None:
        if self._target is None:
            return None
        if float(now_s) - self._target.updated_s > self.hold_s:
            return None
        return self._target

    def age_s(self, *, now_s: float) -> float | None:
        if self._target is None:
            return None
        return float(now_s) - self._target.updated_s

    def clear(self) -> None:
        self._target = None

    def snapshot(self, *, now_s: float) -> dict[str, Any]:
        target = self.latest(now_s=now_s)
        return {
            "active": target is not None,
            "target": None if target is None else target.__dict__,
            "age_s": self.age_s(now_s=now_s),
            "hold_s": self.hold_s,
        }


def select_guidance_gate(
    mapped_gates: list[GateRecord],
    *,
    telemetry: Any,
    min_confidence: float = 0.10,
) -> GateRecord | None:
    """Pick the nearest not-yet-crossed mapped gate in local NED."""

    position = getattr(telemetry, "position_local_ned_m", None)
    if position is None:
        return None
    current = np.asarray(position, dtype=float)
    candidates = [
        gate
        for gate in mapped_gates
        if not gate.crossed and float(gate.confidence) >= float(min_confidence)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda gate: (
            float(np.linalg.norm(np.asarray(gate.position_local_ned_m, dtype=float) - current)),
            gate.sequence is None,
            -1 if gate.sequence is None else gate.sequence,
            gate.gate_id,
        ),
    )
