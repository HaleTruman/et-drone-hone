from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisionGateObservation:
    """A gate observation in camera optical coordinates: [right, up, forward]."""

    gate_id: str
    position_camera_m: tuple[float, float, float]
    position_confidence: float
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VisionGateObservation":
        position = payload.get("position_xyz")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("Vision gate payload must include position_xyz with three values.")
        orientation = payload.get("orientation_xyz")
        return cls(
            gate_id=str(payload.get("id", "")),
            position_camera_m=tuple(float(value) for value in position),
            position_confidence=float(payload.get("position_confidence", 0.0)),
            orientation_camera=None
            if orientation is None
            else tuple(float(value) for value in orientation),
            orientation_confidence=float(payload.get("orientation_confidence", 0.0)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.gate_id,
            "position_xyz": [float(value) for value in self.position_camera_m],
            "position_confidence": float(self.position_confidence),
            "orientation_xyz": None
            if self.orientation_camera is None
            else [float(value) for value in self.orientation_camera],
            "orientation_confidence": float(self.orientation_confidence),
        }


@dataclass(frozen=True)
class VisionObservation:
    frame_id: int
    sim_time_ns: int
    gates: tuple[VisionGateObservation, ...]
    source: str = "vision"

    @classmethod
    def from_controller_payload(cls, payload: dict[str, Any], *, source: str = "vision") -> "VisionObservation":
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        cycle = int(run.get("cycle", 0))
        sim_time_ns = int(run.get("sim_time_ns", 0))
        gates = tuple(
            VisionGateObservation.from_payload(gate)
            for gate in payload.get("gates", [])
            if isinstance(gate, dict)
        )
        return cls(frame_id=cycle, sim_time_ns=sim_time_ns, gates=gates, source=source)

    def to_controller_payload(self, *, output_dir: str = "memory") -> dict[str, Any]:
        return {
            "run": {
                "output_dir": output_dir,
                "cycle": int(self.frame_id),
                "frame_id": f"frame_{int(self.frame_id):06d}",
                "sim_time_ns": int(self.sim_time_ns),
            },
            "gates": [gate.to_payload() for gate in self.gates],
            "obstacles": [],
        }
