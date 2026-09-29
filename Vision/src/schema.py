"""Frame, vehicle-state, and observation schemas copied from Flight."""

from dataclasses import dataclass, field
from typing import Any


Vec3 = tuple[float, float, float]


QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class VehicleState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3
    elapsed_time_ns: int | None = None


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    image: Any | None = None
    saved_path: str | None = None
    elapsed_time_ns: int | None = None


@dataclass(frozen=True)
class VisionGateObservation:
    gate_id: str
    position_local_ned: Vec3
    position_confidence: float
    orientation_local_ned_quat: QuatWxyz | None = None
    orientation_confidence: float | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VisionGateObservation":
        position = payload.get("position_local_ned") or payload.get("position_xyz")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("Vision gate payload must include position_local_ned with three values.")
        orientation = payload.get("orientation_local_ned_quat")
        if orientation is None:
            orientation = payload.get("orientation_quat")
        if orientation is not None and (not isinstance(orientation, (list, tuple)) or len(orientation) != 4):
            raise ValueError("Vision gate payload orientation_local_ned_quat must include four values.")
        return cls(
            gate_id=str(payload.get("id", payload.get("gate_id", ""))),
            position_local_ned=tuple(float(value) for value in position),
            position_confidence=float(payload.get("position_confidence", 0.0)),
            orientation_local_ned_quat=None
            if orientation is None
            else tuple(float(value) for value in orientation),
            orientation_confidence=None
            if payload.get("orientation_confidence") is None
            else float(payload.get("orientation_confidence")),
            trace=dict(payload.get("trace") or {}) if isinstance(payload.get("trace"), dict) else {},
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.gate_id,
            "position_local_ned": [float(value) for value in self.position_local_ned],
            "position_confidence": float(self.position_confidence),
            "orientation_local_ned_quat": None
            if self.orientation_local_ned_quat is None
            else [float(value) for value in self.orientation_local_ned_quat],
            "orientation_confidence": self.orientation_confidence,
        }
        if self.trace:
            payload["trace"] = self.trace
        return payload


@dataclass(frozen=True)
class VisionObservation:
    frame_id: int
    sim_time_ns: int
    gates: list[VisionGateObservation]
    source: str = "vision"
    trace: dict[str, Any] = field(default_factory=dict)
    elapsed_time_ns: int | None = None

    @classmethod
    def from_controller_payload(cls, payload: dict[str, Any], *, source: str = "vision") -> "VisionObservation":
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        cycle = int(run.get("cycle", 0))
        sim_time_ns = int(run.get("sim_time_ns", 0))
        gates = [
            VisionGateObservation.from_payload(gate)
            for gate in payload.get("gates", [])
            if isinstance(gate, dict)
        ]
        trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
        return cls(frame_id=cycle, sim_time_ns=sim_time_ns, gates=gates, source=source, trace=dict(trace))

    def to_controller_payload(self, *, output_dir: str = "memory") -> dict[str, Any]:
        payload = {
            "run": {
                "output_dir": output_dir,
                "cycle": int(self.frame_id),
                "frame_id": f"frame_{int(self.frame_id):06d}",
                "sim_time_ns": int(self.sim_time_ns),
            },
            "gates": [gate.to_payload() for gate in self.gates],
            "obstacles": [],
        }
        if self.trace:
            payload["trace"] = self.trace
        return payload
