"""Flight-compatible dataclasses for 0721Vision instance-frame outputs.

This module belongs to the vision package. It adapts the compact
``0721vision-instance-frame.v1`` JSON into the same controller payload shape
that Flight's ``mapping.perception.VisionObservation`` consumes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, min(1.0, number))


def _float3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def _opencv_camera_to_flight_camera(value: tuple[float, float, float]) -> tuple[float, float, float]:
    x_right, y_down, z_forward = value
    # Vision pose uses OpenCV camera coordinates: +x right, +y down, +z forward.
    # Flight perception expects camera optical coordinates: +x right, +y up, +z forward.
    # Flip y here so downstream Flight GateMap receives Flight-native camera coordinates.
    return (x_right, -y_down, z_forward)


@dataclass(frozen=True)
class VisionGateObservation:
    """A Flight-compatible gate observation in camera optical coordinates.

    position_camera_m is [right, up, forward] in meters relative to the current
    camera origin. orientation_camera is roll-right, pitch-up, yaw-right in
    degrees from the camera frame to the gate frame.
    """

    gate_id: str
    position_camera_m: tuple[float, float, float]
    position_confidence: float
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VisionGateObservation":
        position = _float3(payload.get("position_xyz"))
        if position is None:
            raise ValueError("Vision gate payload must include position_xyz with three numeric values.")
        orientation = _float3(payload.get("orientation_xyz"))
        return cls(
            gate_id=str(payload.get("id", "")),
            position_camera_m=position,
            position_confidence=_clamp01(payload.get("position_confidence")),
            orientation_camera=orientation,
            orientation_confidence=_clamp01(payload.get("orientation_confidence")),
            trace=dict(payload.get("trace") or {}) if isinstance(payload.get("trace"), dict) else {},
        )

    @classmethod
    def from_instance(cls, instance: dict[str, Any]) -> "VisionGateObservation | None":
        pose = instance.get("pose") if isinstance(instance.get("pose"), dict) else {}
        if not pose.get("available"):
            return None
        xyz_camera_m = _float3(pose.get("xyzCameraM"))
        if xyz_camera_m is None:
            return None
        try:
            depth_m = float(pose.get("depthM", xyz_camera_m[2]))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(depth_m) or depth_m <= 0.0:
            return None

        fit_quality = pose.get("fitQuality") if isinstance(pose.get("fitQuality"), dict) else {}
        gate_id = str(instance.get("instanceId") or instance.get("observationId") or "")
        if not gate_id:
            return None
        # rpyCameraDeg is derived from the OpenCV Rodrigues pose. We preserve it
        # as camera-relative gate RPY for v1; validate handedness before using it
        # for high-authority control decisions.
        orientation_camera = _float3(pose.get("rpyCameraDeg"))
        return cls(
            gate_id=gate_id,
            position_camera_m=_opencv_camera_to_flight_camera(xyz_camera_m),
            position_confidence=_clamp01(instance.get("observationQuality")),
            orientation_camera=orientation_camera,
            orientation_confidence=_clamp01(fit_quality.get("overall")),
            trace={
                "backend": "vision_instance_mapping",
                "sourceInstance": instance,
                "bbox": instance.get("source", {}).get("bbox") if isinstance(instance.get("source"), dict) else None,
                "fovClip": instance.get("fovClip") if isinstance(instance.get("fovClip"), dict) else {},
                "pose": pose,
            },
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.gate_id,
            "position_xyz": [float(value) for value in self.position_camera_m],
            "position_confidence": float(self.position_confidence),
            "orientation_xyz": None
            if self.orientation_camera is None
            else [float(value) for value in self.orientation_camera],
            "orientation_confidence": float(self.orientation_confidence),
        }
        if self.trace:
            payload["trace"] = self.trace
        return payload


@dataclass(frozen=True)
class VisionObservation:
    """A Flight-compatible frame-level observation generated by 0721Vision.

    The vision pipeline emits object poses relative to the current camera frame.
    The camera itself is therefore the local origin for this observation.
    """

    frame_id: int
    sim_time_ns: int
    gates: list[VisionGateObservation]
    source: str = "vision_instance_mapping"
    camera_position_camera_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_orientation_camera: tuple[float, float, float] = (0.0, 0.0, 0.0)
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_controller_payload(cls, payload: dict[str, Any], *, source: str = "vision") -> "VisionObservation":
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        gates = [
            VisionGateObservation.from_payload(gate)
            for gate in payload.get("gates", [])
            if isinstance(gate, dict)
        ]
        trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
        return cls(
            frame_id=int(run.get("cycle", 0)),
            sim_time_ns=int(run.get("sim_time_ns", 0)),
            gates=gates,
            source=source,
            trace=dict(trace),
        )

    @classmethod
    def from_instance_frame(
        cls,
        payload: dict[str, Any],
        *,
        sim_time_ns: int | None = None,
        source: str = "vision_instance_mapping",
    ) -> "VisionObservation":
        instances = payload.get("instances")
        if not isinstance(instances, list):
            raise ValueError("Vision instance-frame payload must include an instances list.")
        gates = [
            gate
            for instance in instances
            if isinstance(instance, dict)
            if (gate := VisionGateObservation.from_instance(instance)) is not None
        ]
        return cls(
            frame_id=int(payload.get("frameOrdinal", 0)),
            sim_time_ns=0 if sim_time_ns is None else int(sim_time_ns),
            gates=gates,
            source=source,
            trace={"sourceInstanceFrame": payload},
        )

    @classmethod
    def from_instance_frame_path(
        cls,
        path: str | Path,
        *,
        sim_time_ns: int | None = None,
        source: str = "vision_instance_mapping",
    ) -> "VisionObservation":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Vision instance-frame JSON must contain an object: {path}")
        return cls.from_instance_frame(payload, sim_time_ns=sim_time_ns, source=source)

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
