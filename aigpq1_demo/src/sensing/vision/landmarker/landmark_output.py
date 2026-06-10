"""Build passthrough controller targets from current-frame regressor gates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable


def _distance_from_camera(landmark: dict[str, Any]) -> float:
    position = landmark.get("position_xyz") or [0.0, 0.0, 0.0]
    return math.sqrt(sum(float(value) * float(value) for value in position))


def _egress_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(gate["id"]),
        "position_xyz": [float(value) for value in gate["position_xyz"]],
        "position_confidence": float(gate.get("position_confidence", 0.0)),
        "orientation_xyz": None
        if gate.get("orientation_xyz") is None
        else [float(value) for value in gate["orientation_xyz"]],
        "orientation_confidence": float(gate.get("orientation_confidence", 0.0)),
    }


def controller_gates_from_current_gates(current_gates: Iterable[dict[str, Any]], *, top_k: int = 5) -> list[dict[str, Any]]:
    gates = list(current_gates)
    gates.sort(key=_distance_from_camera)
    return [_egress_gate(gate) for gate in gates[: max(0, int(top_k))]]


def build_passthrough_controller_payload(
    frame_payload: dict[str, Any],
    *,
    output_dir: Path | str,
    top_k: int = 5,
) -> dict[str, Any]:
    return {
        "run": {
            "output_dir": str(Path(output_dir)),
            "cycle": int((frame_payload.get("run") or {}).get("cycle", 0)),
            "frame_id": str((frame_payload.get("run") or {}).get("frame_id", "")),
            "sim_time_ns": int((frame_payload.get("run") or {}).get("sim_time_ns", 0)),
        },
        "gates": controller_gates_from_current_gates(
            (gate for gate in frame_payload.get("gates", []) if isinstance(gate, dict)),
            top_k=top_k,
        ),
        "obstacles": [],
    }


__all__ = [
    "build_passthrough_controller_payload",
    "controller_gates_from_current_gates",
]
