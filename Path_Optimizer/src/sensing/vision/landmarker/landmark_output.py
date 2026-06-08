"""Controller JSON egress for official landmarker gates."""

from __future__ import annotations

import json
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


def controller_gates_from_state(
    state: dict[str, Any],
    *,
    top_k: int = 5,
    landmark_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    landmarks = list((state.get("landmarks") or {}).get("gates") or [])
    if landmark_ids is not None:
        allowed_ids = {str(landmark_id) for landmark_id in landmark_ids}
        landmarks = [landmark for landmark in landmarks if str(landmark.get("id", "")) in allowed_ids]
    landmarks.sort(key=_distance_from_camera)
    return [_egress_gate(landmark) for landmark in landmarks[: max(0, int(top_k))]]


def build_controller_payload(
    state: dict[str, Any],
    *,
    run: dict[str, Any],
    output_dir: Path | str,
    top_k: int = 5,
    landmark_ids: Iterable[str] | None = None,
    current_gates: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gates = (
        controller_gates_from_current_gates(current_gates, top_k=top_k)
        if current_gates is not None
        else controller_gates_from_state(state, top_k=top_k, landmark_ids=landmark_ids)
    )
    return {
        "run": {
            "output_dir": str(Path(output_dir)),
            "cycle": int(run.get("cycle", 0)),
            "frame_id": str(run.get("frame_id", "")),
            "sim_time_ns": int(run.get("sim_time_ns", 0)),
        },
        "gates": gates,
        "obstacles": [],
    }


def write_controller_frame(output_dir: Path, payload: dict[str, Any]) -> Path:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    frame_id = payload.get("run", {}).get("frame_id") or f"frame_{int(payload.get('run', {}).get('cycle', 0)):06d}"
    path = directory / f"{frame_id}_controller.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False) + "\n")


__all__ = [
    "append_jsonl",
    "build_controller_payload",
    "controller_gates_from_current_gates",
    "controller_gates_from_state",
    "write_controller_frame",
]
