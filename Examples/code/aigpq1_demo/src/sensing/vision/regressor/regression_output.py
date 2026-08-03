"""Build the regressor payload consumed by the demo controller path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .logit_inference import FrameRegression


def _alpha_id(index: int) -> str:
    value = int(index)
    output = ""
    while True:
        value, remainder = divmod(value, 26)
        output = chr(ord("a") + remainder) + output
        if value == 0:
            return output
        value -= 1


def frame_name(frame_id: int) -> str:
    return f"frame_{int(frame_id):06d}"


def build_surveyer_payload(frame: FrameRegression, *, output_dir: Path | str) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    for index, gate in enumerate(frame.gates):
        confidence = float(gate.confidence)
        gates.append(
            {
                "id": f"gate-{_alpha_id(index)}{int(frame.frame_id)}",
                "position_xyz": [float(value) for value in gate.position_xyz],
                "position_confidence": confidence,
                "orientation_xyz": [float(value) for value in gate.orientation_xyz],
                "orientation_confidence": confidence,
            }
        )
    return {
        "run": {
            "output_dir": str(Path(output_dir)),
            "cycle": int(frame.frame_id),
            "frame_id": frame_name(frame.frame_id),
            "sim_time_ns": int(frame.sim_time_ns),
        },
        "gates": gates,
        "obstacles": [],
    }


__all__ = ["build_surveyer_payload", "frame_name"]
