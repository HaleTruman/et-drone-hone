"""Surveyer JSON egress for regressor frame observations."""

from __future__ import annotations

import json
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


def write_frame_json(output_dir: Path, frame: FrameRegression) -> tuple[Path, dict[str, Any]]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = build_surveyer_payload(frame, output_dir=directory)
    output_path = directory / f"{frame_name(frame.frame_id)}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return output_path, payload


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False) + "\n")


__all__ = ["append_jsonl", "build_surveyer_payload", "frame_name", "write_frame_json"]
