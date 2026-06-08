from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vision.src.landmarker.landmark_output import build_controller_payload, controller_gates_from_state
from vision.src.landmarker.landmarker import Landmarker, load_landmarker_state, save_landmarker_state
from vision.src.landmarker.landmarker_pipeline import LandmarkerPipelineConfig, run_landmarker_pipeline


def _gate(
    gate_id: str,
    position_xyz: tuple[float, float, float],
    *,
    position_confidence: float = 0.9,
    orientation_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    orientation_confidence: float = 0.8,
) -> dict[str, object]:
    return {
        "id": gate_id,
        "position_xyz": list(position_xyz),
        "position_confidence": position_confidence,
        "orientation_xyz": list(orientation_xyz),
        "orientation_confidence": orientation_confidence,
    }


def _frame(cycle: int, gates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run": {
            "output_dir": "run-test",
            "cycle": cycle,
            "frame_id": f"frame_{cycle:06d}",
            "sim_time_ns": cycle * 1000,
        },
        "gates": gates,
        "obstacles": [],
    }


def _distance(position: list[float]) -> float:
    return math.sqrt(sum(value * value for value in position))


def test_landmarker_associates_near_observations_to_same_official_id() -> None:
    landmarker = Landmarker()

    landmarker.update_frame(_frame(1, [_gate("gate-a1", (10.0, 0.0, 2.0), position_confidence=0.5)]))
    landmarker.update_frame(_frame(2, [_gate("gate-a2", (12.0, 0.0, 2.0), position_confidence=1.0)]))

    landmarks = landmarker.state["landmarks"]["gates"]
    assert len(landmarks) == 1
    assert landmarks[0]["id"] == "gate-001w"
    assert landmarks[0]["observation_count"] == 2
    assert landmarks[0]["source_ids"] == ["gate-a1", "gate-a2"]
    assert landmarks[0]["position_xyz"][0] == pytest.approx((10.0 * 0.5 + 12.0 * 1.0) / 1.5)


def test_landmarker_creates_new_official_id_for_far_observation() -> None:
    landmarker = Landmarker()

    landmarker.update_frame(_frame(1, [_gate("gate-a1", (3.0, 0.0, 0.0))]))
    landmarker.update_frame(_frame(2, [_gate("gate-a2", (30.0, 0.0, 0.0))]))

    landmarks = landmarker.state["landmarks"]["gates"]
    assert [landmark["id"] for landmark in landmarks] == ["gate-001w", "gate-002w"]


def test_landmarker_treats_opposite_orientation_axes_as_bidirectional() -> None:
    landmarker = Landmarker()

    landmarker.update_frame(_frame(1, [_gate("gate-a1", (5.0, 0.0, 0.0), orientation_xyz=(0.0, 0.0, 1.0))]))
    landmarker.update_frame(_frame(2, [_gate("gate-a2", (5.5, 0.0, 0.0), orientation_xyz=(0.0, 0.0, -1.0))]))

    orientation = landmarker.state["landmarks"]["gates"][0]["orientation_xyz"]
    assert orientation is not None
    assert math.sqrt(sum(value * value for value in orientation)) == pytest.approx(1.0)
    assert abs(orientation[2]) == pytest.approx(1.0)


def test_controller_output_returns_closest_five() -> None:
    landmarker = Landmarker()
    gates = [
        _gate("gate-a1", (10.0, 0.0, 0.0)),
        _gate("gate-b1", (2.0, 0.0, 0.0)),
        _gate("gate-c1", (5.0, 0.0, 0.0)),
        _gate("gate-d1", (1.0, 0.0, 0.0)),
        _gate("gate-e1", (8.0, 0.0, 0.0)),
        _gate("gate-f1", (3.0, 0.0, 0.0)),
    ]

    landmarker.update_frame(_frame(1, gates))
    controller_gates = controller_gates_from_state(landmarker.state, top_k=5)

    assert len(controller_gates) == 5
    distances = [_distance(gate["position_xyz"]) for gate in controller_gates]
    assert distances == sorted(distances)
    assert distances == [1.0, 2.0, 3.0, 5.0, 8.0]


def test_controller_output_uses_current_frame_updates_for_top_k() -> None:
    landmarker = Landmarker()

    landmarker.update_frame(_frame(1, [_gate("gate-a1", (1.0, 0.0, 0.0))]))
    update_result = landmarker.update_frame(_frame(2, [_gate("gate-a2", (20.0, 0.0, 0.0))]))
    payload = build_controller_payload(
        landmarker.state,
        run={"cycle": 2, "frame_id": "frame_000002", "sim_time_ns": 2000},
        output_dir="run-test",
        top_k=1,
        current_gates=update_result["current_gates"],
    )

    assert [gate["id"] for gate in payload["gates"]] == ["gate-002w"]


def test_controller_output_uses_current_frame_regression_xyz() -> None:
    landmarker = Landmarker()

    landmarker.update_frame(_frame(1, [_gate("gate-a1", (10.0, 0.0, 0.0), position_confidence=1.0)]))
    update_result = landmarker.update_frame(
        _frame(2, [_gate("gate-a2", (14.0, 0.0, 0.0), position_confidence=1.0)])
    )
    payload = build_controller_payload(
        landmarker.state,
        run={"cycle": 2, "frame_id": "frame_000002", "sim_time_ns": 2000},
        output_dir="run-test",
        top_k=1,
        current_gates=update_result["current_gates"],
    )

    assert payload["gates"][0]["id"] == "gate-001w"
    assert payload["gates"][0]["position_xyz"] == [14.0, 0.0, 0.0]
    assert landmarker.state["landmarks"]["gates"][0]["position_xyz"][0] == pytest.approx(12.0)


def test_landmarker_state_persists_official_ids(tmp_path: Path) -> None:
    state_path = tmp_path / "landmarker_state.json"
    landmarker = Landmarker()
    landmarker.update_frame(_frame(1, [_gate("gate-a1", (4.0, 0.0, 0.0))]))
    save_landmarker_state(state_path, landmarker.state)

    loaded = Landmarker(load_landmarker_state(state_path))
    loaded.update_frame(_frame(2, [_gate("gate-a2", (4.5, 0.0, 0.0))]))

    landmarks = loaded.state["landmarks"]["gates"]
    assert len(landmarks) == 1
    assert landmarks[0]["id"] == "gate-001w"
    assert landmarks[0]["observation_count"] == 2


def test_landmarker_pipeline_reads_regressor_jsonl_and_writes_outputs(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "regressor_frames.jsonl"
    output_dir = tmp_path / "controller"
    state_path = tmp_path / "landmarker_state.json"
    frames = [
        _frame(1, [_gate("gate-a1", (6.0, 0.0, 0.0))]),
        _frame(2, [_gate("gate-a2", (7.0, 0.0, 0.0))]),
    ]
    input_jsonl.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8")

    stats = run_landmarker_pipeline(
        LandmarkerPipelineConfig(
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            state_path=state_path,
            top_k=5,
        )
    )

    assert stats.frames_processed == 2
    assert stats.final_landmark_count == 1
    assert state_path.is_file()
    assert stats.jsonl_path.is_file()
    assert (output_dir / "frame_000001_controller.json").is_file()
    assert (output_dir / "frame_000002_controller.json").is_file()
    lines = stats.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[-1])
    assert payload["run"]["frame_id"] == "frame_000002"
    assert payload["gates"][0]["id"] == "gate-001w"
    assert payload["obstacles"] == []


def test_landmarker_pipeline_outputs_only_current_frame_updates(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "regressor_frames.jsonl"
    output_dir = tmp_path / "controller"
    state_path = tmp_path / "landmarker_state.json"
    frames = [
        _frame(1, [_gate("gate-a1", (1.0, 0.0, 0.0))]),
        _frame(2, [_gate("gate-a2", (20.0, 0.0, 0.0))]),
        _frame(3, []),
    ]
    input_jsonl.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8")

    stats = run_landmarker_pipeline(
        LandmarkerPipelineConfig(
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            state_path=state_path,
            top_k=1,
        )
    )

    assert stats.frames_processed == 3
    payloads = [json.loads(line) for line in stats.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [gate["id"] for gate in payloads[0]["gates"]] == ["gate-001w"]
    assert [gate["id"] for gate in payloads[1]["gates"]] == ["gate-002w"]
    assert payloads[1]["gates"][0]["position_xyz"] == [20.0, 0.0, 0.0]
    assert payloads[2]["gates"] == []
