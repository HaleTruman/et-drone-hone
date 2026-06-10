from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from vision.src.cnn.logits_output import DEPTH_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH, MASK_CHANNELS, serialize_raw_logits
from vision.src.regressor import DEFAULT_REGRESSOR_CHECKPOINT, LogitRegressor
from vision.src.regressor.cnn_ingress import parse_raw_logits
from vision.src.regressor.regression_output import build_surveyer_payload
from vision.src.regressor.regressor_pipeline import RegressorPipelineConfig, run_regressor_pipeline


def _synthetic_logits(*, frame_id: int, with_gate: bool) -> bytes:
    mask_logits = torch.full((MASK_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH), -10.0, dtype=torch.float32)
    depth_logits = torch.zeros((DEPTH_CHANNELS, HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=torch.float32)
    if with_gate:
        x0, x1 = 70, 96
        y0, y1 = 30, 52
        mask_logits[0, y0:y1, x0] = 10.0
        mask_logits[0, y0:y1, x1] = 10.0
        mask_logits[0, y0, x0 : x1 + 1] = 10.0
        mask_logits[0, y1, x0 : x1 + 1] = 10.0
    return serialize_raw_logits(frame_id=frame_id, sim_time_ns=frame_id * 1000, mask_logits=mask_logits, depth_logits=depth_logits)


def _require_checkpoint() -> None:
    if not DEFAULT_REGRESSOR_CHECKPOINT.is_file():
        pytest.skip(f"regressor checkpoint missing: {DEFAULT_REGRESSOR_CHECKPOINT}")


def test_parse_raw_logits_rejects_bad_magic() -> None:
    data = bytearray(_synthetic_logits(frame_id=3, with_gate=False))
    data[0:4] = b"BAD!"

    with pytest.raises(ValueError, match="magic"):
        parse_raw_logits(bytes(data))


def test_regressor_emits_minimal_json_with_quad_pinned_center() -> None:
    _require_checkpoint()
    raw = parse_raw_logits(_synthetic_logits(frame_id=7, with_gate=True))
    runner = LogitRegressor(DEFAULT_REGRESSOR_CHECKPOINT, device="cpu", gate_threshold=0.50, confidence_threshold=0.50)

    frame = runner.run_frame(raw)

    assert frame.frame_id == 7
    assert len(frame.gates) == 1
    gate = frame.gates[0]
    assert gate.center_model_px == gate.quad_center_model_px
    assert gate.confidence > 0.99
    assert all(math.isfinite(value) for value in gate.position_xyz)
    orientation_norm = math.sqrt(sum(value * value for value in gate.orientation_xyz))
    assert orientation_norm == pytest.approx(1.0, abs=1e-5)

    payload = build_surveyer_payload(frame, output_dir=Path("run-test"))
    assert payload["run"]["cycle"] == 7
    assert payload["run"]["frame_id"] == "frame_000007"
    assert payload["obstacles"] == []
    assert len(payload["gates"]) == 1
    assert set(payload["gates"][0]) == {
        "id",
        "position_xyz",
        "position_confidence",
        "orientation_xyz",
        "orientation_confidence",
    }
    assert payload["gates"][0]["id"] == "gate-a7"


def test_regressor_empty_frame_outputs_no_gates() -> None:
    _require_checkpoint()
    raw = parse_raw_logits(_synthetic_logits(frame_id=8, with_gate=False))
    runner = LogitRegressor(DEFAULT_REGRESSOR_CHECKPOINT, device="cpu", gate_threshold=0.50, confidence_threshold=0.50)

    frame = runner.run_frame(raw)
    payload = build_surveyer_payload(frame, output_dir=Path("run-test"))

    assert frame.gates == []
    assert payload["gates"] == []
    assert payload["obstacles"] == []


def test_regressor_pipeline_writes_frame_json_and_jsonl(tmp_path: Path) -> None:
    _require_checkpoint()
    input_dir = tmp_path / "logits"
    output_dir = tmp_path / "json"
    input_dir.mkdir()
    (input_dir / "frame_000005.bin").write_bytes(_synthetic_logits(frame_id=5, with_gate=True))
    (input_dir / "frame_000006.bin").write_bytes(_synthetic_logits(frame_id=6, with_gate=False))

    stats = run_regressor_pipeline(
        RegressorPipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint=DEFAULT_REGRESSOR_CHECKPOINT,
            device="cpu",
        )
    )

    assert stats.frames_processed == 2
    assert stats.gates_emitted == 1
    assert (output_dir / "frame_000005.json").is_file()
    assert (output_dir / "frame_000006.json").is_file()
    lines = stats.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["run"]["frame_id"] == "frame_000005"
    assert len(payload["gates"]) == 1
