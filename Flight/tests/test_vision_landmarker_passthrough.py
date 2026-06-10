import json
from types import SimpleNamespace

from sensing.vision.landmarker.landmark_output import build_passthrough_controller_payload
from sensing.vision.landmarker.landmarker_pipeline import LandmarkerPipelineConfig, run_landmarker_pipeline
from sensing.vision.regressor.logit_inference import FrameRegression, GateRegression
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService


def regressor_frame(gates: list[dict]) -> dict:
    return {
        "run": {"cycle": 9, "frame_id": "frame_000009", "sim_time_ns": 1234},
        "gates": gates,
        "obstacles": [],
    }


def gate(gate_id: str, position_xyz: list[float]) -> dict:
    return {
        "id": gate_id,
        "position_xyz": position_xyz,
        "position_confidence": 0.8,
        "orientation_xyz": [0.0, 0.0, 1.0],
        "orientation_confidence": 0.7,
    }


def regression_gate(position_xyz: tuple[float, float, float]) -> GateRegression:
    return GateRegression(
        center_model_px=(0.0, 0.0),
        center_px=(0.0, 0.0),
        quad_center_model_px=(0.0, 0.0),
        quad_model_px=[],
        position_xyz=position_xyz,
        orientation_xyz=(0.0, 0.0, 1.0),
        confidence=0.8,
        distance_camera_m=0.0,
        quad_fit_source="test",
        quad_is_fallback=False,
    )


def test_passthrough_controller_payload_uses_nearest_regressor_targets() -> None:
    payload = build_passthrough_controller_payload(
        regressor_frame(
            [
                gate("far", [0.0, 0.0, 8.0]),
                gate("near", [0.0, 0.0, 2.0]),
                gate("mid", [0.0, 0.0, 4.0]),
            ]
        ),
        output_dir="memory",
        top_k=2,
    )

    assert [item["id"] for item in payload["gates"]] == ["near", "mid"]
    assert payload["run"] == {
        "output_dir": "memory",
        "cycle": 9,
        "frame_id": "frame_000009",
        "sim_time_ns": 1234,
    }
    assert payload["obstacles"] == []


def test_passthrough_controller_payload_allows_zero_targets() -> None:
    payload = build_passthrough_controller_payload(regressor_frame([gate("near", [0.0, 0.0, 2.0])]), output_dir="memory", top_k=0)

    assert payload["gates"] == []


def test_landmarker_pipeline_passthrough_skips_state_update(tmp_path) -> None:
    input_jsonl = tmp_path / "regressor_frames.jsonl"
    input_jsonl.write_text(
        json.dumps(
            regressor_frame(
                [
                    gate("far", [0.0, 0.0, 8.0]),
                    gate("near", [0.0, 0.0, 2.0]),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "controller"
    state_path = tmp_path / "landmarker_state.json"

    stats = run_landmarker_pipeline(
        LandmarkerPipelineConfig(
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            state_path=state_path,
            top_k=1,
            passthrough_regressor_targets=True,
        )
    )

    line = json.loads((output_dir / "controller_frames.jsonl").read_text(encoding="utf-8"))
    assert [item["id"] for item in line["gates"]] == ["near"]
    assert stats.frames_processed == 1
    assert stats.final_landmark_count == 0
    assert not state_path.exists()


def test_service_passthrough_returns_top_k_without_creating_landmarker(monkeypatch) -> None:
    import sensing.vision.service as service_module

    class FakeCnn:
        def run_frame(self, tensor):
            return SimpleNamespace(mask_logits=None, depth_logits=None)

    class FakeRegressor:
        def run_frame(self, raw_logits):
            return FrameRegression(
                frame_id=7,
                sim_time_ns=555,
                source_width=640,
                source_height=360,
                gates=[
                    regression_gate((0.0, 0.0, 6.0)),
                    regression_gate((0.0, 0.0, 2.0)),
                ],
            )

    monkeypatch.setattr(service_module, "jpeg_bytes_to_tensor", lambda jpeg_bytes: object())
    service = VisionPerceptionService(
        VisionPerceptionConfig(
            run_landmarker=True,
            passthrough_regressor_targets=True,
            top_k=1,
        ),
        cnn=FakeCnn(),
        regressor=FakeRegressor(),
    )

    observation = service.process_frame(frame_id=7, sim_time_ns=555, jpeg_bytes=b"jpeg")

    assert observation.source == "cnn_regressor_passthrough"
    assert len(observation.gates) == 1
    assert observation.gates[0].position_camera_m == (0.0, 0.0, 2.0)
    assert service.snapshot()["landmarker_loaded"] is False
