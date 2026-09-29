from models.deterministic_v2.src.schema import VisionGateResult, VisionResults


def test_vision_gate_result_round_trip():
    gate = VisionGateResult(
        gate_id="gate-001",
        position_camera_m=(1.0, 2.0, 3.0),
        position_confidence=0.8,
        orientation_camera=(4.0, 5.0, 6.0),
        orientation_confidence=0.7,
        trace={"source": "test"},
    )
    payload = gate.to_dict()
    restored = VisionGateResult.from_dict(payload)
    assert restored.gate_id == "gate-001"
    assert restored.position_camera_m == (1.0, 2.0, 3.0)
    assert restored.orientation_camera == (4.0, 5.0, 6.0)
    assert restored.to_controller_payload()["id"] == "gate-001"


def test_vision_results_round_trip():
    result = VisionResults(
        run_id="run-test",
        frame_ordinal=4,
        frame_id="frame_000004",
        source_path="/tmp/frame.jpg",
        image_width=640,
        image_height=360,
        created_at="now",
        gates=[
            VisionGateResult(
                gate_id="gate-001",
                position_camera_m=(1.0, 2.0, 3.0),
                position_confidence=0.9,
            )
        ],
    )
    restored = VisionResults.from_dict(result.to_dict())
    assert restored.gates[0].gate_id == "gate-001"
    assert restored.coordinate_frame == "flight-camera-local"
    assert restored.camera_position_camera_m == (0.0, 0.0, 0.0)
