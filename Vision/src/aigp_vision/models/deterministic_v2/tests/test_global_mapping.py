from aigp_vision.models.deterministic_v2.src.config import DEFAULT_PRESET_PATH, ProjectionConfig
from aigp_vision.models.deterministic_v2.src.global_mapping import GlobalGateMapper


def _solve(source: str, xyz: list[float]) -> dict:
    return {
        "source": source,
        "available": True,
        "xyzCameraM": xyz,
        "rpyCameraDeg": [1.0, 2.0, 3.0],
        "depthM": xyz[2],
        "reprojectionErrorPx": 0.5,
        "source_score": 1.0,
        "fit_score": 1.0,
        "fitQuality": {"overall": 1.0},
    }


def _tracks(profile_id: str, source: str, xyz: list[float]) -> dict:
    return {
        "schema": "projection-inner-void-profile-tracks.v1",
        "profile_id": profile_id,
        "profile_label": f"Config {profile_id}",
        "instances": [
            {
                "instance_id": f"{profile_id.lower()}-001",
                "observations": [
                    {
                        "observation_id": f"{profile_id}-obs-001",
                        "frame_ordinal": 0,
                        "frame_id": "frame_000000",
                        "void_id": f"{profile_id}-void-001",
                        "finalInstanceScore": 1.0,
                        "objects": [_solve(source, xyz)],
                    }
                ],
            }
        ],
    }


def test_global_mapping_merges_profile_samples_into_public_gate():
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    mapper = GlobalGateMapper(config)
    gates_by_frame, payload = mapper.process_run(
        [
            _tracks("A", "inner_void_quad_fit", [1.0, 2.0, 8.0]),
            _tracks("B", "inner_void_ellipse_fit", [1.1, 2.1, 8.1]),
            _tracks("C", "inner_void_detection_rect", [0.9, 1.9, 7.9]),
        ],
        [0],
        {0: "frame_000000"},
    )
    gates = gates_by_frame[0]
    assert payload["profile_count"] == 3
    assert payload["sample_count"] == 3
    assert len(gates) == 1
    assert gates[0].gate_id == "gate-001"
    assert gates[0].position_camera_m[1] < 0.0
    assert gates[0].trace["coordinate_transform"] == "opencv-camera-to-result-camera-flip-y"
