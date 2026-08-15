import json

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.ui.backend.diagnose_shared_pipeline_runtime import (
    PROFILE_STAGES, STAGES, run_diagnostics, summarize_samples,
    validate_limits)
from sensing.vision.models.deterministic_v3.ui.backend.replay_historic_run import (
    HistoricRunSource)


def _fixture(tmp_path, *, touches_frame=False):
    run = tmp_path / "run-runtime-fixture"
    frames = run / "vision_frames"
    frames.mkdir(parents=True)
    image = np.zeros((72, 88, 3), np.uint8)
    left = 0 if touches_frame else 12
    image[8:64, left:76] = 255
    image[26:46, 32:56] = 0
    ok, encoded = cv2.imencode(".jpg", image,
                              [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    jpeg = encoded.tobytes()
    relative = "vision_frames/frame-00000017-123456789.jpg"
    (run / relative).write_bytes(jpeg)
    (run / "frames.jsonl").write_text(json.dumps({
        "frame_id": 17,
        "sim_time_ns": 123_456_789,
        "jpeg_size": len(jpeg),
        "path": relative,
    }) + "\n", encoding="utf-8")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    lut = np.zeros(1 << 24, np.uint8)
    for blue, green, red in np.unique(decoded.reshape(-1, 3), axis=0):
        if int(blue) + int(green) + int(red) > 600:
            lut[(int(red) << 16) | (int(green) << 8) | int(blue)] = 1
    return tuple(HistoricRunSource(run)), lut


def test_runtime_summary_reports_milliseconds_and_reciprocal_hertz():
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])

    assert summary["mean_ms_per_frame"] == 2.5
    assert summary["effective_hz"] == 400.0
    assert summary["p95_ms_per_frame"] > summary["median_ms_per_frame"]


def test_runtime_diagnostics_cover_every_current_shared_stage(tmp_path):
    records, lut = _fixture(tmp_path)
    report = run_diagnostics(records, lut, warmup_frames=0)

    assert report["scope"]["measured_frames"] == 1
    assert report["scope"]["density_mode"] == \
        "all_configured_profiles_for_density_eligible_components"
    assert report["scope"]["density_profile_count"] == 10
    assert report["scope"]["ignore_frame_edge_clipped_for_density"] is True
    assert report["scope"]["maximum_input_components"] == 0
    assert tuple(report["frames"][0]["stages_ms"]) == STAGES
    assert report["frames"][0]["density_record_count"] == len(PROFILE_STAGES)
    assert report["frames"][0]["density_eligible_component_count"] == 1
    assert report["frames"][0]["density_ignored_component_count"] == 0
    for stage in STAGES:
        assert report["summary"][stage]["mean_ms_per_frame"] >= 0
        assert report["summary"][stage]["effective_hz"] > 0


def test_runtime_diagnostics_exclude_frame_edge_clipped_density(tmp_path):
    records, lut = _fixture(tmp_path, touches_frame=True)
    report = run_diagnostics(records, lut, warmup_frames=0)
    frame = report["frames"][0]

    assert frame["component_count"] == 1
    assert frame["density_eligible_component_count"] == 0
    assert frame["density_ignored_component_count"] == 1
    assert frame["density_ignored_component_ids"] == [1]
    assert frame["density_record_count"] == 0
    assert frame["stages_ms"]["density_bank_all_profiles"] >= 0


def test_runtime_limits_make_regressions_machine_readable():
    report = {"summary": {
        "fast": {"p95_ms_per_frame": 4.0},
        "slow": {"p95_ms_per_frame": 9.0},
    }}
    limits = {
        "version": "test-v1",
        "metric": "p95_ms_per_frame",
        "maximum_ms": {"fast": 5.0, "slow": 8.0},
    }

    validation = validate_limits(report, limits)

    assert validation["passed"] is False
    assert [check["passed"] for check in validation["checks"]] == [True, False]
