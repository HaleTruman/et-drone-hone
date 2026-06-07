from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision.tools.review import ReviewConfig, ReviewSink
from vision.tools.review.review_render import (
    ReviewRenderConfig,
    load_jsonl_by_frame_id,
    project_camera_xyz_to_source_px,
    render_gate_overlay_image,
    render_two_pane_review,
)


def _payload(frame_id: int, gates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run": {
            "output_dir": "run-test",
            "cycle": frame_id,
            "frame_id": f"frame_{frame_id:06d}",
            "sim_time_ns": frame_id * 1000,
        },
        "gates": gates,
        "obstacles": [],
    }


def _gate(gate_id: str, position_xyz: tuple[float, float, float]) -> dict[str, object]:
    return {
        "id": gate_id,
        "position_xyz": list(position_xyz),
        "position_confidence": 0.9,
        "orientation_xyz": [0.0, 0.0, 1.0],
        "orientation_confidence": 0.8,
    }


def test_project_camera_xyz_to_source_px_uses_regressor_intrinsics() -> None:
    assert project_camera_xyz_to_source_px([0.0, 0.0, 10.0], source_width=640, source_height=360) == pytest.approx((320.0, 180.0))
    assert project_camera_xyz_to_source_px([1.0, 1.0, 10.0], source_width=640, source_height=360) == pytest.approx((352.0, 148.0))
    assert project_camera_xyz_to_source_px([0.0, 0.0, -1.0], source_width=640, source_height=360) is None


def test_load_jsonl_by_frame_id_matches_frame_name(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    rows = [_payload(0, [_gate("gate-a0", (0.0, 0.0, 10.0))]), _payload(1, [])]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    frames = load_jsonl_by_frame_id(path)

    assert set(frames) == {"frame_000000", "frame_000001"}
    assert frames["frame_000000"]["gates"][0]["id"] == "gate-a0"


def test_render_two_pane_review_writes_mp4(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame_id in range(2):
        image = np.zeros((36, 64, 3), dtype=np.uint8)
        image[:, :, 1] = 40 + (frame_id * 40)
        assert cv2.imwrite(str(frames_dir / f"sample_frame_{frame_id:03d}.jpg"), image)

    regressor_jsonl = tmp_path / "regressor_frames.jsonl"
    controller_jsonl = tmp_path / "controller_frames.jsonl"
    regressor_rows = [
        _payload(0, [_gate("gate-a0", (0.0, 0.0, 10.0))]),
        _payload(1, [_gate("gate-a1", (0.5, 0.0, 10.0))]),
    ]
    controller_rows = [
        _payload(0, [_gate("gate-001w", (0.0, 0.0, 10.0))]),
        _payload(1, [_gate("gate-001w", (0.5, 0.0, 10.0))]),
    ]
    regressor_jsonl.write_text("\n".join(json.dumps(row) for row in regressor_rows) + "\n", encoding="utf-8")
    controller_jsonl.write_text("\n".join(json.dumps(row) for row in controller_rows) + "\n", encoding="utf-8")
    output_mp4 = tmp_path / "review.mp4"

    stats = render_two_pane_review(
        ReviewRenderConfig(
            frames_dir=frames_dir,
            regressor_jsonl=regressor_jsonl,
            controller_jsonl=controller_jsonl,
            output_mp4=output_mp4,
            fps=5.0,
        )
    )

    assert stats.frames_rendered == 2
    assert output_mp4.is_file()
    assert output_mp4.stat().st_size > 0
    capture = cv2.VideoCapture(str(output_mp4))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 128
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 36
    finally:
        capture.release()


def test_render_gate_overlay_image_writes_landmarker_target_overlay(tmp_path: Path) -> None:
    image = np.zeros((36, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    rgb_path = tmp_path / "rgb.jpg"
    assert cv2.imwrite(str(rgb_path), image)
    output_path = tmp_path / "overlay.jpg"

    result = render_gate_overlay_image(
        rgb_path=rgb_path,
        payload=_payload(1, [_gate("gate-001w", (0.0, 0.0, 10.0))]),
        output_path=output_path,
        title="frame_000001 | Landmarker Target Overlay",
        subtitle="controller egress; highlighted marker is nearest target",
    )

    assert result == output_path.resolve()
    assert output_path.is_file()
    assert output_path.stat().st_size > rgb_path.stat().st_size


def test_review_sink_records_landmarker_overlay_in_manifest(tmp_path: Path) -> None:
    image = np.zeros((36, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    review = ReviewSink(ReviewConfig(output_root=tmp_path / "review", run_mode="replay", has_truth=False))
    review.start_run({"run_mode": "replay"})
    review.record_cnn_frame(
        frame=SimpleNamespace(frame_id=1, sim_time_ns=0, jpeg_bytes=encoded.tobytes()),
        logits_path=tmp_path / "pipeline/lightmask_logits/frame_000001.bin",
    )
    controller_jsonl = tmp_path / "controller_frames.jsonl"
    controller_jsonl.write_text(json.dumps(_payload(1, [_gate("gate-001w", (0.0, 0.0, 10.0))])) + "\n", encoding="utf-8")

    review.record_landmarker_outputs(controller_jsonl=controller_jsonl)

    manifest = json.loads((tmp_path / "review/review_manifest.json").read_text(encoding="utf-8"))
    frame = manifest["frames"][0]
    assert frame["landmarker_overlay"] == "frames/frame_000001/landmarker_overlay.jpg"
    assert frame["landmarker_review"] == "frames/frame_000001/landmarker_review.json"
    assert (tmp_path / "review" / frame["landmarker_overlay"]).is_file()
    sidecar = json.loads((tmp_path / "review" / frame["landmarker_review"]).read_text(encoding="utf-8"))
    assert sidecar["nearest_target"]["id"] == "gate-001w"
