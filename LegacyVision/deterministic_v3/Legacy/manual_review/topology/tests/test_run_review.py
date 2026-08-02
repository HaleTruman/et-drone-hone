from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.manual_review.topology.run_review import (
    CANDIDATE_TOPOLOGY_PATH,
    PRODUCTION_TOPOLOGY_PATH,
    generate_review,
    historic_frames,
)
from src.manual_review.topology.topology import assess_frame as assess_candidate
from src.preprocessing import preprocess_frame


def _synthetic_standard_run(root: Path) -> tuple[Path, np.ndarray]:
    run = root / "run-synthetic-standard"
    frames = run / "vision_frames"
    frames.mkdir(parents=True)
    image = np.zeros((96, 96, 3), np.uint8)
    cv2.rectangle(image, (12, 12), (84, 84), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(image, (30, 30), (66, 66), (0, 0, 0), cv2.FILLED)
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    jpeg = payload.tobytes()
    relative = Path("vision_frames") / "frame-00000007-9000.jpg"
    (run / relative).write_bytes(jpeg)
    (run / "frames.jsonl").write_text(json.dumps({
        "frame_id": 7,
        "sim_time_ns": 9000,
        "jpeg_size": len(jpeg),
        "path": relative.as_posix(),
    }) + "\n", encoding="utf-8")
    lut = np.zeros(1 << 24, np.uint8)
    lut[0xFFFFFF] = 1
    return run, lut


def _frame_from_mask(mask: np.ndarray):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask != 0] = 255
    lut = np.zeros(1 << 24, np.uint8)
    lut[0xFFFFFF] = 1
    return preprocess_frame(
        frame_id=11,
        sim_time_ns=22,
        image=image,
        lut=lut,
    )


def test_topology_candidate_is_intentionally_isolated() -> None:
    production = PRODUCTION_TOPOLOGY_PATH.read_bytes()
    candidate = CANDIDATE_TOPOLOGY_PATH.read_bytes()
    assert candidate != production
    assert hashlib.sha256(candidate).digest() != hashlib.sha256(production).digest()
    assert b"standard_minimum_parent_distance_depth_ratio" in candidate
    assert b"standard_minimum_parent_distance_depth_ratio" not in production


def test_baseline_replay_covers_and_renders_every_component(
    tmp_path: Path,
) -> None:
    run, lut = _synthetic_standard_run(tmp_path)
    manifest_path = generate_review(
        run,
        output_root=tmp_path / "output",
        review_id="fixed-test",
        mode="candidate",
        lut=lut,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["processed_frame_count"] == 1
    assert manifest["component_count"] == 1
    assert manifest["visualization_count"] == 1
    assert manifest["candidate_topology_label_counts"] == {"standard": 1}
    assert manifest["topology_source_files_byte_identical"] is False
    assert manifest["decision_equivalent_to_production"] is True

    records_path = manifest_path.parent / manifest["records_path"]
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["identity"]["frame_id"] == 7
    assert record["identity"]["sim_time_ns"] == 9000
    assert record["production_TopologyDecision"] == (
        record["candidate_TopologyDecision"])
    assert record["distance_transform_review"]["foreground_distance_peak_px"] > 0
    assert record["distance_transform_review"]["parent_transform_count"] == 1
    parent = record["distance_transform_review"]["parent_distance_evidence"][0]
    assert parent["parent_distance_depth_ratio"] >= 0.80
    assert parent["accepted"] is True
    assert record["distance_transform_review"]["aperture_transform_count"] == 1
    visualization = manifest_path.parent / record["visualization_path"]
    rendered = cv2.imread(str(visualization), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert rendered.dtype == np.uint8
    assert rendered.size > 0
    assert (manifest_path.parent / "index.html").is_file()
    assert (manifest_path.parent / "standard.html").is_file()

    with pytest.raises(FileExistsError):
        generate_review(
            run,
            output_root=tmp_path / "output",
            review_id="fixed-test",
            mode="candidate",
            lut=lut,
        )


def test_full_parent_distance_rejects_locally_balanced_child() -> None:
    mask = np.zeros((120, 120), np.uint8)
    cv2.rectangle(mask, (10, 10), (78, 65), 1, cv2.FILLED)
    cv2.rectangle(mask, (42, 55), (78, 105), 1, cv2.FILLED)
    cv2.rectangle(mask, (50, 75), (70, 95), 0, cv2.FILLED)
    frame = _frame_from_mask(mask)
    decisions = assess_candidate(frame)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.topology_label == "unknown"
    assert decision.classification_rule == (
        "one_child_parent_distance_below_threshold")
    assert decision.rejection_reason == (
        "standard_parent_distance_depth_below_minimum")
    assert decision.aperture_center_evidence[0].failed_checks == (
        "parent_distance_depth_ratio_below_minimum",)


def test_nested_parent_child_pairs_are_evaluated_independently() -> None:
    mask = np.zeros((128, 128), np.uint8)
    cv2.rectangle(mask, (8, 8), (120, 120), 1, cv2.FILLED)
    cv2.rectangle(mask, (24, 24), (104, 104), 0, cv2.FILLED)
    cv2.rectangle(mask, (40, 40), (88, 88), 1, cv2.FILLED)
    cv2.rectangle(mask, (52, 52), (76, 76), 0, cv2.FILLED)
    decisions = assess_candidate(_frame_from_mask(mask))
    assert len(decisions) == 2
    assert all(decision.topology_label == "standard" for decision in decisions)
    pairs = {
        decision.significant_parent_child_contour_ids[0]
        for decision in decisions
    }
    assert len(pairs) == 2
    assert all(
        "parent_distance_depth_ratio_below_minimum"
        not in decision.aperture_center_evidence[0].failed_checks
        for decision in decisions
    )


def test_historic_frame_identity_validation(tmp_path: Path) -> None:
    run, _ = _synthetic_standard_run(tmp_path)
    records = historic_frames(run)
    assert [(item.frame_id, item.sim_time_ns) for item in records] == [(7, 9000)]
    manifest = run / "frames.jsonl"
    line = manifest.read_text(encoding="utf-8")
    manifest.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate frame_id"):
        historic_frames(run)


def test_production_modules_do_not_import_manual_review() -> None:
    production_root = PRODUCTION_TOPOLOGY_PATH.parent
    for filename in ("pipeline.py", "preprocessing.py", "schema.py", "topology.py"):
        source = (production_root / filename).read_text(encoding="utf-8")
        assert "manual_review" not in source
