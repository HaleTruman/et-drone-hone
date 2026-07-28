"""Focused tests for multi-gate loading, pose solving, and detector plumbing."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.models.geometry import solve_gate_pose
from src.models.run_pipeline import discover_run_split
from src.models.testing.evaluate import clear_generated_overlays, write_overlays
from src.models.tracking import GateTracker
from src.models.training.dataset import GateDetectionDataset, split_train_validation
from src.models.training.losses import gate_pose_loss
from src.models.training.metrics import (
    calculate_metrics,
    circular_absolute_error,
    suppress_duplicate_detections,
    yaw_180_absolute_error,
)
from src.models.training.model import GateDetector
from src.models.training.targets import (
    GateFrame,
    GateTarget,
    canonical_image_corners,
    fit_camera_calibration,
    load_gate_samples,
)


ROOT = Path(__file__).resolve().parents[3]


def test_supplied_run_loads_every_frame_and_gate() -> None:
    frames, rejected = load_gate_samples(ROOT / "runs", ["run_001"])
    assert len(frames) > 0
    assert sum(len(frame.gates) for frame in frames) > len(frames)
    assert isinstance(rejected, list)
    assert max(len(frame.gates) for frame in frames) > 1


def test_dataset_returns_variable_length_detection_targets() -> None:
    frames, _ = load_gate_samples(ROOT / "runs", ["run_001"])
    train, validation = split_train_validation(frames, 0.2, seed=42)
    dataset = GateDetectionDataset(validation, augment=False)
    image, target, index = dataset[0]
    assert image.shape == (3, 360, 640)
    assert target["boxes"].shape == (len(validation[0].gates), 4)
    assert target["keypoints"].shape == (len(validation[0].gates), 4, 3)
    assert target["labels"].dtype == torch.int64
    assert index == 0
    assert len(train) + len(validation) == len(frames)
    assert abs(len(validation) - round(len(frames) * 0.2)) <= 1


def test_zero_gate_frames_are_loaded_as_negative_samples() -> None:
    frames, _ = load_gate_samples(ROOT / "runs", ["run_013"])
    zero_gate_frames = [frame for frame in frames if not frame.gates]
    assert zero_gate_frames
    dataset = GateDetectionDataset(zero_gate_frames[:1], augment=False)
    _, target, _ = dataset[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["keypoints"].shape == (0, 4, 3)


def test_duplicate_suppression_removes_same_gate_fragments() -> None:
    detections = [
        {
            "score": 0.99,
            "bbox_xyxy": np.asarray([100, 50, 300, 250], dtype=np.float32),
        },
        {
            "score": 0.80,
            "bbox_xyxy": np.asarray([120, 70, 260, 180], dtype=np.float32),
        },
        {
            "score": 0.70,
            "bbox_xyxy": np.asarray([420, 50, 520, 160], dtype=np.float32),
        },
    ]
    kept = suppress_duplicate_detections(detections)
    assert [round(item["score"], 2) for item in kept] == [0.99, 0.70]


def test_detector_has_single_gate_class_and_four_keypoints() -> None:
    model = GateDetector(
        pretrained=False,
        input_size=(128, 128),
        detections_per_image=16,
    )
    assert model.detector.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.detector.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 4
    assert model.normalization == "frozen_batch_norm"


def test_detector_loss_aggregation() -> None:
    losses = {
        "loss_classifier": torch.tensor(1.0),
        "loss_keypoint": torch.tensor(2.0),
    }
    total, components = gate_pose_loss(losses)
    assert float(total) == 3.0
    assert set(components) == set(losses)


def test_true_keypoints_recover_pose_and_yaw_modulo_180() -> None:
    frames, _ = load_gate_samples(ROOT / "runs", ["run_005"])
    calibration = fit_camera_calibration(frames)
    gate = next(gate for frame in frames for gate in frame.gates)
    pose = solve_gate_pose(gate.outer_corners, calibration)
    assert pose is not None
    np.testing.assert_allclose(pose["position_cm"], gate.position, atol=0.1)
    assert yaw_180_absolute_error(
        pose["orientation_deg"][0],
        gate.orientation_deg[0],
    ) < 0.01
    np.testing.assert_allclose(
        pose["orientation_deg"][1:],
        gate.orientation_deg[1:],
        atol=0.01,
    )
    assert pose["reprojection_error_px"] < 0.01


def test_image_geometric_corner_order_does_not_depend_on_actor_side() -> None:
    points = np.asarray([[90, 10], [10, 10], [90, 90], [10, 90]], dtype=np.float32)
    ordered = canonical_image_corners(points)
    np.testing.assert_array_equal(
        ordered,
        np.asarray([[10, 10], [90, 10], [10, 90], [90, 90]], dtype=np.float32),
    )


def test_yaw_error_has_180_degree_period() -> None:
    assert yaw_180_absolute_error(89.0, -89.0) == 2.0
    assert yaw_180_absolute_error(10.0, -170.0) == 0.0
    predicted = np.asarray([179.0, -179.0, 10.0])
    target = np.asarray([-179.0, 179.0, 350.0])
    np.testing.assert_allclose(circular_absolute_error(predicted, target), [2, 2, 20])


def test_detection_metrics_count_all_instances() -> None:
    frame = _synthetic_frame(Path("unused.png"))
    detections = [
        {
            "score": 0.9,
            "bbox_xyxy": frame.gates[0].bbox_xyxy.copy(),
            "outer_corners": frame.gates[0].outer_corners.copy(),
            "pose": {
                "position_cm": frame.gates[0].position.copy(),
                "orientation_deg": frame.gates[0].orientation_deg.copy(),
            },
        }
    ]
    metrics = calculate_metrics([detections], [frame])
    assert metrics["target_gate_count"] == 1
    assert metrics["predicted_gate_count"] == 1
    assert metrics["detection_f1"] == 1.0


def test_tracker_preserves_ids_and_creates_new_tracks() -> None:
    tracker = GateTracker()
    first = tracker.update(
        [
            {"bbox_xyxy": np.asarray([10, 10, 30, 30], dtype=np.float32)},
            {"bbox_xyxy": np.asarray([80, 10, 100, 30], dtype=np.float32)},
        ]
    )
    second = tracker.update(
        [
            {"bbox_xyxy": np.asarray([13, 10, 33, 30], dtype=np.float32)},
            {"bbox_xyxy": np.asarray([140, 10, 160, 30], dtype=np.float32)},
        ]
    )
    assert first[0]["track_id"] == second[0]["track_id"]
    assert second[1]["track_id"] not in {
        first[0]["track_id"],
        first[1]["track_id"],
    }


def test_pipeline_auto_discovers_runs_and_holds_out_newest(tmp_path: Path) -> None:
    for run_name in ("run_010", "run_002", "run_001"):
        (tmp_path / run_name).mkdir()
    train_runs, test_runs = discover_run_split(tmp_path, None, None)
    assert train_runs == ["run_001", "run_002"]
    assert test_runs == ["run_010"]


def test_evaluation_clears_only_generated_overlay_images(tmp_path: Path) -> None:
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    for file_name in ("old.jpg", "old.jpeg", "old.png", "keep.txt"):
        (overlay_dir / file_name).write_text("test", encoding="utf-8")
    clear_generated_overlays(overlay_dir)
    assert sorted(path.name for path in overlay_dir.iterdir()) == ["keep.txt"]


def test_evaluation_writes_one_overlay_per_frame(tmp_path: Path) -> None:
    frames = []
    for frame_number in range(1, 4):
        image_path = tmp_path / f"frame_{frame_number:06d}.png"
        Image.new("RGB", (160, 120), (20, 20, 20)).save(image_path)
        frames.append(_synthetic_frame(image_path, frame_number))
    decoded = [
        [
            {
                "score": 0.9,
                "bbox_xyxy": frame.gates[0].bbox_xyxy + 2,
                "outer_corners": frame.gates[0].outer_corners + 2,
                "pose": None,
            }
        ]
        for frame in frames
    ]
    overlay_dir = tmp_path / "overlays"
    write_overlays(overlay_dir, decoded, frames, requested_count=-1)
    assert len(list(overlay_dir.glob("*.jpg"))) == 3


def _synthetic_frame(image_path: Path, frame_number: int = 1) -> GateFrame:
    gate = GateTarget(
        gate_label="gate",
        outer_corners=np.asarray(
            [[20, 20], [80, 20], [20, 80], [80, 80]],
            dtype=np.float32,
        ),
        keypoint_visibility=np.full((4,), 2.0, dtype=np.float32),
        bbox_xyxy=np.asarray([18, 18, 82, 82], dtype=np.float32),
        position=np.asarray([10, 20, 300], dtype=np.float32),
        orientation_deg=np.asarray([5, 10, 15], dtype=np.float32),
    )
    return GateFrame(
        image_path=image_path,
        run_name="run_test",
        frame_number=frame_number,
        width=160,
        height=120,
        gates=(gate,),
    )
