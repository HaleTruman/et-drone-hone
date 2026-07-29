"""Evaluate all-gate detection, keypoints, and solved planar poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from src.models import config
from src.models.training.dataset import (
    GateDetectionDataset,
    detection_collate,
    seed_worker,
)
from src.models.training.metrics import calculate_metrics, decode_detections
from src.models.training.model import GateDetector
from src.models.training.targets import (
    CameraCalibration,
    GateFrame,
    load_gate_samples,
    scale_frames_for_training,
)
from src.models.training.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.MODEL_CHECKPOINT,
    )
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--runs", nargs="+", default=list(config.DEFAULT_TEST_RUNS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=config.LOG_INTERVAL,
        help="Print prediction progress every N batches. Use 0 to disable batch progress.",
    )
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument(
        "--threshold-sweep",
        nargs="*",
        type=float,
        default=list(config.THRESHOLD_SWEEP),
        help="Score thresholds to evaluate from one inference pass.",
    )
    parser.add_argument("--duplicate-iou-threshold", type=float)
    parser.add_argument("--duplicate-containment-area-ratio", type=float)
    parser.add_argument(
        "--suppress-contained-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=config.OVERLAY_COUNT,
        help="Frames to annotate: -1 saves all (default), 0 disables overlays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_interval < 0:
        raise ValueError("--log-interval cannot be negative")
    if (
        args.duplicate_iou_threshold is not None
        and not 0.0 <= args.duplicate_iou_threshold <= 1.0
    ):
        raise ValueError("--duplicate-iou-threshold must be between zero and one")
    if (
        args.duplicate_containment_area_ratio is not None
        and args.duplicate_containment_area_ratio < 0.0
    ):
        raise ValueError("--duplicate-containment-area-ratio cannot be negative")
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.threshold_sweep):
        raise ValueError("--threshold-sweep values must be between zero and one")
    log_status("Selecting device")
    device = choose_device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else checkpoint_path.parent / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    log_status(f"Preparing evaluation output at {output_dir}")
    clear_generated_overlays(overlay_dir)

    log_status(f"Loading checkpoint {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["model_config"]
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(config["score_threshold"])
    )
    duplicate_iou_threshold = (
        args.duplicate_iou_threshold
        if args.duplicate_iou_threshold is not None
        else float(config.get("duplicate_iou_threshold", 0.25))
    )
    suppress_contained_duplicates = (
        args.suppress_contained_duplicates
        if args.suppress_contained_duplicates is not None
        else bool(config.get("suppress_contained_duplicates", True))
    )
    duplicate_containment_area_ratio = (
        args.duplicate_containment_area_ratio
        if args.duplicate_containment_area_ratio is not None
        else float(config.get("duplicate_containment_area_ratio", 0.85))
    )
    calibration = CameraCalibration.from_dict(checkpoint["camera_calibration"])
    log_status(
        f"Loading evaluation samples from {args.data_root} "
        f"for runs: {', '.join(args.runs)}"
    )
    native_samples, rejected = load_gate_samples(args.data_root, args.runs)
    input_height, input_width = tuple(config["input_size"])
    samples = scale_frames_for_training(native_samples, input_width, input_height)
    log_status(
        f"Loaded {len(native_samples)} native frames and scaled them to "
        f"{input_width}x{input_height} for evaluation with "
        f"{sum(len(sample.gates) for sample in samples)} usable gates; "
        f"{len(rejected)} gate/frame records rejected"
    )
    dataset = GateDetectionDataset(samples, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        collate_fn=detection_collate,
    )
    inference_score_threshold = min([score_threshold, *args.threshold_sweep])
    model = build_model_from_checkpoint(checkpoint, inference_score_threshold)
    log_status("Loading model weights")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    log_status("Running predictions")
    outputs, ordered_samples = collect_model_outputs(
        model,
        loader,
        samples,
        device,
        args.log_interval,
    )
    decoded = decode_with_postprocess(
        outputs,
        ordered_samples,
        calibration,
        score_threshold,
        duplicate_iou_threshold,
        suppress_contained_duplicates,
        duplicate_containment_area_ratio,
    )
    log_status("Calculating metrics")
    metrics = calculate_metrics(decoded, samples)
    threshold_sweep = calculate_threshold_sweep(
        outputs,
        ordered_samples,
        calibration,
        args.threshold_sweep,
        duplicate_iou_threshold,
        suppress_contained_duplicates,
        duplicate_containment_area_ratio,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "runs": args.runs,
        "score_threshold": score_threshold,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "suppress_contained_duplicates": suppress_contained_duplicates,
        "duplicate_containment_area_ratio": duplicate_containment_area_ratio,
        "metrics": metrics,
        "threshold_sweep": threshold_sweep,
        "rejected_gates_or_frames": rejected,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    log_status(f"Wrote metrics to {output_dir / 'metrics.json'}")
    write_predictions(output_dir / "predictions.jsonl", decoded, samples)
    log_status(f"Wrote predictions to {output_dir / 'predictions.jsonl'}")
    write_overlays(overlay_dir, decoded, samples, args.overlay_count)
    if args.overlay_count != 0:
        log_status(f"Wrote overlays to {overlay_dir}")
    print(json.dumps(metrics, indent=2))
    if threshold_sweep:
        best_threshold = max(
            threshold_sweep,
            key=lambda item: float(item["detection_f1"]),
        )
        print(
            "Best threshold by F1: "
            f"{best_threshold['score_threshold']:.2f} "
            f"F1={best_threshold['detection_f1']:.3f} "
            f"precision={best_threshold['detection_precision']:.3f} "
            f"recall={best_threshold['detection_recall']:.3f}"
        )
    print(f"Evaluation artifacts: {output_dir}")


def load_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError(
            "This is a legacy single-gate checkpoint. Multi-gate inference requires "
            "a format_version=2 checkpoint; retrain the detector first."
        )
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict[str, object],
    score_threshold: float,
) -> GateDetector:
    config = checkpoint["model_config"]
    return GateDetector(
        pretrained=False,
        input_size=tuple(config["input_size"]),
        score_threshold=score_threshold,
        detections_per_image=int(config.get("detections_per_image", 32)),
        box_nms_threshold=float(config.get("box_nms_threshold", 0.5)),
        keypoint_count=int(config.get("keypoint_count", 4)),
    )


@torch.inference_mode()
def collect_model_outputs(
    model: GateDetector,
    loader: DataLoader,
    samples: Sequence[GateFrame],
    device: torch.device,
    log_interval: int,
) -> tuple[list[dict[str, torch.Tensor]], list[GateFrame]]:
    outputs: list[dict[str, torch.Tensor]] = []
    ordered_samples: list[GateFrame] = []
    batch_count = len(loader)
    for batch_index, (images, _, indices) in enumerate(loader, start=1):
        device_images = [image.to(device, non_blocking=True) for image in images]
        outputs.extend(
            {
                name: value.cpu()
                for name, value in output.items()
                if isinstance(value, torch.Tensor)
            }
            for output in model(device_images)
        )
        ordered_samples.extend(samples[index] for index in indices)
        if should_log_batch(batch_index, batch_count, log_interval):
            log_status(
                f"Prediction batch {batch_index}/{batch_count}: "
                f"frames={len(ordered_samples)}"
            )
    return outputs, ordered_samples


def decode_with_postprocess(
    outputs: Sequence[dict[str, torch.Tensor]],
    ordered_samples: Sequence[GateFrame],
    calibration: CameraCalibration,
    score_threshold: float,
    duplicate_iou_threshold: float,
    suppress_contained_duplicates: bool,
    containment_area_ratio: float,
) -> list[list[dict[str, object]]]:
    return decode_detections(
        outputs,
        ordered_samples,
        calibration,
        score_threshold,
        duplicate_iou_threshold,
        suppress_contained_duplicates,
        containment_area_ratio,
    )


def calculate_threshold_sweep(
    outputs: Sequence[dict[str, torch.Tensor]],
    ordered_samples: Sequence[GateFrame],
    calibration: CameraCalibration,
    thresholds: Sequence[float],
    duplicate_iou_threshold: float,
    suppress_contained_duplicates: bool,
    containment_area_ratio: float,
) -> list[dict[str, object]]:
    sweep: list[dict[str, object]] = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        decoded = decode_with_postprocess(
            outputs,
            ordered_samples,
            calibration,
            threshold,
            duplicate_iou_threshold,
            suppress_contained_duplicates,
            containment_area_ratio,
        )
        metrics = calculate_metrics(decoded, ordered_samples)
        sweep.append(
            {
                "score_threshold": threshold,
                "predicted_gate_count": metrics["predicted_gate_count"],
                "detection_precision": metrics["detection_precision"],
                "detection_recall": metrics["detection_recall"],
                "detection_f1": metrics["detection_f1"],
                "detection_true_positive": metrics["detection_true_positive"],
                "detection_false_positive": metrics["detection_false_positive"],
                "detection_false_negative": metrics["detection_false_negative"],
                "matched_corner_mean_distance_px": metrics[
                    "matched_corner_mean_distance_px"
                ],
                "matched_visible_corner_mean_distance_px": metrics[
                    "matched_visible_corner_mean_distance_px"
                ],
                "matched_all_corner_mean_distance_px": metrics[
                    "matched_all_corner_mean_distance_px"
                ],
                "matched_visible_all_corner_mean_distance_px": metrics[
                    "matched_visible_all_corner_mean_distance_px"
                ],
                "position_l2_mean_m": metrics["position_l2_mean_m"],
                "orientation_mean_mae_deg": metrics["orientation_mean_mae_deg"],
            }
        )
    return sweep


def write_predictions(
    path: Path,
    decoded: Sequence[Sequence[dict[str, object]]],
    samples: Sequence[GateFrame],
) -> None:
    with path.open("w", encoding="utf-8") as predictions_file:
        for detections, sample in zip(decoded, samples):
            record = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "frame_width_px": sample.width,
                "frame_height_px": sample.height,
                "detections": [_detection_to_dict(item) for item in detections],
                "targets": [
                    _target_to_dict(gate)
                    for gate in sample.gates
                ],
            }
            predictions_file.write(json.dumps(record) + "\n")


def write_overlays(
    output_dir: Path,
    decoded: Sequence[Sequence[dict[str, object]]],
    samples: Sequence[GateFrame],
    requested_count: int,
) -> None:
    if requested_count == 0 or not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if requested_count < 0:
        selected_indices = np.arange(len(samples))
    else:
        count = min(requested_count, len(samples))
        selected_indices = np.linspace(0, len(samples) - 1, count, dtype=int)
    for index in selected_indices:
        sample = samples[index]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        if image.size != (sample.width, sample.height):
            image = image.resize(
                (sample.width, sample.height),
                Image.Resampling.BILINEAR,
            )
        draw = ImageDraw.Draw(image)
        for gate in sample.gates:
            _draw_quad(draw, gate.outer_corners, (80, 255, 100), width=2)
            if gate.inner_corners is not None:
                _draw_quad(draw, gate.inner_corners, (80, 180, 100), width=1)
            draw.rectangle(tuple(gate.bbox_xyxy), outline=(80, 255, 100), width=1)
        for detection_index, detection in enumerate(decoded[index]):
            corners = np.asarray(detection["outer_corners"])
            box = np.asarray(detection["bbox_xyxy"])
            _draw_quad(draw, corners, (0, 220, 255), width=3)
            if detection.get("inner_corners") is not None:
                _draw_quad(draw, np.asarray(detection["inner_corners"]), (0, 150, 255), width=2)
            draw.rectangle(tuple(box), outline=(0, 220, 255), width=2)
            pose = detection.get("pose")
            label = f"#{detection_index + 1} {detection['score']:.2f}"
            if pose is not None:
                position = np.asarray(pose["position_m"])
                orientation = np.asarray(pose["orientation_deg"])
                label += (
                    f"  R/U/F {position[0]:.2f}/{position[1]:.2f}/{position[2]:.2f}m"
                    f"  Y180/P/R {orientation[0]:.1f}/{orientation[1]:.1f}/{orientation[2]:.1f}"
                )
            text_position = (float(box[0]), max(0.0, float(box[1]) - 12.0))
            draw.text(text_position, label, fill=(0, 220, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        legend = (
            f"GT gates: {len(sample.gates)} (green)  "
            f"Predicted: {len(decoded[index])} (cyan)"
        )
        draw.text((8, 8), legend, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        image.save(
            output_dir / f"{safe_run_file_stem(sample.run_name)}_frame_{sample.frame_number:06d}.jpg",
            quality=92,
        )


def clear_generated_overlays(output_dir: Path) -> None:
    if not output_dir.is_dir():
        return
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for image_path in output_dir.glob(pattern):
            if image_path.is_file():
                image_path.unlink()


def safe_run_file_stem(run_name: str) -> str:
    return run_name.replace("\\", "__").replace("/", "__")


def log_status(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def should_log_batch(batch_index: int, batch_count: int, log_interval: int) -> bool:
    return (
        batch_index == 1
        or batch_index == batch_count
        or (log_interval > 0 and batch_index % log_interval == 0)
    )


def _detection_to_dict(detection: dict[str, object]) -> dict[str, object]:
    pose = detection.get("pose")
    result: dict[str, object] = {
        "score": detection["score"],
        "bbox_xyxy_px": np.asarray(detection["bbox_xyxy"]).tolist(),
        "outer_corners_px": np.asarray(detection["outer_corners"]).tolist(),
        "keypoint_scores": np.asarray(detection["keypoint_scores"]).tolist(),
    }
    if detection.get("inner_corners") is not None:
        result["inner_corners_px"] = np.asarray(detection["inner_corners"]).tolist()
    if detection.get("keypoints_2d") is not None:
        result["keypoints_2d_px"] = np.asarray(detection["keypoints_2d"]).tolist()
    if "track_id" in detection:
        result["track_id"] = int(detection["track_id"])
    if pose is not None:
        result["relative_position_camera_frame_m"] = dict(
            zip(("right", "up", "forward"), np.asarray(pose["position_m"]).tolist())
        )
        result["relative_orientation_euler_deg"] = dict(
            zip(
                ("yaw_mod_180_deg", "pitch_deg", "roll_deg"),
                np.asarray(pose["orientation_deg"]).tolist(),
            )
        )
        result["pose_reprojection_error_px"] = pose["reprojection_error_px"]
    else:
        result["pose"] = None
    return result


def _target_to_dict(gate: object) -> dict[str, object]:
    result = {
        "gate_label": gate.gate_label,
        "is_target": gate.is_target,
        "outer_corners_px": np.asarray(gate.outer_corners).tolist(),
        "outer_keypoint_visibility": np.asarray(gate.keypoint_visibility).tolist(),
        "bbox_xyxy_px": np.asarray(gate.bbox_xyxy).tolist(),
        "relative_position_camera_frame_m": dict(
            zip(("right", "up", "forward"), np.asarray(gate.position).tolist())
        ),
        "relative_orientation_euler_deg": dict(
            zip(
                ("yaw_mod_180_deg", "pitch_deg", "roll_deg"),
                np.asarray(gate.orientation_deg).tolist(),
            )
        ),
    }
    if gate.inner_corners is not None:
        result["inner_corners_px"] = np.asarray(gate.inner_corners).tolist()
    if gate.all_corners is not None:
        result["keypoints_2d_px"] = np.asarray(gate.all_corners).tolist()
    if gate.all_keypoint_visibility is not None:
        result["keypoint_visibility"] = np.asarray(
            gate.all_keypoint_visibility
        ).tolist()
    return result


def _draw_quad(
    draw: ImageDraw.ImageDraw,
    corners: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if not np.isfinite(corners).all():
        return
    points = [tuple(point) for point in corners[[0, 1, 3, 2, 0]]]
    draw.line(points, fill=color, width=width, joint="curve")


if __name__ == "__main__":
    main()
