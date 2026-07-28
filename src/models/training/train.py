"""Train a detector that returns all gates and four keypoints per gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.models import config
from src.models.training.dataset import (
    GateDetectionDataset,
    detection_collate,
    seed_worker,
    split_train_validation,
)
from src.models.training.losses import gate_pose_loss
from src.models.training.metrics import calculate_metrics, decode_detections
from src.models.training.model import GateDetector
from src.models.training.targets import (
    CameraCalibration,
    GateFrame,
    fit_camera_calibration,
    load_gate_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--train-runs", nargs="+", default=list(config.DEFAULT_TRAIN_RUNS))
    parser.add_argument("--test-runs", nargs="*", default=list(config.DEFAULT_TEST_RUNS))
    parser.add_argument("--output-dir", type=Path, default=config.MODEL_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--backbone-lr-scale", type=float, default=config.BACKBONE_LR_SCALE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--validation-fraction", type=float, default=config.VALIDATION_FRACTION)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--input-height", type=int, default=config.INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=config.INPUT_WIDTH)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=config.LOG_INTERVAL,
        help="Print batch progress every N batches. Use 0 to disable batch progress.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--score-threshold", type=float, default=config.SCORE_THRESHOLD)
    parser.add_argument("--detections-per-image", type=int, default=config.DETECTIONS_PER_IMAGE)
    parser.add_argument("--box-nms-threshold", type=float, default=config.BOX_NMS_THRESHOLD)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=config.DUPLICATE_IOU_THRESHOLD)
    parser.add_argument(
        "--duplicate-containment-area-ratio",
        type=float,
        default=config.DUPLICATE_CONTAINMENT_AREA_RATIO,
        help=(
            "Suppress a lower-score detection whose center lies inside a kept box "
            "when its area is at most this fraction of the kept box area."
        ),
    )
    parser.add_argument(
        "--suppress-contained-duplicates",
        action=argparse.BooleanOptionalAction,
        default=config.SUPPRESS_CONTAINED_DUPLICATES,
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=config.PRETRAINED,
    )
    parser.add_argument(
        "--allow-random-init-on-pretrained-failure",
        action=argparse.BooleanOptionalAction,
        default=config.ALLOW_RANDOM_INIT_ON_PRETRAINED_FAILURE,
        help=(
            "Continue from random initialization if pretrained weights cannot "
            "be loaded. By default, requested pretrained weights must load."
        ),
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=config.FREEZE_BACKBONE_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    log_status("Validating configuration and selecting device")
    set_deterministic_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_status(
        f"Loading training samples from {args.data_root} "
        f"for runs: {', '.join(args.train_runs)}"
    )
    all_train_frames, rejected = load_gate_samples(args.data_root, args.train_runs)
    log_status(
        f"Loaded {len(all_train_frames)} frames with "
        f"{sum(len(frame.gates) for frame in all_train_frames)} usable gates; "
        f"{len(rejected)} gate/frame records rejected"
    )
    train_frames, validation_frames = split_train_validation(
        all_train_frames,
        args.validation_fraction,
        args.seed,
    )
    log_status(
        f"Split data into {len(train_frames)} train frames and "
        f"{len(validation_frames)} validation frames"
    )
    calibration = fit_camera_calibration(train_frames)
    log_status(
        "Estimated camera calibration "
        f"{calibration.frame_width}x{calibration.frame_height}, "
        f"fx={calibration.focal_x_px:.1f}, fy={calibration.focal_y_px:.1f}"
    )
    write_split_manifest(
        output_dir / "split_manifest.json",
        train_frames,
        validation_frames,
        args.test_runs,
        rejected,
    )
    log_status(f"Wrote split manifest to {output_dir / 'split_manifest.json'}")

    log_status(
        f"Building dataloaders with batch_size={args.batch_size}, "
        f"workers={args.workers}, pin_memory={device.type == 'cuda'}"
    )
    train_dataset = GateDetectionDataset(train_frames, augment=True)
    validation_dataset = GateDetectionDataset(validation_frames, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=detection_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        collate_fn=detection_collate,
    )

    input_size = (args.input_height, args.input_width)
    log_status(
        f"Building Keypoint R-CNN model, input_size={input_size}, "
        f"pretrained={args.pretrained}, score_threshold={args.score_threshold}, "
        f"detections_per_image={args.detections_per_image}"
    )
    model = GateDetector(
        pretrained=args.pretrained,
        input_size=input_size,
        score_threshold=args.score_threshold,
        detections_per_image=args.detections_per_image,
        box_nms_threshold=args.box_nms_threshold,
        allow_random_init_on_pretrained_failure=(
            args.allow_random_init_on_pretrained_failure
        ),
    ).to(device)
    freeze_backbone_epochs = (
        args.freeze_backbone_epochs if model.pretrained_loaded else 0
    )
    log_status(
        "Model initialized "
        f"(pretrained_loaded={model.pretrained_loaded}, "
        f"freeze_backbone_epochs={freeze_backbone_epochs})"
    )
    model.set_backbone_trainable(freeze_backbone_epochs <= 0)
    backbone_parameters = list(model.features.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    optimizer = AdamW(
        (
            {
                "params": backbone_parameters,
                "lr": args.learning_rate * args.backbone_lr_scale,
            },
            {"params": head_parameters, "lr": args.learning_rate},
        ),
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_gate_count = sum(len(frame.gates) for frame in train_frames)
    validation_gate_count = sum(len(frame.gates) for frame in validation_frames)
    print(
        f"Training on {device}: {len(train_frames)} frames/{train_gate_count} gates, "
        f"{len(validation_frames)} validation frames/{validation_gate_count} gates, "
        f"{len(rejected)} rejected gate records"
    )
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    best_f1 = -1.0
    best_loss_at_best_f1 = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        log_status(f"Epoch {epoch}/{args.epochs}: starting training")
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)
            log_status(f"Epoch {epoch}: unfroze detector backbone")

        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.log_interval,
        )
        log_status(f"Epoch {epoch}/{args.epochs}: starting validation")
        validation_loss, validation_metrics = validate(
            model,
            validation_loader,
            validation_frames,
            calibration,
            device,
            args.score_threshold,
            args.duplicate_iou_threshold,
            args.suppress_contained_duplicates,
            args.duplicate_containment_area_ratio,
            epoch,
            args.log_interval,
        )
        scheduler.step(validation_loss)
        epoch_record = {
            "epoch": epoch,
            "train": train_losses,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(epoch_record) + "\n")

        validation_f1 = float(validation_metrics["detection_f1"])
        improved = (
            validation_f1 > best_f1 + 1e-6
            or (
                abs(validation_f1 - best_f1) <= 1e-6
                and validation_loss < best_loss_at_best_f1 - 1e-6
            )
        )
        if improved:
            best_f1 = validation_f1
            best_loss_at_best_f1 = validation_loss
            epochs_without_improvement = 0
            log_status(
                f"Epoch {epoch}: validation F1 improved to {validation_f1:.3f}; "
                f"saving best checkpoint"
            )
            save_checkpoint(
                output_dir / "best.pt",
                model,
                calibration,
                args,
                epoch,
                validation_loss,
                validation_metrics,
            )
        else:
            epochs_without_improvement += 1
            log_status(
                f"Epoch {epoch}: validation did not improve "
                f"({epochs_without_improvement}/{args.patience})"
            )
        log_status(f"Epoch {epoch}: saving last checkpoint")
        save_checkpoint(
            output_dir / "last.pt",
            model,
            calibration,
            args,
            epoch,
            validation_loss,
            validation_metrics,
        )
        log_status(
            f"Epoch {epoch:03d} train={train_losses['total_loss']:.4f} "
            f"val={validation_loss:.4f} "
            f"F1={validation_metrics['detection_f1']:.3f} "
            f"corner={_format_metric(validation_metrics['matched_corner_mean_distance_px'], 'px')} "
            f"yaw180={_format_metric(validation_metrics['orientation_mae_deg']['yaw_mod_180'], 'deg')} "
            f"time={epoch_record['elapsed_seconds']:.1f}s"
        )
        if epochs_without_improvement >= args.patience:
            log_status(f"Early stopping after {epoch} epochs")
            break

    log_status(f"Best checkpoint: {output_dir / 'best.pt'}")


def train_one_epoch(
    model: GateDetector,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    frame_count = 0
    batch_count = len(loader)
    progress_start = time.perf_counter()
    for batch_index, (images, targets, _) in enumerate(loader, start=1):
        images, targets = move_batch(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        loss, components = gate_pose_loss(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = len(images)
        totals["total_loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] += float(value) * batch_size
        frame_count += batch_size
        if should_log_batch(batch_index, batch_count, log_interval):
            elapsed = time.perf_counter() - progress_start
            log_status(
                f"Epoch {epoch} train batch {batch_index}/{batch_count}: "
                f"loss={float(loss.detach()):.4f}, "
                f"frames={frame_count}, elapsed={elapsed:.1f}s"
            )
    return {name: value / frame_count for name, value in totals.items()}


@torch.inference_mode()
def validate(
    model: GateDetector,
    loader: DataLoader,
    samples: list[GateFrame],
    calibration: CameraCalibration,
    device: torch.device,
    score_threshold: float,
    duplicate_iou_threshold: float,
    suppress_contained_duplicates: bool,
    containment_area_ratio: float,
    epoch: int,
    log_interval: int,
) -> tuple[float, dict[str, object]]:
    total_loss = 0.0
    frame_count = 0
    model.train()
    batch_count = len(loader)
    progress_start = time.perf_counter()
    for batch_index, (images, targets, _) in enumerate(loader, start=1):
        images, targets = move_batch(images, targets, device)
        losses = model(images, targets)
        loss, _ = gate_pose_loss(losses)
        total_loss += float(loss) * len(images)
        frame_count += len(images)
        if should_log_batch(batch_index, batch_count, log_interval):
            elapsed = time.perf_counter() - progress_start
            log_status(
                f"Epoch {epoch} validation-loss batch {batch_index}/{batch_count}: "
                f"loss={float(loss):.4f}, frames={frame_count}, elapsed={elapsed:.1f}s"
            )

    model.eval()
    outputs: list[dict[str, torch.Tensor]] = []
    ordered_samples: list[GateFrame] = []
    progress_start = time.perf_counter()
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
            elapsed = time.perf_counter() - progress_start
            log_status(
                f"Epoch {epoch} validation-predict batch {batch_index}/{batch_count}: "
                f"frames={len(ordered_samples)}, elapsed={elapsed:.1f}s"
            )
    log_status(f"Epoch {epoch}: decoding detections and calculating metrics")
    decoded = decode_detections(
        outputs,
        ordered_samples,
        calibration,
        score_threshold,
        duplicate_iou_threshold,
        suppress_contained_duplicates,
        containment_area_ratio,
    )
    return total_loss / frame_count, calculate_metrics(decoded, ordered_samples)


def move_batch(
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    return (
        [image.to(device, non_blocking=True) for image in images],
        [
            {name: value.to(device, non_blocking=True) for name, value in target.items()}
            for target in targets
        ],
    )


def save_checkpoint(
    path: Path,
    model: GateDetector,
    calibration: CameraCalibration,
    args: argparse.Namespace,
    epoch: int,
    validation_loss: float,
    validation_metrics: dict[str, object],
) -> None:
    torch.save(
        {
            "format_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": {
                "architecture": "keypointrcnn_resnet50_fpn",
                "pretrained_requested": bool(args.pretrained),
                "pretrained_initialization": model.pretrained_loaded,
                "input_size": [args.input_height, args.input_width],
                "score_threshold": args.score_threshold,
                "detections_per_image": args.detections_per_image,
                "box_nms_threshold": args.box_nms_threshold,
                "duplicate_iou_threshold": args.duplicate_iou_threshold,
                "suppress_contained_duplicates": args.suppress_contained_duplicates,
                "duplicate_containment_area_ratio": (
                    args.duplicate_containment_area_ratio
                ),
                "keypoint_order": ["image_TL", "image_TR", "image_BL", "image_BR"],
                "yaw_period_degrees": 180,
            },
            "camera_calibration": calibration.to_dict(),
            "training_config": vars(args)
            | {
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
            },
            "validation_loss": validation_loss,
            "selection_metric": "detection_f1",
            "selection_metric_value": validation_metrics["detection_f1"],
            "validation_metrics": validation_metrics,
        },
        path,
    )


def write_split_manifest(
    path: Path,
    train_samples: list[GateFrame],
    validation_samples: list[GateFrame],
    test_runs: list[str],
    rejected: list[dict[str, object]],
) -> None:
    contents = {
        "format_version": 2,
        "train": [
            {"frame": sample.sample_id, "gate_count": len(sample.gates)}
            for sample in train_samples
        ],
        "validation": [
            {"frame": sample.sample_id, "gate_count": len(sample.gates)}
            for sample in validation_samples
        ],
        "held_out_test_runs": test_runs,
        "rejected_training_gates_or_frames": rejected,
    }
    path.write_text(json.dumps(contents, indent=2), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.log_interval < 0:
        raise ValueError("--log-interval cannot be negative")
    if args.input_height < 128 or args.input_width < 128:
        raise ValueError("detector input dimensions must each be at least 128 pixels")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between zero and one")
    if not 0.0 <= args.box_nms_threshold <= 1.0:
        raise ValueError("--box-nms-threshold must be between zero and one")
    if not 0.0 <= args.duplicate_iou_threshold <= 1.0:
        raise ValueError("--duplicate-iou-threshold must be between zero and one")
    if args.duplicate_containment_area_ratio < 0.0:
        raise ValueError("--duplicate-containment-area-ratio cannot be negative")
    overlap = set(args.train_runs) & set(args.test_runs)
    if overlap:
        raise ValueError(f"Runs cannot be both training and held-out test data: {overlap}")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _format_metric(value: float | None, suffix: str) -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def log_status(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def should_log_batch(batch_index: int, batch_count: int, log_interval: int) -> bool:
    return (
        batch_index == 1
        or batch_index == batch_count
        or (log_interval > 0 and batch_index % log_interval == 0)
    )


if __name__ == "__main__":
    main()
