"""Run gate-pose training followed by held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SyntheticData.src.models import config
from SyntheticData.src.models.training.targets import discover_run_dirs, run_display_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument(
        "--train-runs",
        nargs="+",
        help=(
            "Training runs. By default, uses every discovered run except one "
            "seeded-random held-out run."
        ),
    )
    parser.add_argument(
        "--test-runs",
        nargs="+",
        help="Held-out runs. By default, selects one discovered run at random.",
    )
    parser.add_argument("--output-dir", type=Path, default=config.MODEL_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=config.LOG_INTERVAL,
        help="Print batch progress every N batches during training/evaluation.",
    )
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument(
        "--partial-frame-sampling-weight",
        type=float,
        default=config.PARTIAL_FRAME_SAMPLING_WEIGHT,
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=config.FREEZE_BACKBONE_EPOCHS)
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=config.OVERLAY_COUNT,
        help="Frames to annotate: -1 saves all (default), 0 disables overlays.",
    )
    parser.add_argument("--input-height", type=int, default=config.INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=config.INPUT_WIDTH)
    parser.add_argument("--score-threshold", type=float, default=config.SCORE_THRESHOLD)
    parser.add_argument("--detections-per-image", type=int, default=config.DETECTIONS_PER_IMAGE)
    parser.add_argument("--box-nms-threshold", type=float, default=config.BOX_NMS_THRESHOLD)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=config.DUPLICATE_IOU_THRESHOLD)
    parser.add_argument(
        "--duplicate-containment-area-ratio",
        type=float,
        default=config.DUPLICATE_CONTAINMENT_AREA_RATIO,
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
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_project_path(args.data_root)
    output_dir = resolve_project_path(args.output_dir)
    checkpoint = output_dir / "best.pt"
    evaluation_dir = output_dir / "evaluation"
    train_runs, test_runs = discover_run_split(
        data_root,
        args.train_runs,
        args.test_runs,
        args.seed,
    )
    print(f"Training runs: {', '.join(train_runs)}")
    print(f"Held-out test runs: {', '.join(test_runs)}")
    print(f"Output directory: {output_dir}", flush=True)

    train_command = [
        sys.executable,
        "-m",
        "SyntheticData.src.models.training.train",
        "--data-root",
        str(data_root),
        "--train-runs",
        *train_runs,
        "--test-runs",
        *test_runs,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--log-interval",
        str(args.log_interval),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
        "--partial-frame-sampling-weight",
        str(args.partial_frame_sampling_weight),
        "--freeze-backbone-epochs",
        str(args.freeze_backbone_epochs),
        "--input-height",
        str(args.input_height),
        "--input-width",
        str(args.input_width),
        "--score-threshold",
        str(args.score_threshold),
        "--detections-per-image",
        str(args.detections_per_image),
        "--box-nms-threshold",
        str(args.box_nms_threshold),
        "--duplicate-iou-threshold",
        str(args.duplicate_iou_threshold),
        "--duplicate-containment-area-ratio",
        str(args.duplicate_containment_area_ratio),
        (
            "--suppress-contained-duplicates"
            if args.suppress_contained_duplicates
            else "--no-suppress-contained-duplicates"
        ),
        "--pretrained" if args.pretrained else "--no-pretrained",
        (
            "--allow-random-init-on-pretrained-failure"
            if args.allow_random_init_on_pretrained_failure
            else "--no-allow-random-init-on-pretrained-failure"
        ),
    ]
    print("Starting training", flush=True)
    subprocess.run(train_command, cwd=REPO_ROOT, check=True)

    evaluate_command = [
        sys.executable,
        "-m",
        "SyntheticData.src.models.testing.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--runs",
        *test_runs,
        "--output-dir",
        str(evaluation_dir),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--overlay-count",
        str(args.overlay_count),
        "--score-threshold",
        str(args.score_threshold),
        "--duplicate-iou-threshold",
        str(args.duplicate_iou_threshold),
        "--duplicate-containment-area-ratio",
        str(args.duplicate_containment_area_ratio),
        (
            "--suppress-contained-duplicates"
            if args.suppress_contained_duplicates
            else "--no-suppress-contained-duplicates"
        ),
    ]
    print("Starting held-out evaluation", flush=True)
    subprocess.run(evaluate_command, cwd=REPO_ROOT, check=True)
    print(f"Best checkpoint: {checkpoint}")
    print(f"Test metrics: {evaluation_dir / 'metrics.json'}")


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def discover_run_split(
    data_root: Path,
    requested_train_runs: list[str] | None,
    requested_test_runs: list[str] | None,
    seed: int = config.SEED,
) -> tuple[list[str], list[str]]:
    data_root = data_root.resolve()
    discovered = sorted(
        (run_display_name(data_root, path) for path in discover_run_dirs(data_root)),
        key=_run_sort_key,
    )
    if not discovered:
        raise FileNotFoundError(f"No run_* directories found under {data_root}")

    if requested_train_runs is None and requested_test_runs is None:
        if len(discovered) < 2:
            raise ValueError(
                "Automatic splitting requires at least two runs so one can remain held out."
            )
        held_out = random.Random(seed).choice(discovered)
        return [run_name for run_name in discovered if run_name != held_out], [held_out]

    test_runs = (
        list(requested_test_runs)
        if requested_test_runs is not None
        else [run_name for run_name in discovered if run_name not in requested_train_runs]
    )
    train_runs = (
        list(requested_train_runs)
        if requested_train_runs is not None
        else [run_name for run_name in discovered if run_name not in test_runs]
    )
    if not train_runs:
        raise ValueError("No training runs remain after applying the requested split.")
    if not test_runs:
        raise ValueError("No held-out test runs remain after applying the requested split.")
    overlap = set(train_runs) & set(test_runs)
    if overlap:
        raise ValueError(f"Runs cannot be both training and test data: {sorted(overlap)}")
    return train_runs, test_runs


def _run_sort_key(run_name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"(?:dataset_(\d+)/)?run_(\d+)", run_name)
    if match:
        dataset_number = int(match.group(1) or 0)
        return 0, dataset_number * 1_000_000 + int(match.group(2))
    return 1, run_name


if __name__ == "__main__":
    main()
