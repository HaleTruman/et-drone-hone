"""Run the latest training-run-02 checkpoint on reference test frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SyntheticData.src.models import config
from SyntheticData.src.models.inference import detect_gates_in_images
from SyntheticData.src.models.testing.evaluate import load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=config.REFERENCE_INPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=config.REFERENCE_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=config.VALIDATION_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--duplicate-iou-threshold", type=float)
    parser.add_argument("--duplicate-containment-area-ratio", type=float)
    parser.add_argument(
        "--suppress-contained-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=config.VALIDATION_BATCH_SIZE)
    parser.add_argument("--tracking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = discover_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else latest_checkpoint(args.model_dir).resolve()
    )
    run_dir = create_run_dir(args.output_root, args.run_name)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Frames: {len(image_paths)} from {args.input_dir}", flush=True)
    print(f"Validation run: {run_dir}", flush=True)

    results = []
    batch_timings = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        print(
            f"Predicting frames {start + 1}-{start + len(batch_paths)} / {len(image_paths)}",
            flush=True,
        )
        batch_started = time.perf_counter()
        results.extend(
            detect_gates_in_images(
                batch_paths,
                checkpoint_path,
                args.device,
                args.score_threshold,
                args.duplicate_iou_threshold,
                args.suppress_contained_duplicates,
                args.duplicate_containment_area_ratio,
                tracking=args.tracking,
            )
        )
        elapsed_seconds = time.perf_counter() - batch_started
        batch_timings.append(
            {
                "start_frame": start + 1,
                "end_frame": start + len(batch_paths),
                "frame_count": len(batch_paths),
                "elapsed_seconds": elapsed_seconds,
                "seconds_per_frame": elapsed_seconds / len(batch_paths),
            }
        )
    records = write_annotated_frames(frames_dir, results)
    write_predictions(run_dir / "predictions.jsonl", records)
    metrics = build_metrics(
        records,
        checkpoint_path,
        load_checkpoint(checkpoint_path),
        args.input_dir,
        run_dir,
        batch_timings,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote annotated frames to {frames_dir}", flush=True)
    print(f"Wrote predictions to {run_dir / 'predictions.jsonl'}", flush=True)
    print(f"Wrote metrics to {run_dir / 'metrics.json'}", flush=True)
    timing = metrics["metrics"]["model_seconds_per_frame_mean"]
    print(f"Average model time per frame: {timing:.4f}s", flush=True)


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = [path for path in model_dir.glob("*.pt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {model_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def create_run_dir(output_root: Path, run_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = output_root / run_name
        if run_dir.exists():
            raise FileExistsError(f"Validation run already exists: {run_dir}")
        run_dir.mkdir()
        return run_dir

    existing_numbers = []
    for path in output_root.iterdir():
        if not path.is_dir() or not path.name.startswith("run_"):
            continue
        suffix = path.name.removeprefix("run_")
        if suffix.isdigit():
            existing_numbers.append(int(suffix))

    next_number = max(existing_numbers, default=0) + 1
    run_dir = output_root / f"run_{next_number:03d}"
    run_dir.mkdir()
    return run_dir


def write_annotated_frames(
    frames_dir: Path,
    results: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for frame_index, result in enumerate(results, start=1):
        image_path = Path(str(result["image_path"]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw_predictions(image, result["detections"])

        output_name = f"frame_{frame_index:06d}_{image_path.stem}.jpg"
        output_path = frames_dir / output_name
        image.save(output_path, quality=92)

        detections = result["detections"]
        records.append(
            {
                "sample_id": f"validation/{frames_dir.parent.name}/frame_{frame_index:06d}",
                "image_path": str(image_path),
                "frame_path": str(output_path),
                "frame_file": output_name,
                "gate_count": len(detections),
                "detections": detections,
                "targets": [],
            }
        )
    return records


def draw_predictions(image: Image.Image, detections: Sequence[dict[str, object]]) -> None:
    draw = ImageDraw.Draw(image)
    for index, detection in enumerate(detections):
        box = detection["bbox_xyxy_px"]
        corners = detection["outer_corners_px"]
        color = (0, 220, 255)
        draw.line([tuple(corners[i]) for i in (0, 1, 3, 2, 0)], fill=color, width=3)
        draw.rectangle(tuple(box), outline=color, width=2)
        for corner in corners:
            x, y = corner
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(0, 0, 0), width=1)
        label = (
            f"track {detection.get('track_id', index + 1)} "
            f"{float(detection['score']):.2f}"
        )
        draw.text(
            (float(box[0]), max(0.0, float(box[1]) - 13.0)),
            label,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    draw.text(
        (8, 8),
        f"Predicted gates: {len(detections)}",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def write_predictions(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as predictions_file:
        for record in records:
            predictions_file.write(json.dumps(record) + "\n")


def build_metrics(
    records: Sequence[dict[str, object]],
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    input_dir: Path,
    run_dir: Path,
    batch_timings: Sequence[dict[str, float | int]],
) -> dict[str, object]:
    scores = [
        float(detection["score"])
        for record in records
        for detection in record["detections"]
    ]
    detection_counts = [int(record["gate_count"]) for record in records]
    pose_solved_count = sum(
        1
        for record in records
        for detection in record["detections"]
        if detection.get("relative_position_camera_frame_m")
        and detection.get("relative_orientation_euler_deg")
    )
    total_detections = sum(detection_counts)
    total_model_seconds = sum(
        float(timing["elapsed_seconds"]) for timing in batch_timings
    )
    model_seconds_per_frame = [
        float(timing["seconds_per_frame"]) for timing in batch_timings
    ]
    return {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "input_dir": str(input_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "score_threshold": checkpoint.get("model_config", {}).get("score_threshold"),
        "model_config": checkpoint.get("model_config", {}),
        "timing": {
            "note": (
                "Measured around detect_gates_in_images for each batch; includes "
                "checkpoint/model setup performed by that helper."
            ),
            "total_model_seconds": total_model_seconds,
            "batch_timings": list(batch_timings),
        },
        "metrics": {
            "frame_count": len(records),
            "predicted_gate_count": total_detections,
            "empty_frame_count": sum(1 for count in detection_counts if count == 0),
            "detections_per_frame_mean": mean(detection_counts) if detection_counts else 0.0,
            "detections_per_frame_min": min(detection_counts, default=0),
            "detections_per_frame_max": max(detection_counts, default=0),
            "score_mean": mean(scores) if scores else None,
            "score_min": min(scores, default=None),
            "score_max": max(scores, default=None),
            "pose_solved_count": pose_solved_count,
            "model_seconds_total": total_model_seconds,
            "model_seconds_per_frame_mean": (
                total_model_seconds / len(records) if records else 0.0
            ),
            "model_milliseconds_per_frame_mean": (
                (total_model_seconds / len(records)) * 1000.0 if records else 0.0
            ),
            "model_seconds_per_frame_min_batch": (
                min(model_seconds_per_frame, default=0.0)
            ),
            "model_seconds_per_frame_max_batch": (
                max(model_seconds_per_frame, default=0.0)
            ),
        },
    }


if __name__ == "__main__":
    main()
