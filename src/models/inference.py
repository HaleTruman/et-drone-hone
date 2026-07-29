"""Run a trained multi-gate detector on arbitrary images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as transform_functional

from src.models import config
from src.models.testing.evaluate import (
    _detection_to_dict,
    build_model_from_checkpoint,
    load_checkpoint,
)
from src.models.tracking import GateTracker
from src.models.training.metrics import decode_detections
from src.models.training.targets import CameraCalibration, GateFrame
from src.models.training.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.MODEL_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--duplicate-iou-threshold", type=float)
    parser.add_argument("--duplicate-containment-area-ratio", type=float)
    parser.add_argument(
        "--suppress-contained-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign persistent IDs across the ordered input images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = detect_gates_in_images(
        args.images,
        args.checkpoint,
        args.device,
        args.score_threshold,
        args.duplicate_iou_threshold,
        args.suppress_contained_duplicates,
        args.duplicate_containment_area_ratio,
        args.tracking,
    )
    lines = [json.dumps(result) for result in results]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Predictions: {args.output.resolve()}")
    else:
        print("\n".join(lines))
    if args.overlay_dir:
        write_inference_overlays(args.overlay_dir, results)


@torch.inference_mode()
def detect_gates_in_images(
    image_paths: Sequence[Path | str],
    checkpoint_path: Path | str,
    requested_device: str = "auto",
    score_threshold: float | None = None,
    duplicate_iou_threshold: float | None = None,
    suppress_contained_duplicates: bool | None = None,
    duplicate_containment_area_ratio: float | None = None,
    tracking: bool = True,
) -> list[dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["model_config"]
    threshold = (
        float(score_threshold)
        if score_threshold is not None
        else float(config["score_threshold"])
    )
    duplicate_iou = (
        float(duplicate_iou_threshold)
        if duplicate_iou_threshold is not None
        else float(config.get("duplicate_iou_threshold", 0.25))
    )
    suppress_contained = (
        suppress_contained_duplicates
        if suppress_contained_duplicates is not None
        else bool(config.get("suppress_contained_duplicates", True))
    )
    containment_area_ratio = (
        float(duplicate_containment_area_ratio)
        if duplicate_containment_area_ratio is not None
        else float(config.get("duplicate_containment_area_ratio", 0.85))
    )
    calibration = CameraCalibration.from_dict(checkpoint["camera_calibration"])
    device = choose_device(requested_device)
    model = build_model_from_checkpoint(checkpoint, threshold)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    tensors: list[torch.Tensor] = []
    frames: list[GateFrame] = []
    resolved_paths: list[Path] = []
    for index, value in enumerate(image_paths):
        path = Path(value).resolve()
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensors.append(
                transform_functional.pil_to_tensor(image).float().to(device) / 255.0
            )
        frames.append(
            GateFrame(
                image_path=path,
                run_name="inference",
                frame_number=index,
                width=width,
                height=height,
                gates=(),
            )
        )
        resolved_paths.append(path)

    outputs = model(tensors)
    detections = decode_detections(
        outputs,
        frames,
        calibration,
        threshold,
        duplicate_iou,
        suppress_contained,
        containment_area_ratio,
    )
    if tracking:
        tracker = GateTracker()
        detections = [tracker.update(frame_detections) for frame_detections in detections]
    return [
        {
            "image_path": str(path),
            "gate_count": len(frame_detections),
            "detections": [
                _detection_to_dict(detection) for detection in frame_detections
            ],
        }
        for path, frame_detections in zip(resolved_paths, detections)
    ]


def write_inference_overlays(
    output_dir: Path,
    results: Sequence[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        image_path = Path(result["image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for index, detection in enumerate(result["detections"]):
            box = detection["bbox_xyxy_px"]
            corners = detection["outer_corners_px"]
            polygon = [corners[i] for i in (0, 1, 3, 2, 0)]
            draw.line([tuple(point) for point in polygon], fill=(0, 220, 255), width=3)
            inner_corners = detection.get("inner_corners_px")
            if inner_corners is not None:
                inner_polygon = [inner_corners[i] for i in (0, 1, 3, 2, 0)]
                draw.line(
                    [tuple(point) for point in inner_polygon],
                    fill=(0, 150, 255),
                    width=2,
                )
            draw.rectangle(tuple(box), outline=(0, 220, 255), width=2)
            draw.text(
                (box[0], max(0.0, box[1] - 12.0)),
                (
                    f"track {detection.get('track_id', index + 1)}: "
                    f"{detection['score']:.2f}"
                ),
                fill=(0, 220, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        image.save(output_dir / f"{image_path.stem}_gates.jpg", quality=92)
    print(f"Overlays: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
