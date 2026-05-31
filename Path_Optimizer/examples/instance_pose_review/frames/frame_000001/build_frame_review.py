#!/usr/bin/env python3
"""Build a single-frame inference-only HTML/JSON review for a v2 checkpoint.

This script is intentionally scoped beside the sample review artifacts. The
current shared gate_mask_center_cross_stride4 code has advanced to a newer
schema, while this review folder points at a v2 checkpoint with 2 mask channels,
2 center channels, and 1 cross channel.

The label JSONL is used only as a frame manifest to find the source image. Gate
labels, target pose, and original instance metadata are not used in the decoded
review output.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import mobilenet_v3_large


REVIEW_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REVIEW_ROOT / "video_review_summary.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "frames" / "frame_000001"
SOURCE_WIDTH = 640
SOURCE_HEIGHT = 360


class MobileNetV3GateMaskCenterCrossDetector(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        mask_channels: int,
        center_channels: int,
        cross_channels: int,
    ) -> None:
        super().__init__()
        backbone = mobilenet_v3_large(weights=None).features
        self.features = nn.Sequential(*list(backbone.children())[:4])
        self.decoder = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
            nn.Upsample(size=(int(image_height), int(image_width)), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
        )
        self.mask_head = nn.Conv2d(64, int(mask_channels), kernel_size=1)
        self.center_head = nn.Conv2d(64, int(center_channels), kernel_size=1)
        self.cross_head = nn.Conv2d(64, int(cross_channels), kernel_size=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded = self.decoder(self.features(images))
        return {
            "mask_logits": self.mask_head(decoded),
            "center_logits": self.center_head(decoded),
            "cross_logits": self.cross_head(decoded),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    parser.add_argument("--frame-id", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--distractor-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--safe-z-static", type=float, default=2.5)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_record(label_jsonl: Path, frame_id: int) -> dict[str, Any]:
    with label_jsonl.open("r", encoding="utf-8") as handle:
        first_record: dict[str, Any] | None = None
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if first_record is None:
                first_record = record
            if int(record.get("frame_id", -1)) == int(frame_id):
                return record
    if first_record is None:
        raise ValueError(f"No records found in {label_jsonl}")
    raise ValueError(f"Frame id {frame_id} was not found in {label_jsonl}")


def image_to_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (int(width), int(height)):
        image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def load_model(checkpoint_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    mask_channels = list(config.get("mask_channels") or [])
    center_channels = list(config.get("center_channels") or [])
    cross_channels = list(config.get("cross_channels") or [])
    image_width = int(config.get("image_width") or 160)
    image_height = int(config.get("image_height") or 90)
    model = MobileNetV3GateMaskCenterCrossDetector(
        image_width=image_width,
        image_height=image_height,
        mask_channels=len(mask_channels),
        center_channels=len(center_channels),
        cross_channels=len(cross_channels),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def sigmoid_outputs(model: nn.Module, image_tensor: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
    return {
        "mask": torch.sigmoid(outputs["mask_logits"])[0].cpu().numpy(),
        "center": torch.sigmoid(outputs["center_logits"])[0].cpu().numpy(),
        "cross": torch.sigmoid(outputs["cross_logits"])[0].cpu().numpy(),
    }


def color_layer(probability: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    clipped = np.clip(probability, 0.0, 1.0)
    alpha = (clipped * 255.0).astype(np.uint8)
    rgba = np.zeros((probability.shape[0], probability.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = int(color[0])
    rgba[:, :, 1] = int(color[1])
    rgba[:, :, 2] = int(color[2])
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def grayscale_layer(probability: np.ndarray) -> Image.Image:
    gray = (np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def font(size: int = 13) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return as_float


def connected_components(
    probability: np.ndarray,
    *,
    threshold: float,
    min_area: int,
    scale_x: float,
    scale_y: float,
) -> list[dict[str, Any]]:
    binary = (probability >= float(threshold)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, Any]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        mask = labels == component_id
        peak = float(probability[mask].max()) if np.any(mask) else 0.0
        mean = float(probability[mask].mean()) if np.any(mask) else 0.0
        cx, cy = centroids[component_id]
        components.append(
            {
                "component_id": int(component_id),
                "area_model_px": area,
                "bbox_model_px": [x, y, w, h],
                "bbox_px": [
                    int(round(x * scale_x)),
                    int(round(y * scale_y)),
                    int(round(w * scale_x)),
                    int(round(h * scale_y)),
                ],
                "center_model_px": [float(cx), float(cy)],
                "center_px": [float(cx * scale_x), float(cy * scale_y)],
                "confidence": peak,
                "mean_probability": mean,
            }
        )
    components.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return components


def probability_at_model_point(probability: np.ndarray | None, point: list[float]) -> float | None:
    if probability is None:
        return None
    if len(point) != 2:
        return None
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or y >= probability.shape[0] or x >= probability.shape[1]:
        return None
    return float(probability[y, x])


def interpreted_gates(
    components: list[dict[str, Any]],
    center_probability: np.ndarray | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        results.append(
            {
                "instance_id": index,
                "kind": "gate",
                "model_component_id": component["component_id"],
                "bbox_px": component["bbox_px"],
                "center_px": component["center_px"],
                "center_model_px": component["center_model_px"],
                "confidence": component["confidence"],
                "center_heatmap_confidence": probability_at_model_point(
                    center_probability,
                    list(component["center_model_px"]),
                ),
                "position_xyz_camera_m": None,
                "orientation_normal_camera_ruf": None,
                "orientation_plane_camera_6": None,
                "pose_source": "not_available_in_inference_output",
                "model_pose_status": "not_predicted_by_this_checkpoint",
            }
        )
    return results


def edge_point_toward_target(
    probability: np.ndarray,
    component: dict[str, Any],
    target_model_xy: tuple[float, float] | None,
    *,
    scale_x: float,
    scale_y: float,
    threshold: float,
) -> list[float] | None:
    if target_model_xy is None:
        return None
    component_id = int(component["component_id"])
    binary = (probability >= float(threshold)).astype(np.uint8)
    _, labels, _, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ys, xs = np.nonzero(labels == component_id)
    if xs.size <= 0:
        return None
    centroid = centroids[component_id]
    direction = np.asarray([float(target_model_xy[0] - centroid[0]), float(target_model_xy[1] - centroid[1])], dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-6:
        return None
    unit = direction / length
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    scores = ((points[:, 0] - float(centroid[0])) * unit[0]) + ((points[:, 1] - float(centroid[1])) * unit[1])
    best = points[int(np.argmax(scores))]
    return [float((best[0] + 0.5) * scale_x), float((best[1] + 0.5) * scale_y)]


def interpreted_obstacles(
    components: list[dict[str, Any]],
    distractor_probability: np.ndarray,
    *,
    center_peak_model_xy: tuple[float, float] | None,
    scale_x: float,
    scale_y: float,
    threshold: float,
    safe_z_static: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        edge = edge_point_toward_target(
            distractor_probability,
            component,
            center_peak_model_xy,
            scale_x=scale_x,
            scale_y=scale_y,
            threshold=threshold,
        )
        results.append(
            {
                "instance_id": index,
                "kind": "avoidance_hint",
                "model_component_id": component["component_id"],
                "bbox_px": component["bbox_px"],
                "center_px": component["center_px"],
                "edge_px_toward_next_gate": edge,
                "safe_z_static": float(safe_z_static),
                "confidence": component["confidence"],
                "position_source": "image_plane_only_no_depth",
            }
        )
    return results


def peak_xy(probability: np.ndarray) -> tuple[int, int, float]:
    flat_index = int(np.argmax(probability))
    y, x = np.unravel_index(flat_index, probability.shape)
    return int(x), int(y), float(probability[y, x])


def center_peak_marker(
    probability: np.ndarray | None,
    *,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any] | None:
    if probability is None:
        return None
    x, y, confidence = peak_xy(probability)
    return {
        "kind": "pose_center_logit_peak",
        "model_px": [int(x), int(y)],
        "px": [float((float(x) + 0.5) * scale_x), float((float(y) + 0.5) * scale_y)],
        "confidence": float(confidence),
        "meaning": "strongest model-predicted target-center heatmap point in the image plane",
        "not_a_label": True,
        "not_3d_position": True,
    }


def selected_gate_from_center_peak(
    gates: list[dict[str, Any]],
    center_peak: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not gates or not isinstance(center_peak, dict):
        return None
    peak_px = center_peak.get("px")
    if not isinstance(peak_px, list) or len(peak_px) != 2:
        return None
    best_gate: dict[str, Any] | None = None
    best_distance = float("inf")
    for gate in gates:
        center_px = gate.get("center_px")
        if not isinstance(center_px, list) or len(center_px) != 2:
            continue
        distance = math.hypot(float(center_px[0]) - float(peak_px[0]), float(center_px[1]) - float(peak_px[1]))
        if distance < best_distance:
            best_distance = distance
            best_gate = gate
    if best_gate is None:
        return None
    return {
        "instance_id": best_gate.get("instance_id"),
        "model_component_id": best_gate.get("model_component_id"),
        "center_px": best_gate.get("center_px"),
        "confidence": best_gate.get("confidence"),
        "center_peak_distance_px": float(best_distance),
        "selection_rule": "nearest_gate_component_center_to_pose_center_logit_peak_px",
        "physical_closest_gate_status": "not_available_without_depth_or_pose_head",
    }


def save_layer_images(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    names: dict[str, list[str]],
) -> list[dict[str, Any]]:
    layer_dir = output_dir / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    layers: list[dict[str, Any]] = []
    for group_key, group_name in (("mask", "mask"), ("center", "center"), ("cross", "cross")):
        for index, channel_name in enumerate(names[group_key]):
            probability = arrays[group_key][index]
            path = layer_dir / f"{group_name}_{index:02d}_{channel_name}.png"
            grayscale_layer(probability).resize((SOURCE_WIDTH, SOURCE_HEIGHT), Image.Resampling.BILINEAR).save(path)
            x, y, max_value = peak_xy(probability)
            layers.append(
                {
                    "group": group_name,
                    "name": channel_name,
                    "prediction_max": max_value,
                    "prediction_peak_model_px": [x, y],
                    "prediction_peak_px": [float((x + 0.5) * (SOURCE_WIDTH / probability.shape[1])), float((y + 0.5) * (SOURCE_HEIGHT / probability.shape[0]))],
                    "image": path.relative_to(output_dir).as_posix(),
                }
            )
    return layers


def quantized_probability_map(probability: np.ndarray, scale: int = 10000) -> list[int]:
    clipped = np.clip(probability, 0.0, 1.0)
    return np.rint(clipped.reshape(-1) * int(scale)).astype(np.uint16).tolist()


def draw_overlay(
    source_image: Image.Image,
    arrays: dict[str, np.ndarray],
    names: dict[str, list[str]],
    gate_components: list[dict[str, Any]],
    obstacle_components: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base = source_image.convert("RGBA")
    colors = {
        "visible_gate_mask": (0, 220, 110),
        "distractor_mask": (255, 88, 48),
        "pose_center_logit": (0, 180, 255),
        "outer_corner_heatmap": (255, 220, 70),
        "diagonal_cross_heatmap": (190, 100, 255),
    }
    for group_key in ("mask", "center", "cross"):
        for index, channel_name in enumerate(names[group_key]):
            color = colors.get(channel_name, (255, 255, 255))
            layer = color_layer(arrays[group_key][index], color).resize(base.size, Image.Resampling.BILINEAR)
            alpha = np.asarray(layer.getchannel("A"), dtype=np.float32)
            alpha = np.clip(alpha * 0.42, 0, 255).astype(np.uint8)
            layer.putalpha(Image.fromarray(alpha, mode="L"))
            base = Image.alpha_composite(base, layer)

    draw = ImageDraw.Draw(base)
    label_font = font(12)
    small_font = font(10)

    for index, component in enumerate(gate_components, start=1):
        x, y, w, h = component["bbox_px"]
        cx, cy = component["center_px"]
        draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 120, 255), width=2)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=(0, 255, 120, 255), width=2)
        draw.text((x + 3, max(0, y - 15)), f"G{index} {component['confidence']:.2f}", fill=(0, 255, 120, 255), font=small_font)

    for index, component in enumerate(obstacle_components, start=1):
        x, y, w, h = component["bbox_px"]
        cx, cy = component["center_px"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 80, 50, 255), width=2)
        draw.line((cx - 5, cy, cx + 5, cy), fill=(255, 80, 50, 255), width=2)
        draw.line((cx, cy - 5, cx, cy + 5), fill=(255, 80, 50, 255), width=2)
        draw.text((x + 3, max(0, y - 15)), f"O{index} {component['confidence']:.2f}", fill=(255, 80, 50, 255), font=small_font)

    center_names = names["center"]
    if "pose_center_logit" in center_names:
        index = center_names.index("pose_center_logit")
        px, py, score = peak_xy(arrays["center"][index])
        source_x = (px + 0.5) * (SOURCE_WIDTH / arrays["center"][index].shape[1])
        source_y = (py + 0.5) * (SOURCE_HEIGHT / arrays["center"][index].shape[0])
        radius = 9
        draw.ellipse((source_x - radius, source_y - radius, source_x + radius, source_y + radius), outline=(0, 180, 255, 255), width=3)
        draw.line((source_x - radius - 5, source_y, source_x + radius + 5, source_y), fill=(0, 180, 255, 255), width=2)
        draw.line((source_x, source_y - radius - 5, source_x, source_y + radius + 5), fill=(0, 180, 255, 255), width=2)
        draw.text((source_x + 12, source_y + 8), f"center {score:.2f}", fill=(0, 180, 255, 255), font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output_path)


def build_html(output_dir: Path, review_json: dict[str, Any]) -> str:
    display_json = dict(review_json)
    display_json.pop("interactive_probability_maps", None)
    embedded_json = html.escape(json.dumps(display_json, indent=2, sort_keys=False))
    review_data_json = json.dumps(review_json, separators=(",", ":"), sort_keys=False).replace("</", "<\\/")
    gates = review_json["interpreted_instances"]["gates"]
    obstacles = review_json["interpreted_instances"]["obstacles"]
    layer_items = "\n".join(
        f"<figure><img src=\"{html.escape(layer['image'])}\" alt=\"{html.escape(layer['name'])}\">"
        f"<figcaption>{html.escape(layer['name'])}<br><span>max {layer['prediction_max']:.3f}</span></figcaption></figure>"
        for layer in review_json["inference"]["layers"]
    )
    source_contract = review_json["source_manifest_usage"]
    source_rows = "\n".join(
        (
            "<tr><th>Decoded From Model</th><td>mask components, center heatmap, corner heatmap, diagonal cross heatmap</td></tr>",
            "<tr><th>Center Peak Marker</th><td>the maximum value in the model's pose_center_logit heatmap; an image-plane target-center hint, not a label or 3D position</td></tr>",
            "<tr><th>Selected Gate Rule</th><td>nearest decoded gate component center to the center peak marker in image pixels; physical closest gate is unavailable without depth</td></tr>",
            f"<tr><th>Manifest Fields Read</th><td>{html.escape(str(source_contract['fields_read']))}</td></tr>",
            f"<tr><th>Original Label Fields Used</th><td>{html.escape(str(source_contract['original_label_fields_used']))}</td></tr>",
            "<tr><th>Position / Orientation</th><td>null because this checkpoint does not predict pose vectors</td></tr>",
        )
    )
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Frame Inference Review</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #14201c;
      --muted: #5b6863;
      --line: #d8ded9;
      --paper: #f7f8f6;
      --panel: #ffffff;
      --gate: #00a964;
      --obstacle: #d94b2b;
      --center: #0576c9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 12px;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 3px 8px;
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr);
      gap: 18px;
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    h2 {{
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      font-weight: 650;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }}
    .image-wrap {{ padding: 12px; }}
    .image-wrap img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #111;
    }}
    .image-wrap canvas {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #111;
    }}
    .overlay-stage {{
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      border: 1px solid var(--line);
      background: #111;
      overflow: hidden;
    }}
    .overlay-stage img {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
      border: 0;
      background: #111;
    }}
    .overlay-stage canvas {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
      border: 0;
      background: transparent;
      pointer-events: none;
    }}
    .render-status {{
      padding: 0 12px 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      padding: 12px;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: 170px minmax(120px, 1fr) 56px;
      align-items: center;
      gap: 10px;
    }}
    .control-row label {{
      font-size: 12px;
      color: var(--ink);
    }}
    .control-row input[type="range"] {{
      width: 100%;
      accent-color: #167466;
    }}
    .control-row output {{
      font-variant-numeric: tabular-nums;
      color: var(--muted);
      text-align: right;
      font-size: 12px;
    }}
    .hint-line {{
      color: var(--muted);
      font-size: 12px;
      padding: 0 12px 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 0 12px 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      display: inline-block;
      margin-right: 5px;
      border-radius: 2px;
      vertical-align: -1px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #fbfcfb;
    }}
    .contract th {{
      width: 180px;
    }}
    .contract td strong {{
      color: var(--ink);
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .layers {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      padding: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      overflow: hidden;
    }}
    figure img {{
      width: 100%;
      display: block;
      image-rendering: auto;
    }}
    figcaption {{
      padding: 7px 8px;
      font-size: 12px;
      color: var(--ink);
    }}
    figcaption span {{ color: var(--muted); }}
    pre {{
      margin: 0;
      padding: 12px;
      overflow: auto;
      max-height: 620px;
      background: #111917;
      color: #e7f0eb;
      font: 12px/1.45 Menlo, Consolas, monospace;
    }}
    .note {{
      padding: 11px 14px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
      background: #fbfcfb;
    }}
    @media (max-width: 980px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .meta {{ justify-content: flex-start; }}
      main {{ grid-template-columns: 1fr; padding: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>__FRAME_ID__</h1>
    <div class="meta">
      <span class="pill">checkpoint step __CHECKPOINT_STEP__</span>
      <span class="pill">schema __SCHEMA_VERSION__</span>
      <span class="pill"><span id="gateCount">__GATE_COUNT__</span> gate components</span>
      <span class="pill"><span id="obstacleCount">__OBSTACLE_COUNT__</span> obstacle components</span>
    </div>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>Overlay</h2>
        <div class="image-wrap">
          <div class="overlay-stage">
            <img id="sourceImage" src="__SOURCE_IMAGE__" alt="source frame">
            <canvas id="overlayCanvas" width="640" height="360" aria-label="interactive inference overlay"></canvas>
          </div>
        </div>
        <div id="renderStatus" class="render-status">Rendering overlay from inference probabilities...</div>
        <div class="legend">
          <span><i class="swatch" style="background:var(--gate)"></i>gate mask</span>
          <span><i class="swatch" style="background:var(--obstacle)"></i>distractor mask</span>
          <span><i class="swatch" style="background:var(--center)"></i>center peak</span>
          <span><i class="swatch" style="background:#fff500"></i>selected gate</span>
          <span><i class="swatch" style="background:#ffdc46"></i>corner heatmap</span>
          <span><i class="swatch" style="background:#be64ff"></i>diagonal cross</span>
        </div>
      </section>
      <section>
        <h2>Threshold Dials</h2>
        <div class="controls">
          <div class="control-row">
            <label for="gateThreshold">Gate mask threshold</label>
            <input id="gateThreshold" type="range" min="0.01" max="0.99" step="0.01" value="__GATE_THRESHOLD__">
            <output id="gateThresholdValue">__GATE_THRESHOLD__</output>
          </div>
          <div class="control-row">
            <label for="distractorThreshold">Obstacle mask threshold</label>
            <input id="distractorThreshold" type="range" min="0.01" max="0.99" step="0.01" value="__DISTRACTOR_THRESHOLD__">
            <output id="distractorThresholdValue">__DISTRACTOR_THRESHOLD__</output>
          </div>
          <div class="control-row">
            <label for="minArea">Minimum component area</label>
            <input id="minArea" type="range" min="1" max="120" step="1" value="__MIN_AREA__">
            <output id="minAreaValue">__MIN_AREA__</output>
          </div>
          <div class="control-row">
            <label for="overlayOpacity">Overlay opacity</label>
            <input id="overlayOpacity" type="range" min="0.05" max="0.85" step="0.05" value="0.35">
            <output id="overlayOpacityValue">0.35</output>
          </div>
        </div>
        <div class="hint-line">Lower thresholds are more lenient. Higher thresholds are stricter.</div>
      </section>
      <section>
        <h2>Prediction Layers</h2>
        <div class="layers">__LAYER_ITEMS__</div>
      </section>
    </div>
    <div class="stack">
      <section>
        <h2>Source Contract</h2>
        <table class="contract">
          <tbody>__SOURCE_ROWS__</tbody>
        </table>
        <div class="note">This page is inference-only. The JSONL is only used to find the PNG file for the selected frame.</div>
      </section>
      <section>
        <h2>Center Peak / Selected Gate</h2>
        <table>
          <tbody id="selectedGateBody"><tr><td>Move a threshold dial to decode.</td></tr></tbody>
        </table>
        <div class="note">The selected gate is the decoded gate closest to the blue center-peak marker in 2D pixels. It is not the physically closest gate.</div>
      </section>
      <section>
        <h2>Interpreted Gates</h2>
        <table>
          <thead><tr><th>ID</th><th>Conf.</th><th>Center PX</th><th>Position XYZ Camera M</th><th>Normal Camera RUF</th></tr></thead>
          <tbody id="gateTableBody"><tr><td colspan="5">Move a threshold dial to decode.</td></tr></tbody>
        </table>
        <div class="note">Inference-only: this checkpoint predicts masks and heatmaps, not position or orientation vectors, so pose fields remain null.</div>
      </section>
      <section>
        <h2>Obstacle Hints</h2>
        <table>
          <thead><tr><th>ID</th><th>Conf.</th><th>Edge PX Toward Gate</th><th>Safe Z</th></tr></thead>
          <tbody id="obstacleTableBody"><tr><td colspan="4">Move a threshold dial to decode.</td></tr></tbody>
        </table>
      </section>
      <section>
        <h2>Frame JSON</h2>
        <pre id="frameJson">__EMBEDDED_JSON__</pre>
      </section>
    </div>
  </main>
  <script id="reviewData" type="application/json">__REVIEW_DATA__</script>
  <script>
    const reviewData = JSON.parse(document.getElementById("reviewData").textContent);
    const maps = reviewData.interactive_probability_maps;
    const mapWidth = maps.width;
    const mapHeight = maps.height;
    const sourceWidth = maps.source_width;
    const sourceHeight = maps.source_height;
    const scaleX = sourceWidth / mapWidth;
    const scaleY = sourceHeight / mapHeight;
    const quantizationScale = maps.quantization_scale || 10000;
    const gateMap = (maps.maps.visible_gate_mask || []).map((value) => value / quantizationScale);
    const obstacleMap = (maps.maps.distractor_mask || []).map((value) => value / quantizationScale);
    const centerMap = (maps.maps.pose_center_logit || []).map((value) => value / quantizationScale);
    const canvas = document.getElementById("overlayCanvas");
    const ctx = canvas.getContext("2d");
    const sourceImage = document.getElementById("sourceImage");
    const gateThreshold = document.getElementById("gateThreshold");
    const distractorThreshold = document.getElementById("distractorThreshold");
    const minArea = document.getElementById("minArea");
    const overlayOpacity = document.getElementById("overlayOpacity");
    const gateThresholdValue = document.getElementById("gateThresholdValue");
    const distractorThresholdValue = document.getElementById("distractorThresholdValue");
    const minAreaOutput = document.getElementById("minAreaValue");
    const overlayOpacityValue = document.getElementById("overlayOpacityValue");
    const gateTableBody = document.getElementById("gateTableBody");
    const obstacleTableBody = document.getElementById("obstacleTableBody");
    const selectedGateBody = document.getElementById("selectedGateBody");
    const gateCount = document.getElementById("gateCount");
    const obstacleCount = document.getElementById("obstacleCount");
    const frameJson = document.getElementById("frameJson");
    const renderStatus = document.getElementById("renderStatus");

    function formatNumber(value, decimals = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "null";
      }
      return Number(value).toFixed(decimals);
    }

    function formatPoint(point) {
      if (!Array.isArray(point)) {
        return "null";
      }
      return "[" + point.map((value) => formatNumber(value, 1)).join(", ") + "]";
    }

    function probabilityAt(probabilityMap, modelPoint) {
      if (!probabilityMap.length || !Array.isArray(modelPoint)) {
        return null;
      }
      const x = Math.round(modelPoint[0]);
      const y = Math.round(modelPoint[1]);
      if (x < 0 || y < 0 || x >= mapWidth || y >= mapHeight) {
        return null;
      }
      return probabilityMap[(y * mapWidth) + x];
    }

    function connectedComponents(probabilityMap, threshold, minAreaValue) {
      if (!probabilityMap.length) {
        return [];
      }
      const labels = new Int32Array(mapWidth * mapHeight);
      const queue = new Int32Array(mapWidth * mapHeight);
      const components = [];
      let label = 0;

      for (let start = 0; start < probabilityMap.length; start += 1) {
        if (labels[start] !== 0 || probabilityMap[start] < threshold) {
          continue;
        }
        label += 1;
        let head = 0;
        let tail = 0;
        queue[tail] = start;
        tail += 1;
        labels[start] = label;
        let area = 0;
        let sumX = 0;
        let sumY = 0;
        let sumProb = 0;
        let maxProb = 0;
        let minX = mapWidth;
        let minY = mapHeight;
        let maxX = 0;
        let maxY = 0;
        const pixels = [];

        while (head < tail) {
          const index = queue[head];
          head += 1;
          const x = index % mapWidth;
          const y = Math.floor(index / mapWidth);
          const prob = probabilityMap[index];
          pixels.push(index);
          area += 1;
          sumX += x;
          sumY += y;
          sumProb += prob;
          maxProb = Math.max(maxProb, prob);
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);

          for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
              if (dx === 0 && dy === 0) {
                continue;
              }
              const nx = x + dx;
              const ny = y + dy;
              if (nx < 0 || ny < 0 || nx >= mapWidth || ny >= mapHeight) {
                continue;
              }
              const next = (ny * mapWidth) + nx;
              if (labels[next] === 0 && probabilityMap[next] >= threshold) {
                labels[next] = label;
                queue[tail] = next;
                tail += 1;
              }
            }
          }
        }

        if (area >= minAreaValue) {
          const centerModel = [sumX / area, sumY / area];
          components.push({
            component_id: label,
            area_model_px: area,
            bbox_model_px: [minX, minY, (maxX - minX) + 1, (maxY - minY) + 1],
            bbox_px: [
              Math.round(minX * scaleX),
              Math.round(minY * scaleY),
              Math.round(((maxX - minX) + 1) * scaleX),
              Math.round(((maxY - minY) + 1) * scaleY)
            ],
            center_model_px: centerModel,
            center_px: [centerModel[0] * scaleX, centerModel[1] * scaleY],
            confidence: maxProb,
            mean_probability: sumProb / area,
            pixels
          });
        }
      }
      components.sort((a, b) => b.confidence - a.confidence);
      return components;
    }

    function peakPoint(probabilityMap) {
      if (!probabilityMap.length) {
        return null;
      }
      let bestIndex = 0;
      let bestValue = probabilityMap[0];
      for (let index = 1; index < probabilityMap.length; index += 1) {
        if (probabilityMap[index] > bestValue) {
          bestValue = probabilityMap[index];
          bestIndex = index;
        }
      }
      const x = bestIndex % mapWidth;
      const y = Math.floor(bestIndex / mapWidth);
      return {
        kind: "pose_center_logit_peak",
        model_px: [x, y],
        px: [(x + 0.5) * scaleX, (y + 0.5) * scaleY],
        confidence: bestValue,
        meaning: "strongest model-predicted target-center heatmap point in the image plane",
        not_a_label: true,
        not_3d_position: true
      };
    }

    function edgePointTowardTarget(component, targetModelPoint) {
      if (!component || !Array.isArray(targetModelPoint) || !component.pixels.length) {
        return null;
      }
      const cx = component.center_model_px[0];
      const cy = component.center_model_px[1];
      const dx = targetModelPoint[0] - cx;
      const dy = targetModelPoint[1] - cy;
      const length = Math.hypot(dx, dy);
      if (length <= 1e-6) {
        return null;
      }
      const ux = dx / length;
      const uy = dy / length;
      let bestPixel = component.pixels[0];
      let bestScore = -Infinity;
      for (const pixel of component.pixels) {
        const x = pixel % mapWidth;
        const y = Math.floor(pixel / mapWidth);
        const score = ((x - cx) * ux) + ((y - cy) * uy);
        if (score > bestScore) {
          bestScore = score;
          bestPixel = pixel;
        }
      }
      const bx = bestPixel % mapWidth;
      const by = Math.floor(bestPixel / mapWidth);
      return [(bx + 0.5) * scaleX, (by + 0.5) * scaleY];
    }

    function interpretedGates(components) {
      return components.map((component, index) => ({
        instance_id: index + 1,
        kind: "gate",
        model_component_id: component.component_id,
        bbox_px: component.bbox_px,
        center_px: component.center_px,
        center_model_px: component.center_model_px,
        confidence: component.confidence,
        center_heatmap_confidence: probabilityAt(centerMap, component.center_model_px),
        position_xyz_camera_m: null,
        orientation_normal_camera_ruf: null,
        orientation_plane_camera_6: null,
        pose_source: "not_available_in_inference_output",
        model_pose_status: "not_predicted_by_this_checkpoint"
      }));
    }

    function interpretedObstacles(components, targetPeak, safeZ) {
      const targetModelPoint = targetPeak ? targetPeak.model_px : null;
      return components.map((component, index) => ({
        instance_id: index + 1,
        kind: "avoidance_hint",
        model_component_id: component.component_id,
        bbox_px: component.bbox_px,
        center_px: component.center_px,
        edge_px_toward_next_gate: edgePointTowardTarget(component, targetModelPoint),
        safe_z_static: safeZ,
        confidence: component.confidence,
        position_source: "image_plane_only_no_depth"
      }));
    }

    function closestGateToCenterPeak(gates, targetPeak) {
      if (!gates.length || !targetPeak || !Array.isArray(targetPeak.px)) {
        return null;
      }
      let bestGate = null;
      let bestDistance = Infinity;
      for (const gate of gates) {
        if (!Array.isArray(gate.center_px)) {
          continue;
        }
        const distance = Math.hypot(
          gate.center_px[0] - targetPeak.px[0],
          gate.center_px[1] - targetPeak.px[1]
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          bestGate = gate;
        }
      }
      if (!bestGate) {
        return null;
      }
      return {
        instance_id: bestGate.instance_id,
        model_component_id: bestGate.model_component_id,
        center_px: bestGate.center_px,
        confidence: bestGate.confidence,
        center_peak_distance_px: bestDistance,
        selection_rule: "nearest_gate_component_center_to_pose_center_logit_peak_px",
        physical_closest_gate_status: "not_available_without_depth_or_pose_head"
      };
    }

    function publicComponent(component) {
      const copy = {...component};
      delete copy.pixels;
      return copy;
    }

    function drawProbabilityMap(probabilityMap, threshold, rgba, opacity) {
      if (!probabilityMap.length) {
        return;
      }
      ctx.save();
      for (let y = 0; y < mapHeight; y += 1) {
        for (let x = 0; x < mapWidth; x += 1) {
          const value = probabilityMap[(y * mapWidth) + x];
          if (value < threshold) {
            continue;
          }
          const alpha = Math.min(0.95, Math.max(0, value * opacity));
          ctx.fillStyle = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${alpha})`;
          ctx.fillRect(x * scaleX, y * scaleY, Math.ceil(scaleX), Math.ceil(scaleY));
        }
      }
      ctx.restore();
    }

    function drawComponents(components, color, prefix) {
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      components.forEach((component, index) => {
        const [x, y, w, h] = component.bbox_px;
        const [cx, cy] = component.center_px;
        ctx.strokeRect(x, y, w, h);
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillText(`${prefix}${index + 1} ${component.confidence.toFixed(2)}`, x + 3, Math.max(12, y - 4));
      });
      ctx.restore();
    }

    function drawCenterPeak(targetPeak) {
      if (!targetPeak) {
        return;
      }
      const [x, y] = targetPeak.px;
      ctx.save();
      ctx.strokeStyle = "rgb(0, 180, 255)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 14, y);
      ctx.lineTo(x + 14, y);
      ctx.moveTo(x, y - 14);
      ctx.lineTo(x, y + 14);
      ctx.stroke();
      ctx.fillStyle = "rgb(0, 105, 180)";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText(`center ${targetPeak.confidence.toFixed(2)}`, x + 12, y + 8);
      ctx.restore();
    }

    function drawSelectedGate(selectedGate) {
      if (!selectedGate || !Array.isArray(selectedGate.center_px)) {
        return;
      }
      const [cx, cy] = selectedGate.center_px;
      ctx.save();
      ctx.strokeStyle = "rgb(255, 245, 0)";
      ctx.fillStyle = "rgb(70, 55, 0)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 17, cy);
      ctx.lineTo(cx + 17, cy);
      ctx.moveTo(cx, cy - 17);
      ctx.lineTo(cx, cy + 17);
      ctx.stroke();
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText(`selected G${selectedGate.instance_id}`, cx + 13, cy - 10);
      ctx.restore();
    }

    function renderGateTable(gates) {
      if (!gates.length) {
        gateTableBody.innerHTML = '<tr><td colspan="5">No gate components crossed the current threshold.</td></tr>';
        return;
      }
      gateTableBody.innerHTML = gates.map((gate) => `
        <tr>
          <td>${gate.instance_id}</td>
          <td>${formatNumber(gate.confidence, 3)}</td>
          <td>${formatPoint(gate.center_px)}</td>
          <td>null</td>
          <td>null</td>
        </tr>
      `).join("");
    }

    function renderObstacleTable(obstacles) {
      if (!obstacles.length) {
        obstacleTableBody.innerHTML = '<tr><td colspan="4">No obstacle components crossed the current threshold.</td></tr>';
        return;
      }
      obstacleTableBody.innerHTML = obstacles.map((obstacle) => `
        <tr>
          <td>${obstacle.instance_id}</td>
          <td>${formatNumber(obstacle.confidence, 3)}</td>
          <td>${formatPoint(obstacle.edge_px_toward_next_gate)}</td>
          <td>${formatNumber(obstacle.safe_z_static, 2)}</td>
        </tr>
      `).join("");
    }

    function renderSelectedGate(targetPeak, selectedGate) {
      if (!targetPeak) {
        selectedGateBody.innerHTML = '<tr><td>No center heatmap is available from this checkpoint.</td></tr>';
        return;
      }
      const selectedRows = selectedGate
        ? `
          <tr><th>Selected gate</th><td>G${selectedGate.instance_id}</td></tr>
          <tr><th>Gate center px</th><td>${formatPoint(selectedGate.center_px)}</td></tr>
          <tr><th>Distance to marker</th><td>${formatNumber(selectedGate.center_peak_distance_px, 1)} px</td></tr>
          <tr><th>Gate confidence</th><td>${formatNumber(selectedGate.confidence, 3)}</td></tr>
          <tr><th>Selection rule</th><td>${selectedGate.selection_rule}</td></tr>
        `
        : '<tr><th>Selected gate</th><td>none at the current gate threshold</td></tr>';
      selectedGateBody.innerHTML = `
        <tr><th>Center peak px</th><td>${formatPoint(targetPeak.px)}</td></tr>
        <tr><th>Center peak confidence</th><td>${formatNumber(targetPeak.confidence, 3)}</td></tr>
        <tr><th>Marker meaning</th><td>maximum pose_center_logit heatmap value; image-plane target-center hint</td></tr>
        ${selectedRows}
        <tr><th>Physical closest gate</th><td>unavailable without depth or pose prediction</td></tr>
      `;
    }

    function buildCurrentJson(gateComponents, obstacleComponents, gates, obstacles, thresholds, targetPeak, selectedGate) {
      const current = {...reviewData};
      delete current.interactive_probability_maps;
      current.thresholds = thresholds;
      current.inference = {
        ...current.inference,
        center_peak_marker: targetPeak,
        selected_gate: selectedGate,
        gate_components: gateComponents.map(publicComponent),
        obstacle_components: obstacleComponents.map(publicComponent)
      };
      current.interpreted_instances = {gates, obstacles};
      current.review_interaction = {
        threshold_controls_are_browser_side: true,
        current_values_are_not_original_labels: true
      };
      return current;
    }

    function updateReview() {
      const gateValue = Number(gateThreshold.value);
      const obstacleValue = Number(distractorThreshold.value);
      const minAreaThreshold = Number(minArea.value);
      const opacityValue = Number(overlayOpacity.value);
      const safeZ = Number(reviewData.thresholds.safe_z_static || 2.5);
      gateThresholdValue.textContent = gateValue.toFixed(2);
      distractorThresholdValue.textContent = obstacleValue.toFixed(2);
      minAreaOutput.textContent = String(minAreaThreshold);
      overlayOpacityValue.textContent = opacityValue.toFixed(2);

      const gateComponents = connectedComponents(gateMap, gateValue, minAreaThreshold);
      const obstacleComponents = connectedComponents(obstacleMap, obstacleValue, minAreaThreshold);
      const targetPeak = peakPoint(centerMap);
      const gates = interpretedGates(gateComponents);
      const obstacles = interpretedObstacles(obstacleComponents, targetPeak, safeZ);
      const selectedGate = closestGateToCenterPeak(gates, targetPeak);
      gateCount.textContent = String(gates.length);
      obstacleCount.textContent = String(obstacles.length);
      renderSelectedGate(targetPeak, selectedGate);
      renderGateTable(gates);
      renderObstacleTable(obstacles);

      canvas.width = sourceWidth;
      canvas.height = sourceHeight;
      ctx.clearRect(0, 0, sourceWidth, sourceHeight);
      drawProbabilityMap(gateMap, gateValue, [0, 220, 110], opacityValue);
      drawProbabilityMap(obstacleMap, obstacleValue, [255, 88, 48], opacityValue);
      drawCenterPeak(targetPeak);
      drawComponents(gateComponents, "rgb(0, 190, 105)", "G");
      drawComponents(obstacleComponents, "rgb(220, 75, 43)", "O");
      drawSelectedGate(selectedGate);

      const thresholds = {
        gate_mask: gateValue,
        distractor_mask: obstacleValue,
        min_component_area_model_px: minAreaThreshold,
        safe_z_static: safeZ
      };
      frameJson.textContent = JSON.stringify(
        buildCurrentJson(gateComponents, obstacleComponents, gates, obstacles, thresholds, targetPeak, selectedGate),
        null,
        2
      );
      renderStatus.textContent = `Rendered ${gates.length} gates and ${obstacles.length} obstacles from inference thresholds. Selected gate: ${selectedGate ? `G${selectedGate.instance_id}` : "none"}.`;
    }

    [gateThreshold, distractorThreshold, minArea, overlayOpacity].forEach((element) => {
      element.addEventListener("input", updateReview);
    });
    if (sourceImage.complete) {
      updateReview();
    } else {
      sourceImage.addEventListener("load", updateReview);
    }
  </script>
</body>
</html>
"""
    replacements = {
        "__FRAME_ID__": html.escape(str(review_json["frame_id"])),
        "__CHECKPOINT_STEP__": str(int(review_json["model"]["checkpoint_step"])),
        "__SCHEMA_VERSION__": html.escape(review_json["model"]["schema_version"]),
        "__GATE_COUNT__": str(len(gates)),
        "__OBSTACLE_COUNT__": str(len(obstacles)),
        "__SOURCE_IMAGE__": html.escape(review_json["paths"]["source_image"]),
        "__GATE_THRESHOLD__": f"{float(review_json['thresholds']['gate_mask']):.2f}",
        "__DISTRACTOR_THRESHOLD__": f"{float(review_json['thresholds']['distractor_mask']):.2f}",
        "__MIN_AREA__": str(int(review_json["thresholds"]["min_component_area_model_px"])),
        "__LAYER_ITEMS__": layer_items,
        "__SOURCE_ROWS__": source_rows,
        "__EMBEDDED_JSON__": embedded_json,
        "__REVIEW_DATA__": review_data_json,
    }
    template = template.replace("{{", "{").replace("}}", "}")
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    summary = read_json(summary_path)
    label_jsonl = Path(summary["label_jsonl"]).expanduser().resolve()
    checkpoint_path = Path(summary["checkpoint"]).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    record = read_record(label_jsonl, int(args.frame_id))
    image_path = Path(record["image_path"]).expanduser().resolve()
    source = Image.open(image_path).convert("RGB")
    source_copy_path = output_dir / image_path.name
    source.save(source_copy_path)

    model, checkpoint = load_model(checkpoint_path)
    config = checkpoint.get("config", {})
    model_width = int(config.get("image_width") or 160)
    model_height = int(config.get("image_height") or 90)
    image_tensor = image_to_tensor(image_path, model_width, model_height)
    arrays = sigmoid_outputs(model, image_tensor)

    names = {
        "mask": list(config.get("mask_channels") or []),
        "center": list(config.get("center_channels") or []),
        "cross": list(config.get("cross_channels") or []),
    }
    scale_x = float(source.width) / float(model_width)
    scale_y = float(source.height) / float(model_height)
    layer_records = save_layer_images(output_dir, arrays, names)

    visible_index = names["mask"].index("visible_gate_mask") if "visible_gate_mask" in names["mask"] else 0
    distractor_index = names["mask"].index("distractor_mask") if "distractor_mask" in names["mask"] else None
    gate_components = connected_components(
        arrays["mask"][visible_index],
        threshold=float(args.gate_threshold),
        min_area=int(args.min_component_area),
        scale_x=scale_x,
        scale_y=scale_y,
    )
    obstacle_components: list[dict[str, Any]] = []
    distractor_probability = np.zeros_like(arrays["mask"][visible_index])
    if distractor_index is not None:
        distractor_probability = arrays["mask"][distractor_index]
        obstacle_components = connected_components(
            distractor_probability,
            threshold=float(args.distractor_threshold),
            min_area=int(args.min_component_area),
            scale_x=scale_x,
            scale_y=scale_y,
        )

    center_peak_model_xy: tuple[float, float] | None = None
    if "pose_center_logit" in names["center"]:
        center_index = names["center"].index("pose_center_logit")
        peak_x, peak_y, _ = peak_xy(arrays["center"][center_index])
        center_peak_model_xy = (float(peak_x), float(peak_y))

    center_probability: np.ndarray | None = None
    if "pose_center_logit" in names["center"]:
        center_probability = arrays["center"][names["center"].index("pose_center_logit")]
    interpreted_gate_rows = interpreted_gates(gate_components, center_probability)
    static_center_peak = center_peak_marker(center_probability, scale_x=scale_x, scale_y=scale_y)
    static_selected_gate = selected_gate_from_center_peak(interpreted_gate_rows, static_center_peak)
    interpreted_obstacle_rows = interpreted_obstacles(
        obstacle_components,
        distractor_probability,
        center_peak_model_xy=center_peak_model_xy,
        scale_x=scale_x,
        scale_y=scale_y,
        threshold=float(args.distractor_threshold),
        safe_z_static=float(args.safe_z_static),
    )

    overlay_path = output_dir / f"frame_{int(record['frame_id']):06d}_overlay.png"
    draw_overlay(source, arrays, names, gate_components, obstacle_components, overlay_path)

    fps = float(summary.get("fps") or 10.0)
    elapsed = max(0.0, (float(record["frame_id"]) - 1.0) / fps)
    probability_quantization_scale = 10000
    interactive_probability_maps: dict[str, Any] = {
        "width": model_width,
        "height": model_height,
        "source_width": source.width,
        "source_height": source.height,
        "quantization": f"uint16_probability_times_{probability_quantization_scale}",
        "quantization_scale": probability_quantization_scale,
        "maps": {
            "visible_gate_mask": quantized_probability_map(
                arrays["mask"][visible_index],
                probability_quantization_scale,
            ),
        },
    }
    if distractor_index is not None:
        interactive_probability_maps["maps"]["distractor_mask"] = quantized_probability_map(
            arrays["mask"][distractor_index],
            probability_quantization_scale,
        )
    if center_probability is not None:
        interactive_probability_maps["maps"]["pose_center_logit"] = quantized_probability_map(
            center_probability,
            probability_quantization_scale,
        )
    review_json: dict[str, Any] = {
        "schema_version": "frame_inference_interpretation_review_v1",
        "frame_id": image_path.name,
        "frame_number": int(record["frame_id"]),
        "elapsed_seconds_estimate": elapsed,
        "elapsed_seconds_source": "estimated_from_video_review_fps",
        "paths": {
            "source_image": source_copy_path.relative_to(output_dir).as_posix(),
            "overlay_image": overlay_path.relative_to(output_dir).as_posix(),
            "json_file": f"frame_{int(record['frame_id']):06d}_review.json",
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint.get("step", 0)),
            "schema_version": str(config.get("schema_version")),
            "image_width": model_width,
            "image_height": model_height,
            "output_stride": int(config.get("output_stride") or 4),
            "mask_channels": names["mask"],
            "center_channels": names["center"],
            "cross_channels": names["cross"],
        },
        "capabilities": {
            "predicted_by_checkpoint": ["visible_gate_mask", "distractor_mask", "pose_center_logit", "outer_corner_heatmap", "diagonal_cross_heatmap"],
            "position_xyz": "not_predicted_by_this_checkpoint",
            "orientation": "not_predicted_by_this_checkpoint",
            "pose_values_in_this_review": "not_populated",
        },
        "source_manifest_usage": {
            "manifest": str(label_jsonl),
            "fields_read": ["frame_id", "image_path"],
            "original_label_fields_used": [],
        },
        "thresholds": {
            "gate_mask": float(args.gate_threshold),
            "distractor_mask": float(args.distractor_threshold),
            "min_component_area_model_px": int(args.min_component_area),
            "safe_z_static": float(args.safe_z_static),
        },
        "interactive_probability_maps": interactive_probability_maps,
        "inference": {
            "layers": layer_records,
            "center_peak_marker": static_center_peak,
            "selected_gate": static_selected_gate,
            "gate_components": gate_components,
            "obstacle_components": obstacle_components,
        },
        "interpreted_instances": {
            "gates": interpreted_gate_rows,
            "obstacles": interpreted_obstacle_rows,
        },
        "label_metadata": None,
        "review_notes": [
            "This sample checkpoint can produce masks and heatmaps for a frame-level review.",
            "Original gate labels, target pose, and instance metadata are not used by this inference-only review.",
            "It cannot independently infer gate XYZ position or orientation; those fields remain null.",
            "The center peak marker is the maximum pose_center_logit heatmap point; selected_gate is the decoded gate nearest to that marker in image pixels.",
            "The physically closest gate is not available from this checkpoint because it has no depth or pose head.",
            "Obstacle output is an image-plane avoidance hint. No single-frame depth or true obstacle shape is inferred here.",
        ],
    }

    json_path = output_dir / review_json["paths"]["json_file"]
    write_json(json_path, review_json)
    html_path = output_dir / "index.html"
    html_path.write_text(build_html(output_dir, review_json), encoding="utf-8")
    manifest_path = output_dir / "review_manifest.json"
    write_json(
        manifest_path,
        {
            "index_html": str(html_path),
            "frame_json": str(json_path),
            "overlay": str(overlay_path),
            "source": str(source_copy_path),
        },
    )
    script_copy = output_dir / "build_frame_review.py"
    if script_copy.resolve() != Path(__file__).resolve():
        shutil.copy2(Path(__file__).resolve(), script_copy)

    print(f"html: {html_path}")
    print(f"json: {json_path}")
    print(f"overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
