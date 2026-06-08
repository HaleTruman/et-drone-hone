"""RGB visual review for regressor and landmarker extractor output."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.src.io.udp_protocol import frame_id_from_path, sorted_jpeg_paths
from vision.src.regressor.logit_inference import CAMERA_CX_PX, CAMERA_CY_PX, CAMERA_FX_PX, CAMERA_FY_PX
from vision.src.regressor.logit_inference import SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH


@dataclass(frozen=True)
class ReviewRenderConfig:
    frames_dir: Path
    regressor_jsonl: Path
    controller_jsonl: Path
    output_mp4: Path
    fps: float = 30.0
    max_frames: int = 0


@dataclass(frozen=True)
class ReviewRenderStats:
    frames_rendered: int
    elapsed_seconds: float
    output_mp4: Path


def frame_name(frame_id: int) -> str:
    return f"frame_{int(frame_id):06d}"


def load_jsonl_by_frame_id(path: Path) -> dict[str, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSONL not found: {resolved}")
    frames: dict[str, dict[str, Any]] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row is not an object: {resolved}")
            run = payload.get("run")
            if not isinstance(run, dict):
                raise ValueError(f"JSONL row is missing run object: {resolved}")
            name = str(run.get("frame_id") or frame_name(int(run.get("cycle", 0))))
            frames[name] = payload
    return frames


def project_camera_xyz_to_source_px(
    position_xyz: list[float] | tuple[float, float, float],
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float] | None:
    if len(position_xyz) != 3:
        return None
    x, y, z = [float(value) for value in position_xyz]
    if z <= 1.0e-6 or not all(math.isfinite(value) for value in (x, y, z)):
        return None
    sx = float(source_width) / float(SOURCE_IMAGE_WIDTH)
    sy = float(source_height) / float(SOURCE_IMAGE_HEIGHT)
    fx = CAMERA_FX_PX * sx
    fy = CAMERA_FY_PX * sy
    cx = CAMERA_CX_PX * sx
    cy = CAMERA_CY_PX * sy
    return ((x / z) * fx) + cx, cy - ((y / z) * fy)


def _project_axis_segment(
    position_xyz: list[float],
    orientation_xyz: list[float] | None,
    *,
    source_width: int,
    source_height: int,
    scale_m: float = 2.0,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if orientation_xyz is None or len(orientation_xyz) != 3:
        return None
    axis = np.asarray([float(value) for value in orientation_xyz], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-6 or not math.isfinite(norm):
        return None
    axis = axis / norm
    center = np.asarray([float(value) for value in position_xyz], dtype=np.float32)
    endpoints = [center - (axis * float(scale_m)), center + (axis * float(scale_m))]
    projected = [
        project_camera_xyz_to_source_px(endpoint.tolist(), source_width=source_width, source_height=source_height)
        for endpoint in endpoints
    ]
    if projected[0] is None or projected[1] is None:
        return None
    return (
        (int(round(projected[0][0])), int(round(projected[0][1]))),
        (int(round(projected[1][0])), int(round(projected[1][1]))),
    )


def _draw_header(image: np.ndarray, title: str, subtitle: str) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 52), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    cv2.putText(image, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)


def _draw_marker(image: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], index: int) -> None:
    x, y = point
    cv2.circle(image, (x, y), 7, color, 2, cv2.LINE_AA)
    cv2.line(image, (x - 10, y), (x + 10, y), color, 2, cv2.LINE_AA)
    cv2.line(image, (x, y - 10), (x, y + 10), color, 2, cv2.LINE_AA)
    cv2.putText(image, f"{index}", (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_gate_overlay(
    image: np.ndarray,
    payload: dict[str, Any] | None,
    *,
    title: str,
    subtitle: str,
    marker_color: tuple[int, int, int],
    axis_color: tuple[int, int, int],
    nearest_label: str | None = None,
    nearest_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    rendered = image.copy()
    _draw_header(rendered, title, subtitle)
    gates = payload.get("gates", []) if isinstance(payload, dict) else []
    if not isinstance(gates, list):
        gates = []
    source_height, source_width = rendered.shape[:2]
    text_y = 72
    for index, gate in enumerate(gate for gate in gates if isinstance(gate, dict)):
        position = gate.get("position_xyz")
        if not isinstance(position, list) or len(position) != 3:
            continue
        projected = project_camera_xyz_to_source_px(position, source_width=source_width, source_height=source_height)
        if projected is None:
            continue
        px = (int(round(projected[0])), int(round(projected[1])))
        orientation = gate.get("orientation_xyz")
        current_marker_color = nearest_color if nearest_color is not None and index == 0 else marker_color
        if isinstance(orientation, list):
            segment = _project_axis_segment(
                position,
                orientation,
                source_width=source_width,
                source_height=source_height,
            )
            if segment is not None:
                cv2.line(rendered, segment[0], segment[1], axis_color, 2, cv2.LINE_AA)
        _draw_marker(rendered, px, current_marker_color, index + 1)
        confidence = float(gate.get("position_confidence", 0.0))
        prefix = f"{nearest_label} " if nearest_label and index == 0 else ""
        label = (
            f"{prefix}{index + 1}:{gate.get('id', '')} "
            f"xyz({float(position[0]):.1f},{float(position[1]):.1f},{float(position[2]):.1f}) "
            f"c{confidence:.2f}"
        )
        label_x = max(8, min(source_width - 300, px[0] + 12))
        label_y = max(64, min(source_height - 12, px[1] + 18))
        cv2.putText(rendered, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rendered, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, current_marker_color, 1, cv2.LINE_AA)
        if text_y < source_height - 8:
            cv2.putText(rendered, label, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(rendered, label, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, current_marker_color, 1, cv2.LINE_AA)
            text_y += 16
    return rendered


def render_gate_overlay_image(
    *,
    rgb_path: Path,
    payload: dict[str, Any] | None,
    output_path: Path,
    title: str,
    subtitle: str,
    marker_color: tuple[int, int, int] = (90, 245, 125),
    axis_color: tuple[int, int, int] = (215, 80, 240),
    nearest_label: str | None = "TARGET",
    nearest_color: tuple[int, int, int] = (40, 255, 255),
) -> Path:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read RGB frame: {rgb_path}")
    rendered = _draw_gate_overlay(
        image,
        payload,
        title=title,
        subtitle=subtitle,
        marker_color=marker_color,
        axis_color=axis_color,
        nearest_label=nearest_label,
        nearest_color=nearest_color,
    )
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(resolved_output_path), rendered):
        raise RuntimeError(f"Unable to write overlay image: {resolved_output_path}")
    return resolved_output_path


def render_two_pane_review(config: ReviewRenderConfig) -> ReviewRenderStats:
    frame_paths = sorted_jpeg_paths(Path(config.frames_dir).expanduser())
    if int(config.max_frames) > 0:
        frame_paths = frame_paths[: int(config.max_frames)]
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png frames found in {config.frames_dir}")

    regressor_frames = load_jsonl_by_frame_id(config.regressor_jsonl)
    controller_frames = load_jsonl_by_frame_id(config.controller_jsonl)

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Unable to read RGB frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    output_path = Path(config.output_mp4).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.fps),
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open MP4 writer: {output_path}")

    started = time.perf_counter()
    frames_rendered = 0
    try:
        for path in frame_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to read RGB frame: {path}")
            frame_id = frame_id_from_path(path)
            name = frame_name(frame_id)
            left = _draw_gate_overlay(
                image,
                regressor_frames.get(name),
                title=f"{name} | Regressor RGB Overlay",
                subtitle="all gate predictions from regressor JSON",
                marker_color=(255, 210, 45),
                axis_color=(255, 120, 30),
            )
            right = _draw_gate_overlay(
                image,
                controller_frames.get(name),
                title=f"{name} | Landmarker Current Top 5",
                subtitle="controller egress, nearest matched current-frame XYZ only",
                marker_color=(90, 245, 125),
                axis_color=(215, 80, 240),
            )
            writer.write(np.hstack([left, right]))
            frames_rendered += 1
    finally:
        writer.release()

    return ReviewRenderStats(
        frames_rendered=frames_rendered,
        elapsed_seconds=time.perf_counter() - started,
        output_mp4=output_path,
    )


__all__ = [
    "ReviewRenderConfig",
    "ReviewRenderStats",
    "frame_name",
    "load_jsonl_by_frame_id",
    "project_camera_xyz_to_source_px",
    "render_gate_overlay_image",
    "render_two_pane_review",
]
