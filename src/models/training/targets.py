"""Frame-level, multi-gate targets for detection and keypoint estimation."""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


OUTER_CORNER_NAMES = (
    "Corner_outer_TL",
    "Corner_outer_TR",
    "Corner_outer_BL",
    "Corner_outer_BR",
)
INNER_CORNER_NAMES = (
    "Corner_inner_TL",
    "Corner_inner_TR",
    "Corner_inner_BL",
    "Corner_inner_BR",
)
POSITION_NAMES = ("right", "up", "forward")
ORIENTATION_NAMES = ("yaw_deg", "pitch_deg", "roll_deg")
OUTER_KEYPOINT_COUNT = 4
KEYPOINT_COUNT = 8
LABEL_KEYPOINT_COUNT = 8


@dataclass(frozen=True)
class GateTarget:
    """One gate instance in a frame."""

    gate_label: str
    outer_corners: np.ndarray
    keypoint_visibility: np.ndarray
    bbox_xyxy: np.ndarray
    position: np.ndarray
    orientation_deg: np.ndarray
    inner_corners: np.ndarray | None = None
    all_corners: np.ndarray | None = None
    all_keypoint_visibility: np.ndarray | None = None
    is_target: bool = False

    @property
    def yaw_mod_180_deg(self) -> float:
        return float(wrap_yaw_180(self.orientation_deg[0]))

    def scaled_pixels(self, scale_x: float, scale_y: float) -> "GateTarget":
        corners = self.outer_corners.copy()
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y
        bbox = self.bbox_xyxy.copy()
        bbox[[0, 2]] *= scale_x
        bbox[[1, 3]] *= scale_y
        inner_corners = _scale_optional_points(self.inner_corners, scale_x, scale_y)
        all_corners = _scale_optional_points(self.all_corners, scale_x, scale_y)
        return dataclasses.replace(
            self,
            outer_corners=corners.astype(np.float32),
            bbox_xyxy=bbox.astype(np.float32),
            inner_corners=inner_corners,
            all_corners=all_corners,
        )


@dataclass(frozen=True)
class GateFrame:
    """One image and all usable gates visible in it."""

    image_path: Path
    run_name: str
    frame_number: int
    width: int
    height: int
    gates: tuple[GateTarget, ...]

    @property
    def sample_id(self) -> str:
        return f"{self.run_name}/frame_{self.frame_number:06d}"

    def scaled_pixels(self, width: int, height: int) -> "GateFrame":
        if width == self.width and height == self.height:
            return self
        scale_x = width / self.width
        scale_y = height / self.height
        return dataclasses.replace(
            self,
            width=width,
            height=height,
            gates=tuple(gate.scaled_pixels(scale_x, scale_y) for gate in self.gates),
        )


# Kept as an import-compatible name for downstream code.
GateSample = GateFrame


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole calibration and known gate dimensions used by the pose solver."""

    focal_x_px: float
    focal_y_px: float
    principal_x_px: float
    principal_y_px: float
    frame_width: int
    frame_height: int
    gate_width_m: float
    gate_height_m: float

    def scaled_to(self, width: int, height: int) -> "CameraCalibration":
        scale_x = width / self.frame_width
        scale_y = height / self.frame_height
        return CameraCalibration(
            focal_x_px=self.focal_x_px * scale_x,
            focal_y_px=self.focal_y_px * scale_y,
            principal_x_px=self.principal_x_px * scale_x,
            principal_y_px=self.principal_y_px * scale_y,
            frame_width=width,
            frame_height=height,
            gate_width_m=self.gate_width_m,
            gate_height_m=self.gate_height_m,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "focal_x_px": self.focal_x_px,
            "focal_y_px": self.focal_y_px,
            "principal_x_px": self.principal_x_px,
            "principal_y_px": self.principal_y_px,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "gate_width_m": self.gate_width_m,
            "gate_height_m": self.gate_height_m,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float | int]) -> "CameraCalibration":
        normalized = dict(values)
        if "gate_width_m" not in normalized and "gate_width_cm" in normalized:
            normalized["gate_width_m"] = float(normalized.pop("gate_width_cm")) / 100.0
        if "gate_height_m" not in normalized and "gate_height_cm" in normalized:
            normalized["gate_height_m"] = float(normalized.pop("gate_height_cm")) / 100.0
        return cls(**normalized)


def wrap_yaw_180(value: float | np.ndarray) -> float | np.ndarray:
    """Map directed yaw to the side-invariant interval [-90, 90)."""

    return (np.asarray(value) + 90.0) % 180.0 - 90.0


def canonical_image_corners(corners: np.ndarray) -> np.ndarray:
    """Order a quadrilateral as image TL, TR, BL, BR.

    Input uses the actor's TL, TR, BL, BR order. Rear views reverse only the
    horizontal pairs, so comparing the two projected side midpoints removes
    actor-front semantics without fragile sorting under strong perspective.
    """

    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    actor_left_x = float(corners[[0, 2], 0].mean())
    actor_right_x = float(corners[[1, 3], 0].mean())
    if actor_left_x <= actor_right_x:
        return corners.copy()
    return corners[[1, 0, 3, 2]].copy()


def _scale_optional_points(
    points: np.ndarray | None,
    scale_x: float,
    scale_y: float,
) -> np.ndarray | None:
    if points is None:
        return None
    scaled = points.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled.astype(np.float32)


def load_gate_samples(
    data_root: Path | str,
    run_names: Iterable[str] | None = None,
) -> tuple[list[GateFrame], list[dict[str, object]]]:
    """Load frames with every usable visible gate and explicit rejection details."""

    data_root = Path(data_root).resolve()
    if run_names is None:
        run_dirs = discover_run_dirs(data_root)
    else:
        run_dirs = [resolve_run_dir(data_root, run_name) for run_name in run_names]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {data_root}")

    frames: list[GateFrame] = []
    rejected: list[dict[str, object]] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    run_name = run_display_name(data_root, run_dir)
                    frame, gate_rejections = _frame_from_record(run_dir, run_name, record)
                    frames.append(frame)
                    for rejection in gate_rejections:
                        rejected.append(
                            {
                                "run_name": run_name,
                                "line_number": line_number,
                                "frame_number": record.get("frame_number"),
                                **rejection,
                            }
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    rejected.append(
                        {
                            "run_name": run_name,
                            "line_number": line_number,
                            "frame_number": record.get("frame_number"),
                            "reason": str(exc),
                        }
                    )
    return frames, rejected


def scale_frames_for_training(
    frames: Sequence[GateFrame],
    width: int,
    height: int,
) -> list[GateFrame]:
    return [frame.scaled_pixels(width, height) for frame in frames]


def fit_camera_calibration(frames: Sequence[GateFrame]) -> CameraCalibration:
    """Return capture calibration.

    The current Unreal capture has a 90-degree horizontal FOV and square
    pixels. Keeping this derivation frame-relative also supports resized
    copies and preserves the capture calibration convention (fx=fy=width/2).
    """

    if not frames:
        raise ValueError("Cannot fit camera calibration without frames.")
    dimensions = {(frame.width, frame.height) for frame in frames}
    if len(dimensions) != 1:
        raise ValueError(f"Mixed frame dimensions are not supported: {dimensions}")
    width, height = dimensions.pop()
    focal = width * 0.5
    return CameraCalibration(
        focal_x_px=focal,
        focal_y_px=focal,
        principal_x_px=width * 0.5,
        principal_y_px=height * 0.5,
        frame_width=width,
        frame_height=height,
        gate_width_m=2.7,
        gate_height_m=2.7,
    )


def _frame_from_record(
    run_dir: Path,
    run_name: str,
    record: dict[str, object],
) -> tuple[GateFrame, list[dict[str, object]]]:
    frame_number = int(record["frame_number"])
    width = int(record.get("frame_width_px", record.get("frame_width")))
    height = int(record.get("frame_height_px", record.get("frame_height")))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame dimensions {width}x{height}")
    image_path = run_dir / "frames" / f"frame_{frame_number:06d}.png"
    if not image_path.is_file():
        raise ValueError(f"frame image is missing: {image_path}")

    gates: list[GateTarget] = []
    rejected: list[dict[str, object]] = []
    target_gate = str(record.get("target_gate", ""))
    for gate in record.get("gates", ()):
        try:
            gates.append(_gate_from_metadata(gate, width, height, target_gate))
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "gate_label": gate.get("label") if isinstance(gate, dict) else None,
                    "reason": str(exc),
                }
            )
    return (
        GateFrame(
            image_path=image_path,
            run_name=run_name,
            frame_number=frame_number,
            width=width,
            height=height,
            gates=tuple(gates),
        ),
        rejected,
    )


def _gate_from_metadata(
    gate: dict[str, object],
    width: int,
    height: int,
    target_gate: str = "",
) -> GateTarget:
    if isinstance(gate.get("labels"), dict):
        return _gate_from_clean_labels(gate, width, height, target_gate)

    label = str(gate["label"])
    corner_data = gate["corners"]
    original_corners, original_visibility = _legacy_corner_points(
        corner_data,
        OUTER_CORNER_NAMES,
        width,
        height,
    )
    corners = canonical_image_corners(original_corners)
    original_inner_corners, original_inner_visibility = _legacy_corner_points(
        corner_data,
        INNER_CORNER_NAMES,
        width,
        height,
        required=False,
    )
    if original_corners[[0, 2], 0].mean() <= original_corners[[1, 3], 0].mean():
        outer_order = [0, 1, 2, 3]
    else:
        outer_order = [1, 0, 3, 2]
    visibility = original_visibility[outer_order]
    inner_corners = original_inner_corners[outer_order]
    inner_visibility = original_inner_visibility[outer_order]

    bbox_data = gate.get("bbox_2d_px", gate.get("bbox_2d"))
    if not isinstance(bbox_data, dict):
        raise ValueError("gate has no bbox_2d_px")
    bbox = _clipped_bbox_from_mapping(bbox_data, width, height)

    if "relative_position_camera_frame_m" in gate:
        position_data = gate["relative_position_camera_frame_m"]
        position_scale = 1.0
    else:
        position_data = gate["relative_position_camera_frame"]
        position_scale = 0.01
    orientation_data = gate["relative_orientation_euler_deg"]
    position = (
        np.asarray([position_data[name] for name in POSITION_NAMES], dtype=np.float32)
        * position_scale
    )
    orientation = np.asarray(
        [orientation_data[name] for name in ORIENTATION_NAMES],
        dtype=np.float32,
    )
    directed_yaw = float(orientation[0])
    side_invariant_yaw = float(wrap_yaw_180(directed_yaw))
    if abs(directed_yaw - side_invariant_yaw) > 90.0:
        orientation[1:] *= -1.0
    orientation[0] = side_invariant_yaw
    if not all(np.isfinite(value).all() for value in (corners, bbox, position, orientation)):
        raise ValueError("gate target contains non-finite values")
    return GateTarget(
        gate_label=label,
        outer_corners=corners,
        keypoint_visibility=visibility,
        bbox_xyxy=bbox,
        position=position,
        orientation_deg=orientation,
        inner_corners=inner_corners,
        all_corners=np.concatenate((corners, inner_corners), axis=0),
        all_keypoint_visibility=np.concatenate((visibility, inner_visibility), axis=0),
        is_target=bool(gate.get("is_target", False) or gate.get("label") == target_gate),
    )


def _gate_from_clean_labels(
    gate: dict[str, object],
    width: int,
    height: int,
    target_gate: str = "",
) -> GateTarget:
    labels = gate["labels"]
    if not isinstance(labels, dict):
        raise ValueError("gate labels block is not an object")
    label = str(labels.get("label", gate.get("label", "")))
    keypoints_2d = np.asarray(labels["keypoints_2d"], dtype=np.float32).reshape(-1, 2)
    if keypoints_2d.shape[0] < OUTER_KEYPOINT_COUNT:
        raise ValueError("labels.keypoints_2d must contain at least four corners")
    if keypoints_2d.shape[0] < LABEL_KEYPOINT_COUNT:
        keypoints_2d = np.concatenate(
            (
                keypoints_2d,
                np.full(
                    (LABEL_KEYPOINT_COUNT - keypoints_2d.shape[0], 2),
                    np.nan,
                    dtype=np.float32,
                ),
            ),
            axis=0,
        )
    visibility = np.asarray(
        labels.get("keypoint_visibility", np.ones((len(keypoints_2d),), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if len(visibility) < len(keypoints_2d):
        visibility = np.pad(visibility, (0, len(keypoints_2d) - len(visibility)))
    visibility = np.where(visibility > 0.0, 2.0, 0.0).astype(np.float32)
    bbox_data = labels.get("bbox_xyxy", labels.get("bbox_xyxy_px", gate.get("bbox_2d_px")))
    bbox = (
        _clipped_bbox_from_sequence(bbox_data, width, height)
        if isinstance(bbox_data, (list, tuple))
        else _clipped_bbox_from_mapping(bbox_data, width, height)
        if isinstance(bbox_data, dict)
        else _bbox_from_points(keypoints_2d[visibility > 0.0], width, height)
    )
    keypoints_2d = _replace_nonfinite_points(keypoints_2d, bbox)
    position_data = labels.get("t_cam_m", labels.get("relative_position_camera_frame_m"))
    if position_data is None:
        raise ValueError("labels block has no t_cam_m")
    position = _position_array(position_data, scale=1.0)
    orientation = _orientation_array(labels)
    outer = canonical_image_corners(keypoints_2d[:OUTER_KEYPOINT_COUNT])
    if keypoints_2d[:OUTER_KEYPOINT_COUNT][[0, 2], 0].mean() <= keypoints_2d[:OUTER_KEYPOINT_COUNT][[1, 3], 0].mean():
        order = [0, 1, 2, 3]
    else:
        order = [1, 0, 3, 2]
    outer_visibility = visibility[:OUTER_KEYPOINT_COUNT][order]
    inner = keypoints_2d[OUTER_KEYPOINT_COUNT:LABEL_KEYPOINT_COUNT][order]
    inner_visibility = visibility[OUTER_KEYPOINT_COUNT:LABEL_KEYPOINT_COUNT][order]
    return GateTarget(
        gate_label=label,
        outer_corners=outer.astype(np.float32),
        keypoint_visibility=outer_visibility.astype(np.float32),
        bbox_xyxy=bbox,
        position=position,
        orientation_deg=orientation,
        inner_corners=inner.astype(np.float32),
        all_corners=np.concatenate((outer, inner), axis=0).astype(np.float32),
        all_keypoint_visibility=np.concatenate((outer_visibility, inner_visibility), axis=0),
        is_target=bool(
            labels.get("is_target", gate.get("is_target", False))
            or label == target_gate
        ),
    )


def _legacy_corner_points(
    corners: dict[str, object],
    names: Sequence[str],
    width: int,
    height: int,
    required: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    visibility: list[float] = []
    for name in names:
        try:
            projection = _corner_projection(corners, name)
        except KeyError:
            if required:
                raise
            points.append((np.nan, np.nan))
            visibility.append(0.0)
            continue
        pixel = _pixel_xy(projection)
        visible = _is_visible_corner(projection, pixel, width, height)
        points.append(pixel if pixel is not None else (np.nan, np.nan))
        visibility.append(2.0 if visible else 0.0)
    point_array = _replace_nonfinite_points(
        np.asarray(points, dtype=np.float32),
        fallback_bbox=None,
    )
    return point_array, np.asarray(visibility, dtype=np.float32)


def _pixel_xy(projection: dict[str, object]) -> tuple[float, float] | None:
    pixel = projection.get("pixel_px", projection.get("pixel"))
    if not isinstance(pixel, dict):
        return None
    return float(pixel["x"]), float(pixel["y"])


def _is_visible_corner(
    projection: dict[str, object],
    pixel: tuple[float, float] | None,
    width: int,
    height: int,
) -> bool:
    if pixel is None:
        return False
    x_value, y_value = pixel
    inside_frame = bool(
        projection.get(
            "inside_frame",
            0.0 <= x_value < width and 0.0 <= y_value < height,
        )
    )
    visibility = projection.get("visibility")
    line_of_sight_clear = True
    if isinstance(visibility, dict):
        line_of_sight = visibility.get("line_of_sight")
        if isinstance(line_of_sight, dict):
            line_of_sight_clear = not bool(line_of_sight.get("occluded", False))
        if "visible" in visibility:
            return bool(visibility["visible"]) and inside_frame and line_of_sight_clear
    return inside_frame and line_of_sight_clear


def _corner_projection(corners: dict[str, object], name: str) -> dict[str, object]:
    if name in corners:
        value = corners[name]
        if isinstance(value, dict):
            return value
    parts = name.split("_")
    if len(parts) >= 3:
        ring = parts[1].lower()
        location = parts[2].lower()
        ring_corners = corners.get(ring)
        if isinstance(ring_corners, dict):
            value = ring_corners.get(location)
            if isinstance(value, dict):
                return value
    raise KeyError(f"missing corner projection {name}")


def _position_array(value: object, scale: float) -> np.ndarray:
    if isinstance(value, dict):
        return np.asarray([value[name] for name in POSITION_NAMES], dtype=np.float32) * scale
    return np.asarray(value, dtype=np.float32).reshape(3) * scale


def _orientation_array(labels: dict[str, object]) -> np.ndarray:
    orientation_data = labels.get(
        "relative_orientation_euler_deg",
        labels.get("orientation_euler_deg"),
    )
    if isinstance(orientation_data, dict):
        orientation = np.asarray(
            [
                orientation_data.get("yaw_mod_180_deg", orientation_data.get("yaw_deg")),
                orientation_data["pitch_deg"],
                orientation_data["roll_deg"],
            ],
            dtype=np.float32,
        )
        orientation[0] = float(wrap_yaw_180(orientation[0]))
        return orientation
    if orientation_data is not None:
        orientation = np.asarray(orientation_data, dtype=np.float32).reshape(3)
        orientation[0] = float(wrap_yaw_180(orientation[0]))
        return orientation
    quaternion = labels.get("q_cam")
    if quaternion is not None:
        return _orientation_from_quaternion(np.asarray(quaternion, dtype=np.float64).reshape(4))
    raise ValueError("labels block has no orientation")


def _orientation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w_value, x_value, y_value, z_value = quaternion
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        raise ValueError("q_cam has zero length")
    w_value, x_value, y_value, z_value = quaternion / norm
    rotation = np.asarray(
        [
            [
                1 - 2 * (y_value * y_value + z_value * z_value),
                2 * (x_value * y_value - z_value * w_value),
                2 * (x_value * z_value + y_value * w_value),
            ],
            [
                2 * (x_value * y_value + z_value * w_value),
                1 - 2 * (x_value * x_value + z_value * z_value),
                2 * (y_value * z_value - x_value * w_value),
            ],
            [
                2 * (x_value * z_value - y_value * w_value),
                2 * (y_value * z_value + x_value * w_value),
                1 - 2 * (x_value * x_value + y_value * y_value),
            ],
        ],
        dtype=np.float64,
    )
    normal = rotation[:, 2]
    if normal[2] < 0.0:
        normal = -normal
    right, up, forward = normal[0], -normal[1], normal[2]
    yaw = float(wrap_yaw_180(np.degrees(np.arctan2(right, forward))))
    pitch = float(np.degrees(np.arctan2(up, np.sqrt(right * right + forward * forward))))
    return np.asarray([yaw, pitch, 0.0], dtype=np.float32)


def _clipped_bbox_from_sequence(value: object, width: int, height: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32).reshape(4)
    return _clip_bbox(values, width, height)


def _clipped_bbox_from_mapping(value: object, width: int, height: int) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError("gate has no bbox_2d_px")
    bbox = np.asarray(
        [
            float(value.get("x_min_px", value.get("x_min"))),
            float(value.get("y_min_px", value.get("y_min"))),
            float(value.get("x_max_px", value.get("x_max"))),
            float(value.get("y_max_px", value.get("y_max"))),
        ],
        dtype=np.float32,
    )
    return _clip_bbox(bbox, width, height)


def _bbox_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    finite = np.asarray(points, dtype=np.float32)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("cannot derive bbox without finite visible keypoints")
    bbox = np.asarray(
        [
            finite[:, 0].min(),
            finite[:, 1].min(),
            finite[:, 0].max(),
            finite[:, 1].max(),
        ],
        dtype=np.float32,
    )
    return _clip_bbox(bbox, width, height)


def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = bbox.astype(np.float32).copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0.0, width - 1.0)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0.0, height - 1.0)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox_2d does not define a positive-area in-frame box")
    return clipped


def _replace_nonfinite_points(
    points: np.ndarray,
    fallback_bbox: np.ndarray | None,
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    if np.isfinite(result).all():
        return result
    if fallback_bbox is not None:
        fallback = np.asarray(
            [
                (float(fallback_bbox[0]) + float(fallback_bbox[2])) * 0.5,
                (float(fallback_bbox[1]) + float(fallback_bbox[3])) * 0.5,
            ],
            dtype=np.float32,
        )
    else:
        finite = result[np.isfinite(result).all(axis=1)]
        fallback = (
            finite.mean(axis=0).astype(np.float32)
            if len(finite)
            else np.zeros((2,), dtype=np.float32)
        )
    result[~np.isfinite(result).all(axis=1)] = fallback
    return result


def discover_run_dirs(data_root: Path) -> list[Path]:
    nested = sorted(
        path for path in data_root.glob("dataset_*/runs/run_*") if path.is_dir()
    )
    if nested:
        return nested
    return sorted(path for path in data_root.glob("run_*") if path.is_dir())


def resolve_run_dir(data_root: Path, run_name: str) -> Path:
    normalized = run_name.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) == 2:
        return data_root / parts[0] / "runs" / parts[1]
    if len(parts) == 3 and parts[1] == "runs":
        return data_root / parts[0] / "runs" / parts[2]

    flat = data_root / run_name
    if flat.is_dir():
        return flat

    matches = [
        path for path in data_root.glob(f"dataset_*/runs/{run_name}") if path.is_dir()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Run name '{run_name}' is ambiguous under {data_root}; use dataset_###/{run_name}."
        )
    return flat


def run_display_name(data_root: Path, run_dir: Path) -> str:
    relative = run_dir.resolve().relative_to(data_root)
    parts = relative.parts
    if len(parts) == 3 and parts[1] == "runs":
        return f"{parts[0]}/{parts[2]}"
    return relative.as_posix()
