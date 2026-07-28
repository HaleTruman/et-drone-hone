"""Frame-level, multi-gate targets for detection and keypoint estimation."""

from __future__ import annotations

from dataclasses import dataclass
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
POSITION_NAMES = ("right", "up", "forward")
ORIENTATION_NAMES = ("yaw_deg", "pitch_deg", "roll_deg")
KEYPOINT_COUNT = 4


@dataclass(frozen=True)
class GateTarget:
    """One gate instance in a frame."""

    gate_label: str
    outer_corners: np.ndarray
    keypoint_visibility: np.ndarray
    bbox_xyxy: np.ndarray
    position: np.ndarray
    orientation_deg: np.ndarray

    @property
    def yaw_mod_180_deg(self) -> float:
        return float(wrap_yaw_180(self.orientation_deg[0]))


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
    gate_width_cm: float
    gate_height_cm: float

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
            gate_width_cm=self.gate_width_cm,
            gate_height_cm=self.gate_height_cm,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "focal_x_px": self.focal_x_px,
            "focal_y_px": self.focal_y_px,
            "principal_x_px": self.principal_x_px,
            "principal_y_px": self.principal_y_px,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "gate_width_cm": self.gate_width_cm,
            "gate_height_cm": self.gate_height_cm,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float | int]) -> "CameraCalibration":
        return cls(**values)


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


def fit_camera_calibration(frames: Sequence[GateFrame]) -> CameraCalibration:
    """Return capture calibration.

    The current Unreal capture has a 90-degree horizontal FOV and square
    pixels. Keeping this derivation frame-relative also supports resized
    copies and preserves the exact 640x360 capture calibration (fx=fy=320).
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
        gate_width_cm=270.0,
        gate_height_cm=270.0,
    )


def _frame_from_record(
    run_dir: Path,
    run_name: str,
    record: dict[str, object],
) -> tuple[GateFrame, list[dict[str, object]]]:
    frame_number = int(record["frame_number"])
    width = int(record["frame_width"])
    height = int(record["frame_height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame dimensions {width}x{height}")
    image_path = run_dir / "frames" / f"frame_{frame_number:06d}.png"
    if not image_path.is_file():
        raise ValueError(f"frame image is missing: {image_path}")

    gates: list[GateTarget] = []
    rejected: list[dict[str, object]] = []
    for gate in record.get("gates", ()):
        try:
            gates.append(_gate_from_metadata(gate, width, height))
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
) -> GateTarget:
    label = str(gate["label"])
    original_corners = np.asarray(
        [_pixel_xy(gate["corners"][name], name) for name in OUTER_CORNER_NAMES],
        dtype=np.float32,
    )
    corners = canonical_image_corners(original_corners)
    visibility = np.asarray(
        [
            2.0
            if 0.0 <= point[0] < width and 0.0 <= point[1] < height
            else 0.0
            for point in corners
        ],
        dtype=np.float32,
    )

    bbox_data = gate.get("bbox_2d")
    if not isinstance(bbox_data, dict):
        raise ValueError("gate has no bbox_2d")
    bbox = np.asarray(
        [
            np.clip(float(bbox_data["x_min"]), 0.0, width - 1.0),
            np.clip(float(bbox_data["y_min"]), 0.0, height - 1.0),
            np.clip(float(bbox_data["x_max"]), 0.0, width - 1.0),
            np.clip(float(bbox_data["y_max"]), 0.0, height - 1.0),
        ],
        dtype=np.float32,
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("bbox_2d does not define a positive-area in-frame box")

    position_data = gate["relative_position_camera_frame"]
    orientation_data = gate["relative_orientation_euler_deg"]
    position = np.asarray([position_data[name] for name in POSITION_NAMES], dtype=np.float32)
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
    )


def _pixel_xy(projection: dict[str, object], name: str) -> tuple[float, float]:
    pixel = projection.get("pixel")
    if not isinstance(pixel, dict):
        raise ValueError(f"{name} has no projectable pixel coordinate")
    return float(pixel["x"]), float(pixel["y"])


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
