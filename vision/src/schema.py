from __future__ import annotations

import json
import base64
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LUT_PATH = "assets/color_lut_v1.npz"
DEFAULT_FILL_RADIUS_PX = 0
DEFAULT_BRIDGE_MAX_PX = 25
DEFAULT_MIN_MASK_REGION_PX = 150
DEFAULT_MIN_VOID_PX = 75
DEFAULT_MIN_VOID_DISTANCE_PX = 2.0
DEFAULT_MIN_ORIGINAL_COVERAGE_DEG = 270.0


@dataclass
class MaskFrame:
    frame_path: str
    width: int
    height: int
    base_mask: np.ndarray


@dataclass
class FillFrame:
    frame: MaskFrame
    yellow_mask: np.ndarray
    filled_mask: np.ndarray
    fill_radius_px: float = DEFAULT_FILL_RADIUS_PX


@dataclass
class BridgeFrame:
    fill: FillFrame
    green_mask: np.ndarray
    final_mask: np.ndarray
    bridge_max_px: int = DEFAULT_BRIDGE_MAX_PX
    min_mask_region_px: int = DEFAULT_MIN_MASK_REGION_PX


@dataclass
class MotionCompensation:
    predicted_center: tuple[float, float]
    source_frame_path: str
    compensated_distance_px: float
    translation_scale: float
    tolerance_px: float


@dataclass
class TrackMatch:
    iou: float | None
    raw_distance_px: float
    area_agreement: float
    lifetime_area_agreement: float | None
    score: float
    streak: int
    motion_compensation: MotionCompensation | None = None


@dataclass
class MotionForecast:
    confidence: str
    history: list[tuple[float, float, float]]
    forecast: list[tuple[float, float]]


@dataclass
class TrailPoint:
    frame_path: str
    predicted_center: tuple[float, float]


@dataclass
class MotionTrail:
    points: list[TrailPoint]


@dataclass
class VoidRecord:
    void_id: str
    pixel_count: int
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    distance_px: float
    mask_clipping: bool = False
    track_id: str = ""
    reliable: bool = False
    track_match: TrackMatch | None = None
    forecast: MotionForecast | None = None
    motion_trail: MotionTrail | None = None
    possible_reconnection: str | None = None
    reconnection_error_px: float | None = None
    region_track_id: str | None = None


@dataclass
class RegionRecord:
    region_id: str
    pixel_count: int
    bbox: tuple[int, int, int, int]
    mask_clipping: bool = False
    voids: list[VoidRecord] = field(default_factory=list)
    track_id: str = ""
    reliable: bool = False
    track_match: TrackMatch | None = None
    motion_trail: MotionTrail | None = None
    forecast: MotionForecast | None = None


@dataclass
class FinalEstimate:
    track_id: str
    center: tuple[float, float]
    source: str
    confidence: float
    missing: int


@dataclass
class FinalVoidEstimate:
    final_void_id: int
    center: tuple[float, float]
    source: str
    confidence: float
    clipped: bool
    track_id: str


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    image: Any | None = None
    saved_path: str | None = None


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class VehicleState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3


@dataclass(frozen=True)
class VisionGateObservation:
    gate_id: str
    position_camera_m: tuple[float, float, float]
    position_confidence: float
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionObservation:
    frame_id: int
    sim_time_ns: int
    gates: list[VisionGateObservation]
    source: str = "vision"
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoidAnalysis:
    bridge: BridgeFrame
    regions: list[RegionRecord]
    min_void_px: int = DEFAULT_MIN_VOID_PX
    final_estimates: list[FinalEstimate] = field(default_factory=list)
    final_void_estimates: list[FinalVoidEstimate] = field(default_factory=list)
    vision_observation: VisionObservation | None = None


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def mask_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def pack_bits(mask: np.ndarray) -> np.ndarray:
    return np.packbits(as_bool_mask(mask).reshape(-1).astype(np.uint8), bitorder="little")


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def frame_stem(frame_path: str) -> str:
    return Path(frame_path).stem


def artifact_path(out_dir: Path, frame_path: str) -> Path:
    return out_dir / f"{frame_stem(frame_path)}.json"


def bitplane_json(mask: np.ndarray) -> dict:
    packed = pack_bits(mask)
    return {
        "encoding": "base64-packed-lsb0",
        "byte_length": int(packed.size),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _serialize(value):
    """The only path from a dataclass to JSON: walk every field, by name, in
    declared order. No key is ever renamed, dropped, or invented here — add a
    field to any dataclass above and it appears in the output automatically.
    This is the standard: schema.py's JSON is always a structural mirror of
    its dataclasses, never a hand-picked subset."""
    if isinstance(value, np.ndarray):
        return bitplane_json(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def save_review_artifacts(analysis: VoidAnalysis, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = analysis.bridge.fill.frame.frame_path
    json_path = artifact_path(out_dir, frame_path)
    stale_npz_path = out_dir / f"{frame_stem(frame_path)}.npz"
    if stale_npz_path.exists():
        stale_npz_path.unlink()

    meta = {"version": "mask-review.v2", "analysis": _serialize(analysis)}
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"json": str(json_path), "metadata": meta}
