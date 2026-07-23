"""Dataclass contracts for the vision pipeline."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar


FRAME_SCHEMA = "vision-frame.v1"
RUN_MANIFEST_SCHEMA = "vision-run-manifest.v1"
RUN_STATUS_SCHEMA = "vision-run-status.v1"
VISION_RESULTS_SCHEMA = "vision-results.v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {item.name: json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        return json_safe(value.tolist())
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tuple3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three numeric values")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain exactly three numeric values") from exc


def _optional_tuple3(value: Any, field_name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _tuple3(value, field_name)


class JsonDataclassMixin:
    _nested_lists: ClassVar[dict[str, type["JsonDataclassMixin"]]] = {}
    _nested_objects: ClassVar[dict[str, type["JsonDataclassMixin"]]] = {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for item in fields(self):
            if item.metadata.get("serialize") is False:
                continue
            payload[item.name] = json_safe(getattr(self, item.name))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {}
        for item in fields(cls):
            if not item.init or item.metadata.get("serialize") is False:
                continue
            if item.name not in payload:
                continue
            value = payload[item.name]
            if item.name in cls._nested_lists and isinstance(value, list):
                nested = cls._nested_lists[item.name]
                value = [nested.from_dict(entry) if isinstance(entry, dict) else entry for entry in value]
            elif item.name in cls._nested_objects and isinstance(value, dict):
                value = cls._nested_objects[item.name].from_dict(value)
            kwargs[item.name] = value
        return cls(**kwargs)


#
# final result
#
# External systems should import these two dataclasses first. 
#
@dataclass
class VisionGateResult(JsonDataclassMixin):
    gate_id: str = ""
    position_camera_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_confidence: float = 0.0
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position_camera_m = _tuple3(self.position_camera_m, "position_camera_m")
        self.orientation_camera = _optional_tuple3(self.orientation_camera, "orientation_camera")
        self.position_confidence = float(self.position_confidence)
        self.orientation_confidence = float(self.orientation_confidence)

    def to_controller_payload(self) -> dict[str, Any]:
        return {
            "id": self.gate_id,
            "position_xyz": [float(value) for value in self.position_camera_m],
            "position_confidence": float(self.position_confidence),
            "orientation_xyz": None
            if self.orientation_camera is None
            else [float(value) for value in self.orientation_camera],
            "orientation_confidence": float(self.orientation_confidence),
        }


@dataclass
class VisionResults(JsonDataclassMixin):
    schema: str = VISION_RESULTS_SCHEMA
    run_id: str = ""
    frame_ordinal: int = 0
    frame_id: str = ""
    source_path: str = ""
    image_width: int = 0
    image_height: int = 0
    created_at: str = ""
    timing_ms: dict[str, float] = field(default_factory=dict)
    coordinate_frame: str = "flight-camera-local"
    camera_position_camera_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_orientation_camera: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gates: list[VisionGateResult] = field(default_factory=list)
    obstacles: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"gates": VisionGateResult}

    def __post_init__(self) -> None:
        self.camera_position_camera_m = _tuple3(self.camera_position_camera_m, "camera_position_camera_m")
        self.camera_orientation_camera = _tuple3(self.camera_orientation_camera, "camera_orientation_camera")

    def to_controller_payload(self, *, output_dir: str = "memory") -> dict[str, Any]:
        return {
            "run": {
                "output_dir": output_dir,
                "cycle": int(self.frame_ordinal),
                "frame_id": self.frame_id,
                "sim_time_ns": 0,
            },
            "gates": [gate.to_controller_payload() for gate in self.gates],
            "obstacles": list(self.obstacles),
        }


@dataclass
class PipelinePreset(JsonDataclassMixin):
    schema: str
    image: dict[str, Any]
    camera: dict[str, Any]
    lut: dict[str, Any]
    maskLayers: list[dict[str, Any]]
    classes: list[dict[str, Any]]
    bbox: dict[str, Any]
    clipping: dict[str, Any]
    contours: dict[str, Any]
    pose: dict[str, Any]
    instanceTracking: dict[str, Any]
    flightBridge: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "PipelinePreset":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"preset JSON must contain an object: {path}")
        return cls.from_dict(payload)

    def enabled_bits(self) -> list[int]:
        return [int(layer["bit"]) for layer in self.maskLayers if bool(layer.get("enabled", True))]

    def layer_by_bit(self) -> dict[int, dict[str, Any]]:
        return {int(layer["bit"]): dict(layer) for layer in self.maskLayers}

    def class_by_prefix(self) -> dict[str, dict[str, Any]]:
        return {str(item["prefix"]): dict(item) for item in self.classes}


@dataclass
class FrameMeta(JsonDataclassMixin):
    run_id: str
    frame_ordinal: int
    frame_id: str
    source_path: str
    image_width: int
    image_height: int
    created_at: str
    timing_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameDataclass(JsonDataclassMixin):
    run_id: str
    frame_ordinal: int
    frame_id: str
    source_path: str
    image_width: int
    image_height: int
    created_at: str
    timing_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class SourceFrame(FrameDataclass):
    source_sha256: str = ""
    decoder: str = "PIL.Image.open(path).convert('RGB')"


@dataclass
class ColorMaskFrame(FrameDataclass):
    lut_path: str = ""
    lut_sha256: str = ""
    decoder: str = "PIL.Image.open(path).convert('RGB')"
    mask_width: int = 0
    mask_height: int = 0
    nonzero_pixel_count: int = 0
    layer_pixel_counts: dict[str, int] = field(default_factory=dict)
    bitfield_counts: dict[str, int] = field(default_factory=dict)
    mask_path: str | None = None
    mask_bits: Any = field(default=None, repr=False, compare=False, metadata={"serialize": False})


@dataclass
class BBoxObservation(JsonDataclassMixin):
    bbox_id: str
    component_label: int
    bbox_px: list[int]
    bbox_uv: list[float]
    center_px: list[float]
    center_uv: list[float]
    width_px: int
    height_px: int
    area_px: int
    pixel_count: int
    layer_counts: dict[str, int]
    dominant_prefix: str | None
    quality: dict[str, Any]
    quad_fit: dict[str, Any] = field(default_factory=dict)


@dataclass
class BBoxFrame(FrameDataclass):
    observations: list[BBoxObservation] = field(default_factory=list)
    bbox_count: int = 0
    selected_pixel_count: int = 0
    enabled_bits: list[int] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    component_labels: Any = field(default=None, repr=False, compare=False, metadata={"serialize": False})
    selected_mask: Any = field(default=None, repr=False, compare=False, metadata={"serialize": False})

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"observations": BBoxObservation}


@dataclass
class ClippingObservation(JsonDataclassMixin):
    bbox_id: str
    enabled: bool
    status: str
    severity: float
    sides: list[str]
    near_sides: list[str]
    bbox_touches: dict[str, bool]
    contact_pixels: dict[str, int]
    contact_ratio: dict[str, float]
    side_severity: dict[str, float]
    side_trust: dict[str, float]
    warn_only: bool


@dataclass
class ClippingFrame(FrameDataclass):
    observations: list[ClippingObservation] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"observations": ClippingObservation}


@dataclass
class ContourObservation(JsonDataclassMixin):
    bbox_id: str
    outer: dict[str, Any] | None
    voids: list[dict[str, Any]] = field(default_factory=list)
    additional_outers: list[dict[str, Any]] = field(default_factory=list)
    selected_pixel_count: int = 0
    contour_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    quad_points_px: list[list[float]] | None = None
    bbox_px: list[int] = field(default_factory=list)


@dataclass
class ContourFrame(FrameDataclass):
    observations: list[ContourObservation] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"observations": ContourObservation}


@dataclass
class PoseObservation(JsonDataclassMixin):
    bbox_id: str
    available: bool
    reason: str | None
    xyzCameraM: list[float] | None = None
    rpyCameraDeg: list[float] | None = None
    depthM: float | None = None
    fitQuality: dict[str, Any] = field(default_factory=dict)
    reprojectionErrorPx: float | None = None
    solvePnPMethod: str | None = None
    imagePointsPx: list[list[float]] | None = None
    objectPointsM: list[list[float]] | None = None


@dataclass
class PoseFrame(FrameDataclass):
    observations: list[PoseObservation] = field(default_factory=list)
    available_count: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"observations": PoseObservation}


@dataclass
class InstanceObservation(JsonDataclassMixin):
    observation_id: str
    instance_id: str
    bbox_id: str
    bbox: dict[str, Any]
    pose: dict[str, Any]
    association_score: float
    observationQuality: float
    tracking_status: str
    age_frames: int
    missed_frame_gap: int
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InstanceFrame(FrameDataclass):
    observations: list[InstanceObservation] = field(default_factory=list)
    instances: list[dict[str, Any]] = field(default_factory=list)
    active_track_count: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"observations": InstanceObservation}


@dataclass
class FlightGateObservation(VisionGateResult):
    """Backward-compatible stage name for a final gate result."""


@dataclass
class FlightObservation(FrameDataclass):
    gates: list[FlightGateObservation] = field(default_factory=list)
    obstacles: list[dict[str, Any]] = field(default_factory=list)
    source: str = "vision_instance_tracking"
    camera_position_camera_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    camera_orientation_camera: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    _nested_lists: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"gates": FlightGateObservation}

    def to_controller_payload(self, *, output_dir: str = "memory") -> dict[str, Any]:
        return {
            "run": {
                "output_dir": output_dir,
                "cycle": int(self.frame_ordinal),
                "frame_id": self.frame_id,
                "sim_time_ns": 0,
            },
            "gates": [gate.to_controller_payload() for gate in self.gates],
            "obstacles": list(self.obstacles),
        }


@dataclass
class StageDebugEnvelope(JsonDataclassMixin):
    schema: str
    run_id: str
    frame_ordinal: int
    frame_id: str
    stage: str
    created_at: str
    timing_ms: float
    payload: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineFrameResult(FrameDataclass):
    vision_results: VisionResults = field(default_factory=VisionResults)
    source: dict[str, Any] = field(default_factory=dict)
    color_masking: dict[str, Any] = field(default_factory=dict)
    bboxing: dict[str, Any] = field(default_factory=dict)
    clipping: dict[str, Any] = field(default_factory=dict)
    contouring: dict[str, Any] = field(default_factory=dict)
    pose_estimation: dict[str, Any] = field(default_factory=dict)
    instance_tracking: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    _nested_objects: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"vision_results": VisionResults}


@dataclass
class RunStatus(JsonDataclassMixin):
    schema: str
    run_id: str
    status: str
    run_root: str
    started_at: str
    updated_at: str
    frame_count: int = 0
    latest_frame: str | None = None
    completed_at: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class RunManifest(JsonDataclassMixin):
    schema: str
    run_id: str
    mode: str
    run_root: str
    source_dir: str | None
    single_frame: str | None
    output_root: str
    created_at: str
    completed_at: str | None
    debug: bool
    preset_path: str
    preset_sha256: str
    preset: dict[str, Any]
    stages: list[str]
    frame_count: int
    frame_index: list[dict[str, Any]]
    timings_ms: dict[str, Any]
    output_paths: dict[str, str]
