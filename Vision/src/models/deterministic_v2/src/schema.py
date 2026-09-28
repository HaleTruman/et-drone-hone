"""Public and minimal internal dataclass contracts for projection."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar


VISION_RESULTS_SCHEMA = "vision-results.v1"
PROJECTION_FRAME_SCHEMA = "projection-frame.v1"
RUN_MANIFEST_SCHEMA = "projection-run-manifest.v1"
RUN_STATUS_SCHEMA = "projection-run-status.v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {item.name: json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
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


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
        handle.write("\n")
    temp.replace(path)


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
        return {
            item.name: json_safe(getattr(self, item.name))
            for item in fields(self)
            if item.metadata.get("serialize") is not False
        }

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
# Final result.
#
# External systems should import this dataclass. It is intentionally kept at
# the top of the schema module and stable across projection runtime changes.
#
@dataclass
class VisionGateResult(JsonDataclassMixin):
    gate_id: str
    position_camera_m: tuple[float, float, float]
    position_confidence: float
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gate_id = str(self.gate_id)
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
class FrameMeta(JsonDataclassMixin):
    run_id: str
    frame_ordinal: int
    frame_id: str
    source_path: str
    image_width: int = 0
    image_height: int = 0
    created_at: str = field(default_factory=utc_now)
    timing_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ProjectionFrameResult(JsonDataclassMixin):
    schema: str
    run_id: str
    frame_ordinal: int
    frame_id: str
    source_path: str
    image_width: int
    image_height: int
    created_at: str
    vision_results: VisionResults
    stage_counts: dict[str, int] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    _nested_objects: ClassVar[dict[str, type[JsonDataclassMixin]]] = {"vision_results": VisionResults}


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
    stages: list[str]
    frame_count: int
    frame_index: list[dict[str, Any]]
    timings_ms: dict[str, Any]
    output_paths: dict[str, str]
