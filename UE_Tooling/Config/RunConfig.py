"""
Desired behavior:
- Compile YAML config inputs into one deterministic runtime `SET_CONFIG` envelope.
- Keep YAML-to-JSON mapping centralized to avoid config drift across tooling scripts.
- Guarantee required v1 fields are present, with explicit defaults only where fallback behavior is still allowed.

Interfaces:
- Used by `UE_Tooling/WebSocket/ws_bridge.py` to obtain canonical startup config payloads.
- Reads config files in `UE_Tooling/Config/Data_Interface_Config` and `UE_Tooling/Config/Drone_Controller_Config`.
- Exposes `compile_set_config(...)` and a CLI for ad-hoc payload generation.

Assumptions:
- Movement and telemetry YAML may still be partial during scaffold stage and default where documented.
- Runtime image-capture YAML is treated as required input and should fail fast when malformed or incomplete.
- Unreal runtime does not parse YAML; it consumes only compiled JSON envelope.
- Canonical schema version is `1.0`.

Success conditions:
- Output envelope always includes required v1 `SET_CONFIG` fields and traceability fields.
- `config_hash` is deterministic for identical semantic payloads.
- `config_id` is derived from the canonical hash for stable run auditing.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from UE_Tooling.WebSocket import protocol

SCRIPT_NAME = "RunConfig"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-13"

DEFAULT_MOVEMENT = {
    "max_speed_cmps": 1400.0,
    "max_accel_cmps2": 600.0,
    "max_yaw_degps": 90.0,
    "max_pitch_degps": 60.0,
    "max_roll_degps": 60.0,
    "damping": 0.12,
}

DEFAULT_SENSOR_RIG = {
    "active_viewpoint": "front",
    "front_fov_deg": 90.0,
    "capture_width": 1280,
    "capture_height": 720,
    "front_offset_cm": {"x": 80.0, "y": 0.0, "z": 0.0},
    "front_rotation_deg": {"pitch": 0.0, "roll": 0.0, "yaw": 0.0},
}

DEFAULT_TELEMETRY = {
    "include_all_visible_meshes": True,
    "distance_units": "cm",
    "distance_precision_decimals": 3,
    "center_distance_mode": "pivot_origin",
    "edge_distance_mode": "nearest_triangle_edge",
    "line_of_sight_filtering": False,
}


@dataclass(frozen=True)
class RunConfigPaths:
    pose_yaml: Path
    viewpoint_yaml: Path
    telemetry_yaml: Path
    movement_yaml: Path


@dataclass(frozen=True)
class CompiledConfig:
    envelope: dict[str, Any]
    assumptions: list[str]
    source_paths: dict[str, str]


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Failed to locate repo root (.git not found)")


def _default_paths() -> RunConfigPaths:
    config_root = Path(__file__).resolve().parent
    return RunConfigPaths(
        pose_yaml=config_root / "Data_Interface_Config/SensorRigProfileConfig_Pose.yaml",
        viewpoint_yaml=config_root / "Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml",
        telemetry_yaml=config_root / "Data_Interface_Config/SensorRigProfileConfig_Telemetry.yaml",
        movement_yaml=config_root / "Drone_Controller_Config/Config_DroneMovementTuning.yaml",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML object in {path}, got {type(raw).__name__}")
    return raw


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now_utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Missing/invalid required object: {path}")
    return value


def _require_positive_float(value: Any, path: str) -> float:
    parsed = _as_float(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"Missing/invalid required positive float: {path}")
    return parsed


def _require_float(value: Any, path: str) -> float:
    parsed = _as_float(value)
    if parsed is None:
        raise ValueError(f"Missing/invalid required float: {path}")
    return parsed


def _require_positive_int(value: Any, path: str) -> int:
    parsed = _as_int(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"Missing/invalid required positive int: {path}")
    return parsed


def compile_set_config(
    *,
    run_id: str,
    seq: int = 1,
    timestamp: str | None = None,
    schema_version: str = protocol.SCHEMA_VERSION_V1,
    paths: RunConfigPaths | None = None,
) -> CompiledConfig:
    """Compile YAML inputs into deterministic `SET_CONFIG` envelope."""
    selected_paths = paths or _default_paths()
    pose_yaml = _load_yaml(selected_paths.pose_yaml)
    viewpoint_yaml = _load_yaml(selected_paths.viewpoint_yaml)
    telemetry_yaml = _load_yaml(selected_paths.telemetry_yaml)
    movement_yaml = _load_yaml(selected_paths.movement_yaml)

    assumptions: list[str] = []

    movement_block = dict(DEFAULT_MOVEMENT)
    movement_input = movement_yaml.get("movement_tuning", {}) if isinstance(movement_yaml.get("movement_tuning"), dict) else {}
    movement_mappings = {
        "max_speed": "max_speed_cmps",
        "max_accel": "max_accel_cmps2",
        "max_yaw_rate": "max_yaw_degps",
        "max_pitch_rate": "max_pitch_degps",
        "max_roll_rate": "max_roll_degps",
        "damping": "damping",
    }
    for source_key, target_key in movement_mappings.items():
        parsed = _as_float(movement_input.get(source_key))
        if parsed is None:
            assumptions.append(f"movement.{target_key} defaulted to {movement_block[target_key]}")
            continue
        movement_block[target_key] = parsed

    sensor_block = deepcopy(DEFAULT_SENSOR_RIG)
    viewpoint_input = viewpoint_yaml.get("viewpoints", {}) if isinstance(viewpoint_yaml.get("viewpoints"), dict) else {}
    active_viewpoints = viewpoint_input.get("active")
    if isinstance(active_viewpoints, list) and active_viewpoints:
        sensor_block["active_viewpoint"] = str(active_viewpoints[0])
    else:
        assumptions.append("sensor_rig.active_viewpoint defaulted to 'front'")

    resolution_input = _require_object(viewpoint_input.get("resolution"), "viewpoints.resolution")
    capture_rig_input = _require_object(viewpoint_input.get("capture_rig"), "viewpoints.capture_rig")
    viewpoint_offset_input = _require_object(
        capture_rig_input.get("offset_cm"),
        "viewpoints.capture_rig.offset_cm",
    )
    viewpoint_rotation_input = _require_object(
        capture_rig_input.get("rotation_deg"),
        "viewpoints.capture_rig.rotation_deg",
    )

    sensor_block["front_fov_deg"] = _require_positive_float(
        resolution_input.get("fov_deg"),
        "viewpoints.resolution.fov_deg",
    )
    sensor_block["capture_width"] = _require_positive_int(
        resolution_input.get("width"),
        "viewpoints.resolution.width",
    )
    sensor_block["capture_height"] = _require_positive_int(
        resolution_input.get("height"),
        "viewpoints.resolution.height",
    )

    for axis in ("x", "y", "z"):
        sensor_block["front_offset_cm"][axis] = _require_float(
            viewpoint_offset_input.get(axis),
            f"viewpoints.capture_rig.offset_cm.{axis}",
        )

    for axis in ("pitch", "roll", "yaw"):
        sensor_block["front_rotation_deg"][axis] = _require_float(
            viewpoint_rotation_input.get(axis),
            f"viewpoints.capture_rig.rotation_deg.{axis}",
        )

    telemetry_block = dict(DEFAULT_TELEMETRY)
    telemetry_input = telemetry_yaml.get("telemetry", {}) if isinstance(telemetry_yaml.get("telemetry"), dict) else {}
    include_all = telemetry_input.get("include_all_visible_meshes")
    if isinstance(include_all, bool):
        telemetry_block["include_all_visible_meshes"] = include_all
    else:
        assumptions.append(
            "telemetry.include_all_visible_meshes defaulted to True (all visible meshes)"
        )

    precision = _as_int(telemetry_input.get("distance_precision_decimals"))
    if precision is not None and precision >= 0:
        telemetry_block["distance_precision_decimals"] = precision
    else:
        assumptions.append(
            f"telemetry.distance_precision_decimals defaulted to {telemetry_block['distance_precision_decimals']}"
        )

    distance_units = telemetry_input.get("distance_units")
    if isinstance(distance_units, str) and distance_units.strip():
        telemetry_block["distance_units"] = distance_units.strip()
    else:
        assumptions.append("telemetry.distance_units defaulted to 'cm'")

    for key in ["center_distance_mode", "edge_distance_mode"]:
        override = telemetry_input.get(key)
        if isinstance(override, str) and override.strip():
            telemetry_block[key] = override.strip()
        else:
            assumptions.append(f"telemetry.{key} defaulted to {telemetry_block[key]!r}")

    los_override = telemetry_input.get("line_of_sight_filtering")
    if isinstance(los_override, bool):
        telemetry_block["line_of_sight_filtering"] = los_override
    else:
        assumptions.append("telemetry.line_of_sight_filtering defaulted to False")

    config_payload_no_hash = {
        "movement": movement_block,
        "sensor_rig": sensor_block,
        "telemetry": telemetry_block,
    }
    config_hash = _canonical_hash(config_payload_no_hash)
    config_id = f"cfg_{config_hash[:12]}"
    config_payload = {
        "config_id": config_id,
        "config_hash": config_hash,
        **config_payload_no_hash,
    }

    envelope = protocol.make_set_config(
        run_id=run_id,
        payload=config_payload,
        seq=seq,
        timestamp=timestamp or _now_utc_timestamp(),
    )
    envelope["schema_version"] = schema_version

    return CompiledConfig(
        envelope=envelope,
        assumptions=sorted(set(assumptions)),
        source_paths={
            "pose_yaml": str(selected_paths.pose_yaml),
            "viewpoint_yaml": str(selected_paths.viewpoint_yaml),
            "telemetry_yaml": str(selected_paths.telemetry_yaml),
            "movement_yaml": str(selected_paths.movement_yaml),
        },
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--schema-version", default=protocol.SCHEMA_VERSION_V1)
    parser.add_argument("--output", default="")
    parser.add_argument("--show-assumptions", action="store_true")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    compiled = compile_set_config(
        run_id=str(args.run_id),
        seq=int(args.seq),
        schema_version=str(args.schema_version),
    )

    output_payload = {
        "script": {"name": SCRIPT_NAME, "version": SCRIPT_VERSION, "date": SCRIPT_DATE},
        "source_paths": compiled.source_paths,
        "assumptions": compiled.assumptions,
        "set_config": compiled.envelope,
    }
    text = json.dumps(output_payload, indent=2, sort_keys=True)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[{SCRIPT_NAME}] wrote {out_path}")
    else:
        print(text)

    if args.show_assumptions and compiled.assumptions:
        print(f"[{SCRIPT_NAME}] assumptions:")
        for assumption in compiled.assumptions:
            print(f"  - {assumption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
