"""
Desired behavior:
- Provide one reusable validator for all WebSocket protocol envelopes.
- Enforce the v1 required field contract with explicit, human-readable errors.
- Allow strict and warn-only modes so rollout can start permissive and tighten safely.

Interfaces:
- Imported by `UE_Tooling/WebSocket/ws_bridge.py` and test utilities.
- `validate_message(...)` returns structured errors/warnings.
- `ensure_valid_message(...)` raises immediately in strict paths.

Assumptions:
- Envelope schema follows `UE_Tooling/WebSocket/protocol.py`.
- Unknown extra fields are allowed; required fields and basic types are enforced.
- v1 OBS carries image bytes in `payload.image_bytes_b64`.

Success conditions:
- Missing/invalid required fields are surfaced deterministically.
- `SET_CONFIG` v1 canonical field list is enforced in strict mode.
- Validation can run in both strict and log-first modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from UE_Tooling.WebSocket import protocol

SCRIPT_NAME = "schemas"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-13"

SET_CONFIG_REQUIRED_PATHS = [
    "type",
    "schema_version",
    "run_id",
    "seq",
    "timestamp",
    "payload.config_id",
    "payload.config_hash",
    "payload.movement.max_speed_cmps",
    "payload.movement.max_accel_cmps2",
    "payload.movement.max_yaw_degps",
    "payload.movement.max_pitch_degps",
    "payload.movement.max_roll_degps",
    "payload.movement.damping",
    "payload.sensor_rig.active_viewpoint",
    "payload.sensor_rig.front_fov_deg",
    "payload.sensor_rig.capture_width",
    "payload.sensor_rig.capture_height",
    "payload.sensor_rig.front_offset_cm.x",
    "payload.sensor_rig.front_offset_cm.y",
    "payload.sensor_rig.front_offset_cm.z",
    "payload.sensor_rig.front_rotation_deg.pitch",
    "payload.sensor_rig.front_rotation_deg.roll",
    "payload.sensor_rig.front_rotation_deg.yaw",
    "payload.telemetry.include_all_visible_meshes",
    "payload.telemetry.distance_units",
    "payload.telemetry.distance_precision_decimals",
    "payload.telemetry.center_distance_mode",
    "payload.telemetry.edge_distance_mode",
    "payload.telemetry.line_of_sight_filtering",
]

PER_TYPE_REQUIRED_PATHS = {
    protocol.TYPE_SET_CONFIG: SET_CONFIG_REQUIRED_PATHS,
    protocol.TYPE_ACK: ["type", "run_id", "schema_version", "timestamp", "payload"],
    protocol.TYPE_ACK_RECEIVED: ["type", "run_id", "schema_version", "timestamp", "payload"],
    protocol.TYPE_CONFIG_READY: [
        "type",
        "run_id",
        "schema_version",
        "timestamp",
        "payload.config_id",
        "payload.config_hash",
    ],
    protocol.TYPE_ACK_APPLIED: [
        "type",
        "run_id",
        "schema_version",
        "timestamp",
        "payload.config_id",
        "payload.config_hash",
    ],
    protocol.TYPE_SPAWN_DRONES: ["type", "run_id", "schema_version", "timestamp", "payload"],
    protocol.TYPE_CMD: ["type", "run_id", "schema_version", "timestamp", "drone_id", "payload"],
    protocol.TYPE_CAPTURE_NOW: ["type", "run_id", "schema_version", "timestamp", "drone_id", "capture_id", "payload"],
    protocol.TYPE_OBS: [
        "type",
        "run_id",
        "schema_version",
        "timestamp",
        "drone_id",
        "capture_id",
        "payload",
        "payload.config_ref.config_id",
        "payload.config_ref.config_hash",
        "payload.viewpoint.fov_deg",
        "payload.viewpoint.width",
        "payload.viewpoint.height",
    ],
    protocol.TYPE_STATUS: ["type", "run_id", "schema_version", "timestamp", "payload"],
    protocol.TYPE_ERROR: ["type", "run_id", "schema_version", "timestamp", "payload"],
}


class SchemaValidationError(ValueError):
    """Raised when strict validation fails."""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


_MISSING = object()


def _read_path(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, dict):
            return _MISSING
        if part not in value:
            return _MISSING
        value = value.get(part)
    return value


def _enforce_required_paths(message: dict[str, Any], paths: list[str], errors: list[str]) -> None:
    for path in paths:
        value = _read_path(message, path)
        if value is _MISSING:
            errors.append(f"Missing required field: {path}")
            continue
        if value is None:
            errors.append(f"Required field is null: {path}")
            continue
        if isinstance(value, str) and not value.strip():
            errors.append(f"Required field is empty string: {path}")


def _validate_obs_message(message: dict[str, Any], errors: list[str]) -> None:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return

    image_block = payload.get(protocol.IMAGE_BLOCK_KEY)
    canonical_image_value = image_block.get(protocol.IMAGE_BYTES_FIELD) if isinstance(image_block, dict) else _MISSING
    legacy_image_value = payload.get(protocol.IMAGE_PAYLOAD_KEY, _MISSING)
    if canonical_image_value is _MISSING and legacy_image_value is _MISSING:
        errors.append(
            "OBS requires image bytes at payload.image.bytes_b64 or payload.image_bytes_b64"
        )
    else:
        for label, value in (
            (f"payload.{protocol.IMAGE_BLOCK_KEY}.{protocol.IMAGE_BYTES_FIELD}", canonical_image_value),
            (f"payload.{protocol.IMAGE_PAYLOAD_KEY}", legacy_image_value),
        ):
            if value is _MISSING:
                continue
            if not isinstance(value, str):
                errors.append(f"{label} must be string")

    viewpoint = payload.get("viewpoint")
    if isinstance(viewpoint, dict):
        width = viewpoint.get("width")
        height = viewpoint.get("height")
        fov_deg = viewpoint.get("fov_deg")
        if not isinstance(width, int) or width <= 0:
            errors.append("payload.viewpoint.width must be positive int")
        if not isinstance(height, int) or height <= 0:
            errors.append("payload.viewpoint.height must be positive int")
        if not isinstance(fov_deg, (int, float)) or float(fov_deg) <= 0:
            errors.append("payload.viewpoint.fov_deg must be positive number")

    config_ref = payload.get("config_ref")
    if isinstance(config_ref, dict):
        for field_name in ("config_id", "config_hash"):
            value = config_ref.get(field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"payload.config_ref.{field_name} must be non-empty string")


def validate_message(message: dict[str, Any], strict: bool = True) -> ValidationResult:
    """Validate one protocol message in strict or warn-only mode."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(message, dict):
        return ValidationResult(valid=False, errors=[f"Message must be dict, got {type(message).__name__}"], warnings=[])

    message_type_raw = message.get("type")
    try:
        message_type = protocol.normalize_type(message_type_raw)
    except Exception:
        message_type = str(message_type_raw or "")
        errors.append(f"Invalid message type: {message_type_raw!r}")

    run_id = message.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("Missing/invalid run_id")

    schema_version = message.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        errors.append("Missing/invalid schema_version")
    elif schema_version != protocol.SCHEMA_VERSION_V1:
        warnings.append(
            f"Schema version '{schema_version}' differs from supported '{protocol.SCHEMA_VERSION_V1}'"
        )

    required_paths = PER_TYPE_REQUIRED_PATHS.get(message_type)
    if required_paths is None:
        warnings.append(f"Unknown message type '{message_type}'")
    else:
        _enforce_required_paths(message, required_paths, errors)

    if message_type in protocol.ACTION_TYPES:
        if "drone_id" not in message and message_type != protocol.TYPE_SPAWN_DRONES:
            errors.append(f"{message_type} requires drone_id")

    if message_type == protocol.TYPE_OBS:
        _validate_obs_message(message, errors)

    valid = len(errors) == 0
    if not strict and errors:
        warnings.extend(f"[demoted] {item}" for item in errors)
        errors = []
        valid = True

    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def ensure_valid_message(message: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """Validate and raise on failure in strict mode."""
    result = validate_message(message, strict=strict)
    if not result.valid:
        raise SchemaValidationError("; ".join(result.errors))
    return message
