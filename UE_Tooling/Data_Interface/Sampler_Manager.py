"""
Use case:
- Sole tooling-side owner of periodic sample triggering.
- Read sampling cadence from the image-capture YAML and emit minimal `CAPTURE_NOW`
  actions over an already-running bridge.
- Keep ownership narrow: sampling owns capture timing, capture IDs, image-sample
  validation, accepted image persistence, and sample-status logging only.

Interfaces:
- Reads `UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml`
  for tooling-side capture cadence.
- Uses `UE_Tooling/WebSocket/protocol.py` to build minimal `CAPTURE_NOW` envelopes.
- Runs against an existing `UE_Tooling/WebSocket/ws_bridge.py` server instance that
  already owns transport and config-ready state.

Assumptions:
- Bridge startup and `SET_CONFIG` application are owned elsewhere.
- `CAPTURE_NOW` stays minimal: `run_id`, `drone_id`, `capture_id`, envelope timestamp,
  and an empty payload object.
- Image behavior comes from already-applied runtime config, not from per-capture overrides.
- Raw bridge persistence is optional; accepted sample persistence is always owned here.

Success conditions:
- Sampling cadence can be loaded from YAML with a default of `15.0` seconds.
- The manager waits for bridge connection and config-ready state, then emits one capture
  every configured interval.
- Each returned image is validated against active config identity and expected viewpoint
  settings before it is accepted.
- Accepted images are written deterministically under the run artifact root.
- Each capture attempt logs exactly one terminal sample status: `received` or `failed`.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UE_Tooling.WebSocket import protocol
from UE_Tooling.WebSocket.ws_bridge import WebSocketBridgeServer

SCRIPT_NAME = "Sampler_Manager"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-14"

DEFAULT_CAPTURE_INTERVAL_SECONDS = 15.0
DEFAULT_OBSERVATION_TIMEOUT_SECONDS = 15.0
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class SamplerManagerConfig:
    run_id: str
    drone_id: str
    capture_interval_seconds: float
    observation_timeout_seconds: float
    connection_timeout_seconds: float
    viewpoint_yaml_path: Path


@dataclass(frozen=True)
class ActiveImageConfig:
    config_id: str
    config_hash: str
    width: int
    height: int
    fov_deg: float


@dataclass(frozen=True)
class ValidatedImageSample:
    run_id: str
    drone_id: str
    capture_id: str
    seq: str
    timestamp_utc: str
    config_id: str
    config_hash: str
    viewpoint_width: int
    viewpoint_height: int
    viewpoint_fov_deg: float
    actual_width: int
    actual_height: int
    image_bytes: bytes


class AcceptedImageStore:
    """Persist sampler-validated image samples for one run."""

    def __init__(self, run_root: Path) -> None:
        self.root = run_root / "samples"
        self.images_dir = self.root / "images"
        self.accepted_samples_jsonl = self.root / "accepted_samples.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._accepted_count = 0

    def persist(
        self,
        sample: ValidatedImageSample,
        *,
        capture_interval_seconds: float,
        raw_image_file: str = "",
    ) -> str:
        self._accepted_count += 1
        capture_dir = self.images_dir / sample.capture_id
        capture_dir.mkdir(parents=True, exist_ok=True)
        interval_token = _format_interval_token(capture_interval_seconds)
        image_filename = (
            f"{sample.drone_id}__{sample.seq}__"
            f"{sample.viewpoint_width}-{sample.viewpoint_height}__{interval_token}.png"
        )
        image_path = capture_dir / image_filename
        image_path.write_bytes(sample.image_bytes)
        relative_image_path = image_path.relative_to(self.root).as_posix()
        record = {
            "accepted_index": self._accepted_count,
            "timestamp_utc": sample.timestamp_utc,
            "run_id": sample.run_id,
            "drone_id": sample.drone_id,
            "capture_id": sample.capture_id,
            "seq": sample.seq,
            "config_ref": {
                "config_id": sample.config_id,
                "config_hash": sample.config_hash,
            },
            "viewpoint": {
                "width": sample.viewpoint_width,
                "height": sample.viewpoint_height,
                "fov_deg": sample.viewpoint_fov_deg,
            },
            "image": {
                "width": sample.actual_width,
                "height": sample.actual_height,
                "sample_file": relative_image_path,
                "raw_image_file": raw_image_file,
            },
        }
        with self.accepted_samples_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return relative_image_path


_MISSING = object()


def _default_viewpoint_yaml_path() -> Path:
    return REPO_ROOT / "UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML object in {path}, got {type(raw).__name__}")
    return raw


def load_capture_interval_seconds(
    *,
    viewpoint_yaml_path: Path | None = None,
    default_seconds: float = DEFAULT_CAPTURE_INTERVAL_SECONDS,
) -> float:
    """Load tooling-side capture cadence from viewpoint YAML, defaulting to 15 seconds."""
    path = (viewpoint_yaml_path or _default_viewpoint_yaml_path()).resolve()
    parsed = _load_yaml(path)
    viewpoints = parsed.get("viewpoints", {}) if isinstance(parsed.get("viewpoints"), dict) else {}
    value = viewpoints.get("capture_interval_seconds")
    try:
        interval = float(value)
    except (TypeError, ValueError):
        return float(default_seconds)
    if interval <= 0:
        return float(default_seconds)
    return interval


def _read_path(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value.get(part)
    return value


def _require_non_empty_string(root: dict[str, Any], path: str) -> str:
    value = _read_path(root, path)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/invalid required string: {path}")
    return value.strip()


def _require_positive_int(root: dict[str, Any], path: str) -> int:
    value = _read_path(root, path)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Missing/invalid required positive int: {path}")
    return value


def _require_positive_float(root: dict[str, Any], path: str) -> float:
    value = _read_path(root, path)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"Missing/invalid required positive number: {path}")
    return float(value)


def _format_interval_token(seconds: float) -> str:
    normalized = f"{float(seconds):.6f}".rstrip("0").rstrip(".")
    return f"{normalized.replace('.', 'p')}s"


def _read_png_dimensions(image_bytes: bytes) -> tuple[int, int]:
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError("image_bytes must be bytes-like")
    png_signature = b"\x89PNG\r\n\x1a\n"
    if len(image_bytes) < 24 or bytes(image_bytes[:8]) != png_signature:
        raise ValueError("Expected PNG image bytes for image validation")
    width, height = struct.unpack(">II", bytes(image_bytes[16:24]))
    if width <= 0 or height <= 0:
        raise ValueError("PNG reported non-positive dimensions")
    return int(width), int(height)


class SamplerManager:
    """Own periodic capture intent over an already-running websocket bridge."""

    def __init__(self, config: SamplerManagerConfig) -> None:
        self.config = config
        self._capture_index = 0
        self._next_action_seq = 1

    @staticmethod
    def _active_image_config_from_bridge(bridge: WebSocketBridgeServer) -> ActiveImageConfig:
        config_ref = bridge.get_active_config_reference()
        sensor_rig = bridge.get_active_sensor_rig()
        config_id = str(config_ref.get("config_id", "")).strip()
        config_hash = str(config_ref.get("config_hash", "")).strip()
        if not config_id or not config_hash:
            raise ValueError("Bridge has no active config reference for image validation")
        try:
            width = int(sensor_rig["capture_width"])
            height = int(sensor_rig["capture_height"])
            fov_deg = float(sensor_rig["front_fov_deg"])
        except Exception as exc:
            raise ValueError("Bridge has no active sensor_rig image settings for image validation") from exc
        if width <= 0 or height <= 0 or fov_deg <= 0:
            raise ValueError("Bridge active sensor_rig contains invalid image settings")
        return ActiveImageConfig(
            config_id=config_id,
            config_hash=config_hash,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )

    @classmethod
    def from_viewpoint_yaml(
        cls,
        *,
        run_id: str,
        drone_id: str,
        viewpoint_yaml_path: Path | None = None,
        capture_interval_seconds: float | None = None,
        observation_timeout_seconds: float = DEFAULT_OBSERVATION_TIMEOUT_SECONDS,
        connection_timeout_seconds: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS,
    ) -> "SamplerManager":
        resolved_path = (viewpoint_yaml_path or _default_viewpoint_yaml_path()).resolve()
        interval = (
            float(capture_interval_seconds)
            if capture_interval_seconds is not None
            else load_capture_interval_seconds(viewpoint_yaml_path=resolved_path)
        )
        if interval <= 0:
            interval = DEFAULT_CAPTURE_INTERVAL_SECONDS
        return cls(
            SamplerManagerConfig(
                run_id=str(run_id),
                drone_id=str(drone_id),
                capture_interval_seconds=float(interval),
                observation_timeout_seconds=max(0.1, float(observation_timeout_seconds)),
                connection_timeout_seconds=max(0.1, float(connection_timeout_seconds)),
                viewpoint_yaml_path=resolved_path,
            )
        )

    @staticmethod
    def _utc_now() -> str:
        return protocol.utc_timestamp()

    @staticmethod
    def _extract_capture_id(message: dict[str, Any]) -> str:
        payload = message.get("payload", {}) if isinstance(message.get("payload"), dict) else {}
        return str(message.get("capture_id") or payload.get("capture_id") or "")

    @staticmethod
    def _extract_message_timestamp(message: dict[str, Any]) -> str:
        payload = message.get("payload", {}) if isinstance(message.get("payload"), dict) else {}
        return str(message.get("timestamp") or payload.get("timestamp_utc") or SamplerManager._utc_now())

    def validate_image_observation(
        self,
        observation: dict[str, Any],
        *,
        expected_config: ActiveImageConfig,
    ) -> ValidatedImageSample:
        payload = observation.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("OBS payload must be object for image validation")

        run_id = str(observation.get("run_id") or "").strip()
        drone_id = str(observation.get("drone_id") or "").strip()
        capture_id = str(observation.get("capture_id") or "").strip()
        seq = str(observation.get("seq", "na")).strip() or "na"
        timestamp_utc = self._extract_message_timestamp(observation)

        if run_id != self.config.run_id:
            raise ValueError(f"OBS run_id mismatch: expected={self.config.run_id} got={run_id}")
        if drone_id != self.config.drone_id:
            raise ValueError(f"OBS drone_id mismatch: expected={self.config.drone_id} got={drone_id}")
        if not capture_id:
            raise ValueError("OBS missing capture_id")

        observed_config_id = _require_non_empty_string(payload, "config_ref.config_id")
        observed_config_hash = _require_non_empty_string(payload, "config_ref.config_hash")
        if observed_config_id != expected_config.config_id:
            raise ValueError(
                f"OBS config_id mismatch: expected={expected_config.config_id} got={observed_config_id}"
            )
        if observed_config_hash != expected_config.config_hash:
            raise ValueError(
                f"OBS config_hash mismatch: expected={expected_config.config_hash} got={observed_config_hash}"
            )

        viewpoint_width = _require_positive_int(payload, "viewpoint.width")
        viewpoint_height = _require_positive_int(payload, "viewpoint.height")
        viewpoint_fov_deg = _require_positive_float(payload, "viewpoint.fov_deg")
        if viewpoint_width != expected_config.width:
            raise ValueError(
                f"OBS viewpoint.width mismatch: expected={expected_config.width} got={viewpoint_width}"
            )
        if viewpoint_height != expected_config.height:
            raise ValueError(
                f"OBS viewpoint.height mismatch: expected={expected_config.height} got={viewpoint_height}"
            )
        if viewpoint_fov_deg != expected_config.fov_deg:
            raise ValueError(
                f"OBS viewpoint.fov_deg mismatch: expected={expected_config.fov_deg} got={viewpoint_fov_deg}"
            )

        image_bytes = protocol.decode_image_bytes(payload)
        if image_bytes is None:
            raise ValueError("OBS missing image bytes")
        actual_width, actual_height = _read_png_dimensions(image_bytes)
        if actual_width != viewpoint_width or actual_height != viewpoint_height:
            raise ValueError(
                "Decoded PNG dimensions do not match OBS viewpoint metadata: "
                f"png={actual_width}x{actual_height} metadata={viewpoint_width}x{viewpoint_height}"
            )

        return ValidatedImageSample(
            run_id=run_id,
            drone_id=drone_id,
            capture_id=capture_id,
            seq=seq,
            timestamp_utc=timestamp_utc,
            config_id=observed_config_id,
            config_hash=observed_config_hash,
            viewpoint_width=viewpoint_width,
            viewpoint_height=viewpoint_height,
            viewpoint_fov_deg=viewpoint_fov_deg,
            actual_width=actual_width,
            actual_height=actual_height,
            image_bytes=image_bytes,
        )

    def _log_sample_event(self, status: str, *, capture_id: str, timestamp: str, note: str = "") -> None:
        parts = [
            f"[{SCRIPT_NAME}]",
            status,
            f"run_id={self.config.run_id}",
            f"drone_id={self.config.drone_id}",
            f"capture_id={capture_id}",
            f"timestamp={timestamp}",
        ]
        if note:
            parts.append(f"note={note}")
        print(" ".join(parts))

    def _next_capture_id(self) -> str:
        self._capture_index += 1
        return f"capture_{self._capture_index:06d}"

    def build_capture_action(self, capture_id: str) -> dict[str, Any]:
        """Build the minimal `CAPTURE_NOW` action owned by the sampler."""
        action = protocol.make_action(
            action_type=protocol.TYPE_CAPTURE_NOW,
            run_id=self.config.run_id,
            payload={},
            seq=self._next_action_seq,
            drone_id=self.config.drone_id,
            capture_id=capture_id,
        )
        self._next_action_seq += 1
        return action

    async def run_one_capture_cycle(
        self,
        bridge: WebSocketBridgeServer,
        *,
        accepted_store: AcceptedImageStore | None = None,
    ) -> dict[str, Any]:
        """Run exactly one sampler-owned capture cycle and return structured evidence."""
        if accepted_store is None:
            accepted_store = AcceptedImageStore(bridge.artifacts.root)

        capture_id = self._next_capture_id()
        action = self.build_capture_action(capture_id)
        sent_timestamp = self._extract_message_timestamp(action)
        result: dict[str, Any] = {
            "status": "failed",
            "error": "",
            "sample_status": "failed",
            "capture_id": capture_id,
            "action_seq": action.get("seq"),
            "sent_timestamp_utc": sent_timestamp,
            "observation_received": False,
            "observation_seq": "",
            "observation_timestamp_utc": "",
            "validation_passed": False,
            "accepted_sample_file": "",
            "raw_image_file": "",
            "expected_config": {},
            "observed_config": {},
        }

        try:
            await bridge.send_action(action)
            self._log_sample_event(
                "sent",
                capture_id=capture_id,
                timestamp=sent_timestamp,
            )
        except Exception as exc:
            result["error"] = f"send_error={exc}"
            self._log_sample_event(
                "failed",
                capture_id=capture_id,
                timestamp=self._utc_now(),
                note=result["error"],
            )
            return result

        observation = await self._wait_for_matching_observation(bridge, capture_id)
        if observation is None:
            result["error"] = f"observation_timeout={self.config.observation_timeout_seconds}s"
            self._log_sample_event(
                "failed",
                capture_id=capture_id,
                timestamp=self._utc_now(),
                note=result["error"],
            )
            return result

        observation_timestamp = self._extract_message_timestamp(observation)
        result["observation_received"] = True
        result["observation_seq"] = str(observation.get("seq", ""))
        result["observation_timestamp_utc"] = observation_timestamp

        try:
            expected_image_config = self._active_image_config_from_bridge(bridge)
            result["expected_config"] = {
                "config_id": expected_image_config.config_id,
                "config_hash": expected_image_config.config_hash,
                "width": expected_image_config.width,
                "height": expected_image_config.height,
                "fov_deg": expected_image_config.fov_deg,
            }
            validated_sample = self.validate_image_observation(
                observation,
                expected_config=expected_image_config,
            )
            accepted_path = accepted_store.persist(
                validated_sample,
                capture_interval_seconds=self.config.capture_interval_seconds,
                raw_image_file=bridge.describe_raw_capture_path(observation),
            )
            result["status"] = "success"
            result["sample_status"] = "received"
            result["validation_passed"] = True
            result["accepted_sample_file"] = accepted_path
            result["raw_image_file"] = bridge.describe_raw_capture_path(observation)
            result["observed_config"] = {
                "config_id": validated_sample.config_id,
                "config_hash": validated_sample.config_hash,
                "viewpoint_width": validated_sample.viewpoint_width,
                "viewpoint_height": validated_sample.viewpoint_height,
                "viewpoint_fov_deg": validated_sample.viewpoint_fov_deg,
                "actual_width": validated_sample.actual_width,
                "actual_height": validated_sample.actual_height,
            }
            self._log_sample_event(
                "received",
                capture_id=capture_id,
                timestamp=validated_sample.timestamp_utc,
                note=f"accepted_sample={accepted_path}",
            )
            return result
        except Exception as exc:
            result["error"] = f"validation_error={exc}"
            self._log_sample_event(
                "failed",
                capture_id=capture_id,
                timestamp=observation_timestamp,
                note=result["error"],
            )
            return result

    async def _wait_for_matching_observation(
        self,
        bridge: WebSocketBridgeServer,
        capture_id: str,
    ) -> dict[str, Any] | None:
        deadline = asyncio.get_running_loop().time() + self.config.observation_timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            observation = await bridge.wait_for_observation(timeout_seconds=remaining)
            if observation is None:
                return None
            if self._extract_capture_id(observation) == capture_id:
                return observation

    async def run(self, bridge: WebSocketBridgeServer) -> int:
        """Run periodic sampling against an already-running, externally owned bridge."""
        connected = await bridge.wait_for_client_connection(
            timeout_seconds=self.config.connection_timeout_seconds
        )
        if not connected:
            self._log_sample_event(
                "failed",
                capture_id="connection",
                timestamp=self._utc_now(),
                note=f"ue_connection_timeout={self.config.connection_timeout_seconds}s",
            )
            return 1

        ready = await bridge.wait_for_config_ready()
        if not ready:
            self._log_sample_event(
                "failed",
                capture_id="config",
                timestamp=self._utc_now(),
                note="bridge_never_reached_config_ready",
            )
            return 1

        accepted_store = AcceptedImageStore(bridge.artifacts.root)

        loop = asyncio.get_running_loop()
        next_capture_due = loop.time()
        while True:
            sleep_seconds = next_capture_due - loop.time()
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
            await self.run_one_capture_cycle(bridge, accepted_store=accepted_store)

            next_capture_due += self.config.capture_interval_seconds
