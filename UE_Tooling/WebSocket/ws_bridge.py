"""
Desired behavior:
- Run one authoritative WebSocket bridge process for UE runtime control + observation transport.
- Enforce config gating so `SPAWN_DRONES`, `CMD`, and `CAPTURE_NOW` are blocked until config-ready.
- Keep bridge ownership narrow: transport, gate state, optional raw artifact persistence, and
  in-memory observation handoff.

Interfaces:
- Starts a WebSocket server UE connects to as a client.
- Accepts one active run context (`run_id`) and optional startup `SET_CONFIG`.
- Exposes send helpers for config and runtime actions with strict gate checks.

Assumptions:
- v1 `OBS` image bytes arrive as base64 text at `payload.image_bytes_b64`.
- One active UE client connection is sufficient for MVP.
- Validation can run strict or warn-only through `schemas.py`.

Success conditions:
- Bridge logs connection lifecycle and explicit state transitions.
- Config handshakes (`ACK` then `CONFIG_READY`) are recognized and tracked.
- Gated actions are blocked pre-ready and delivered post-ready.
- Raw run artifacts can be written deterministically when enabled, with explicit failure/error records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from websockets.legacy.server import WebSocketServerProtocol

# Allow direct script execution via `python3 UE_Tooling/WebSocket/ws_bridge.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UE_Tooling.Config.RunConfig import compile_set_config
from UE_Tooling.WebSocket import protocol, schemas

SCRIPT_NAME = "ws_bridge"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-22"


@dataclass
class BridgeConfig:
    host: str
    port: int
    run_id: str
    config_timeout_seconds: int
    strict_validation: bool
    artifacts_root: Path
    exit_after_actions: bool
    post_actions_timeout_seconds: int
    persist_raw_artifacts: bool


@dataclass
class BridgeRuntimeState:
    state: str = "BRIDGE_UP"
    ue_connected: bool = False
    config_acked: bool = False
    config_ready: bool = False
    run_failed: bool = False
    run_complete: bool = False
    config_id: str = ""
    config_hash: str = ""
    last_error: str = ""


class RunArtifactStore:
    """Persist optional raw websocket bridge artifacts for one run."""

    def __init__(self, run_id: str, artifacts_root: Path, *, persist_raw_artifacts: bool) -> None:
        self.run_id = run_id
        self.root = artifacts_root / run_id
        self.persist_raw_artifacts = persist_raw_artifacts
        self.raw_captures_dir = self.root / "raw_captures"
        self.raw_messages_jsonl = self.root / "raw_messages.jsonl"
        self.raw_observations_jsonl = self.root / "raw_observations.jsonl"
        self.summary_json = self.root / "summary.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.persist_raw_artifacts:
            self.raw_captures_dir.mkdir(parents=True, exist_ok=True)
        self._message_count = 0
        self._obs_count = 0

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def append_message(
        self,
        *,
        direction: str,
        message: dict[str, Any],
        validation: schemas.ValidationResult | None = None,
        note: str = "",
    ) -> None:
        self._message_count += 1
        record = {
            "timestamp_utc": self._utc_now(),
            "index": self._message_count,
            "direction": direction,
            "note": note,
            "message": message,
        }
        if validation is not None:
            record["validation"] = asdict(validation)
        if self.persist_raw_artifacts:
            with self.raw_messages_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def append_internal(self, note: str, payload: dict[str, Any] | None = None) -> None:
        self.append_message(
            direction="internal",
            message=payload or {},
            validation=None,
            note=note,
        )

    @staticmethod
    def _observation_components(message: dict[str, Any], obs_index: int) -> tuple[str, str, str]:
        payload = message.get("payload", {})
        capture_id = str(message.get("capture_id") or payload.get("capture_id") or f"capture_{obs_index:06d}")
        drone_id = str(message.get("drone_id") or payload.get("drone_id") or "drone_unknown")
        seq_value = str(message.get("seq", "na"))
        return capture_id, drone_id, seq_value

    def raw_capture_relative_path(self, message: dict[str, Any], *, obs_index: int | None = None) -> str:
        index = obs_index if obs_index is not None else max(self._obs_count, 1)
        capture_id, drone_id, seq_value = self._observation_components(message, index)
        return (Path("raw_captures") / capture_id / f"raw__{drone_id}__{seq_value}.png").as_posix()

    def persist_observation(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = message.get("payload", {})
        image_bytes = protocol.decode_image_bytes(payload) if isinstance(payload, dict) else None

        self._obs_count += 1
        capture_id, drone_id, seq_value = self._observation_components(message, self._obs_count)
        raw_image_file = ""
        if image_bytes is not None and self.persist_raw_artifacts:
            image_dir = self.root / "raw_captures" / capture_id
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"raw__{drone_id}__{seq_value}.png"
            image_path.write_bytes(image_bytes)
            raw_image_file = image_path.relative_to(self.root).as_posix()

        obs_record = {
            "timestamp_utc": self._utc_now(),
            "observation_index": self._obs_count,
            "run_id": message.get("run_id"),
            "drone_id": message.get("drone_id"),
            "capture_id": message.get("capture_id"),
            "seq": message.get("seq"),
            "raw_image_file": raw_image_file,
            "message": message,
        }
        if self.persist_raw_artifacts:
            with self.raw_observations_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(obs_record, sort_keys=True) + "\n")
        return {
            "observation_index": self._obs_count,
            "raw_image_file": raw_image_file,
        }

    def write_summary(self, state: BridgeRuntimeState) -> None:
        payload = {
            "script": {"name": SCRIPT_NAME, "version": SCRIPT_VERSION, "date": SCRIPT_DATE},
            "run_id": self.run_id,
            "written_at_utc": self._utc_now(),
            "message_count": self._message_count,
            "observation_count": self._obs_count,
            "state": asdict(state),
        }
        self.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class WebSocketBridgeServer:
    """Bridge server with config gating, validation, and deterministic artifact persistence."""

    def __init__(
        self,
        *,
        config: BridgeConfig,
        startup_set_config: dict[str, Any] | None = None,
        queued_actions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.state = BridgeRuntimeState()
        self.startup_set_config = startup_set_config
        self.queued_actions = queued_actions or []
        self.artifacts = RunArtifactStore(
            config.run_id,
            config.artifacts_root,
            persist_raw_artifacts=config.persist_raw_artifacts,
        )
        self._server: websockets.server.Serve | None = None
        self._client: WebSocketServerProtocol | None = None
        self._connected_event = asyncio.Event()
        self._config_ack_event = asyncio.Event()
        self._config_ready_event = asyncio.Event()
        self._error_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._obs_event = asyncio.Event()
        self._observations_received = 0
        self._observation_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._active_set_config_message: dict[str, Any] | None = None

    def _log(self, message: str) -> None:
        print(f"[{SCRIPT_NAME}] {message}")

    def _transition(self, next_state: str) -> None:
        if self.state.state != next_state:
            self._log(f"state: {self.state.state} -> {next_state}")
            self.state.state = next_state
            self.artifacts.append_internal("state_transition", {"state": next_state})

    @staticmethod
    def _copy_message(message: dict[str, Any] | None) -> dict[str, Any] | None:
        return deepcopy(message) if message is not None else None

    async def start(self) -> None:
        self._server = await websockets.serve(self._on_connect, self.config.host, self.config.port, max_size=None)
        self._log(f"listening ws://{self.config.host}:{self.config.port} run_id={self.config.run_id}")
        self.artifacts.append_internal(
            "bridge_started",
            {
                "host": self.config.host,
                "port": self.config.port,
                "run_id": self.config.run_id,
                "strict_validation": self.config.strict_validation,
            },
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self.state.run_complete = not self.state.run_failed
        self.artifacts.write_summary(self.state)
        self._log("bridge stopped")

    async def _on_connect(self, websocket: WebSocketServerProtocol) -> None:
        if self._client is not None:
            await websocket.close(code=1013, reason="Bridge already has active UE client")
            return

        self._client = websocket
        self.state.ue_connected = True
        self._connected_event.set()
        self._transition("UE_CONNECTED")
        self._log("UE client connected")
        self.artifacts.append_internal("ue_connected")

        try:
            if self.startup_set_config is not None:
                await self.send_set_config(self.startup_set_config)
            async for raw_message in websocket:
                await self._handle_incoming(raw_message)
        except websockets.ConnectionClosed:
            self._log("UE client disconnected")
        finally:
            self.state.ue_connected = False
            self._client = None
            self._transition("BRIDGE_UP")
            self.artifacts.append_internal("ue_disconnected")

    async def _handle_incoming(self, raw_message: str) -> None:
        parsed = protocol.loads_message(raw_message)
        validation = schemas.validate_message(parsed, strict=self.config.strict_validation)
        self.artifacts.append_message(direction="inbound", message=parsed, validation=validation)

        if not validation.valid:
            reason = "; ".join(validation.errors)
            self._mark_error(f"inbound_validation_failed: {reason}")
            return
        for warning in validation.warnings:
            self._log(f"validation warning: {warning}")

        message_type = protocol.normalize_type(parsed["type"])
        if message_type in protocol.CONFIG_ACK_TYPES:
            self.state.config_acked = True
            self._config_ack_event.set()
            self._transition("CONFIG_ACKED")
            self._log(f"received {message_type}")
            return

        if message_type in protocol.CONFIG_READY_TYPES:
            payload = parsed.get("payload", {}) if isinstance(parsed.get("payload"), dict) else {}
            self.state.config_ready = True
            self.state.config_id = str(payload.get("config_id", ""))
            self.state.config_hash = str(payload.get("config_hash", ""))
            self._transition("CONFIG_READY")
            self._config_ready_event.set()
            self._log(f"received {message_type} config_id={self.state.config_id} config_hash={self.state.config_hash}")
            return

        if message_type == protocol.TYPE_OBS:
            self._observations_received += 1
            self._obs_event.set()
            self.artifacts.persist_observation(parsed)
            await self._observation_queue.put(parsed)
            self._log("received OBS")
            return

        if message_type == protocol.TYPE_STATUS:
            await self._status_queue.put(parsed)
            self._log("received STATUS")
            return

        if message_type == protocol.TYPE_ERROR:
            error_payload = parsed.get("payload", {})
            self._mark_error(f"UE_ERROR: {error_payload}")
            return

        self._log(f"received unhandled type: {message_type}")

    def _mark_error(self, message: str) -> None:
        self.state.run_failed = True
        self.state.last_error = message
        self._error_event.set()
        self._transition("RUN_FAILED")
        self._log(f"ERROR: {message}")
        self.artifacts.append_internal("error", {"error": message})

    async def _wait_for_signal_or_failure(
        self,
        *,
        signal_event: asyncio.Event,
        timeout_seconds: float | None,
        timeout_error: str,
    ) -> bool:
        if signal_event.is_set():
            return True
        if self._error_event.is_set() or self.state.run_failed:
            return False

        signal_task = asyncio.create_task(signal_event.wait())
        error_task = asyncio.create_task(self._error_event.wait())
        wait_set = {signal_task, error_task}
        done: set[asyncio.Task[Any]]
        pending: set[asyncio.Task[Any]]
        try:
            done, pending = await asyncio.wait(
                wait_set,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in wait_set:
                if not task.done():
                    task.cancel()

        if signal_task in done:
            return True
        if error_task in done or self._error_event.is_set() or self.state.run_failed:
            return False
        if timeout_seconds is not None and not done and timeout_error:
            self._mark_error(timeout_error)
        return False

    async def send_message(self, message: dict[str, Any], *, gated_action: bool = False) -> None:
        if self._client is None:
            raise RuntimeError("No UE client connected")
        normalized_type = protocol.normalize_type(str(message.get("type", "")))
        if gated_action and normalized_type in protocol.ACTION_TYPES and not self.state.config_ready:
            reason = f"blocked pre-config-ready action: {normalized_type}"
            self.artifacts.append_internal("action_blocked", {"type": normalized_type, "reason": reason})
            raise RuntimeError(reason)

        validation = schemas.validate_message(message, strict=self.config.strict_validation)
        self.artifacts.append_message(direction="outbound", message=message, validation=validation)
        if not validation.valid:
            raise schemas.SchemaValidationError("; ".join(validation.errors))

        await self._client.send(protocol.dumps_message(message))
        self._log(f"sent {normalized_type}")

    async def send_set_config(self, set_config_message: dict[str, Any]) -> None:
        self._active_set_config_message = self._copy_message(set_config_message)
        self.state.config_acked = False
        self.state.config_ready = False
        self._config_ack_event.clear()
        self._config_ready_event.clear()
        self._transition("CONFIG_WAITING")
        await self.send_message(set_config_message, gated_action=False)

    async def send_action(self, action_message: dict[str, Any]) -> None:
        await self.send_message(action_message, gated_action=True)
        if self.state.state == "CONFIG_READY":
            self._transition("RUN_ACTIVE")

    async def wait_for_config_ready(self) -> bool:
        return await self._wait_for_signal_or_failure(
            signal_event=self._config_ready_event,
            timeout_seconds=float(self.config.config_timeout_seconds),
            timeout_error=(
                f"config_ready timeout after {self.config.config_timeout_seconds}s "
                f"for run_id={self.config.run_id}"
            ),
        )

    async def wait_for_config_ack(self, timeout_seconds: float | None = None) -> bool:
        wait_timeout = float(self.config.config_timeout_seconds) if timeout_seconds is None else float(timeout_seconds)
        return await self._wait_for_signal_or_failure(
            signal_event=self._config_ack_event,
            timeout_seconds=wait_timeout,
            timeout_error=f"config_ack timeout after {wait_timeout}s for run_id={self.config.run_id}",
        )

    async def wait_for_config_handshake(self, timeout_seconds: float | None = None) -> bool:
        ack_ok = await self.wait_for_config_ack(timeout_seconds=timeout_seconds)
        if not ack_ok:
            return False
        return await self.wait_for_config_ready()

    async def wait_for_error(self, timeout_seconds: float | None = None) -> str | None:
        if self._error_event.is_set() or self.state.run_failed:
            return str(self.state.last_error)
        if timeout_seconds is None:
            await self._error_event.wait()
            return str(self.state.last_error)
        try:
            await asyncio.wait_for(self._error_event.wait(), timeout=float(timeout_seconds))
            return str(self.state.last_error)
        except asyncio.TimeoutError:
            return None

    async def wait_for_client_connection(self, timeout_seconds: float | None = None) -> bool:
        if timeout_seconds is None:
            return await self._wait_for_signal_or_failure(
                signal_event=self._connected_event,
                timeout_seconds=None,
                timeout_error="",
            )
        return await self._wait_for_signal_or_failure(
            signal_event=self._connected_event,
            timeout_seconds=float(timeout_seconds),
            timeout_error=f"client_connection timeout after {timeout_seconds}s for run_id={self.config.run_id}",
        )

    async def wait_for_observation(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        try:
            if timeout_seconds is None:
                return await self._observation_queue.get()
            return await asyncio.wait_for(self._observation_queue.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    async def wait_for_status(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        try:
            if timeout_seconds is None:
                return await self._status_queue.get()
            return await asyncio.wait_for(self._status_queue.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    async def wait_for_expected_observations(self, expected_count: int) -> bool:
        if expected_count <= 0:
            return True
        deadline = asyncio.get_running_loop().time() + float(self.config.post_actions_timeout_seconds)
        while self._observations_received < expected_count:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self._mark_error(
                    "post-action timeout waiting for OBS "
                    f"(expected={expected_count}, received={self._observations_received})"
                )
                return False
            try:
                await asyncio.wait_for(self._obs_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                self._mark_error(
                    "post-action timeout waiting for OBS "
                    f"(expected={expected_count}, received={self._observations_received})"
                )
                return False
            self._obs_event.clear()
        return True

    def get_active_set_config(self) -> dict[str, Any] | None:
        return self._copy_message(self._active_set_config_message)

    def get_active_config_reference(self) -> dict[str, str]:
        return {
            "config_id": str(self.state.config_id),
            "config_hash": str(self.state.config_hash),
        }

    def get_active_sensor_rig(self) -> dict[str, Any]:
        active_set_config = self.get_active_set_config() or {}
        payload = active_set_config.get("payload", {})
        if not isinstance(payload, dict):
            return {}
        sensor_rig = payload.get("sensor_rig", {})
        return deepcopy(sensor_rig) if isinstance(sensor_rig, dict) else {}

    def describe_raw_capture_path(self, message: dict[str, Any]) -> str:
        if not self.config.persist_raw_artifacts:
            return ""
        return self.artifacts.raw_capture_relative_path(message)

    async def run(self) -> int:
        await self.start()
        try:
            await self._connected_event.wait()
            if self.startup_set_config is not None:
                ready = await self.wait_for_config_handshake()
                if not ready:
                    return 1

            for action in self.queued_actions:
                try:
                    await self.send_action(action)
                except Exception as exc:
                    self._mark_error(f"failed to send action: {exc}")
                    return 1

            if self.config.exit_after_actions:
                expected_observations = sum(
                    1
                    for action in self.queued_actions
                    if protocol.normalize_type(str(action.get("type", ""))) == protocol.TYPE_CAPTURE_NOW
                )
                ready_to_exit = await self.wait_for_expected_observations(expected_observations)
                if not ready_to_exit:
                    return 1
                return 1 if self.state.run_failed else 0

            await self._stop_event.wait()
            return 1 if self.state.run_failed else 0
        finally:
            await self.stop()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "set_config" in payload and isinstance(payload["set_config"], dict):
        return payload["set_config"]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_actions(path: Path, run_id: str) -> list[dict[str, Any]]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected action list in {path}")
    output: list[dict[str, Any]] = []
    for index, action in enumerate(parsed, start=1):
        if not isinstance(action, dict):
            raise ValueError(f"Action at index {index} is not an object")
        action_type = protocol.normalize_type(str(action.get("type", "")))
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError(f"Action payload at index {index} must be object")
        output.append(
            protocol.make_action(
                action_type=action_type,
                run_id=run_id,
                payload=payload,
                seq=action.get("seq"),
                drone_id=action.get("drone_id"),
                capture_id=action.get("capture_id"),
                in_reply_to=action.get("in_reply_to"),
            )
        )
    return output


def _default_artifacts_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "UE_Tooling/Artifacts/websocket"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-id", default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--config-timeout-seconds", type=int, default=30)
    parser.add_argument("--set-config-json", default="")
    parser.add_argument("--set-config-from-yaml", action="store_true")
    parser.add_argument("--actions-json", default="")
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--warn-validation", action="store_true")
    parser.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    parser.add_argument("--no-raw-artifacts", action="store_true")
    parser.add_argument("--exit-after-actions", action="store_true")
    parser.add_argument("--post-actions-timeout-seconds", type=int, default=20)
    return parser


async def _async_main() -> int:
    args = _build_arg_parser().parse_args()
    strict_validation = True
    if args.warn_validation:
        strict_validation = False
    if args.strict_validation:
        strict_validation = True

    startup_set_config: dict[str, Any] | None = None
    if args.set_config_json:
        startup_set_config = _load_json(Path(args.set_config_json).resolve())
    elif args.set_config_from_yaml:
        compiled = compile_set_config(run_id=str(args.run_id))
        startup_set_config = compiled.envelope

    queued_actions: list[dict[str, Any]] = []
    if args.actions_json:
        queued_actions = _load_actions(Path(args.actions_json).resolve(), str(args.run_id))

    config = BridgeConfig(
        host=str(args.host),
        port=int(args.port),
        run_id=str(args.run_id),
        config_timeout_seconds=max(1, int(args.config_timeout_seconds)),
        strict_validation=bool(strict_validation),
        artifacts_root=Path(args.artifacts_root).resolve(),
        exit_after_actions=bool(args.exit_after_actions),
        post_actions_timeout_seconds=max(1, int(args.post_actions_timeout_seconds)),
        persist_raw_artifacts=not bool(args.no_raw_artifacts),
    )
    bridge = WebSocketBridgeServer(
        config=config,
        startup_set_config=startup_set_config,
        queued_actions=queued_actions,
    )
    return await bridge.run()


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
