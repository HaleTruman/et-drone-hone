"""
Use case:
- Tooling-side owner of spawn action shaping and spawn-capability probe semantics.
- Keep spawn behavior explicit and supervised through the shared websocket bridge.

Interfaces:
- Builds canonical `SPAWN_DRONES` actions using `UE_Tooling/WebSocket/protocol.py`.
- Executes probe flows against an existing `WebSocketBridgeServer` transport instance.

Assumptions:
- Bridge startup/config apply are owned by a higher-level runtime orchestrator.
- Spawn capability probing is runtime-scoped and should be intentionally minimal.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UE_Tooling.WebSocket import protocol
from UE_Tooling.WebSocket.ws_bridge import WebSocketBridgeServer

SCRIPT_NAME = "Drone_Spawner"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-22"

DEFAULT_STATUS_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class SpawnCapabilityConfig:
    run_id: str
    probe_drone_id: str
    status_timeout_seconds: float = DEFAULT_STATUS_TIMEOUT_SECONDS
    spawn_count: int = 1
    action_seq: int | None = 1


def build_spawn_action(config: SpawnCapabilityConfig) -> dict[str, Any]:
    spawn_count = max(1, int(config.spawn_count))
    payload: dict[str, Any] = {"spawn_count": spawn_count}
    probe_drone_id = str(config.probe_drone_id).strip()
    if probe_drone_id:
        payload["drone_ids"] = [probe_drone_id]
    return protocol.make_action(
        action_type=protocol.TYPE_SPAWN_DRONES,
        run_id=str(config.run_id),
        payload=payload,
        seq=config.action_seq,
    )


def _status_details(message: dict[str, Any]) -> dict[str, Any]:
    payload_raw = message.get("payload", {})
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    drone_ids_raw = payload.get("drone_ids", [])
    drone_ids = [str(item) for item in drone_ids_raw] if isinstance(drone_ids_raw, list) else []
    return {
        "message_type": str(message.get("type", "")),
        "run_id": str(message.get("run_id", "")),
        "seq": message.get("seq"),
        "timestamp": str(message.get("timestamp", "")),
        "event": str(payload.get("event", "")),
        "spawned_count": int(payload.get("spawned_count", 0)) if isinstance(payload.get("spawned_count"), (int, float)) else 0,
        "drone_ids": drone_ids,
        "raw_message": message,
    }


def _is_spawn_accepted_status(details: dict[str, Any], *, expected_run_id: str) -> bool:
    if str(details.get("message_type", "")).upper() != protocol.TYPE_STATUS:
        return False
    if str(details.get("run_id", "")) != str(expected_run_id):
        return False
    return str(details.get("event", "")) == "SPAWN_DRONES_ACCEPTED"


async def wait_for_spawn_accepted_status(
    bridge: WebSocketBridgeServer,
    *,
    run_id: str,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    deadline = asyncio.get_running_loop().time() + max(0.1, float(timeout_seconds))
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        status_message = await bridge.wait_for_status(timeout_seconds=remaining)
        if status_message is None:
            return None
        details = _status_details(status_message)
        if _is_spawn_accepted_status(details, expected_run_id=run_id):
            return details


def describe_module_surface() -> dict[str, Any]:
    return {
        "script": {"name": SCRIPT_NAME, "version": SCRIPT_VERSION, "date": SCRIPT_DATE},
        "action_type": protocol.TYPE_SPAWN_DRONES,
        "supports_runtime_probe": True,
        "requires_config_ready": True,
        "status_event": "SPAWN_DRONES_ACCEPTED",
    }


async def run_spawn_capability_probe(
    bridge: WebSocketBridgeServer,
    *,
    config: SpawnCapabilityConfig,
) -> dict[str, Any]:
    result = {
        "status": "failed",
        "capability_status": "failed",
        "error": "",
        "probe_action": {},
        "status_details": {},
    }
    if bridge.state.run_failed:
        result["status"] = "blocked"
        result["capability_status"] = "blocked"
        result["error"] = bridge.state.last_error or "Bridge is already in failed state."
        return result
    if not bridge.state.config_ready:
        result["status"] = "blocked"
        result["capability_status"] = "blocked"
        result["error"] = "Bridge is not config-ready; spawn probe cannot run."
        return result

    action = build_spawn_action(config)
    result["probe_action"] = action
    try:
        await bridge.send_action(action)
    except Exception as exc:
        result["status"] = "failed"
        result["capability_status"] = "failed"
        result["error"] = f"Failed to send SPAWN_DRONES probe action: {exc}"
        return result

    accepted = await wait_for_spawn_accepted_status(
        bridge,
        run_id=config.run_id,
        timeout_seconds=config.status_timeout_seconds,
    )
    if accepted is None:
        result["status"] = "failed"
        result["capability_status"] = "failed"
        if bridge.state.run_failed:
            result["error"] = bridge.state.last_error or "Bridge entered failed state during spawn probe."
        else:
            result["error"] = (
                "Timed out waiting for SPAWN_DRONES_ACCEPTED "
                f"after {max(0.1, float(config.status_timeout_seconds))}s."
            )
        return result

    result["status_details"] = accepted
    result["status"] = "success"
    result["capability_status"] = "ready"
    return result
