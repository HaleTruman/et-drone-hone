"""
Status: in development.
Top-level runtime orchestration entrypoint for Unreal IO sessions.

Current scope (phase 2-8):
- Consume and validate canonical build handoff from `Artifacts/build/<run_name>.json`.
- Prove runtime bootstrap topology assumptions before UE launch/config apply stages.
- Start and supervise websocket bridge lifecycle with explicit failure reporting.
- Launch Unreal against the authoritative level in a render-enabled runtime mode.
- Compile/load runtime config and apply it through explicit handshake supervision.
- Validate startup topology readiness against canonical preplacement + live owner evidence.
- Supervise a one-shot sampler-owned image smoke capture and validate returned OBS truth.
- Integrate and report a supervised spawn-capability probe through `Drone_Spawner`.
- Capture Unreal runtime launch/log evidence into the runtime session manifest.
- Emit operator-readable manifest summary fields (`operator_summary`) for quick triage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from asyncio.subprocess import STDOUT
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow direct script execution via `python3 UE_Tooling/run_unreal_io.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UE_Tooling.Config.RunConfig import compile_set_config
from UE_Tooling.Data_Interface.Sampler_Manager import SamplerManager
from UE_Tooling.Drone_Controller.Drone_Spawner import (
    SpawnCapabilityConfig,
    describe_module_surface as describe_spawner_module_surface,
    run_spawn_capability_probe,
)
from UE_Tooling.WebSocket.ws_bridge import BridgeConfig, WebSocketBridgeServer

SCRIPT_NAME = "run_unreal_io"
SCRIPT_VERSION = "0.10.0"
SCRIPT_DATE = "2026-03-22"

REQUIRED_BUILD_STAGES = ("websocket", "course", "drone", "io")
REQUIRED_STARTUP_PLACEMENTS = {
    "set_data_config": "BP_SetDataConfig_Main",
    "sample_manager": "BP_SampleManager_Main",
    "startup_drone_pawn": "BP_DronePawn_Startup_Main",
}
RUNTIME_PHASE_ORDER = (
    "phase2_build_handoff_ingest",
    "phase3_bridge_supervision",
    "phase4_unreal_runtime_launch",
    "phase5_config_apply",
    "phase6_startup_topology_validation",
    "phase7_sampler_smoke_capture",
    "phase8_spawner_capability",
    "phase9_controller_capability",
)


@dataclass(frozen=True)
class RuntimeArgs:
    build_manifest: str
    build_run_name: str
    runtime_run_id: str
    host: str
    port: int
    config_timeout_seconds: int
    connect_timeout_seconds: int
    bridge_probe_seconds: int
    require_client_connection: bool
    strict_validation: bool
    artifacts_root: str
    set_config_json: str
    set_config_from_yaml: bool
    unreal_startup_timeout_seconds: int
    unreal_shutdown_timeout_seconds: int
    unreal_userdir: str
    skip_phase7_smoke: bool
    smoke_drone_id: str
    smoke_observation_timeout_seconds: float
    smoke_viewpoint_yaml: str
    skip_phase8_spawn_probe: bool
    spawn_probe_drone_id: str
    spawn_probe_count: int
    spawn_probe_timeout_seconds: float


@dataclass(frozen=True)
class UnrealLaunchPlan:
    command: list[str]
    editor_binary: str
    project_path: str
    authoritative_level_path: str
    userdir: str
    stdout_log_path: str
    expected_engine_log_path: str


@dataclass(frozen=True)
class Phase5ConfigContext:
    set_config_message: dict[str, Any]
    mode: str
    source_path: str
    source_paths: dict[str, str]
    assumptions: list[str]
    expected_config_id: str
    expected_config_hash: str


def _parse_live_sensor_owner_apply_evidence(engine_log_path: Path) -> dict[str, Any]:
    evidence = {
        "engine_log_path": str(engine_log_path),
        "engine_log_exists": engine_log_path.exists(),
        "sensor_owner_apply_records": [],
        "no_live_sensor_owner_error_seen": False,
        "config_ready_line_seen": False,
    }
    if not engine_log_path.exists():
        return evidence

    apply_pattern = re.compile(r"set_config applied to\s+(\d+)/(\d+)\s+live sensor owners")
    no_owner_phrase = "no live sensor owners found"
    config_ready_phrase = "config ready run_id="

    with engine_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, start=1):
            if no_owner_phrase in line:
                evidence["no_live_sensor_owner_error_seen"] = True
            if config_ready_phrase in line:
                evidence["config_ready_line_seen"] = True
            match = apply_pattern.search(line)
            if not match:
                continue
            applied = int(match.group(1))
            found = int(match.group(2))
            evidence["sensor_owner_apply_records"].append(
                {
                    "line_number": index,
                    "applied": applied,
                    "found": found,
                    "line": line.strip(),
                }
            )
    return evidence


def _validate_phase6_startup_topology(
    *,
    phase2_result: dict[str, Any],
    phase5_result: dict[str, Any],
    phase4_result: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "status": "success",
        "error": "",
        "failure_classification": "",
        "canonical_preplacement_valid": False,
        "transport_config_ready_proven": False,
        "startup_pawn_evidence": {},
        "live_sensor_owner_valid": False,
        "live_sensor_owner_evidence": {},
        "post_config_state_ready": False,
        "action_surface_topology_ready": False,
        "sampler_surface_ready": False,
        "spawner_surface_ready": False,
        "controller_surface_ready": False,
        "errors": [],
    }

    phase5_status = str(phase5_result.get("status", ""))
    if phase5_status != "success":
        result["status"] = "blocked"
        result["failure_classification"] = "config_failure"
        result["error"] = "Phase 6 blocked: Phase 5 config apply did not succeed."
        result["errors"] = [result["error"]]
        return result

    required = phase2_result.get("required_startup_placements", {})
    startup_entry = required.get("startup_drone_pawn", {}) if isinstance(required, dict) else {}
    if not isinstance(startup_entry, dict):
        startup_entry = {}
    result["startup_pawn_evidence"] = {
        "present": bool(startup_entry.get("present", False)),
        "status": str(startup_entry.get("status", "")),
        "actor_label": str(startup_entry.get("actor_label", "")),
        "expected_actor_label": str(startup_entry.get("expected_actor_label", "")),
        "target_level": str(startup_entry.get("target_level", "")),
        "target_level_matches_authoritative": bool(startup_entry.get("target_level_matches_authoritative", False)),
    }
    result["canonical_preplacement_valid"] = bool(
        result["startup_pawn_evidence"]["present"]
        and result["startup_pawn_evidence"]["status"] == "success"
        and result["startup_pawn_evidence"]["actor_label"] == result["startup_pawn_evidence"]["expected_actor_label"]
        and result["startup_pawn_evidence"]["target_level_matches_authoritative"]
    )
    if not result["canonical_preplacement_valid"]:
        result["errors"].append("Canonical startup pawn preplacement is invalid or missing.")

    result["transport_config_ready_proven"] = bool(
        phase5_result.get("config_ack_observed", False)
        and phase5_result.get("config_ready_observed", False)
        and phase5_result.get("config_reference_match", False)
    )

    engine_log_path = Path(str(phase4_result.get("expected_engine_log_path", "")))
    live_owner_evidence = _parse_live_sensor_owner_apply_evidence(engine_log_path)
    result["live_sensor_owner_evidence"] = live_owner_evidence
    apply_records = live_owner_evidence.get("sensor_owner_apply_records", [])
    latest_apply = apply_records[-1] if apply_records else {}
    found = int(latest_apply.get("found", 0)) if latest_apply else 0
    applied = int(latest_apply.get("applied", 0)) if latest_apply else 0
    log_apply_proven = bool(found > 0 and applied > 0)
    result["live_sensor_owner_valid"] = bool(log_apply_proven or result["transport_config_ready_proven"])
    if not result["live_sensor_owner_valid"]:
        if bool(live_owner_evidence.get("no_live_sensor_owner_error_seen", False)):
            result["errors"].append("Runtime reported no live sensor owners during SET_CONFIG apply.")
        else:
            result["errors"].append("No positive live sensor owner apply evidence was found in Unreal log.")

    result["post_config_state_ready"] = bool(result["transport_config_ready_proven"])
    if not result["post_config_state_ready"]:
        result["errors"].append("Post-config runtime readiness evidence is incomplete.")

    topology_ready = bool(
        result["canonical_preplacement_valid"]
        and result["live_sensor_owner_valid"]
        and result["post_config_state_ready"]
    )
    result["action_surface_topology_ready"] = topology_ready
    result["sampler_surface_ready"] = topology_ready
    result["spawner_surface_ready"] = topology_ready
    result["controller_surface_ready"] = topology_ready

    if result["errors"]:
        result["status"] = "failed"
        result["failure_classification"] = "topology_failure"
        result["error"] = result["errors"][0]
    return result


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Failed to locate repo root (.git not found).")


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _default_build_artifacts_dir(repo_root: Path) -> Path:
    return repo_root / "UE_Tooling/Artifacts/build"


def _default_runtime_artifacts_root(repo_root: Path) -> Path:
    return repo_root / "UE_Tooling/Artifacts/runtime"


def _build_arg_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", default="")
    parser.add_argument("--build-run-name", default="")
    parser.add_argument("--runtime-run-id", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config-timeout-seconds", type=int, default=30)
    parser.add_argument("--connect-timeout-seconds", type=int, default=15)
    parser.add_argument("--bridge-probe-seconds", type=int, default=0)
    parser.add_argument("--require-client-connection", action="store_true")
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--warn-validation", action="store_true")
    parser.add_argument("--artifacts-root", default=str(_default_runtime_artifacts_root(repo_root)))
    parser.add_argument("--set-config-json", default="")
    parser.add_argument("--set-config-from-yaml", action="store_true")
    parser.add_argument("--unreal-startup-timeout-seconds", type=int, default=120)
    parser.add_argument("--unreal-shutdown-timeout-seconds", type=int, default=20)
    parser.add_argument("--unreal-userdir", default="")
    parser.add_argument("--skip-phase7-smoke", action="store_true")
    parser.add_argument("--smoke-drone-id", default="")
    parser.add_argument("--smoke-observation-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--smoke-viewpoint-yaml", default="")
    parser.add_argument("--skip-phase8-spawn-probe", action="store_true")
    parser.add_argument("--spawn-probe-drone-id", default="")
    parser.add_argument("--spawn-probe-count", type=int, default=1)
    parser.add_argument("--spawn-probe-timeout-seconds", type=float, default=10.0)
    return parser


def _parse_args(repo_root: Path) -> RuntimeArgs:
    parser = _build_arg_parser(repo_root)
    raw = parser.parse_args()
    strict_validation = True
    if raw.warn_validation:
        strict_validation = False
    if raw.strict_validation:
        strict_validation = True
    return RuntimeArgs(
        build_manifest=str(raw.build_manifest).strip(),
        build_run_name=str(raw.build_run_name).strip(),
        runtime_run_id=_sanitize(str(raw.runtime_run_id).strip()),
        host=str(raw.host).strip(),
        port=max(1, int(raw.port)),
        config_timeout_seconds=max(1, int(raw.config_timeout_seconds)),
        connect_timeout_seconds=max(1, int(raw.connect_timeout_seconds)),
        bridge_probe_seconds=max(0, int(raw.bridge_probe_seconds)),
        require_client_connection=bool(raw.require_client_connection),
        strict_validation=bool(strict_validation),
        artifacts_root=str(raw.artifacts_root).strip(),
        set_config_json=str(raw.set_config_json).strip(),
        set_config_from_yaml=bool(raw.set_config_from_yaml),
        unreal_startup_timeout_seconds=max(1, int(raw.unreal_startup_timeout_seconds)),
        unreal_shutdown_timeout_seconds=max(1, int(raw.unreal_shutdown_timeout_seconds)),
        unreal_userdir=str(raw.unreal_userdir).strip(),
        skip_phase7_smoke=bool(raw.skip_phase7_smoke),
        smoke_drone_id=str(raw.smoke_drone_id).strip(),
        smoke_observation_timeout_seconds=max(0.1, float(raw.smoke_observation_timeout_seconds)),
        smoke_viewpoint_yaml=str(raw.smoke_viewpoint_yaml).strip(),
        skip_phase8_spawn_probe=bool(raw.skip_phase8_spawn_probe),
        spawn_probe_drone_id=str(raw.spawn_probe_drone_id).strip(),
        spawn_probe_count=max(1, int(raw.spawn_probe_count)),
        spawn_probe_timeout_seconds=max(0.1, float(raw.spawn_probe_timeout_seconds)),
    )
