"""
Status: in development.
Renamed from `run_io_handshake_wiring.py` to `run_io_build.py` to mirror the
Course/Drone/IO/WebSocket orchestration layer naming pattern.

Desired behavior:
- Execute minimal IO content generation needed for websocket config handshake runtime integration.
- Rebuild invalid placeholder IO assets as valid Blueprint/Struct assets.
- Ensure required IO runtime actors are placed deterministically in the target map.

Interfaces:
- Runs in Unreal Editor Python through `UE_Tooling/UE_Build/run_unreal_build_gen.py`.
- Executes, in order:
  - `Data_Config/gen_st_runconfig.py`
  - `Data_Config/gen_bp_setdataconfig.py`
  - `Drone_Controller/gen_bp_dronespawner.py`
  - `Drone_Controller/gen_bp_dronecontroller.py`
  - `Data_Interface/gen_bp_samplemanager.py`

Assumptions:
- DroneWebSocket plugin is built/deployed and the game module is compiled so
  `SetDataConfigRuntimeActor` is available as the BP_SetDataConfig parent.
- Target map package path is provided or defaults to the versioned course map.

Success conditions:
- All generation scripts execute without exception.
- IO assets are regenerated and loadable.
- Target map contains deterministic singleton instances for:
  - `BP_SetDataConfig_Main`
  - `BP_SampleManager_Main`
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import unreal

SCRIPT_NAME = "run_io_build"
SCRIPT_VERSION = "1.4.0"
SCRIPT_DATE = "2026-03-16"

BP_SET_DATA_CONFIG_PATH = "/Game/io/Data_Config/BP_SetDataConfig"
BP_SAMPLE_MANAGER_PATH = "/Game/io/Data_Interface/BP_SampleManager"
DEFAULT_LEVEL = "/Game/Course_Content/L_CourseTorus_v004_20260309_083500/Maps/L_CourseTorus"
SET_DATA_CONFIG_ACTOR_LABEL = "BP_SetDataConfig_Main"
SAMPLE_MANAGER_ACTOR_LABEL = "BP_SampleManager_Main"

ORDERED_SCRIPT_PATHS = [
    "Data_Config/gen_st_runconfig.py",
    "Data_Config/gen_bp_setdataconfig.py",
    "Drone_Controller/gen_bp_dronespawner.py",
    "Drone_Controller/gen_bp_dronecontroller.py",
    "Data_Interface/gen_bp_samplemanager.py",
]


@dataclass(frozen=True)
class RunConfig:
    kit_name: str
    version_id: str
    timestamp: str
    level_path: str
    continue_on_failure: bool

    @property
    def run_name(self) -> str:
        return f"{self.kit_name}_{self.version_id}_{self.timestamp}"


@dataclass
class ScriptExecution:
    relative_script_path: str
    absolute_script_path: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    status: str
    error: str | None
    metadata: dict


@dataclass(frozen=True)
class PlacementSpec:
    key: str
    asset_path: str
    actor_label: str
    spawn_xyz: tuple[float, float, float]


REQUIRED_PLACEMENTS = [
    PlacementSpec(
        key="set_data_config",
        asset_path=BP_SET_DATA_CONFIG_PATH,
        actor_label=SET_DATA_CONFIG_ACTOR_LABEL,
        spawn_xyz=(0.0, 0.0, 120.0),
    ),
    PlacementSpec(
        key="sample_manager",
        asset_path=BP_SAMPLE_MANAGER_PATH,
        actor_label=SAMPLE_MANAGER_ACTOR_LABEL,
        spawn_xyz=(200.0, 0.0, 120.0),
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-name", default="IOBuild")
    parser.add_argument("--version-id", default="v001")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    return RunConfig(
        kit_name=_sanitize(str(args.kit_name)),
        version_id=_sanitize(str(args.version_id)),
        timestamp=_sanitize(str(args.timestamp)),
        level_path=str(args.level),
        continue_on_failure=bool(args.continue_on_failure),
    )


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_script_metadata(script_path: Path) -> dict:
    content = script_path.read_text(encoding="utf-8")

    def extract(name: str) -> str | None:
        match = re.search(rf'^{name}\s*=\s*"([^"]+)"', content, re.MULTILINE)
        return match.group(1) if match else None

    return {
        "path": str(script_path),
        "script_name": extract("SCRIPT_NAME"),
        "script_version": extract("SCRIPT_VERSION"),
        "script_date": extract("SCRIPT_DATE"),
        "file_mtime_utc": datetime.fromtimestamp(script_path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def _load_blueprint_class(asset_path: str) -> object:
    blueprint = unreal.EditorAssetLibrary.load_asset(asset_path)
    if blueprint is None:
        raise RuntimeError(f"Missing blueprint asset: {asset_path}")
    generated_class = unreal.BlueprintEditorLibrary.generated_class(blueprint)
    if generated_class is None:
        raise RuntimeError(f"Blueprint generated class unavailable: {asset_path}")
    return generated_class


def _place_single_actor(spec: PlacementSpec, bp_class: object) -> dict:
    actors = list(unreal.EditorLevelLibrary.get_all_level_actors() or [])
    matching = [actor for actor in actors if actor.get_class() == bp_class]
    preexisting_count = len(matching)

    created = False
    destroyed_count = 0
    if not matching:
        spawn_location = unreal.Vector(spec.spawn_xyz[0], spec.spawn_xyz[1], spec.spawn_xyz[2])
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(bp_class, spawn_location)
        if actor is None:
            raise RuntimeError(f"Failed to spawn actor for {spec.asset_path}.")
        matching = [actor]
        created = True

    primary = matching[0]
    for extra in matching[1:]:
        if unreal.EditorLevelLibrary.destroy_actor(extra):
            destroyed_count += 1

    unreal.log(
        f"[{SCRIPT_NAME}] placed actor key={spec.key} label={spec.actor_label} "
        f"created={created} preexisting={preexisting_count} deduplicated={destroyed_count} count_after=1"
    )
    primary.set_actor_label(spec.actor_label)
    return {
        "status": "success",
        "key": spec.key,
        "asset_path": spec.asset_path,
        "actor_label": spec.actor_label,
        "created": created,
        "reused_existing": not created,
        "preexisting_count": preexisting_count,
        "deduplicated_count": destroyed_count,
        "count_after": 1,
        "resolved_actor_name": primary.get_name(),
    }


def _place_required_io_actors(level: str) -> dict:
    loaded_map = unreal.EditorLoadingAndSavingUtils.load_map(str(level))
    if loaded_map is None:
        raise RuntimeError(f"Failed to load map: {level}")

    placements_by_key: dict[str, dict] = {}
    for spec in REQUIRED_PLACEMENTS:
        try:
            bp_class = _load_blueprint_class(spec.asset_path)
            record = _place_single_actor(spec, bp_class)
            record["target_level"] = str(level)
            placements_by_key[spec.key] = record
        except Exception as exc:
            placements_by_key[spec.key] = {
                "status": "failed",
                "key": spec.key,
                "asset_path": spec.asset_path,
                "actor_label": spec.actor_label,
                "target_level": str(level),
                "error": str(exc),
            }

    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    all_success = all(item.get("status") == "success" for item in placements_by_key.values())
    return {
        "attempted": True,
        "status": "success" if all_success else "failed",
        "target_level": str(level),
        "placements_by_key": placements_by_key,
    }


def _run_single_script(script_path: Path) -> ScriptExecution:
    started = datetime.now(tz=timezone.utc)
    start_monotonic = time.monotonic()
    error: str | None = None
    status = "success"
    try:
        module = _load_module(script_path)
        run_fn = getattr(module, "run", None)
        if run_fn is None:
            raise RuntimeError(f"Missing run() in {script_path}")
        run_fn()
    except Exception as exc:
        status = "failed"
        error = str(exc)
    finished = datetime.now(tz=timezone.utc)
    duration = max(0.0, time.monotonic() - start_monotonic)
    return ScriptExecution(
        relative_script_path=str(script_path.relative_to(script_path.parents[1])),
        absolute_script_path=str(script_path),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=round(duration, 3),
        status=status,
        error=error,
        metadata=_extract_script_metadata(script_path),
    )


def _write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def run() -> None:
    cfg = _parse_args()
    repo_root = _repo_root()
    root = Path(__file__).resolve().parent
    unreal.log(f"[{SCRIPT_NAME}] Script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")
    unreal.log(f"[{SCRIPT_NAME}] target level={cfg.level_path}")

    created_at = datetime.now(tz=timezone.utc)
    executions: list[ScriptExecution] = []
    overall_status = "success"
    for relative_path in ORDERED_SCRIPT_PATHS:
        script_path = root / relative_path
        unreal.log(f"[{SCRIPT_NAME}] running {relative_path}")
        execution = _run_single_script(script_path)
        executions.append(execution)
        if execution.status != "success":
            overall_status = "failed"
            unreal.log_error(f"[{SCRIPT_NAME}] failed {relative_path}: {execution.error}")
            if not cfg.continue_on_failure:
                break

    placements: dict
    if overall_status == "success":
        unreal.log(f"[{SCRIPT_NAME}] placing required IO actors into target map")
        try:
            placements = _place_required_io_actors(level=str(cfg.level_path))
        except Exception as exc:
            placements = {
                "attempted": True,
                "status": "failed",
                "target_level": str(cfg.level_path),
                "error": str(exc),
            }
            overall_status = "failed"
    else:
        placements = {
            "attempted": False,
            "status": "skipped",
            "target_level": str(cfg.level_path),
            "placements_by_key": {
                spec.key: {
                    "status": "skipped",
                    "key": spec.key,
                    "asset_path": spec.asset_path,
                    "actor_label": spec.actor_label,
                    "target_level": str(cfg.level_path),
                    "reason": "io_generation_failed",
                }
                for spec in REQUIRED_PLACEMENTS
            },
            "reason": "io_generation_failed",
        }

    placement_compat = (
        placements.get("placements_by_key", {}).get("set_data_config")
        if isinstance(placements, dict)
        else None
    ) or {
        "status": "unknown",
        "key": "set_data_config",
        "asset_path": BP_SET_DATA_CONFIG_PATH,
        "actor_label": SET_DATA_CONFIG_ACTOR_LABEL,
    }

    artifact = {
        "script": {"name": SCRIPT_NAME, "version": SCRIPT_VERSION, "date": SCRIPT_DATE},
        "run_name": cfg.run_name,
        "kit_name": cfg.kit_name,
        "version_id": cfg.version_id,
        "timestamp": cfg.timestamp,
        "created_at_utc": created_at.isoformat(),
        "finished_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": "success" if overall_status == "success" and placements["status"] == "success" else "failed",
        "strict_failure_policy": not cfg.continue_on_failure,
        "target_level": cfg.level_path,
        "ordered_scripts": ORDERED_SCRIPT_PATHS,
        "executed_count": len(executions),
        "required_count": len(ORDERED_SCRIPT_PATHS),
        "executions": [asdict(item) for item in executions],
        "placement": placement_compat,
        "placements": placements,
    }
    artifact_path = repo_root / "UE_Tooling/Artifacts/io_build" / f"{cfg.run_name}.json"
    _write_artifact(artifact_path, artifact)
    unreal.log(f"[{SCRIPT_NAME}] wrote artifact {artifact_path}")

    if artifact["status"] != "success":
        raise RuntimeError(f"{SCRIPT_NAME} failed; see artifact: {artifact_path}")

    unreal.log(f"[{SCRIPT_NAME}] success")


if __name__ == "__main__":
    run()
