"""
Status: in development.
Top-level build-layer orchestrator shell for UE tooling.

Current scope:
- Build one deterministic orchestration context for the full build flow.
- Resolve required stage runner inputs before execution begins.
- Declare explicit build-stage order and stage commands.
- Execute the WebSocket, Course, Drone, and IO build stages through domain runners.
- Validate expected stage artifacts and record them in a top-level build manifest.
- Extract `ue_paths.level_path` from the Course artifact as the authoritative
  level handoff for downstream stages, including explicit Drone/IO `--level`.
- Enforce fail-fast gating if WebSocket, Course, Drone, or IO stage fails.
- Validate IO placement success and target-level consistency from the IO artifact,
  including canonical startup drone preplacement proof.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPT_NAME = "run_unreal_build"
SCRIPT_VERSION = "0.5.0"
SCRIPT_DATE = "2026-03-22"

BUILD_KIT_NAME = "UEBuild"
WEBSOCKET_KIT_NAME = "WebSocketPlugin"
COURSE_NAME = "L_CourseTorus"
DRONE_KIT_NAME = "DroneContentKit"
IO_KIT_NAME = "IOBuild"
AUTHORITATIVE_LEVEL_PLACEHOLDER = "<AUTHORITATIVE_COURSE_LEVEL_FROM_COURSE_STAGE_ARTIFACT>"
IO_REQUIRED_PLACEMENTS = {
    "set_data_config": "BP_SetDataConfig_Main",
    "sample_manager": "BP_SampleManager_Main",
    "startup_drone_pawn": "BP_DronePawn_Startup_Main",
}


@dataclass(frozen=True)
class BuildConfig:
    timestamp: str
    version_id: str
    unreal_editor: str
    project: str
    engine_root: str
    build_script: str
    editor_binary: str
    timeout_seconds: int
    overwrite_project_plugin: bool

    @property
    def run_name(self) -> str:
        return f"{BUILD_KIT_NAME}_{self.version_id}_{self.timestamp}"


@dataclass(frozen=True)
class StagePlan:
    name: str
    runner_kind: str
    runner_script: str
    expected_artifact_path: str
    command: list[str]
    requires_authoritative_course_level: bool


@dataclass(frozen=True)
class BuildContext:
    script_name: str
    script_version: str
    script_date: str
    run_name: str
    timestamp: str
    version_id: str
    fail_fast: bool
    timeout_seconds: int
    repo_root: str
    project: str
    engine_root: str
    build_script: str
    editor_binary: str
    unreal_editor: str
    overwrite_project_plugin: bool
    stage_order: list[str]
    stage_plans: list[StagePlan]


@dataclass(frozen=True)
class StageExecution:
    name: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    command: list[str]
    runner_script: str
    expected_artifact_path: str
    artifact_path: str
    artifact_exists: bool
    artifact_status: Optional[str]
    return_code: int
    stdout: str
    stderr: str
    error: str


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Failed to locate repo root (.git not found).")


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _default_unreal_editor() -> Optional[Path]:
    candidates = [
        Path("/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_existing_path(raw_value: str, repo_root: Path, label: str) -> Path:
    raw = Path(raw_value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(repo_root / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(f"Missing required path for {label}: {raw_value}")


def _derive_engine_root(editor_binary: Path) -> Path:
    for parent in editor_binary.parents:
        if parent.name == "Engine":
            return parent
    raise RuntimeError(
        f"Unable to derive engine root from editor binary: {editor_binary}. "
        "Expected path under <Engine>/Binaries/..."
    )


def _parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--version-id", default="v001")
    parser.add_argument("--unreal-editor", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--engine-root", default="")
    parser.add_argument("--build-script", default="")
    parser.add_argument("--editor-binary", default="")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--overwrite-project-plugin", action="store_true")
    args = parser.parse_args()

    return BuildConfig(
        timestamp=_sanitize(str(args.timestamp)),
        version_id=_sanitize(str(args.version_id)),
        unreal_editor=str(args.unreal_editor).strip(),
        project=str(args.project).strip(),
        engine_root=str(args.engine_root).strip(),
        build_script=str(args.build_script).strip(),
        editor_binary=str(args.editor_binary).strip(),
        timeout_seconds=max(60, int(args.timeout_seconds)),
        overwrite_project_plugin=bool(args.overwrite_project_plugin),
    )


def _resolve_context(cfg: BuildConfig) -> BuildContext:
    root = _repo_root()
    ue_tooling_root = root / "UE_Tooling"

    default_project = str((root / "UE_Drone_Env/UE_Drone_Env.uproject").resolve())
    project = _resolve_existing_path(cfg.project or default_project, root, "project")

    if cfg.editor_binary:
        editor_binary = _resolve_existing_path(cfg.editor_binary, root, "editor_binary")
    elif cfg.unreal_editor:
        editor_binary = _resolve_existing_path(cfg.unreal_editor, root, "unreal_editor")
    else:
        default_editor = _default_unreal_editor()
        if default_editor is None:
            raise RuntimeError(
                "Unreal editor binary not found. Pass --editor-binary or --unreal-editor."
            )
        editor_binary = default_editor

    if cfg.unreal_editor:
        unreal_editor = _resolve_existing_path(cfg.unreal_editor, root, "unreal_editor")
    else:
        unreal_editor = editor_binary

    if cfg.engine_root:
        engine_root = _resolve_existing_path(cfg.engine_root, root, "engine_root")
    else:
        engine_root = _derive_engine_root(editor_binary)

    if cfg.build_script:
        build_script = _resolve_existing_path(cfg.build_script, root, "build_script")
    else:
        default_build_script = engine_root / "Build/BatchFiles/Mac/Build.sh"
        build_script = _resolve_existing_path(str(default_build_script), root, "build_script")

    websocket_runner = _resolve_existing_path(
        str(ue_tooling_root / "UE_Build/WebSocket/run_websocket_build.py"),
        root,
        "websocket_runner",
    )
    course_runner = _resolve_existing_path(
        str(ue_tooling_root / "UE_Build/Content_Generation/Course/run_course_build.py"),
        root,
        "course_runner",
    )
    drone_runner = _resolve_existing_path(
        str(ue_tooling_root / "UE_Build/Content_Generation/Drone/run_drone_build.py"),
        root,
        "drone_runner",
    )
    io_runner = _resolve_existing_path(
        str(ue_tooling_root / "UE_Build/Content_Generation/IO/run_io_build.py"),
        root,
        "io_runner",
    )
    low_level_executor = _resolve_existing_path(
        str(ue_tooling_root / "UE_Build/run_unreal_build_gen.py"),
        root,
        "low_level_executor",
    )

    stage_order = ["websocket", "course", "drone", "io"]
    python_bin = str(Path(sys.executable).resolve())

    websocket_run_name = f"{WEBSOCKET_KIT_NAME}_{cfg.version_id}_{cfg.timestamp}"
    course_run_name = f"{COURSE_NAME}_{cfg.version_id}_{cfg.timestamp}"
    drone_run_name = f"{DRONE_KIT_NAME}_{cfg.version_id}_{cfg.timestamp}"
    io_run_name = f"{IO_KIT_NAME}_{cfg.version_id}_{cfg.timestamp}"

    stage_plans = [
        StagePlan(
            name="websocket",
            runner_kind="host_subprocess",
            runner_script=str(websocket_runner),
            expected_artifact_path=str(
                ue_tooling_root / "Artifacts/websocket_build" / websocket_run_name / "run_manifest.json"
            ),
            command=[
                python_bin,
                str(websocket_runner),
                "--kit-name",
                WEBSOCKET_KIT_NAME,
                "--version-id",
                cfg.version_id,
                "--timestamp",
                cfg.timestamp,
                "--project",
                str(project),
                "--engine-root",
                str(engine_root),
                "--build-script",
                str(build_script),
                "--editor-binary",
                str(editor_binary),
                "--timeout-seconds",
                str(cfg.timeout_seconds),
            ]
            + (["--overwrite-project-plugin"] if cfg.overwrite_project_plugin else []),
            requires_authoritative_course_level=False,
        ),
        StagePlan(
            name="course",
            runner_kind="host_subprocess",
            runner_script=str(course_runner),
            expected_artifact_path=str(ue_tooling_root / "Artifacts/runs" / f"{course_run_name}.json"),
            command=[
                python_bin,
                str(course_runner),
                "--course-name",
                COURSE_NAME,
                "--version-id",
                cfg.version_id,
                "--timestamp",
                cfg.timestamp,
                "--timeout-seconds",
                str(cfg.timeout_seconds),
                "--unreal-editor",
                str(unreal_editor),
            ],
            requires_authoritative_course_level=False,
        ),
        StagePlan(
            name="drone",
            runner_kind="host_subprocess",
            runner_script=str(drone_runner),
            expected_artifact_path=str(ue_tooling_root / "Artifacts/drone_content" / f"{drone_run_name}.json"),
            command=[
                python_bin,
                str(drone_runner),
                "--kit-name",
                DRONE_KIT_NAME,
                "--version-id",
                cfg.version_id,
                "--timestamp",
                cfg.timestamp,
                "--level",
                AUTHORITATIVE_LEVEL_PLACEHOLDER,
                "--timeout-seconds",
                str(cfg.timeout_seconds),
                "--unreal-editor",
                str(unreal_editor),
                "--project",
                str(project),
            ],
            requires_authoritative_course_level=True,
        ),
        StagePlan(
            name="io",
            runner_kind="ue_python_entrypoint_via_executor",
            runner_script=str(io_runner),
            expected_artifact_path=str(ue_tooling_root / "Artifacts/io_build" / f"{io_run_name}.json"),
            command=[
                python_bin,
                str(low_level_executor),
                "--script",
                str(io_runner),
                "--timeout-seconds",
                str(cfg.timeout_seconds),
                "--unreal-editor",
                str(unreal_editor),
                "--project",
                str(project),
                "--",
                "--kit-name",
                IO_KIT_NAME,
                "--version-id",
                cfg.version_id,
                "--timestamp",
                cfg.timestamp,
                "--level",
                AUTHORITATIVE_LEVEL_PLACEHOLDER,
            ],
            requires_authoritative_course_level=True,
        ),
    ]

    return BuildContext(
        script_name=SCRIPT_NAME,
        script_version=SCRIPT_VERSION,
        script_date=SCRIPT_DATE,
        run_name=cfg.run_name,
        timestamp=cfg.timestamp,
        version_id=cfg.version_id,
        fail_fast=True,
        timeout_seconds=cfg.timeout_seconds,
        repo_root=str(root),
        project=str(project),
        engine_root=str(engine_root),
        build_script=str(build_script),
        editor_binary=str(editor_binary),
        unreal_editor=str(unreal_editor),
        overwrite_project_plugin=cfg.overwrite_project_plugin,
        stage_order=stage_order,
        stage_plans=stage_plans,
    )


def _manifest_path(ctx: BuildContext) -> Path:
    manifest_dir = Path(ctx.repo_root) / "UE_Tooling/Artifacts/build"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"{ctx.run_name}.json"


def _legacy_manifest_path(ctx: BuildContext) -> Path:
    manifest_dir = Path(ctx.repo_root) / "UE_Tooling/Artifacts/build_runs" / ctx.run_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / "run_manifest.json"


def _write_manifest(
    ctx: BuildContext,
    status: str,
    stage_executions: list[StageExecution],
    not_run_stages: list[dict],
    execution_scope: str,
    stopped_before_stage: str,
    authoritative_course_level_path: str,
    error: str,
    io_validation: dict,
) -> Path:
    manifest_path = _manifest_path(ctx)
    legacy_manifest_path = _legacy_manifest_path(ctx)

    stage_status_by_name = {stage.name: "not_run" for stage in ctx.stage_plans}
    stage_artifact_paths_expected = {
        stage.name: stage.expected_artifact_path for stage in ctx.stage_plans
    }
    stage_artifact_paths_actual = {stage.name: "" for stage in ctx.stage_plans}
    stage_commands_issued = {stage.name: [] for stage in ctx.stage_plans}
    for stage in stage_executions:
        stage_status_by_name[stage.name] = stage.status
        stage_artifact_paths_actual[stage.name] = stage.artifact_path
        stage_commands_issued[stage.name] = list(stage.command)

    io_placement_summary = {
        "status": "",
        "target_level": "",
        "target_level_matches_authoritative": False,
        "set_data_config": {},
        "sample_manager": {},
        "startup_drone_pawn": {},
    }
    if isinstance(io_validation, dict) and io_validation:
        required = io_validation.get("required_placements", {})
        io_placement_summary = {
            "status": io_validation.get("placements_status", ""),
            "target_level": io_validation.get("placements_target_level", ""),
            "target_level_matches_authoritative": bool(
                io_validation.get("placements_target_level_matches_authoritative", False)
            ),
            "set_data_config": required.get("set_data_config", {}),
            "sample_manager": required.get("sample_manager", {}),
            "startup_drone_pawn": required.get("startup_drone_pawn", {}),
        }

    payload = {
        "manifest_schema_version": "1.0",
        "manifest_path": str(manifest_path),
        "legacy_manifest_path": str(legacy_manifest_path),
        "build_run_id": ctx.run_name,
        "script_name": ctx.script_name,
        "script_version": ctx.script_version,
        "script_date": ctx.script_date,
        "run_name": ctx.run_name,
        "timestamp": ctx.timestamp,
        "version_id": ctx.version_id,
        "status": status,
        "execution_scope": execution_scope,
        "fail_fast": ctx.fail_fast,
        "stopped_before_stage": stopped_before_stage,
        "authoritative_course_level_path": authoritative_course_level_path,
        "io_validation": io_validation,
        "io_placement_summary": io_placement_summary,
        "stage_status_by_name": stage_status_by_name,
        "stage_artifact_paths_expected": stage_artifact_paths_expected,
        "stage_artifact_paths_actual": stage_artifact_paths_actual,
        "stage_commands_issued": stage_commands_issued,
        "websocket_status": stage_status_by_name.get("websocket", "not_run"),
        "course_status": stage_status_by_name.get("course", "not_run"),
        "drone_status": stage_status_by_name.get("drone", "not_run"),
        "io_status": stage_status_by_name.get("io", "not_run"),
        "runtime_handoff": {
            "authoritative_level_path": authoritative_course_level_path,
            "stage_status_by_name": stage_status_by_name,
            "stage_artifact_paths_actual": stage_artifact_paths_actual,
            "io_placement_summary": io_placement_summary,
        },
        "error": error,
        "timeout_seconds": ctx.timeout_seconds,
        "resolved_inputs": {
            "repo_root": ctx.repo_root,
            "project": ctx.project,
            "engine_root": ctx.engine_root,
            "build_script": ctx.build_script,
            "editor_binary": ctx.editor_binary,
            "unreal_editor": ctx.unreal_editor,
            "overwrite_project_plugin": ctx.overwrite_project_plugin,
        },
        "stage_order": ctx.stage_order,
        "stage_plans": [asdict(stage) for stage in ctx.stage_plans],
        "stage_executions": [asdict(stage) for stage in stage_executions],
        "not_run_stages": not_run_stages,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    legacy_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _stage_plan_by_name(ctx: BuildContext, name: str) -> StagePlan:
    for stage in ctx.stage_plans:
        if stage.name == name:
            return stage
    raise RuntimeError(f"Missing required stage plan: {name}")


def _remaining_stages(ctx: BuildContext, current_stage_name: str) -> list[str]:
    if current_stage_name not in ctx.stage_order:
        return []
    index = ctx.stage_order.index(current_stage_name)
    return ctx.stage_order[index + 1 :]


def _run_stage(stage: StagePlan) -> StageExecution:
    started = datetime.now(tz=timezone.utc)
    started_monotonic = time.monotonic()

    completed = subprocess.run(stage.command, capture_output=True, text=True)

    finished = datetime.now(tz=timezone.utc)
    duration_seconds = round(max(0.0, time.monotonic() - started_monotonic), 3)

    artifact_path = Path(stage.expected_artifact_path)
    artifact_exists = artifact_path.exists()
    artifact_status: Optional[str] = None
    error = ""

    if completed.returncode == 0:
        if not artifact_exists:
            error = f"Expected stage artifact missing: {artifact_path}"
        else:
            try:
                artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact_status = artifact_payload.get("status")
            except Exception as exc:
                error = f"Failed to parse stage artifact JSON: {exc}"
            else:
                if artifact_status != "success":
                    error = (
                        f"Stage artifact status is not success for {stage.name}: "
                        f"{artifact_status!r}"
                    )
    else:
        error = f"Runner exited non-zero with code {completed.returncode}"

    status = "success" if not error else "failed"

    return StageExecution(
        name=stage.name,
        status=status,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=duration_seconds,
        command=list(stage.command),
        runner_script=stage.runner_script,
        expected_artifact_path=stage.expected_artifact_path,
        artifact_path=str(artifact_path),
        artifact_exists=artifact_exists,
        artifact_status=artifact_status,
        return_code=int(completed.returncode),
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        error=error,
    )


def _extract_course_level_path(course_stage: StageExecution) -> str:
    artifact_path = Path(course_stage.artifact_path)
    if not artifact_path.exists():
        raise RuntimeError(f"Course artifact missing: {artifact_path}")

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Course artifact JSON: {exc}") from exc

    ue_paths = payload.get("ue_paths")
    if not isinstance(ue_paths, dict):
        raise RuntimeError("Course artifact missing `ue_paths` object.")
    level_path = ue_paths.get("level_path")
    if not isinstance(level_path, str) or not level_path.strip():
        raise RuntimeError("Course artifact missing non-empty `ue_paths.level_path`.")
    return level_path.strip()


def _resolve_command_authoritative_level(command: list[str], level_path: str) -> list[str]:
    resolved: list[str] = []
    for token in command:
        if token == AUTHORITATIVE_LEVEL_PLACEHOLDER:
            resolved.append(level_path)
        else:
            resolved.append(token)
    return resolved


def _stage_with_authoritative_level(stage: StagePlan, level_path: str) -> StagePlan:
    resolved_command = _resolve_command_authoritative_level(stage.command, level_path)

    if stage.requires_authoritative_course_level and AUTHORITATIVE_LEVEL_PLACEHOLDER in resolved_command:
        raise RuntimeError(
            f"Stage still contains unresolved authoritative-level placeholder: {stage.name}"
        )

    if stage.name == "drone":
        if "--level" not in resolved_command:
            raise RuntimeError("Drone stage command missing required --level handoff.")
        level_index = resolved_command.index("--level")
        if level_index + 1 >= len(resolved_command):
            raise RuntimeError("Drone stage command has --level without a value.")
        if resolved_command[level_index + 1] != level_path:
            raise RuntimeError(
                "Drone stage command level handoff mismatch with authoritative course level."
            )

    if stage.name == "io":
        if "--script" not in resolved_command:
            raise RuntimeError("IO stage command missing required --script binding.")
        script_index = resolved_command.index("--script")
        if script_index + 1 >= len(resolved_command):
            raise RuntimeError("IO stage command has --script without a value.")
        if resolved_command[script_index + 1] != stage.runner_script:
            raise RuntimeError("IO stage --script target does not match run_io_build.py path.")

        if "--" not in resolved_command:
            raise RuntimeError("IO stage command missing script-arg separator `--`.")
        separator_index = resolved_command.index("--")
        script_tail = resolved_command[separator_index + 1 :]
        if "--level" not in script_tail:
            raise RuntimeError("IO stage script args missing required --level handoff.")
        level_index = script_tail.index("--level")
        if level_index + 1 >= len(script_tail):
            raise RuntimeError("IO stage script args have --level without a value.")
        if script_tail[level_index + 1] != level_path:
            raise RuntimeError(
                "IO stage command level handoff mismatch with authoritative course level."
            )

    return StagePlan(
        name=stage.name,
        runner_kind=stage.runner_kind,
        runner_script=stage.runner_script,
        expected_artifact_path=stage.expected_artifact_path,
        command=resolved_command,
        requires_authoritative_course_level=stage.requires_authoritative_course_level,
    )


def _validate_io_artifact(io_stage: StageExecution, authoritative_level_path: str) -> tuple[bool, dict, str]:
    artifact_path = Path(io_stage.artifact_path)
    if not artifact_path.exists():
        return False, {}, f"IO artifact missing: {artifact_path}"

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, {}, f"Failed to parse IO artifact JSON: {exc}"

    target_level = payload.get("target_level")
    placements = payload.get("placements")
    placements_by_key = {}
    placements_status = ""
    placements_target_level = ""
    if isinstance(placements, dict):
        placements_by_key = placements.get("placements_by_key", {})
        placements_status = str(placements.get("status", ""))
        placements_target_level = str(placements.get("target_level", ""))

    validation = {
        "artifact_path": str(artifact_path),
        "artifact_status": payload.get("status"),
        "artifact_target_level": target_level,
        "artifact_target_level_matches_authoritative": target_level == authoritative_level_path,
        "placements_status": placements_status,
        "placements_target_level": placements_target_level,
        "placements_target_level_matches_authoritative": placements_target_level == authoritative_level_path,
        "required_placements": {},
    }

    if payload.get("status") != "success":
        return False, validation, f"IO artifact status is not success: {payload.get('status')!r}"
    if target_level != authoritative_level_path:
        return (
            False,
            validation,
            "IO artifact target_level does not match authoritative course level.",
        )
    if placements_status != "success":
        return (
            False,
            validation,
            f"IO placements status is not success: {placements_status!r}",
        )
    if placements_target_level != authoritative_level_path:
        return (
            False,
            validation,
            "IO placements.target_level does not match authoritative course level.",
        )
    if not isinstance(placements_by_key, dict):
        return False, validation, "IO artifact placements_by_key is missing or invalid."

    for placement_key, expected_label in IO_REQUIRED_PLACEMENTS.items():
        item = placements_by_key.get(placement_key)
        entry = {
            "present": isinstance(item, dict),
            "status": "",
            "actor_label": "",
            "expected_actor_label": expected_label,
            "target_level": "",
            "target_level_matches_authoritative": False,
        }
        if isinstance(item, dict):
            entry["status"] = str(item.get("status", ""))
            entry["actor_label"] = str(item.get("actor_label", ""))
            entry["target_level"] = str(item.get("target_level", ""))
            entry["target_level_matches_authoritative"] = entry["target_level"] == authoritative_level_path
        validation["required_placements"][placement_key] = entry

        if not entry["present"]:
            return False, validation, f"IO placement record missing for {placement_key}."
        if entry["status"] != "success":
            return False, validation, f"IO placement status is not success for {placement_key}."
        if entry["actor_label"] != expected_label:
            return (
                False,
                validation,
                f"IO placement actor label mismatch for {placement_key}: {entry['actor_label']!r}",
            )
        if not entry["target_level_matches_authoritative"]:
            return (
                False,
                validation,
                f"IO placement target level mismatch for {placement_key}.",
            )

    return True, validation, ""


def main() -> int:
    cfg = _parse_args()
    try:
        ctx = _resolve_context(cfg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        websocket_stage = _stage_plan_by_name(ctx, "websocket")
        course_stage = _stage_plan_by_name(ctx, "course")
        drone_stage = _stage_plan_by_name(ctx, "drone")
        io_stage = _stage_plan_by_name(ctx, "io")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    websocket_execution = _run_stage(websocket_stage)
    after_websocket = _remaining_stages(ctx, websocket_stage.name)
    if websocket_execution.status != "success":
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=[websocket_execution],
            not_run_stages=[
                {"name": stage, "reason": "blocked_by_websocket_stage_failure"} for stage in after_websocket
            ],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage=after_websocket[0] if after_websocket else "",
            authoritative_course_level_path="",
            error=websocket_execution.error,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": websocket_execution.name,
                    "manifest_path": str(manifest_path),
                    "error": websocket_execution.error,
                },
                sort_keys=True,
            )
        )
        return 1

    course_execution = _run_stage(course_stage)
    after_course = _remaining_stages(ctx, course_stage.name)
    stage_executions = [websocket_execution, course_execution]
    if course_execution.status != "success":
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[
                {"name": stage, "reason": "blocked_by_course_stage_failure"} for stage in after_course
            ],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage=after_course[0] if after_course else "",
            authoritative_course_level_path="",
            error=course_execution.error,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": course_execution.name,
                    "manifest_path": str(manifest_path),
                    "error": course_execution.error,
                },
                sort_keys=True,
            )
        )
        return 1

    try:
        authoritative_course_level_path = _extract_course_level_path(course_execution)
    except RuntimeError as exc:
        error_message = str(exc)
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[
                    {"name": stage, "reason": "blocked_by_course_artifact_validation_failure"}
                    for stage in after_course
                ],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage=after_course[0] if after_course else "",
            authoritative_course_level_path="",
            error=error_message,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": course_execution.name,
                    "manifest_path": str(manifest_path),
                    "error": error_message,
                },
                sort_keys=True,
            )
        )
        return 1

    try:
        resolved_drone_stage = _stage_with_authoritative_level(drone_stage, authoritative_course_level_path)
    except RuntimeError as exc:
        error_message = str(exc)
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[
                {"name": stage, "reason": "blocked_by_drone_level_handoff_failure"}
                for stage in after_course
            ],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage=drone_stage.name,
            authoritative_course_level_path=authoritative_course_level_path,
            error=error_message,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": drone_stage.name,
                    "manifest_path": str(manifest_path),
                    "error": error_message,
                },
                sort_keys=True,
            )
        )
        return 1

    drone_execution = _run_stage(resolved_drone_stage)
    after_drone = _remaining_stages(ctx, resolved_drone_stage.name)
    stage_executions = [websocket_execution, course_execution, drone_execution]
    if drone_execution.status != "success":
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[
                {"name": stage, "reason": "blocked_by_drone_stage_failure"} for stage in after_drone
            ],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage=after_drone[0] if after_drone else "",
            authoritative_course_level_path=authoritative_course_level_path,
            error=drone_execution.error,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": drone_execution.name,
                    "manifest_path": str(manifest_path),
                    "error": drone_execution.error,
                },
                sort_keys=True,
            )
        )
        return 1

    try:
        resolved_io_stage = _stage_with_authoritative_level(io_stage, authoritative_course_level_path)
    except RuntimeError as exc:
        error_message = str(exc)
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[
                {"name": stage, "reason": "blocked_by_io_level_handoff_failure"} for stage in after_drone
            ],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage=io_stage.name,
            authoritative_course_level_path=authoritative_course_level_path,
            error=error_message,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": io_stage.name,
                    "manifest_path": str(manifest_path),
                    "error": error_message,
                },
                sort_keys=True,
            )
        )
        return 1

    io_execution = _run_stage(resolved_io_stage)
    stage_executions = [websocket_execution, course_execution, drone_execution, io_execution]
    if io_execution.status != "success":
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage="",
            authoritative_course_level_path=authoritative_course_level_path,
            error=io_execution.error,
            io_validation={},
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": io_execution.name,
                    "manifest_path": str(manifest_path),
                    "error": io_execution.error,
                },
                sort_keys=True,
            )
        )
        return 1

    io_ok, io_validation, io_error = _validate_io_artifact(io_execution, authoritative_course_level_path)
    if not io_ok:
        manifest_path = _write_manifest(
            ctx=ctx,
            status="failed",
            stage_executions=stage_executions,
            not_run_stages=[],
            execution_scope="step7_full_build_through_io",
            stopped_before_stage="",
            authoritative_course_level_path=authoritative_course_level_path,
            error=io_error,
            io_validation=io_validation,
        )
        print(
            json.dumps(
                {
                    "script_name": SCRIPT_NAME,
                    "run_name": ctx.run_name,
                    "status": "failed",
                    "failed_stage": io_execution.name,
                    "manifest_path": str(manifest_path),
                    "error": io_error,
                },
                sort_keys=True,
            )
        )
        return 1

    manifest_path = _write_manifest(
        ctx=ctx,
        status="io_stage_success",
        stage_executions=stage_executions,
        not_run_stages=[],
        execution_scope="step7_full_build_through_io",
        stopped_before_stage="",
        authoritative_course_level_path=authoritative_course_level_path,
        error="",
        io_validation=io_validation,
    )
    print(
        json.dumps(
            {
                "script_name": SCRIPT_NAME,
                "run_name": ctx.run_name,
                "status": "io_stage_success",
                "manifest_path": str(manifest_path),
                "websocket_stage_artifact_path": websocket_execution.artifact_path,
                "course_stage_artifact_path": course_execution.artifact_path,
                "drone_stage_artifact_path": drone_execution.artifact_path,
                "io_stage_artifact_path": io_execution.artifact_path,
                "authoritative_course_level_path": authoritative_course_level_path,
                "stage_order": ctx.stage_order,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
