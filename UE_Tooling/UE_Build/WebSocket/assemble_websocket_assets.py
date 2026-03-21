"""
Status: in development.
Renamed from `websocket_asset_assembly.py` to `assemble_websocket_assets.py` to
mirror the Course/Drone/IO/WebSocket build-layer naming pattern.

Desired behavior:
- Provide one shared helper module for websocket plugin bootstrap/build and validation scripts.
- Keep plugin path resolution, source staging, project build invocation, validation inputs, and result logging consistent.
- Make it easy to evolve the websocket plugin source tree while keeping startup-readiness checks deterministic.

Interfaces:
- Used by `bootstrap_plugin.py`, `plugin_validate.py`, and `run_websocket_build.py`.
- Operates on `UE_Tooling/UE_Build/WebSocket/Plugin_Source/<PluginName>` as the source-of-truth.
- Stages and builds against `UE_Drone_Env/Plugins/<PluginName>`.

Assumptions:
- The websocket plugin source-of-truth lives in `Plugin_Source/<PluginName>`.
- The minimum required plugin files are the currently known transport/protocol/handshake files.
- Validation should compare source-of-truth, project install, descriptor compatibility, project build outputs, and headless startup readiness.

Success conditions:
- Bootstrap/build scripts can stage source into the UE project and invoke a UE-side build deterministically.
- Validation scripts can assess whether the project plugin is structurally aligned and startup-ready.
- Shared helpers emit stable metadata for runner artifacts and troubleshooting.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_NAME = "assemble_websocket_assets"
SCRIPT_VERSION = "2.1.0"
SCRIPT_DATE = "2026-03-15"

RESULT_MARKER = "[WebSocketBuildResult]"
DEFAULT_PLUGIN_NAME = "DroneWebSocket"
DEFAULT_PROJECT_PATH = "UE_Drone_Env/UE_Drone_Env.uproject"
DEFAULT_STARTUP_MAP = "/Engine/Maps/Entry"
STARTUP_PROBE_MARKER = "WEBSOCKET_PLUGIN_VALIDATE_STARTUP_OK"

IGNORED_STAGE_DIRS = {"Binaries", "Intermediate", "Saved", "__pycache__"}
IGNORED_STAGE_FILES = {".DS_Store"}


@dataclass(frozen=True)
class PluginPaths:
    plugin_name: str
    repo_root: Path
    tooling_root: Path
    project_file: Path
    project_name: str
    project_editor_target: str
    source_root: Path
    project_root: Path
    engine_root: Path
    build_script: Path
    editor_binary: Path
    host_platform: str


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Failed to locate repo root.")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def log(message: str) -> None:
    print(f"[{SCRIPT_NAME}] {message}")


def emit_result(script_name: str, status: str, details: dict | None = None) -> None:
    payload = {
        "script_name": script_name,
        "status": status,
        "timestamp_utc": utc_now_iso(),
        "details": details or {},
    }
    print(f"{RESULT_MARKER} {json.dumps(payload, sort_keys=True)}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def extract_script_metadata(script_path: Path) -> dict:
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


def project_file(default_path: str | None = None) -> Path:
    root = repo_root()
    relative = default_path or DEFAULT_PROJECT_PATH
    project = root / relative
    require(project.exists(), f"Project file is missing: {project}")
    return project


def _engine_association(project_descriptor: dict[str, Any]) -> str:
    association = str(project_descriptor.get("EngineAssociation") or "").strip()
    require(association, "Project .uproject is missing EngineAssociation.")
    return association


def _default_engine_root(project_descriptor: dict[str, Any]) -> Path:
    association = _engine_association(project_descriptor)
    return Path(f"/Users/Shared/Epic Games/UE_{association}")


def _default_build_script(engine_root: Path) -> Path:
    return engine_root / "Engine/Build/BatchFiles/Mac/Build.sh"


def _default_editor_binary(engine_root: Path) -> Path:
    return engine_root / "Engine/Binaries/Mac/UnrealEditor"


def _host_platform_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Mac"
    if system == "Linux":
        return "Linux"
    if system == "Windows":
        return "Win64"
    raise RuntimeError(f"Unsupported host platform: {system}")


def plugin_paths(
    *,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
    project_override: str | None = None,
    engine_root_override: str | None = None,
    build_script_override: str | None = None,
    editor_binary_override: str | None = None,
) -> PluginPaths:
    root = repo_root()
    tooling_root = root / "UE_Tooling/UE_Build/WebSocket"
    project = Path(project_override).resolve() if project_override else project_file()
    descriptor = read_json(project)
    engine_root = Path(engine_root_override).resolve() if engine_root_override else _default_engine_root(descriptor)
    build_script = Path(build_script_override).resolve() if build_script_override else _default_build_script(engine_root)
    editor_binary = Path(editor_binary_override).resolve() if editor_binary_override else _default_editor_binary(engine_root)
    project_name = project.stem

    return PluginPaths(
        plugin_name=plugin_name,
        repo_root=root,
        tooling_root=tooling_root,
        project_file=project,
        project_name=project_name,
        project_editor_target=f"{project_name}Editor",
        source_root=tooling_root / "Plugin_Source" / plugin_name,
        project_root=root / "UE_Drone_Env/Plugins" / plugin_name,
        engine_root=engine_root,
        build_script=build_script,
        editor_binary=editor_binary,
        host_platform=_host_platform_name(),
    )


def required_relative_paths(plugin_name: str) -> tuple[str, ...]:
    return (
        f"{plugin_name}.uplugin",
        "Config/FilterPlugin.ini",
        f"Source/{plugin_name}/{plugin_name}.Build.cs",
        f"Source/{plugin_name}/Public/WSClientComponent.h",
        f"Source/{plugin_name}/Public/WSConfigHandshakeActor.h",
        f"Source/{plugin_name}/Public/WSProtocolTypes.h",
        f"Source/{plugin_name}/Private/DroneWebSocketModule.cpp",
        f"Source/{plugin_name}/Private/WSClientComponent.cpp",
        f"Source/{plugin_name}/Private/WSConfigHandshakeActor.cpp",
        f"Source/{plugin_name}/Private/WSProtocolTypes.cpp",
    )


def expected_binary_relative_paths(plugin_name: str, host_platform: str) -> tuple[str, ...]:
    if host_platform == "Mac":
        return (
            f"Binaries/Mac/UnrealEditor-{plugin_name}.dylib",
            "Binaries/Mac/UnrealEditor.modules",
        )
    if host_platform == "Linux":
        return (
            f"Binaries/Linux/libUnrealEditor-{plugin_name}.so",
            "Binaries/Linux/UnrealEditor.modules",
        )
    if host_platform == "Win64":
        return (
            f"Binaries/Win64/UnrealEditor-{plugin_name}.dll",
            "Binaries/Win64/UnrealEditor.modules",
        )
    raise RuntimeError(f"Unsupported host platform: {host_platform}")


def validate_required_files(plugin_root: Path, plugin_name: str) -> dict[str, Any]:
    missing = [relative for relative in required_relative_paths(plugin_name) if not (plugin_root / relative).exists()]
    return {
        "plugin_root": str(plugin_root),
        "exists": plugin_root.exists(),
        "is_directory": plugin_root.is_dir(),
        "missing_required_paths": missing,
        "is_valid": plugin_root.exists() and plugin_root.is_dir() and not missing,
    }


def validate_expected_binaries(plugin_root: Path, plugin_name: str, host_platform: str) -> dict[str, Any]:
    expected = expected_binary_relative_paths(plugin_name, host_platform)
    missing = [relative for relative in expected if not (plugin_root / relative).exists()]
    return {
        "plugin_root": str(plugin_root),
        "host_platform": host_platform,
        "expected_binary_paths": list(expected),
        "missing_binary_paths": missing,
        "is_valid": not missing,
    }


def _is_ignored_relative_path(relative_path: Path) -> bool:
    parts = set(relative_path.parts)
    if parts & IGNORED_STAGE_DIRS:
        return True
    return relative_path.name in IGNORED_STAGE_FILES


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_snapshot(plugin_root: Path) -> dict[str, str]:
    if not plugin_root.exists():
        return {}
    snapshot: dict[str, str] = {}
    for path in sorted(plugin_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(plugin_root)
        if _is_ignored_relative_path(relative):
            continue
        snapshot[str(relative)] = file_hash(path)
    return snapshot


def compare_source_snapshots(source_root: Path, project_root: Path) -> dict[str, Any]:
    source_snapshot = collect_source_snapshot(source_root)
    project_snapshot = collect_source_snapshot(project_root)
    missing_in_project = sorted(set(source_snapshot) - set(project_snapshot))
    unexpected_in_project = sorted(set(project_snapshot) - set(source_snapshot))
    mismatched_hashes = sorted(
        relative
        for relative in set(source_snapshot).intersection(project_snapshot)
        if source_snapshot[relative] != project_snapshot[relative]
    )
    return {
        "source_file_count": len(source_snapshot),
        "project_file_count": len(project_snapshot),
        "missing_in_project": missing_in_project,
        "unexpected_in_project": unexpected_in_project,
        "mismatched_hashes": mismatched_hashes,
        "is_in_sync": not missing_in_project and not unexpected_in_project and not mismatched_hashes,
    }


def plugin_descriptor_validation(plugin_root: Path, plugin_name: str, project_descriptor: dict[str, Any]) -> dict[str, Any]:
    descriptor_path = plugin_root / f"{plugin_name}.uplugin"
    require(descriptor_path.exists(), f"Plugin descriptor is missing: {descriptor_path}")
    descriptor = read_json(descriptor_path)

    module_names = [str(module.get("Name")) for module in descriptor.get("Modules", []) if isinstance(module, dict)]
    declared_engine = str(descriptor.get("EngineVersion") or "").strip()
    project_engine = _engine_association(project_descriptor)

    if declared_engine:
        compatible = declared_engine.startswith(project_engine)
        reason = (
            f"Descriptor EngineVersion {declared_engine} matches project EngineAssociation {project_engine}."
            if compatible
            else f"Descriptor EngineVersion {declared_engine} does not match project EngineAssociation {project_engine}."
        )
    else:
        compatible = True
        reason = f"Descriptor does not declare EngineVersion; treating project EngineAssociation {project_engine} as authoritative."

    return {
        "descriptor_path": str(descriptor_path),
        "project_engine_association": project_engine,
        "declared_engine_version": declared_engine or None,
        "module_names": module_names,
        "required_module_present": plugin_name in module_names,
        "engine_compatible": compatible,
        "reason": reason,
        "is_valid": compatible and plugin_name in module_names,
    }


def ensure_bootstrap_inputs(paths: PluginPaths) -> dict[str, Any]:
    result = {
        "source_root_exists": paths.source_root.exists(),
        "source_root_is_directory": paths.source_root.is_dir(),
        "project_file_exists": paths.project_file.exists(),
        "build_script_exists": paths.build_script.exists(),
    }
    require(result["source_root_exists"], f"Plugin source root is missing: {paths.source_root}")
    require(result["source_root_is_directory"], f"Plugin source root is not a directory: {paths.source_root}")
    require(result["project_file_exists"], f"Project file is missing: {paths.project_file}")
    require(result["build_script_exists"], f"Build script is missing: {paths.build_script}")
    return result


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def stage_source_in_project(source_root: Path, project_root: Path, overwrite: bool) -> str:
    if project_root.exists() and not overwrite:
        return "kept_existing"

    if project_root.exists():
        remove_tree(project_root)

    project_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_root,
        project_root,
        dirs_exist_ok=False,
        ignore=shutil.ignore_patterns(*IGNORED_STAGE_DIRS, *IGNORED_STAGE_FILES),
    )
    return "rebuilt_project_plugin" if overwrite else "built_missing_plugin"


def run_command(command: list[str], *, timeout_seconds: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        return_code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nCommand timed out after {timeout_seconds} seconds."
        timed_out = True
    duration = max(0.0, time.monotonic() - started)
    return {
        "command": command,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_seconds": round(duration, 3),
        "timed_out": timed_out,
    }


def build_project_plugin(paths: PluginPaths, *, timeout_seconds: int) -> dict[str, Any]:
    command = [
        str(paths.build_script),
        paths.project_editor_target,
        paths.host_platform,
        "Development",
        str(paths.project_file),
        "-WaitMutex",
        "-NoHotReloadFromIDE",
    ]
    result = run_command(command, timeout_seconds=timeout_seconds)
    result["success"] = result["return_code"] == 0 and not result["timed_out"]
    return result


def startup_probe(paths: PluginPaths, *, timeout_seconds: int, startup_map: str = DEFAULT_STARTUP_MAP) -> dict[str, Any]:
    require(paths.editor_binary.exists(), f"UnrealEditor binary is missing: {paths.editor_binary}")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write("import unreal\n")
        handle.write(f'unreal.log("{STARTUP_PROBE_MARKER}")\n')
        probe_script = Path(handle.name)

    userdir = paths.repo_root / "UE_Drone_Env/Saved/WebSocketValidationUserDir" / datetime.now().strftime("%Y%m%d_%H%M%S")
    userdir.mkdir(parents=True, exist_ok=True)
    editor_log_path = userdir / "Saved/Logs" / f"{paths.project_name}.log"
    stdout_log_path = userdir / "Saved/Logs" / f"{paths.project_name}.startup_probe_stdout.log"
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(paths.editor_binary),
        str(paths.project_file),
        startup_map,
        "-unattended",
        "-nop4",
        "-nosplash",
        "-nullrhi",
        "-nosound",
        "-forcelogflush",
        "-stdout",
        "-FullStdOutLogOutput",
        "-AllowStdOutLogVerbosity",
        "-CrashForUAT",
        "-ScriptErrorsAreFatal",
        f"-userdir={userdir}",
        f"-ExecutePythonScript={probe_script}",
    ]

    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    timed_out = False
    return_code = 1
    try:
        with stdout_log_path.open("w", encoding="utf-8") as stdout_handle:
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=subprocess.STDOUT, text=True)
            try:
                return_code = int(process.wait(timeout=timeout_seconds))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    return_code = int(process.wait(timeout=30))
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = int(process.wait(timeout=30))
    finally:
        probe_script.unlink(missing_ok=True)

    duration = max(0.0, time.monotonic() - started)
    stdout_text = stdout_log_path.read_text(encoding="utf-8", errors="ignore") if stdout_log_path.exists() else ""
    editor_log_text = editor_log_path.read_text(encoding="utf-8", errors="ignore") if editor_log_path.exists() else ""
    marker_seen = STARTUP_PROBE_MARKER in stdout_text or STARTUP_PROBE_MARKER in editor_log_text
    return {
        "command": command,
        "return_code": return_code,
        "duration_seconds": round(duration, 3),
        "timed_out": timed_out,
        "startup_map": startup_map,
        "userdir": str(userdir),
        "stdout_log_path": str(stdout_log_path),
        "editor_log_path": str(editor_log_path),
        "stdout_tail": stdout_text[-4000:] if stdout_text else "",
        "editor_log_tail": editor_log_text[-4000:] if editor_log_text else "",
        "stdout_length": len(stdout_text),
        "editor_log_length": len(editor_log_text),
        "marker_seen": marker_seen,
        "success": return_code == 0 and not timed_out and marker_seen,
    }
