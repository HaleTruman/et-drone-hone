"""
Desired behavior:
- Run the canonical websocket bootstrap/build flow in ordered steps and record a versioned artifact for the run.
- Treat bootstrap + validation as one reproducible generation pass for the UE websocket plugin state.
- Preserve explicit per-step logs, script metadata, and pass/fail status for review.

Interfaces:
- Invokes `bootstrap_plugin.py` and `plugin_validate.py` in order.
- Writes artifacts to `UE_Tooling/Artifacts/websocket_build`.
- Uses `websocket_asset_assembly.py` metadata conventions for stable reporting.

Assumptions:
- `bootstrap_plugin.py` owns staging and project build invocation.
- `plugin_validate.py` is the required readiness validation pass.
- This runner is for versioning/logging the websocket build workflow, not runtime websocket traffic.

Success conditions:
- Ordered websocket generation steps execute successfully.
- A run artifact captures script versions, timestamps, step results, build/validation logs, and chosen overwrite policy.
- Failures are explicit and stop the run unless continue-on-failure is requested.
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

from websocket_asset_assembly import RESULT_MARKER, extract_script_metadata, repo_root, write_json

SCRIPT_NAME = "run_websocket_generation"
SCRIPT_VERSION = "2.0.0"
SCRIPT_DATE = "2026-03-14"


@dataclass(frozen=True)
class RunConfig:
    kit_name: str
    version_id: str
    timestamp: str
    plugin_name: str
    overwrite_project_plugin: bool
    continue_on_failure: bool
    project: str
    engine_root: str
    build_script: str
    editor_binary: str
    timeout_seconds: int
    startup_map: str

    @property
    def run_name(self) -> str:
        return f"{self.kit_name}_{self.version_id}_{self.timestamp}"


@dataclass
class StepExecution:
    script_name: str
    script_path: str
    log_path: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    return_code: int
    status: str
    stdout: str
    stderr: str
    results: list[dict]
    metadata: dict


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-name", default="WebSocketPlugin")
    parser.add_argument("--version-id", default="v001")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--plugin-name", default="DroneWebSocket")
    parser.add_argument("--overwrite-project-plugin", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--project", default="")
    parser.add_argument("--engine-root", default="")
    parser.add_argument("--build-script", default="")
    parser.add_argument("--editor-binary", default="")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--startup-map", default="/Engine/Maps/Entry")
    args = parser.parse_args()
    return RunConfig(
        kit_name=_sanitize(str(args.kit_name)),
        version_id=_sanitize(str(args.version_id)),
        timestamp=_sanitize(str(args.timestamp)),
        plugin_name=str(args.plugin_name),
        overwrite_project_plugin=bool(args.overwrite_project_plugin),
        continue_on_failure=bool(args.continue_on_failure),
        project=str(args.project),
        engine_root=str(args.engine_root),
        build_script=str(args.build_script),
        editor_binary=str(args.editor_binary),
        timeout_seconds=max(60, int(args.timeout_seconds)),
        startup_map=str(args.startup_map),
    )


def _parse_results(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    parsed: list[dict] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if RESULT_MARKER not in line:
            continue
        payload = line.split(RESULT_MARKER, 1)[1].strip()
        try:
            parsed.append(json.loads(payload))
        except json.JSONDecodeError:
            parsed.append({"parse_error": True, "raw": payload})
    return parsed


def _build_step_command(script_path: Path, cfg: RunConfig) -> list[str]:
    command = [sys.executable, str(script_path), "--plugin-name", cfg.plugin_name]
    if cfg.project:
        command.extend(["--project", cfg.project])
    if cfg.engine_root:
        command.extend(["--engine-root", cfg.engine_root])
    if script_path.name == "bootstrap_plugin.py":
        if cfg.build_script:
            command.extend(["--build-script", cfg.build_script])
        if cfg.overwrite_project_plugin:
            command.append("--overwrite")
    if script_path.name == "plugin_validate.py":
        if cfg.editor_binary:
            command.extend(["--editor-binary", cfg.editor_binary])
        command.extend(["--startup-map", cfg.startup_map])
    command.extend(["--timeout-seconds", str(cfg.timeout_seconds)])
    return command


def _run_step(script_path: Path, log_path: Path, cfg: RunConfig) -> StepExecution:
    started = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    command = _build_step_command(script_path, cfg)
    completed = subprocess.run(command, capture_output=True, text=True)
    finished = datetime.now(timezone.utc)
    duration = max(0.0, time.monotonic() - start_monotonic)

    combined_output = []
    if completed.stdout:
        combined_output.append(completed.stdout.rstrip())
    if completed.stderr:
        combined_output.append(completed.stderr.rstrip())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(combined_output) + ("\n" if combined_output else ""), encoding="utf-8")

    return StepExecution(
        script_name=script_path.stem,
        script_path=str(script_path),
        log_path=str(log_path),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=round(duration, 3),
        return_code=int(completed.returncode),
        status="success" if completed.returncode == 0 else "failed",
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        results=_parse_results(log_path),
        metadata=extract_script_metadata(script_path),
    )


def main() -> int:
    cfg = _parse_args()
    root = repo_root()
    websocket_root = root / "UE_Tooling/UE_Build/WebSocket"
    scripts = [
        websocket_root / "bootstrap_plugin.py",
        websocket_root / "plugin_validate.py",
    ]

    missing = [str(path) for path in scripts if not path.exists()]
    if missing:
        print("ERROR: Missing websocket generation scripts:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    logs_dir = root / "UE_Tooling/Artifacts/websocket_build/logs" / cfg.run_name
    artifact_dir = root / "UE_Tooling/Artifacts/websocket_build" / cfg.run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    executions: list[StepExecution] = []
    overall_status = "success"
    for script_path in scripts:
        execution = _run_step(script_path=script_path, log_path=logs_dir / f"{script_path.stem}.log", cfg=cfg)
        executions.append(execution)
        if execution.return_code != 0:
            overall_status = "failed"
            if not cfg.continue_on_failure:
                break

    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "script_date": SCRIPT_DATE,
        "run_name": cfg.run_name,
        "kit_name": cfg.kit_name,
        "version_id": cfg.version_id,
        "timestamp": cfg.timestamp,
        "plugin_name": cfg.plugin_name,
        "overwrite_project_plugin": cfg.overwrite_project_plugin,
        "continue_on_failure": cfg.continue_on_failure,
        "project": cfg.project,
        "engine_root": cfg.engine_root,
        "build_script": cfg.build_script,
        "editor_binary": cfg.editor_binary,
        "timeout_seconds": cfg.timeout_seconds,
        "startup_map": cfg.startup_map,
        "status": overall_status,
        "steps": [asdict(execution) for execution in executions],
    }

    write_json(artifact_dir / "run_manifest.json", manifest)
    print(json.dumps({"run_name": cfg.run_name, "status": overall_status, "artifact_dir": str(artifact_dir)}, sort_keys=True))
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
