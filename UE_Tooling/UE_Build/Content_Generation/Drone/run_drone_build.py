"""
Status: in development.
Renamed from `run_drone_content_generation.py` to `run_drone_build.py` to
mirror the Course/Drone/IO/WebSocket orchestration layer naming pattern.

Desired behavior:
- Run all Drone_Content generation scripts in strict dependency order through headless Unreal.
- Fail fast on generator errors while preserving explicit per-script logs and artifact metadata.
- Emit a complete run artifact containing script versions, statuses, logs, and generator execution results.

Interfaces:
- Invokes `UE_Tooling/UE_Build/run_unreal_build_gen.py` for each generator script.
- Consumes scripts under `UE_Tooling/UE_Build/Content_Generation/Drone`.
- Writes run artifacts to `UE_Tooling/Artifacts/drone_content`.

Assumptions:
- Unreal editor binary/project are available via `UE_Tooling/UE_Build/run_unreal_build_gen.py`.
- Generation scripts use `[DroneContentResult]` log records for asset-level reporting.

Success conditions:
- All ordered generators execute successfully in headless Unreal.
- Failures are explicit and stop the run by default.
- Artifact JSON captures run-level and script-level validation details for review.
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

SCRIPT_NAME = "run_drone_build"
SCRIPT_VERSION = "1.5.0"
SCRIPT_DATE = "2026-03-15"

ORDERED_GENERATORS = [
    "Structs/gen_st_dronecommandnormalized.py",
    "Structs/gen_st_droneviewpointsnapshot.py",
    "Structs/gen_st_dronetelemetrysnapshot.py",
    "Interfaces/gen_bpi_dronecommandreceiver.py",
    "Interfaces/gen_bpi_droneviewpointprovider.py",
    "Interfaces/gen_bpi_dronetelemetryprovider.py",
    "Data/gen_da_dronemovementdefault.py",
    "Data/gen_da_sensorrigprofiledefault.py",
    "Materials/gen_m_dronebody_base.py",
    "Materials/gen_mi_dronebody_default.py",
    "Meshes/gen_sm_dronebody.py",
    "Meshes/gen_sm_dronecollisionproxy.py",
    "Drone_Blueprints/gen_bp_dronemovement_6dof.py",
    "Drone_Blueprints/gen_bp_dronesensors.py",
    "Drone_Blueprints/gen_bp_dronetelemetrysampler.py",
    "Drone_Blueprints/gen_bp_dronepawn.py",
]

@dataclass(frozen=True)
class RunConfig:
    kit_name: str
    version_id: str
    timestamp: str
    level_path: str
    timeout_seconds: int
    continue_on_failure: bool
    unreal_editor: str
    project: str

    @property
    def run_name(self) -> str:
        return f"{self.kit_name}_{self.version_id}_{self.timestamp}"


@dataclass
class GeneratorExecution:
    relative_script_path: str
    absolute_script_path: str
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


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Failed to locate repo root (.git not found).")


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-name", default="DroneContentKit")
    parser.add_argument("--version-id", default="v001")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--level", default="/Game/Course_Content/Maps/L_CourseTorus")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--unreal-editor", default="")
    parser.add_argument("--project", default="")
    args = parser.parse_args()

    return RunConfig(
        kit_name=_sanitize(str(args.kit_name)),
        version_id=_sanitize(str(args.version_id)),
        timestamp=_sanitize(str(args.timestamp)),
        level_path=str(args.level),
        timeout_seconds=max(60, int(args.timeout_seconds)),
        continue_on_failure=bool(args.continue_on_failure),
        unreal_editor=str(args.unreal_editor),
        project=str(args.project),
    )


def _extract_script_metadata(script_path: Path) -> dict:
    content = script_path.read_text(encoding="utf-8")

    def extract(name: str) -> Optional[str]:
        match = re.search(rf'^{name}\s*=\s*"([^"]+)"', content, re.MULTILINE)
        return match.group(1) if match else None

    return {
        "path": str(script_path),
        "script_name": extract("SCRIPT_NAME"),
        "script_version": extract("SCRIPT_VERSION"),
        "script_date": extract("SCRIPT_DATE"),
        "file_mtime_utc": datetime.fromtimestamp(script_path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def _parse_result_lines(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    results: list[dict] = []
    marker = "[DroneContentResult]"
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        payload_raw = line.split(marker, 1)[1].strip()
        try:
            results.append(json.loads(payload_raw))
        except json.JSONDecodeError:
            results.append({"raw": payload_raw, "parse_error": True})
    return results


def _run_single_generator(
    run_cfg: RunConfig,
    run_unreal_script: Path,
    script_path: Path,
    log_path: Path,
) -> GeneratorExecution:
    started = datetime.now(tz=timezone.utc)
    start_monotonic = time.monotonic()

    cmd = [
        sys.executable,
        str(run_unreal_script),
        "--script",
        str(script_path),
        "--log",
        str(log_path),
        "--level",
        run_cfg.level_path,
        "--timeout-seconds",
        str(run_cfg.timeout_seconds),
    ]
    if run_cfg.unreal_editor:
        cmd.extend(["--unreal-editor", run_cfg.unreal_editor])
    if run_cfg.project:
        cmd.extend(["--project", run_cfg.project])

    completed = subprocess.run(cmd, capture_output=True, text=True)

    finished = datetime.now(tz=timezone.utc)
    duration = max(0.0, time.monotonic() - start_monotonic)
    status = "success" if completed.returncode == 0 else "failed"

    return GeneratorExecution(
        relative_script_path=str(script_path.relative_to(script_path.parents[1])),
        absolute_script_path=str(script_path),
        log_path=str(log_path),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=round(duration, 3),
        return_code=int(completed.returncode),
        status=status,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        results=_parse_result_lines(log_path),
        metadata=_extract_script_metadata(script_path),
    )


def _write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main() -> int:
    cfg = _parse_args()
    repo_root = _repo_root()
    drone_content_root = repo_root / "UE_Tooling/UE_Build/Content_Generation/Drone"
    run_unreal_script = repo_root / "UE_Tooling/UE_Build/run_unreal_build_gen.py"
    logs_dir = repo_root / "UE_Tooling/Artifacts/drone_content/logs" / cfg.run_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    ordered_paths = [drone_content_root / rel_path for rel_path in ORDERED_GENERATORS]
    missing = [str(path) for path in ordered_paths if not path.exists()]
    if missing:
        print("ERROR: Missing generator scripts:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    if not run_unreal_script.exists():
        print(f"ERROR: Missing runner: {run_unreal_script}", file=sys.stderr)
        return 2

    created_at = datetime.now(tz=timezone.utc)
    executions: list[GeneratorExecution] = []
    overall_status = "success"

    for index, script_path in enumerate(ordered_paths, start=1):
        log_path = logs_dir / f"{index:02d}_{script_path.stem}.log"
        execution = _run_single_generator(cfg, run_unreal_script, script_path, log_path)
        executions.append(execution)
        print(f"[{index}/{len(ordered_paths)}] {script_path.name}: {execution.status} (rc={execution.return_code})")
        if execution.status != "success":
            overall_status = "failed"
            if not cfg.continue_on_failure:
                break

    artifact = {
        "script": {"name": SCRIPT_NAME, "version": SCRIPT_VERSION, "date": SCRIPT_DATE},
        "run_name": cfg.run_name,
        "kit_name": cfg.kit_name,
        "version_id": cfg.version_id,
        "timestamp": cfg.timestamp,
        "created_at_utc": created_at.isoformat(),
        "finished_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": overall_status,
        "strict_failure_policy": not cfg.continue_on_failure,
        "ordered_generators": [str(path) for path in ordered_paths],
        "executed_count": len(executions),
        "required_count": len(ordered_paths),
        "executions": [asdict(item) for item in executions],
    }

    artifacts_dir = repo_root / "UE_Tooling/Artifacts/drone_content"
    artifact_path = artifacts_dir / f"{cfg.run_name}.json"
    _write_artifact(artifact_path, artifact)
    print(f"Artifact: {artifact_path}")

    if overall_status != "success":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
