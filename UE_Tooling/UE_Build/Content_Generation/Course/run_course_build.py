"""
Status: in development.
Renamed from `run_versioned_course_generation.py` to `run_course_build.py` to
mirror the Course/Drone/IO/WebSocket orchestration layer naming pattern.

Role:
- Orchestrate the ordered Course build run.
- Record script/version metadata and generated output manifests.
- Stay focused on build orchestration, not reusable asset assembly behavior.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPT_NAME = "run_course_build"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-15"


@dataclass(frozen=True)
class RunConfig:
    course_name: str
    version_id: str
    timestamp: str
    count: int
    x_min_cm: float
    x_max_cm: float
    y_min_cm: float
    y_max_cm: float
    ground_z_cm: float
    ground_offset_cm: float
    z_jitter_cm: float
    timeout_seconds: int
    unreal_editor: str

    @property
    def run_name(self) -> str:
        return f"{self.course_name}_{self.version_id}_{self.timestamp}"

    @property
    def asset_dir(self) -> str:
        return f"/Game/Course_Content/{self.run_name}"

    @property
    def level_path(self) -> str:
        return f"{self.asset_dir}/Maps/{self.course_name}"

    @property
    def material_name(self) -> str:
        return "M_CourseTorus_Red"

    @property
    def mesh_name(self) -> str:
        return "SM_CourseTorus_1mOpening"

    @property
    def light_blueprint_name(self) -> str:
        return "BP_CourseLight_Main"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--course-name", default="L_CourseTorus")
    parser.add_argument("--version-id", default="v001")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--x-min-cm", type=float, default=-5000.0)
    parser.add_argument("--x-max-cm", type=float, default=5000.0)
    parser.add_argument("--y-min-cm", type=float, default=-5000.0)
    parser.add_argument("--y-max-cm", type=float, default=5000.0)
    parser.add_argument("--ground-z-cm", type=float, default=0.0)
    parser.add_argument("--ground-offset-cm", type=float, default=180.0)
    parser.add_argument("--z-jitter-cm", type=float, default=200.0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--unreal-editor", default="")
    args = parser.parse_args()

    def sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")

    course_name = sanitize(str(args.course_name))
    version_id = sanitize(str(args.version_id))
    timestamp = sanitize(str(args.timestamp))

    return RunConfig(
        course_name=course_name,
        version_id=version_id,
        timestamp=timestamp,
        count=max(1, int(args.count)),
        x_min_cm=float(args.x_min_cm),
        x_max_cm=float(args.x_max_cm),
        y_min_cm=float(args.y_min_cm),
        y_max_cm=float(args.y_max_cm),
        ground_z_cm=float(args.ground_z_cm),
        ground_offset_cm=float(args.ground_offset_cm),
        z_jitter_cm=max(0.0, float(args.z_jitter_cm)),
        timeout_seconds=max(1, int(args.timeout_seconds)),
        unreal_editor=str(args.unreal_editor).strip(),
    )


def _extract_script_metadata(script_path: Path) -> dict:
    content = script_path.read_text(encoding="utf-8")

    def extract(name: str) -> Optional[str]:
        match = re.search(rf'^{name}\s*=\s*"([^"]+)"', content, re.MULTILINE)
        return match.group(1) if match else None

    mtime_utc = datetime.fromtimestamp(script_path.stat().st_mtime, tz=timezone.utc).isoformat()
    return {
        "path": str(script_path),
        "script_name": extract("SCRIPT_NAME"),
        "script_version": extract("SCRIPT_VERSION"),
        "script_date": extract("SCRIPT_DATE"),
        "file_mtime_utc": mtime_utc,
    }


def _game_path_to_file(content_root: Path, game_path: str, extension: str) -> Path:
    if not game_path.startswith("/Game/"):
        raise ValueError(f"Unsupported game path: {game_path}")
    relative = game_path[len("/Game/") :]
    return content_root / f"{relative}.{extension}"


def main() -> int:
    cfg = _parse_args()
    repo_root = _repo_root()
    ue_tooling_root = repo_root / "UE_Tooling"
    content_root = repo_root / "UE_Drone_Env/Content"

    run_unreal_path = ue_tooling_root / "UE_Build/run_unreal_build_gen.py"
    map_script = repo_root / "UE_Tooling/UE_Build/Content_Generation/Course/Maps/gen_l_coursetorus.py"
    material_script = repo_root / "UE_Tooling/UE_Build/Content_Generation/Course/Materials/gen_m_coursetorus_red.py"
    mesh_script = repo_root / "UE_Tooling/UE_Build/Content_Generation/Course/Meshes/gen_sm_coursetorus_1mopening.py"
    blueprint_script = repo_root / "UE_Tooling/UE_Build/Content_Generation/Course/Course_Blueprints/gen_bp_courselight_main.py"
    assembly_script = repo_root / "UE_Tooling/UE_Build/Content_Generation/Course/assemble_course_assets.py"

    artifacts_dir = ue_tooling_root / "Artifacts/runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/tmp/{cfg.run_name}.log")
    artifact_path = artifacts_dir / f"{cfg.run_name}.json"

    cmd = [
        sys.executable,
        str(run_unreal_path),
        "--script",
        str(map_script),
        "--log",
        str(log_path),
        "--timeout-seconds",
        str(cfg.timeout_seconds),
    ]
    if cfg.unreal_editor:
        cmd.extend(["--unreal-editor", cfg.unreal_editor])
    cmd.extend(
        [
            "--",
            "--asset-dir",
            cfg.asset_dir,
            "--level",
            cfg.level_path,
            "--material-name",
            cfg.material_name,
            "--mesh-name",
            cfg.mesh_name,
            "--light-blueprint-name",
            cfg.light_blueprint_name,
            "--count",
            str(cfg.count),
            "--x-min-cm",
            str(cfg.x_min_cm),
            "--x-max-cm",
            str(cfg.x_max_cm),
            "--y-min-cm",
            str(cfg.y_min_cm),
            "--y-max-cm",
            str(cfg.y_max_cm),
            "--ground-z-cm",
            str(cfg.ground_z_cm),
            "--ground-offset-cm",
            str(cfg.ground_offset_cm),
            "--z-jitter-cm",
            str(cfg.z_jitter_cm),
        ]
    )

    proc = subprocess.run(cmd, check=False)

    expected_files = {
        "material_uasset": _game_path_to_file(content_root, f"{cfg.asset_dir}/{cfg.material_name}", "uasset"),
        "mesh_uasset": _game_path_to_file(content_root, f"{cfg.asset_dir}/{cfg.mesh_name}", "uasset"),
        "light_blueprint_uasset": _game_path_to_file(content_root, f"{cfg.asset_dir}/{cfg.light_blueprint_name}", "uasset"),
        "map_umap": _game_path_to_file(content_root, cfg.level_path, "umap"),
    }

    script_paths = [assembly_script, material_script, mesh_script, blueprint_script, map_script]
    artifact = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "script_date": SCRIPT_DATE,
        "run_name": cfg.run_name,
        "course_name": cfg.course_name,
        "version_id": cfg.version_id,
        "timestamp": cfg.timestamp,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": "success" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "command": cmd,
        "log_path": str(log_path),
        "ue_paths": {
            "asset_dir": cfg.asset_dir,
            "level_path": cfg.level_path,
            "material_name": cfg.material_name,
            "mesh_name": cfg.mesh_name,
            "light_blueprint_name": cfg.light_blueprint_name,
        },
        "scripts": [_extract_script_metadata(path) for path in script_paths],
        "generated_files": {
            key: {"path": str(path), "exists": path.exists()}
            for key, path in expected_files.items()
        },
    }

    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(str(artifact_path))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
