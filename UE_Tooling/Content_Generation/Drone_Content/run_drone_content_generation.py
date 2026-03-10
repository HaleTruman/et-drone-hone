import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPT_NAME = "run_drone_content_generation"
SCRIPT_VERSION = "0.1.0"
SCRIPT_DATE = "2026-03-10"

ORDERED_GENERATORS = [
    "Interfaces/gen_bpi_dronecommandreceiver.py",
    "Interfaces/gen_bpi_droneposeprovider.py",
    "Interfaces/gen_bpi_dronetelemetryprovider.py",
    "Interfaces/gen_bpi_droneviewpointprovider.py",
    "Data/gen_da_dronemovementdefault.py",
    "Data/gen_da_sensorrigprofiledefault.py",
    "Meshes/gen_sm_dronebody.py",
    "Meshes/gen_sm_dronecollisionproxy.py",
    "Materials/gen_m_dronebody_base.py",
    "Materials/gen_mi_dronebody_default.py",
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

    @property
    def run_name(self) -> str:
        return f"{self.kit_name}_{self.version_id}_{self.timestamp}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-name", default="DroneContentKit")
    parser.add_argument("--version-id", default="v001")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    def sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")

    return RunConfig(
        kit_name=sanitize(str(args.kit_name)),
        version_id=sanitize(str(args.version_id)),
        timestamp=sanitize(str(args.timestamp)),
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


def main() -> int:
    cfg = _parse_args()
    repo_root = _repo_root()
    drone_content_root = repo_root / "UE_Tooling/Content_Generation/Drone_Content"

    ordered_paths = [drone_content_root / rel_path for rel_path in ORDERED_GENERATORS]
    missing = [str(path) for path in ordered_paths if not path.exists()]
    script_paths = [path for path in ordered_paths if path.exists()]

    artifacts_dir = repo_root / "UE_Tooling/Artifacts/drone_content"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / f"{cfg.run_name}.json"

    artifact = {
        "script": {
            "name": SCRIPT_NAME,
            "version": SCRIPT_VERSION,
            "date": SCRIPT_DATE,
        },
        "run_name": cfg.run_name,
        "kit_name": cfg.kit_name,
        "version_id": cfg.version_id,
        "timestamp": cfg.timestamp,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": "placeholder_not_executed",
        "notes": [
            "This scaffold records intended generator order and metadata.",
            "Asset generation execution wiring is intentionally pending.",
        ],
        "ordered_generators": [str(path) for path in ordered_paths],
        "missing_generators": missing,
        "scripts": [_extract_script_metadata(path) for path in script_paths],
    }

    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(str(artifact_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
