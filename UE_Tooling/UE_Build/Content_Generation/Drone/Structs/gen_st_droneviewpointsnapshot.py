"""
Creates `ST_DroneViewpointSnapshot` at `/Game/Drone_Content/Structs/ST_DroneViewpointSnapshot`.

Role in pipeline:
- Explicit schema owner for image/viewpoint snapshot metadata.
- Keeps viewpoint field ownership out of the shared assembly helper.
- Defines the canonical snapshot shape used by viewpoint interfaces and later `OBS` assembly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_st_droneviewpointsnapshot"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-15"

TARGET_ASSET_PATH = "/Game/Drone_Content/Structs/ST_DroneViewpointSnapshot"

SCHEMA = {
    "description": "Single named viewpoint capture settings + transform data.",
    "fields": [
        {"name": "timestamp_utc", "type": "string"},
        {"name": "run_id", "type": "string"},
        {"name": "capture_id", "type": "string"},
        {"name": "viewpoint_name", "type": "string"},
        {"name": "fov_deg", "type": "real"},
        {"name": "width", "type": "int"},
        {"name": "height", "type": "int"},
        {"name": "rig_offset_from_drone_body_cm", "type": "vector3_real"},
        {"name": "rig_rotation_from_drone_body_deg", "type": "vector3_real"},
    ],
}


def _load_assets_module():
    module_path = Path(__file__).resolve().parent.parent / "assemble_drone_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_drone_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    assets.generate_struct_asset(
        asset_path=TARGET_ASSET_PATH,
        generated_by=SCRIPT_NAME,
        schema=SCHEMA,
        regenerate=True,
    )
    capability = assets.struct_member_authoring_capability()
    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "UserDefinedStruct")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"schema": SCHEMA, "member_authoring": capability},
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
