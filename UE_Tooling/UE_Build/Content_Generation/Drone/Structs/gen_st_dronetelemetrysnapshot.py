"""
Creates `ST_DroneTelemetrySnapshot` at `/Game/Drone_Content/Structs/ST_DroneTelemetrySnapshot`.

Role in pipeline:
- Explicit schema owner for the canonical top-level telemetry snapshot payload.
- Keeps telemetry snapshot field ownership out of the shared assembly helper.
- Provides one stable struct asset consumed by telemetry providers and runtime telemetry samplers.
- Carries pose and telemetry together so the top-level `ST` / `BPI` / `BP` module boundary stays symmetric.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_st_dronetelemetrysnapshot"
SCRIPT_VERSION = "1.3.0"
SCRIPT_DATE = "2026-03-15"

TARGET_ASSET_PATH = "/Game/Drone_Content/Structs/ST_DroneTelemetrySnapshot"

SCHEMA = {
    "description": (
        "Canonical top-level telemetry payload emitted per capture trigger. "
        "Contains pose-aligned fields and telemetry detail records so telemetry remains "
        "one top-level runtime contract."
    ),
    "fields": [
        {"name": "timestamp_utc", "type": "string"},
        {"name": "run_id", "type": "string"},
        {"name": "capture_id", "type": "string"},
        {"name": "location_cm", "type": "vector3_real"},
        {"name": "rotation_quat_xyzw", "type": "vector4_real"},
        {"name": "forward_vector", "type": "vector3_real"},
        {"name": "right_vector", "type": "vector3_real"},
        {"name": "up_vector", "type": "vector3_real"},
        {"name": "distance_units", "type": "string"},
        {"name": "distance_precision_decimals", "type": "int"},
        {
            "name": "mesh_proximity_records",
            "type": (
                "array<object{mesh_name:string,mesh_path:string,"
                "distance_to_pivot_cm:real,distance_to_edge_cm:real}>"
            ),
        },
    ],
    "notes": [
        "This is the canonical top-level telemetry contract.",
        "Pose-aligned fields and per-mesh proximity records are carried directly inside "
        "ST_DroneTelemetrySnapshot to keep the top-level runtime contract symmetric.",
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
