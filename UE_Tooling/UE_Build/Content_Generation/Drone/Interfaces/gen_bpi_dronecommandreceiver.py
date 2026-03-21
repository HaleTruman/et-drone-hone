"""
Desired behavior:
- Regenerate `BPI_DroneCommandReceiver` as the canonical command-ingress interface for drone control.
- Expose one stable function surface: `ApplyCommandNormalized`.
- Ensure shared v1 schema struct assets exist before interface save.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver`.
- Depends on shared struct schema assets in `/Game/Drone_Content/Structs/*`.
- Consumed by runtime controller flows and implemented by the movement module path
  (`BP_DroneMovement_6DOF`) rather than the pawn composition root.

Assumptions:
- Script runs inside Unreal Editor Python with Blueprint editor APIs available.
- Interface pin-level signature authoring is deferred; function-graph names are source-of-truth in v1.
- Struct assets are generated explicitly by `Drone/Structs/gen_st_*` before this script runs.

Success conditions:
- Interface asset exists and compiles.
- Expected function graph is present.
- Schema metadata is stamped for downstream runtime/tooling validation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_bpi_dronecommandreceiver"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver"
EXPECTED_FUNCTIONS = ["ApplyCommandNormalized"]
STRUCT_DEPENDENCIES = ["ST_DroneCommandNormalized"]
STRUCT_PATHS = {
    "ST_DroneCommandNormalized": "/Game/Drone_Content/Structs/ST_DroneCommandNormalized",
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

    for struct_name in STRUCT_DEPENDENCIES:
        assets.ensure_loaded_asset_type(STRUCT_PATHS[struct_name], "UserDefinedStruct")
    bpi = assets.create_blueprint_interface(TARGET_ASSET_PATH, regenerate=True)
    assets.reset_function_graphs(bpi, EXPECTED_FUNCTIONS)
    assets.stamp_common_metadata(
        bpi,
        generated_by=SCRIPT_NAME,
        extra={
            "interface_name": "BPI_DroneCommandReceiver",
            "function_names": EXPECTED_FUNCTIONS,
            "struct_dependencies": {name: STRUCT_PATHS[name] for name in STRUCT_DEPENDENCIES},
        },
    )
    assets.save_blueprint_asset(TARGET_ASSET_PATH, bpi)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    assets.ensure_blueprint_has_functions(loaded, EXPECTED_FUNCTIONS)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"functions": EXPECTED_FUNCTIONS, "struct_dependencies": STRUCT_DEPENDENCIES},
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
