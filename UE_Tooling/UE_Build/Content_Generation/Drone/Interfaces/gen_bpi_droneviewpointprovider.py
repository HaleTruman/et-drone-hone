"""
Desired behavior:
- Regenerate `BPI_DroneViewpointProvider` for stable viewpoint-query access.
- Expose two v1 function surfaces: `ListViewpoints` and `GetViewpointSnapshot`.
- Record only interface identity and explicit struct dependencies.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Interfaces/BPI_DroneViewpointProvider`.
- Function graphs: `ListViewpoints`, `GetViewpointSnapshot`.
- Struct contract dependency: `ST_DroneViewpointSnapshot`.

Assumptions:
- Struct assets are generated explicitly under `/Game/Drone_Content/Structs` before this script runs.

Success conditions:
- Interface asset exists and compiles.
- Both function graphs are present.
- Metadata contains interface identity + struct dependency only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_bpi_droneviewpointprovider"
SCRIPT_VERSION = "1.2.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneViewpointProvider"
EXPECTED_FUNCTIONS = ["ListViewpoints", "GetViewpointSnapshot"]
STRUCT_DEPENDENCIES = ["ST_DroneViewpointSnapshot"]
STRUCT_PATHS = {
    "ST_DroneViewpointSnapshot": "/Game/Drone_Content/Structs/ST_DroneViewpointSnapshot",
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
            "interface_name": "BPI_DroneViewpointProvider",
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
        {
            "functions": EXPECTED_FUNCTIONS,
            "struct_dependencies": STRUCT_DEPENDENCIES,
        },
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
