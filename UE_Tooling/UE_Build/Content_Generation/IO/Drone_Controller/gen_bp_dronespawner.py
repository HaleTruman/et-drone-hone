"""
Creates `BP_DroneSpawner` at `/Game/io/Drone_Controller/BP_DroneSpawner`.

Role in pipeline:
- Runtime spawner actor that instantiates one or many `BP_DronePawn` instances in-level.
- Supports BeginPlay and/or command-driven spawning for episode setup/reset flows.
- Assigns stable names/IDs/tags so downstream systems can address drones deterministically.

Related generators:
- Drone asset container: `Drone/Drone_Blueprints/gen_bp_dronepawn.py`
- Command router: `gen_bp_dronecontroller.py`
- Data producer: `../Data_Interface/gen_bp_samplemanager.py`
- Config gate: `../Data_Config/gen_bp_setdataconfig.py`

Runtime linkage:
- Works with tooling-side spawner/control scripts to ensure expected drone population before control/capture.

Current status:
- Minimal generator that creates a valid spawner blueprint shell with deterministic defaults.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

TARGET_ASSET_PATH = "/Game/io/Drone_Controller/BP_DroneSpawner"
SCRIPT_NAME = "gen_bp_dronespawner"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-13"

EXPECTED_FUNCTIONS = [
    "SpawnDrones",
]

DEFAULTS = {
    "spawn_count_default": 1,
    "drone_id_prefix": "drone_",
    "drone_pawn_blueprint_path": "/Game/Drone_Content/Blueprints/BP_DronePawn",
}


def _load_assets_module():
    module_path = Path(__file__).resolve().parents[1] / "assemble_io_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_io_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    blueprint = assets.create_blueprint(TARGET_ASSET_PATH, unreal.Actor, regenerate=True)
    assets.add_member_variable_if_missing(blueprint, "spawn_count_default", assets.pin_int())
    assets.add_member_variable_if_missing(blueprint, "drone_id_prefix", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "drone_pawn_blueprint_path", assets.pin_string())
    assets.reset_function_graphs(blueprint, EXPECTED_FUNCTIONS)
    assets.set_blueprint_defaults(blueprint, DEFAULTS)

    assets.set_asset_metadata(blueprint, "io.generated_by", SCRIPT_NAME)
    assets.set_asset_metadata(blueprint, "io.script_version", SCRIPT_VERSION)
    assets.set_asset_metadata(blueprint, "io.script_date", SCRIPT_DATE)
    assets.save_blueprint_asset(TARGET_ASSET_PATH, blueprint)
    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"functions": EXPECTED_FUNCTIONS, "defaults": DEFAULTS},
    )


if __name__ == "__main__":
    run()
