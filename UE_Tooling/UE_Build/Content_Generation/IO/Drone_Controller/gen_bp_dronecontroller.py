"""
Creates `BP_DroneController` at `/Game/io/Drone_Controller/BP_DroneController`.

Role in pipeline:
- In-engine command router for inbound control messages (`CMD`) from transport endpoints.
- Resolves target drone by stable ID/tag/name and dispatches commands to the correct pawn.
- Decouples network transport details from pawn movement implementation.

Related generators:
- Command contract: `Drone/Interfaces/gen_bpi_dronecommandreceiver.py`
- Target pawn: `Drone/Drone_Blueprints/gen_bp_dronepawn.py`
- Spawn lifecycle peer: `gen_bp_dronespawner.py`
- Config gate: `../Data_Config/gen_bp_setdataconfig.py`

Runtime linkage:
- Receives normalized command intent from runtime IO and applies it through pawn command interfaces.

Current status:
- Minimal generator that creates a valid controller blueprint shell with deterministic defaults.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

TARGET_ASSET_PATH = "/Game/io/Drone_Controller/BP_DroneController"
SCRIPT_NAME = "gen_bp_dronecontroller"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-13"

EXPECTED_FUNCTIONS = [
    "ApplyCommandToDrone",
]

DEFAULTS = {
    "command_receiver_interface_path": "/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver",
    "enforce_config_ready_gate": True,
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
    assets.add_member_variable_if_missing(blueprint, "command_receiver_interface_path", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "enforce_config_ready_gate", assets.pin_bool())
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
