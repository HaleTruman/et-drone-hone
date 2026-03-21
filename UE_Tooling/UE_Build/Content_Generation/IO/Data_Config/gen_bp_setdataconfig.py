"""
Creates `BP_SetDataConfig` at `/Game/io/Data_Config/BP_SetDataConfig`.

Role in pipeline:
- In-engine configuration ingress endpoint for `SET_CONFIG` payloads.
- Parses inbound payloads into `ST_RunConfig`, validates schema/shape expectations, and applies runtime overrides.
- Publishes config-ready state/event so spawn/control/sample flows can gate execution on successful config apply.

Related generators:
- Typed config contract: `gen_st_runconfig.py`
- Runtime consumers of applied config:
  - `../Drone_Controller/gen_bp_dronespawner.py`
  - `../Drone_Controller/gen_bp_dronecontroller.py`
  - `../Data_Interface/gen_bp_samplemanager.py`
- Drone-side defaults affected by overrides:
  - `Drone/Data/gen_da_dronemovementdefault.py`
  - `Drone/Data/gen_da_sensorrigprofiledefault.py`

Runtime linkage:
- Bridges tooling-compiled config payloads into in-memory UE runtime state without mutating default assets.

Current status:
- Minimal generator that creates a valid Blueprint subclass of `SetDataConfigRuntimeActor`.
- Step 8B contract metadata is stamped and runtime ingress is owned by parent-class
  `OnSetConfigReceived` behavior that validates sensor fields, applies runtime sensor state,
  and only emits `CONFIG_READY` after successful apply.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import unreal

TARGET_ASSET_PATH = "/Game/io/Data_Config/BP_SetDataConfig"
ST_RUNCONFIG_ASSET_PATH = "/Game/io/Data_Config/ST_RunConfig"
SCRIPT_NAME = "gen_bp_setdataconfig"
SCRIPT_VERSION = "1.4.0"
SCRIPT_DATE = "2026-03-16"

EXPECTED_FUNCTIONS = [
    "OnConfigAccepted",
    "OnConfigRejected",
]

DEFAULTS = {
    "server_url": "ws://127.0.0.1:8765",
    "auto_connect_on_begin_play": True,
    "fail_on_run_id_mismatch": True,
    "expected_run_id": "",
    "config_gate_enabled": True,
    "set_config_ingress_mode": "parent_default_on_set_config_received_v1",
    "set_config_runtime_owner": "BP_SetDataConfig",
    "active_config_reference_owner": "BP_SetDataConfig",
    "active_config_reference_read_mode": "authoritative_runtime_seam_v1",
    "active_config_reference_read_function": "GetActiveConfigReference",
}


def _load_assets_module():
    module_path = Path(__file__).resolve().parents[1] / "assemble_io_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_io_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_runconfig_contract() -> dict:
    struct_asset = unreal.EditorAssetLibrary.load_asset(ST_RUNCONFIG_ASSET_PATH)
    if struct_asset is None:
        return {
            "source_struct_asset": ST_RUNCONFIG_ASSET_PATH,
            "missing": True,
            "required_payload_paths": [],
            "runtime_apply_targets": {},
        }

    required_payload_paths_raw = unreal.EditorAssetLibrary.get_metadata_tag(struct_asset, "io.required_payload_paths")
    runtime_apply_targets_raw = unreal.EditorAssetLibrary.get_metadata_tag(struct_asset, "io.runtime_apply_targets")

    required_payload_paths = []
    runtime_apply_targets = {}
    try:
        required_payload_paths = json.loads(required_payload_paths_raw) if required_payload_paths_raw else []
    except Exception:
        required_payload_paths = []
    try:
        runtime_apply_targets = json.loads(runtime_apply_targets_raw) if runtime_apply_targets_raw else {}
    except Exception:
        runtime_apply_targets = {}

    return {
        "source_struct_asset": ST_RUNCONFIG_ASSET_PATH,
        "missing": False,
        "required_payload_paths": required_payload_paths,
        "runtime_apply_targets": runtime_apply_targets,
    }


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    parent_class = getattr(unreal, "SetDataConfigRuntimeActor", None)
    assets.require(parent_class is not None, "SetDataConfigRuntimeActor class is unavailable. Build/deploy game module first.")

    blueprint = assets.create_blueprint(TARGET_ASSET_PATH, parent_class, regenerate=True)
    assets.add_member_variable_if_missing(blueprint, "config_gate_enabled", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "set_config_ingress_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "set_config_runtime_owner", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "active_config_reference_owner", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "active_config_reference_read_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "active_config_reference_read_function", assets.pin_string())
    assets.reset_function_graphs(blueprint, EXPECTED_FUNCTIONS)
    assets.set_blueprint_defaults(blueprint, DEFAULTS)

    runconfig_contract = _read_runconfig_contract()

    assets.set_asset_metadata(blueprint, "io.generated_by", SCRIPT_NAME)
    assets.set_asset_metadata(blueprint, "io.script_version", SCRIPT_VERSION)
    assets.set_asset_metadata(blueprint, "io.script_date", SCRIPT_DATE)
    assets.set_asset_metadata(
        blueprint,
        "io.handshake_contract",
        "Receives SET_CONFIG through SetDataConfigRuntimeActor.OnSetConfigReceived, validates required image sensor fields, applies runtime sensor state, and emits ACK/CONFIG_READY only after successful apply.",
    )
    assets.set_asset_metadata(blueprint, "io.config_ingress_phase", "8B_complete")
    assets.set_asset_metadata(blueprint, "io.set_config_ingress_mode", DEFAULTS["set_config_ingress_mode"])
    assets.set_asset_metadata(blueprint, "io.active_config_reference_owner", DEFAULTS["active_config_reference_owner"])
    assets.set_asset_metadata(blueprint, "io.active_config_reference_read_mode", DEFAULTS["active_config_reference_read_mode"])
    assets.set_asset_metadata(blueprint, "io.active_config_reference_read_function", DEFAULTS["active_config_reference_read_function"])
    assets.set_asset_metadata_json(
        blueprint,
        "io.active_config_reference_contract",
        {
            "owner": DEFAULTS["active_config_reference_owner"],
            "read_function": DEFAULTS["active_config_reference_read_function"],
            "read_mode": DEFAULTS["active_config_reference_read_mode"],
            "required_fields": ["config_id", "config_hash"],
            "authoritative_source_note": "BP_SetDataConfig is the single runtime owner of active config identity.",
        },
    )
    assets.set_asset_metadata_json(blueprint, "io.required_payload_paths", runconfig_contract["required_payload_paths"])
    assets.set_asset_metadata_json(blueprint, "io.runtime_apply_targets", runconfig_contract["runtime_apply_targets"])
    assets.set_asset_metadata_json(blueprint, "io.runconfig_contract_source", runconfig_contract)

    assets.save_blueprint_asset(TARGET_ASSET_PATH, blueprint)
    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"parent_class": "SetDataConfigRuntimeActor", "default_server_url": DEFAULTS["server_url"]},
    )


if __name__ == "__main__":
    run()
