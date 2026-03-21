"""
Creates `ST_RunConfig` at `/Game/io/Data_Config/ST_RunConfig`.

Role in pipeline:
- UE-side typed struct for runtime configuration payloads (`SET_CONFIG`).
- Defines stable fields for movement tuning, sensor/viewpoint settings, telemetry selection, and traceability IDs.
- Serves as the typed contract boundary between network JSON and in-engine Blueprint application logic.

Related generators:
- Config endpoint: `gen_bp_setdataconfig.py`
- Runtime consumers: `../Drone_Controller/gen_bp_dronecontroller.py`, `../Data_Interface/gen_bp_samplemanager.py`
- Tooling compiler counterpart: `UE_Tooling/Config/RunConfig.py`

Runtime linkage:
- Used to parse/validate inbound config into structured data before in-memory override application.

Current status:
- Minimal generator that creates a valid `UserDefinedStruct` asset with schema metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

TARGET_ASSET_PATH = "/Game/io/Data_Config/ST_RunConfig"
SCRIPT_NAME = "gen_st_runconfig"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-16"

SET_CONFIG_TYPED_CONTRACT = {
    "message_type": "SET_CONFIG",
    "payload_root": "payload",
    "required_payload_paths": [
        "payload.config_id",
        "payload.config_hash",
        "payload.movement",
        "payload.sensor_rig",
        "payload.telemetry",
        "payload.sensor_rig.active_viewpoint",
        "payload.sensor_rig.front_fov_deg",
        "payload.sensor_rig.capture_width",
        "payload.sensor_rig.capture_height",
        "payload.sensor_rig.front_offset_cm.x",
        "payload.sensor_rig.front_offset_cm.y",
        "payload.sensor_rig.front_offset_cm.z",
        "payload.sensor_rig.front_rotation_deg.pitch",
        "payload.sensor_rig.front_rotation_deg.roll",
        "payload.sensor_rig.front_rotation_deg.yaw",
    ],
    "payload_shape": {
        "config_id": {"type": "string", "required": True},
        "config_hash": {"type": "string", "required": True},
        "movement": {
            "type": "object",
            "required": True,
            "fields": {
                "max_speed_cmps": {"type": "real", "required": True},
                "max_accel_cmps2": {"type": "real", "required": True},
                "max_yaw_degps": {"type": "real", "required": True},
                "max_pitch_degps": {"type": "real", "required": True},
                "max_roll_degps": {"type": "real", "required": True},
                "damping": {"type": "real", "required": True},
            },
        },
        "sensor_rig": {
            "type": "object",
            "required": True,
            "fields": {
                "active_viewpoint": {"type": "string", "required": True},
                "front_fov_deg": {"type": "real", "required": True},
                "capture_width": {"type": "int", "required": True},
                "capture_height": {"type": "int", "required": True},
                "front_offset_cm": {
                    "type": "object",
                    "required": True,
                    "fields": {
                        "x": {"type": "real", "required": True},
                        "y": {"type": "real", "required": True},
                        "z": {"type": "real", "required": True},
                    },
                },
                "front_rotation_deg": {
                    "type": "object",
                    "required": True,
                    "fields": {
                        "pitch": {"type": "real", "required": True},
                        "roll": {"type": "real", "required": True},
                        "yaw": {"type": "real", "required": True},
                    },
                },
            },
        },
        "telemetry": {
            "type": "object",
            "required": True,
            "fields": {
                "include_all_visible_meshes": {"type": "bool", "required": True},
                "distance_units": {"type": "string", "required": True},
                "distance_precision_decimals": {"type": "int", "required": True},
                "center_distance_mode": {"type": "string", "required": True},
                "edge_distance_mode": {"type": "string", "required": True},
                "line_of_sight_filtering": {"type": "bool", "required": True},
            },
        },
    },
    "runtime_apply_targets": {
        "payload.sensor_rig.active_viewpoint": "BP_DroneSensors.active_viewpoint",
        "payload.sensor_rig.front_fov_deg": "BP_DroneSensors.front_fov_deg",
        "payload.sensor_rig.capture_width": "BP_DroneSensors.capture_width",
        "payload.sensor_rig.capture_height": "BP_DroneSensors.capture_height",
        "payload.sensor_rig.front_offset_cm.x": "BP_DroneSensors.front_offset_x_cm",
        "payload.sensor_rig.front_offset_cm.y": "BP_DroneSensors.front_offset_y_cm",
        "payload.sensor_rig.front_offset_cm.z": "BP_DroneSensors.front_offset_z_cm",
        "payload.sensor_rig.front_rotation_deg.pitch": "BP_DroneSensors.front_rotation_pitch_deg",
        "payload.sensor_rig.front_rotation_deg.roll": "BP_DroneSensors.front_rotation_roll_deg",
        "payload.sensor_rig.front_rotation_deg.yaw": "BP_DroneSensors.front_rotation_yaw_deg",
    },
    "contract_sources": {
        "tooling_compiler": "UE_Tooling/Config/RunConfig.py",
        "image_contract_doc": "docs/data/image_capture_contract.md",
        "ws_contract_doc": "docs/data/data_contract.md",
    },
    "runtime_policy": {
        "config_required_before_actions": True,
        "mid_run_set_config_allowed": False,
    },
}


def _struct_member_authoring_capability() -> dict:
    utils = getattr(unreal, "StructureEditorUtils", None)
    if utils is None:
        return {
            "mode": "metadata_only",
            "available": False,
            "reason": "unreal.StructureEditorUtils is not exposed to Unreal Python in this environment.",
        }

    required_names = [
        "add_variable",
        "rename_variable",
        "change_variable_type",
        "get_var_desc",
        "on_structure_changed",
    ]
    missing = [name for name in required_names if not hasattr(utils, name)]
    if missing:
        return {
            "mode": "metadata_only",
            "available": False,
            "reason": "StructureEditorUtils binding is incomplete for Python struct-member authoring.",
            "missing_python_members": missing,
        }

    return {
        "mode": "reflected_editor_utils",
        "available": True,
        "reason": "StructureEditorUtils Python binding appears available for deep struct-member authoring.",
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

    assets.delete_legacy_file_if_exists("io/Data_Config/ST_RunConfig")
    struct_asset = assets.create_user_defined_struct(TARGET_ASSET_PATH, regenerate=True)
    capability = _struct_member_authoring_capability()
    assets.set_asset_metadata(struct_asset, "io.generated_by", SCRIPT_NAME)
    assets.set_asset_metadata(struct_asset, "io.script_version", SCRIPT_VERSION)
    assets.set_asset_metadata(struct_asset, "io.script_date", SCRIPT_DATE)
    assets.set_asset_metadata_json(struct_asset, "io.set_config_typed_contract", SET_CONFIG_TYPED_CONTRACT)
    assets.set_asset_metadata_json(struct_asset, "io.required_payload_paths", SET_CONFIG_TYPED_CONTRACT["required_payload_paths"])
    assets.set_asset_metadata_json(struct_asset, "io.runtime_apply_targets", SET_CONFIG_TYPED_CONTRACT["runtime_apply_targets"])
    assets.set_asset_metadata(struct_asset, "io.struct_member_authoring_mode", str(capability["mode"]))
    assets.set_asset_metadata_json(struct_asset, "io.struct_member_authoring", capability)
    assets.save_asset(TARGET_ASSET_PATH)
    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "UserDefinedStruct")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "asset_kind": "UserDefinedStruct",
            "typed_contract": SET_CONFIG_TYPED_CONTRACT,
            "member_authoring": capability,
        },
    )


if __name__ == "__main__":
    run()
