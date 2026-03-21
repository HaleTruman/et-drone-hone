"""
Desired behavior:
- Regenerate `BP_DroneMovement_6DOF` as a minimal deterministic movement module.
- Expose command ingress function `ApplyCommandNormalized` without physics complexity in v1.
- Persist baseline tuning placeholders that runtime config can overwrite in-memory.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Blueprints/BP_DroneMovement_6DOF`.
- Parent class: `ActorComponent`.
- Function surface: `ApplyCommandNormalized`.

Assumptions:
- v1 movement is command-to-transform intent only (no physics simulation).
- Runtime config gate enforces replacement of placeholder tuning before run actions.

Success conditions:
- Blueprint exists/compiles with required function + variables.
- Physics toggle is false by default.
- Sentinel baseline tuning values are present for validation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_bp_dronemovement_6dof"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Blueprints/BP_DroneMovement_6DOF"
MOVEMENT_DEFAULTS_PATH = "/Game/Drone_Content/Data/DA_DroneMovementDefault"
EXPECTED_FUNCTIONS = ["ApplyCommandNormalized"]

VARIABLE_DEFAULTS = {
    "config_required": True,
    "use_physics": False,
    "input_pitch": 0.0,
    "input_roll": 0.0,
    "input_yaw": 0.0,
    "input_throttle": 0.0,
    "max_speed_cmps": -9999.0,
    "max_accel_cmps2": -9999.0,
    "max_yaw_degps": -9999.0,
    "max_pitch_degps": -9999.0,
    "max_roll_degps": -9999.0,
    "damping": -9999.0,
    "movement_mode": "minimal_deterministic_no_physics",
}


def _load_assets_module():
    module_path = Path(__file__).resolve().parent.parent / "assemble_drone_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_drone_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_defaults(assets, blueprint) -> None:
    cdo = assets.blueprint_default_object(blueprint)
    for key, expected in VARIABLE_DEFAULTS.items():
        actual = cdo.get_editor_property(key)
        if isinstance(expected, float):
            assets.require(abs(float(actual) - expected) < 0.0001, f"Unexpected default for {key}: {actual} != {expected}")
        else:
            assets.require(actual == expected, f"Unexpected default for {key}: {actual} != {expected}")


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    blueprint = assets.create_blueprint(TARGET_ASSET_PATH, unreal.ActorComponent, regenerate=True)
    assets.reset_function_graphs(blueprint, EXPECTED_FUNCTIONS)

    assets.add_member_variable_if_missing(blueprint, "config_required", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "use_physics", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "input_pitch", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "input_roll", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "input_yaw", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "input_throttle", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "max_speed_cmps", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "max_accel_cmps2", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "max_yaw_degps", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "max_pitch_degps", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "max_roll_degps", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "damping", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "movement_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "movement_defaults_asset", assets.pin_object_reference(unreal.Object))

    for variable_name in VARIABLE_DEFAULTS:
        assets.set_variable_instance_editable(blueprint, variable_name, True)

    defaults_asset = assets.load_asset(MOVEMENT_DEFAULTS_PATH)
    merged_defaults = dict(VARIABLE_DEFAULTS)
    merged_defaults["movement_defaults_asset"] = defaults_asset
    assets.set_blueprint_defaults(blueprint, merged_defaults)
    assets.stamp_common_metadata(
        blueprint,
        generated_by=SCRIPT_NAME,
        extra={
            "function_names": EXPECTED_FUNCTIONS,
            "movement_defaults_asset": MOVEMENT_DEFAULTS_PATH,
            "v1_policy": "minimal_no_physics",
        },
    )
    assets.save_blueprint_asset(TARGET_ASSET_PATH, blueprint)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    assets.ensure_blueprint_has_functions(loaded, EXPECTED_FUNCTIONS)
    assets.ensure_blueprint_has_variables(loaded, list(merged_defaults.keys()))
    _validate_defaults(assets, loaded)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"functions": EXPECTED_FUNCTIONS, "defaults_asset": MOVEMENT_DEFAULTS_PATH},
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
