"""
Desired behavior:
- Regenerate `DA_SensorRigProfileDefault` as the canonical sensor-rig default data asset.
- Persist valid-but-obviously-wrong bootstrap defaults so runtime override failures are easy to detect.
- Reflect v1 front-view-only policy while still requiring runtime config application.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Data/DA_SensorRigProfileDefault`.
- Consumed by sensor rig module generation/runtime (`BP_DroneSensors`).
- Designed for runtime `SET_CONFIG` overwrite, not direct operational defaults.

Assumptions:
- Defaults are represented as a Blueprint-derived data asset.
- Bootstrap values must be valid enough for startup, but visibly incorrect for configured runs.

Success conditions:
- Asset exists and compiles.
- Required sensor default variables exist.
- Bootstrap defaults are validated after save on both the generated class and the asset instance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_da_sensorrigprofiledefault"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Data/DA_SensorRigProfileDefault"
CLASS_ASSET_PATH = "/Game/Drone_Content/Data/Classes/BP_DA_SensorRigProfileDefaultClass"

BOOTSTRAP_DEFAULTS = {
    "config_required": True,
    "active_viewpoint": "front",
    "front_fov_deg": 37.0,
    "capture_width": 222,
    "capture_height": 333,
    "front_offset_x_cm": 17.0,
    "front_offset_y_cm": -23.0,
    "front_offset_z_cm": 41.0,
    "front_rotation_pitch_deg": 11.0,
    "front_rotation_roll_deg": -7.0,
    "front_rotation_yaw_deg": 19.0,
    "sentinel_marker": "__BOOTSTRAP_DEFAULT_SET_CONFIG_REQUIRED__",
}

VARIABLE_SPECS = [
    ("config_required", "bool"),
    ("active_viewpoint", "name"),
    ("front_fov_deg", "real"),
    ("capture_width", "int"),
    ("capture_height", "int"),
    ("front_offset_x_cm", "real"),
    ("front_offset_y_cm", "real"),
    ("front_offset_z_cm", "real"),
    ("front_rotation_pitch_deg", "real"),
    ("front_rotation_roll_deg", "real"),
    ("front_rotation_yaw_deg", "real"),
    ("sentinel_marker", "string"),
]


def _load_assets_module():
    module_path = Path(__file__).resolve().parent.parent / "assemble_drone_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_drone_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_defaults_metadata(assets, data_asset) -> None:
    raw = unreal.EditorAssetLibrary.get_metadata_tag(data_asset, "drone.defaults")
    assets.require(bool(raw), "Missing drone.defaults metadata on DA_SensorRigProfileDefault")


def _pin_for_kind(assets, kind: str):
    pin_factories = {
        "bool": assets.pin_bool,
        "name": assets.pin_name,
        "real": assets.pin_real,
        "int": assets.pin_int,
        "string": assets.pin_string,
    }
    factory = pin_factories.get(kind)
    assets.require(factory is not None, f"Unsupported pin kind for DA_SensorRigProfileDefault: {kind}")
    return factory()


def _add_required_variables(assets, class_blueprint) -> None:
    for variable_name, kind in VARIABLE_SPECS:
        assets.add_member_variable_if_missing(class_blueprint, variable_name, _pin_for_kind(assets, kind))
        assets.set_variable_instance_editable(class_blueprint, variable_name, True)


def _apply_defaults_to_object(assets, target_object, defaults: dict[str, object], object_name: str) -> None:
    for key, value in defaults.items():
        try:
            target_object.set_editor_property(key, value)
        except Exception as exc:
            raise RuntimeError(f"Failed to set bootstrap default {key} on {object_name}: {exc}") from exc


def _validate_defaults_on_object(assets, target_object, object_name: str) -> None:
    for key, expected in BOOTSTRAP_DEFAULTS.items():
        actual = target_object.get_editor_property(key)
        if isinstance(expected, float):
            assets.require(abs(float(actual) - expected) < 0.0001, f"Unexpected default for {object_name}.{key}: {actual} != {expected}")
        elif isinstance(expected, str):
            assets.require(str(actual) == expected, f"Unexpected default for {object_name}.{key}: {actual} != {expected}")
        else:
            assets.require(actual == expected, f"Unexpected default for {object_name}.{key}: {actual} != {expected}")


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    class_blueprint, _ = assets.create_concrete_data_asset_class(
        class_asset_path=CLASS_ASSET_PATH,
        regenerate=True,
    )
    _add_required_variables(assets, class_blueprint)
    assets.set_blueprint_defaults(class_blueprint, BOOTSTRAP_DEFAULTS)
    assets.stamp_common_metadata(
        class_blueprint,
        generated_by=SCRIPT_NAME,
        extra={
            "asset_kind": "DataAssetClass",
            "for_asset": TARGET_ASSET_PATH,
            "class_asset_path": CLASS_ASSET_PATH,
            "bootstrap_defaults": BOOTSTRAP_DEFAULTS,
        },
    )
    assets.save_blueprint_asset(CLASS_ASSET_PATH, class_blueprint)

    data_asset = assets.create_data_asset(
        asset_path=TARGET_ASSET_PATH,
        class_asset_path=CLASS_ASSET_PATH,
        regenerate_asset=True,
        regenerate_class=False,
    )
    _apply_defaults_to_object(assets, data_asset, BOOTSTRAP_DEFAULTS, TARGET_ASSET_PATH)
    assets.stamp_common_metadata(
        data_asset,
        generated_by=SCRIPT_NAME,
        extra={
            "asset_kind": "DataDefault",
            "policy": "bootstrap_defaults_require_runtime_override",
            "bootstrap_defaults": BOOTSTRAP_DEFAULTS,
            "viewpoint_policy_v1": "front_only",
        },
    )
    assets.set_asset_metadata_json(data_asset, "drone.defaults", BOOTSTRAP_DEFAULTS)
    assets.save_asset(TARGET_ASSET_PATH)

    loaded_class_blueprint = assets.ensure_loaded_asset_type(CLASS_ASSET_PATH, "Blueprint")
    assets.ensure_blueprint_has_variables(loaded_class_blueprint, [name for name, _ in VARIABLE_SPECS])
    _validate_defaults_on_object(
        assets,
        assets.blueprint_default_object(loaded_class_blueprint),
        CLASS_ASSET_PATH,
    )

    loaded = assets.ensure_loaded_data_asset(
        TARGET_ASSET_PATH,
        expected_class_asset_path=CLASS_ASSET_PATH,
    )
    _validate_defaults_metadata(assets, loaded)
    _validate_defaults_on_object(assets, loaded, TARGET_ASSET_PATH)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "bootstrap_defaults": BOOTSTRAP_DEFAULTS,
            "class_asset_path": CLASS_ASSET_PATH,
        },
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
