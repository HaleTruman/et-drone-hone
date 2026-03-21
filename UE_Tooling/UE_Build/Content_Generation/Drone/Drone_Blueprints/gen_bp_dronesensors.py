"""
Desired behavior:
- Regenerate `BP_DroneSensors` as the runtime owner of image-capture state for a single front viewpoint.
- Expose the public function surfaces needed for config apply, capture, and viewpoint snapshot queries.
- Bootstrap from `DA_SensorRigProfileDefault`, then hold the active in-memory sensor-rig state that later runtime config will overwrite.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Blueprints/BP_DroneSensors`.
- Parent class: `DroneSensorsRuntimeComponent` (native apply behavior).
- Function surfaces:
  - `ApplySensorRigConfig`
  - `CaptureActiveViewpointPngBytes`
  - `ListViewpoints`
  - `GetViewpointSnapshot`

Assumptions:
- v1 uses only one viewpoint (`front`).
- Image transport is PNG-only and color space is sRGB-only for this stage.
- Sensor defaults can be overwritten by runtime config after config-ready gate.

Success conditions:
- Blueprint exists and compiles.
- Required variables are present.
- Public function surfaces are provided by the native parent component.
- Defaults are loaded from `DA_SensorRigProfileDefault` rather than hardcoded authoritative runtime values.
- Blueprint metadata clearly declares `BP_DroneSensors` as the runtime owner of active image state.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_bp_dronesensors"
SCRIPT_VERSION = "1.3.0"
SCRIPT_DATE = "2026-03-16"

TARGET_ASSET_PATH = "/Game/Drone_Content/Blueprints/BP_DroneSensors"
SENSOR_DEFAULTS_PATH = "/Game/Drone_Content/Data/DA_SensorRigProfileDefault"
VIEWPOINT_STRUCT_PATH = "/Game/Drone_Content/Structs/ST_DroneViewpointSnapshot"
# Keep native parent implementations for capture/snapshot/list surfaces.
# Creating empty Blueprint override graphs here would shadow truthful native logic.
FUNCTION_GRAPHS: list[str] = []
PUBLIC_FUNCTION_SURFACES = [
    "ApplySensorRigConfig",
    "CaptureActiveViewpointPngBytes",
    "ListViewpoints",
    "GetViewpointSnapshot",
]

BOOTSTRAP_FIELD_NAMES = [
    "config_required",
    "active_viewpoint",
    "front_fov_deg",
    "capture_width",
    "capture_height",
    "front_offset_x_cm",
    "front_offset_y_cm",
    "front_offset_z_cm",
    "front_rotation_pitch_deg",
    "front_rotation_roll_deg",
    "front_rotation_yaw_deg",
    "sentinel_marker",
]

RUNTIME_VARIABLE_DEFAULTS = {
    "runtime_config_applied": False,
    "config_apply_last_error": "",
    "capture_output_mode": "image_bytes_png",
    "capture_color_space": "srgb",
    "active_image_state_owner": "BP_DroneSensors",
    "viewpoint_count": 1,
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
    for key, expected in _load_bootstrap_defaults(assets).items():
        actual = cdo.get_editor_property(key)
        if isinstance(expected, float):
            assets.require(abs(float(actual) - expected) < 0.0001, f"Unexpected default for {key}: {actual} != {expected}")
        elif isinstance(expected, str):
            assets.require(str(actual) == expected, f"Unexpected default for {key}: {actual} != {expected}")
        else:
            assets.require(actual == expected, f"Unexpected default for {key}: {actual} != {expected}")
    for key, expected in RUNTIME_VARIABLE_DEFAULTS.items():
        actual = cdo.get_editor_property(key)
        if isinstance(expected, float):
            assets.require(abs(float(actual) - expected) < 0.0001, f"Unexpected default for {key}: {actual} != {expected}")
        elif isinstance(expected, str):
            assets.require(str(actual) == expected, f"Unexpected default for {key}: {actual} != {expected}")
        else:
            assets.require(actual == expected, f"Unexpected default for {key}: {actual} != {expected}")


def _load_bootstrap_defaults(assets) -> dict[str, object]:
    defaults_asset = assets.ensure_loaded_data_asset(SENSOR_DEFAULTS_PATH)
    bootstrap: dict[str, object] = {}
    for field_name in BOOTSTRAP_FIELD_NAMES:
        target_name = "bootstrap_sentinel_marker" if field_name == "sentinel_marker" else field_name
        try:
            bootstrap[target_name] = defaults_asset.get_editor_property(field_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read bootstrap field '{field_name}' from {SENSOR_DEFAULTS_PATH}: {exc}"
            ) from exc
    return bootstrap


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    parent_class = getattr(unreal, "DroneSensorsRuntimeComponent", None)
    assets.require(
        parent_class is not None,
        "DroneSensorsRuntimeComponent class is unavailable. Build the UE_Drone_Env module before generating BP_DroneSensors.",
    )
    blueprint = assets.create_blueprint(TARGET_ASSET_PATH, parent_class, regenerate=True)
    # Always reset to the desired set. An empty desired list intentionally removes
    # legacy BP override graphs so native parent functions remain the live behavior.
    assets.reset_function_graphs(blueprint, FUNCTION_GRAPHS)

    assets.add_member_variable_if_missing(blueprint, "config_required", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "runtime_config_applied", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "config_apply_last_error", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "active_viewpoint", assets.pin_name())
    assets.add_member_variable_if_missing(blueprint, "front_fov_deg", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "capture_width", assets.pin_int())
    assets.add_member_variable_if_missing(blueprint, "capture_height", assets.pin_int())
    assets.add_member_variable_if_missing(blueprint, "front_offset_x_cm", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "front_offset_y_cm", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "front_offset_z_cm", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "front_rotation_pitch_deg", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "front_rotation_roll_deg", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "front_rotation_yaw_deg", assets.pin_real())
    assets.add_member_variable_if_missing(blueprint, "capture_output_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "capture_color_space", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "active_image_state_owner", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "bootstrap_sentinel_marker", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "viewpoint_count", assets.pin_int())
    assets.add_member_variable_if_missing(blueprint, "sensor_defaults_asset", assets.pin_object_reference(unreal.Object))

    viewpoint_struct = assets.load_asset(VIEWPOINT_STRUCT_PATH)
    assets.add_member_variable_if_missing(blueprint, "viewpoint_snapshot_template", assets.pin_struct(viewpoint_struct))

    for variable_name in (
        list(_load_bootstrap_defaults(assets).keys())
        + list(RUNTIME_VARIABLE_DEFAULTS.keys())
        + ["sensor_defaults_asset"]
    ):
        assets.set_variable_instance_editable(blueprint, variable_name, True)

    defaults_asset = assets.load_asset(SENSOR_DEFAULTS_PATH)
    merged_defaults = dict(_load_bootstrap_defaults(assets))
    merged_defaults.update(RUNTIME_VARIABLE_DEFAULTS)
    merged_defaults["sensor_defaults_asset"] = defaults_asset
    assets.set_blueprint_defaults(blueprint, merged_defaults)
    assets.stamp_common_metadata(
        blueprint,
        generated_by=SCRIPT_NAME,
        extra={
            "runtime_owner": "active_image_state",
            "function_graph_names": FUNCTION_GRAPHS,
            "public_function_surfaces": PUBLIC_FUNCTION_SURFACES,
            "sensor_defaults_asset": SENSOR_DEFAULTS_PATH,
            "viewpoint_policy_v1": "front_only",
            "bootstrap_source": "DA_SensorRigProfileDefault",
            "bootstrap_loaded_fields": sorted(_load_bootstrap_defaults(assets).keys()),
            "apply_behavior_source": "native_parent_component",
            "fixed_policy_v1": {
                "viewpoint_count": 1,
                "capture_output_mode": "image_bytes_png",
                "capture_color_space": "srgb",
            },
            "function_roles": {
                "ApplySensorRigConfig": "native runtime validation/apply entrypoint inherited from DroneSensorsRuntimeComponent",
                "CaptureActiveViewpointPngBytes": "capture the active front viewpoint and return PNG bytes for OBS assembly",
                "ListViewpoints": "report currently supported viewpoints for the runtime sensor owner",
                "GetViewpointSnapshot": "report current active image state for validation and OBS assembly",
            },
        },
    )
    assets.save_blueprint_asset(TARGET_ASSET_PATH, blueprint)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    if FUNCTION_GRAPHS:
        assets.ensure_blueprint_has_functions(loaded, FUNCTION_GRAPHS)
    assets.ensure_blueprint_has_variables(loaded, list(merged_defaults.keys()) + ["viewpoint_snapshot_template"])
    _validate_defaults(assets, loaded)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "function_graphs": FUNCTION_GRAPHS,
            "public_function_surfaces": PUBLIC_FUNCTION_SURFACES,
            "viewpoint_policy_v1": "front_only",
            "bootstrap_source": SENSOR_DEFAULTS_PATH,
            "runtime_state_owner": "BP_DroneSensors",
            "apply_behavior_source": "native_parent_component",
            "resolution": [merged_defaults["capture_width"], merged_defaults["capture_height"]],
        },
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
