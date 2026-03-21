"""
Desired behavior:
- Regenerate `BP_DroneTelemetrySampler` as the canonical top-level telemetry runtime module.
- Expose one v1 function surface: `GetTelemetrySnapshot`.
- Encode telemetry policy defaults: all visible meshes, cm units, 3-decimal precision, no LOS filtering.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Blueprints/BP_DroneTelemetrySampler`.
- Parent class: `ActorComponent`.
- Uses the canonical top-level telemetry struct schema asset.

Assumptions:
- Telemetry is captured on trigger and computed fully per request.
- Pose-aligned telemetry fields now travel inside `ST_DroneTelemetrySnapshot`.
- Variable-length mesh proximity records remain valid within the telemetry snapshot payload.

Success conditions:
- Blueprint exists and compiles.
- Required function graph + telemetry policy variables are present.
- Canonical telemetry struct-template variable is present for schema traceability.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_bp_dronetelemetrysampler"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-15"

TARGET_ASSET_PATH = "/Game/Drone_Content/Blueprints/BP_DroneTelemetrySampler"
EXPECTED_FUNCTIONS = ["GetTelemetrySnapshot"]

TELEMETRY_STRUCT_PATH = "/Game/Drone_Content/Structs/ST_DroneTelemetrySnapshot"

VARIABLE_DEFAULTS = {
    "telemetry_units": "cm",
    "telemetry_precision_decimals": 3,
    "include_all_visible_meshes": True,
    "distance_center_mode": "pivot_origin",
    "distance_edge_mode": "nearest_triangle_edge",
    "line_of_sight_filtering": False,
    "telemetry_array_mode": "variable_length",
    "telemetry_schema_version": "1.0",
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
        assets.require(actual == expected, f"Unexpected default for {key}: {actual} != {expected}")


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    blueprint = assets.create_blueprint(TARGET_ASSET_PATH, unreal.ActorComponent, regenerate=True)
    assets.reset_function_graphs(blueprint, EXPECTED_FUNCTIONS)

    assets.add_member_variable_if_missing(blueprint, "telemetry_units", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "telemetry_precision_decimals", assets.pin_int())
    assets.add_member_variable_if_missing(blueprint, "include_all_visible_meshes", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "distance_center_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "distance_edge_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "line_of_sight_filtering", assets.pin_bool())
    assets.add_member_variable_if_missing(blueprint, "telemetry_array_mode", assets.pin_string())
    assets.add_member_variable_if_missing(blueprint, "telemetry_schema_version", assets.pin_string())

    telemetry_struct = assets.load_asset(TELEMETRY_STRUCT_PATH)
    assets.add_member_variable_if_missing(blueprint, "telemetry_snapshot_template", assets.pin_struct(telemetry_struct))

    for variable_name in VARIABLE_DEFAULTS:
        assets.set_variable_instance_editable(blueprint, variable_name, True)

    assets.set_blueprint_defaults(blueprint, VARIABLE_DEFAULTS)
    assets.stamp_common_metadata(
        blueprint,
        generated_by=SCRIPT_NAME,
        extra={
            "contract_role": "canonical_top_level_telemetry_module",
            "function_names": EXPECTED_FUNCTIONS,
            "compatibility_note": (
                "Pose-aligned telemetry now travels inside ST_DroneTelemetrySnapshot. "
                "Legacy pose-only surfaces remain elsewhere temporarily until full convergence."
            ),
            "policy": {
                "mesh_selection": "all_visible_meshes",
                "distance_units": "cm",
                "precision_decimals": 3,
                "line_of_sight_filtering": False,
            },
            "struct_templates": {
                "telemetry": TELEMETRY_STRUCT_PATH,
            },
        },
    )
    assets.save_blueprint_asset(TARGET_ASSET_PATH, blueprint)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    assets.ensure_blueprint_has_functions(loaded, EXPECTED_FUNCTIONS)
    assets.ensure_blueprint_has_variables(
        loaded,
        list(VARIABLE_DEFAULTS.keys())
        + [
            "telemetry_snapshot_template",
        ],
    )
    _validate_defaults(assets, loaded)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "functions": EXPECTED_FUNCTIONS,
            "distance_units": "cm",
            "precision_decimals": 3,
            "mesh_policy": "all_visible_meshes",
        },
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
