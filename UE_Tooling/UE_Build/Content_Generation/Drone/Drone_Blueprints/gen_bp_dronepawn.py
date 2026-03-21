"""
Desired behavior:
- Regenerate `BP_DronePawn` as the canonical drone runtime composition root.
- Host the generated movement, sensors, and telemetry modules as real Blueprint
  component templates on the pawn.
- Expose clean composition metadata without duplicating command, telemetry, or
  image behavior on the pawn itself.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Blueprints/BP_DronePawn`.
- Parent class: `Pawn`.
- Required function surfaces: none in the canonical composition-only shape.

Assumptions:
- v1 focuses on true hosted Blueprint component composition, not behavior
  implementation on the pawn.
- Runtime spawning/controller/sample systems resolve this pawn through stable path + metadata contract.

Success conditions:
- Pawn blueprint exists and compiles.
- Composition-root variables and hosted module metadata exist.
- Hosted movement, sensors, and telemetry component templates persist after save/reload.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_bp_dronepawn"
SCRIPT_VERSION = "1.2.0"
SCRIPT_DATE = "2026-03-15"

TARGET_ASSET_PATH = "/Game/Drone_Content/Blueprints/BP_DronePawn"

BODY_MESH_PATH = "/Game/Drone_Content/Meshes/SM_DroneBody"
COLLISION_MESH_PATH = "/Game/Drone_Content/Meshes/SM_DroneCollisionProxy"
BODY_MATERIAL_PATH = "/Game/Drone_Content/Materials/MI_DroneBody_Default"

MOVEMENT_BP_PATH = "/Game/Drone_Content/Blueprints/BP_DroneMovement_6DOF"
SENSORS_BP_PATH = "/Game/Drone_Content/Blueprints/BP_DroneSensors"
TELEMETRY_BP_PATH = "/Game/Drone_Content/Blueprints/BP_DroneTelemetrySampler"

MOVEMENT_COMPONENT_NAME = "DroneMovementModule"
SENSORS_COMPONENT_NAME = "DroneSensorsModule"
TELEMETRY_COMPONENT_NAME = "DroneTelemetryModule"

BPI_COMMAND_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver"
BPI_VIEWPOINT_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneViewpointProvider"
BPI_TELEMETRY_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneTelemetryProvider"

EXPECTED_FUNCTIONS: list[str] = []

VARIABLE_DEFAULTS = {
    "drone_identity_type": "uuid",
    "composition_root_role": "drone_runtime_container",
    "module_discovery_mode": "hosted_component_templates_v1",
    "movement_module_role_name": "movement",
    "sensor_module_role_name": "sensors",
    "telemetry_module_role_name": "telemetry",
    "primary_sensor_owner_role_name": "sensors",
    "movement_component_name": MOVEMENT_COMPONENT_NAME,
    "sensor_component_name": SENSORS_COMPONENT_NAME,
    "telemetry_component_name": TELEMETRY_COMPONENT_NAME,
    "primary_sensor_component_name": SENSORS_COMPONENT_NAME,
}

HOSTED_COMPONENT_SPECS = [
    {
        "role": "movement",
        "component_name": MOVEMENT_COMPONENT_NAME,
        "component_blueprint_path": MOVEMENT_BP_PATH,
    },
    {
        "role": "sensors",
        "component_name": SENSORS_COMPONENT_NAME,
        "component_blueprint_path": SENSORS_BP_PATH,
    },
    {
        "role": "telemetry",
        "component_name": TELEMETRY_COMPONENT_NAME,
        "component_blueprint_path": TELEMETRY_BP_PATH,
    },
]


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
        elif isinstance(expected, str):
            assets.require(str(actual) == expected, f"Unexpected default for {key}: {actual} != {expected}")
        else:
            assets.require(actual == expected, f"Unexpected default for {key}: {actual} != {expected}")


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    pawn = assets.create_blueprint(TARGET_ASSET_PATH, unreal.Pawn, regenerate=True)
    assets.reset_function_graphs(pawn, EXPECTED_FUNCTIONS)

    assets.add_member_variable_if_missing(pawn, "drone_identity_type", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "composition_root_role", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "module_discovery_mode", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "movement_module_role_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "sensor_module_role_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "telemetry_module_role_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "primary_sensor_owner_role_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "movement_component_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "sensor_component_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "telemetry_component_name", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "primary_sensor_component_name", assets.pin_string())

    assets.add_member_variable_if_missing(pawn, "body_mesh_asset", assets.pin_object_reference(unreal.StaticMesh))
    assets.add_member_variable_if_missing(pawn, "collision_mesh_asset", assets.pin_object_reference(unreal.StaticMesh))
    assets.add_member_variable_if_missing(pawn, "body_material_asset", assets.pin_object_reference(unreal.MaterialInterface))
    assets.add_member_variable_if_missing(pawn, "command_receiver_interface_asset", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "viewpoint_provider_interface_asset", assets.pin_string())
    assets.add_member_variable_if_missing(pawn, "telemetry_provider_interface_asset", assets.pin_string())

    for variable_name in list(VARIABLE_DEFAULTS.keys()) + [
        "body_mesh_asset",
        "collision_mesh_asset",
        "body_material_asset",
        "command_receiver_interface_asset",
        "viewpoint_provider_interface_asset",
        "telemetry_provider_interface_asset",
    ]:
        assets.set_variable_instance_editable(pawn, variable_name, True)

    merged_defaults = dict(VARIABLE_DEFAULTS)
    merged_defaults.update(
        {
            "body_mesh_asset": assets.load_asset(BODY_MESH_PATH),
            "collision_mesh_asset": assets.load_asset(COLLISION_MESH_PATH),
            "body_material_asset": assets.load_asset(BODY_MATERIAL_PATH),
            "command_receiver_interface_asset": BPI_COMMAND_PATH,
            "viewpoint_provider_interface_asset": BPI_VIEWPOINT_PATH,
            "telemetry_provider_interface_asset": BPI_TELEMETRY_PATH,
        }
    )

    assets.set_blueprint_defaults(pawn, merged_defaults)
    hosted_components = assets.rebuild_blueprint_hosted_components(pawn, HOSTED_COMPONENT_SPECS)
    assets.stamp_common_metadata(
        pawn,
        generated_by=SCRIPT_NAME,
        extra={
            "contract_role": "composition_root_only",
            "function_names": EXPECTED_FUNCTIONS,
            "composition_note": (
                "BP_DronePawn no longer declares direct command, pose, viewpoint, or telemetry "
                "function surfaces. Runtime behavior is owned by the hosted movement, sensors, "
                "and telemetry component templates on this pawn."
            ),
            "module_discovery": {
                "mode": "hosted_component_templates_v1",
                "movement_role": "movement",
                "sensor_role": "sensors",
                "telemetry_role": "telemetry",
                "primary_sensor_owner_role": "sensors",
                "movement_component_name": MOVEMENT_COMPONENT_NAME,
                "sensor_component_name": SENSORS_COMPONENT_NAME,
                "telemetry_component_name": TELEMETRY_COMPONENT_NAME,
                "primary_sensor_component_name": SENSORS_COMPONENT_NAME,
            },
            "asset_references": {
                "body_mesh": BODY_MESH_PATH,
                "collision_mesh": COLLISION_MESH_PATH,
                "body_material": BODY_MATERIAL_PATH,
                "interfaces": [BPI_COMMAND_PATH, BPI_VIEWPOINT_PATH, BPI_TELEMETRY_PATH],
            },
            "hosted_components": HOSTED_COMPONENT_SPECS,
        },
    )
    assets.save_blueprint_asset(TARGET_ASSET_PATH, pawn)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    if EXPECTED_FUNCTIONS:
        assets.ensure_blueprint_has_functions(loaded, EXPECTED_FUNCTIONS)
    assets.ensure_blueprint_has_variables(loaded, list(merged_defaults.keys()))
    persisted_hosted_components = assets.ensure_blueprint_has_hosted_components(loaded, HOSTED_COMPONENT_SPECS)
    _validate_defaults(assets, loaded)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "functions": EXPECTED_FUNCTIONS,
            "body_mesh": BODY_MESH_PATH,
            "hosted_components_created": hosted_components,
            "hosted_components_persisted": persisted_hosted_components,
            "primary_sensor_component_name": SENSORS_COMPONENT_NAME,
            "interfaces": [BPI_COMMAND_PATH, BPI_VIEWPOINT_PATH, BPI_TELEMETRY_PATH],
        },
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
