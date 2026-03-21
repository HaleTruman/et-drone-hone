"""
Desired behavior:
- Regenerate `MI_DroneBody_Default` as the default instance of `M_DroneBody_Base`.
- Apply conservative default parameter values for immediate visibility in-map.
- Keep instance setup deterministic for mesh/pawn assignment.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Materials/MI_DroneBody_Default`.
- Depends on parent material: `/Game/Drone_Content/Materials/M_DroneBody_Base`.

Assumptions:
- Parent material is generated before this script.
- Material instance parameter APIs are available in `MaterialEditingLibrary`.

Success conditions:
- Material instance asset exists.
- Parent material reference is set correctly.
- Default parameter values are applied (or hard-fail if API calls fail).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_mi_dronebody_default"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Materials/MI_DroneBody_Default"
PARENT_MATERIAL_PATH = "/Game/Drone_Content/Materials/M_DroneBody_Base"


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

    parent = assets.load_asset(PARENT_MATERIAL_PATH)
    instance = assets.create_material_instance(TARGET_ASSET_PATH, regenerate=True)
    instance.set_editor_property("parent", parent)

    assets.require(
        hasattr(unreal.MaterialEditingLibrary, "set_material_instance_vector_parameter_value"),
        "MaterialEditingLibrary.set_material_instance_vector_parameter_value is unavailable.",
    )
    assets.require(
        hasattr(unreal.MaterialEditingLibrary, "set_material_instance_scalar_parameter_value"),
        "MaterialEditingLibrary.set_material_instance_scalar_parameter_value is unavailable.",
    )

    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
        instance,
        "BaseColor",
        unreal.LinearColor(0.15, 0.35, 0.90, 1.0),
    )
    unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(instance, "Roughness", 0.45)
    unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(instance, "Metallic", 0.10)

    assets.stamp_common_metadata(
        instance,
        generated_by=SCRIPT_NAME,
        extra={"parent_material": PARENT_MATERIAL_PATH},
    )
    assets.save_asset(TARGET_ASSET_PATH)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "MaterialInstanceConstant")
    loaded_parent = loaded.get_editor_property("parent")
    assets.require(loaded_parent is not None, "Material instance parent is not set")
    assets.require(
        str(loaded_parent.get_path_name()).startswith(PARENT_MATERIAL_PATH),
        f"Unexpected parent material: {loaded_parent.get_path_name()}",
    )
    assets.log_result(TARGET_ASSET_PATH, "regenerated", {"parent_material": PARENT_MATERIAL_PATH})
    assets.log("Success")


if __name__ == "__main__":
    run()
