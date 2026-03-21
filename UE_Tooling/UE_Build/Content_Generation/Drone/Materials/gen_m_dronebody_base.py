"""
Desired behavior:
- Regenerate `M_DroneBody_Base` as the canonical base material for drone visuals.
- Keep graph minimal: vector parameter for base color + scalar parameters for roughness/metallic.
- Provide a stable parent for `MI_DroneBody_Default`.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Materials/M_DroneBody_Base`.
- Parameter contract: `BaseColor`, `Roughness`, `Metallic`.

Assumptions:
- Material editing library is available in headless Unreal editor mode.
- Minimal parameterized graph is sufficient for v1 visuals.

Success conditions:
- Material asset exists and recompiles.
- Parameter expressions are created and connected to expected material properties.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_m_dronebody_base"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Materials/M_DroneBody_Base"


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

    material = assets.create_material(TARGET_ASSET_PATH, regenerate=True)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    base_color_expr = unreal.MaterialEditingLibrary.create_material_expression(
        material,
        unreal.MaterialExpressionVectorParameter,
        -450,
        -100,
    )
    assets.require(base_color_expr is not None, "Failed creating BaseColor expression")
    base_color_expr.set_editor_property("parameter_name", "BaseColor")
    base_color_expr.set_editor_property("default_value", unreal.LinearColor(0.20, 0.20, 0.20, 1.0))
    assets.require(
        unreal.MaterialEditingLibrary.connect_material_property(
            base_color_expr,
            "",
            unreal.MaterialProperty.MP_BASE_COLOR,
        ),
        "Failed connecting BaseColor to BaseColor property",
    )

    roughness_expr = unreal.MaterialEditingLibrary.create_material_expression(
        material,
        unreal.MaterialExpressionScalarParameter,
        -450,
        0,
    )
    assets.require(roughness_expr is not None, "Failed creating Roughness expression")
    roughness_expr.set_editor_property("parameter_name", "Roughness")
    roughness_expr.set_editor_property("default_value", 0.55)
    assets.require(
        unreal.MaterialEditingLibrary.connect_material_property(
            roughness_expr,
            "",
            unreal.MaterialProperty.MP_ROUGHNESS,
        ),
        "Failed connecting Roughness to Roughness property",
    )

    metallic_expr = unreal.MaterialEditingLibrary.create_material_expression(
        material,
        unreal.MaterialExpressionScalarParameter,
        -450,
        100,
    )
    assets.require(metallic_expr is not None, "Failed creating Metallic expression")
    metallic_expr.set_editor_property("parameter_name", "Metallic")
    metallic_expr.set_editor_property("default_value", 0.05)
    assets.require(
        unreal.MaterialEditingLibrary.connect_material_property(
            metallic_expr,
            "",
            unreal.MaterialProperty.MP_METALLIC,
        ),
        "Failed connecting Metallic to Metallic property",
    )

    unreal.MaterialEditingLibrary.recompile_material(material)
    assets.stamp_common_metadata(
        material,
        generated_by=SCRIPT_NAME,
        extra={"parameters": ["BaseColor", "Roughness", "Metallic"]},
    )
    assets.save_asset(TARGET_ASSET_PATH)

    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Material")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"parameters": ["BaseColor", "Roughness", "Metallic"]},
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
