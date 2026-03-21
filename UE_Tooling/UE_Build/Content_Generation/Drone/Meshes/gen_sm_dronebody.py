"""
Desired behavior:
- Regenerate `SM_DroneBody` from scratch on every build run.
- Use a deterministic primitive source mesh and assign default drone body material instance.
- Keep visual mesh generation simple and robust in headless editor workflows.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Meshes/SM_DroneBody`.
- Source mesh template: `/Engine/BasicShapes/Cube.Cube`.
- Material dependency: `/Game/Drone_Content/Materials/MI_DroneBody_Default`.

Assumptions:
- Engine basic shape meshes are available in the target UE install.
- Static mesh material assignment API is available on duplicated mesh assets.

Success conditions:
- Static mesh asset exists and is a valid `StaticMesh`.
- Material slot 0 is assigned to `MI_DroneBody_Default`.
- Result metadata is stamped and saved.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_sm_dronebody"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Meshes/SM_DroneBody"
SOURCE_TEMPLATE_MESH_PATH = "/Engine/BasicShapes/Cube.Cube"
MATERIAL_INSTANCE_PATH = "/Game/Drone_Content/Materials/MI_DroneBody_Default"


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

    mesh = assets.duplicate_static_mesh_asset(SOURCE_TEMPLATE_MESH_PATH, TARGET_ASSET_PATH, regenerate=True)
    material_instance = assets.load_asset(MATERIAL_INSTANCE_PATH)
    assets.set_static_mesh_material(mesh, material_instance, slot=0)
    assets.stamp_common_metadata(
        mesh,
        generated_by=SCRIPT_NAME,
        extra={"source_template": SOURCE_TEMPLATE_MESH_PATH, "material_instance": MATERIAL_INSTANCE_PATH},
    )
    assets.save_asset(TARGET_ASSET_PATH)

    loaded = assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "StaticMesh")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"source_template": SOURCE_TEMPLATE_MESH_PATH, "material_instance": MATERIAL_INSTANCE_PATH},
    )
    assets.log(f"Mesh name: {loaded.get_name()}")
    assets.log("Success")


if __name__ == "__main__":
    run()
