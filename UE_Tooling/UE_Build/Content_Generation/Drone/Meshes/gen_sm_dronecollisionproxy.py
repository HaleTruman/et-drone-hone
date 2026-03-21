"""
Desired behavior:
- Regenerate `SM_DroneCollisionProxy` from scratch on every build run.
- Use a deterministic simplified primitive mesh as collision proxy source.
- Keep collision representation decoupled from visual mesh topology.

Interfaces:
- Produced asset path: `/Game/Drone_Content/Meshes/SM_DroneCollisionProxy`.
- Source mesh template: `/Engine/BasicShapes/Sphere.Sphere`.

Assumptions:
- Engine primitive source mesh exists.
- Proxy mesh does not need material customization for v1.

Success conditions:
- Static mesh asset exists and is valid.
- Asset is regenerated deterministically each build run.
- Metadata records source template and usage intent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_sm_dronecollisionproxy"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Meshes/SM_DroneCollisionProxy"
SOURCE_TEMPLATE_MESH_PATH = "/Engine/BasicShapes/Sphere.Sphere"


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

    proxy_mesh = assets.duplicate_static_mesh_asset(SOURCE_TEMPLATE_MESH_PATH, TARGET_ASSET_PATH, regenerate=True)
    assets.stamp_common_metadata(
        proxy_mesh,
        generated_by=SCRIPT_NAME,
        extra={
            "source_template": SOURCE_TEMPLATE_MESH_PATH,
            "usage": "collision_proxy_v1",
        },
    )
    assets.save_asset(TARGET_ASSET_PATH)

    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "StaticMesh")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {"source_template": SOURCE_TEMPLATE_MESH_PATH},
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
