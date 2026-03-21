"""
Desired behavior:
- Regenerate `DA_DroneMovementDefault` as the canonical movement-default data asset.
- Encode explicit sentinel placeholders so external validation can detect missing runtime override.
- Keep defaults minimal and deterministic for v1 command-only movement (no physics).

Interfaces:
- Produced asset path: `/Game/Drone_Content/Data/DA_DroneMovementDefault`.
- Consumed by movement module generation/runtime (`BP_DroneMovement_6DOF`).
- Intended to be overridden in-memory via runtime `SET_CONFIG`.

Assumptions:
- Data defaults are represented as a Blueprint-derived data asset for editable defaults in UE.
- Sentinel placeholders are allowed and expected in baseline assets.

Success conditions:
- Asset exists and compiles.
- Required movement default variables exist.
- Sentinel defaults are persisted and validated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_da_dronemovementdefault"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-12"

TARGET_ASSET_PATH = "/Game/Drone_Content/Data/DA_DroneMovementDefault"
CLASS_ASSET_PATH = "/Game/Drone_Content/Data/Classes/BP_DA_DroneMovementDefaultClass"

VARIABLE_DEFAULTS = {
    "config_required": True,
    "max_speed_cmps": -9999.0,
    "max_accel_cmps2": -9999.0,
    "max_yaw_degps": -9999.0,
    "max_pitch_degps": -9999.0,
    "max_roll_degps": -9999.0,
    "damping": -9999.0,
    "sentinel_marker": "__UNSET_BY_SET_CONFIG__",
}


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
    assets.require(bool(raw), "Missing drone.defaults metadata on DA_DroneMovementDefault")


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    class_blueprint, _ = assets.create_concrete_data_asset_class(
        class_asset_path=CLASS_ASSET_PATH,
        regenerate=True,
    )
    assets.stamp_common_metadata(
        class_blueprint,
        generated_by=SCRIPT_NAME,
        extra={
            "asset_kind": "DataAssetClass",
            "for_asset": TARGET_ASSET_PATH,
            "class_asset_path": CLASS_ASSET_PATH,
        },
    )
    assets.save_blueprint_asset(CLASS_ASSET_PATH, class_blueprint)

    data_asset = assets.create_data_asset(
        asset_path=TARGET_ASSET_PATH,
        class_asset_path=CLASS_ASSET_PATH,
        regenerate_asset=True,
        regenerate_class=False,
    )
    assets.stamp_common_metadata(
        data_asset,
        generated_by=SCRIPT_NAME,
        extra={
            "asset_kind": "DataDefault",
            "policy": "sentinel_placeholders_required",
            "defaults": VARIABLE_DEFAULTS,
            "movement_mode_v1": "minimal_deterministic_no_physics",
        },
    )
    assets.set_asset_metadata_json(data_asset, "drone.defaults", VARIABLE_DEFAULTS)
    assets.save_asset(TARGET_ASSET_PATH)

    loaded = assets.ensure_loaded_data_asset(
        TARGET_ASSET_PATH,
        expected_class_asset_path=CLASS_ASSET_PATH,
    )
    _validate_defaults_metadata(assets, loaded)
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "sentinel_defaults": VARIABLE_DEFAULTS,
            "class_asset_path": CLASS_ASSET_PATH,
        },
    )
    assets.log("Success")


if __name__ == "__main__":
    run()
