"""
Creates `BP_SampleManager` at `/Game/io/Data_Interface/BP_SampleManager`.

Role in pipeline:
- In-engine observation producer that orchestrates atomic capture snapshots.
- Queries pose/viewpoint/telemetry from drone-facing interfaces, triggers scene capture, and bundles outputs.
- Stamps outputs with traceability fields (`capture_id`, `run_id`, `config_id`, timestamps) for replayable datasets.

Related generators:
- Drone providers:
  - `Drone/Interfaces/gen_bpi_droneviewpointprovider.py`
  - `Drone/Interfaces/gen_bpi_dronetelemetryprovider.py`
- Config runtime:
  - `../Data_Config/gen_st_runconfig.py`
  - `../Data_Config/gen_bp_setdataconfig.py`
- Control/spawn peers:
  - `../Drone_Controller/gen_bp_dronespawner.py`
  - `../Drone_Controller/gen_bp_dronecontroller.py`

Runtime linkage:
- Emits observation payloads consumed by tooling-side sampler/manager and transport bridge layers.

Current status:
- Generated on top of native `SampleManagerRuntimeActor` so orchestration behavior lives in runtime code.
- Blueprint generation only stamps defaults/metadata and preserves deterministic discovery identity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import unreal

TARGET_ASSET_PATH = "/Game/io/Data_Interface/BP_SampleManager"
SCRIPT_NAME = "gen_bp_samplemanager"
SCRIPT_VERSION = "1.3.0"
SCRIPT_DATE = "2026-03-16"

# Keep native parent implementations for orchestration/config-ref function surfaces.
# Creating Blueprint override graphs here would shadow truthful native runtime behavior.
EXPECTED_FUNCTIONS: list[str] = []
PUBLIC_FUNCTION_SURFACES = [
    "CaptureNow",
    "ResolveActiveConfigReference",
]

DISCOVERY_ID = "BP_SampleManager_Main"
DISCOVERY_CONTRACT = {
    "identity_role": "sample_orchestrator",
    "identity_id": DISCOVERY_ID,
    "discovery_mode": "deterministic_level_singleton_v1",
    "placement_owner": "run_io_build",
    "target_actor_label": DISCOVERY_ID,
}

DEFAULTS = {
    "capture_output_mode": "image_bytes_png",
    "default_width": 1280,
    "default_height": 720,
    "default_fov_deg": 90.0,
    "runtime_discovery_role": "sample_orchestrator",
    "runtime_discovery_id": DISCOVERY_ID,
    "runtime_discovery_mode": "deterministic_level_singleton_v1",
    "config_reference_source_role": "set_config_ingress",
    "config_reference_source_actor_label": "BP_SetDataConfig_Main",
    "config_reference_read_function": "GetActiveConfigReference",
    "config_reference_read_mode": "runtime_query_per_capture_v1",
    "cache_config_reference": False,
}


def _load_assets_module():
    module_path = Path(__file__).resolve().parents[1] / "assemble_io_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_io_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> None:
    assets = _load_assets_module()
    assets.log_script_metadata()
    assets.log(f"Script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")

    parent_class = getattr(unreal, "SampleManagerRuntimeActor", None)
    assets.require(
        parent_class is not None,
        "SampleManagerRuntimeActor class is unavailable. Build the UE_Drone_Env module before generating BP_SampleManager.",
    )

    blueprint = assets.create_blueprint(TARGET_ASSET_PATH, parent_class, regenerate=True)
    assets.reset_function_graphs(blueprint, EXPECTED_FUNCTIONS)
    assets.set_blueprint_defaults(blueprint, DEFAULTS)

    assets.set_asset_metadata(blueprint, "io.generated_by", SCRIPT_NAME)
    assets.set_asset_metadata(blueprint, "io.script_version", SCRIPT_VERSION)
    assets.set_asset_metadata(blueprint, "io.script_date", SCRIPT_DATE)
    assets.set_asset_metadata(blueprint, "io.runtime_owner", "BP_SampleManager")
    assets.set_asset_metadata(blueprint, "io.orchestration_owner", "atomic_capture_transaction")
    assets.set_asset_metadata(blueprint, "io.function_graph_names", "[]")
    assets.set_asset_metadata_json(blueprint, "io.public_function_surfaces", PUBLIC_FUNCTION_SURFACES)
    assets.set_asset_metadata(
        blueprint,
        "io.orchestration_note",
        "Capture orchestration behavior is owned by native parent SampleManagerRuntimeActor. Blueprint generation does not author graph bodies.",
    )
    assets.set_asset_metadata_json(blueprint, "io.runtime_discovery", DISCOVERY_CONTRACT)
    assets.set_asset_metadata_json(
        blueprint,
        "io.active_config_reference_contract",
        {
            "source_role": DEFAULTS["config_reference_source_role"],
            "source_actor_label": DEFAULTS["config_reference_source_actor_label"],
            "read_function": DEFAULTS["config_reference_read_function"],
            "read_mode": DEFAULTS["config_reference_read_mode"],
            "cache_config_reference": DEFAULTS["cache_config_reference"],
            "required_fields": ["config_id", "config_hash"],
        },
    )
    assets.save_blueprint_asset(TARGET_ASSET_PATH, blueprint)
    assets.ensure_loaded_asset_type(TARGET_ASSET_PATH, "Blueprint")
    assets.log_result(
        TARGET_ASSET_PATH,
        "regenerated",
        {
            "function_graphs": EXPECTED_FUNCTIONS,
            "public_function_surfaces": PUBLIC_FUNCTION_SURFACES,
            "defaults": DEFAULTS,
            "runtime_discovery": DISCOVERY_CONTRACT,
            "parent_class": "SampleManagerRuntimeActor",
        },
    )


if __name__ == "__main__":
    run()
