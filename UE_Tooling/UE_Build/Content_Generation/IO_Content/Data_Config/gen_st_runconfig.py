"""
Creates `ST_RunConfig` at `/Game/io/Data_Config/ST_RunConfig`.

Role in pipeline:
- UE-side typed struct for runtime configuration payloads (`SET_CONFIG`).
- Defines stable fields for movement tuning, sensor/viewpoint settings, telemetry selection, and traceability IDs.
- Serves as the typed contract boundary between network JSON and in-engine Blueprint application logic.

Related generators:
- Config endpoint: `gen_bp_setdataconfig.py`
- Runtime consumers: `../Drone_Controller/gen_bp_dronecontroller.py`, `../Data_Interface/gen_bp_samplemanager.py`
- Tooling compiler counterpart: `UE_Tooling/Config/RunConfig.py`

Runtime linkage:
- Used to parse/validate inbound config into structured data before in-memory override application.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/io/Data_Config/ST_RunConfig"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
