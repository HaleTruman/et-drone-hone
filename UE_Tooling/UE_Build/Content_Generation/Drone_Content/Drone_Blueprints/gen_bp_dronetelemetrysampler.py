"""
Creates `BP_DroneTelemetrySampler` at `/Game/Drone_Content/Blueprints/BP_DroneTelemetrySampler`.

Role in pipeline:
- Telemetry module intended to sample pose/velocity/angular state from the drone.
- Provides the pawn-side source for pose and extended telemetry interface responses.
- Enables synchronized data reads for capture pipelines.

Related generators:
- Interfaces: `gen_bpi_droneposeprovider.py`, `gen_bpi_dronetelemetryprovider.py`
- Consumer blueprints: `gen_bp_dronepawn.py`, `io/Data_Interface/gen_bp_samplemanager.py`

Runtime linkage:
- Sample manager and related data interfaces query this data path through pawn BPIs.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Blueprints/BP_DroneTelemetrySampler"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
