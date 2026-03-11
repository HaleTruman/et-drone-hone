"""
Creates `BPI_DronePoseProvider` at `/Game/Drone_Content/Interfaces/BPI_DronePoseProvider`.

Role in pipeline:
- Defines the pose query contract exposed by the drone to external systems.
- Expected to include a stable snapshot API for location/rotation and optional velocity fields.
- Keeps sampling/capture orchestration decoupled from pawn internals.

Related generators:
- Implementer: `gen_bp_dronepawn.py` (delegating to telemetry where appropriate)
- Telemetry source: `gen_bp_dronetelemetrysampler.py`
- Consumer: `io/Data_Interface/gen_bp_samplemanager.py`

Runtime linkage:
- Used during observation capture so image frames can be paired with aligned pose metadata.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Interfaces/BPI_DronePoseProvider"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
