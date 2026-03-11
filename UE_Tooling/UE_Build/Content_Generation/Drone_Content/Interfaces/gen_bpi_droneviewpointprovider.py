"""
Creates `BPI_DroneViewpointProvider` at `/Game/Drone_Content/Interfaces/BPI_DroneViewpointProvider`.

Role in pipeline:
- Defines how the drone exposes named viewpoint transforms and camera-oriented parameters.
- Provides a stable API between sensor rig definitions and scene capture orchestration.
- Allows capture logic to remain independent of direct component hierarchy assumptions.

Related generators:
- Sensor rig: `gen_bp_dronesensors.py`
- Implementer: `gen_bp_dronepawn.py`
- Consumer: `io/Data_Interface/gen_bp_samplemanager.py`
- Default profile: `gen_da_sensorrigprofiledefault.py`

Runtime linkage:
- Sample/capture systems query this interface to position scene capture viewpoints consistently.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneViewpointProvider"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
