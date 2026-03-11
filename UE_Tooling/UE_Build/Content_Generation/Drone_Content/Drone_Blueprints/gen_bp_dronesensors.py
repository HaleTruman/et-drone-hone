"""
Creates `BP_DroneSensors` at `/Game/Drone_Content/Blueprints/BP_DroneSensors`.

Role in pipeline:
- Defines sensor rig mount/viewpoint transforms (for example front/down/left/right).
- Provides default camera-oriented values (such as FOV targets) for viewpoint queries.
- Feeds viewpoint data used by the pawn and downstream sampling/capture orchestration.

Related generators:
- Interface: `gen_bpi_droneviewpointprovider.py`
- Data profile: `gen_da_sensorrigprofiledefault.py`
- Consumer blueprint: `gen_bp_dronepawn.py`

Runtime linkage:
- Supports stable named viewpoints that sample/capture systems can query consistently.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Blueprints/BP_DroneSensors"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
