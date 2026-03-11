"""
Creates `DA_SensorRigProfileDefault` at `/Game/Drone_Content/Data/DA_SensorRigProfileDefault`.

Role in pipeline:
- Stores default sensor rig profile data (mount names/offsets and camera-oriented defaults).
- Initializes viewpoint behavior for sensor rig and pawn integration.
- Provides a stable profile baseline that runtime config flows can override in memory.

Related generators:
- Sensor rig: `gen_bp_dronesensors.py`
- Pawn integration: `gen_bp_dronepawn.py`
- Viewpoint contract: `gen_bpi_droneviewpointprovider.py`
- Runtime override path (future): `io/Data_Config/gen_bp_setdataconfig.py`

Runtime linkage:
- Used to seed named viewpoints for capture/query pipelines before runtime overrides are applied.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Data/DA_SensorRigProfileDefault"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
