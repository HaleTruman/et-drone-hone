"""
Creates `SM_DroneBody` at `/Game/Drone_Content/Meshes/SM_DroneBody`.

Role in pipeline:
- Provides the drone's primary visible static mesh asset.
- Supplies the geometry bound to default drone materials for in-world visualization.
- Serves as the visual body reference used by the pawn blueprint.

Related generators:
- Materials: `gen_m_dronebody_base.py`, `gen_mi_dronebody_default.py`
- Pawn consumer: `gen_bp_dronepawn.py`
- Collision counterpart: `gen_sm_dronecollisionproxy.py`

Runtime linkage:
- Used as the pawn's visible mesh while collision/physics can be split to a separate proxy asset.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Meshes/SM_DroneBody"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
