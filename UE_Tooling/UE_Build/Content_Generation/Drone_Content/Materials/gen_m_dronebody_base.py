"""
Creates `M_DroneBody_Base` at `/Game/Drone_Content/Materials/M_DroneBody_Base`.

Role in pipeline:
- Defines the base drone-body material graph used by default drone visuals.
- Provides a stable parent for material instances so appearance can be tuned without rebuilding the base graph.
- Keeps rendering setup modular and reusable across drone variants.

Related generators:
- Material instance: `gen_mi_dronebody_default.py`
- Visual mesh consumer path: `gen_sm_dronebody.py` + `gen_bp_dronepawn.py`

Runtime linkage:
- The pawn's visible mesh path uses this base material through an instance-based assignment.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Materials/M_DroneBody_Base"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
