"""
Creates `MI_DroneBody_Default` at `/Game/Drone_Content/Materials/MI_DroneBody_Default`.

Role in pipeline:
- Defines the default material instance for the drone body visual.
- References the base material and sets default parameter values for immediate visual consistency.
- Allows appearance iteration via instance parameters without altering the base material graph.

Related generators:
- Parent material: `gen_m_dronebody_base.py`
- Visual mesh + pawn consumers: `gen_sm_dronebody.py`, `gen_bp_dronepawn.py`

Runtime linkage:
- Assigned to drone visual mesh components to give the pawn a stable default look in-level.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Materials/MI_DroneBody_Default"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
