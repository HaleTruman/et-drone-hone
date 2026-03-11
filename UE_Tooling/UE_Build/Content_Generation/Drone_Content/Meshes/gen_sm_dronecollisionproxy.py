"""
Creates `SM_DroneCollisionProxy` at `/Game/Drone_Content/Meshes/SM_DroneCollisionProxy`.

Role in pipeline:
- Provides a simplified geometry asset intended for collision/physics stability.
- Decouples physics representation from visual mesh complexity.
- Enables visual iteration on `SM_DroneBody` without tightly coupling collision behavior.

Related generators:
- Visual mesh: `gen_sm_dronebody.py`
- Pawn consumer: `gen_bp_dronepawn.py`

Runtime linkage:
- Intended for invisible collision/physics usage while the visible mesh remains a separate asset path.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Meshes/SM_DroneCollisionProxy"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
