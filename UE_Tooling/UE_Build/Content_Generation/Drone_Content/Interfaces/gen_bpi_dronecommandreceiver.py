"""
Creates `BPI_DroneCommandReceiver` at `/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver`.

Role in pipeline:
- Defines the control-ingress contract for normalized command application on the drone.
- Keeps command transport/controller layers decoupled from specific movement implementation details.
- Supports future command routing strategies without changing the pawn's external surface.

Related generators:
- Implementer: `gen_bp_dronepawn.py`
- Movement module target: `gen_bp_dronemovement_6dof.py`
- Upstream callers: `io/Drone_Controller/gen_bp_dronecontroller.py` and transport-facing flows

Runtime linkage:
- Command messages flow through controller/transport logic into this interface, then into movement logic.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
