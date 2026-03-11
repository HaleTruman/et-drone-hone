"""
Creates `BPI_DroneTelemetryProvider` at `/Game/Drone_Content/Interfaces/BPI_DroneTelemetryProvider`.

Role in pipeline:
- Defines the extended telemetry contract beyond basic pose.
- Preserves a stable query surface while telemetry detail expands over time.
- Prevents downstream data collection systems from depending on pawn/module internals.

Related generators:
- Implementer: `gen_bp_dronepawn.py` (typically delegating to telemetry sampler)
- Telemetry module: `gen_bp_dronetelemetrysampler.py`
- Consumer: `io/Data_Interface/gen_bp_samplemanager.py`

Runtime linkage:
- Sampling/data interfaces request telemetry through this contract for observation metadata.

Current status:
- Placeholder generator (logs TODO only).
"""

import unreal

TARGET_ASSET_PATH = "/Game/Drone_Content/Interfaces/BPI_DroneTelemetryProvider"


def run() -> None:
    unreal.log_warning(f"[Content_Generation] TODO generate or update {TARGET_ASSET_PATH}")


if __name__ == "__main__":
    run()
