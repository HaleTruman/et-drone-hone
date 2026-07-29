"""Main Unreal entrypoint for gate dataset generation."""

from pathlib import Path
import sys


SCRIPT_PATH = Path(
    globals().get(
        "__file__",
        r"C:\Users\brend\Projects\UE_5\TestProject\ML\src\dataset_generation\generate.py",
    )
).resolve()
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # Unreal keeps imported Python modules alive between Execute Python Script runs.
    # Clear this project's modules so edits are picked up without restarting Unreal.
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]

import unreal

from src.dataset_generation import config
from src.dataset_generation.dataset_collection import outputs
from src.dataset_generation.dataset_collection import runtime as dataset_collection_runtime
from src.dataset_generation.gate_randomization import runtime as gate_randomization_runtime


def main():
    validate_config()
    dataset_dir = outputs.create_dataset_dir()
    unreal.log(f"Starting dataset generation in {dataset_dir}")
    start_batch(1)


def validate_config():
    if config.RUN_COUNT < 1:
        raise ValueError("RUN_COUNT must be at least 1.")
    if not 0.0 <= config.ZERO_GATE_FRAME_PERCENT <= 100.0:
        raise ValueError("ZERO_GATE_FRAME_PERCENT must be between 0.0 and 100.0.")
    if config.ZERO_GATE_CAMERA_POSITIONS < 1:
        raise ValueError("ZERO_GATE_CAMERA_POSITIONS must be at least 1.")


def start_batch(batch_number):
    unreal.log(f"Starting dataset generation batch {batch_number}/{config.RUN_COUNT}")
    gate_randomization_runtime.start(
        batch_number=batch_number,
        on_complete=start_collection,
    )


def start_collection(camera, gates, batch_number):
    dataset_collection_runtime.start(
        camera=camera,
        gates=gates,
        batch_number=batch_number,
        on_complete=finish_batch,
    )


def finish_batch(batch_number):
    if batch_number < config.RUN_COUNT:
        start_batch(batch_number + 1)
        return

    if config.SAVE_FRAMES:
        dataset_collection_runtime.restore_capture_every_frame()
    unreal.log(f"All {config.RUN_COUNT} dataset generation batch(es) complete")


if __name__ == "__main__":
    main()
