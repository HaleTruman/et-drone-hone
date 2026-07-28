"""Compatibility entrypoint for gate dataset collection.

Run this from inside Unreal Editor with:

Tools > Execute Python Script

This delegates to `src.dataset_generation.generate`. The dataset collection
implementation lives in `src/dataset_generation/dataset_collection/`, while
`src/dataset_generation/gate_randomization/` owns gate placement.
"""

from pathlib import Path
import runpy
import sys


SCRIPT_PATH = Path(globals().get("__file__", r"C:\Users\brend\Projects\UE_5\TestProject\ML\src\dataset_generation\dataset_collection\collect_gate_dataset.py")).resolve()
ROOT = SCRIPT_PATH.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    runpy.run_module("src.dataset_generation.generate", run_name="__main__")


if __name__ == "__main__":
    main()
