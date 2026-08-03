"""Compatibility entrypoint for gate dataset generation."""

from pathlib import Path
import runpy
import sys


SCRIPT_PATH = Path(globals().get("__file__", r"C:\Users\brend\Projects\UE_5\TestProject\ML\SyntheticData\src\dataset_generation\gate_randomization\run_gate_randomization.py")).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    runpy.run_module("SyntheticData.src.dataset_generation.generate", run_name="__main__")


if __name__ == "__main__":
    main()
