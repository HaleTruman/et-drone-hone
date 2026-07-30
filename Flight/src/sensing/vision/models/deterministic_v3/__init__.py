"""Deterministic Vision Pipeline"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deterministic_vision import DeterministicVision

__all__ = [
    "DeterministicVision"
]
