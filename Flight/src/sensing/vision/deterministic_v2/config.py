from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DeterministicVisionV2Config:
    preset_path: Path = BACKEND_DIR / "assets" / "pipeline_presets.json"
    scratch_root: Path = BACKEND_DIR / "assets" / "scratch"
    expected_width: int = 640
    expected_height: int = 360
    debug: bool = False
    keep_scratch: bool = False
