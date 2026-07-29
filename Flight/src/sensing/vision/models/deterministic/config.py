from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DeterministicVisionConfig:
    config_path: Path = BACKEND_DIR / "assets" / "config" / "alpha_classes.json"
    review_path: Path = BACKEND_DIR / "assets" / "review" / "review_state.json"
    lut_path: Path = BACKEND_DIR / "src" / "color_masks" / "artifacts" / "color_lut_v1.npz"
    scratch_root: Path = BACKEND_DIR / "assets" / "scratch"
    expected_width: int = 640
    expected_height: int = 360
    parallel_stateless: bool = True
    keep_scratch: bool = False
