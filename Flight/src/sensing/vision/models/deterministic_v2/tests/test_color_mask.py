from pathlib import Path

import numpy as np
from PIL import Image

from vision.src.color_mask import ColorMasker
from vision.src.config import DEFAULT_PRESET_PATH, ProjectionConfig
from vision.src.schema import FrameMeta


def first_nonzero_lut_color(config: ProjectionConfig) -> tuple[int, tuple[int, int, int]]:
    with np.load(config.lut_path, allow_pickle=False) as payload:
        lut = payload["lut"]
        indices = np.flatnonzero(lut)
        assert indices.size > 0
        key = int(indices[0])
        value = int(lut[key])
    return value, ((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF)


def test_color_masker_processes_synthetic_rgb(tmp_path: Path):
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    expected, rgb = first_nonzero_lut_color(config)
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (4, 3), rgb).save(path, quality=100, subsampling=0)
    masker = ColorMasker(config)
    mask, payload = masker.process(
        FrameMeta(run_id="run-test", frame_ordinal=0, frame_id="frame_000000", source_path=str(path))
    )
    assert mask.shape == (3, 4)
    assert int(mask[0, 0]) == expected
    assert payload["nonzero_pixel_count"] == 12
