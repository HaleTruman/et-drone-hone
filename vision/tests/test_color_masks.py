from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from color_masks import mask_frames  # noqa: E402


class ColorMaskRuntimeTests(unittest.TestCase):
    def test_apply_lut_preserves_overlapping_bits(self) -> None:
        lut = np.zeros(4, dtype=np.uint8)
        lut[1] = 1
        lut[2] = 2 | 8
        rgb_keys = np.array([[0, 1], [2, 3]], dtype=np.uint32)

        mask_bits = mask_frames.apply_lut_to_rgb_keys(lut, rgb_keys)

        self.assertEqual(mask_bits.tolist(), [[0, 1], [10, 0]])
        stats = mask_frames.mask_stats(
            mask_bits,
            [
                {"prefix": "001", "bit": 0},
                {"prefix": "002", "bit": 1},
                {"prefix": "007", "bit": 3},
            ],
        )
        self.assertEqual(stats["selectedPixels"], 2)
        self.assertEqual(stats["overlapPixels"], 1)
        self.assertEqual(stats["layerPixelCounts"], {"001": 1, "002": 1, "007": 1})

    def test_pil_rgb_decode_uses_exact_rgb_key_packing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.png"
            image = Image.new("RGB", (2, 1))
            image.putdata([(0, 0, 1), (0, 0, 2)])
            image.save(path)

            rgb_keys, width, height = mask_frames.decode_rgb_keys(path)

        self.assertEqual((width, height), (2, 1))
        self.assertEqual(rgb_keys.tolist(), [[1, 2]])


if __name__ == "__main__":
    unittest.main()
