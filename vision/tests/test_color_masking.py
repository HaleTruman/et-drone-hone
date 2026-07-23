from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.src.color_masking import ColorMasker
from vision.src.schema import PipelinePreset, SourceFrame, utc_now


VISION_ROOT = Path(__file__).resolve().parents[1]


def _rgb_from_key(key: int) -> tuple[int, int, int]:
    return ((key >> 16) & 255, (key >> 8) & 255, key & 255)


class ColorMaskingTests(unittest.TestCase):
    def test_lut_shape_and_dtype(self) -> None:
        path = VISION_ROOT / "assets" / "color_lut_v1.npz"
        with np.load(path, allow_pickle=False) as payload:
            lut = payload["lut"]
            self.assertEqual(lut.shape, (16777216,))
            self.assertEqual(lut.dtype, np.uint8)

    def test_synthetic_rgb_image_maps_to_expected_bitfields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            preset = PipelinePreset.from_path(VISION_ROOT / "assets" / "pipeline_presets.json")
            lut_path = VISION_ROOT / preset.lut["path"]
            with np.load(lut_path, allow_pickle=False) as payload:
                lut = payload["lut"]
                key_bit0 = int(np.flatnonzero(lut & np.uint8(1 << 0))[0])
                key_bit1 = int(np.flatnonzero(lut & np.uint8(1 << 1))[0])
                key_bit2 = int(np.flatnonzero(lut & np.uint8(1 << 2))[0])
                key_bit3 = int(np.flatnonzero(lut & np.uint8(1 << 3))[0])
            rgb = np.array(
                [
                    [_rgb_from_key(key_bit0), _rgb_from_key(key_bit1), (0, 0, 0)],
                    [_rgb_from_key(key_bit2), _rgb_from_key(key_bit3), (0, 0, 0)],
                ],
                dtype=np.uint8,
            )
            image_path = tmp_path / "tiny.png"
            Image.fromarray(rgb).save(image_path)
            source = SourceFrame(
                run_id="run-test",
                frame_ordinal=0,
                frame_id="frame_000000",
                source_path=str(image_path),
                image_width=3,
                image_height=2,
                created_at=utc_now(),
            )
            frame = ColorMasker(preset, VISION_ROOT).process(source)
            self.assertTrue(int(frame.mask_bits[0, 0]) & 1)
            self.assertTrue(int(frame.mask_bits[0, 1]) & 2)
            self.assertTrue(int(frame.mask_bits[1, 0]) & 4)
            self.assertTrue(int(frame.mask_bits[1, 1]) & 8)
            self.assertEqual(int(frame.mask_bits[0, 2]), 0)
            self.assertGreaterEqual(frame.layer_pixel_counts["001"], 1)
            self.assertGreaterEqual(frame.layer_pixel_counts["002"], 1)
            self.assertGreaterEqual(frame.layer_pixel_counts["003"], 1)
            self.assertGreaterEqual(frame.layer_pixel_counts["007"], 1)


if __name__ == "__main__":
    unittest.main()
