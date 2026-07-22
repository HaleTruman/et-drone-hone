from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))
sys.path.insert(0, str(APP_ROOT / "tools"))

from mask_bbox.maskbits_input import build_maskbits_context, layer_hits_from_mask_bits, load_mask_bits  # noqa: E402


class MaskbitsBboxInputTests(unittest.TestCase):
    def test_layer_hits_preserve_bits_and_disable_review_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "mask_manifest.json"
            manifest = {
                "frameCount": 1,
                "width": 3,
                "height": 2,
                "lut": {
                    "sha256": "fake-lut",
                    "metadata": {
                        "layers": [
                            {"prefix": "001", "bit": 0, "rgbKeyCount": 10},
                            {"prefix": "002", "bit": 1, "rgbKeyCount": 20},
                            {"prefix": "003", "bit": 2, "rgbKeyCount": 30},
                        ]
                    },
                },
                "layers": [
                    {"prefix": "001", "bit": 0},
                    {"prefix": "002", "bit": 1},
                    {"prefix": "003", "bit": 2},
                ],
                "summary": {"layerPixelCounts": {"001": 2, "002": 2, "003": 1}},
                "frames": [],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config = {
                "classes": [
                    {"prefix": "001", "displayName": "one", "confidence": 0.99},
                    {"prefix": "002", "displayName": "two", "confidence": 0.90},
                    {"prefix": "003", "displayName": "three", "confidence": 0.80},
                ]
            }
            review = {"visualization": {"layerEnabledByPrefix": {"002": False}}}
            context = build_maskbits_context(config, review, manifest, manifest_path)
            mask_bits = np.array([[0, 1, 2], [3, 4, 7]], dtype=np.uint8)

            layer_hits = layer_hits_from_mask_bits(mask_bits, context)

            self.assertEqual(layer_hits.shape, (3, 2, 3))
            self.assertEqual(layer_hits[0].tolist(), [[False, True, False], [True, False, True]])
            self.assertFalse(layer_hits[1].any())
            self.assertEqual(layer_hits[2].tolist(), [[False, False, False], [False, True, True]])
            self.assertEqual(context["enabledPrefixes"], ["001", "003"])

    def test_load_mask_bits_rejects_wrong_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.maskbits.u8.bin"
            np.array([1, 2, 3], dtype=np.uint8).tofile(path)

            with self.assertRaises(ValueError):
                load_mask_bits({"maskBits": str(path)}, width=2, height=2)


if __name__ == "__main__":
    unittest.main()
