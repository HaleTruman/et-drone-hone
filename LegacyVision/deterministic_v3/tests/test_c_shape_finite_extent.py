import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_finite_extent import (
    estimate_c_shape_extent)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)


class CShapeFiniteExtentTests(unittest.TestCase):
    def test_extends_missing_line_beyond_full_mask_boundary(self):
        expected = np.array(((30, 30), (90, 30), (90, 90), (30, 90)))
        mask = np.zeros((120, 120), np.uint8)
        for side in (0, 2, 3):
            cv2.line(mask, tuple(expected[side]),
                     tuple(expected[(side + 1) % 4]), 1, 9, cv2.LINE_8)

        result = estimate_c_shape_extent(CShapeDensityInput(
            "synthetic-c", mask, mask.astype(np.float64)))

        self.assertIsNotNone(result)
        self.assertEqual(result.missing_side_index, 1)
        self.assertEqual(result.spine_side_index, 3)
        self.assertGreater(result.half_width_px, 4.0)
        exits = np.asarray(result.outer_exits_uv)
        endpoints = np.asarray(result.extended_endpoints_uv)
        distances = np.linalg.norm(endpoints - exits, axis=1)
        np.testing.assert_allclose(
            distances, result.half_width_px, rtol=0, atol=1e-6)
        self.assertTrue(np.all(endpoints[:, 0] > exits[:, 0]))

    def test_rejects_non_quadrilateral_evidence(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[2, 2] = mask[2, 17] = mask[17, 10] = 1
        self.assertIsNone(estimate_c_shape_extent(
            CShapeDensityInput("triangle", mask, mask.astype(float))))


if __name__ == "__main__":
    unittest.main()
