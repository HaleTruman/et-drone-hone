import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_parallel_aligned import (
    fit_parallel_aligned_c_shape)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)


class CShapeParallelAlignedTests(unittest.TestCase):
    def test_parallel_consensus_guides_three_density_lines(self):
        corners = np.array(((30, 30), (90, 30), (90, 90), (30, 90)))
        mask = np.zeros((120, 120), np.uint8)
        density = np.zeros(mask.shape, np.float64)
        for side in (0, 2, 3):
            start, end = tuple(corners[side]), tuple(corners[(side + 1) % 4])
            cv2.line(mask, start, end, 1, 11, cv2.LINE_8)
            cv2.line(density, start, end, 1.0, 3, cv2.LINE_8)
        density[mask != 0] = np.maximum(density[mask != 0], 0.1)

        result = fit_parallel_aligned_c_shape(CShapeDensityInput(
            "synthetic-c", mask, density))

        self.assertIsNotNone(result)
        self.assertEqual(len(result.parallel_lines), 3)
        self.assertEqual(result.parallel_extent.missing_side_index, 1)
        for line in result.parallel_lines:
            self.assertTrue(np.isfinite(line.rail_angle_gap_degrees))
            self.assertTrue(np.isfinite(line.p90_angle_delta_degrees))
            self.assertTrue(np.isfinite(line.p90_rmse_px))


if __name__ == "__main__":
    unittest.main()
