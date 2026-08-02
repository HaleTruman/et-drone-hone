import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_contour_guided import (
    fit_contour_guided_c_shape)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)


class CShapeContourGuidedTests(unittest.TestCase):
    def test_outer_rails_guide_three_p90_centerlines(self):
        corners = np.array(((30, 30), (90, 30), (90, 90), (30, 90)))
        mask = np.zeros((120, 120), np.uint8)
        density = np.zeros(mask.shape, np.float64)
        for side in (0, 2, 3):
            start, end = tuple(corners[side]), tuple(corners[(side + 1) % 4])
            cv2.line(mask, start, end, 1, 11, cv2.LINE_8)
            cv2.line(density, start, end, 1.0, 3, cv2.LINE_8)
        density[mask != 0] = np.maximum(density[mask != 0], 0.1)

        result = fit_contour_guided_c_shape(CShapeDensityInput(
            "synthetic-c", mask, density))

        self.assertIsNotNone(result)
        self.assertEqual(len(result.guided_lines), 3)
        self.assertEqual(result.guided_extent.missing_side_index, 1)
        self.assertLess(
            len(result.outer_contour_uv), len(result.raw_outer_contour_uv))
        for line in result.guided_lines:
            self.assertEqual(len(line.rail_coefficients), 2)
            self.assertGreater(len(line.midpoint_samples_uv), 2)
            self.assertTrue(np.isfinite(line.angle_delta_degrees))


if __name__ == "__main__":
    unittest.main()
