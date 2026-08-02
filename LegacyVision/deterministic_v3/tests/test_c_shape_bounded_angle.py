import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_bounded_angle import (
    bounded_axial_direction, fit_bounded_angle_c_shape)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)


class CShapeBoundedAngleTests(unittest.TestCase):
    def test_weighted_angular_correction_is_capped(self):
        p90 = np.asarray((1.0, 0.0))
        contour = np.asarray((0.5, np.sqrt(3) / 2))

        _, raw, weighted, applied = bounded_axial_direction(p90, contour)

        self.assertAlmostEqual(np.degrees(raw), 60.0)
        self.assertAlmostEqual(np.degrees(weighted), 30.0)
        self.assertAlmostEqual(np.degrees(applied), 15.0)

    def test_bounded_fit_preserves_three_lines(self):
        corners = np.array(((30, 30), (90, 30), (90, 90), (30, 90)))
        mask = np.zeros((120, 120), np.uint8)
        density = np.zeros(mask.shape, np.float64)
        for side in (0, 2, 3):
            start, end = tuple(corners[side]), tuple(corners[(side + 1) % 4])
            cv2.line(mask, start, end, 1, 11, cv2.LINE_8)
            cv2.line(density, start, end, 1.0, 3, cv2.LINE_8)
        density[mask != 0] = np.maximum(density[mask != 0], 0.1)

        result = fit_bounded_angle_c_shape(CShapeDensityInput(
            "synthetic-c", mask, density))

        self.assertIsNotNone(result)
        self.assertEqual(len(result.bounded_lines), 3)
        self.assertEqual(result.bounded_extent.missing_side_index, 1)
        for line in result.bounded_lines:
            self.assertLessEqual(abs(line.applied_delta_degrees), 15.0 + 1e-9)
            self.assertLessEqual(abs(line.applied_delta_degrees),
                                 abs(line.raw_contour_delta_degrees) + 1e-9)
            self.assertTrue(np.isfinite(line.p90_rmse_px))


if __name__ == "__main__":
    unittest.main()
