import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput, solve_c_shape_pose)
from sensing.vision.models.deterministic_v3.src.gate_centerline_pnp import (
    K, OBJECT_POINTS_M)


class CShapeThreeLinePoseTests(unittest.TestCase):
    def test_projects_missing_side_from_three_visible_sides(self):
        rvec = np.array([[np.pi], [0.0], [0.0]])
        tvec = np.array([[0.2], [-0.1], [10.0]])
        projected, _ = cv2.projectPoints(OBJECT_POINTS_M, rvec, tvec, K, None)
        corners = np.rint(projected.reshape(4, 2)).astype(np.int32)
        mask = np.zeros((360, 640), np.uint8)
        for side in (0, 2, 3):
            cv2.line(mask, tuple(corners[side]), tuple(corners[(side + 1) % 4]),
                     1, 5, cv2.LINE_8)

        result = solve_c_shape_pose(CShapeDensityInput(
            "synthetic-c", mask, mask.astype(np.float64)))

        self.assertIsNotNone(result)
        self.assertEqual(result.instance_id, "synthetic-c")
        self.assertEqual(result.missing_side_index, 1)
        self.assertEqual(len(result.visible_lines), 3)
        self.assertTrue(result.candidates)
        self.assertLess(min(item.line_rmse_px for item in result.candidates), 1e-5)

    def test_rejects_shape_without_a_quadrilateral_scaffold(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[2, 2] = mask[2, 17] = mask[17, 10] = 1
        self.assertIsNone(solve_c_shape_pose(
            CShapeDensityInput("triangle", mask, mask.astype(float))))


if __name__ == "__main__":
    unittest.main()
