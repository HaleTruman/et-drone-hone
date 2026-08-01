import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.gate_centerline_pnp import (
    GateCenterlineQuad, K, OBJECT_POINTS_M, solve_gate_pose)


class GateCenterlinePnpTests(unittest.TestCase):
    def test_recovers_known_camera_pose(self):
        expected_rvec = np.array([[0.08], [-0.12], [0.03]], np.float64)
        expected_tvec = np.array([[0.25], [-0.10], [8.0]], np.float64)
        projected, _ = cv2.projectPoints(
            OBJECT_POINTS_M, expected_rvec, expected_tvec, K, None)
        quad = GateCenterlineQuad(
            "gate-7", tuple(map(tuple, projected.reshape(4, 2))))

        pose = solve_gate_pose(quad)

        self.assertEqual(pose.instance_id, "gate-7")
        self.assertGreater(pose.confidence, 0.999)
        self.assertLess(pose.reprojection_rmse_px, 1e-6)
        self.assertTrue(np.allclose(pose.tvec_camera_m, expected_tvec.ravel()))

    def test_rejects_any_input_other_than_four_finite_points(self):
        pose = solve_gate_pose(GateCenterlineQuad("bad", ((1.0, 2.0),)))
        self.assertIsNone(pose.rvec_camera)
        self.assertEqual(pose.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
