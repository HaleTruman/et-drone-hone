import unittest

import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_decoupled_extension import (
    contour_exits_from_paths, extend_from_observed_paths,
    extend_to_longest_green_path, ray_contour_exit)
from sensing.vision.models.deterministic_v3.src.c_shape_finite_extent import (
    CShapeExtentResult)


class CShapeDecoupledExtensionTests(unittest.TestCase):
    def test_ray_exit_is_exact_contour_intersection(self):
        contour = np.asarray(((-2.0, -2.0), (12.0, -2.0),
                              (12.0, 12.0), (-2.0, 12.0)))

        exit_point = ray_contour_exit((0.0, 5.0), (1.0, 0.0), contour)

        np.testing.assert_allclose(exit_point, (12.0, 5.0))

    def test_both_exits_replace_mask_extent_with_contour_intersection(self):
        extent = CShapeExtentResult(
            "synthetic", 0.9, 1, 3, (), ((0.0, 0.0), (0.0, 10.0)),
            2.0, 4.0, (8.0, 8.0), ((8.0, 0.0), (8.0, 10.0)),
            ((10.0, 0.0), (10.0, 10.0)),
            ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        contour = np.asarray(((-2.0, -2.0), (12.0, -2.0),
                              (12.0, 12.0), (-2.0, 12.0)))

        exits = contour_exits_from_paths(extent, contour)

        np.testing.assert_allclose(exits, ((12.0, 0.0), (12.0, 10.0)))

    def test_extension_moves_only_along_fixed_observed_paths(self):
        extent = CShapeExtentResult(
            "synthetic", 0.9, 1, 3, (), ((0.0, 0.0), (0.0, 10.0)),
            2.0, 4.0, (10.0, 10.0), ((10.0, 0.0), (10.0, 10.0)),
            ((12.0, 0.0), (12.0, 10.0)),
            ((0.0, 0.0), (12.0, 0.0), (12.0, 10.0), (0.0, 10.0)))

        candidate = extend_from_observed_paths(extent, 2.0, 3.0)

        self.assertEqual(candidate.observed_exits_uv, extent.outer_exits_uv)
        np.testing.assert_allclose(
            candidate.extended_endpoints_uv, ((14.0, 0.0), (16.0, 10.0)))
        self.assertTrue(candidate.convex)
        self.assertGreater(candidate.area_px2, 0)

    def test_each_arm_receives_shortfall_from_scaled_longest_green(self):
        extent = CShapeExtentResult(
            "synthetic", 0.9, 1, 3, (), ((0.0, 0.0), (0.0, 10.0)),
            2.0, 4.0, (6.0, 8.0), ((6.0, 0.0), (8.0, 10.0)),
            ((8.0, 0.0), (10.0, 10.0)),
            ((0.0, 0.0), (8.0, 0.0), (10.0, 10.0), (0.0, 10.0)))

        candidate = extend_to_longest_green_path(
            extent, ((6.0, 0.0), (8.0, 10.0)))

        self.assertEqual(candidate.extension_scales, (0.75, 0.0))
        np.testing.assert_allclose(
            candidate.extended_endpoints_uv, ((7.5, 0.0), (8.0, 10.0)))


if __name__ == "__main__":
    unittest.main()
