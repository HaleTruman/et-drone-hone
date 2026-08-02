import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.c_shape_decoupled_extension import (
    fit_longest_green_extension)
from sensing.vision.models.deterministic_v3.src.c_shape_finite_extent import (
    CShapeExtentResult)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)
from sensing.vision.models.deterministic_v3.src.c_shape.c_shape_tailored_shortfall_baseline import (
    CONTOUR_ANGULAR_INFLUENCE, LONGEST_GREEN_TARGET_SCALE,
    MAX_CONTOUR_CORRECTION_DEGREES, extend_by_scaled_green_shortfall,
    fit_c_shape_geometry_baseline)


class CShapeGeometryBaselineTests(unittest.TestCase):
    def test_approved_settings_are_pinned(self):
        self.assertEqual(CONTOUR_ANGULAR_INFLUENCE, 0.50)
        self.assertEqual(MAX_CONTOUR_CORRECTION_DEGREES, 15.0)
        self.assertEqual(LONGEST_GREEN_TARGET_SCALE, 0.75)

    def test_tailored_shortfall_matches_approved_rule(self):
        extent = CShapeExtentResult(
            "synthetic", 0.9, 1, 3, (), ((0.0, 0.0), (0.0, 10.0)),
            2.0, 4.0, (6.0, 8.0), ((6.0, 0.0), (8.0, 10.0)),
            ((8.0, 0.0), (10.0, 10.0)),
            ((0.0, 0.0), (8.0, 0.0), (10.0, 10.0), (0.0, 10.0)))

        candidate = extend_by_scaled_green_shortfall(
            extent, ((6.0, 0.0), (8.0, 10.0)))

        self.assertEqual(candidate.extension_scales, (0.75, 0.0))
        np.testing.assert_allclose(
            candidate.extended_endpoints_uv, ((7.5, 0.0), (8.0, 10.0)))

    def test_pinned_baseline_matches_working_source(self):
        corners = np.array(((30, 30), (90, 30), (90, 90), (30, 90)))
        mask = np.zeros((120, 120), np.uint8)
        density = np.zeros(mask.shape, np.float64)
        for side in (0, 2, 3):
            start, end = tuple(corners[side]), tuple(corners[(side + 1) % 4])
            cv2.line(mask, start, end, 1, 11, cv2.LINE_8)
            cv2.line(density, start, end, 1.0, 3, cv2.LINE_8)
        density[mask != 0] = np.maximum(density[mask != 0], 0.1)
        source = CShapeDensityInput("synthetic-c", mask, density)

        working = fit_longest_green_extension(source)
        pinned = fit_c_shape_geometry_baseline(source)

        self.assertIsNotNone(working)
        self.assertIsNotNone(pinned)
        np.testing.assert_allclose(
            working.candidates[0].quadrilateral_uv,
            pinned.candidate.quadrilateral_uv)


if __name__ == "__main__":
    unittest.main()
