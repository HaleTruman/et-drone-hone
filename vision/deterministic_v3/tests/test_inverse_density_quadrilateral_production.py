import unittest

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.inverse_density_quadrilateral_production import (
    INVERSE_GAMMA,
    MAX_CONNECTED_COMPONENTS,
    RELATIVE_CAP,
    RIDGE_GAMMA,
    detect_density_quadrilaterals,
    fit_p90_quadrilateral,
    inverse_density_field,
    iter_component_inputs,
    radius_profile,
    raw_square_density,
)


MASK_RGB = (10, 20, 30)


def test_lut():
    lut = np.zeros(1 << 24, np.uint8)
    lut[(MASK_RGB[0] << 16) | (MASK_RGB[1] << 8) | MASK_RGB[2]] = 1
    return lut


def reference_field(mask, density_radius, ridge_radius):
    inside = mask != 0
    height, width = mask.shape
    density = np.zeros(mask.shape, np.float64)
    for y, x in np.argwhere(inside):
        y0, y1 = max(0, y - density_radius), min(height, y + density_radius + 1)
        x0, x1 = max(0, x - density_radius), min(width, x + density_radius + 1)
        density[y, x] = inside[y0:y1, x0:x1].mean()
    mean_density = density[inside].mean()
    normalized = np.zeros(mask.shape, np.float64)
    normalized[inside] = np.clip(
        density[inside] / (mean_density * RELATIVE_CAP), 0, 1)
    inverse = np.zeros(mask.shape, np.float64)
    inverse[inside] = np.power(
        1.0 - normalized[inside], 1.0 / INVERSE_GAMMA)
    final = np.zeros(mask.shape, np.float64)
    for y, x in np.argwhere(inside):
        y0, y1 = max(0, y - ridge_radius), min(height, y + ridge_radius + 1)
        x0, x1 = max(0, x - ridge_radius), min(width, x + ridge_radius + 1)
        local = inverse[y0:y1, x0:x1]
        mass = local.sum()
        if mass <= 0:
            continue
        yy, xx = np.indices(local.shape)
        centroid_x = float((local * (xx + x0)).sum() / mass)
        centroid_y = float((local * (yy + y0)).sum() / mass)
        distance = np.hypot(x - centroid_x, y - centroid_y)
        ridge = np.power(
            1.0 - np.clip(distance / ridge_radius, 0, 1), RIDGE_GAMMA)
        final[y, x] = inverse[y, x] * ridge
    return final


class InverseDensityQuadrilateralProductionTests(unittest.TestCase):
    def test_square_density_includes_diagonal_corner_of_window(self):
        mask = np.zeros((9, 9), np.uint8)
        mask[4, 4] = 1
        mask[6, 6] = 1
        mask[7, 7] = 1

        density = raw_square_density(mask, 2)

        self.assertAlmostEqual(float(density[4, 4]), 2 / 25, places=7)

    def test_optimized_formula_matches_direct_reference(self):
        mask = np.zeros((31, 31), np.uint8)
        mask[3:28, 4:27] = 1
        mask[10:21, 11:20] = 0
        expected = reference_field(mask, 4, 3)

        actual = inverse_density_field(mask, 4, 3)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)

    def test_profile_boundaries_remain_calibrated(self):
        expected = {
            1: ("compact", 2, 2), 35: ("compact", 2, 2),
            36: ("small", 4, 3), 59: ("small", 4, 3),
            60: ("medium", 7, 6), 89: ("medium", 7, 6),
            90: ("large", 20, 16),
        }
        for dimension, values in expected.items():
            profile = radius_profile(dimension)
            self.assertEqual(
                (profile.name, profile.density_radius, profile.ridge_radius),
                values)

    def test_component_square_isolated_from_neighbor(self):
        isolated = np.zeros((70, 90), np.uint8)
        isolated[10:40, 10:35] = 1
        neighboring = isolated.copy()
        neighboring[10:40, 45:70] = 1

        first_isolated = next(iter_component_inputs(isolated))
        first_neighboring = next(iter_component_inputs(neighboring))

        self.assertTrue(np.array_equal(
            first_isolated.mask, first_neighboring.mask))
        self.assertEqual(first_isolated.profile, first_neighboring.profile)

    def test_p90_fit_is_convex_and_uses_positive_values_only(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[4:36, 4:36] = 1
        mask[12:28, 12:28] = 0
        field = np.zeros(mask.shape, np.float64)
        field[mask > 0] = 0.1
        field[7:10, 7:33] = 1.0
        field[30:33, 7:33] = 1.0
        field[7:33, 7:10] = 1.0
        field[7:33, 30:33] = 1.0

        fit = fit_p90_quadrilateral(mask, field)

        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.threshold, 1.0)
        self.assertTrue(cv2.isContourConvex(
            fit.corners.astype(np.float32).reshape(-1, 1, 2)))

    def test_p90_fit_rejects_support_hull_with_fewer_than_four_vertices(self):
        mask = np.ones((12, 12), np.uint8)
        field = np.zeros(mask.shape, np.float64)
        field[2, 2] = field[2, 9] = field[9, 5] = 1.0

        fit = fit_p90_quadrilateral(mask, field)

        self.assertIsNone(fit)

    def test_detection_assigns_one_profile_per_component(self):
        image = np.zeros((120, 180, 3), np.uint8)
        color = tuple(reversed(MASK_RGB))
        image[10:45, 10:45] = color
        image[19:36, 19:36] = 0
        image[20:100, 80:160] = color
        image[40:80, 100:140] = 0

        batch = detect_density_quadrilaterals(image, test_lut())

        self.assertFalse(batch.gated)
        self.assertEqual(batch.retained_components, 2)
        self.assertEqual(len(batch.quadrilaterals), 2)
        self.assertEqual(
            [(item.profile.name, item.profile.density_radius,
              item.profile.ridge_radius) for item in batch.quadrilaterals],
            [("compact", 2, 2), ("medium", 7, 6)])

    def test_detection_gates_500_input_components(self):
        image = np.zeros((100, 100, 3), np.uint8)
        color = tuple(reversed(MASK_RGB))
        points = [(y, x) for y in range(0, 100, 2)
                  for x in range(0, 100, 2)][:MAX_CONNECTED_COMPONENTS]
        for y, x in points:
            image[y, x] = color

        batch = detect_density_quadrilaterals(image, test_lut())

        self.assertTrue(batch.gated)
        self.assertGreaterEqual(batch.input_components,
                                MAX_CONNECTED_COMPONENTS)
        self.assertFalse(batch.quadrilaterals)


if __name__ == "__main__":
    unittest.main()
