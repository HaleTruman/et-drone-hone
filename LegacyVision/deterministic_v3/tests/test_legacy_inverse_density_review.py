import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.legacy_inverse_density import (
    DENSITY_RADIUS_PX,
    INVERSE_GAMMA,
    LAYERS,
    RELATIVE_CAP,
    RIDGE_GAMMA,
    RIDGE_RADIUS_PX,
    compute_density_fields,
    compute_preprocessing,
    generate_review,
    heat_layer,
    render_layers,
)
from sensing.vision.models.deterministic_v3.src.legacy_inverse_density_production import (
    DENSITY_RADIUS_PX as PRODUCTION_DENSITY_RADIUS_PX,
    RIDGE_RADIUS_PX as PRODUCTION_RIDGE_RADIUS_PX,
    _field_from_mask,
    component_radius_bucket,
    final_inverse_density,
    fit_density_guided_centerline,
    fit_density_percentile_marker,
)


MASK_RGB = (10, 20, 30)


def test_lut():
    lut = np.zeros(1 << 24, np.uint8)
    lut[(MASK_RGB[0] << 16) | (MASK_RGB[1] << 8) | MASK_RGB[2]] = 1
    return lut


def test_image():
    image = np.zeros((48, 64, 3), np.uint8)
    color = tuple(reversed(MASK_RGB))
    image[8:40, 12:52] = color
    image[3:6, 3:6] = color
    image[20:24, 12:32] = 0
    return image


class LegacyInverseDensityReviewTests(unittest.TestCase):
    def test_preprocessing_exposes_five_ordered_stages(self):
        stages = compute_preprocessing(test_image(), test_lut())

        self.assertEqual(stages.base_mask.shape, test_image().shape[:2])
        self.assertTrue(stages.rejected_small_pixels[4, 4])
        self.assertEqual(int(stages.size_filtered_mask[4, 4]), 0)
        self.assertGreaterEqual(cv2.countNonZero(stages.first_closed_mask),
                                cv2.countNonZero(stages.size_filtered_mask))
        self.assertGreaterEqual(cv2.countNonZero(stages.second_closed_mask),
                                cv2.countNonZero(stages.first_closed_mask))
        self.assertEqual([item[0] for item in LAYERS[:5]], [
            "base_mask", "rejected_small_components", "size_filtered_mask",
            "first_closed_mask", "second_closed_mask",
        ])

    def test_calibrated_full_resolution_math(self):
        mask = np.full((41, 41), 255, np.uint8)
        fields = compute_density_fields(mask)
        center = (20, 20)
        expected_inverse = np.power(1 - 1 / RELATIVE_CAP,
                                    1 / INVERSE_GAMMA)

        self.assertEqual(DENSITY_RADIUS_PX, 12)
        self.assertEqual(RIDGE_RADIUS_PX, 10)
        self.assertEqual(RIDGE_GAMMA, 4.15)
        self.assertAlmostEqual(fields.foreground_mean_density, 1.0, places=6)
        self.assertAlmostEqual(float(fields.normalized_density[center]), 0.5,
                               places=6)
        self.assertAlmostEqual(float(fields.base_inverse_density[center]),
                               expected_inverse, places=6)
        self.assertAlmostEqual(float(fields.ridge_multiplier[center]), 1.0,
                               places=5)
        self.assertAlmostEqual(float(fields.final_inverse_density[center]),
                               expected_inverse, places=5)

    def test_render_layers_are_complete_and_independent(self):
        layers, stages, fields = render_layers(test_image(), test_lut())

        self.assertEqual(set(layers), {item[0] for item in LAYERS})
        self.assertTrue(all(
            not np.shares_memory(left, right)
            for index, left in enumerate(layers.values())
            for right in list(layers.values())[index + 1:]))
        background = heat_layer(fields.final_inverse_density,
                                stages.second_closed_mask > 0)
        self.assertTrue(np.array_equal(background[0, 0], [7, 6, 5]))

    def test_optimized_field_helper_tracks_final_review_field(self):
        image = test_image()
        lut = test_lut()
        stages = compute_preprocessing(image, lut)
        expected = compute_density_fields(
            stages.second_closed_mask).final_inverse_density

        actual = _field_from_mask(
            stages.second_closed_mask, DENSITY_RADIUS_PX, RIDGE_RADIUS_PX)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)

    def test_component_radius_bucket_boundaries(self):
        expected = {
            1: ("compact", 2, 2),
            35: ("compact", 2, 2),
            36: ("small", 4, 3),
            59: ("small", 4, 3),
            60: ("medium", 7, 6),
            89: ("medium", 7, 6),
            90: ("large", 20, 16),
        }

        for dimension, bucket in expected.items():
            selected = component_radius_bucket(dimension)
            self.assertEqual((selected[0], selected[2], selected[3]), bucket)

    def test_large_only_mask_preserves_global_result(self):
        image = np.zeros((140, 160, 3), np.uint8)
        image[20:120, 30:140] = tuple(reversed(MASK_RGB))
        lut = test_lut()
        mask = compute_preprocessing(image, lut).second_closed_mask
        expected = _field_from_mask(
            mask, PRODUCTION_DENSITY_RADIUS_PX,
            PRODUCTION_RIDGE_RADIUS_PX)

        actual = final_inverse_density(image, lut)

        self.assertTrue(np.array_equal(actual, expected))

    def test_component_local_bucket_isolates_nearby_component(self):
        color = tuple(reversed(MASK_RGB))
        isolated = np.zeros((64, 80, 3), np.uint8)
        isolated[12:36, 8:32] = color
        neighboring = isolated.copy()
        neighboring[12:36, 40:64] = color

        isolated_field = final_inverse_density(isolated, test_lut())
        neighboring_field = final_inverse_density(neighboring, test_lut())

        self.assertTrue(np.array_equal(
            isolated_field[12:36, 8:32],
            neighboring_field[12:36, 8:32]))

    def test_centerline_uses_density_while_staying_between_contours(self):
        component = np.zeros((40, 40), np.uint8)
        component[3:37, 3:37] = 1
        component[12:28, 12:28] = 0
        density = np.zeros(component.shape, np.float64)
        density[component > 0] = 0.1
        density[3:7, 3:37] = 1.0
        density[33:37, 3:37] = 1.0
        density[3:37, 3:7] = 1.0
        density[3:37, 33:37] = 1.0

        fit = fit_density_guided_centerline(component, density)

        self.assertIsNotNone(fit)
        self.assertEqual(fit.geometric_corners.shape, (4, 2))
        self.assertEqual(fit.density_corners.shape, (4, 2))
        self.assertTrue(all(0.15 <= value <= 0.85
                            for value in fit.density_offsets))
        self.assertGreater(float(np.mean(fit.density_offsets)), 0.5)

    def test_density_percentile_marker_uses_only_highest_evidence(self):
        component = np.zeros((40, 40), np.uint8)
        component[3:37, 3:37] = 1
        component[12:28, 12:28] = 0
        density = np.zeros(component.shape, np.float64)
        density[component > 0] = 0.1
        density[7:10, 7:33] = 1.0
        density[30:33, 7:33] = 1.0
        density[7:33, 7:10] = 1.0
        density[7:33, 30:33] = 1.0

        marker = fit_density_percentile_marker(component, density)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.corners.shape, (4, 2))
        self.assertAlmostEqual(marker.threshold, 1.0)
        self.assertEqual(marker.point_count, int(np.count_nonzero(density == 1)))

    def test_minimal_production_path_gates_500_components(self):
        image = np.zeros((96, 96, 3), np.uint8)
        color = tuple(reversed(MASK_RGB))
        image[:24, :24] = color
        points = [(y, x) for y in range(26, 96, 2)
                  for x in range(0, 96, 2)][:499]
        for y, x in points:
            image[y, x] = color

        field = final_inverse_density(image, test_lut())

        self.assertFalse(np.any(field))

    def test_generation_writes_profile_and_all_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "run-test"
            frames = run / "vision_frames"
            frames.mkdir(parents=True)
            source = frames / "frame-00000001-1.jpg"
            self.assertTrue(cv2.imwrite(str(source), test_image()))
            output = root / "deterministic_v3" / "legacy_review_runs"
            manifest = (root / "deterministic_v3" / "ui" / "frontend" /
                        "data" / "legacy-runs-manifest.json")

            counts = generate_review(
                [run], output, manifest, test_lut(), static_root=root)

            self.assertEqual(counts, {"run-test": 1})
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["profile"]["density_radius_px"], 12)
            self.assertEqual(payload["profile"]["inverse_gamma"], 1.65)
            self.assertEqual(payload["profile"]["resolution"],
                             "native full frame")
            self.assertEqual(len(payload["layers"]), 10)
            frame = payload["runs"][0]["frames"][0]
            self.assertEqual(set(frame["layers"]), {item[0] for item in LAYERS})
            for layer_id, _, _ in LAYERS:
                self.assertTrue((output / "run-test" / "layers" / layer_id /
                                 "frame-00000001-1.png").is_file())


if __name__ == "__main__":
    unittest.main()
