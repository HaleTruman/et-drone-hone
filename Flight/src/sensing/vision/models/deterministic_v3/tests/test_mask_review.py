from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3 import (
    DetectionVision,
    DeterministicVision,
)
from sensing.vision.models.deterministic_v3.src.void_detection import (
    CALIBRATED_DENSITY_RADIUS_PX,
    CALIBRATED_COMPONENT_RADIUS_PROFILES,
    CALIBRATED_INVERSE_GAMMA,
    CALIBRATED_RELATIVE_CAP,
    CALIBRATED_RIDGE_GAMMA,
    CALIBRATED_RIDGE_RADIUS_PX,
    DEFAULT_DISTANCE_GAMMA,
    DEFAULT_LOCAL_DISTANCE_RADIUS,
    DENSE_LINE_MAX_SEGMENTS,
    DENSE_LINE_MIN_SEGMENTS,
    INVERSE_DENSITY_RADIUS_PX,
    LEGACY_INVERSE_DENSITY_GAMMA,
    LEGACY_INVERSE_DENSITY_RADIUS_PX,
    LEGACY_INVERSE_DENSITY_SWEEP,
    LOCAL_INVERSE_DENSITY_RADIUS_PX,
    LOCAL_INVERSE_DENSITY_GAMMA,
    REVIEW_LAYERS,
    calibrated_final_inverse_density_layer,
    calibrated_component_radii,
    calibrated_component_scaled_field,
    compute_mask_stages,
    contour_to_local_maxima_layer,
    dense_contour_quadrilateral_layer,
    dense_contour_quadrilaterals,
    dense_pixel_line_layer,
    dense_pixel_line_segments,
    detect_image,
    distance_transform_layer,
    find_contours,
    generate_review,
    include_latest_logged_run,
    inverse_mask_density_layer,
    legacy_inverse_mask_density_layer,
    legacy_inverse_density_sweep,
    legacy_sweep_id,
    local_inverse_mask_density_layer,
    local_distance_transform_layer,
    render_review_layers,
    selected_runs,
)
from sensing.vision.models.deterministic_v3.src.void_geometry import (
    parent_obtuse_corners,
)


MASK_RGB = (10, 20, 30)


def test_lut() -> np.ndarray:
    lut = np.zeros(1 << 24, dtype=np.uint8)
    lut[(MASK_RGB[0] << 16) | (MASK_RGB[1] << 8) | MASK_RGB[2]] = 1
    return lut


def test_image() -> np.ndarray:
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    mask_color_bgr = tuple(reversed(MASK_RGB))
    image[8:64, 12:84] = mask_color_bgr
    image[25:49, 34:62] = 0
    image[35:39, 12:36] = 0
    image[3:6, 3:6] = mask_color_bgr
    return image


class MaskReviewTests(unittest.TestCase):
    def test_public_compatibility_alias(self) -> None:
        self.assertIs(DeterministicVision, DetectionVision)

    def test_mask_stages_capture_filtering_and_morphology(self) -> None:
        image = test_image()
        lut = test_lut()

        stages = compute_mask_stages(image, lut)

        self.assertEqual(stages.base_mask.dtype, np.uint8)
        self.assertEqual(stages.contour_mask.shape, image.shape[:2])
        self.assertGreater(cv2.countNonZero(stages.close_added_mask), 0)
        self.assertTrue(stages.small_pixels[4, 4])
        self.assertEqual(int(stages.size_filtered_mask[4, 4]), 0)

    def test_geometry_contains_corners_without_circle_features(self) -> None:
        context = {
            "id_start": 0,
            "run_id": "run-test",
            "frame_id": "frame-00000001-1",
            "frame_index": 0,
            "frame_count": 1,
        }
        _, estimates, geometry = detect_image(test_image(), context, test_lut())

        self.assertTrue(estimates)
        records = geometry["void_geometry.json"]
        self.assertTrue(records)
        self.assertTrue(records[0].outer_corners)
        self.assertFalse(hasattr(records[0], "outer_corner_circles"))
        self.assertFalse(hasattr(records[0], "inner_corner_circles"))
        self.assertFalse(hasattr(records[0], "connection_circles"))

    def test_parent_obtuse_corners_are_ellipse_independent(self) -> None:
        contour = np.array([[[0, 0]], [[10, 0]], [[20, 2]], [[20, 20]],
                            [[0, 20]]], dtype=np.int32)
        corners = parent_obtuse_corners(contour)
        match = next(corner for corner in corners if corner.point_px == (10, 0))
        self.assertGreater(match.angle_degrees, 160.0)

    def test_review_layers_are_complete_and_independent(self) -> None:
        layers, _, _ = render_review_layers(test_image(), {
            "id_start": 0, "run_id": "run-test",
            "frame_id": "frame-00000001-1", "frame_index": 0,
            "frame_count": 1,
        }, test_lut())
        self.assertEqual(set(layers), {item[0] for item in REVIEW_LAYERS})
        self.assertTrue(all(
            not np.shares_memory(left, right)
            for index, left in enumerate(layers.values())
            for right in list(layers.values())[index + 1:]))

    def test_low_distance_gamma_produces_a_bright_visible_layer(self) -> None:
        stages = compute_mask_stages(test_image(), test_lut())
        contours, _ = find_contours(stages.contour_mask)

        bright = distance_transform_layer(
            stages.contour_mask, contours, DEFAULT_DISTANCE_GAMMA)
        dark = distance_transform_layer(stages.contour_mask, contours, 1.5)
        interior = stages.contour_mask > 0
        luminance = np.array([0.114, 0.587, 0.299])

        self.assertGreater(float((bright[interior] @ luminance).mean()),
                           float((dark[interior] @ luminance).mean()))
        contour_band = cv2.dilate(
            interior.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.assertTrue(np.all(bright[~contour_band] == 0))

    def test_inverse_mask_density_highlights_thin_mask_regions(self) -> None:
        mask = np.zeros((96, 96), np.uint8)
        mask[8:48, 8:48] = 255
        mask[68:71, 8:60] = 255

        layer = inverse_mask_density_layer(mask)
        local = local_inverse_mask_density_layer(mask)

        self.assertEqual(INVERSE_DENSITY_RADIUS_PX, 9)
        self.assertEqual(LOCAL_INVERSE_DENSITY_RADIUS_PX, 5)
        self.assertEqual(LOCAL_INVERSE_DENSITY_GAMMA, 3.0)
        self.assertGreater(int(layer[69, 30, 0]), int(layer[28, 28, 0]))
        self.assertGreater(int(local[69, 30, 0]), int(local[28, 28, 0]))
        self.assertTrue(np.all(layer[mask == 0] == 0))
        self.assertFalse(np.array_equal(layer, local))

    def test_calibrated_inverse_density_uses_native_mask_resolution(self) -> None:
        mask = np.zeros((47, 73), np.uint8)
        mask[7:40, 11:62] = 255
        mask[18:29, 11:32] = 0

        layer = calibrated_final_inverse_density_layer(mask)

        self.assertEqual(layer.shape, (47, 73, 3))
        self.assertEqual(CALIBRATED_DENSITY_RADIUS_PX, 12)
        self.assertEqual(CALIBRATED_RELATIVE_CAP, 2.0)
        self.assertEqual(CALIBRATED_INVERSE_GAMMA, 1.65)
        self.assertEqual(CALIBRATED_RIDGE_RADIUS_PX, 10)
        self.assertEqual(CALIBRATED_RIDGE_GAMMA, 4.15)
        self.assertTrue(np.array_equal(layer[0, 0], [7, 6, 5]))
        self.assertTrue(np.any(layer[mask > 0] != np.array([7, 6, 5])))
        layer_ids = [item[0] for item in REVIEW_LAYERS]
        contour_index = layer_ids.index("contour_mask")
        self.assertEqual(layer_ids[contour_index + 1],
                         "calibrated_final_inverse_density")

    def test_component_scaled_density_has_five_proportional_profiles(self) -> None:
        self.assertEqual(
            [profile[1] for profile in CALIBRATED_COMPONENT_RADIUS_PROFILES],
            [20, 16, 11, 7, 2])
        self.assertEqual(calibrated_component_radii(20 ** 2, 20, 17), (2, 2))
        self.assertEqual(calibrated_component_radii(90 ** 2, 20, 17), (20, 17))
        middle = calibrated_component_radii(55 ** 2, 20, 17)
        self.assertEqual(middle, (11, 10))

        layer_ids = [item[0] for item in REVIEW_LAYERS]
        fixed = layer_ids.index("calibrated_final_inverse_density")
        self.assertEqual(layer_ids[fixed + 1:fixed + 6], [
            f"calibrated_component_scaled_{profile[0]}"
            for profile in CALIBRATED_COMPONENT_RADIUS_PROFILES
        ])

    def test_component_scaled_density_isolates_neighboring_components(self) -> None:
        isolated = np.zeros((96, 128), np.uint8)
        isolated[20:60, 12:52] = 255
        neighboring = isolated.copy()
        neighboring[20:60, 76:116] = 255

        isolated_field = calibrated_component_scaled_field(isolated, 20, 17)
        neighboring_field = calibrated_component_scaled_field(
            neighboring, 20, 17)

        self.assertTrue(np.array_equal(
            isolated_field[20:60, 12:52],
            neighboring_field[20:60, 12:52]))
        small = calibrated_component_scaled_field(isolated, 2, 2)
        self.assertFalse(np.array_equal(isolated_field, small))

    def test_dense_pixel_line_fits_each_dense_component_independently(self) -> None:
        mask = np.zeros((80, 120), np.uint8)
        mask[12:28, 8:50] = 255
        mask[52:68, 70:112] = 255

        layer = dense_pixel_line_layer(mask)

        bright = (layer.max(axis=2) > 220) & (layer.min(axis=2) < 100)
        self.assertTrue(np.any(bright[12:28, 8:50]))
        self.assertTrue(np.any(bright[52:68, 70:112]))
        self.assertFalse(np.any(bright[28:52, 50:70]))
        colors = np.unique(layer[bright], axis=0)
        self.assertGreaterEqual(len(colors), 2)
        self.assertFalse(np.any(dense_pixel_line_layer(
            np.zeros_like(mask))))

    def test_dense_pixel_line_adapts_between_two_and_eight_fits(self) -> None:
        mask = np.zeros((80, 120), np.uint8)
        mask[24:48, 8:112] = 255

        _, groups = dense_pixel_line_segments(mask)

        self.assertGreaterEqual(len(groups), DENSE_LINE_MIN_SEGMENTS)
        self.assertLessEqual(len(groups), DENSE_LINE_MAX_SEGMENTS)

        fragmented = np.zeros((120, 240), np.uint8)
        for index in range(10):
            x = 4 + index * 23
            fragmented[40:56, x:x + 16] = 255
        _, fragmented_groups = dense_pixel_line_segments(fragmented)
        self.assertLessEqual(len(fragmented_groups), DENSE_LINE_MAX_SEGMENTS)

    def test_dense_contours_are_fitted_as_independent_quadrilaterals(self) -> None:
        mask = np.zeros((96, 128), np.uint8)
        mask[12:44, 8:56] = 255
        mask[56:88, 72:120] = 255

        candidates, contours, quadrilaterals = dense_contour_quadrilaterals(
            mask)
        layer = dense_contour_quadrilateral_layer(mask)

        self.assertTrue(np.any(candidates))
        self.assertEqual(len(contours), 2)
        self.assertEqual(len(quadrilaterals), 2)
        self.assertTrue(all(points.shape == (4, 2)
                            for points in quadrilaterals))
        self.assertTrue(np.any(layer[12:44, 8:56]))
        self.assertTrue(np.any(layer[56:88, 72:120]))
        self.assertFalse(np.any(dense_contour_quadrilateral_layer(
            np.zeros_like(mask))))

        triangle = np.zeros((64, 64), np.uint8)
        cv2.fillConvexPoly(
            triangle, np.array([[32, 4], [4, 58], [58, 58]], np.int32), 255)
        _, triangle_contours, triangle_quads = dense_contour_quadrilaterals(
            triangle)
        self.assertEqual(len(triangle_contours), 1)
        self.assertEqual(len(triangle_quads), 1)
        self.assertEqual(triangle_quads[0].shape, (4, 2))

    def test_legacy_inverse_density_matches_square_field(self) -> None:
        mask = np.zeros((13, 13), np.uint8)
        mask[6, 6] = 255

        layer = legacy_inverse_mask_density_layer(mask)
        inverse = np.power(120 / 121, LEGACY_INVERSE_DENSITY_GAMMA)
        expected_rgb = np.array([
            round(20 + 235 * inverse),
            round(40 + 178 * np.sqrt(inverse)),
            round(52 - 42 * inverse),
        ], np.uint8)

        self.assertTrue(np.array_equal(layer[6, 6], expected_rgb[::-1]))
        self.assertEqual(LEGACY_INVERSE_DENSITY_RADIUS_PX, 5)
        self.assertEqual(LEGACY_INVERSE_DENSITY_GAMMA, 1.5)
        self.assertTrue(np.array_equal(layer[0, 0], [52, 40, 20]))
        solid = legacy_inverse_mask_density_layer(np.full((13, 13), 255, np.uint8))
        self.assertTrue(np.all(solid == np.array([52, 40, 20], np.uint8)))
        sweep = legacy_inverse_density_sweep(mask)
        self.assertEqual(len(sweep), 9)
        default_id = legacy_sweep_id(
            LEGACY_INVERSE_DENSITY_RADIUS_PX, LEGACY_INVERSE_DENSITY_GAMMA)
        self.assertTrue(np.array_equal(sweep[default_id], layer))

    def test_local_distance_is_not_rescaled_by_a_remote_mask_area(self) -> None:
        local_only = np.zeros((48, 96), np.uint8)
        local_only[8:20, 8:20] = 255
        with_remote_area = local_only.copy()
        with_remote_area[8:40, 48:88] = 255
        local_contours, _ = find_contours(local_only)
        remote_contours, _ = find_contours(with_remote_area)

        local = local_distance_transform_layer(
            local_only, local_contours, DEFAULT_LOCAL_DISTANCE_RADIUS)
        remote = local_distance_transform_layer(
            with_remote_area, remote_contours, DEFAULT_LOCAL_DISTANCE_RADIUS)
        global_local = distance_transform_layer(local_only, local_contours)
        global_remote = distance_transform_layer(
            with_remote_area, remote_contours)

        self.assertTrue(np.array_equal(local[:28, :28], remote[:28, :28]))
        self.assertFalse(np.array_equal(
            global_local[:28, :28], global_remote[:28, :28]))

    def test_contour_to_maxima_transform_is_component_isolated(self) -> None:
        local_only = np.zeros((48, 96), np.uint8)
        local_only[6:30, 6:30] = 255
        with_remote_area = local_only.copy()
        with_remote_area[8:42, 48:90] = 255
        local_contours, _ = find_contours(local_only)
        remote_contours, _ = find_contours(with_remote_area)

        local = contour_to_local_maxima_layer(local_only, local_contours)
        remote = contour_to_local_maxima_layer(
            with_remote_area, remote_contours)

        self.assertTrue(np.array_equal(local[:36, :36], remote[:36, :36]))
        self.assertGreater(len(np.unique(local[7:29, 7:29].reshape(-1, 3),
                                         axis=0)), 3)

    def test_generation_replaces_only_the_active_review_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = root / "detection_v2" / "runs"
            run = runs_root / "run-test"
            frames = run / "vision_frames"
            frames.mkdir(parents=True)
            sources = [
                frames / "frame-00000003-3.jpg",
                frames / "frame-00000001-1.jpg",
                frames / "frame-00000002-2.jpg",
            ]
            for source in sources:
                self.assertTrue(cv2.imwrite(str(source), test_image()))
            original_sources = {source: source.read_bytes() for source in sources}

            output_root = root / "deterministic_v3" / "review_runs"
            output_root.mkdir(parents=True)
            stale = output_root / "stale.txt"
            stale.write_text("old generated output")
            manifest = (root / "deterministic_v3" / "frame-viewer" / "data" /
                        "runs-manifest.json")

            counts = generate_review(
                [run], output_root, manifest, test_lut(), start=1, limit=1,
                static_root=root)

            self.assertEqual(counts, {"run-test": 1})
            self.assertFalse(stale.exists())
            for source, original in original_sources.items():
                self.assertEqual(source.read_bytes(), original)
            output = (output_root / "run-test" / "layers" / "composite" /
                      "frame-00000002-2.png")
            self.assertTrue(output.is_file())
            source_image = cv2.imread(str(frames / "frame-00000002-2.jpg"))
            expected, _, _ = detect_image(source_image, {
                "id_start": 0,
                "run_id": "run-test",
                "frame_id": "frame-00000002-2",
                "frame_index": 1,
                "frame_count": 3,
            }, test_lut())
            rendered = cv2.imread(str(output))
            self.assertTrue(np.array_equal(rendered, expected))
            payload = json.loads(manifest.read_text())
            self.assertEqual(
                payload["settings"]["distance_gamma"], DEFAULT_DISTANCE_GAMMA)
            self.assertEqual(
                payload["settings"]["local_distance_radius"],
                DEFAULT_LOCAL_DISTANCE_RADIUS)
            sweep = payload["sweeps"][0]
            self.assertEqual(sweep["default_preset"], "r5_g1p5")
            self.assertEqual(len(sweep["presets"]), 9)
            self.assertEqual(
                {layer["id"] for layer in payload["layers"]},
                {item[0] for item in REVIEW_LAYERS})
            self.assertEqual(payload["runs"][0]["id"], "run-test")
            self.assertEqual(len(payload["runs"][0]["frames"]), 1)
            frame_payload = payload["runs"][0]["frames"][0]
            self.assertEqual(frame_payload["id"], "frame-00000002-2")
            self.assertEqual(
                set(frame_payload["layers"]), {item[0] for item in REVIEW_LAYERS})
            self.assertEqual(
                set(frame_payload["sweeps"]["legacy_inverse_mask_density"]),
                {legacy_sweep_id(*preset)
                 for preset in LEGACY_INVERSE_DENSITY_SWEEP})
            self.assertTrue(frame_payload["layers"]["composite"].startswith(
                "/deterministic_v3/review_runs/run-test/layers/composite/"))
            for layer_id, _, _ in REVIEW_LAYERS:
                self.assertTrue((output_root / "run-test" / "layers" /
                                 layer_id / "frame-00000002-2.png").is_file())
            for radius, gamma in LEGACY_INVERSE_DENSITY_SWEEP:
                self.assertTrue((
                    output_root / "run-test" / "sweeps" /
                    "legacy_inverse_mask_density" /
                    legacy_sweep_id(radius, gamma) /
                    "frame-00000002-2.png").is_file())
            self.assertFalse(list(output_root.rglob("*.json")))

    def test_run_selection_accepts_only_vision_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            current = runs_root / "run-current"
            legacy = runs_root / "run-legacy"
            (current / "vision_frames").mkdir(parents=True)
            (legacy / "mask_frames").mkdir(parents=True)

            selected = selected_runs(runs_root, all_runs=True)
            self.assertEqual(selected, [current])
            with self.assertRaisesRegex(ValueError, "no vision_frames"):
                selected_runs(runs_root, [legacy.name])

    def test_latest_repository_log_run_is_automatically_included(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "selected" / "run-selected"
            logs = root / "Flight" / "logs" / "runs"
            older = logs / "run-20260731T120000Z"
            latest = logs / "run-20260801T031401Z"
            for run in (selected, older, latest):
                frames = run / "vision_frames"
                frames.mkdir(parents=True)
                self.assertTrue(cv2.imwrite(str(frames / "frame-1.png"),
                                            test_image()))

            included = include_latest_logged_run([selected], logs)

            self.assertEqual(included, [selected, latest])
            self.assertEqual(
                include_latest_logged_run([latest], logs), [latest])


if __name__ == "__main__":
    unittest.main()
