"""Corpus-policy tests for the isolated spectrum asset builder."""

from __future__ import annotations

import unittest

import numpy as np

from ....preprocessing import DEFAULT_CONFIG
from ..build_assets import (
    ComponentAsset,
    SPECTRUM_PREPROCESSING_CONFIG,
    _analysis_square_crop,
    _balanced_bin_sizes,
    _bin_records,
)


def _asset(area_px: int, frame_id: int) -> ComponentAsset:
    return ComponentAsset(
        run_id="run-test",
        frame_id=frame_id,
        sim_time_ns=frame_id,
        source_relative_path=f"vision_frames/{frame_id}.jpg",
        component_id=1,
        production_profile_id="scale_01",
        bbox_x=1,
        bbox_y=1,
        bbox_width=2,
        bbox_height=2,
        image_origin_u=0,
        image_origin_v=0,
        analysis_height=2,
        analysis_width=2,
        area_px=area_px,
        fill_ratio=1.0,
        solidity=1.0,
        source_png=b"source",
        closed_mask_png=b"mask",
    )


class SpectrumBuilderTests(unittest.TestCase):
    def test_review_config_changes_only_area_and_version(self) -> None:
        self.assertEqual(
            SPECTRUM_PREPROCESSING_CONFIG.minimum_component_area_px, 250)
        self.assertIn("a250", SPECTRUM_PREPROCESSING_CONFIG.version)
        for field in (
            "maximum_input_components",
            "close_kernel_px",
            "connectivity",
            "contour_simplify_epsilon_px",
            "distance_mask_size",
        ):
            self.assertEqual(
                getattr(SPECTRUM_PREPROCESSING_CONFIG, field),
                getattr(DEFAULT_CONFIG, field),
                field,
            )

    def test_known_corpus_divides_into_one_245_and_nine_244_bins(self) -> None:
        self.assertEqual(_balanced_bin_sizes(2_441), (245,) + (244,) * 9)

    def test_rank_assignment_is_exhaustive_and_largest_first(self) -> None:
        assets = [_asset(1_000 - index, index) for index in range(23)]
        bins, assignments = _bin_records(assets)
        self.assertEqual(len(assignments), len(assets))
        self.assertEqual(tuple(sorted(set(assignments))), tuple(range(10)))
        self.assertEqual(bins[0].default_profile_id, "scale_10")
        self.assertEqual(bins[-1].default_profile_id, "scale_01")
        self.assertGreaterEqual(
            bins[0].minimum_area_px, bins[-1].maximum_area_px)

    def test_analysis_crop_preserves_alignment_with_black_padding(self) -> None:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        crop = _analysis_square_crop(image, (-2, 1), (5, 5))
        self.assertEqual(crop.shape, (5, 5, 3))
        np.testing.assert_array_equal(crop[0:3, 2:5], image[1:4, 0:3])
        self.assertFalse(np.any(crop[:, :2]))
        self.assertFalse(np.any(crop[3:, 2:]))


if __name__ == "__main__":
    unittest.main()

