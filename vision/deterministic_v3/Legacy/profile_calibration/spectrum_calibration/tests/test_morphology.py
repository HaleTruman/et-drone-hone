"""Candidate-mask morphology contract tests for the isolated laboratory."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from ..server import CandidateMorphology, apply_candidate_morphology


class SpectrumMorphologyTests(unittest.TestCase):
    def test_zero_zero_is_an_exact_binary_identity_with_metrics(self) -> None:
        mask = np.zeros((11, 13), np.uint8)
        mask[2:7, 3:9] = 255
        mask[8, 11] = 7

        result = apply_candidate_morphology(mask, CandidateMorphology(0, 0))

        np.testing.assert_array_equal(result.mask, (mask != 0).astype(np.uint8))
        self.assertEqual(result.mask.dtype, np.uint8)
        self.assertEqual(result.provenance["operation_order"], [
            "MORPH_OPEN", "MORPH_CLOSE"])
        self.assertFalse(result.provenance["open_enabled"])
        self.assertFalse(result.provenance["close_enabled"])
        self.assertEqual(result.provenance["padding_px"], 0)
        self.assertEqual(
            result.provenance["foreground_area_before_px"],
            result.provenance["foreground_area_after_px"],
        )
        self.assertEqual(
            result.provenance["connected_component_count_before"],
            result.provenance["connected_component_count_after"],
        )
        self.assertEqual(result.provenance["split_count"], 0)

    def test_radius_one_is_a_three_by_three_rectangular_kernel(self) -> None:
        settings = CandidateMorphology(1, 1)
        self.assertEqual(settings.open_kernel_width_px, 3)
        self.assertEqual(settings.close_kernel_width_px, 3)

        mask = np.zeros((9, 9), np.uint8)
        mask[2:7, 2:7] = 1
        result = apply_candidate_morphology(mask, settings)
        self.assertEqual(result.provenance["kernel_shape"], "MORPH_RECT")
        self.assertEqual(result.provenance["open_kernel_width_px"], 3)
        self.assertEqual(result.provenance["close_kernel_width_px"], 3)
        self.assertEqual(result.provenance["padding_px"], 2)

    def test_open_is_applied_before_close(self) -> None:
        mask = np.zeros((11, 11), np.uint8)
        mask[3:8, 3:8] = 1
        mask[5, 3:5] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        padded = np.pad(mask, 2, mode="constant", constant_values=0)
        expected = cv2.morphologyEx(
            cv2.morphologyEx(
                padded,
                cv2.MORPH_OPEN,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
            cv2.MORPH_CLOSE,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[2:-2, 2:-2]
        reversed_order = cv2.morphologyEx(
            cv2.morphologyEx(
                padded,
                cv2.MORPH_CLOSE,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
            cv2.MORPH_OPEN,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[2:-2, 2:-2]
        self.assertFalse(np.array_equal(expected, reversed_order))

        result = apply_candidate_morphology(mask, CandidateMorphology(1, 1))

        np.testing.assert_array_equal(result.mask, expected)
        self.assertEqual(
            result.provenance["operation_order"],
            ["MORPH_OPEN", "MORPH_CLOSE"],
        )
        self.assertEqual(
            result.provenance["foreground_area_after_px"],
            int(np.count_nonzero(expected)),
        )

    def test_open_that_erases_all_foreground_raises_stable_error(self) -> None:
        mask = np.zeros((7, 7), np.uint8)
        mask[3, 3] = 1
        with self.assertRaisesRegex(ValueError, "^candidate_mask_empty$"):
            apply_candidate_morphology(mask, CandidateMorphology(1, 0))


if __name__ == "__main__":
    unittest.main()
