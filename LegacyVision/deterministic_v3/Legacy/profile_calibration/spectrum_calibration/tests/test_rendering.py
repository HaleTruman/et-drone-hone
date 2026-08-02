"""Rendering-contract tests for the isolated ten-column review surface."""

from __future__ import annotations

import unittest

import numpy as np

from ....schema import DensityEvidence, DensityProfile
from ..rendering import (
    PANEL_HEADER_PX,
    PANEL_LABELS,
    PANEL_SIDE_PX,
    vertical_evidence_sheet,
)


class SpectrumRenderingTests(unittest.TestCase):
    def test_vertical_sheet_contains_seven_150_by_172_panels(self) -> None:
        mask = np.zeros((31, 31), np.uint8)
        mask[4:27, 6:25] = 1
        field = mask.astype(np.float32) * 0.75
        profile = DensityProfile(
            profile_id="scale_05",
            calibration_version="test",
            maximum_component_area_px=500,
            density_radius_px=6,
            ridge_radius_px=5,
            relative_cap=2.0,
            inverse_gamma=1.4,
            ridge_gamma=5.0,
        )
        evidence = DensityEvidence(
            component_id=1,
            profile=profile,
            final_field=field,
            positive_points_xy=np.empty((0, 2), np.int32),
            positive_weights=np.empty((0,), np.float32),
            p70_threshold=0.1,
            p70_mask=mask,
            p80_threshold=0.2,
            p80_mask=mask,
            p90_threshold=0.3,
            p90_mask=mask,
        )

        candidate_mask = mask.copy()
        candidate_mask[4, 6] = 0
        rendered = vertical_evidence_sheet(
            np.full((31, 31, 3), 127, np.uint8),
            mask,
            candidate_mask,
            evidence,
        )

        self.assertEqual(PANEL_SIDE_PX, 150)
        self.assertEqual(PANEL_HEADER_PX, 22)
        self.assertEqual(len(PANEL_LABELS), 7)
        self.assertIn("candidate DensityBank input mask", PANEL_LABELS)
        self.assertEqual(rendered.shape, (1_204, 150, 3))
        self.assertEqual(rendered.dtype, np.uint8)

    def test_vertical_sheet_rejects_misaligned_evidence(self) -> None:
        mask = np.ones((15, 15), np.uint8)
        profile = DensityProfile(
            "scale_01", "test", 100, 2, 2, 2.0, 1.2, 5.5)
        evidence = DensityEvidence(
            1,
            profile,
            np.ones((14, 14), np.float32),
            np.empty((0, 2), np.int32),
            np.empty((0,), np.float32),
            0.1,
            mask,
            0.2,
            mask,
            0.3,
            mask,
        )
        with self.assertRaisesRegex(ValueError, "final_field"):
            vertical_evidence_sheet(
                np.zeros((15, 15, 3), np.uint8), mask, mask, evidence)

    def test_vertical_sheet_rejects_empty_candidate_mask(self) -> None:
        mask = np.ones((9, 9), np.uint8)
        profile = DensityProfile(
            "scale_01", "test", 100, 2, 2, 2.0, 1.2, 5.5)
        evidence = DensityEvidence(
            1,
            profile,
            mask.astype(np.float32),
            np.empty((0, 2), np.int32),
            np.empty((0,), np.float32),
            0.1,
            mask,
            0.2,
            mask,
            0.3,
            mask,
        )
        with self.assertRaisesRegex(ValueError, "candidate DensityBank input"):
            vertical_evidence_sheet(
                np.zeros((9, 9, 3), np.uint8),
                mask,
                np.zeros_like(mask),
                evidence,
            )


if __name__ == "__main__":
    unittest.main()
