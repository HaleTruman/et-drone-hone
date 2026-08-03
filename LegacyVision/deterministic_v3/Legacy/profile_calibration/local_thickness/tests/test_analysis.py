"""Focused tests for component-local thickness and profile assignment."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from ..analysis import (
    PROFILE_RADII,
    THICKNESS_PROFILE_BOUNDARIES_PX,
    analyze_frame,
    assign_profiles,
    component_thickness_evidence,
    covering_local_radius,
    equivalent_area_profile_assignment,
    padded_distance_transform,
    segment_thickness_populations,
    thickness_band_assignment,
)
from ...density_runtime import _synthetic_frame


class LocalThicknessTests(unittest.TestCase):
    @staticmethod
    def _overlap_evidence(mask: np.ndarray):
        _, radius, _ = covering_local_radius(mask)
        return component_thickness_evidence(
            1,
            (0, 0, mask.shape[1], mask.shape[0]),
            (0, 0),
            False,
            mask,
            2.0 * radius,
        )[0]

    def test_distance_transform_has_an_explicit_zero_boundary(self) -> None:
        mask = np.ones((5, 5), np.uint8)
        distance = padded_distance_transform(mask)
        self.assertAlmostEqual(float(distance[2, 2]), 3.0, places=4)
        self.assertLess(float(distance.max()), 4.0)

    def test_covering_radius_separates_thick_body_and_thin_arm(self) -> None:
        mask = np.zeros((120, 160), np.uint8)
        mask[10:100, 10:50] = 1
        mask[52:58, 50:145] = 1
        _, radius, centers = covering_local_radius(mask)
        self.assertTrue(np.any(centers))
        self.assertGreater(float(radius[50, 25]), 15.0)
        self.assertLess(float(radius[55, 100]), 4.0)

    def test_l2_assignment_uses_finer_profile_on_thin_arm(self) -> None:
        mask = np.zeros((120, 160), np.uint8)
        mask[10:100, 10:50] = 1
        mask[52:58, 50:145] = 1
        _, radius, _ = covering_local_radius(mask)
        assignment = assign_profiles(
            mask,
            radius,
            assignment_mode="l2_covering_disk",
            minimum_support_ratio=1.0,
        )
        self.assertLess(int(assignment[55, 100]), int(assignment[50, 25]))
        self.assertEqual(PROFILE_RADII[int(assignment[55, 100])], 3)

    def test_square_assignment_matches_full_window_support(self) -> None:
        mask = np.zeros((120, 160), np.uint8)
        mask[10:100, 10:50] = 1
        mask[52:58, 50:145] = 1
        _, radius, _ = covering_local_radius(mask)
        assignment = assign_profiles(
            mask,
            radius,
            assignment_mode="square_window",
            minimum_support_ratio=1.0,
        )
        self.assertEqual(PROFILE_RADII[int(assignment[55, 100])], 2)
        self.assertGreater(int(assignment[50, 25]), int(assignment[55, 100]))

    def test_empirical_thickness_boundaries_keep_ties_together(self) -> None:
        values = np.array([
            6.0, 6.25, 7.99, 8.0, 9.99, 10.0, 11.99, 12.0,
            13.99, 14.0, 15.99, 16.0, 17.99, 18.0, 20.99,
            21.0, 24.99, 25.0,
        ], np.float32)[None, :]
        assignment = thickness_band_assignment(
            np.ones(values.shape, np.uint8), values)

        self.assertEqual(
            THICKNESS_PROFILE_BOUNDARIES_PX,
            (6.25, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 25.0),
        )
        self.assertEqual(
            assignment.tolist()[0],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        )

    def test_equivalent_area_assignment_is_clamped_around_baseline(self) -> None:
        mask = np.ones((1, 3), np.uint8)
        thickness = np.array([[1.0, 10.0, 100.0]], np.float32)

        assignment = equivalent_area_profile_assignment(
            mask, thickness, component_area_px=827)

        self.assertEqual(assignment.tolist()[0], [3, 5, 7])

    def test_two_supported_thickness_populations_are_segmented(self) -> None:
        mask = np.ones((20, 20), np.uint8)
        thickness = np.full(mask.shape, 6.0, np.float32)
        thickness[:, 10:] = 16.0

        evidence, thinner, thicker = segment_thickness_populations(
            mask, thickness)

        self.assertTrue(evidence.available)
        self.assertGreaterEqual(evidence.median_ratio, 2.0)
        self.assertEqual(int(np.count_nonzero(thinner)), 200)
        self.assertEqual(int(np.count_nonzero(thicker)), 200)
        self.assertFalse(np.any(thinner & thicker))
        self.assertTrue(np.array_equal(thinner | thicker, mask != 0))

    def test_uniform_thickness_is_not_forced_into_two_populations(self) -> None:
        mask = np.ones((20, 20), np.uint8)
        evidence, thinner, thicker = segment_thickness_populations(
            mask, np.full(mask.shape, 10.0, np.float32))

        self.assertFalse(evidence.available)
        self.assertEqual(evidence.reason, "uniform_thickness")
        self.assertFalse(np.any(thinner))
        self.assertFalse(np.any(thicker))

    def test_one_toggle_controls_clipped_component_processing(self) -> None:
        mask = np.zeros((25, 25), np.uint8)
        mask[3:22, 3:22] = 1
        frame, component = _synthetic_frame(mask)
        clipped = replace(component, touches_frame=True)
        frame = replace(frame, components=(clipped,))

        excluded = analyze_frame(
            frame, include_clipped_components=False)
        included = analyze_frame(
            frame, include_clipped_components=True)

        self.assertTrue(np.all(
            excluded.adaptive_profile_index[mask != 0] == -2))
        self.assertFalse(np.any(excluded.adaptive_final_field))
        self.assertTrue(np.any(
            included.adaptive_profile_index[mask != 0] >= 0))
        self.assertTrue(np.any(included.adaptive_final_field > 0))

    def test_thickness_peak_at_ordinary_corner_is_not_overlap(self) -> None:
        mask = np.zeros((120, 120), np.uint8)
        mask[15:105, 15:31] = 1
        mask[89:105, 15:105] = 1

        evidence = self._overlap_evidence(mask)

        self.assertGreater(evidence.peak_ratio, 1.20)
        self.assertFalse(evidence.experimental_overlap_candidate)
        self.assertTrue(evidence.junctions)
        self.assertTrue(all(
            junction.branch_count <= 2 for junction in evidence.junctions))

    def test_three_arm_thickness_peak_is_overlap_candidate(self) -> None:
        mask = np.zeros((120, 120), np.uint8)
        mask[15:105, 52:68] = 1
        mask[15:31, 15:105] = 1

        evidence = self._overlap_evidence(mask)

        self.assertTrue(evidence.experimental_overlap_candidate)
        self.assertEqual(evidence.assessment, "candidate")
        self.assertTrue(any(
            junction.branch_count >= 3 and junction.opposing_pair
            for junction in evidence.junctions))

    def test_clipped_thickness_peak_is_indeterminate(self) -> None:
        mask = np.zeros((120, 120), np.uint8)
        mask[15:105, 52:68] = 1
        mask[15:31, 15:105] = 1
        _, radius, _ = covering_local_radius(mask)

        evidence, core = component_thickness_evidence(
            1, (0, 0, 120, 120), (0, 0), True, mask, 2.0 * radius)

        self.assertEqual(evidence.assessment, "indeterminate")
        self.assertFalse(evidence.experimental_overlap_candidate)
        self.assertFalse(np.any(core))

    def test_observed_clipped_candidate_is_retained_for_review_only(self) -> None:
        mask = np.zeros((160, 160), np.uint8)
        mask[35:125, 72:88] = 1
        mask[72:88, 35:125] = 1
        _, radius, _ = covering_local_radius(mask)

        evidence, core = component_thickness_evidence(
            1,
            (0, 0, 160, 160),
            (0, 0),
            True,
            mask,
            2.0 * radius,
            frame_shape=(160, 160),
        )

        self.assertTrue(evidence.experimental_overlap_candidate)
        self.assertEqual(evidence.assessment, "candidate_clipped_review_only")
        self.assertEqual(evidence.evidence_scope, "frame_edge_clipped_partial")
        self.assertFalse(evidence.routing_eligible)
        self.assertTrue(np.any(core))
        self.assertTrue(any(
            junction.support_fully_observed and junction.candidate
            for junction in evidence.junctions))

    def test_adaptive_regional_percentiles_are_nested_and_mask_owned(self) -> None:
        mask = np.zeros((40, 40), np.uint8)
        mask[5:35, 8:32] = 1
        frame, _ = _synthetic_frame(mask)

        analysis = analyze_frame(frame)

        inside = mask != 0
        self.assertFalse(np.any(analysis.adaptive_p70_mask[~inside]))
        self.assertFalse(np.any(analysis.adaptive_p80_mask[~inside]))
        self.assertFalse(np.any(analysis.adaptive_p90_mask[~inside]))
        self.assertFalse(np.any(
            (analysis.adaptive_p90_mask != 0)
            & (analysis.adaptive_p80_mask == 0)))
        self.assertFalse(np.any(
            (analysis.adaptive_p80_mask != 0)
            & (analysis.adaptive_p70_mask == 0)))

    def test_multiple_apertures_support_a_thickness_candidate(self) -> None:
        mask = np.zeros((120, 120), np.uint8)
        mask[15:105, 15:31] = 1
        mask[89:105, 15:105] = 1
        _, radius, _ = covering_local_radius(mask)

        evidence, core = component_thickness_evidence(
            1,
            (0, 0, 120, 120),
            (0, 0),
            False,
            mask,
            2.0 * radius,
            closed_hole_count=2,
        )

        self.assertTrue(evidence.experimental_overlap_candidate)
        self.assertEqual(
            evidence.reason,
            "thickness_excess_with_multiple_apertures",
        )
        self.assertTrue(np.any(core))

    def test_multiple_apertures_do_not_bypass_excess_scale_support(self) -> None:
        mask = np.ones((20, 20), np.uint8)
        thickness = np.full(mask.shape, 10.0, np.float32)
        thickness[10, 10] = 12.0

        evidence, core = component_thickness_evidence(
            1,
            (0, 0, 20, 20),
            (0, 0),
            False,
            mask,
            thickness,
            closed_hole_count=2,
        )

        self.assertFalse(evidence.experimental_overlap_candidate)
        self.assertEqual(
            evidence.reason,
            "thickness_excess_below_scale_support",
        )
        self.assertFalse(np.any(core))


if __name__ == "__main__":
    unittest.main()
