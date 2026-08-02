"""Parity and catalog tests for the isolated profile-calibration tool."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ...configurations import DEFAULT_DENSITY_CONFIGURATION
from ...density_bank import DensityBank
from ...preprocessing import component_mask, decode_jpeg, load_lut, preprocess_frame
from ..build_assets import _profile_for_area, discover_latest_runs
from ..density_runtime import CandidateSettings, compute_density_evidence
from ..historic_source import HistoricRunSource
from ..server import _candidate


FLIGHT_ROOT = Path(__file__).resolve().parents[8]


class ProfileCalibrationTests(unittest.TestCase):
    def test_area_grouping_uses_the_production_profile_boundaries(self) -> None:
        profiles = DEFAULT_DENSITY_CONFIGURATION.profiles
        minimum = 1
        for profile in profiles:
            self.assertIs(_profile_for_area(minimum), profile)
            maximum = profile.maximum_component_area_px
            if maximum is not None:
                self.assertIs(_profile_for_area(maximum), profile)
                minimum = maximum + 1
        self.assertEqual(_profile_for_area(10**9).profile_id, "scale_10")

    def test_latest_run_discovery_is_timestamp_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in ("01", "04", "02", "05", "03"):
                run = root / f"run-{suffix}"
                (run / "vision_frames").mkdir(parents=True)
                (run / "frames.jsonl").touch()
            selected = discover_latest_runs(root)
            self.assertEqual(
                tuple(path.name for path in selected),
                ("run-03", "run-04", "run-05"),
            )

    def test_saved_square_mask_recomputes_exact_production_evidence(self) -> None:
        run = FLIGHT_ROOT / "logs" / "run-20260801T031401Z"
        lut = load_lut()
        selected = None
        for record in HistoricRunSource(run):
            frame = preprocess_frame(
                frame_id=record.frame_id,
                sim_time_ns=record.sim_time_ns,
                image=decode_jpeg(record.pipeline_input()["jpeg_bytes"]),
                lut=lut,
            )
            selected = next(
                (component for component in frame.components
                 if not component.touches_frame),
                None,
            )
            if selected is not None:
                break
        self.assertIsNotNone(selected)
        assert selected is not None
        mask = component_mask(frame, selected, stage="closed")
        profile = DensityBank(
            frame, DEFAULT_DENSITY_CONFIGURATION,
        ).recommended_standard_profile(selected)
        direct = DensityBank(frame, DEFAULT_DENSITY_CONFIGURATION).get(
            selected, profile.profile_id)
        restored = compute_density_evidence(
            mask,
            DEFAULT_DENSITY_CONFIGURATION.profiles,
            profile.profile_id,
        )

        self.assertEqual(restored.profile, direct.profile)
        self.assertEqual(restored.p70_threshold, direct.p70_threshold)
        self.assertEqual(restored.p80_threshold, direct.p80_threshold)
        self.assertEqual(restored.p90_threshold, direct.p90_threshold)
        for name in (
            "final_field",
            "positive_points_xy",
            "positive_weights",
            "p70_mask",
            "p80_mask",
            "p90_mask",
        ):
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(direct, name), err_msg=name)

    def test_candidate_values_are_schema_owned_and_do_not_mutate_baseline(self) -> None:
        profile = DEFAULT_DENSITY_CONFIGURATION.profiles[0]
        mask = np.zeros((25, 25), np.uint8)
        mask[3:22, 5:20] = 1
        baseline = compute_density_evidence(
            mask, DEFAULT_DENSITY_CONFIGURATION.profiles, profile.profile_id)
        candidate_settings = CandidateSettings(
            density_radius_px=6,
            ridge_radius_px=5,
            relative_cap=profile.relative_cap,
            inverse_gamma=1.55,
            ridge_gamma=4.25,
        )
        candidate = compute_density_evidence(
            mask,
            DEFAULT_DENSITY_CONFIGURATION.profiles,
            profile.profile_id,
            candidate_settings,
        )

        self.assertEqual(baseline.profile, profile)
        self.assertEqual(candidate.profile.density_radius_px, 6)
        self.assertEqual(candidate.profile.ridge_radius_px, 5)
        self.assertEqual(candidate.profile.inverse_gamma, 1.55)
        self.assertEqual(candidate.profile.ridge_gamma, 4.25)
        self.assertEqual(
            DEFAULT_DENSITY_CONFIGURATION.profiles[0], profile,
            "candidate recomputation must not mutate production configuration",
        )

    def test_gamma_candidate_values_are_capped_at_one_hundred(self) -> None:
        profile = DEFAULT_DENSITY_CONFIGURATION.profiles[0]
        candidate = _candidate(
            {
                "inverse_gamma": ["100"],
                "ridge_gamma": ["100"],
            },
            profile,
        )
        self.assertEqual(candidate.inverse_gamma, 100.0)
        self.assertEqual(candidate.ridge_gamma, 100.0)
        with self.assertRaisesRegex(ValueError, "inverse_gamma"):
            _candidate({"inverse_gamma": ["100.01"]}, profile)
        with self.assertRaisesRegex(ValueError, "ridge_gamma"):
            _candidate({"ridge_gamma": ["100.01"]}, profile)


if __name__ == "__main__":
    unittest.main()
