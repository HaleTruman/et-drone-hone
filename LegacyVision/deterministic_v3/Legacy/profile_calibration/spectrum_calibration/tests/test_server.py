"""Candidate interpolation tests for the spectrum backend."""

from __future__ import annotations

import unittest

from ....configurations import DEFAULT_DENSITY_CONFIGURATION
from ..server import (
    AssetStore,
    BinSnapshot,
    PROFILE_FIELDS,
    _endpoint_values,
    resolve_candidate,
)


class _Store:
    def __init__(self) -> None:
        self.profiles = DEFAULT_DENSITY_CONFIGURATION.profiles
        self.profile_map = {
            profile.profile_id: profile for profile in self.profiles}
        self.bins = tuple(BinSnapshot(
            bin_index=index,
            count=1,
            min_area=1_000 - index,
            max_area=1_000 - index,
            median_area=1_000 - index,
            default_profile_id=f"scale_{10 - index:02d}",
        ) for index in range(10))

    default_endpoints = AssetStore.default_endpoints


class SpectrumServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _Store()

    def test_endpoint_columns_are_exact_and_intermediate_radii_are_half_up(self) -> None:
        query = {
            "large_density_radius_px": ["1"],
            "small_density_radius_px": ["2"],
        }
        result = resolve_candidate(self.store, query)
        self.assertEqual(result.resolved_bins[0]["density_radius_px"], 1)
        self.assertEqual(result.resolved_bins[4]["density_radius_px"], 1)
        self.assertEqual(result.resolved_bins[5]["density_radius_px"], 2)
        self.assertEqual(result.resolved_bins[9]["density_radius_px"], 2)
        for field in PROFILE_FIELDS:
            self.assertEqual(
                result.resolved_bins[0][field], result.endpoints[f"large_{field}"])
            self.assertEqual(
                result.resolved_bins[9][field], result.endpoints[f"small_{field}"])

    def test_default_morphology_is_one_for_large_and_zero_for_small(self) -> None:
        defaults = self.store.default_endpoints()
        self.assertEqual(defaults["large_open_kernel_radius_px"], 1)
        self.assertEqual(defaults["large_close_kernel_radius_px"], 1)
        self.assertEqual(defaults["small_open_kernel_radius_px"], 0)
        self.assertEqual(defaults["small_close_kernel_radius_px"], 0)

        result = resolve_candidate(self.store, {})
        for field in ("open_kernel_radius_px", "close_kernel_radius_px"):
            self.assertEqual(
                [item[field] for item in result.resolved_bins],
                [1] * 5 + [0] * 5,
                field,
            )

    def test_relative_cap_endpoints_interpolate_and_enforce_one_to_five(self) -> None:
        result = resolve_candidate(self.store, {
            "large_relative_cap": ["1"],
            "small_relative_cap": ["5"],
        })
        self.assertEqual(result.resolved_bins[0]["relative_cap"], 1.0)
        self.assertEqual(result.resolved_bins[9]["relative_cap"], 5.0)

        defaults = self.store.default_endpoints()
        accepted = _endpoint_values({
            "large_relative_cap": ["1.0"],
            "small_relative_cap": ["5.0"],
        }, defaults)
        self.assertEqual(accepted["large_relative_cap"], 1.0)
        self.assertEqual(accepted["small_relative_cap"], 5.0)
        for name, value in (
            ("large_relative_cap", "0.99"),
            ("small_relative_cap", "5.01"),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    _endpoint_values({name: [value]}, defaults)

    def test_gamma_bounds_retain_the_approved_one_hundred_cap(self) -> None:
        defaults = self.store.default_endpoints()
        values = _endpoint_values({
            "large_inverse_gamma": ["100"],
            "small_ridge_gamma": ["100"],
        }, defaults)
        self.assertEqual(values["large_inverse_gamma"], 100.0)
        self.assertEqual(values["small_ridge_gamma"], 100.0)
        with self.assertRaisesRegex(ValueError, "large_inverse_gamma"):
            _endpoint_values(
                {"large_inverse_gamma": ["100.01"]}, defaults)

    def test_candidate_signature_is_canonical(self) -> None:
        first = resolve_candidate(self.store, {
            "large_inverse_gamma": ["2.100000"],
        })
        second = resolve_candidate(self.store, {
            "large_inverse_gamma": ["2.1"],
        })
        self.assertEqual(first.signature, second.signature)


if __name__ == "__main__":
    unittest.main()
