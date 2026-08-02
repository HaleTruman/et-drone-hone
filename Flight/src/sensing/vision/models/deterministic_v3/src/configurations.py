"""Central, code-owned configuration for deterministic-v3 geometry."""

from dataclasses import replace

from .schema import (
    CShapeConfiguration, CShapeDensityProfileRule,
    DensityBankConfiguration, DensityProfile, GateRegressionConfiguration,
    MultiGateConfiguration,
    MultiGateDensityProfileRule, MultiGateIdentificationConfiguration,
    StandardGateConfiguration)


# Density profiles

DENSITY_CALIBRATION_VERSION = (
    "inverse-density-v3:runs-20260731T093159Z+093616Z+093732Z:"
    "nonclipped-area-deciles-higher:"
    "cap2.0:visual-gamma-radius-interpolation-v1"
)

# Provisional visual-sweep interpolation anchors:
# density radius 2/12/20 -> inverse gamma 1.20/1.65/2.10;
# ridge radius 2/10/16 -> ridge gamma 5.50/4.15/3.00.
DEFAULT_DENSITY_CONFIGURATION = DensityBankConfiguration(
    calibration_version=DENSITY_CALIBRATION_VERSION,
    ignore_frame_edge_clipped=True,
    profiles=(
        DensityProfile(
            profile_id="scale_01",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=146,
            density_radius_px=2,
            ridge_radius_px=2,
            relative_cap=2.0,
            inverse_gamma=1.20,
            ridge_gamma=5.50,
        ),
        DensityProfile(
            profile_id="scale_02",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=190,
            density_radius_px=3,
            ridge_radius_px=2,
            relative_cap=2.0,
            inverse_gamma=1.25,
            ridge_gamma=5.50,
        ),
        DensityProfile(
            profile_id="scale_03",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=242,
            density_radius_px=4,
            ridge_radius_px=3,
            relative_cap=2.0,
            inverse_gamma=1.29,
            ridge_gamma=5.33,
        ),
        DensityProfile(
            profile_id="scale_04",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=260,
            density_radius_px=5,
            ridge_radius_px=4,
            relative_cap=2.0,
            inverse_gamma=1.34,
            ridge_gamma=5.16,
        ),
        DensityProfile(
            profile_id="scale_05",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=372,
            density_radius_px=6,
            ridge_radius_px=5,
            relative_cap=2.0,
            inverse_gamma=1.38,
            ridge_gamma=4.99,
        ),
        DensityProfile(
            profile_id="scale_06",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=827,
            density_radius_px=7,
            ridge_radius_px=6,
            relative_cap=2.0,
            inverse_gamma=1.43,
            ridge_gamma=4.83,
        ),
        DensityProfile(
            profile_id="scale_07",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=1_370,
            density_radius_px=9,
            ridge_radius_px=7,
            relative_cap=2.0,
            inverse_gamma=1.52,
            ridge_gamma=4.66,
        ),
        DensityProfile(
            profile_id="scale_08",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=3_113,
            density_radius_px=12,
            ridge_radius_px=10,
            relative_cap=2.0,
            inverse_gamma=1.65,
            ridge_gamma=4.15,
        ),
        DensityProfile(
            profile_id="scale_09",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=4_671,
            density_radius_px=16,
            ridge_radius_px=13,
            relative_cap=2.0,
            inverse_gamma=1.88,
            ridge_gamma=3.58,
        ),
        DensityProfile(
            profile_id="scale_10",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=None,
            density_radius_px=20,
            ridge_radius_px=16,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
    ),
)


# Gate pose regression

DEFAULT_GATE_REGRESSION_CONFIGURATION = GateRegressionConfiguration(
    configuration_version="post-pnp-gate-regression-v2",
    history_half_life_frames=15.0,
    maximum_missed_frames=15,
    minimum_camera_depth_m=0.20,
    pixel_noise_px=1.5,
    association_floor_px=18.0,
    maximum_reprojection_rmse_px=8.0,
)


# Standard-gate processing

STANDARD_GATE_CONFIGURATION_VERSION = (
    "standard-gate-independent-p70-coverage-v1"
)

DEFAULT_STANDARD_GATE_CONFIGURATION = StandardGateConfiguration(
    configuration_version=STANDARD_GATE_CONFIGURATION_VERSION,
    minimum_hole_area_px2=12.0,
    minimum_p70_evidence_points=12,
    minimum_side_support_points=2,
    minimum_side_coverage_ratio=0.25,
    maximum_line_rmse_scale=0.08,
    minimum_side_mask_support_ratio=0.50,
    maximum_aperture_center_offset_ratio=0.30,
    minimum_fit_confidence=0.75,
    mask_support_radius_px=2,
)


# C-shape processing

C_SHAPE_CONFIGURATION_VERSION = (
    "c-shape-tailored-shortfall-v1:legacy-dimension-profile-map-v1"
)

DEFAULT_C_SHAPE_CONFIGURATION = CShapeConfiguration(
    configuration_version=C_SHAPE_CONFIGURATION_VERSION,
    density_profile_rules=(
        CShapeDensityProfileRule(35, "scale_01"),
        CShapeDensityProfileRule(59, "scale_03"),
        CShapeDensityProfileRule(89, "scale_06"),
        CShapeDensityProfileRule(None, "scale_10"),
    ),
)


# Multi-gate identification and processing

MULTI_GATE_IDENTIFICATION_CONFIGURATION_VERSION = (
    "multi-gate-overlap-identification-v1:"
    "covering-local-thickness:regional-sigma1:excess1.20:"
    "multi-arm-or-two-aperture:nonclipped-live"
)

DEFAULT_MULTI_GATE_IDENTIFICATION_CONFIGURATION = (
    MultiGateIdentificationConfiguration(
        configuration_version=MULTI_GATE_IDENTIFICATION_CONFIGURATION_VERSION,
        analyze_clipped_components=False,
        regional_smoothing_sigma_px=1.0,
        excess_ratio=1.20,
        minimum_excess_area_scale=0.15,
        angle_bin_count=36,
        minimum_branch_bins=2,
        minimum_branch_count=3,
        opposing_tolerance_degrees=35.0,
        population_boundaries_px=(
            6.25, 8.0, 10.0, 12.0, 14.0,
            16.0, 18.0, 21.0, 25.0,
        ),
        population_minimum_fraction=0.10,
        population_minimum_pixels=24,
        population_minimum_median_ratio=1.25,
        population_minimum_explained_variance=0.35,
    )
)

REVIEW_MULTI_GATE_IDENTIFICATION_CONFIGURATION = replace(
    DEFAULT_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
    configuration_version=(
        "multi-gate-overlap-identification-v1:"
        "covering-local-thickness:regional-sigma1:excess1.20:"
        "multi-arm-or-two-aperture:clipped-review"
    ),
    analyze_clipped_components=True,
)

MULTI_GATE_CONFIGURATION_VERSION = (
    "multi-gate-two-aperture-coverage-adapter-v1:"
    "legacy-dimension-profile-map-v1:shared-density-bank-v1"
)

DEFAULT_MULTI_GATE_CONFIGURATION = MultiGateConfiguration(
    configuration_version=MULTI_GATE_CONFIGURATION_VERSION,
    density_profile_rules=(
        MultiGateDensityProfileRule(35, "scale_01"),
        MultiGateDensityProfileRule(59, "scale_03"),
        MultiGateDensityProfileRule(89, "scale_06"),
        MultiGateDensityProfileRule(None, "scale_10"),
    ),
    minimum_p90_evidence_points=4,
)
