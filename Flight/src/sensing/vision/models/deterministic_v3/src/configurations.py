"""Central, code-owned configuration for deterministic-v3 geometry."""

from .schema import (
    CShapeConfiguration, CShapeDensityProfileRule,
    DensityBankConfiguration, DensityProfile)


# Density profiles

DENSITY_CALIBRATION_VERSION = (
    "inverse-density-v3:runs-20260731T093159Z+093616Z+093732Z:"
    "nonclipped-area-deciles-higher:"
    "cap2.0:inverse2.10:ridge3.00"
)

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
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_02",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=190,
            density_radius_px=3,
            ridge_radius_px=2,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_03",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=242,
            density_radius_px=4,
            ridge_radius_px=3,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_04",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=260,
            density_radius_px=5,
            ridge_radius_px=4,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_05",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=372,
            density_radius_px=6,
            ridge_radius_px=5,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_06",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=827,
            density_radius_px=7,
            ridge_radius_px=6,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_07",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=1_370,
            density_radius_px=9,
            ridge_radius_px=7,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_08",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=3_113,
            density_radius_px=12,
            ridge_radius_px=10,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
        ),
        DensityProfile(
            profile_id="scale_09",
            calibration_version=DENSITY_CALIBRATION_VERSION,
            maximum_component_area_px=4_671,
            density_radius_px=16,
            ridge_radius_px=13,
            relative_cap=2.0,
            inverse_gamma=2.10,
            ridge_gamma=3.00,
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
