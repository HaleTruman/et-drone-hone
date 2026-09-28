import numpy as np

from models.deterministic_v2.src.config import DEFAULT_PRESET_PATH, ProjectionConfig


def test_projection_preset_is_reduced_and_valid():
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    assert config.image_width == 640
    assert config.image_height == 360
    assert config.camera["fx"] == 320.0
    assert config.enabled_bits == [0, 1, 2, 3]
    assert [profile["profile_id"] for profile in config.profiles] == ["A", "B", "C"]
    assert config.section("globalGateMapping")["minSampleScore"] == 0.85
    assert config.section("globalGateMapping")["positionSmoothingAlpha"] == 0.6
    assert "segmentation" not in config.payload
    assert "flightBridge" not in config.payload


def test_promoted_profile_settings_are_available():
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    profiles = {profile["profile_id"]: config.profile_config(profile) for profile in config.profiles}
    assert profiles["A"].section("innerVoids")["minVoidPixels"] == 1500
    assert profiles["A"].section("innerVoidQuadFit")["requireExactlyFourPoints"] is True
    assert profiles["B"].section("innerVoids")["minVoidPixels"] == 400
    assert profiles["B"].section("innerVoidQuadFit")["requireExactlyFourPoints"] is False
    assert profiles["C"].section("innerVoids")["maxVoidPixels"] == 399
    assert profiles["C"].section("innerVoidInstanceTracking")["maxCenterDistancePx"] == 80


def test_color_lut_asset_shape_and_dtype():
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    with np.load(config.lut_path, allow_pickle=False) as payload:
        lut = payload["lut"]
    assert lut.shape == (16777216,)
    assert lut.dtype == np.uint8
