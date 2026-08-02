from dataclasses import replace

import cv2
import numpy as np
import pytest

from sensing.vision.models.deterministic_v3.src.configurations import (
    DEFAULT_DENSITY_CONFIGURATION)
from sensing.vision.models.deterministic_v3.src.density_bank import DensityBank
from sensing.vision.models.deterministic_v3.src.inverse_density_quadrilateral_production import (
    inverse_density_field, iter_component_inputs, prepare_mask)
from sensing.vision.models.deterministic_v3.src.preprocessing import (
    DEFAULT_CONFIG, PreprocessingConfig, component_mask, preprocess_frame)
from sensing.vision.models.deterministic_v3.src.schema import (
    CLIPPED_TOPOLOGY, C_SHAPE_ROUTE, C_SHAPE_TOPOLOGY, MULTI_GATE_ROUTE,
    MULTI_VOID_TOPOLOGY, STANDARD_ROUTE, STANDARD_TOPOLOGY, UNKNOWN_TOPOLOGY,
    TopologyEvidence)
from sensing.vision.models.deterministic_v3.src.topology import assess_frame
from sensing.vision.models.deterministic_v3.src.topology import (
    DEFAULT_POLICY)


MASK_BGR = (30, 20, 10)


def _lut():
    lut = np.zeros(1 << 24, np.uint8)
    blue, green, red = MASK_BGR
    lut[(red << 16) | (green << 8) | blue] = 1
    return lut


def _three_topologies_image():
    mask = np.zeros((90, 310), bool)

    # Stable one-aperture parent.
    mask[10:70, 10:70] = True
    mask[27:53, 27:53] = False

    # Stable open C-shaped parent.
    mask[10:70, 90:150] = True
    mask[25:55, 105:135] = False
    mask[34:46, 135:150] = False

    # Stable two-aperture parent.
    mask[10:70, 170:290] = True
    mask[25:55, 190:215] = False
    mask[25:55, 245:270] = False

    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = MASK_BGR
    return image


def _frame():
    return preprocess_frame(
        frame_id=41,
        sim_time_ns=123_000,
        image=_three_topologies_image(),
        lut=_lut(),
    )


def _mixed_multi_void_c_shape_image():
    mask = np.zeros((110, 180), bool)
    mask[10:100, 10:170] = True
    mask[30:55, 90:115] = False
    mask[65:90, 125:150] = False
    mask[34:76, 10:80] = False
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = MASK_BGR
    return image


def test_preprocessing_preserves_identity_without_component_mask_copies():
    frame = _frame()
    assert DEFAULT_CONFIG.maximum_input_components == 0
    assert frame.frame_id == 41
    assert frame.sim_time_ns == 123_000
    assert frame.image_shape == (90, 310)
    assert not frame.gated
    assert len(frame.components) == 3
    assert not frame.component_labels.flags.writeable

    for component in frame.components:
        assert not hasattr(component, "raw_mask")
        assert not hasattr(component, "closed_mask")
        local = component_mask(frame, component)
        assert local.shape == component.analysis_shape
        assert int(local.sum()) == component.area_px
        assert not component.touches_frame


def test_preprocessing_matches_the_accepted_mask_and_component_baseline():
    image = _three_topologies_image()
    lut = _lut()
    expected_mask, gated, input_count = prepare_mask(image, lut)
    frame = preprocess_frame(
        frame_id=41, sim_time_ns=123_000, image=image, lut=lut)

    assert not gated
    assert input_count == frame.input_component_count
    np.testing.assert_array_equal(frame.closed_mask, expected_mask)

    expected_components = tuple(iter_component_inputs(expected_mask))
    assert len(expected_components) == len(frame.components)
    for expected, actual in zip(expected_components, frame.components):
        assert expected.label == actual.component_id
        assert expected.bbox == actual.bbox_xywh
        assert expected.area == actual.area_px
        np.testing.assert_array_equal(
            expected.mask, component_mask(frame, actual))


def test_topology_routes_from_closed_hierarchy_without_density():
    frame = _frame()
    standard, c_shape, multi = frame.components
    bank = DensityBank(frame)
    standard_decision, c_shape_decision, multi_decision = assess_frame(
        frame, density_bank=bank)

    for decision, component in zip(
            (standard_decision, c_shape_decision, multi_decision),
            frame.components):
        assert decision.frame_id == frame.frame_id
        assert decision.sim_time_ns == frame.sim_time_ns
        assert decision.component_id == component.component_id

    assert standard_decision.accepted
    assert standard_decision.topology_label == STANDARD_TOPOLOGY
    assert standard_decision.route == STANDARD_ROUTE
    assert standard_decision.density_profile_id is None
    assert standard_decision.closed_significant_holes == 1
    assert standard_decision.classification_rule == \
        "one_child_balanced_aperture"
    assert standard_decision.aperture_center_evidence[0].accepted
    assert bank.cached_profile_ids(standard) == ()

    assert c_shape_decision.accepted
    assert c_shape_decision.topology_label == C_SHAPE_TOPOLOGY
    assert c_shape_decision.route == C_SHAPE_ROUTE
    assert c_shape_decision.closed_significant_holes == 0
    assert c_shape_decision.exterior_void_span_px > 0
    assert c_shape_decision.exterior_void_segment_uv is not None
    assert c_shape_decision.classification_rule == \
        "zero_child_bounded_opening"

    assert multi_decision.accepted
    assert multi_decision.topology_label == MULTI_VOID_TOPOLOGY
    assert multi_decision.route == MULTI_GATE_ROUTE
    assert multi_decision.closed_significant_holes == 2
    assert multi_decision.classification_rule == "two_direct_children"

    raw_disagreement = replace(
        standard,
        topology=TopologyEvidence(
            (), (), standard.topology.closed_hole_contours,
            standard.topology.closed_hole_areas_px2),
    )
    disagreement_frame = replace(
        frame, components=(raw_disagreement, *frame.components[1:]))
    disagreement_decision = assess_frame(disagreement_frame)[0]
    assert disagreement_decision.raw_significant_holes == 0
    assert disagreement_decision.closed_significant_holes == 1
    assert disagreement_decision.topology_label == STANDARD_TOPOLOGY

    clipped = replace(standard, touches_frame=True)
    clipped_frame = replace(
        frame, components=(clipped, *frame.components[1:]))
    clipped_bank = DensityBank(clipped_frame)
    clipped_decision = assess_frame(
        clipped_frame, density_bank=clipped_bank)[0]
    assert clipped_decision.topology_label == CLIPPED_TOPOLOGY
    assert clipped_decision.rejection_reason == "frame_edge_clipped"
    assert clipped_decision.density_profile_id is None
    assert clipped_bank.cached_profile_ids(clipped) == ()


def test_two_direct_children_are_automatically_multi_gate():
    frame = preprocess_frame(
        frame_id=42,
        sim_time_ns=124_000,
        image=_mixed_multi_void_c_shape_image(),
        lut=_lut(),
    )

    assert len(frame.components) == 1
    decision = assess_frame(frame)[0]

    assert decision.raw_significant_holes == 2
    assert decision.closed_significant_holes == 2
    assert decision.frame_id == 42
    assert decision.sim_time_ns == 124_000
    assert decision.component_id == frame.components[0].component_id
    assert decision.topology_label == MULTI_VOID_TOPOLOGY
    assert decision.route == MULTI_GATE_ROUTE
    assert decision.accepted
    assert decision.classification_rule == "two_direct_children"


def test_closed_contour_tree_retains_nested_standard_pairs():
    mask = np.zeros((140, 140), bool)
    mask[10:130, 10:130] = True
    mask[25:115, 25:115] = False
    mask[45:95, 45:95] = True
    mask[58:82, 58:82] = False
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = MASK_BGR
    frame = preprocess_frame(
        frame_id=43, sim_time_ns=125_000, image=image, lut=_lut())

    assert len(frame.components) == 2
    assert [decision.topology_label for decision in assess_frame(frame)] == [
        STANDARD_TOPOLOGY, STANDARD_TOPOLOGY]
    assert any(
        not node.is_hole and node.depth == 2
        for node in frame.closed_contour_nodes)
    for component in frame.components:
        assert len(component.contours.parent_contour_ids) == 1
        assert len(component.contours.parent_child_contour_ids) == 1


def test_zero_child_open_parent_is_c_shape_but_solid_is_unknown():
    mask = np.zeros((100, 190), bool)
    mask[10:90, 10:80] = True
    mask[25:75, 30:80] = False
    mask[10:90, 110:180] = True
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = MASK_BGR
    frame = preprocess_frame(
        frame_id=44, sim_time_ns=126_000, image=image, lut=_lut())
    c_shape, solid = assess_frame(frame)

    assert c_shape.topology_label == C_SHAPE_TOPOLOGY
    assert c_shape.exterior_void_span_px > 0
    assert solid.topology_label == UNKNOWN_TOPOLOGY
    assert solid.classification_rule == "zero_child_solid"
    assert solid.rejection_reason == "solid_without_void"


def test_one_child_large_opening_precedes_standard_center_confirmation():
    mask = np.zeros((170, 230), bool)
    mask[10:160, 10:220] = True
    mask[45:85, 35:80] = False
    mask[35:135, 125:220] = False
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = MASK_BGR
    frame = preprocess_frame(
        frame_id=45, sim_time_ns=127_000, image=image, lut=_lut())
    decision = assess_frame(frame)[0]

    assert decision.closed_significant_holes == 1
    assert decision.exterior_void_span_px >= 75
    assert decision.topology_label == MULTI_VOID_TOPOLOGY
    assert decision.classification_rule == "one_child_plus_large_opening"
    higher_threshold = replace(
        DEFAULT_POLICY,
        mixed_opening_minimum_span_px=decision.exterior_void_span_px + 1,
    )
    rerouted = assess_frame(frame, policy=higher_threshold)[0]
    assert rerouted.topology_label == STANDARD_TOPOLOGY


def test_unbalanced_aperture_center_is_rejected_with_explicit_evidence():
    mask = np.zeros((110, 110), np.uint8)
    mask[10:100, 10:100] = 1
    child = np.array(
        [[25, 25], [85, 25], [35, 40], [85, 80], [25, 80]],
        np.int32,
    )
    cv2.fillPoly(mask, [child], 0)
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask != 0] = MASK_BGR
    frame = preprocess_frame(
        frame_id=47, sim_time_ns=129_000, image=image, lut=_lut())
    decision = assess_frame(frame)[0]

    assert decision.topology_label == UNKNOWN_TOPOLOGY
    assert decision.rejection_reason == \
        "standard_aperture_center_unbalanced"
    assert "radial_balance_below_minimum" in \
        decision.aperture_center_evidence[0].failed_checks


def test_more_than_two_direct_children_are_unknown():
    mask = np.zeros((110, 190), bool)
    mask[10:100, 10:180] = True
    for left in (30, 80, 130):
        mask[35:75, left:left + 25] = False
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = MASK_BGR
    frame = preprocess_frame(
        frame_id=46, sim_time_ns=128_000, image=image, lut=_lut())
    decision = assess_frame(frame)[0]

    assert decision.closed_significant_holes == 3
    assert decision.topology_label == UNKNOWN_TOPOLOGY
    assert decision.rejection_reason == "unsupported_direct_child_count"


def test_density_bank_is_lazy_profile_complete_and_component_local():
    frame = _frame()
    component = frame.components[0]
    bank = DensityBank(frame)

    assert bank.profile_ids() == tuple(
        f"scale_{index:02d}" for index in range(1, 11))
    assert DEFAULT_DENSITY_CONFIGURATION.calibration_version.startswith(
        "inverse-density-v3:runs-20260731T093159Z+093616Z+093732Z:"
        "nonclipped-area-deciles-higher:")
    assert tuple(
        (
            profile.maximum_component_area_px,
            profile.density_radius_px,
            profile.ridge_radius_px,
            profile.relative_cap,
            profile.inverse_gamma,
            profile.ridge_gamma,
        )
        for profile in DEFAULT_DENSITY_CONFIGURATION.profiles
    ) == (
        (146, 2, 2, 2.0, 2.10, 3.00),
        (190, 3, 2, 2.0, 2.10, 3.00),
        (242, 4, 3, 2.0, 2.10, 3.00),
        (260, 5, 4, 2.0, 2.10, 3.00),
        (372, 6, 5, 2.0, 2.10, 3.00),
        (827, 7, 6, 2.0, 2.10, 3.00),
        (1_370, 9, 7, 2.0, 2.10, 3.00),
        (3_113, 12, 10, 2.0, 2.10, 3.00),
        (4_671, 16, 13, 2.0, 2.10, 3.00),
        (None, 20, 16, 2.0, 2.10, 3.00),
    )
    assert bank.recommended_standard_profile(component).profile_id == "scale_08"
    assert bank.cached_profile_ids(component) == ()

    first = bank.get(component, "scale_02")
    assert first is bank.get(component, "scale_02")
    assert bank.cached_profile_ids(component) == ("scale_02",)
    assert first.final_field.shape == component.analysis_shape
    assert not first.final_field.flags.writeable
    assert np.count_nonzero(first.p90_mask) <= np.count_nonzero(first.p80_mask)
    assert np.count_nonzero(first.p80_mask) <= np.count_nonzero(first.p70_mask)

    variants = bank.precompute(component)
    assert len(variants) == 10
    assert bank.cached_profile_ids(component) == bank.profile_ids()
    configuration = bank.configuration()
    assert configuration.ignore_frame_edge_clipped is True
    assert configuration.profiles == tuple(bank.profiles.values())


def test_density_bank_uses_one_fully_configurable_profile_deck():
    frame = _frame()
    first_profile = replace(
        DEFAULT_DENSITY_CONFIGURATION.profiles[0],
        maximum_component_area_px=150,
        density_radius_px=3,
        ridge_radius_px=4,
        relative_cap=1.75,
        inverse_gamma=2.50,
        ridge_gamma=3.50,
    )
    configuration = replace(
        DEFAULT_DENSITY_CONFIGURATION,
        profiles=(first_profile,
                  *DEFAULT_DENSITY_CONFIGURATION.profiles[1:]),
    )

    bank = DensityBank(frame, configuration)
    evidence = bank.get(frame.components[0], "scale_01")

    assert bank.configuration() is configuration
    assert evidence.profile is first_profile
    assert evidence.profile.maximum_component_area_px == 150
    assert evidence.profile.density_radius_px == 3
    assert evidence.profile.ridge_radius_px == 4
    assert evidence.profile.relative_cap == 1.75
    assert evidence.profile.inverse_gamma == 2.50
    assert evidence.profile.ridge_gamma == 3.50


def test_density_bank_rejects_nonpositive_profile_calibration():
    frame = _frame()
    invalid_profile = replace(
        DEFAULT_DENSITY_CONFIGURATION.profiles[0], inverse_gamma=0.0)
    configuration = replace(
        DEFAULT_DENSITY_CONFIGURATION,
        profiles=(invalid_profile,
                  *DEFAULT_DENSITY_CONFIGURATION.profiles[1:]),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        DensityBank(frame, configuration)


def test_density_bank_ignores_only_clipped_components_by_default():
    frame = _frame()
    clipped = replace(frame.components[0], touches_frame=True)
    clipped_frame = replace(
        frame, components=(clipped, *frame.components[1:]))
    bank = DensityBank(clipped_frame)

    assert DEFAULT_DENSITY_CONFIGURATION.ignore_frame_edge_clipped is True
    assert not bank.includes_component(clipped)
    assert bank.includes_component(clipped_frame.components[1])
    assert bank.precompute(clipped) == ()
    with pytest.raises(ValueError, match="ignored for density"):
        bank.get(clipped, "scale_01")

    review_bank = DensityBank(clipped_frame, replace(
        DEFAULT_DENSITY_CONFIGURATION,
        ignore_frame_edge_clipped=False,
    ))
    assert len(review_bank.precompute(clipped)) == 10


def test_density_bank_reproduces_the_accepted_inverse_density_math():
    frame = _frame()
    component = frame.components[0]
    bank = DensityBank(frame)

    for profile_id in bank.profile_ids():
        evidence = bank.get(component, profile_id)
        mask = component_mask(frame, component)
        expected = inverse_density_field(
            mask,
            evidence.profile.density_radius_px,
            evidence.profile.ridge_radius_px,
        )
        np.testing.assert_allclose(evidence.final_field, expected, rtol=0, atol=0)


def test_component_count_gate_retains_a_rejected_frame_record():
    config = PreprocessingConfig(maximum_input_components=3)
    frame = preprocess_frame(
        frame_id=9,
        sim_time_ns=10,
        image=_three_topologies_image(),
        lut=_lut(),
        config=config,
    )
    assert frame.gated
    assert frame.rejection_reason == "frame_component_gate"
    assert frame.input_component_count == 3
    assert frame.components == ()
    assert np.count_nonzero(frame.base_mask) > 0
    assert np.count_nonzero(frame.closed_mask) == 0


def test_zero_component_count_gate_is_explicitly_disabled():
    frame = preprocess_frame(
        frame_id=9,
        sim_time_ns=10,
        image=_three_topologies_image(),
        lut=_lut(),
        config=PreprocessingConfig(maximum_input_components=0),
    )

    assert frame.input_component_count == 3
    assert not frame.gated
    assert len(frame.components) == 3

    with pytest.raises(ValueError, match="zero or positive"):
        PreprocessingConfig(maximum_input_components=-1)
