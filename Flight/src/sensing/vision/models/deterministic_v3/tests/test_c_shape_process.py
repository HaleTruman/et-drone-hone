from dataclasses import replace

import cv2
import numpy as np
import pytest

from sensing.vision.models.deterministic_v3.src.c_shape.process import (
    process_c_shape, quadrilateral_from_c_shape,
    select_c_shape_density_profile_id)
from sensing.vision.models.deterministic_v3.src.c_shape.c_shape_tailored_shortfall_baseline import (
    fit_c_shape_geometry_baseline)
from sensing.vision.models.deterministic_v3.src.c_shape_three_line_pose import (
    CShapeDensityInput)
from sensing.vision.models.deterministic_v3.src.density_bank import DensityBank
from sensing.vision.models.deterministic_v3.src.preprocessing import (
    component_mask, preprocess_frame)
from sensing.vision.models.deterministic_v3.src.topology import assess_frame
from sensing.vision.models.deterministic_v3.ui.backend.schema_json import (
    assert_runtime_equal, runtime_object, runtime_value)


def _c_shape_frame():
    image = np.zeros((140, 140, 3), np.uint8)
    corners = np.array(((30, 30), (110, 30), (110, 110), (30, 110)))
    for side_index in (0, 2, 3):
        cv2.line(
            image, tuple(corners[side_index]),
            tuple(corners[(side_index + 1) % 4]),
            (255, 255, 255), 11, cv2.LINE_8)
    lut = np.zeros(1 << 24, np.uint8)
    lut[0xffffff] = 1
    return preprocess_frame(
        frame_id=41, sim_time_ns=1234, image=image, lut=lut)


def test_c_shape_dimension_rules_reuse_exact_density_bank_profiles():
    component = _c_shape_frame().components[0]

    selections = tuple(
        select_c_shape_density_profile_id(replace(
            component, bbox_xywh=(0, 0, dimension, dimension)))
        for dimension in (35, 36, 59, 60, 89, 90)
    )

    assert selections == (
        "scale_01", "scale_03", "scale_03",
        "scale_06", "scale_06", "scale_10")


def test_c_shape_route_publishes_three_lines_and_completed_refined_mask():
    frame = _c_shape_frame()
    component = frame.components[0]
    decision = assess_frame(frame)[0]
    bank = DensityBank(frame)

    result = process_c_shape(frame, component, decision, bank)

    assert decision.topology_label == "c_shape"
    assert decision.route == "c_shape"
    assert result.accepted is True
    assert result.topology_label == "c_shape"
    assert result.route == "c_shape"
    assert result.selected_density_profile.profile_id == "scale_10"
    assert len(result.visible_lines) == 3
    assert result.missing_side_index is not None
    assert result.refined_mask.dtype == np.uint8
    assert result.refined_mask.flags.writeable is False
    assert result.refined_mask_line_width_px == 2
    assert np.any(result.refined_mask)

    missing_start = np.asarray(
        result.completed_quadrilateral_uv[result.missing_side_index])
    missing_end = np.asarray(result.completed_quadrilateral_uv[
        (result.missing_side_index + 1) % 4])
    midpoint = np.rint(0.5 * (missing_start + missing_end)).astype(int)
    origin = np.asarray(result.refined_mask_origin_uv)
    local_x, local_y = midpoint - origin
    assert result.refined_mask[local_y, local_x] == 1

    evidence = bank.get(component, result.selected_density_profile.profile_id)
    baseline = fit_c_shape_geometry_baseline(CShapeDensityInput(
        "41:1", component_mask(frame, component), evidence.final_field,
        component.image_origin_uv))
    np.testing.assert_allclose(
        result.completed_quadrilateral_uv,
        baseline.candidate.quadrilateral_uv)

    restored = runtime_object(runtime_value(result))
    assert_runtime_equal(result, restored)

    quadrilateral = quadrilateral_from_c_shape(result)
    assert quadrilateral.accepted is True
    assert quadrilateral.route == "c_shape"
    assert quadrilateral.fitter == result.fitter
    assert quadrilateral.selected_density_profile == \
        result.selected_density_profile
    assert quadrilateral.p90_evidence_points == result.p90_evidence_points
    assert quadrilateral.corner_order == (
        "upper_left", "upper_right", "lower_right", "lower_left")
    assert quadrilateral.corners_uv is not None


def test_c_shape_route_rejects_a_non_c_shape_label():
    frame = _c_shape_frame()
    component = frame.components[0]
    decision = replace(
        assess_frame(frame)[0], topology_label="standard",
        route="standard_gate")

    with pytest.raises(ValueError, match="accepted c_shape route"):
        process_c_shape(frame, component, decision, DensityBank(frame))
