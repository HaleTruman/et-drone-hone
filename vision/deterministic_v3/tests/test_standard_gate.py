from dataclasses import replace

import cv2
import numpy as np
import pytest

from sensing.vision.models.deterministic_v3.src.density_bank import DensityBank
from sensing.vision.models.deterministic_v3.src.preprocessing import (
    preprocess_frame)
from sensing.vision.models.deterministic_v3.src.schema import (
    QUADRILATERAL_CORNER_ORDER, UNKNOWN_TOPOLOGY)
from sensing.vision.models.deterministic_v3.src.standard_gate import (
    FITTER_NAME, fit_standard_gate)
from sensing.vision.models.deterministic_v3.src.topology import assess_frame


MASK_BGR = (30, 20, 10)


def _lut():
    lut = np.zeros(1 << 24, np.uint8)
    blue, green, red = MASK_BGR
    lut[(red << 16) | (green << 8) | blue] = 1
    return lut


def _frame():
    image = np.zeros((360, 640, 3), np.uint8)
    image[90:270, 220:420] = MASK_BGR
    image[140:220, 280:360] = 0
    return preprocess_frame(
        frame_id=71, sim_time_ns=987_654_321, image=image, lut=_lut())


def test_standard_gate_uses_recommended_cached_profile_and_provenance():
    frame = _frame()
    component = frame.components[0]
    bank = DensityBank(frame)
    decision = assess_frame(frame, density_bank=bank)[0]
    profile = bank.recommended_standard_profile(component)
    density = bank.get(component, profile.profile_id)

    result = fit_standard_gate(frame, component, decision, density)

    assert result.accepted
    assert (result.frame_id, result.sim_time_ns, result.component_id) == (
        frame.frame_id, frame.sim_time_ns, component.component_id)
    assert result.fitter == FITTER_NAME
    assert result.selected_density_profile == profile
    assert result.corner_order == QUADRILATERAL_CORNER_ORDER
    assert result.p90_threshold == density.p90_threshold
    assert result.p90_evidence_points == int(np.count_nonzero(density.p90_mask))
    corners = np.asarray(result.corners_uv, np.float32)
    assert cv2.isContourConvex(corners.reshape(-1, 1, 2))
    assert result.area_px2 == pytest.approx(cv2.contourArea(
        corners.reshape(-1, 1, 2)))


def test_standard_gate_refuses_a_nonstandard_topology_decision():
    frame = _frame()
    component = frame.components[0]
    bank = DensityBank(frame)
    decision = assess_frame(frame, density_bank=bank)[0]
    profile = bank.recommended_standard_profile(component)
    density = bank.get(component, profile.profile_id)
    invalid = replace(
        decision, topology_label=UNKNOWN_TOPOLOGY, route=None, accepted=False)

    with pytest.raises(ValueError, match="accepted standard route"):
        fit_standard_gate(frame, component, invalid, density)
