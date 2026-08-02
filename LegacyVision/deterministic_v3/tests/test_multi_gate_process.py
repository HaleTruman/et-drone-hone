"""Focused overlap-identification and multi-gate process contracts."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.configurations import (
    REVIEW_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
)
from sensing.vision.models.deterministic_v3.src.density_bank import DensityBank
from sensing.vision.models.deterministic_v3.src.multi_gate.identification import (
    identify_multi_gate_candidates,
)
from sensing.vision.models.deterministic_v3.src.multi_gate.process import (
    process_multi_gate,
    quadrilaterals_from_multi_gate,
)
from sensing.vision.models.deterministic_v3.src.pipeline import (
    GateGeometryPipeline,
)
from sensing.vision.models.deterministic_v3.src.preprocessing import (
    preprocess_frame,
)
from sensing.vision.models.deterministic_v3.src.schema import MULTI_GATE_ROUTE
from sensing.vision.models.deterministic_v3.src.topology import assess_frame
from sensing.vision.models.deterministic_v3.ui.backend.schema_json import (
    assert_runtime_equal,
    runtime_object,
    runtime_value,
)


MASK_BGR = (220, 120, 40)


def _frame(mask: np.ndarray):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask != 0] = 255
    lut = np.zeros(1 << 24, np.uint8)
    lut[0xffffff] = 1
    return preprocess_frame(
        frame_id=91,
        sim_time_ns=987_654_321,
        image=image,
        lut=lut,
    )


def _two_aperture_mask(shape=(190, 190)) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.rectangle(mask, (15, 30), (90, 120), 1, 13)
    cv2.rectangle(mask, (90, 30), (165, 120), 1, 13)
    return mask


def _pipeline_input():
    mask = np.zeros((360, 640), np.uint8)
    cv2.rectangle(mask, (230, 120), (330, 220), 1, 13)
    cv2.rectangle(mask, (330, 120), (430, 220), 1, 13)
    image = np.zeros((360, 640, 3), np.uint8)
    image[mask != 0] = MASK_BGR
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    bgr = decoded.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    lut = np.zeros(1 << 24, np.uint8)
    lut[np.unique(keys[mask != 0])] = 1
    return encoded.tobytes(), lut


def test_identifier_scans_all_components_without_overriding_topology():
    mask = np.zeros((180, 360), np.uint8)
    mask[20:150, 72:88] = 1
    mask[20:36, 25:135] = 1
    mask[35:145, 250:266] = 1
    mask[129:145, 250:335] = 1
    frame = _frame(mask)
    decisions = assess_frame(frame)

    result = identify_multi_gate_candidates(
        frame, decisions, DensityBank(frame)
    )

    assert len(result.component_assessments) == len(frame.components) == 2
    candidates = [
        item for item in result.component_assessments
        if item.experimental_overlap_candidate
    ]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_reason == \
        "thickness_excess_with_multi_arm_junction"
    assert candidate.routing_eligible is True
    assert candidate.fitter_ready is False
    assert candidate.topology_route is None
    assert candidate.fitter_readiness_reason == \
        "authoritative_topology_route_not_multi_gate"
    assert candidate.density_overlap is not None
    assert result.fitter_ready_component_ids == ()


def test_two_aperture_candidate_reaches_two_normalized_gate_fits():
    frame = _frame(_two_aperture_mask())
    component = frame.components[0]
    decision = assess_frame(frame)[0]
    bank = DensityBank(frame)
    identification = identify_multi_gate_candidates(
        frame, (decision,), bank
    ).component_assessments[0]

    result = process_multi_gate(
        frame,
        component,
        decision,
        bank,
        identification=identification,
    )
    quadrilaterals = quadrilaterals_from_multi_gate(result)

    assert decision.route == MULTI_GATE_ROUTE
    assert decision.closed_significant_holes == 2
    assert identification.fitter_ready is True
    assert result.accepted is True
    assert result.selected_density_profile == bank.get(
        component, result.selected_density_profile.profile_id
    ).profile
    assert 0.0 <= result.weighted_union_coverage <= 1.0
    assert len(result.aperture_fits) == 2
    assert tuple(fit.aperture_role for fit in result.aperture_fits) == (
        "larger", "smaller"
    )
    assert tuple(item.gate_index for item in quadrilaterals) == (0, 1)
    assert all(item.accepted for item in quadrilaterals)
    assert all(item.corners_uv is not None for item in quadrilaterals)
    assert len({
        fit.child_contour_id for fit in result.aperture_fits
    }) == 2

    restored = runtime_object(runtime_value(result))
    assert_runtime_equal(result, restored)


def test_clipped_candidate_is_retained_as_review_only_evidence():
    mask = np.zeros((200, 200), np.uint8)
    mask[60:150, 92:108] = 1
    mask[60:76, 45:155] = 1
    mask[66:70, 0:46] = 1
    frame = _frame(mask)
    decision = assess_frame(frame)[0]
    live = identify_multi_gate_candidates(
        frame, (decision,), DensityBank(frame)
    ).component_assessments[0]

    assessment = identify_multi_gate_candidates(
        frame,
        (decision,),
        DensityBank(frame),
        REVIEW_MULTI_GATE_IDENTIFICATION_CONFIGURATION,
    ).component_assessments[0]

    assert live.assessment == "indeterminate"
    assert live.candidate_reason == "frame_edge_clipped_review_disabled"
    assert live.junctions == ()
    assert assessment.touches_frame is True
    assert assessment.assessment == "candidate_clipped_review_only"
    assert assessment.experimental_overlap_candidate is True
    assert assessment.routing_eligible is False
    assert assessment.fitter_ready is False
    assert assessment.fitter_readiness_reason == "frame_edge_clipped"
    assert assessment.density_overlap is None
    assert any(
        junction.candidate and junction.support_fully_observed
        for junction in assessment.junctions
    )


def test_pipeline_keeps_both_multi_gate_identities_through_pnp():
    jpeg_bytes, lut = _pipeline_input()
    pipeline = GateGeometryPipeline(lut)

    result = pipeline.process_frame(
        frame_id=92,
        sim_time_ns=987_654_322,
        jpeg_bytes=jpeg_bytes,
        vehicle_state=SimpleNamespace(
            position_local_ned_m=(0.0, 0.0, 0.0),
            attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
            sim_time_ns=987_654_322,
        ),
    )

    assert MULTI_GATE_ROUTE in result.processed_routes
    assert result.multi_gate_identification is not None
    assert result.multi_gate_identification.fitter_ready_component_ids == (1,)
    assert len(result.multi_gate_results) == 1
    assert result.multi_gate_results[0].accepted is True
    assert tuple(item.gate_index for item in result.quadrilateral_estimates) == (
        0, 1
    )
    assert tuple(
        item.gate_index for item in result.pnp_relative_pose_estimates
    ) == (
        0, 1
    )
    assert tuple(item.gate_index for item in result.camera_pose_estimates) == (
        0, 1
    )
    assert all(item.route == MULTI_GATE_ROUTE
               for item in result.quadrilateral_estimates)
    assert all(item.accepted for item in result.pnp_relative_pose_estimates)
    assert all(item.accepted for item in result.camera_pose_estimates)
    assert pipeline.regressor.active_track_count == 2
