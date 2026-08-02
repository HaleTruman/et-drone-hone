from dataclasses import replace

import cv2
import numpy as np
import pytest

from sensing.vision.models.deterministic_v3.src.density_bank import DensityBank
from sensing.vision.models.deterministic_v3.src.pipeline import (
    StandardGatePipeline,
)
from sensing.vision.models.deterministic_v3.src.preprocessing import (
    preprocess_frame,
)
from sensing.vision.models.deterministic_v3.src.standard_gate_processing import (
    process_standard_components,
    process_standard_gate,
    quadrilateral_from_standard,
)


MASK_BGR = (30, 20, 10)


def _lut():
    lut = np.zeros(1 << 24, np.uint8)
    blue, green, red = MASK_BGR
    lut[(red << 16) | (green << 8) | blue] = 1
    return lut


def _frame(mask, *, frame_id=31):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask != 0] = MASK_BGR
    return preprocess_frame(
        frame_id=frame_id,
        sim_time_ns=987_654_321,
        image=image,
        lut=_lut(),
    )


def _ring_mask():
    mask = np.zeros((300, 440), np.uint8)
    mask[60:240, 120:320] = 1
    mask[110:190, 180:260] = 0
    return mask


def _calibrated_ring_mask():
    mask = np.zeros((360, 640), np.uint8)
    mask[90:270, 230:410] = 1
    mask[140:220, 280:360] = 0
    return mask


def _jpeg_and_decoded_lut(mask):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask != 0] = MASK_BGR
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.uint32)
    keys = (
        (decoded[:, :, 2] << 16)
        | (decoded[:, :, 1] << 8)
        | decoded[:, :, 0]
    )
    lut = np.zeros(1 << 24, np.uint8)
    lut[np.unique(keys[mask != 0])] = 1
    return encoded.tobytes(), lut


def test_independent_standard_process_accepts_one_supported_aperture():
    frame = _frame(_ring_mask())
    component = frame.components[0]
    bank = DensityBank(frame)

    result = process_standard_gate(frame, component, bank)
    quadrilateral = quadrilateral_from_standard(result)

    assert result.accepted is True
    assert result.high_confidence is True
    assert result.fit_confidence >= result.high_confidence_threshold
    assert result.significant_parent_child_contour_ids
    density = bank.get(
        component, result.selected_density_profile.profile_id
    )
    assert result.p70_threshold == density.p70_threshold
    assert result.p70_evidence_points == int(
        np.count_nonzero(density.p70_mask)
    )
    assert 0.80 <= result.p70_coverage_ratio <= 1.0
    assert len(result.side_evidence) == 4
    assert all(side.support_points >= 2 for side in result.side_evidence)
    assert (result.frame_id, result.sim_time_ns, result.component_id) == (
        frame.frame_id, frame.sim_time_ns, component.component_id
    )
    assert quadrilateral.accepted is True
    assert quadrilateral.fit_confidence == result.fit_confidence
    assert quadrilateral.corners_uv == result.fitted_corners_uv


def test_live_pipeline_sends_only_the_high_confidence_fit_to_pnp():
    jpeg_bytes, lut = _jpeg_and_decoded_lut(_calibrated_ring_mask())

    frame_result = StandardGatePipeline(lut).process_frame(
        frame_id=41,
        sim_time_ns=123_456_789,
        jpeg_bytes=jpeg_bytes,
    )

    assert len(frame_result.standard_gate_results) == 1
    standard = frame_result.standard_gate_results[0]
    assert standard.accepted and standard.high_confidence
    assert len(frame_result.quadrilateral_estimates) == 1
    assert frame_result.quadrilateral_estimates[0].fit_confidence == \
        standard.fit_confidence
    assert len(frame_result.pnp_relative_pose_estimates) == 1
    assert frame_result.pnp_relative_pose_estimates[0].accepted is True
    assert frame_result.camera_pose_estimates == ()


def test_live_pipeline_does_not_send_rejected_standard_candidate_to_pnp():
    mask = np.zeros((360, 640), np.uint8)
    mask[90:270, 230:410] = 1
    jpeg_bytes, lut = _jpeg_and_decoded_lut(mask)

    frame_result = StandardGatePipeline(lut).process_frame(
        frame_id=42,
        sim_time_ns=223_456_789,
        jpeg_bytes=jpeg_bytes,
    )

    assert len(frame_result.standard_gate_results) == 1
    assert frame_result.standard_gate_results[0].accepted is False
    assert frame_result.standard_gate_results[0].rejection_reason == \
        "standard_aperture_missing"
    assert frame_result.quadrilateral_estimates == ()
    assert frame_result.pnp_relative_pose_estimates == ()
    assert frame_result.camera_pose_estimates == ()


@pytest.mark.parametrize(
    ("mask_factory", "reason"),
    (
        (
            lambda: cv2.rectangle(
                np.zeros((300, 440), np.uint8),
                (120, 60),
                (319, 239),
                1,
                cv2.FILLED,
            ),
            "standard_aperture_missing",
        ),
        (
            lambda: _two_aperture_mask(),
            "standard_multiple_apertures",
        ),
        (
            lambda: _clipped_ring_mask(),
            "standard_frame_edge_clipped",
        ),
    ),
)
def test_independent_standard_process_rejects_nonstandard_evidence(
    mask_factory, reason
):
    frame = _frame(mask_factory())
    result = process_standard_gate(
        frame, frame.components[0], DensityBank(frame)
    )

    assert result.accepted is False
    assert result.high_confidence is False
    assert result.rejection_reason == reason


def _two_aperture_mask():
    mask = np.zeros((300, 440), np.uint8)
    mask[60:240, 100:340] = 1
    mask[105:195, 135:205] = 0
    mask[105:195, 235:305] = 0
    return mask


def _clipped_ring_mask():
    mask = np.zeros((300, 440), np.uint8)
    mask[0:180, 100:300] = 1
    mask[45:125, 160:240] = 0
    return mask


def test_frame_processor_retains_one_ordered_assessment_per_component():
    mask = np.zeros((300, 640), np.uint8)
    mask[60:240, 50:210] = 1
    mask[110:190, 100:160] = 0
    mask[60:240, 400:560] = 1
    frame = _frame(mask)
    results = process_standard_components(frame, DensityBank(frame))

    assert tuple(result.component_id for result in results) == tuple(
        component.component_id for component in frame.components
    )
    assert tuple(result.accepted for result in results) == (True, False)
    assert results[1].rejection_reason == "standard_aperture_missing"


def test_standard_process_rejects_foreign_frame_and_bank_provenance():
    frame = _frame(_ring_mask(), frame_id=31)
    other = _frame(_ring_mask(), frame_id=32)

    with pytest.raises(ValueError, match="component does not belong"):
        process_standard_gate(frame, other.components[0], DensityBank(frame))
    with pytest.raises(ValueError, match="density bank belongs"):
        process_standard_gate(frame, frame.components[0], DensityBank(other))


def test_confidence_threshold_is_authoritative_for_quad_admission():
    frame = _frame(_ring_mask())
    result = process_standard_gate(
        frame, frame.components[0], DensityBank(frame)
    )
    rejected = replace(
        result,
        fit_confidence=result.high_confidence_threshold - 0.01,
        high_confidence=False,
        accepted=False,
        rejection_reason="standard_fit_confidence_below_minimum",
    )

    quadrilateral = quadrilateral_from_standard(rejected)

    assert quadrilateral.accepted is False
    assert quadrilateral.corners_uv is None
    assert quadrilateral.rejection_reason == \
        "standard_fit_confidence_below_minimum"
