from dataclasses import asdict
import json
from types import SimpleNamespace

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.pipeline import (
    GateGeometryPipeline, StandardGatePipeline)
from sensing.vision.models.deterministic_v3.src.schema import (
    C_SHAPE_ROUTE, MULTI_GATE_ROUTE, STANDARD_ROUTE)


MASK_BGR = (220, 120, 40)


def _jpeg_and_lut():
    mask = np.zeros((360, 640), bool)
    mask[90:270, 230:410] = True
    mask[140:220, 280:360] = False
    image = np.zeros((360, 640, 3), np.uint8)
    image[mask] = MASK_BGR
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    bgr = decoded.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    lut = np.zeros(1 << 24, np.uint8)
    lut[np.unique(keys[mask])] = 1
    return encoded.tobytes(), lut


def _c_shape_jpeg_and_lut():
    mask = np.zeros((360, 640), np.uint8)
    corners = np.array(((230, 90), (410, 90), (410, 270), (230, 270)))
    for side_index in (0, 2, 3):
        cv2.line(
            mask, tuple(corners[side_index]),
            tuple(corners[(side_index + 1) % 4]), 1, 15, cv2.LINE_8)
    image = np.zeros((360, 640, 3), np.uint8)
    image[mask != 0] = MASK_BGR
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    bgr = decoded.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    lut = np.zeros(1 << 24, np.uint8)
    lut[np.unique(keys[mask != 0])] = 1
    return encoded.tobytes(), lut


def _vehicle(sim_time_ns):
    return SimpleNamespace(
        position_local_ned_m=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        sim_time_ns=sim_time_ns,
    )


def test_standard_pipeline_reaches_pnp_with_exact_json_identity():
    jpeg_bytes, lut = _jpeg_and_lut()
    result = StandardGatePipeline(lut).process_frame(
        frame_id=81,
        sim_time_ns=123_456_789,
        jpeg_bytes=jpeg_bytes,
        vehicle_state=_vehicle(123_456_789),
    )

    assert (result.frame_id, result.sim_time_ns) == (81, 123_456_789)
    assert result.processed_routes == (
        STANDARD_ROUTE, C_SHAPE_ROUTE, MULTI_GATE_ROUTE)
    assert len(result.topology_decisions) == 1
    assert len(result.quadrilateral_estimates) == 1
    assert len(result.pnp_relative_pose_estimates) == 1
    assert len(result.camera_pose_estimates) == 1
    decision = result.topology_decisions[0]
    quadrilateral = result.quadrilateral_estimates[0]
    pnp_pose = result.pnp_relative_pose_estimates[0]
    pose = result.camera_pose_estimates[0]
    assert decision.accepted and quadrilateral.accepted
    assert pnp_pose.accepted and pose.accepted
    identities = {
        (item.frame_id, item.sim_time_ns, item.component_id)
        for item in (decision, quadrilateral, pnp_pose, pose)
    }
    assert identities == {(81, 123_456_789, 1)}
    assert quadrilateral.selected_density_profile.profile_id == "scale_10"
    assert (
        pnp_pose.candidates[
            pnp_pose.selected_candidate_rank
        ].position_camera_m[2] > 0
    )
    assert pose.position_camera_m[2] > 0

    payload = asdict(result)
    restored = json.loads(json.dumps(payload, allow_nan=False))
    assert restored["frame_id"] == 81
    assert restored["sim_time_ns"] == 123_456_789
    assert restored["quadrilateral_estimates"][0]["component_id"] == 1
    assert restored["pnp_relative_pose_estimates"][0]["component_id"] == 1
    assert restored["pnp_relative_pose_estimates"][0]["candidates"][0][
        "position_camera_m"][2] > 0
    assert restored["camera_pose_estimates"][0]["component_id"] == 1
    assert restored["camera_pose_estimates"][0]["position_camera_m"][2] > 0
    assert restored["camera_calibration"]["camera_matrix"] == [
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 180.0],
        [0.0, 0.0, 1.0],
    ]
    assert restored["gate_model"]["side_length_m"] == 2.1


def test_c_shape_pipeline_publishes_quad_and_official_camera_pose():
    jpeg_bytes, lut = _c_shape_jpeg_and_lut()
    result = GateGeometryPipeline(lut).process_frame(
        frame_id=82,
        sim_time_ns=223_456_789,
        jpeg_bytes=jpeg_bytes,
        vehicle_state=_vehicle(223_456_789),
    )

    assert GateGeometryPipeline is StandardGatePipeline
    assert result.processed_routes == (
        STANDARD_ROUTE, C_SHAPE_ROUTE, MULTI_GATE_ROUTE)
    assert len(result.topology_decisions) == 1
    assert len(result.quadrilateral_estimates) == 1
    assert len(result.pnp_relative_pose_estimates) == 1
    assert len(result.camera_pose_estimates) == 1
    decision = result.topology_decisions[0]
    quadrilateral = result.quadrilateral_estimates[0]
    pnp_pose = result.pnp_relative_pose_estimates[0]
    pose = result.camera_pose_estimates[0]
    assert decision.topology_label == "c_shape"
    assert decision.route == C_SHAPE_ROUTE
    assert decision.accepted is True
    assert quadrilateral.route == C_SHAPE_ROUTE
    assert quadrilateral.accepted is True
    assert quadrilateral.corners_uv is not None
    assert pnp_pose.route == C_SHAPE_ROUTE
    assert pnp_pose.accepted is True
    assert pose.route == C_SHAPE_ROUTE
    assert pose.accepted is True
    assert pose.position_camera_m[2] > 0
    assert {
        (item.frame_id, item.sim_time_ns, item.component_id)
        for item in (decision, quadrilateral, pnp_pose, pose)
    } == {(82, 223_456_789, 1)}


def test_pipeline_withholds_final_pose_without_vehicle_state():
    jpeg_bytes, lut = _jpeg_and_lut()

    result = StandardGatePipeline(lut).process_frame(
        frame_id=83,
        sim_time_ns=323_456_789,
        jpeg_bytes=jpeg_bytes,
    )

    assert len(result.pnp_relative_pose_estimates) == 1
    assert result.pnp_relative_pose_estimates[0].accepted is True
    assert result.camera_pose_estimates == ()
