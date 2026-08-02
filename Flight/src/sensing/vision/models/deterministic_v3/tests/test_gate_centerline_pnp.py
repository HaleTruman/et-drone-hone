import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.configurations import (
    DEFAULT_DENSITY_CONFIGURATION)
from sensing.vision.models.deterministic_v3.src.gate_centerline_pnp import (
    CAMERA_CALIBRATION, GATE_MODEL, K, OBJECT_POINTS_M, solve_gate_pose)
from sensing.vision.models.deterministic_v3.src.schema import (
    QUADRILATERAL_CORNER_ORDER, STANDARD_ROUTE, QuadrilateralEstimate)


def _quadrilateral(corners, *, image_shape=(360, 640), accepted=True):
    return QuadrilateralEstimate(
        frame_id=17,
        sim_time_ns=123_456_789,
        component_id=3,
        image_shape=image_shape,
        route=STANDARD_ROUTE,
        fitter="test",
        selected_density_profile=DEFAULT_DENSITY_CONFIGURATION.profiles[0],
        corner_order=QUADRILATERAL_CORNER_ORDER,
        corners_uv=corners,
        p90_threshold=0.5,
        p90_evidence_points=40,
        area_px2=100.0 if accepted else None,
        accepted=accepted,
        rejection_reason=None if accepted else "test_rejection",
    )


def test_recovers_known_camera_pose_with_schema_identity():
    expected_rvec = np.array([[0.08], [-0.12], [0.03]], np.float64)
    expected_tvec = np.array([[0.25], [-0.10], [8.0]], np.float64)
    projected, _ = cv2.projectPoints(
        OBJECT_POINTS_M, expected_rvec, expected_tvec, K, None)

    pose = solve_gate_pose(_quadrilateral(
        tuple(map(tuple, projected.reshape(4, 2)))))

    assert pose.accepted
    assert (pose.frame_id, pose.sim_time_ns, pose.component_id) == (
        17, 123_456_789, 3)
    assert pose.camera_calibration_id == CAMERA_CALIBRATION.calibration_id
    assert pose.gate_model_id == GATE_MODEL.model_id
    assert pose.position_confidence > 0.0
    assert pose.orientation_confidence > 0.0
    assert pose.reprojection_rmse_px < 1e-6
    assert np.allclose(pose.position_camera_m, expected_tvec.ravel())


def test_rejects_invalid_or_unaccepted_quadrilateral():
    invalid = solve_gate_pose(_quadrilateral(((1.0, 2.0),)))
    assert not invalid.accepted
    assert invalid.rejection_reason == "invalid_quadrilateral_points"

    unaccepted = solve_gate_pose(_quadrilateral(None, accepted=False))
    assert not unaccepted.accepted
    assert unaccepted.rejection_reason == "quadrilateral_not_accepted"


def test_rejects_image_shape_outside_calibration():
    corners = ((100.0, 100.0), (200.0, 100.0),
               (200.0, 200.0), (100.0, 200.0))
    pose = solve_gate_pose(_quadrilateral(corners, image_shape=(720, 1280)))

    assert not pose.accepted
    assert pose.rejection_reason == "camera_calibration_shape_mismatch"
