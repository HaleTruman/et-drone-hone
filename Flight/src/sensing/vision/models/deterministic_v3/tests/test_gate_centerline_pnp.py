import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.configurations import (
    DEFAULT_DENSITY_CONFIGURATION)
from sensing.vision.models.deterministic_v3.src.gate_centerline_pnp import (
    CAMERA_CALIBRATION, GATE_MODEL, K, OBJECT_POINTS_M, solve_gate_pose)
from sensing.vision.models.deterministic_v3.src.schema import (
    C_SHAPE_ROUTE, QUADRILATERAL_CORNER_ORDER, STANDARD_ROUTE,
    QuadrilateralEstimate)


def _quadrilateral(
    corners, *, image_shape=(360, 640), accepted=True,
    route=STANDARD_ROUTE,
):
    return QuadrilateralEstimate(
        frame_id=17,
        sim_time_ns=123_456_789,
        component_id=3,
        image_shape=image_shape,
        route=route,
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
    assert pose.selected_candidate_rank == 0
    assert pose.candidate_count == len(pose.candidates)
    assert pose.candidate_count == 2
    assert (pose.frame_id, pose.sim_time_ns, pose.component_id) == (
        17, 123_456_789, 3)
    assert pose.camera_calibration_id == CAMERA_CALIBRATION.calibration_id
    assert pose.gate_model_id == GATE_MODEL.model_id
    assert pose.position_confidence > 0.0
    assert pose.orientation_confidence > 0.0
    assert pose.candidates[0].reprojection_rmse_px < 1e-6
    assert np.allclose(
        pose.candidates[0].position_camera_m, expected_tvec.ravel())

    assert tuple(item.candidate_rank for item in pose.candidates) == (0, 1)
    rmses = tuple(item.reprojection_rmse_px for item in pose.candidates)
    assert rmses == tuple(sorted(rmses))
    image_points = projected.reshape(4, 2)
    for candidate in pose.candidates:
        reprojection, _ = cv2.projectPoints(
            OBJECT_POINTS_M,
            np.asarray(candidate.rotation_vector_model_to_camera),
            np.asarray(candidate.position_camera_m),
            K,
            None,
        )
        reproduced_rmse = float(np.sqrt(np.mean(np.sum(
            (reprojection.reshape(4, 2) - image_points) ** 2, axis=1))))
        assert np.isclose(
            reproduced_rmse, candidate.reprojection_rmse_px, atol=1e-12)


def test_rejects_invalid_or_unaccepted_quadrilateral():
    invalid = solve_gate_pose(_quadrilateral(((1.0, 2.0),)))
    assert not invalid.accepted
    assert invalid.rejection_reason == "invalid_quadrilateral_points"
    assert invalid.candidates == ()
    assert invalid.selected_candidate_rank is None
    assert invalid.candidate_count == 0

    unaccepted = solve_gate_pose(_quadrilateral(None, accepted=False))
    assert not unaccepted.accepted
    assert unaccepted.rejection_reason == "quadrilateral_not_accepted"


def test_rejects_image_shape_outside_calibration():
    corners = ((100.0, 100.0), (200.0, 100.0),
               (200.0, 200.0), (100.0, 200.0))
    pose = solve_gate_pose(_quadrilateral(corners, image_shape=(720, 1280)))

    assert not pose.accepted
    assert pose.rejection_reason == "camera_calibration_shape_mismatch"


def test_shared_solver_preserves_c_shape_route_in_accepted_pose():
    projected, _ = cv2.projectPoints(
        OBJECT_POINTS_M, np.zeros((3, 1)), np.array(((0.0,), (0.0,), (7.0,))),
        K, None)

    pose = solve_gate_pose(_quadrilateral(
        tuple(map(tuple, projected.reshape(4, 2))), route=C_SHAPE_ROUTE))

    assert pose.accepted is True
    assert pose.route == C_SHAPE_ROUTE
    assert pose.candidates[0].position_camera_m[2] > 0


def test_high_rmse_rejection_preserves_every_candidate(monkeypatch):
    rvecs = (
        np.zeros((3, 1), np.float64),
        np.array(((0.20,), (-0.10,), (0.05,)), np.float64),
    )
    tvecs = (
        np.array(((0.0,), (0.0,), (10.0,)), np.float64),
        np.array(((0.5,), (0.2,), (12.0,)), np.float64),
    )

    def _synthetic_candidates(*_args, **_kwargs):
        return True, rvecs, tvecs, np.zeros((2, 1), np.float64)

    monkeypatch.setattr(cv2, "solvePnPGeneric", _synthetic_candidates)
    pose = solve_gate_pose(_quadrilateral(
        ((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0))))

    assert pose.accepted is False
    assert pose.rejection_reason == "pnp_reprojection_error"
    assert pose.selected_candidate_rank == 0
    assert pose.candidate_count == len(pose.candidates) == 2
    assert tuple(item.candidate_rank for item in pose.candidates) == (0, 1)
    assert pose.candidates[0].reprojection_rmse_px <= \
        pose.candidates[1].reprojection_rmse_px
    assert pose.position_confidence > 0.0
    assert pose.ambiguity_gap_px is not None
