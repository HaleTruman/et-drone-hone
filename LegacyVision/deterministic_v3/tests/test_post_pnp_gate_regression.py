from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.src.configurations import (
    DEFAULT_DENSITY_CONFIGURATION,
    DEFAULT_GATE_REGRESSION_CONFIGURATION,
)
from sensing.vision.models.deterministic_v3.src.gate_centerline_pnp import (
    CAMERA_CALIBRATION,
    GATE_MODEL,
    SOLVER_NAME,
)
from sensing.vision.models.deterministic_v3.src.gate_regression import (
    REGRESSOR_NAME,
    GatePoseRegressor,
    PostPnPGateRegressor,
)
from sensing.vision.models.deterministic_v3.src.schema import (
    QUADRILATERAL_CORNER_ORDER,
    STANDARD_ROUTE,
    GeometryFrameResult,
    PnPCandidateEstimate,
    PnPRelativePoseEstimate,
    QuadrilateralEstimate,
)


ATTITUDE = (1.0, 0.0, 0.0, 0.0)
K = np.asarray(CAMERA_CALIBRATION.camera_matrix, np.float64)
DISTORTION = np.asarray(
    CAMERA_CALIBRATION.distortion_coefficients, np.float64
)
OBJECT_POINTS = np.asarray(GATE_MODEL.object_points_m, np.float64)


def _vehicle(origin, sim_time_ns):
    return SimpleNamespace(
        position_local_ned_m=tuple(map(float, origin)),
        attitude_quaternion=ATTITUDE,
        sim_time_ns=sim_time_ns,
    )


def _candidate(rank, position, rotation=(0.0, 0.0, 0.0), rmse=0.5):
    return PnPCandidateEstimate(
        candidate_rank=rank,
        rotation_vector_model_to_camera=tuple(map(float, rotation)),
        position_camera_m=tuple(map(float, position)),
        reprojection_rmse_px=float(rmse),
    )


def _frame(
    frame_id,
    sim_time_ns,
    component_id,
    candidates,
    *,
    image_candidate_rank=0,
    accepted=True,
    corner_offset=None,
):
    candidates = tuple(candidates)
    image_candidate = next(
        candidate for candidate in candidates
        if candidate.candidate_rank == image_candidate_rank
    )
    corners, _ = cv2.projectPoints(
        OBJECT_POINTS,
        np.asarray(image_candidate.rotation_vector_model_to_camera, np.float64),
        np.asarray(image_candidate.position_camera_m, np.float64),
        K,
        DISTORTION,
    )
    corners = corners.reshape(4, 2)
    if corner_offset is not None:
        corners = corners + np.asarray(corner_offset, np.float64)
    quadrilateral = QuadrilateralEstimate(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        component_id=component_id,
        image_shape=CAMERA_CALIBRATION.image_shape,
        route=STANDARD_ROUTE,
        fitter="test",
        selected_density_profile=DEFAULT_DENSITY_CONFIGURATION.profiles[0],
        corner_order=QUADRILATERAL_CORNER_ORDER,
        corners_uv=tuple(tuple(map(float, point)) for point in corners),
        p90_threshold=0.5,
        p90_evidence_points=40,
        area_px2=1000.0,
        accepted=True,
        rejection_reason=None,
    )
    raw = PnPRelativePoseEstimate(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        component_id=component_id,
        route=STANDARD_ROUTE,
        solver=SOLVER_NAME,
        camera_calibration_id=CAMERA_CALIBRATION.calibration_id,
        gate_model_id=GATE_MODEL.model_id,
        candidates=candidates,
        selected_candidate_rank=(0 if candidates else None),
        candidate_count=len(candidates),
        ambiguity_gap_px=(
            candidates[1].reprojection_rmse_px
            - candidates[0].reprojection_rmse_px
            if len(candidates) > 1 else None
        ),
        position_confidence=0.93,
        orientation_confidence=0.07,
        accepted=accepted,
        rejection_reason=None if accepted else "pnp_reprojection_error",
    )
    return GeometryFrameResult(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        preprocessing_version="test",
        density_configuration=DEFAULT_DENSITY_CONFIGURATION,
        gate_regression_configuration=DEFAULT_GATE_REGRESSION_CONFIGURATION,
        camera_calibration=CAMERA_CALIBRATION,
        gate_model=GATE_MODEL,
        processed_routes=(STANDARD_ROUTE,),
        topology_decisions=(),
        quadrilateral_estimates=(quadrilateral,),
        pnp_relative_pose_estimates=(raw,),
        camera_pose_estimates=(),
    )


def _empty_frame(frame_id, sim_time_ns):
    source = _frame(
        frame_id,
        sim_time_ns,
        1,
        (_candidate(0, (0.0, 0.0, 10.0)),),
    )
    return replace(
        source,
        quadrilateral_estimates=(),
        pnp_relative_pose_estimates=(),
    )


def test_public_api_keeps_original_import_name_as_an_alias():
    assert PostPnPGateRegressor is GatePoseRegressor
    assert REGRESSOR_NAME == DEFAULT_GATE_REGRESSION_CONFIGURATION.configuration_version


def test_full_pose_regression_is_not_locked_to_current_bearing():
    regressor = GatePoseRegressor()
    first = _frame(
        1, 100, 1,
        (_candidate(0, (0.10, 0.0, 10.4), (0.0, 0.08, 0.0)),),
    )
    second = _frame(
        2, 200, 7,
        (_candidate(0, (-0.10, 0.0, 9.6), (0.0, -0.08, 0.0)),),
    )

    first_result = regressor.refine(first, _vehicle((0.0, 0.0, 0.0), 100))
    second_result = regressor.refine(second, _vehicle((0.0, 0.0, 0.0), 200))

    assert first_result.pnp_relative_pose_estimates == first.pnp_relative_pose_estimates
    assert len(second_result.camera_pose_estimates) == 1
    refined = second_result.camera_pose_estimates[0]
    assert refined.accepted
    raw_position = np.asarray(
        second.pnp_relative_pose_estimates[0].candidates[0].position_camera_m
    )
    final_position = np.asarray(refined.position_camera_m)
    assert not np.allclose(
        final_position[:2] / final_position[2],
        raw_position[:2] / raw_position[2],
    )
    assert not np.allclose(
        refined.rotation_vector_model_to_camera,
        second.pnp_relative_pose_estimates[0]
        .candidates[0].rotation_vector_model_to_camera,
    )
    assert refined.solver == f"{SOLVER_NAME}+{REGRESSOR_NAME}"
    assert refined.regression_evidence.observation_count == 2
    assert regressor.active_track_count == 1
    assert regressor.active_observation_count == 2


def test_temporal_orientation_selects_secondary_candidate():
    regressor = GatePoseRegressor()
    first = _frame(
        1, 100, 1,
        (
            _candidate(0, (0.0, 0.0, 10.0), (0.0, 0.10, 0.0), 0.30),
            _candidate(1, (0.0, 0.0, 10.0), (0.0, 0.75, 0.0), 0.60),
        ),
    )
    second = _frame(
        2, 200, 1,
        (
            _candidate(0, (0.0, 0.0, 10.0), (0.0, 0.70, 0.0), 0.25),
            _candidate(1, (0.0, 0.0, 10.0), (0.0, 0.12, 0.0), 0.55),
        ),
        image_candidate_rank=1,
    )

    regressor.refine(first, _vehicle((0.0, 0.0, 0.0), 100))
    result = regressor.refine(second, _vehicle((0.0, 0.0, 0.0), 200))

    pose = result.camera_pose_estimates[0]
    assert pose.accepted
    assert pose.regression_evidence.selected_candidate_rank == 1
    assert pose.regression_evidence.orientation_residual_deg < 5.0


def test_retains_observations_applies_decay_and_avoids_duplicate_updates():
    immediate = GatePoseRegressor()
    delayed = GatePoseRegressor()
    first = _frame(1, 100, 1, (_candidate(0, (0.0, 0.0, 10.0)),))
    second = _frame(2, 200, 1, (_candidate(0, (0.0, 0.0, 10.0)),))

    immediate.refine(first, _vehicle((0.0, 0.0, 0.0), 100))
    immediate_result = immediate.refine(
        second, _vehicle((0.0, 0.0, 0.0), 200)
    )
    delayed.refine(first, _vehicle((0.0, 0.0, 0.0), 100))
    for frame_id in range(2, 7):
        delayed.refine(
            _empty_frame(frame_id, frame_id * 100),
            _vehicle((0.0, 0.0, 0.0), frame_id * 100),
        )
    delayed_second = replace(second, frame_id=7, sim_time_ns=700)
    delayed_raw = replace(
        delayed_second.pnp_relative_pose_estimates[0],
        frame_id=7,
        sim_time_ns=700,
    )
    delayed_quad = replace(
        delayed_second.quadrilateral_estimates[0],
        frame_id=7,
        sim_time_ns=700,
    )
    delayed_second = replace(
        delayed_second,
        pnp_relative_pose_estimates=(delayed_raw,),
        quadrilateral_estimates=(delayed_quad,),
    )
    delayed_result = delayed.refine(
        delayed_second, _vehicle((0.0, 0.0, 0.0), 700)
    )

    immediate_weight = immediate_result.camera_pose_estimates[0].regression_evidence.effective_history_weight
    delayed_weight = delayed_result.camera_pose_estimates[0].regression_evidence.effective_history_weight
    assert delayed_weight < immediate_weight
    assert delayed.active_observation_count == 2
    assert delayed.refine(
        delayed_second, _vehicle((0.0, 0.0, 0.0), 700)
    ) is delayed_result
    assert delayed.active_observation_count == 2


def test_track_is_evicted_only_after_more_than_fifteen_missed_frames():
    regressor = GatePoseRegressor()
    first = _frame(1, 100, 1, (_candidate(0, (0.0, 0.0, 10.0)),))
    regressor.refine(first, _vehicle((0.0, 0.0, 0.0), 100))

    for frame_id in range(2, 17):
        regressor.refine(
            _empty_frame(frame_id, frame_id * 100),
            _vehicle((0.0, 0.0, 0.0), frame_id * 100),
        )
    assert regressor.active_track_count == 1

    regressor.refine(
        _empty_frame(17, 1700), _vehicle((0.0, 0.0, 0.0), 1700)
    )
    assert regressor.active_track_count == 0
    assert regressor.active_observation_count == 0


def test_gross_outlier_is_retained_as_history_without_corrupting_posterior():
    regressor = GatePoseRegressor()
    first = _frame(
        1, 100, 1,
        (_candidate(0, (0.0, 0.0, 10.0), rmse=0.5),),
    )
    outlier = _frame(
        2, 200, 1,
        (_candidate(0, (0.55, 0.0, 10.0), rmse=3.0),),
    )
    reconfirmed = _frame(
        3, 300, 1,
        (_candidate(0, (0.0, 0.0, 10.0), rmse=0.5),),
    )

    first_result = regressor.refine(
        first, _vehicle((0.0, 0.0, 0.0), 100)
    )
    outlier_result = regressor.refine(
        outlier, _vehicle((0.0, 0.0, 0.0), 200)
    )
    final_result = regressor.refine(
        reconfirmed, _vehicle((0.0, 0.0, 0.0), 300)
    )

    assert not outlier_result.camera_pose_estimates[0].accepted
    assert outlier_result.camera_pose_estimates[0].rejection_reason == \
        "regression_reprojection_error"
    assert final_result.camera_pose_estimates[0].accepted
    assert final_result.camera_pose_estimates[0].regression_evidence.track_id == \
        first_result.camera_pose_estimates[0].regression_evidence.track_id
    assert final_result.camera_pose_estimates[0].regression_evidence.observation_count == 3
    assert regressor.active_observation_count == 3


def test_missing_vehicle_state_keeps_raw_pnp_and_publishes_no_final_pose():
    regressor = GatePoseRegressor()
    frame = _frame(3, 300, 1, (_candidate(0, (0.1, -0.1, 8.0)),))

    result = regressor.refine(frame)

    assert result.pnp_relative_pose_estimates == frame.pnp_relative_pose_estimates
    assert result.camera_pose_estimates == ()
    assert regressor.active_track_count == 0


def test_rejected_raw_pnp_keeps_candidate_evidence_but_has_no_final_pose():
    regressor = GatePoseRegressor()
    frame = _frame(
        4, 400, 1,
        (_candidate(0, (0.0, 0.0, 8.0), rmse=9.0),),
        accepted=False,
    )

    result = regressor.refine(frame, _vehicle((0.0, 0.0, 0.0), 400))

    assert result.pnp_relative_pose_estimates == frame.pnp_relative_pose_estimates
    assert len(result.pnp_relative_pose_estimates[0].candidates) == 1
    assert result.camera_pose_estimates == ()
    assert regressor.active_track_count == 0


def test_final_metrics_are_recomputed_and_reprojection_failure_is_recorded():
    regressor = GatePoseRegressor()
    frame = _frame(
        5,
        500,
        1,
        (_candidate(0, (0.0, 0.0, 8.0), rmse=0.01),),
        corner_offset=((20.0, 0.0),) * 4,
    )

    result = regressor.refine(frame, _vehicle((0.0, 0.0, 0.0), 500))

    pose = result.camera_pose_estimates[0]
    assert not pose.accepted
    assert pose.reprojection_rmse_px > 8.0
    assert pose.rejection_reason == "regression_reprojection_error"
    assert pose.position_camera_m is None
    assert pose.rotation_vector_model_to_camera is None
    assert pose.regression_evidence.observation_count == 1
    assert pose.position_confidence == 0.0
    assert pose.orientation_confidence == 0.0
