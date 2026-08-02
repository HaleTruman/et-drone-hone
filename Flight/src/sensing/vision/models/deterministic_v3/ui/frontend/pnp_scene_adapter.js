export const PNP_SCENE_SCHEMA_READY = true;
export const PNP_SCENE_COORDINATE_FRAME =
  'OpenCV optical camera: +X right, +Y down, +Z forward';

function finiteVector(value, length) {
  return Array.isArray(value) && value.length === length &&
    value.every((item) => Number.isFinite(Number(item)));
}

function validCameraCalibration(value) {
  return value && typeof value.calibration_id === 'string' &&
    finiteVector(value.image_shape, 2) &&
    Array.isArray(value.camera_matrix) && value.camera_matrix.length === 3 &&
    value.camera_matrix.every((row) => finiteVector(row, 3)) &&
    Array.isArray(value.distortion_coefficients) &&
    value.distortion_coefficients.every((item) => Number.isFinite(Number(item)));
}

function validGateModel(value) {
  return value && typeof value.model_id === 'string' &&
    Number.isFinite(Number(value.side_length_m)) &&
    Array.isArray(value.object_points_m) && value.object_points_m.length === 4 &&
    value.object_points_m.every((point) => finiteVector(point, 3));
}

function candidatePose(estimate, candidate, role) {
  if (!candidate || !Number.isInteger(candidate.candidate_rank) ||
      !finiteVector(candidate.rotation_vector_model_to_camera, 3) ||
      !finiteVector(candidate.position_camera_m, 3) ||
      !Number.isFinite(Number(candidate.reprojection_rmse_px))) return null;
  return {
    ...candidate,
    frame_id: estimate.frame_id,
    sim_time_ns: estimate.sim_time_ns,
    component_id: estimate.component_id,
    gate_index: estimate.gate_index,
    route: estimate.route,
    camera_calibration_id: estimate.camera_calibration_id,
    gate_model_id: estimate.gate_model_id,
    candidate_role: role,
    source_estimate: estimate,
  };
}

function pnpCandidates(estimate, cameraCalibration, gateModel) {
  if (!estimate || estimate.accepted !== true ||
      estimate.camera_calibration_id !== cameraCalibration.calibration_id ||
      estimate.gate_model_id !== gateModel.model_id ||
      !Array.isArray(estimate.candidates)) return null;
  const selectedRank = estimate.selected_candidate_rank;
  if (!Number.isInteger(selectedRank)) return null;
  const selected = estimate.candidates.find(
    (candidate) => Number.isInteger(candidate?.candidate_rank) &&
      candidate.candidate_rank === selectedRank);
  const secondary = estimate.candidates.find(
    (candidate) => Number.isInteger(candidate?.candidate_rank) &&
      candidate.candidate_rank !== selectedRank);
  const selectedPose = candidatePose(estimate, selected, 'selected');
  if (!selectedPose) return null;
  return {
    selected: selectedPose,
    secondary: candidatePose(estimate, secondary, 'secondary'),
  };
}

function renderableFinalPose(pose, cameraCalibration, gateModel) {
  return pose?.accepted === true &&
    pose.camera_calibration_id === cameraCalibration.calibration_id &&
    pose.gate_model_id === gateModel.model_id &&
    finiteVector(pose.rotation_vector_model_to_camera, 3) &&
    finiteVector(pose.position_camera_m, 3);
}

function emptyResult(reason, runtimeResult = undefined) {
  return {
    schemaAvailable: false,
    renderSupported: false,
    runtime_result: runtimeResult,
    pnp_relative_pose_estimates: [],
    renderable_selected_pnp_candidates: [],
    renderable_secondary_pnp_candidates: [],
    camera_pose_estimates: [],
    renderable_camera_pose_estimates: [],
    reason,
  };
}

/**
 * Validate and expose the exact production GeometryFrameResult JSON fields.
 * Rejected PnPRelativePoseEstimate and CameraPoseEstimate records remain in
 * the readout but are never rendered.
 */
export function pnpSceneFromReview(reviewPayload) {
  const runtimeResult = reviewPayload?.runtime_result;
  if (!runtimeResult || typeof runtimeResult !== 'object') {
    return emptyResult('GeometryFrameResult.runtime_result is unavailable.');
  }
  const cameraCalibration = runtimeResult.camera_calibration;
  const gateModel = runtimeResult.gate_model;
  const pnpEstimates = runtimeResult.pnp_relative_pose_estimates;
  const finalPoses = runtimeResult.camera_pose_estimates;
  if (!validCameraCalibration(cameraCalibration) || !validGateModel(gateModel) ||
      !Array.isArray(pnpEstimates) || !Array.isArray(finalPoses)) {
    const result = emptyResult(
      'GeometryFrameResult PnP fields do not match the runtime schema.',
      runtimeResult);
    result.pnp_relative_pose_estimates = Array.isArray(pnpEstimates)
      ? pnpEstimates : [];
    result.camera_pose_estimates = Array.isArray(finalPoses) ? finalPoses : [];
    return result;
  }

  const selectedCandidates = [];
  const secondaryCandidates = [];
  let malformedAcceptedPnp = 0;
  pnpEstimates.forEach((estimate) => {
    if (estimate?.accepted !== true) return;
    const candidates = pnpCandidates(estimate, cameraCalibration, gateModel);
    if (!candidates) {
      malformedAcceptedPnp += 1;
      return;
    }
    selectedCandidates.push(candidates.selected);
    if (candidates.secondary) secondaryCandidates.push(candidates.secondary);
  });
  const renderableFinal = finalPoses.filter((pose) =>
    renderableFinalPose(pose, cameraCalibration, gateModel));
  const malformedAcceptedFinal = finalPoses.filter(
    (pose) => pose?.accepted === true).length - renderableFinal.length;
  const distortionUnsupported = cameraCalibration.distortion_coefficients
    .some((item) => Math.abs(Number(item)) > 1e-12);
  const malformedCount = malformedAcceptedPnp + malformedAcceptedFinal;
  const reason = distortionUnsupported
    ? 'Non-zero CameraCalibration.distortion_coefficients require a distortion-aware renderer.'
    : (malformedCount
      ? `${malformedCount} accepted PnP/final pose record(s) are not renderable.`
      : null);
  return {
    schemaAvailable: true,
    renderSupported: !distortionUnsupported && malformedCount === 0,
    runtime_result: runtimeResult,
    camera_calibration: cameraCalibration,
    gate_model: gateModel,
    pnp_relative_pose_estimates: pnpEstimates,
    renderable_selected_pnp_candidates: selectedCandidates,
    renderable_secondary_pnp_candidates: secondaryCandidates,
    camera_pose_estimates: finalPoses,
    renderable_camera_pose_estimates: renderableFinal,
    reason,
  };
}
