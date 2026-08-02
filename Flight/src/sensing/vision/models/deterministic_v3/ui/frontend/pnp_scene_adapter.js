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

function renderablePose(pose, cameraCalibration, gateModel) {
  return pose?.accepted === true &&
    pose.camera_calibration_id === cameraCalibration.calibration_id &&
    pose.gate_model_id === gateModel.model_id &&
    finiteVector(pose.rotation_vector_model_to_camera, 3) &&
    finiteVector(pose.position_camera_m, 3);
}

/**
 * Validate and expose the exact production GeometryFrameResult JSON fields.
 * Rejected CameraPoseEstimate records remain present but are never rendered.
 */
export function pnpSceneFromReview(reviewPayload) {
  const runtimeResult = reviewPayload?.runtime_result;
  if (!runtimeResult || typeof runtimeResult !== 'object') {
    return {
      schemaAvailable: false,
      renderSupported: false,
      camera_pose_estimates: [],
      renderable_camera_pose_estimates: [],
      reason: 'GeometryFrameResult.runtime_result is unavailable.',
    };
  }
  const cameraCalibration = runtimeResult.camera_calibration;
  const gateModel = runtimeResult.gate_model;
  const poses = runtimeResult.camera_pose_estimates;
  if (!validCameraCalibration(cameraCalibration) ||
      !validGateModel(gateModel) || !Array.isArray(poses)) {
    return {
      schemaAvailable: false,
      renderSupported: false,
      runtime_result: runtimeResult,
      camera_pose_estimates: Array.isArray(poses) ? poses : [],
      renderable_camera_pose_estimates: [],
      reason: 'GeometryFrameResult PnP fields do not match the runtime schema.',
    };
  }
  const distortionUnsupported = cameraCalibration.distortion_coefficients
    .some((item) => Math.abs(Number(item)) > 1e-12);
  const renderable = poses.filter((pose) =>
    renderablePose(pose, cameraCalibration, gateModel));
  const malformedAcceptedCount = poses.filter((pose) => pose?.accepted === true).length -
    renderable.length;
  const reason = distortionUnsupported
    ? 'Non-zero CameraCalibration.distortion_coefficients require a distortion-aware renderer.'
    : (malformedAcceptedCount
      ? `${malformedAcceptedCount} accepted CameraPoseEstimate record(s) are not renderable.`
      : null);
  return {
    schemaAvailable: true,
    renderSupported: !distortionUnsupported && malformedAcceptedCount === 0,
    runtime_result: runtimeResult,
    camera_calibration: cameraCalibration,
    gate_model: gateModel,
    camera_pose_estimates: poses,
    renderable_camera_pose_estimates: renderable,
    reason,
  };
}
