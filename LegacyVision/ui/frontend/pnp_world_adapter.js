export const PNP_WORLD_REPLAY_VERSION =
  'deterministic-v3.pnp-world-replay.v2';

export function pnpWorldFrameKey(frameId, simTimeNs) {
  return `${Number(frameId)}:${Number(simTimeNs)}`;
}

function finiteVector(value, length) {
  if (!Array.isArray(value) || value.length !== length) return null;
  const result = value.map(Number);
  return result.every(Number.isFinite) ? result : null;
}

function finiteMatrix3(value) {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const rows = value.map((row) => finiteVector(row, 3));
  return rows.every(Boolean) ? rows : null;
}

function projectedTransform(value, sourceTransform) {
  if (!value || typeof value !== 'object' || !sourceTransform) return null;
  const position = finiteVector(value.position_local_ned_m, 3);
  const corners = value.corners_local_ned_m === null
    ? null
    : (Array.isArray(value.corners_local_ned_m)
      ? value.corners_local_ned_m.map((point) => finiteVector(point, 3))
      : null);
  if (!position || (corners &&
      (corners.length !== 4 || !corners.every(Boolean)))) return null;
  return {
    ...value,
    position_local_ned_m: position,
    corners_local_ned_m: corners,
    sourcePose: sourceTransform,
  };
}

function finalProjectionRecord(value, runtimeResult) {
  if (!value || typeof value !== 'object' ||
      value.source_field !== 'GeometryFrameResult.camera_pose_estimates') {
    return null;
  }
  const poseIndex = Number(value.pose_index);
  const poses = runtimeResult.camera_pose_estimates;
  const pose = Number.isInteger(poseIndex) && Array.isArray(poses)
    ? poses[poseIndex] : null;
  if (!pose || pose.accepted !== true ||
      Number(pose.component_id) !== Number(value.component_id) ||
      Number(pose.gate_index) !== Number(value.gate_index) ||
      String(pose.route) !== String(value.route)) return null;
  const projected = projectedTransform(value, pose);
  return projected ? { ...projected, fieldName: 'camera_pose_estimates' } : null;
}

function pnpProjectionRecord(value, runtimeResult) {
  if (!value || typeof value !== 'object' ||
      value.source_field !== 'GeometryFrameResult.pnp_relative_pose_estimates') {
    return null;
  }
  const poseIndex = Number(value.pose_index);
  const estimates = runtimeResult.pnp_relative_pose_estimates;
  const estimate = Number.isInteger(poseIndex) && Array.isArray(estimates)
    ? estimates[poseIndex] : null;
  if (!estimate || estimate.accepted !== true ||
      Number(estimate.component_id) !== Number(value.component_id) ||
      Number(estimate.gate_index) !== Number(value.gate_index) ||
      String(estimate.route) !== String(value.route) ||
      !Array.isArray(estimate.candidates)) return null;
  const selectedRank = estimate.selected_candidate_rank;
  if (!Number.isInteger(selectedRank)) return null;
  const selected = estimate.candidates.find(
    (candidate) => Number.isInteger(candidate?.candidate_rank) &&
      candidate.candidate_rank === selectedRank);
  const secondary = estimate.candidates.find(
    (candidate) => Number.isInteger(candidate?.candidate_rank) &&
      candidate.candidate_rank !== selectedRank);
  const selectedProjection = projectedTransform(value.selected_candidate, selected);
  if (!selectedProjection ||
      Number(value.selected_candidate?.candidate_rank) !== selectedRank) return null;
  const secondaryProjection = value.secondary_candidate === null
    ? null : projectedTransform(value.secondary_candidate, secondary);
  if (value.secondary_candidate !== null && (!secondaryProjection ||
      Number(value.secondary_candidate?.candidate_rank) !==
      Number(secondary?.candidate_rank))) return null;
  return {
    ...value,
    fieldName: 'pnp_relative_pose_estimates',
    sourceEstimate: estimate,
    selected_candidate: selectedProjection,
    secondary_candidate: secondaryProjection,
  };
}

function replayFrame(value) {
  const frame = value?.frame;
  const runtime = value?.runtime_result;
  const projection = value?.ui_projection;
  const frameId = Number(frame?.frame_id);
  const simTimeNs = Number(frame?.sim_time_ns);
  if (!Number.isInteger(frameId) || !Number.isInteger(simTimeNs) ||
      Number(runtime?.frame_id) !== frameId ||
      Number(runtime?.sim_time_ns) !== simTimeNs || !projection) return null;
  const vehicle = value.vehicle_state;
  const vehiclePosition = vehicle
    ? finiteVector(vehicle.position_local_ned_m, 3) : null;
  const vehicleAttitude = vehicle
    ? finiteVector(vehicle.attitude_quaternion, 4) : null;
  const available = projection.available === true;
  const cameraPosition = finiteVector(
    projection.camera_position_local_ned_m, 3);
  const bodyRotation = finiteMatrix3(
    projection.rotation_local_ned_from_body_frd);
  const cameraRotation = finiteMatrix3(
    projection.rotation_local_ned_from_camera_cv);
  if (available && (!vehiclePosition || !vehicleAttitude || !cameraPosition ||
      !bodyRotation || !cameraRotation)) return null;
  const raw = (Array.isArray(projection.pnp_relative_pose_estimates)
    ? projection.pnp_relative_pose_estimates : [])
    .map((item) => pnpProjectionRecord(item, runtime)).filter(Boolean);
  const final = (Array.isArray(projection.camera_pose_estimates)
    ? projection.camera_pose_estimates : [])
    .map((item) => finalProjectionRecord(item, runtime)).filter(Boolean);
  return {
    source: value,
    frameId,
    simTimeNs,
    sync: value.sync,
    vehicleState: vehicle ? {
      ...vehicle,
      position_local_ned_m: vehiclePosition,
      attitude_quaternion: vehicleAttitude,
    } : null,
    uiProjection: {
      ...projection,
      camera_position_local_ned_m: cameraPosition,
      rotation_local_ned_from_body_frd: bodyRotation,
      rotation_local_ned_from_camera_cv: cameraRotation,
      pnp_relative_pose_estimates: raw,
      camera_pose_estimates: final,
    },
  };
}

export function pnpWorldReplayFromJson(payload) {
  if (payload?.version !== PNP_WORLD_REPLAY_VERSION ||
      payload?.coordinate_system?.world !== 'local_ned' ||
      payload?.coordinate_system?.three_display !== 'north_up_east') {
    return { available: false, reason: 'PnP world replay contract unavailable.' };
  }
  const trajectory = (Array.isArray(payload.trajectory) ? payload.trajectory : [])
    .map((sample) => {
      const position = finiteVector(sample?.position_local_ned_m, 3);
      const cameraPosition = finiteVector(
        sample?.ui_projection?.camera_position_local_ned_m, 3);
      const cameraAvailable = sample?.ui_projection?.available === true;
      const attitude = finiteVector(sample?.attitude_quaternion, 4);
      const sampleIndex = Number(sample?.sample_index);
      return position && attitude && Number.isInteger(sampleIndex) &&
          (!cameraAvailable || cameraPosition)
        ? { ...sample, position_local_ned_m: position,
          attitude_quaternion: attitude, sample_index: sampleIndex,
          ui_projection: {
            ...sample.ui_projection,
            camera_position_local_ned_m: cameraPosition,
          } }
        : null;
    }).filter(Boolean);
  const frames = (Array.isArray(payload.frames) ? payload.frames : [])
    .map(replayFrame).filter(Boolean);
  const framesByKey = new Map(frames.map((frame) => [
    pnpWorldFrameKey(frame.frameId, frame.simTimeNs), frame,
  ]));
  return {
    available: true,
    source: payload,
    runId: String(payload.run_id),
    configuration: payload.configuration,
    coordinateSystem: payload.coordinate_system,
    cameraMount: payload.camera_mount,
    trajectory,
    frames,
    framesByKey,
  };
}
