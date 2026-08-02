import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import {
  PNP_SCENE_COORDINATE_FRAME,
  PNP_SCENE_SCHEMA_READY,
  pnpSceneFromReview,
} from './pnp_scene_adapter.js?v=5';
import { pnpWorldReplayFromJson } from './pnp_world_adapter.js?v=2';
import {
  PNP_WORLD_DISPLAY_BACKGROUND,
  PnpWorldScene,
} from './pnp_world_scene.js?v=5';
import {
  preferredRunId,
  rememberFrame,
  rememberedFrameIndex,
} from './frame_navigation_state.js?v=1';

const runSelect = document.getElementById('run-select');
const frameSlider = document.getElementById('frame-slider');
const frameMeta = document.getElementById('frame-meta');
const previousButton = document.getElementById('previous-frame');
const nextButton = document.getElementById('next-frame');
const canvas = document.getElementById('pnp-scene-canvas');
const stage = canvas.closest('.pnp-scene-stage');
const status = document.getElementById('pnp-scene-status');
const metadata = document.getElementById('pnp-scene-metadata');
const rawPoseToggle = document.getElementById('raw-pnp-toggle');
const secondaryPoseToggle = document.getElementById('secondary-pnp-toggle');
const finalPoseToggle = document.getElementById('final-pose-toggle');
const historicRawPoseToggle = document.getElementById('historic-raw-pose-toggle');
const historicFinalPoseToggle = document.getElementById(
  'historic-final-pose-toggle');
const cameraProjectionModeButton = document.getElementById('camera-projection-mode');
const open3dModeButton = document.getElementById('open-3d-mode');
const fit3dEvidenceButton = document.getElementById('fit-3d-evidence');
const viewCoordinate = document.getElementById('pnp-view-coordinate');
const worldCameraPathToggle = document.getElementById('world-camera-path-toggle');
const worldFullPathToggle = document.getElementById('world-full-path-toggle');
const worldControls = document.getElementById('pnp-world-controls');
const worldFullPathControls = document.getElementById('pnp-world-full-path-controls');
const cameraReadout = document.getElementById('pnp-camera-readout');
const worldReadout = document.getElementById('pnp-world-readout');
const worldMetadata = document.getElementById('pnp-world-metadata');

const CAMERA_PROJECTION_MODE = 'camera-projection';
const WORLD_3D_MODE = 'world-3d';
const RAW_PNP_COLOR = 0x52e8ff;
const SECONDARY_PNP_COLOR = 0xffb347;
const FINAL_POSE_COLOR = 0xff4fd8;
const HISTORIC_RAW_PNP_COLOR = 0xffd84a;
const HISTORIC_FINAL_POSE_COLOR = 0x55f29a;
const BORDER_HALO_COLOR = 0x020305;
const RAW_PNP_STYLE = Object.freeze({
  color: RAW_PNP_COLOR,
  fillOpacity: 0.13,
  borderWidthPx: 8,
  haloWidthPx: 11,
  renderOrder: 20,
});
const SECONDARY_PNP_STYLE = Object.freeze({
  color: SECONDARY_PNP_COLOR,
  fillOpacity: 0.08,
  borderWidthPx: 3,
  haloWidthPx: 5,
  renderOrder: 25,
});
const FINAL_POSE_STYLE = Object.freeze({
  color: FINAL_POSE_COLOR,
  fillOpacity: 0.24,
  borderWidthPx: 4,
  haloWidthPx: 6,
  renderOrder: 30,
});
const HISTORIC_RAW_PNP_STYLE = Object.freeze({
  color: HISTORIC_RAW_PNP_COLOR,
  fillOpacity: 0,
  borderOpacity: 0.72,
  haloOpacity: 0.6,
  borderWidthPx: 4,
  haloWidthPx: 6,
  renderOrder: 8,
  showAxes: false,
});
const HISTORIC_FINAL_POSE_STYLE = Object.freeze({
  color: HISTORIC_FINAL_POSE_COLOR,
  fillOpacity: 0,
  borderOpacity: 0.78,
  haloOpacity: 0.65,
  borderWidthPx: 2.5,
  haloWidthPx: 4.5,
  renderOrder: 12,
  showAxes: false,
});
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);
const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.05, 1000);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);
const projectionReference = cameraSpaceReference();
const historyRoot = new THREE.Group();
const poseRoot = new THREE.Group();
scene.add(projectionReference, historyRoot, poseRoot);
const worldScene = new PnpWorldScene({ scene, renderer, canvas, stage });

const state = {
  runs: [],
  run: null,
  frameIndex: 0,
  loadVersion: 0,
  backgroundTexture: null,
  camera_calibration: null,
  sceneResult: null,
  poseStatus: '',
  historyRunId: null,
  historyResults: [],
  historyRequestVersion: 0,
  historyStatus: '',
  viewMode: CAMERA_PROJECTION_MODE,
  viewStatus: 'Camera projection',
  worldRunId: null,
  worldReplay: null,
  worldRequestVersion: 0,
  worldStatus: '',
  worldFrameStatus: '',
};

runSelect.addEventListener('change', () => void selectRun(runSelect.value));
frameSlider.addEventListener('input', () => void selectFrame(Number(frameSlider.value)));
previousButton.addEventListener('click', () => void selectFrame(state.frameIndex - 1));
nextButton.addEventListener('click', () => void selectFrame(state.frameIndex + 1));
rawPoseToggle.addEventListener('change', updatePoseLayers);
secondaryPoseToggle.addEventListener('change', updatePoseLayers);
finalPoseToggle.addEventListener('change', updatePoseLayers);
historicRawPoseToggle.addEventListener(
  'change', () => void setHistoryVisibility());
historicFinalPoseToggle.addEventListener(
  'change', () => void setHistoryVisibility());
worldCameraPathToggle.addEventListener('change', updateWorldLayers);
worldFullPathToggle.addEventListener('change', updateWorldLayers);
cameraProjectionModeButton.addEventListener(
  'click', () => setViewMode(CAMERA_PROJECTION_MODE));
open3dModeButton.addEventListener('click', () => setViewMode(WORLD_3D_MODE));
fit3dEvidenceButton.addEventListener(
  'click', () => worldScene.fitSelectedEvidence(true));
new ResizeObserver(resizeScene).observe(stage);

function historyRequested() {
  return historicRawPoseToggle.checked || historicFinalPoseToggle.checked;
}

function worldHistoryStatus() {
  const fields = [];
  if (historicRawPoseToggle.checked) {
    fields.push('prior ui_projection.pnp_relative_pose_estimates');
  }
  if (historicFinalPoseToggle.checked) {
    fields.push('prior ui_projection.camera_pose_estimates');
  }
  return fields.join(' · ');
}

function cameraSpaceReference() {
  const group = new THREE.Group();
  const depthM = 10;
  const grid = new THREE.GridHelper(10, 10, 0x54a8ff, 0x22405c);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -depthM;
  grid.material.transparent = true;
  grid.material.opacity = 0.3;
  group.add(grid);

  const horizontal = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-5, 0, -depthM),
    new THREE.Vector3(5, 0, -depthM),
  ]);
  const vertical = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, -5, -depthM),
    new THREE.Vector3(0, 5, -depthM),
  ]);
  group.add(
    new THREE.Line(horizontal, new THREE.LineBasicMaterial({ color: 0xff5a67 })),
    new THREE.Line(vertical, new THREE.LineBasicMaterial({ color: 0x9aabba })),
  );
  return group;
}

async function init() {
  resizeScene();
  try {
    const response = await fetch('/api/runs', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Run catalog fetch failed: ${response.status}`);
    const payload = await response.json();
    state.runs = (Array.isArray(payload.runs) ? payload.runs : [])
      .filter((run) => run.has_logged_frames && run.geometry_frame_result_count > 0);
    runSelect.replaceChildren(...state.runs.map((run) => new Option(
      `${run.id} · ${run.geometry_frame_result_count} GeometryFrameResult · ` +
      `${run.accepted_pnp_relative_pose_estimate_count} ` +
      'PnPRelativePoseEstimate.accepted=true · ' +
      `${run.accepted_camera_pose_estimate_count} ` +
      'CameraPoseEstimate.accepted=true',
      run.id,
    )));
    if (!state.runs.length) {
      return showStatus('No GeometryFrameResult review JSON is available.');
    }
    const newest = state.runs.reduce((latest, run) =>
      run.id.localeCompare(latest.id) > 0 ? run : latest);
    await selectRun(preferredRunId(state.runs, newest.id));
  } catch (error) {
    showStatus(error.message || 'Unable to initialize CameraPoseEstimate projection.');
  }
}

async function selectRun(runId) {
  const summary = state.runs.find((run) => run.id === runId);
  if (!summary) return showStatus('Unknown source.run_id.');
  state.historyRequestVersion += 1;
  state.worldRequestVersion += 1;
  state.historyRunId = null;
  state.historyResults = [];
  state.historyStatus = '';
  state.worldRunId = null;
  state.worldReplay = null;
  state.worldStatus = '';
  state.worldFrameStatus = '';
  worldScene.clearReplay();
  clearPoseGroup(historyRoot);
  showStatus(`Loading source.run_id=${runId}…`);
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/frames`, { cache: 'no-store' });
  if (!response.ok) return showStatus(`Unable to load source.run_id=${runId}.`);
  state.run = { ...summary, ...await response.json() };
  runSelect.value = runId;
  const count = state.run.frames?.length || 0;
  frameSlider.max = String(Math.max(count - 1, 0));
  frameSlider.disabled = count === 0;
  if (!count) return showStatus('This run has no source.relative_path values.');
  await selectFrame(rememberedFrameIndex(state.run.id, state.run.frames));
  if (state.viewMode === WORLD_3D_MODE) void ensureWorldReplay();
}

async function selectFrame(index) {
  const frames = state.run?.frames || [];
  if (!frames.length) return;
  const loadVersion = ++state.loadVersion;
  state.frameIndex = Math.min(Math.max(index, 0), frames.length - 1);
  frameSlider.value = String(state.frameIndex);
  previousButton.disabled = state.frameIndex === 0;
  nextButton.disabled = state.frameIndex === frames.length - 1;
  const frame = frames[state.frameIndex];
  rememberFrame(state.run.id, state.frameIndex, frame);
  frameMeta.textContent =
    `UI frame ${state.frameIndex} / ${frames.length - 1} · ` +
    `source.relative_path=${frame.filename}`;
  showStatus('Loading source.relative_path…');
  await setBackground(frame.image_url, loadVersion);
  if (loadVersion !== state.loadVersion) return;

  if (!PNP_SCENE_SCHEMA_READY || !frame.geometry_result_url) {
    const result = pnpSceneFromReview(null);
    setPoses(result);
    updateReadout(frame, result);
    if (historyRequested() && state.viewMode === CAMERA_PROJECTION_MODE) {
      void ensureHistory();
    }
    return;
  }
  try {
    const response = await fetch(frame.geometry_result_url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`GeometryFrameResult fetch failed: ${response.status}`);
    }
    const review = await response.json();
    if (loadVersion !== state.loadVersion) return;
    const result = pnpSceneFromReview(review);
    setPoses(result);
    updateReadout(frame, result);
    if (historyRequested() && state.viewMode === CAMERA_PROJECTION_MODE) {
      void ensureHistory();
    }
  } catch (error) {
    const result = {
      schemaAvailable: false,
      renderSupported: false,
      pnp_relative_pose_estimates: [],
      renderable_selected_pnp_candidates: [],
      renderable_secondary_pnp_candidates: [],
      camera_pose_estimates: [],
      renderable_camera_pose_estimates: [],
      reason: error.message,
    };
    setPoses(result);
    updateReadout(frame, result);
    if (historyRequested() && state.viewMode === CAMERA_PROJECTION_MODE) {
      void ensureHistory();
    }
  }
}

function setBackground(url, loadVersion) {
  return new Promise((resolve) => {
    new THREE.TextureLoader().load(url, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      if (loadVersion !== state.loadVersion) {
        texture.dispose();
        return resolve();
      }
      state.backgroundTexture?.dispose();
      state.backgroundTexture = texture;
      if (state.viewMode === CAMERA_PROJECTION_MODE) scene.background = texture;
      renderScene();
      resolve();
    }, undefined, () => {
      state.backgroundTexture?.dispose();
      state.backgroundTexture = null;
      if (state.viewMode === CAMERA_PROJECTION_MODE) {
        scene.background = new THREE.Color(0x05070a);
      }
      renderScene();
      resolve();
    });
  });
}

function setPoses(result) {
  clearPoses();
  state.sceneResult = result;
  if (!result) {
    renderScene();
    return;
  }
  if (result.schemaAvailable && result.camera_calibration) {
    configureProjection(result.camera_calibration);
  }
  if (result.renderSupported) {
    if (rawPoseToggle.checked) {
      result.renderable_selected_pnp_candidates.forEach((pose) => {
        poseRoot.add(gatePlane(pose, result.gate_model, RAW_PNP_STYLE));
      });
    }
    if (secondaryPoseToggle.checked) {
      result.renderable_secondary_pnp_candidates.forEach((pose) => {
        poseRoot.add(gatePlane(pose, result.gate_model, SECONDARY_PNP_STYLE));
      });
    }
    if (finalPoseToggle.checked) {
      result.renderable_camera_pose_estimates.forEach((pose) => {
        poseRoot.add(gatePlane(pose, result.gate_model, FINAL_POSE_STYLE));
      });
    }
  }
  const total = result.pnp_relative_pose_estimates?.length || 0;
  const accepted = result.renderable_selected_pnp_candidates?.length || 0;
  const secondary = result.renderable_secondary_pnp_candidates?.length || 0;
  const final = result.renderable_camera_pose_estimates?.length || 0;
  state.poseStatus = result.schemaAvailable
    ? (result.renderSupported
      ? `PnPRelativePoseEstimate.accepted=true ${accepted} / ${total} · ` +
        `secondary candidate ${secondary} · ` +
        `GeometryFrameResult.camera_pose_estimates ${final}`
      : result.reason)
    : result.reason || 'GeometryFrameResult is unavailable.';
  updateHistoricPoseVisibility();
  selectWorldFrame();
  updateSceneStatus();
  renderScene();
}

function updatePoseLayers() {
  setPoses(state.sceneResult);
  updateWorldLayers();
}

function updateWorldLayers() {
  worldScene.setLayers({
    raw: rawPoseToggle.checked,
    secondary: secondaryPoseToggle.checked,
    final: finalPoseToggle.checked,
    historyRaw: historicRawPoseToggle.checked,
    historyFinal: historicFinalPoseToggle.checked,
    path: worldCameraPathToggle.checked,
    fullPath: worldFullPathToggle.checked,
  });
}

function setViewMode(mode) {
  if (mode !== CAMERA_PROJECTION_MODE && mode !== WORLD_3D_MODE) return;
  const isWorld = mode === WORLD_3D_MODE;
  state.viewMode = mode;
  state.viewStatus = isWorld
    ? 'World 3-D · LOCAL_NED'
    : 'Camera projection';
  projectionReference.visible = !isWorld;
  poseRoot.visible = !isWorld;
  historyRoot.visible = !isWorld && historyRequested();
  worldScene.setActive(isWorld);
  stage.classList.toggle('open-3d', isWorld);
  cameraProjectionModeButton.classList.toggle('active', !isWorld);
  open3dModeButton.classList.toggle('active', isWorld);
  cameraProjectionModeButton.setAttribute('aria-pressed', String(!isWorld));
  open3dModeButton.setAttribute('aria-pressed', String(isWorld));
  fit3dEvidenceButton.hidden = !isWorld;
  worldControls.hidden = !isWorld;
  worldFullPathControls.hidden = !isWorld;
  cameraReadout.hidden = isWorld;
  worldReadout.hidden = !isWorld;
  viewCoordinate.textContent = isWorld
    ? 'LOCAL_NED · north / up / east · orbit / pan / zoom'
    : 'OpenCV camera projection';
  scene.background = isWorld
    ? new THREE.Color(PNP_WORLD_DISPLAY_BACKGROUND)
    : state.backgroundTexture || new THREE.Color(0x05070a);
  updateStageAspect();
  updateWorldLayers();
  if (isWorld) {
    state.historyStatus = worldHistoryStatus();
    void ensureWorldReplay();
  } else {
    state.historyStatus = '';
    updateHistoricPoseVisibility();
    if (historyRequested()) void ensureHistory();
  }
  updateSceneStatus();
  resizeScene();
}

function updateStageAspect() {
  if (state.viewMode === WORLD_3D_MODE) {
    stage.style.aspectRatio = '16 / 9';
    return;
  }
  const shape = state.camera_calibration?.image_shape;
  if (Array.isArray(shape) && shape.length === 2) {
    stage.style.aspectRatio = `${Number(shape[1])} / ${Number(shape[0])}`;
  }
}

async function setHistoryVisibility() {
  updateWorldLayers();
  if (state.viewMode === WORLD_3D_MODE) {
    state.historyStatus = worldHistoryStatus();
    updateSceneStatus();
    return renderScene();
  }
  if (!historyRequested()) {
    state.historyStatus = '';
    historyRoot.visible = false;
    updateSceneStatus();
    return renderScene();
  }
  await ensureHistory();
}

async function ensureWorldReplay() {
  const runId = state.run?.id;
  if (!runId) return;
  if (state.worldRunId === runId && state.worldReplay) {
    selectWorldFrame();
    return;
  }
  const requestVersion = ++state.worldRequestVersion;
  state.worldStatus = 'loading pnp-world-replay ui_projection…';
  updateSceneStatus();
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/pnp-world-replay`,
      { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`PnP world replay fetch failed: ${response.status}`);
    }
    const replay = pnpWorldReplayFromJson(await response.json());
    if (requestVersion !== state.worldRequestVersion || state.run?.id !== runId) {
      return;
    }
    if (!replay.available) throw new Error(replay.reason);
    state.worldRunId = runId;
    state.worldReplay = replay;
    worldScene.setReplay(replay);
    updateWorldLayers();
    const usable = replay.cameraMount?.usable === true;
    state.worldStatus =
      `trajectory ${replay.trajectory.length} VehicleState samples · ` +
      `${replay.frames.length} exact GeometryFrameResult frames · ` +
      (usable
        ? 'logged camera mount matches production transform'
        : `world gates unavailable: ${replay.cameraMount?.unavailable_reason ||
          'camera_mount unusable'}`);
    selectWorldFrame();
  } catch (error) {
    if (requestVersion !== state.worldRequestVersion) return;
    state.worldRunId = null;
    state.worldReplay = null;
    worldScene.clearReplay();
    state.worldStatus = error.message || 'PnP world replay unavailable.';
    updateWorldReadout(null);
  }
  updateSceneStatus();
  renderScene();
}

function selectWorldFrame() {
  if (!state.worldReplay || !state.sceneResult) return;
  const runtime = state.sceneResult.runtime_result || {};
  const frameId = Number(runtime.frame_id);
  const simTimeNs = Number(runtime.sim_time_ns);
  if (!Number.isInteger(frameId) || !Number.isInteger(simTimeNs)) {
    state.worldFrameStatus = 'exact world frame identity unavailable';
    updateWorldReadout(null);
    return;
  }
  const selection = worldScene.selectFrame(frameId, simTimeNs);
  const projection = selection.frame?.uiProjection;
  const orientationStatuses = new Set([
    ...(projection?.pnp_relative_pose_estimates || []).map(
      (record) => record.selected_candidate).filter(Boolean),
    ...(projection?.camera_pose_estimates || []),
  ].map((record) => record.orientation_status));
  state.worldFrameStatus = selection.available
    ? `ui_projection selected raw ${selection.rawCount} · secondary ` +
      `${selection.secondaryCount} · final ${selection.finalCount} · ` +
      'orientation_status ' +
      `${[...orientationStatuses].join(', ') || 'none'}`
    : `ui_projection unavailable: ${selection.reason || 'unknown reason'}`;
  updateWorldReadout(selection);
  updateSceneStatus();
}

function updateWorldReadout(selection) {
  if (!worldMetadata) return;
  const replay = state.worldReplay;
  const frame = selection?.frame;
  const projection = frame?.uiProjection;
  const values = [
    ['version', replay?.source?.version ?? 'unavailable'],
    ['run_id', replay?.runId ?? state.run?.id ?? 'unavailable'],
    ['coordinate_system.world', replay?.coordinateSystem?.world ?? 'unavailable'],
    ['coordinate_system.three_display',
      replay?.coordinateSystem?.three_display ?? 'unavailable'],
    ['configuration.maximum_alignment_error_ms',
      replay?.configuration?.maximum_alignment_error_ms ?? 'unavailable'],
    ['camera_mount.source', replay?.cameraMount?.source ?? 'unavailable'],
    ['camera_mount.VIO_CAMERA_TILT_DEG',
      replay?.cameraMount?.VIO_CAMERA_TILT_DEG ?? 'unavailable'],
    ['camera_mount.VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M',
      formatValue(replay?.cameraMount?.VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M)],
    ['camera_mount.matches_production_transform',
      replay?.cameraMount?.matches_production_transform ?? 'unavailable'],
    ['trajectory.length', replay?.trajectory?.length ?? 0],
    ['frame.frame_id', frame?.frameId ?? 'unavailable'],
    ['frame.sim_time_ns', frame?.simTimeNs ?? 'unavailable'],
    ['sync.match_source', frame?.sync?.match_source ?? 'unavailable'],
    ['sync.alignment_error_ms', frame?.sync?.alignment_error_ms ?? 'unavailable'],
    ['sync.within_tolerance', frame?.sync?.within_tolerance ?? 'unavailable'],
    ['vehicle_state.position_local_ned_m',
      formatValue(frame?.vehicleState?.position_local_ned_m)],
    ['ui_projection.available', projection?.available ?? false],
    ['ui_projection.unavailable_reason',
      projection?.unavailable_reason ?? 'null'],
    ['ui_projection.camera_position_local_ned_m',
      formatValue(projection?.camera_position_local_ned_m)],
    ['ui_projection.pnp_relative_pose_estimates.length',
      projection?.pnp_relative_pose_estimates?.length ?? 0],
    ['ui_projection.camera_pose_estimates.length',
      projection?.camera_pose_estimates?.length ?? 0],
  ];
  appendWorldPnpReadout(
    values, projection?.pnp_relative_pose_estimates || []);
  appendWorldProjectionReadout(
    values, 'camera_pose_estimates', projection?.camera_pose_estimates || []);
  worldMetadata.replaceChildren(...values.flatMap(([term, description]) => {
    const key = document.createElement('dt');
    key.textContent = term;
    const value = document.createElement('dd');
    value.textContent = String(description ?? 'null');
    return [key, value];
  }));
}

function appendWorldPnpReadout(values, records) {
  records.forEach((record) => {
    const root = 'ui_projection.pnp_relative_pose_estimates' +
      `[pose_index=${record.pose_index}]`;
    values.push(
      [`${root}.component_id`, record.component_id],
      [`${root}.route`, record.route],
      [`${root}.selected_candidate.candidate_rank`,
        record.selected_candidate?.candidate_rank ?? 'null'],
      [`${root}.selected_candidate.position_local_ned_m`,
        formatValue(record.selected_candidate?.position_local_ned_m)],
      [`runtime_result.pnp_relative_pose_estimates` +
        `[pose_index=${record.pose_index}].candidates` +
        `[candidate_rank=${record.selected_candidate?.candidate_rank}]` +
        '.reprojection_rmse_px',
      record.selected_candidate?.sourcePose?.reprojection_rmse_px ?? 'null'],
      [`${root}.secondary_candidate.candidate_rank`,
        record.secondary_candidate?.candidate_rank ?? 'null'],
      [`${root}.secondary_candidate.position_local_ned_m`,
        formatValue(record.secondary_candidate?.position_local_ned_m)],
      [`runtime_result.pnp_relative_pose_estimates` +
        `[pose_index=${record.pose_index}].candidates` +
        `[candidate_rank=${record.secondary_candidate?.candidate_rank}]` +
        '.reprojection_rmse_px',
      record.secondary_candidate?.sourcePose?.reprojection_rmse_px ?? 'null'],
    );
  });
}

function appendWorldProjectionReadout(values, fieldName, records) {
  records.forEach((record) => {
    const projected = `ui_projection.${fieldName}` +
      `[pose_index=${record.pose_index}]`;
    const runtime = `runtime_result.${fieldName}` +
      `[pose_index=${record.pose_index}]`;
    values.push(
      [`${projected}.component_id`, record.component_id],
      [`${projected}.route`, record.route],
      [`${projected}.position_local_ned_m`,
        formatValue(record.position_local_ned_m)],
      [`${projected}.camera_depth_m`, record.camera_depth_m],
      [`${projected}.orientation_status`, record.orientation_status],
      [`${runtime}.reprojection_rmse_px`,
        record.sourcePose.reprojection_rmse_px],
      [`${runtime}.position_confidence`, record.sourcePose.position_confidence],
      [`${runtime}.orientation_confidence`,
        record.sourcePose.orientation_confidence],
    );
  });
}

async function ensureHistory() {
  const runId = state.run?.id;
  if (!historyRequested() || !runId) return;
  if (state.historyRunId === runId) {
    if (!historyRoot.children.length && state.historyResults.length) {
      buildHistoricPoseObjects();
    }
    updateHistoricPoseVisibility();
    updateSceneStatus();
    return renderScene();
  }
  const requestVersion = ++state.historyRequestVersion;
  state.historyStatus = 'loading prior GeometryFrameResult pose fields…';
  updateSceneStatus();
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/geometry-history`,
      { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`GeometryFrameResult history fetch failed: ${response.status}`);
    }
    const payload = await response.json();
    if (requestVersion !== state.historyRequestVersion || state.run?.id !== runId) {
      return;
    }
    state.historyRunId = runId;
    state.historyResults = (Array.isArray(payload.frames) ? payload.frames : [])
      .map((review) => pnpSceneFromReview(review));
    buildHistoricPoseObjects();
    updateHistoricPoseVisibility();
  } catch (error) {
    if (requestVersion !== state.historyRequestVersion) return;
    state.historyResults = [];
    state.historyStatus = error.message || 'GeometryFrameResult history unavailable.';
  }
  updateSceneStatus();
  renderScene();
}

function buildHistoricPoseObjects() {
  clearPoseGroup(historyRoot);
  state.historyResults.forEach((result) => {
    const runtime = result.runtime_result;
    const frameId = Number(runtime?.frame_id);
    const timeNs = Number(runtime?.sim_time_ns);
    if (!Number.isFinite(frameId) || !Number.isFinite(timeNs) ||
        !result.renderSupported) return;
    result.renderable_camera_pose_estimates.forEach((pose) => {
      const gate = gatePlane(pose, result.gate_model, HISTORIC_FINAL_POSE_STYLE);
      gate.userData.historyFrameId = frameId;
      gate.userData.historyTimeNs = timeNs;
      gate.userData.historyField = 'camera_pose_estimates';
      gate.userData.historyKind = 'final';
      historyRoot.add(gate);
    });
    result.renderable_selected_pnp_candidates.forEach((pose) => {
      const gate = gatePlane(pose, result.gate_model, HISTORIC_RAW_PNP_STYLE);
      gate.userData.historyFrameId = frameId;
      gate.userData.historyTimeNs = timeNs;
      gate.userData.historyField = 'pnp_relative_pose_estimates';
      gate.userData.historyKind = 'raw';
      historyRoot.add(gate);
    });
  });
}

function updateHistoricPoseVisibility() {
  if (state.viewMode !== CAMERA_PROJECTION_MODE ||
      !historyRequested() || state.historyRunId !== state.run?.id) {
    historyRoot.visible = false;
    return;
  }
  historyRoot.visible = true;
  const current = state.sceneResult?.runtime_result;
  const currentFrameId = Number(current?.frame_id);
  const currentTimeNs = Number(current?.sim_time_ns);
  if (!Number.isFinite(currentFrameId) || !Number.isFinite(currentTimeNs)) {
    historyRoot.children.forEach((gate) => { gate.visible = false; });
    state.historyStatus = 'prior GeometryFrameResult pose fields unavailable';
    return;
  }
  let rawCount = 0;
  let finalCount = 0;
  historyRoot.children.forEach((gate) => {
    const frameId = gate.userData.historyFrameId;
    const timeNs = gate.userData.historyTimeNs;
    const isPrior = timeNs < currentTimeNs || (
      timeNs === currentTimeNs && frameId < currentFrameId);
    const fieldEnabled = gate.userData.historyField === 'pnp_relative_pose_estimates'
      ? historicRawPoseToggle.checked
      : historicFinalPoseToggle.checked;
    gate.visible = isPrior && fieldEnabled;
    if (!gate.visible) return;
    if (gate.userData.historyField === 'pnp_relative_pose_estimates') rawCount += 1;
    if (gate.userData.historyField === 'camera_pose_estimates') finalCount += 1;
  });
  const counts = [];
  if (historicRawPoseToggle.checked) {
    counts.push(
      `prior GeometryFrameResult.pnp_relative_pose_estimates ${rawCount}`);
  }
  if (historicFinalPoseToggle.checked) {
    counts.push(
      `prior GeometryFrameResult.camera_pose_estimates ${finalCount}`);
  }
  state.historyStatus = counts.join(' · ');
}

function updateSceneStatus() {
  showStatus([
    state.viewStatus, state.poseStatus, state.historyStatus,
    state.viewMode === WORLD_3D_MODE ? state.worldStatus : '',
    state.viewMode === WORLD_3D_MODE ? state.worldFrameStatus : '',
  ].filter(Boolean).join(' · '));
}

function configureProjection(cameraCalibration) {
  const [imageHeight, imageWidth] = cameraCalibration.image_shape.map(Number);
  const matrix = cameraCalibration.camera_matrix;
  const fx = Number(matrix[0][0]);
  const skew = Number(matrix[0][1]);
  const cx = Number(matrix[0][2]);
  const fy = Number(matrix[1][1]);
  const cy = Number(matrix[1][2]);
  const near = 0.05;
  const far = 1000;
  camera.near = near;
  camera.far = far;
  camera.projectionMatrix.set(
    2 * fx / imageWidth, -2 * skew / imageWidth, 1 - 2 * cx / imageWidth, 0,
    0, 2 * fy / imageHeight, 2 * cy / imageHeight - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  state.camera_calibration = cameraCalibration;
  updateStageAspect();
  resizeScene();
}

function gatePlane(pose, gateModel, style) {
  const points = gateModel.object_points_m.map(
    ([x, y, z]) => new THREE.Vector3(Number(x), Number(y), Number(z)));
  const group = new THREE.Group();
  if (style.fillOpacity > 0) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    const plane = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: style.color,
        transparent: true,
        opacity: style.fillOpacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    plane.renderOrder = style.renderOrder - 1;
    group.add(plane);
  }
  const halo = gateBorder(
    points, BORDER_HALO_COLOR, style.haloWidthPx, style.renderOrder,
    style.haloOpacity ?? 1);
  const border = gateBorder(
    points, style.color, style.borderWidthPx, style.renderOrder + 1,
    style.borderOpacity ?? 1);
  group.add(halo, border);
  if (style.showAxes !== false) {
    group.add(new THREE.AxesHelper(Number(gateModel.side_length_m) * 0.3));
  }

  const [rx, ry, rz] = pose.rotation_vector_model_to_camera.map(Number);
  const angle = Math.hypot(rx, ry, rz);
  const modelToCamera = new THREE.Quaternion();
  if (angle > 1e-12) {
    modelToCamera.setFromAxisAngle(new THREE.Vector3(rx, ry, rz).normalize(), angle);
  }
  const cvToThree = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0), Math.PI);
  group.quaternion.copy(cvToThree).multiply(modelToCamera);
  const [tx, ty, tz] = pose.position_camera_m.map(Number);
  group.position.set(tx, -ty, -tz);
  group.userData = { pose };
  return group;
}

function gateBorder(points, color, widthPx, renderOrder, opacity) {
  const closedPoints = [...points, points[0]];
  const geometry = new LineGeometry();
  geometry.setPositions(closedPoints.flatMap((point) => point.toArray()));
  const material = new LineMaterial({
    color,
    linewidth: widthPx,
    transparent: true,
    opacity,
    depthTest: false,
    depthWrite: false,
    worldUnits: false,
  });
  material.resolution.set(
    Math.max(1, stage.clientWidth), Math.max(1, stage.clientHeight));
  const border = new Line2(geometry, material);
  border.computeLineDistances();
  border.renderOrder = renderOrder;
  return border;
}

function clearPoses() {
  clearPoseGroup(poseRoot);
}

function clearPoseGroup(root) {
  while (root.children.length) {
    const child = root.children[0];
    root.remove(child);
    child.traverse((value) => {
      value.geometry?.dispose();
      if (Array.isArray(value.material)) value.material.forEach((item) => item.dispose());
      else value.material?.dispose();
    });
  }
}

function updateReadout(frame, result) {
  const runtime = result.runtime_result || {};
  const calibration = result.camera_calibration;
  const model = result.gate_model;
  const pnpEstimates = result.pnp_relative_pose_estimates || [];
  const finalPoses = result.camera_pose_estimates || [];
  const values = [
    ['source.run_id', state.run.id],
    ['source.relative_path', frame.filename],
    ['GeometryFrameResult.frame_id', runtime.frame_id ?? 'unavailable'],
    ['GeometryFrameResult.sim_time_ns', runtime.sim_time_ns ?? 'unavailable'],
    ['GeometryFrameResult.processed_routes', formatValue(runtime.processed_routes)],
    ['camera optical coordinate frame', PNP_SCENE_COORDINATE_FRAME],
    ['CameraCalibration.calibration_id', calibration?.calibration_id ?? 'unavailable'],
    ['CameraCalibration.image_shape', formatValue(calibration?.image_shape)],
    ['CameraCalibration.camera_matrix', formatValue(calibration?.camera_matrix)],
    ['CameraCalibration.distortion_coefficients',
      formatValue(calibration?.distortion_coefficients)],
    ['PlanarGateModel.model_id', model?.model_id ?? 'unavailable'],
    ['PlanarGateModel.side_length_m', model?.side_length_m ?? 'unavailable'],
    ['PlanarGateModel.corner_order', formatValue(model?.corner_order)],
    ['PlanarGateModel.object_points_m', formatValue(model?.object_points_m)],
    ['GeometryFrameResult.pnp_relative_pose_estimates.length',
      pnpEstimates.length],
    ['GeometryFrameResult.camera_pose_estimates.length', finalPoses.length],
  ];
  appendPnpReadout(values, pnpEstimates);
  appendPoseReadout(values, 'camera_pose_estimates', finalPoses);
  metadata.replaceChildren(...values.flatMap(([term, description]) => {
    const key = document.createElement('dt');
    key.textContent = term;
    const value = document.createElement('dd');
    value.textContent = String(description ?? 'null');
    return [key, value];
  }));
}

function appendPnpReadout(values, estimates) {
  estimates.forEach((estimate) => {
    const root = 'GeometryFrameResult.pnp_relative_pose_estimates' +
      `[component_id=${estimate.component_id};gate_index=${estimate.gate_index}]`;
    values.push(
      [`${root}.route`, estimate.route],
      [`${root}.solver`, estimate.solver],
      [`${root}.selected_candidate_rank`, estimate.selected_candidate_rank],
      [`${root}.candidate_count`, estimate.candidate_count],
      [`${root}.ambiguity_gap_px`, estimate.ambiguity_gap_px],
      [`${root}.position_confidence`, estimate.position_confidence],
      [`${root}.orientation_confidence`, estimate.orientation_confidence],
      [`${root}.accepted`, estimate.accepted],
      [`${root}.rejection_reason`, estimate.rejection_reason],
    );
    (Array.isArray(estimate.candidates) ? estimate.candidates : [])
      .forEach((candidate) => {
        const candidateRoot = `${root}.candidates` +
          `[candidate_rank=${candidate.candidate_rank}]`;
        values.push(
          [`${candidateRoot}.rotation_vector_model_to_camera`,
            formatValue(candidate.rotation_vector_model_to_camera)],
          [`${candidateRoot}.position_camera_m`,
            formatValue(candidate.position_camera_m)],
          [`${candidateRoot}.reprojection_rmse_px`,
            candidate.reprojection_rmse_px],
        );
      });
  });
}

function appendPoseReadout(values, fieldName, poses) {
  poses.forEach((pose) => {
    const root = `GeometryFrameResult.${fieldName}` +
      `[component_id=${pose.component_id}]`;
    values.push(
      [`${root}.route`, pose.route],
      [`${root}.solver`, pose.solver],
      [`${root}.rotation_vector_model_to_camera`,
        formatValue(pose.rotation_vector_model_to_camera)],
      [`${root}.position_camera_m`, formatValue(pose.position_camera_m)],
      [`${root}.reprojection_rmse_px`, pose.reprojection_rmse_px],
      [`${root}.position_confidence`, pose.position_confidence],
      [`${root}.orientation_confidence`, pose.orientation_confidence],
      [`${root}.regression_evidence`, formatValue(pose.regression_evidence)],
      [`${root}.accepted`, pose.accepted],
      [`${root}.rejection_reason`, pose.rejection_reason],
    );
  });
}

function formatValue(value) {
  return value === undefined ? 'unavailable' : JSON.stringify(value);
}

function showStatus(message) {
  status.textContent = message || '';
  status.hidden = false;
}

function resizeScene() {
  const width = Math.max(1, stage.clientWidth);
  const height = Math.max(1, stage.clientHeight);
  renderer.setSize(width, height, false);
  [historyRoot, poseRoot].forEach((root) => root.traverse((value) => {
    if (value.material?.isLineMaterial) value.material.resolution.set(width, height);
  }));
  worldScene.resize(width, height);
  renderScene();
}

function renderScene() {
  if (state.viewMode === WORLD_3D_MODE) worldScene.render();
  else renderer.render(scene, camera);
}

init();
