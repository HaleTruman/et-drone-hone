import * as THREE from 'three';
import {
  PNP_SCENE_COORDINATE_FRAME,
  PNP_SCENE_SCHEMA_READY,
  pnpSceneFromReview,
} from './pnp_scene_adapter.js?v=3';
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

const POSE_COLORS = [0xffd84a, 0x52e8ff, 0xff66d4, 0x78ef72];
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);
const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.05, 1000);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);
const poseRoot = new THREE.Group();
scene.add(cameraSpaceReference(), poseRoot);

const state = {
  runs: [],
  run: null,
  frameIndex: 0,
  loadVersion: 0,
  backgroundTexture: null,
  camera_calibration: null,
};

runSelect.addEventListener('change', () => void selectRun(runSelect.value));
frameSlider.addEventListener('input', () => void selectFrame(Number(frameSlider.value)));
previousButton.addEventListener('click', () => void selectFrame(state.frameIndex - 1));
nextButton.addEventListener('click', () => void selectFrame(state.frameIndex + 1));
new ResizeObserver(resizeScene).observe(stage);

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
    new THREE.Line(vertical, new THREE.LineBasicMaterial({ color: 0x55f29a })),
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
      `${run.accepted_camera_pose_estimate_count} CameraPoseEstimate.accepted=true`,
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
    return updateReadout(frame, result);
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
  } catch (error) {
    const result = {
      schemaAvailable: false,
      renderSupported: false,
      camera_pose_estimates: [],
      renderable_camera_pose_estimates: [],
      reason: error.message,
    };
    setPoses(result);
    updateReadout(frame, result);
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
      scene.background = texture;
      renderScene();
      resolve();
    }, undefined, () => {
      state.backgroundTexture?.dispose();
      state.backgroundTexture = null;
      scene.background = new THREE.Color(0x05070a);
      renderScene();
      resolve();
    });
  });
}

function setPoses(result) {
  clearPoses();
  if (result.schemaAvailable && result.camera_calibration) {
    configureProjection(result.camera_calibration);
  }
  if (result.renderSupported) {
    result.renderable_camera_pose_estimates.forEach((pose, index) => {
      poseRoot.add(gatePlane(pose, result.gate_model, index));
    });
  }
  const total = result.camera_pose_estimates?.length || 0;
  const accepted = result.renderable_camera_pose_estimates?.length || 0;
  showStatus(result.schemaAvailable
    ? (result.renderSupported
      ? `CameraPoseEstimate.accepted=true ${accepted} / ${total}`
      : result.reason)
    : result.reason || 'GeometryFrameResult is unavailable.');
  renderScene();
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
  stage.style.aspectRatio = `${imageWidth} / ${imageHeight}`;
  state.camera_calibration = cameraCalibration;
  resizeScene();
}

function gatePlane(pose, gateModel, index) {
  const color = POSE_COLORS[index % POSE_COLORS.length];
  const points = gateModel.object_points_m.map(
    ([x, y, z]) => new THREE.Vector3(Number(x), Number(y), Number(z)));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const plane = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  const border = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color }),
  );
  const group = new THREE.Group();
  group.add(plane, border, new THREE.AxesHelper(Number(gateModel.side_length_m) * 0.3));

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
  group.userData = pose;
  return group;
}

function clearPoses() {
  while (poseRoot.children.length) {
    const child = poseRoot.children[0];
    poseRoot.remove(child);
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
  const poses = result.camera_pose_estimates || [];
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
    ['GeometryFrameResult.camera_pose_estimates.length', poses.length],
  ];
  poses.forEach((pose) => {
    const root = `CameraPoseEstimate[component_id=${pose.component_id}]`;
    values.push(
      [`${root}.route`, pose.route],
      [`${root}.solver`, pose.solver],
      [`${root}.rotation_vector_model_to_camera`,
        formatValue(pose.rotation_vector_model_to_camera)],
      [`${root}.position_camera_m`, formatValue(pose.position_camera_m)],
      [`${root}.candidate_count`, pose.candidate_count],
      [`${root}.reprojection_rmse_px`, pose.reprojection_rmse_px],
      [`${root}.secondary_reprojection_rmse_px`,
        pose.secondary_reprojection_rmse_px],
      [`${root}.ambiguity_gap_px`, pose.ambiguity_gap_px],
      [`${root}.position_confidence`, pose.position_confidence],
      [`${root}.orientation_confidence`, pose.orientation_confidence],
      [`${root}.accepted`, pose.accepted],
      [`${root}.rejection_reason`, pose.rejection_reason],
    );
  });
  metadata.replaceChildren(...values.flatMap(([term, description]) => {
    const key = document.createElement('dt');
    key.textContent = term;
    const value = document.createElement('dd');
    value.textContent = String(description ?? 'null');
    return [key, value];
  }));
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
  renderScene();
}

function renderScene() {
  renderer.render(scene, camera);
}

init();
