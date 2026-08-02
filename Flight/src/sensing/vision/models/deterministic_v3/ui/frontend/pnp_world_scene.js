import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import {
  PNP_WORLD_COLORS,
  PNP_WORLD_SCENE_CONFIGURATION as CONFIG,
} from './pnp_world_config.js?v=4';
import { pnpWorldFrameKey } from './pnp_world_adapter.js?v=2';

const DARK_BACKGROUND = 0x05070a;

/** Map serialized LOCAL_NED [north, east, down] into Three [x, y, z]. */
function displayPoint([north, east, down]) {
  return new THREE.Vector3(Number(north), -Number(down), Number(east));
}

function trajectoryPoint(sample) {
  return displayPoint(
    sample.ui_projection?.camera_position_local_ned_m ||
    sample.position_local_ned_m);
}

function addNedVector(origin, rotation, column, length) {
  return [0, 1, 2].map((row) =>
    Number(origin[row]) + Number(rotation[row][column]) * length);
}

function disposeTree(root) {
  while (root.children.length) {
    const child = root.children[0];
    root.remove(child);
    child.traverse((value) => {
      value.geometry?.dispose();
      if (Array.isArray(value.material)) {
        value.material.forEach((material) => material.dispose());
      } else {
        value.material?.dispose();
      }
    });
  }
}

function basicLine(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    depthWrite: false,
  });
  return new THREE.Line(geometry, material);
}

function screenLine(points, color, widthPx, opacity, renderOrder, stage) {
  const geometry = new LineGeometry();
  geometry.setPositions(points.flatMap((point) => point.toArray()));
  const material = new LineMaterial({
    color,
    linewidth: widthPx,
    transparent: opacity < 1,
    opacity,
    depthTest: false,
    depthWrite: false,
    worldUnits: false,
  });
  material.resolution.set(
    Math.max(1, stage.clientWidth), Math.max(1, stage.clientHeight));
  const line = new Line2(geometry, material);
  line.computeLineDistances();
  line.renderOrder = renderOrder;
  return line;
}

function gateOutline(record, style, stage) {
  const root = new THREE.Group();
  const corners = record.corners_local_ned_m?.map(displayPoint) || [];
  if (corners.length !== 4) {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 12, 8),
      new THREE.MeshBasicMaterial({ color: style.color }),
    );
    marker.position.copy(displayPoint(record.position_local_ned_m));
    root.add(marker);
    return root;
  }
  if (style.fillOpacity > 0) {
    const fillGeometry = new THREE.BufferGeometry().setFromPoints(corners);
    fillGeometry.setIndex([0, 1, 2, 0, 2, 3]);
    const fill = new THREE.Mesh(fillGeometry, new THREE.MeshBasicMaterial({
      color: style.color,
      transparent: true,
      opacity: style.fillOpacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    }));
    fill.renderOrder = style.renderOrder - 1;
    root.add(fill);
  }
  const closed = [...corners, corners[0]];
  root.add(
    screenLine(
      closed, PNP_WORLD_COLORS.depthHalo, style.haloWidthPx,
      style.haloOpacity, style.renderOrder, stage),
    screenLine(
      closed, style.color, style.borderWidthPx,
      style.borderOpacity, style.renderOrder + 1, stage),
  );
  root.userData.projection = record;
  return root;
}

function axisArrow(origin, endpoint, color) {
  const start = displayPoint(origin);
  const end = displayPoint(endpoint);
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length <= 1e-9) return new THREE.Group();
  return new THREE.ArrowHelper(
    delta.normalize(), start, length, color,
    Math.min(0.2, length * 0.22), Math.min(0.1, length * 0.12));
}

function orientedAxes(origin, rotation, length, colors) {
  const root = new THREE.Group();
  colors.forEach((color, column) => root.add(axisArrow(
    origin, addNedVector(origin, rotation, column, length), color)));
  return root;
}

function worldReference() {
  const root = new THREE.Group();
  const grid = new THREE.GridHelper(
    CONFIG.gridSizeM, CONFIG.gridDivisions, 0x46545e, 0x202a31);
  grid.material.transparent = true;
  grid.material.opacity = 0.48;
  root.add(grid);
  const origin = [0, 0, 0];
  root.add(
    axisArrow(origin, [2, 0, 0], 0xff5a67),
    axisArrow(origin, [0, 2, 0], 0x55f29a),
    axisArrow(origin, [0, 0, 2], 0x69a7ff),
  );
  return root;
}

const CURRENT_RAW_STYLE = Object.freeze({
  color: PNP_WORLD_COLORS.rawCurrent,
  fillOpacity: 0.12,
  borderOpacity: 1,
  haloOpacity: 1,
  borderWidthPx: CONFIG.currentRawBorderPx,
  haloWidthPx: CONFIG.currentRawBorderPx + 3,
  renderOrder: 40,
});
const CURRENT_SECONDARY_STYLE = Object.freeze({
  color: PNP_WORLD_COLORS.secondaryCurrent,
  fillOpacity: 0.06,
  borderOpacity: 0.9,
  haloOpacity: 0.85,
  borderWidthPx: CONFIG.currentSecondaryBorderPx,
  haloWidthPx: CONFIG.currentSecondaryBorderPx + 2,
  renderOrder: 45,
});
const CURRENT_FINAL_STYLE = Object.freeze({
  color: PNP_WORLD_COLORS.finalCurrent,
  fillOpacity: 0.22,
  borderOpacity: 1,
  haloOpacity: 1,
  borderWidthPx: CONFIG.currentFinalBorderPx,
  haloWidthPx: CONFIG.currentFinalBorderPx + 2,
  renderOrder: 50,
});
const HISTORY_RAW_STYLE = Object.freeze({
  color: PNP_WORLD_COLORS.rawHistory,
  fillOpacity: 0,
  borderOpacity: 0.58,
  haloOpacity: 0.42,
  borderWidthPx: CONFIG.historyRawBorderPx,
  haloWidthPx: CONFIG.historyRawBorderPx + 2,
  renderOrder: 15,
});
const HISTORY_FINAL_STYLE = Object.freeze({
  color: PNP_WORLD_COLORS.finalHistory,
  fillOpacity: 0,
  borderOpacity: 0.68,
  haloOpacity: 0.48,
  borderWidthPx: CONFIG.historyFinalBorderPx,
  haloWidthPx: CONFIG.historyFinalBorderPx + 2,
  renderOrder: 20,
});

export class PnpWorldScene {
  constructor({ scene, renderer, canvas, stage }) {
    this.scene = scene;
    this.renderer = renderer;
    this.canvas = canvas;
    this.stage = stage;
    this.camera = new THREE.PerspectiveCamera(
      CONFIG.cameraFovDeg, 16 / 9, CONFIG.cameraNearM, CONFIG.cameraFarM);
    this.camera.position.set(8, 5, 8);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enabled = false;
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = true;
    this.controls.target.set(0, 0, 0);

    this.root = new THREE.Group();
    this.root.visible = false;
    this.referenceRoot = worldReference();
    this.fullPathRoot = new THREE.Group();
    this.elapsedPathRoot = new THREE.Group();
    this.historyRawRoot = new THREE.Group();
    this.historyFinalRoot = new THREE.Group();
    this.currentRawRoot = new THREE.Group();
    this.currentSecondaryRoot = new THREE.Group();
    this.currentFinalRoot = new THREE.Group();
    this.currentReferenceRoot = new THREE.Group();
    this.depthRoot = new THREE.Group();
    this.root.add(
      this.referenceRoot,
      this.fullPathRoot,
      this.elapsedPathRoot,
      this.historyRawRoot,
      this.historyFinalRoot,
      this.depthRoot,
      this.currentRawRoot,
      this.currentSecondaryRoot,
      this.currentFinalRoot,
      this.currentReferenceRoot,
    );
    this.scene.add(this.root);
    this.replay = null;
    this.selectedFrame = null;
    this.active = false;
    this.animationFrame = null;
    this.fitRunId = null;
    this.layers = {
      raw: true,
      secondary: false,
      final: true,
      historyRaw: false,
      historyFinal: false,
      path: true,
      fullPath: true,
    };
  }

  clearReplay() {
    [
      this.fullPathRoot, this.elapsedPathRoot, this.historyRawRoot,
      this.historyFinalRoot, this.currentRawRoot, this.currentSecondaryRoot,
      this.currentFinalRoot, this.currentReferenceRoot, this.depthRoot,
    ].forEach(disposeTree);
    this.replay = null;
    this.selectedFrame = null;
    this.fitRunId = null;
  }

  setReplay(replay) {
    this.clearReplay();
    this.replay = replay?.available ? replay : null;
    if (!this.replay) return;
    const path = this.replay.trajectory.map(trajectoryPoint);
    if (path.length >= 2) {
      this.fullPathRoot.add(basicLine(
        path, PNP_WORLD_COLORS.fullTrajectory, 0.42));
    }
    this.replay.frames.forEach((frame) => {
      this.addHistoryFrame(
        frame, 'pnp_relative_pose_estimates',
        this.historyRawRoot, HISTORY_RAW_STYLE);
      this.addHistoryFrame(
        frame, 'camera_pose_estimates',
        this.historyFinalRoot, HISTORY_FINAL_STYLE);
    });
    this.updateLayerVisibility();
  }

  addHistoryFrame(frame, fieldName, root, style) {
    const frameRoot = new THREE.Group();
    frameRoot.userData.frameId = frame.frameId;
    frameRoot.userData.simTimeNs = frame.simTimeNs;
    frameRoot.userData.sampleIndex = Number(frame.vehicleState?.sample_index);
    frame.uiProjection[fieldName].forEach((record) => {
      const projection = fieldName === 'pnp_relative_pose_estimates'
        ? record.selected_candidate : record;
      if (projection) frameRoot.add(gateOutline(projection, style, this.stage));
    });
    if (frameRoot.children.length) root.add(frameRoot);
  }

  setLayers(layers) {
    this.layers = { ...this.layers, ...layers };
    this.updateLayerVisibility();
    if (this.selectedFrame) this.buildCurrent(this.selectedFrame);
    this.render();
  }

  setActive(active) {
    this.active = Boolean(active);
    this.root.visible = this.active;
    this.controls.enabled = this.active;
    if (this.active) this.startRenderLoop();
  }

  selectFrame(frameId, simTimeNs) {
    disposeTree(this.currentRawRoot);
    disposeTree(this.currentSecondaryRoot);
    disposeTree(this.currentFinalRoot);
    disposeTree(this.currentReferenceRoot);
    disposeTree(this.depthRoot);
    disposeTree(this.elapsedPathRoot);
    this.selectedFrame = this.replay?.framesByKey.get(
      pnpWorldFrameKey(frameId, simTimeNs)) || null;
    if (!this.selectedFrame) {
      this.updateHistoryVisibility();
      this.render();
      return {
        available: false,
        reason: 'exact GeometryFrameResult world replay frame unavailable',
      };
    }
    this.buildElapsedPath(this.selectedFrame);
    this.buildCurrent(this.selectedFrame);
    this.updateHistoryVisibility();
    this.updateLayerVisibility();
    if (this.fitRunId !== this.replay.runId) this.fitSelectedEvidence();
    this.render();
    const projection = this.selectedFrame.uiProjection;
    return {
      available: projection.available === true,
      reason: projection.unavailable_reason,
      frame: this.selectedFrame,
      rawCount: projection.pnp_relative_pose_estimates.length,
      secondaryCount: projection.pnp_relative_pose_estimates.filter(
        (record) => record.secondary_candidate).length,
      finalCount: projection.camera_pose_estimates.length,
    };
  }

  buildElapsedPath(frame) {
    const sampleIndex = Number(frame.vehicleState?.sample_index);
    if (!Number.isInteger(sampleIndex)) return;
    const points = this.replay.trajectory.slice(0, sampleIndex + 1).map(
      trajectoryPoint);
    if (points.length >= 2) {
      this.elapsedPathRoot.add(screenLine(
        points, PNP_WORLD_COLORS.elapsedTrajectory, 3, 0.9, 5, this.stage));
    }
  }

  buildCurrent(frame) {
    disposeTree(this.currentRawRoot);
    disposeTree(this.currentSecondaryRoot);
    disposeTree(this.currentFinalRoot);
    disposeTree(this.currentReferenceRoot);
    disposeTree(this.depthRoot);
    const projection = frame.uiProjection;
    if (!projection.available) return;
    const vehiclePosition = frame.vehicleState.position_local_ned_m;
    const cameraPosition = projection.camera_position_local_ned_m;
    const vehicleMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 10),
      new THREE.MeshBasicMaterial({ color: PNP_WORLD_COLORS.drone }));
    vehicleMarker.position.copy(displayPoint(vehiclePosition));
    const cameraMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 16, 10),
      new THREE.MeshBasicMaterial({ color: PNP_WORLD_COLORS.camera }));
    cameraMarker.position.copy(displayPoint(cameraPosition));
    this.currentReferenceRoot.add(
      vehicleMarker,
      cameraMarker,
      orientedAxes(
        vehiclePosition, projection.rotation_local_ned_from_body_frd,
        CONFIG.droneAxisLengthM, [0xff5a67, 0x55f29a, 0x69a7ff]),
      orientedAxes(
        cameraPosition, projection.rotation_local_ned_from_camera_cv,
        CONFIG.cameraAxisLengthM, [0xff5a67, 0x55f29a, 0x69a7ff]),
    );
    if (this.layers.raw) {
      projection.pnp_relative_pose_estimates.forEach((record) => {
        const selected = record.selected_candidate;
        this.currentRawRoot.add(gateOutline(selected, CURRENT_RAW_STYLE, this.stage));
        this.depthRoot.add(this.depthSegment(
          cameraPosition, selected, CURRENT_RAW_STYLE));
      });
    }
    if (this.layers.secondary) {
      projection.pnp_relative_pose_estimates.forEach((record) => {
        const secondary = record.secondary_candidate;
        if (!secondary) return;
        this.currentSecondaryRoot.add(
          gateOutline(secondary, CURRENT_SECONDARY_STYLE, this.stage));
        this.depthRoot.add(this.depthSegment(
          cameraPosition, secondary, CURRENT_SECONDARY_STYLE));
      });
    }
    if (this.layers.final) {
      projection.camera_pose_estimates.forEach((record) => {
        this.currentFinalRoot.add(
          gateOutline(record, CURRENT_FINAL_STYLE, this.stage));
        this.depthRoot.add(this.depthSegment(
          cameraPosition, record, CURRENT_FINAL_STYLE));
      });
    }
  }

  depthSegment(cameraPosition, record, style) {
    return screenLine(
      [displayPoint(cameraPosition), displayPoint(record.position_local_ned_m)],
      style.color, CONFIG.depthBorderPx, 0.62,
      style.renderOrder - 2, this.stage);
  }

  updateLayerVisibility() {
    this.fullPathRoot.visible = this.layers.fullPath;
    this.elapsedPathRoot.visible = this.layers.path;
    this.currentRawRoot.visible = this.layers.raw;
    this.currentSecondaryRoot.visible = this.layers.secondary;
    this.currentFinalRoot.visible = this.layers.final;
    this.historyRawRoot.visible = this.layers.historyRaw;
    this.historyFinalRoot.visible = this.layers.historyFinal;
    this.updateHistoryVisibility();
  }

  updateHistoryVisibility() {
    const selected = this.selectedFrame;
    [
      [this.historyRawRoot, this.layers.historyRaw],
      [this.historyFinalRoot, this.layers.historyFinal],
    ].forEach(([root, enabled]) => {
      root.children.forEach((frameRoot) => {
        frameRoot.visible = Boolean(enabled && selected && (
          frameRoot.userData.simTimeNs < selected.simTimeNs ||
          (frameRoot.userData.simTimeNs === selected.simTimeNs &&
            frameRoot.userData.frameId < selected.frameId)));
      });
    });
  }

  fitSelectedEvidence(force = false) {
    if (!this.replay || (!force && this.fitRunId === this.replay.runId)) return;
    this.scene.updateMatrixWorld(true);
    const box = new THREE.Box3();
    if (this.currentReferenceRoot.children.length) {
      box.expandByObject(this.currentReferenceRoot, true);
    }
    if (this.currentRawRoot.children.length) box.expandByObject(this.currentRawRoot, true);
    if (this.currentSecondaryRoot.children.length) {
      box.expandByObject(this.currentSecondaryRoot, true);
    }
    if (this.currentFinalRoot.children.length) {
      box.expandByObject(this.currentFinalRoot, true);
    }
    if (box.isEmpty()) {
      const sample = this.replay.trajectory[0];
      const point = sample ? trajectoryPoint(sample) : new THREE.Vector3();
      box.expandByPoint(point);
      box.expandByPoint(point.clone().addScalar(1));
    }
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(0.75, sphere.radius);
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov * 0.5);
    const distance = Math.max(2.5, radius / Math.sin(halfFov) * 1.4);
    const direction = new THREE.Vector3(-0.9, 0.7, 1.15).normalize();
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center.clone().addScaledVector(direction, distance));
    this.camera.near = Math.max(0.01, distance / 1000);
    this.camera.far = Math.max(CONFIG.cameraFarM, distance * 120);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.fitRunId = this.replay.runId;
    this.render();
  }

  resize(width, height) {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.root.traverse((value) => {
      if (value.material?.isLineMaterial) {
        value.material.resolution.set(width, height);
      }
    });
  }

  render() {
    if (this.active) this.renderer.render(this.scene, this.camera);
  }

  startRenderLoop() {
    if (this.animationFrame !== null) return;
    const animate = () => {
      if (!this.active) {
        this.animationFrame = null;
        return;
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.animationFrame = requestAnimationFrame(animate);
    };
    this.animationFrame = requestAnimationFrame(animate);
  }
}

export const PNP_WORLD_DISPLAY_BACKGROUND = DARK_BACKGROUND;
