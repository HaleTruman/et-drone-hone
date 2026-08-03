import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const VISION_CAMERA = {
  widthPx: 640,
  heightPx: 360,
  fx: 320,
  fy: 320,
  cx: 320,
  cy: 180
};

const TELEMETRY_DEFAULT_CHART_HEIGHT_PX = 750;
const TELEMETRY_MIN_CHART_HEIGHT_PX = 240;
const TELEMETRY_MAX_CHART_HEIGHT_PX = 1600;
const TELEMETRY_MIN_SCALE = 0.1;
const TELEMETRY_MAX_SCALE = 80;

function verticalFovFromCamera(camera = VISION_CAMERA) {
  return 2 * Math.atan(camera.heightPx / (2 * camera.fy)) * 180 / Math.PI;
}

const state = {
  runs: [],
  runPath: '',
  summary: null,
  frame: null,
  frameIndex: 0,
  playing: false,
  playTimer: 0,
  inspectorMode: 'telemetry',
  plotMode: 'position',
  viewMode: 'frame',
  hoveredGateIndex: null,
  telemetryChartHeights: {},
  telemetryScales: {},
  telemetryXDomains: {},
  imageNatural: { width: 0, height: 0 },
  frameViewportState: { scrollLeft: 0, scrollTop: 0 },
  plannedPathVisibilityUserSet: false,
  settings: {
    showObservations: true,
    showGateBoxes: true,
    showCenters: true,
    showLabels: true,
    overlayOpacity: 0.9,
    gateSizeM: 2.7,
    verticalFovDeg: verticalFovFromCamera(),
    zoom: 1,
    show3dGrid: true,
    show3dAxes: true,
    show3dDrone: true,
    show3dObservations: true,
    show3dGateMap: true,
    show3dTargets: true,
    show3dTargetLine: true,
    show3dTestPath: true,
    show3dPlannedPath: true,
    show3dTrail: true,
    show3dLookahead: true,
    show3dVelocity: true,
    show3dCurrentDesiredAcceleration: true,
    show3dAutipilotDesiredAcceleration: true,
    show3dLabels: true,
    showTelemetryActual: true,
    showTelemetryTruth: true
  }
};

const map3d = {
  initialized: false,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  root: null,
  animationFrame: 0,
  labels: [],
  originNed: [0, 0, 0],
  referenceDistance: 20
};

const els = {
  runSubtitle: document.getElementById('runSubtitle'),
  runSelect: document.getElementById('runSelect'),
  runState: document.getElementById('runState'),
  refreshButton: document.getElementById('refreshButton'),
  playButton: document.getElementById('playButton'),
  prevButton: document.getElementById('prevButton'),
  nextButton: document.getElementById('nextButton'),
  frameReadout: document.getElementById('frameReadout'),
  frameSlider: document.getElementById('frameSlider'),
  frameCountLabel: document.getElementById('frameCountLabel'),
  frameMeta: document.getElementById('frameMeta'),
  frameStage: document.getElementById('frameStage'),
  frameViewport: document.getElementById('frameViewport'),
  frameImage: document.getElementById('frameImage'),
  overlayCanvas: document.getElementById('overlayCanvas'),
  emptyFrame: document.getElementById('emptyFrame'),
  syncReadout: document.getElementById('syncReadout'),
  hoverReadout: document.getElementById('hoverReadout'),
  statusText: document.getElementById('statusText'),
  summaryGrid: document.getElementById('summaryGrid'),
  timelineCanvas: document.getElementById('timelineCanvas'),
  nearbyFrames: document.getElementById('nearbyFrames'),
  telemetryCards: document.getElementById('telemetryCards'),
  telemetryStatus: document.getElementById('telemetryStatus'),
  telemetryPlot: document.getElementById('telemetryPlot'),
  plotStatus: document.getElementById('plotStatus'),
  inspectorTitle: document.getElementById('inspectorTitle'),
  jsonInspector: document.getElementById('jsonInspector'),
  jsonSize: document.getElementById('jsonSize'),
  jsonPopoutButton: document.getElementById('jsonPopoutButton'),
  jsonPopout: document.getElementById('jsonPopout'),
  jsonPopoutTitle: document.getElementById('jsonPopoutTitle'),
  jsonPopoutInspector: document.getElementById('jsonPopoutInspector'),
  jsonPopoutClose: document.getElementById('jsonPopoutClose'),
  gateList: document.getElementById('gateList'),
  gateCount: document.getElementById('gateCount'),
  overlayStatus: document.getElementById('overlayStatus'),
  showObservations: document.getElementById('showObservations'),
  showGateBoxes: document.getElementById('showGateBoxes'),
  showCenters: document.getElementById('showCenters'),
  showLabels: document.getElementById('showLabels'),
  overlayOpacity: document.getElementById('overlayOpacity'),
  overlayOpacityValue: document.getElementById('overlayOpacityValue'),
  gateSize: document.getElementById('gateSize'),
  gateSizeValue: document.getElementById('gateSizeValue'),
  verticalFov: document.getElementById('verticalFov'),
  verticalFovValue: document.getElementById('verticalFovValue'),
  zoomOutButton: document.getElementById('zoomOutButton'),
  zoomSlider: document.getElementById('zoomSlider'),
  zoomValue: document.getElementById('zoomValue'),
  zoomInButton: document.getElementById('zoomInButton'),
  zoomFitButton: document.getElementById('zoomFitButton'),
  viewTabs: document.querySelectorAll('.viewTab'),
  viewPanes: document.querySelectorAll('.viewPane'),
  map3dMeta: document.getElementById('map3dMeta'),
  map3dStage: document.getElementById('map3dStage'),
  map3dCanvas: document.getElementById('map3dCanvas'),
  map3dEmpty: document.getElementById('map3dEmpty'),
  map3dResetButton: document.getElementById('map3dResetButton'),
  map3dReadout: document.getElementById('map3dReadout'),
  map3dCounts: document.getElementById('map3dCounts'),
  show3dGrid: document.getElementById('show3dGrid'),
  show3dAxes: document.getElementById('show3dAxes'),
  show3dDrone: document.getElementById('show3dDrone'),
  show3dObservations: document.getElementById('show3dObservations'),
  show3dGateMap: document.getElementById('show3dGateMap'),
  show3dTargets: document.getElementById('show3dTargets'),
  show3dTargetLine: document.getElementById('show3dTargetLine'),
  show3dTestPath: document.getElementById('show3dTestPath'),
  show3dPlannedPath: document.getElementById('show3dPlannedPath'),
  show3dTrail: document.getElementById('show3dTrail'),
  show3dLookahead: document.getElementById('show3dLookahead'),
  show3dVelocity: document.getElementById('show3dVelocity'),
  show3dCurrentDesiredAcceleration: document.getElementById('show3dCurrentDesiredAcceleration'),
  show3dAutipilotDesiredAcceleration: document.getElementById('show3dAutipilotDesiredAcceleration'),
  show3dLabels: document.getElementById('show3dLabels'),
  showTelemetryActual: document.getElementById('showTelemetryActual'),
  showTelemetryTruth: document.getElementById('showTelemetryTruth'),
  telemetryMeta: document.getElementById('telemetryMeta'),
  telemetryLegend: document.getElementById('telemetryLegend'),
  telemetryPlotGrid: document.getElementById('telemetryPlotGrid'),
  telemetryEmpty: document.getElementById('telemetryEmpty'),
  controlsHelp: document.getElementById('controlsHelp'),
  controlsHelpClose: document.getElementById('controlsHelpClose')
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function setStatus(text) {
  els.statusText.textContent = text;
}

async function loadRuns({ keepSelection = true } = {}) {
  setStatus('loading run index');
  const payload = await fetchJson('/api/runs');
  state.runs = payload.runs || [];
  els.runSelect.innerHTML = '';
  for (const run of state.runs) {
    const option = document.createElement('option');
    option.value = run.path;
    const counts = run.counts || {};
    option.textContent = `${run.label} | ${run.name} | ${counts.frames || 0} frames`;
    els.runSelect.appendChild(option);
  }
  const previous = state.runPath;
  const paths = new Set(state.runs.map((run) => run.path));
  state.runPath = keepSelection && paths.has(previous) ? previous : (state.runs[0]?.path || '');
  els.runSelect.value = state.runPath;
  if (!state.runPath) {
    setStatus('no run logs found');
    renderEmptyRun();
    return;
  }
  await loadRun();
}

async function loadRun() {
  setStatus('loading run');
  state.plannedPathVisibilityUserSet = false;
  const payload = await fetchJson(`/api/run?run=${encodeURIComponent(state.runPath)}`);
  state.summary = payload.summary;
  state.frameIndex = Math.min(state.frameIndex, Math.max((state.summary.counts?.frames || 1) - 1, 0));
  renderSummary();
  await loadFrame(state.frameIndex);
}

async function loadFrame(index) {
  if (!state.runPath) return;
  const maxFrame = Math.max((state.summary?.counts?.frames || 1) - 1, 0);
  state.frameIndex = Math.max(0, Math.min(Number(index) || 0, maxFrame));
  setStatus(`loading frame ${state.frameIndex + 1}`);
  const payload = await fetchJson(`/api/frame?run=${encodeURIComponent(state.runPath)}&frame=${state.frameIndex}`);
  state.frame = payload;
  applyDefault3dPathVisibility();
  els.frameImage.src = `/api/frame-image?run=${encodeURIComponent(state.runPath)}&frame=${state.frameIndex}&v=${payload.frame.jpeg_size}`;
  els.frameImage.onload = () => {
    state.imageNatural = { width: els.frameImage.naturalWidth || 0, height: els.frameImage.naturalHeight || 0 };
    if (isFrameViewVisible()) {
      syncCanvasToImage();
      restoreFrameViewport();
      renderOverlay();
    }
  };
  renderFrame();
  setStatus('ready');
}

function renderEmptyRun() {
  els.runSubtitle.textContent = 'no logs';
  els.summaryGrid.innerHTML = '';
  els.frameSlider.max = 0;
  els.frameSlider.value = 0;
  els.emptyFrame.hidden = false;
  els.frameMeta.textContent = 'No frame loaded.';
  els.jsonInspector.textContent = '';
}

function renderSummary() {
  const summary = state.summary || {};
  const counts = summary.counts || {};
  els.runSubtitle.textContent = summary.name || 'no run selected';
  els.runState.textContent = `${counts.cycles || 0} cycles`;
  els.frameCountLabel.textContent = `${counts.frames || 0}`;
  els.frameSlider.max = Math.max((counts.frames || 1) - 1, 0);
  els.frameSlider.value = state.frameIndex;
  const span = summary.telemetry_span_s == null ? 'n/a' : `${summary.telemetry_span_s.toFixed(3)}s`;
  const values = [
    ['Frames', counts.frames || 0],
    ['Cycles', counts.cycles || 0],
    ['Observations', counts.observations || 0],
    ['Events', counts.events || 0],
    ['Span', span],
    ['Modes', (summary.system_modes || []).join(', ') || 'n/a'],
    ['Scenario', summary.metadata?.scenario || 'n/a'],
    ['Frame IDs', summary.frame_id_range ? `${summary.frame_id_range[0]}-${summary.frame_id_range[1]}` : 'n/a']
  ];
  els.summaryGrid.innerHTML = values.map(([label, value]) => (
    `<div class="metric"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(String(value))}">${escapeHtml(String(value))}</strong></div>`
  )).join('');
}

function observedFrameIndices() {
  const values = state.summary?.observed_frame_indices;
  return Array.isArray(values)
    ? values.map((value) => Number(value)).filter((value) => Number.isInteger(value) && value >= 0)
    : [];
}

function jumpObservedFrame(direction) {
  const observed = observedFrameIndices();
  if (!observed.length) {
    setStatus('no observed frames in this run');
    return;
  }
  const current = Number(state.frameIndex) || 0;
  const step = direction < 0 ? -1 : 1;
  let target = null;
  if (step > 0) {
    target = observed.find((index) => index > current) ?? observed[0];
  } else {
    for (let index = observed.length - 1; index >= 0; index -= 1) {
      if (observed[index] < current) {
        target = observed[index];
        break;
      }
    }
    if (target == null) target = observed[observed.length - 1];
  }
  loadFrame(target);
}

function renderFrame() {
  const payload = state.frame;
  if (!payload) return;
  const frame = payload.frame;
  els.emptyFrame.hidden = true;
  els.frameSlider.value = payload.index;
  els.frameReadout.textContent = `Frame ${payload.index + 1}/${payload.count}`;
  els.frameMeta.textContent = `id=${frame.frame_id} cycle=${frame.cycle ?? 'n/a'} sim=${frame.sim_time_ns} path=${frame.path}`;
  const error = payload.sync?.alignment_error_ms;
  els.syncReadout.textContent = `telemetry cycle ${payload.sync?.cycle_index ?? 'n/a'} | alignment ${error == null ? 'n/a' : `${error.toFixed(3)} ms`}`;
  renderNearbyFrames();
  renderTelemetryCards();
  renderGateList();
  renderInspector();
  renderTimeline();
  renderTelemetryPlot();
  renderTelemetryDashboard();
  renderOverlay();
  renderMap3d();
}

function renderNearbyFrames() {
  const rows = state.frame?.nearby || [];
  els.nearbyFrames.innerHTML = '';
  for (const row of rows) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `frameRow${row.index === state.frameIndex ? ' active' : ''}`;
    button.innerHTML = `<span>#${row.index + 1}</span><span>${row.frame_id}</span><span>cy ${row.cycle ?? '-'}</span><i class="dot ${row.has_observation ? 'on' : ''}"></i>`;
    button.addEventListener('click', () => loadFrame(row.index));
    els.nearbyFrames.appendChild(button);
  }
}

function renderTelemetryCards() {
  const telemetry = state.frame?.telemetry?.telemetry || {};
  const geometricPathFollower = state.frame?.telemetry?.geometric_path_follower || {};
  const values = [
    ['Position NED', formatVec(telemetry.position_local_ned_m, 'm')],
    ['Velocity NED', formatVec(telemetry.velocity_local_ned_mps, 'm/s')],
    ['Acceleration', formatVec(telemetry.acceleration_local_ned_mps2, 'm/s2')],
    ['Desired Accel NED', formatVec(geometricPathFollower.desired_acceleration_local_ned_mps2, 'm/s2')],
    ['Body Rates', formatVec(telemetry.body_rates_frd_rps || telemetry.body_rates_rps, 'rad/s')],
    ['Attitude', formatVec(telemetry.attitude_quaternion || telemetry.attitude, '')],
    ['Sim Time', telemetry.sim_time_ns ?? state.frame?.telemetry?.sim_time_ns ?? 'n/a']
  ];
  els.telemetryCards.innerHTML = values.map(([label, value]) => (
    `<div class="telemetryCard"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`
  )).join('');
  els.telemetryStatus.textContent = state.frame?.telemetry ? 'aligned' : 'missing';
}

function renderGateList() {
  const gates = state.frame?.observation_gates || [];
  const selectedGates = selectedGateEntries();
  els.gateCount.textContent = String(gates.length + selectedGates.length);
  els.overlayStatus.textContent = `${gates.length} gates | ${selectedGates.length} selected`;
  state.hoveredGateIndex = null;
  if (!gates.length && !selectedGates.length) {
    els.gateList.innerHTML = '<div class="gateCard"><span>No observation matched this frame.</span></div>';
    return;
  }
  const observationCards = gates.map((gate, index) => {
    const position = gate.position_xyz
      ? `${formatVec(gate.position_xyz, 'm')} camera`
      : `${formatVec(gate.position_local_ned_m || gate.position_local_ned || gate.position_relative_ned_m, 'm')} local NED`;
    const orientation = gateHasOrientation(gate)
      ? (gate.orientation_xyz
        ? formatVec(gate.orientation_xyz, 'deg')
        : formatQuat(gate.orientation_local_ned_quat || gate.orientation_quat))
      : 'none';
    const card = document.createElement('div');
    card.className = 'gateCard';
    card.dataset.gateIndex = String(index);
    card.innerHTML = `
      <strong>${escapeHtml(gate.id || `gate-${index + 1}`)}</strong>
      <span>pos ${escapeHtml(position)} | conf ${formatNumber(gate.position_confidence)}</span>
      <span>orientation ${escapeHtml(orientation)} | conf ${formatNumber(gate.orientation_confidence)}</span>
    `;
    card.addEventListener('mouseenter', () => {
      state.hoveredGateIndex = index;
      card.classList.add('hovered');
      renderOverlay();
      renderMap3d();
    });
    card.addEventListener('mouseleave', () => {
      if (state.hoveredGateIndex === index) state.hoveredGateIndex = null;
      card.classList.remove('hovered');
      renderOverlay();
      renderMap3d();
    });
    return card;
  });
  const selectedCards = selectedGates.map(({ role, gate }) => {
    const position = `${formatVec(gate.position_local_ned_m || gate.position_local_ned || gate.position_relative_ned_m, 'm')} local NED`;
    const card = document.createElement('div');
    card.className = `gateCard selectedGate ${role}`;
    card.innerHTML = `
      <strong>${escapeHtml(role === 'target' ? 'Target Gate' : 'Next Gate')} · ${escapeHtml(gate.id || 'n/a')}</strong>
      <span>pos ${escapeHtml(position)} | conf ${formatNumber(gate.confidence)}</span>
      <span>sequence ${escapeHtml(String(gate.sequence ?? 'n/a'))} | crossed ${gate.crossed ? 'yes' : 'no'}</span>
    `;
    return card;
  });
  els.gateList.replaceChildren(...selectedCards, ...observationCards);
}

function renderInspector() {
  const mode = state.inspectorMode;
  let title = 'Telemetry JSON';
  let payload = state.frame?.sync || {};
  if (mode === 'observation') {
    title = 'Observation JSON';
    payload = state.frame?.observation || { note: 'No observation matched this frame.' };
  } else if (mode === 'raw') {
    title = 'Frame Payload JSON';
    payload = state.frame || {};
  }
  const text = JSON.stringify(payload, null, 2);
  els.inspectorTitle.textContent = title;
  els.jsonInspector.textContent = text;
  els.jsonSize.textContent = `${text.length.toLocaleString()} chars`;
  els.jsonPopoutTitle.textContent = title;
  els.jsonPopoutInspector.textContent = text;
}

function renderTimeline() {
  const canvas = els.timelineCanvas;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#050708';
  ctx.fillRect(0, 0, width, height);
  const frames = state.frame?.timeline?.frames || [];
  const count = Math.max(state.frame?.count || 1, 1);
  ctx.strokeStyle = '#2a333b';
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  for (const row of frames) {
    const x = Math.round((row.index / Math.max(count - 1, 1)) * (width - 20)) + 10;
    ctx.strokeStyle = row.observed ? '#ffd45a' : '#3a4650';
    ctx.beginPath();
    ctx.moveTo(x, 18);
    ctx.lineTo(x, height - 18);
    ctx.stroke();
  }
  const selectedX = Math.round((state.frameIndex / Math.max(count - 1, 1)) * (width - 20)) + 10;
  ctx.strokeStyle = '#5cf2ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(selectedX, 8);
  ctx.lineTo(selectedX, height - 8);
  ctx.stroke();
  ctx.lineWidth = 1;
}

function renderTelemetryPlot() {
  const series = state.frame?.telemetry_series;
  const canvas = els.telemetryPlot;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#050708';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#2a333b';
  ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
  if (!series || !series.times_s.length) return;
  const values = series[state.plotMode] || [];
  const labels = state.plotMode === 'rates' ? ['p', 'q', 'r'] : ['x', 'y', 'z'];
  const colors = ['#5cf2ff', '#ff6048', '#71e989'];
  const plot = { left: 42, top: 14, right: 10, bottom: 26 };
  const xs = series.times_s;
  const yValues = values.flat().filter((value) => Number.isFinite(value));
  if (!yValues.length) return;
  const minX = xs[0];
  const maxX = xs[xs.length - 1] || minX + 1;
  let minY = Math.min(...yValues);
  let maxY = Math.max(...yValues);
  if (Math.abs(maxY - minY) < 1e-9) {
    minY -= 1;
    maxY += 1;
  }
  drawGrid(ctx, canvas, plot, minX, maxX, minY, maxY);
  for (let axis = 0; axis < 3; axis += 1) {
    ctx.strokeStyle = colors[axis];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i += 1) {
      const yRaw = values[i]?.[axis];
      if (!Number.isFinite(yRaw)) continue;
      const x = map(xs[i], minX, maxX, plot.left, canvas.width - plot.right);
      const y = map(yRaw, minY, maxY, canvas.height - plot.bottom, plot.top);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.fillStyle = colors[axis];
    ctx.fillText(labels[axis], canvas.width - plot.right - 36 + axis * 12, plot.top + 10);
  }
  if (series.selected != null) {
    const x = map(series.selected, minX, maxX, plot.left, canvas.width - plot.right);
    ctx.strokeStyle = '#ffd45a';
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, canvas.height - plot.bottom);
    ctx.stroke();
  }
  const hover = nearestCompactTelemetryPoint(canvas, series, values, labels, colors, plot, minX, maxX, minY, maxY);
  if (hover) drawTelemetryHoverTip(ctx, hover, canvas.width, canvas.height);
  els.plotStatus.textContent = state.plotMode;
}

function renderTelemetryDashboard() {
  if (!els.telemetryPlotGrid) return;
  const series = state.frame?.telemetry_series;
  const groups = Array.isArray(series?.groups) ? series.groups : [];
  const hasTelemetry = Boolean(series?.times_s?.length && groups.length);
  els.telemetryEmpty.hidden = hasTelemetry;
  els.telemetryPlotGrid.hidden = !hasTelemetry;
  els.telemetryMeta.textContent = hasTelemetry
    ? `${series.times_s.length} samples | selected t=${series.selected == null ? 'n/a' : `${series.selected.toFixed(3)}s`}`
    : 'No telemetry samples loaded.';
  renderTelemetryLegend(groups);
  if (!hasTelemetry) {
    els.telemetryPlotGrid.innerHTML = '';
    return;
  }
  const existing = new Map([...els.telemetryPlotGrid.querySelectorAll('.telemetryChart')].map((item) => [item.dataset.groupId, item]));
  const ordered = [];
  for (const group of groups) {
    let chart = existing.get(group.id);
    if (!chart) {
      chart = document.createElement('section');
      chart.className = 'telemetryChart';
      chart.dataset.groupId = group.id;
      chart.innerHTML = `
        <div class="telemetryChartHead">
          <div class="telemetryChartTitle">
            <strong></strong>
            <span class="telemetryChartScale"></span>
          </div>
          <div class="telemetryChartLegend"></div>
        </div>
        <canvas width="760" height="250"></canvas>
        <div class="telemetryResizeHandle" role="separator" aria-orientation="horizontal"></div>
      `;
      installTelemetryChartInteractions(chart);
    }
    const height = telemetryChartHeight(group.id);
    chart.style.setProperty('--telemetry-chart-height', `${height}px`);
    chart.querySelector('strong').textContent = group.title || group.id;
    chart.querySelector('.telemetryChartScale').textContent = telemetryScaleLabel(group);
    renderTelemetryChartLegend(chart.querySelector('.telemetryChartLegend'), group);
    chart._telemetryGroup = group;
    ordered.push(chart);
  }
  els.telemetryPlotGrid.replaceChildren(...ordered);
  for (const chart of ordered) {
    drawTelemetryGroup(chart.querySelector('canvas'), series, chart._telemetryGroup);
  }
}

function installTelemetryChartInteractions(chart) {
  if (chart.dataset.telemetryInteractions === 'installed') return;
  chart.dataset.telemetryInteractions = 'installed';
  const canvas = chart.querySelector('canvas');
  const handle = chart.querySelector('.telemetryResizeHandle');

  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    const groupId = chart.dataset.groupId;
    const startY = event.clientY;
    const startHeight = telemetryChartHeight(groupId);
    handle.setPointerCapture(event.pointerId);

    const onMove = (moveEvent) => {
      const nextHeight = clamp(
        startHeight + moveEvent.clientY - startY,
        TELEMETRY_MIN_CHART_HEIGHT_PX,
        TELEMETRY_MAX_CHART_HEIGHT_PX
      );
      state.telemetryChartHeights[groupId] = Math.round(nextHeight);
      chart.style.setProperty('--telemetry-chart-height', `${state.telemetryChartHeights[groupId]}px`);
      if (state.frame?.telemetry_series && chart._telemetryGroup) {
        drawTelemetryGroup(canvas, state.frame.telemetry_series, chart._telemetryGroup);
      }
    };

    const onUp = (upEvent) => {
      handle.releasePointerCapture?.(upEvent.pointerId);
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
    };

    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  });

  canvas.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    const groupId = chart.dataset.groupId;
    const series = state.frame?.telemetry_series;
    const group = chart._telemetryGroup;
    const metrics = telemetryPlotMetrics(canvas, series, group);
    if (!metrics) return;
    const pointer = canvasPointer(canvas, event);
    const startScale = telemetryChartScale(groupId);
    const dragMode = pointer.x < metrics.plot.left ? 'scale-y' : 'select-x';
    const startPointer = pointer;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add(dragMode === 'scale-y' ? 'scaling' : 'selecting');
    canvas._telemetryDragMode = dragMode;
    canvas._telemetryPointer = pointer;

    const onMove = (moveEvent) => {
      const nextPointer = canvasPointer(canvas, moveEvent);
      canvas._telemetryPointer = nextPointer;
      if (dragMode === 'scale-y') {
        const deltaY = nextPointer.y - startPointer.y;
        state.telemetryScales[groupId] = clamp(
          startScale * Math.exp(-deltaY * 0.01),
          TELEMETRY_MIN_SCALE,
          TELEMETRY_MAX_SCALE
        );
        chart.querySelector('.telemetryChartScale').textContent = telemetryScaleLabel(chart._telemetryGroup);
      } else {
        canvas._telemetrySelection = {
          startX: clamp(startPointer.x, metrics.plot.left, metrics.cssWidth - metrics.plot.right),
          endX: clamp(nextPointer.x, metrics.plot.left, metrics.cssWidth - metrics.plot.right)
        };
      }
      if (state.frame?.telemetry_series && chart._telemetryGroup) {
        drawTelemetryGroup(canvas, state.frame.telemetry_series, chart._telemetryGroup);
      }
    };

    const onUp = (upEvent) => {
      const endPointer = canvasPointer(canvas, upEvent);
      if (dragMode === 'select-x' && metrics) {
        const startX = clamp(startPointer.x, metrics.plot.left, metrics.cssWidth - metrics.plot.right);
        const endX = clamp(endPointer.x, metrics.plot.left, metrics.cssWidth - metrics.plot.right);
        if (Math.abs(endX - startX) >= 8) {
          const rangeStart = map(Math.min(startX, endX), metrics.plot.left, metrics.cssWidth - metrics.plot.right, metrics.minX, metrics.maxX);
          const rangeEnd = map(Math.max(startX, endX), metrics.plot.left, metrics.cssWidth - metrics.plot.right, metrics.minX, metrics.maxX);
          if (Number.isFinite(rangeStart) && Number.isFinite(rangeEnd) && rangeEnd > rangeStart) {
            state.telemetryXDomains[groupId] = [rangeStart, rangeEnd];
          }
        }
      }
      canvas.releasePointerCapture?.(upEvent.pointerId);
      canvas.classList.remove('scaling', 'selecting');
      canvas._telemetryDragMode = null;
      canvas._telemetrySelection = null;
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      if (state.frame?.telemetry_series && chart._telemetryGroup) {
        chart.querySelector('.telemetryChartScale').textContent = telemetryScaleLabel(chart._telemetryGroup);
        drawTelemetryGroup(canvas, state.frame.telemetry_series, chart._telemetryGroup);
      }
    };

    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
  });

  canvas.addEventListener('pointermove', (event) => {
    if (canvas._telemetryDragMode || !state.frame?.telemetry_series || !chart._telemetryGroup) return;
    canvas._telemetryPointer = canvasPointer(canvas, event);
    const metrics = telemetryPlotMetrics(canvas, state.frame.telemetry_series, chart._telemetryGroup);
    canvas.classList.toggle('axisHover', Boolean(metrics && canvas._telemetryPointer.x < metrics.plot.left));
    canvas.classList.toggle('plotHover', Boolean(metrics && canvas._telemetryPointer.x >= metrics.plot.left));
    drawTelemetryGroup(canvas, state.frame.telemetry_series, chart._telemetryGroup);
  });

  canvas.addEventListener('pointerleave', () => {
    if (canvas._telemetryDragMode) return;
    canvas._telemetryPointer = null;
    canvas.classList.remove('axisHover', 'plotHover');
    if (state.frame?.telemetry_series && chart._telemetryGroup) {
      drawTelemetryGroup(canvas, state.frame.telemetry_series, chart._telemetryGroup);
    }
  });

  canvas.addEventListener('dblclick', (event) => {
    event.preventDefault();
    delete state.telemetryScales[chart.dataset.groupId];
    delete state.telemetryXDomains[chart.dataset.groupId];
    canvas._telemetrySelection = null;
    chart.querySelector('.telemetryChartScale').textContent = telemetryScaleLabel(chart._telemetryGroup);
    if (state.frame?.telemetry_series && chart._telemetryGroup) {
      drawTelemetryGroup(canvas, state.frame.telemetry_series, chart._telemetryGroup);
    }
  });
}

function telemetryChartHeight(groupId) {
  const value = Number(state.telemetryChartHeights[groupId]);
  return Number.isFinite(value)
    ? clamp(value, TELEMETRY_MIN_CHART_HEIGHT_PX, TELEMETRY_MAX_CHART_HEIGHT_PX)
    : TELEMETRY_DEFAULT_CHART_HEIGHT_PX;
}

function telemetryChartScale(groupId) {
  const value = Number(state.telemetryScales[groupId]);
  return Number.isFinite(value)
    ? clamp(value, TELEMETRY_MIN_SCALE, TELEMETRY_MAX_SCALE)
    : 1;
}

function telemetryXDomain(groupId, minX, maxX) {
  const domain = state.telemetryXDomains[groupId];
  if (!Array.isArray(domain) || domain.length < 2) return [minX, maxX];
  const start = clamp(Number(domain[0]), minX, maxX);
  const end = clamp(Number(domain[1]), minX, maxX);
  return end > start ? [start, end] : [minX, maxX];
}

function telemetryScaleLabel(group) {
  if (!group) return '';
  const yScale = telemetryChartScale(group.id);
  const xDomain = state.telemetryXDomains[group.id];
  const xLabel = Array.isArray(xDomain) && xDomain.length >= 2
    ? ` | x ${formatNumber(xDomain[0])}-${formatNumber(xDomain[1])}s`
    : '';
  return `${group.unit || ''}${yScale === 1 ? '' : ` | y x${yScale.toFixed(2)}`}${xLabel}`;
}

function renderTelemetryLegend(groups) {
  if (!els.telemetryLegend) return;
  const hasSimTruth = groups.some((group) => (group.series || []).some((item) => item.label === 'sim truth'));
  const items = [
    ['x / p / qw', 'north'],
    ['y / q / qx', 'east'],
    ['z / r / qy', 'down'],
    ['qz', 'telemetryFourth'],
    ['actual', 'telemetryActual']
  ];
  if (hasSimTruth) items.push(['truth', 'telemetryTruth']);
  els.telemetryLegend.innerHTML = items.map(([label, cls]) => (
    `<span><i class="legendSwatch ${cls}"></i>${escapeHtml(label)}</span>`
  )).join('');
}

function visibleTelemetryGroupSeries(group) {
  return (group.series || []).filter((item) => {
    const isTruth = item.label === 'sim truth';
    return isTruth ? state.settings.showTelemetryTruth : state.settings.showTelemetryActual;
  });
}

function renderTelemetryChartLegend(container, group) {
  if (!container) return;
  const axes = Array.isArray(group.axes) ? group.axes : [];
  const groupSeries = visibleTelemetryGroupSeries(group);
  const items = [];
  for (const item of groupSeries) {
    const isTruth = item.label === 'sim truth';
    axes.forEach((axisLabel, axis) => {
      items.push({
        label: `${axisLabel} ${isTruth ? 'truth' : item.label}`,
        color: telemetryAxisColor(axis),
        truth: isTruth
      });
    });
  }
  container.innerHTML = items.map((item) => (
    `<span><i class="telemetryPlotSwatch${item.truth ? ' truth' : ''}" style="border-top-color:${item.color}"></i>${escapeHtml(item.label)}</span>`
  )).join('');
}

function telemetryAxisColor(axis) {
  return ['#5cf2ff', '#ff6048', '#71e989', '#ffd45a'][axis % 4];
}

function drawTelemetryGroup(canvas, series, group) {
  const metrics = telemetryPlotMetrics(canvas, series, group, { resize: true });
  if (!metrics) return;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(metrics.dpr, 0, 0, metrics.dpr, 0, 0);
  const { cssWidth, cssHeight, plot, minX, maxX, minY, maxY, groupSeries } = metrics;
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = '#050708';
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.strokeStyle = '#2a333b';
  ctx.strokeRect(0.5, 0.5, cssWidth - 1, cssHeight - 1);
  drawGrid(ctx, { width: cssWidth, height: cssHeight }, plot, minX, maxX, minY, maxY);
  const sourceStyles = {
    'estimate': { alpha: 1, dash: [] },
    'sim truth': { alpha: 0.85, dash: [7, 4] },
    'imu': { alpha: 0.95, dash: [] }
  };
  groupSeries.forEach((item) => {
    const style = sourceStyles[item.label] || { alpha: 0.9, dash: [] };
    (group.axes || []).forEach((axisLabel, axis) => {
      ctx.save();
      ctx.globalAlpha = style.alpha;
      ctx.strokeStyle = telemetryAxisColor(axis);
      ctx.lineWidth = item.label === 'sim truth' ? 1.2 : 1.6;
      ctx.setLineDash(style.dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < metrics.xs.length; i += 1) {
        const yRaw = item.values?.[i]?.[axis];
        const xRaw = metrics.xs[i];
        if (!Number.isFinite(Number(xRaw)) || xRaw < minX || xRaw > maxX) {
          started = false;
          continue;
        }
        if (!Number.isFinite(Number(yRaw))) {
          started = false;
          continue;
        }
        const x = map(xRaw, minX, maxX, plot.left, cssWidth - plot.right);
        const y = map(Number(yRaw), minY, maxY, cssHeight - plot.bottom, plot.top);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.restore();
    });
  });
  if (series.selected != null) {
    const x = map(series.selected, minX, maxX, plot.left, cssWidth - plot.right);
    ctx.strokeStyle = '#ffffff';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, cssHeight - plot.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const hover = nearestTelemetryGroupPoint(canvas, series, group, groupSeries, plot, minX, maxX, minY, maxY, cssWidth, cssHeight);
  if (hover) drawTelemetryHoverTip(ctx, hover, cssWidth, cssHeight);
  if (canvas._telemetrySelection) drawTelemetrySelection(ctx, canvas._telemetrySelection, plot, cssWidth, cssHeight);
}

function telemetryPlotMetrics(canvas, series, group, { resize = false } = {}) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.floor((rect.width || 760) * dpr));
  const height = Math.max(190, Math.floor((rect.height || 250) * dpr));
  if (resize && (canvas.width !== width || canvas.height !== height)) {
    canvas.width = width;
    canvas.height = height;
  }
  const cssWidth = width / dpr;
  const cssHeight = height / dpr;
  const xs = Array.isArray(series?.times_s) ? series.times_s : [];
  const groupSeries = group ? visibleTelemetryGroupSeries(group) : [];
  const fullMinX = xs[0];
  const fullMaxX = xs[xs.length - 1] || fullMinX + 1;
  if (!xs.length || !Number.isFinite(Number(fullMinX)) || !Number.isFinite(Number(fullMaxX))) return null;
  const [minX, maxX] = telemetryXDomain(group.id, Number(fullMinX), Number(fullMaxX));
  const values = [];
  for (const item of groupSeries) {
    for (let i = 0; i < xs.length; i += 1) {
      const x = Number(xs[i]);
      if (!Number.isFinite(x) || x < minX || x > maxX) continue;
      for (const value of item.values?.[i] || []) {
        if (Number.isFinite(Number(value))) values.push(Number(value));
      }
    }
  }
  if (!values.length) return null;
  let minY = Math.min(...values);
  let maxY = Math.max(...values);
  if (Math.abs(maxY - minY) < 1e-9) {
    minY -= 1;
    maxY += 1;
  }
  const yScale = telemetryChartScale(group.id);
  if (yScale !== 1) {
    const centerY = (minY + maxY) / 2;
    const halfRange = ((maxY - minY) / 2) / yScale;
    minY = centerY - halfRange;
    maxY = centerY + halfRange;
  }
  return {
    dpr,
    cssWidth,
    cssHeight,
    xs,
    groupSeries,
    plot: { left: 48, top: 14, right: 14, bottom: 28 },
    minX,
    maxX,
    minY,
    maxY
  };
}

function canvasPointer(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return {
    x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * (canvas.width / dpr),
    y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * (canvas.height / dpr)
  };
}

function canvasPixelPointer(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * canvas.width,
    y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * canvas.height
  };
}

function nearestCompactTelemetryPoint(canvas, series, values, labels, colors, plot, minX, maxX, minY, maxY) {
  const pointer = canvas._compactTelemetryPointer;
  if (!pointer) return null;
  let best = null;
  const xs = series.times_s || [];
  for (let axis = 0; axis < 3; axis += 1) {
    for (let i = 0; i < xs.length; i += 1) {
      const value = Number(values[i]?.[axis]);
      if (!Number.isFinite(value)) continue;
      const x = map(xs[i], minX, maxX, plot.left, canvas.width - plot.right);
      const y = map(value, minY, maxY, canvas.height - plot.bottom, plot.top);
      const distance = Math.hypot(pointer.x - x, pointer.y - y);
      if (!best || distance < best.distance) {
        best = { x, y, distance, color: colors[axis], title: labels[axis], time: xs[i], value };
      }
    }
  }
  return best && best.distance <= 16 ? best : null;
}

function nearestTelemetryGroupPoint(canvas, series, group, groupSeries, plot, minX, maxX, minY, maxY, cssWidth, cssHeight) {
  const pointer = canvas._telemetryPointer;
  if (!pointer) return null;
  const xs = series.times_s || [];
  let best = null;
  groupSeries.forEach((item) => {
    (group.axes || []).forEach((axisLabel, axis) => {
      for (let i = 0; i < xs.length; i += 1) {
        const xRaw = Number(xs[i]);
        if (!Number.isFinite(xRaw) || xRaw < minX || xRaw > maxX) continue;
        const value = Number(item.values?.[i]?.[axis]);
        if (!Number.isFinite(value)) continue;
        const x = map(xRaw, minX, maxX, plot.left, cssWidth - plot.right);
        const y = map(value, minY, maxY, cssHeight - plot.bottom, plot.top);
        const distance = Math.hypot(pointer.x - x, pointer.y - y);
        if (!best || distance < best.distance) {
          best = {
            x,
            y,
            distance,
            color: telemetryAxisColor(axis),
            title: `${axisLabel} ${item.label}`,
            time: xs[i],
            value
          };
        }
      }
    });
  });
  return best && best.distance <= 16 ? best : null;
}

function drawTelemetrySelection(ctx, selection, plot, width, height) {
  const startX = clamp(selection.startX, plot.left, width - plot.right);
  const endX = clamp(selection.endX, plot.left, width - plot.right);
  const left = Math.min(startX, endX);
  const selectionWidth = Math.abs(endX - startX);
  ctx.save();
  ctx.fillStyle = 'rgba(92, 242, 255, .16)';
  ctx.strokeStyle = 'rgba(92, 242, 255, .7)';
  ctx.fillRect(left, plot.top, selectionWidth, height - plot.top - plot.bottom);
  ctx.strokeRect(left + 0.5, plot.top + 0.5, Math.max(0, selectionWidth - 1), height - plot.top - plot.bottom - 1);
  ctx.restore();
}

function drawTelemetryHoverTip(ctx, hover, width, height) {
  const lines = [
    hover.title,
    `t=${formatNumber(hover.time)}s`,
    `v=${formatNumber(hover.value)}`
  ];
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = hover.color;
  ctx.fillStyle = hover.color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(hover.x, hover.y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(hover.x, hover.y, 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  const tooltipWidth = Math.max(...lines.map((line) => ctx.measureText(line).width)) + 18;
  const tooltipHeight = 52;
  const x = Math.min(Math.max(hover.x + 12, 6), width - tooltipWidth - 6);
  const y = Math.min(Math.max(hover.y - tooltipHeight - 10, 6), height - tooltipHeight - 6);
  ctx.fillStyle = 'rgba(8, 11, 14, .94)';
  ctx.strokeStyle = 'rgba(255, 255, 255, .18)';
  ctx.fillRect(x, y, tooltipWidth, tooltipHeight);
  ctx.strokeRect(x + 0.5, y + 0.5, tooltipWidth - 1, tooltipHeight - 1);
  ctx.fillStyle = hover.color;
  lines.forEach((line, index) => ctx.fillText(line, x + 9, y + 16 + index * 14));
  ctx.restore();
}

function drawGrid(ctx, canvas, plot, minX, maxX, minY, maxY) {
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  ctx.fillStyle = '#98a6b3';
  ctx.strokeStyle = '#1f2830';
  for (let i = 0; i <= 4; i += 1) {
    const t = i / 4;
    const y = map(t, 0, 1, canvas.height - plot.bottom, plot.top);
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(canvas.width - plot.right, y);
    ctx.stroke();
    const value = map(t, 0, 1, minY, maxY);
    ctx.fillText(value.toFixed(2), 5, y + 3);
  }
  ctx.fillText(`${minX.toFixed(1)}s`, plot.left, canvas.height - 8);
  ctx.fillText(`${maxX.toFixed(1)}s`, canvas.width - plot.right - 54, canvas.height - 8);
}

function syncCanvasToImage() {
  if (!isFrameViewVisible()) return;
  const width = state.imageNatural.width || 1280;
  const height = state.imageNatural.height || 720;
  const stageWidth = Math.max(1, els.frameStage.clientWidth);
  const stageHeight = Math.max(1, els.frameStage.clientHeight);
  const fitScale = Math.min(stageWidth / width, stageHeight / height);
  const displayScale = Math.max(0.05, fitScale * state.settings.zoom);
  const displayWidth = Math.max(1, Math.round(width * displayScale));
  const displayHeight = Math.max(1, Math.round(height * displayScale));
  const marginLeft = Math.max(0, Math.floor((stageWidth - displayWidth) / 2));
  const marginTop = Math.max(0, Math.floor((stageHeight - displayHeight) / 2));

  Object.assign(els.frameViewport.style, {
    width: `${displayWidth}px`,
    height: `${displayHeight}px`,
    marginLeft: `${marginLeft}px`,
    marginTop: `${marginTop}px`,
    marginRight: displayWidth < stageWidth ? `${marginLeft}px` : '0',
    marginBottom: displayHeight < stageHeight ? `${marginTop}px` : '0'
  });

  if (els.overlayCanvas.width !== width || els.overlayCanvas.height !== height) {
    els.overlayCanvas.width = width;
    els.overlayCanvas.height = height;
  }
}

function renderOverlay() {
  if (!isFrameViewVisible()) return;
  syncCanvasToImage();
  const canvas = els.overlayCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const gates = state.frame?.observation_gates || [];
  const selectedGates = selectedGateEntries();
  if (!state.settings.showObservations || (!gates.length && !selectedGates.length)) return;
  const alpha = state.settings.overlayOpacity;
  gates.forEach((gate, index) => drawGateObservation(ctx, canvas, gate, index, alpha));
  selectedGates.forEach((entry) => drawSelectedGateOverlay(ctx, canvas, entry, alpha));
}

function initMap3d() {
  if (map3d.initialized || !els.map3dCanvas) return;
  map3d.scene = new THREE.Scene();
  map3d.scene.background = new THREE.Color(0x050607);
  map3d.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 5000);
  map3d.camera.position.set(-10, 8, 14);
  map3d.renderer = new THREE.WebGLRenderer({ canvas: els.map3dCanvas, antialias: true });
  map3d.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  map3d.controls = new OrbitControls(map3d.camera, els.map3dCanvas);
  map3d.controls.enableDamping = true;
  map3d.controls.screenSpacePanning = true;
  configureMap3dControls();
  map3d.controls.userData = { fitted: false };
  map3d.root = new THREE.Group();
  map3d.scene.add(map3d.root);
  map3d.scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(-8, 12, 5);
  map3d.scene.add(sun);
  map3d.initialized = true;
  animateMap3d();
  resizeMap3d();
}

function animateMap3d() {
  if (!map3d.initialized) return;
  map3d.animationFrame = requestAnimationFrame(animateMap3d);
  updateMap3dControlSensitivity();
  map3d.controls.update();
  map3d.renderer.render(map3d.scene, map3d.camera);
}

function resizeMap3d() {
  if (!map3d.initialized || !els.map3dStage) return;
  const rect = els.map3dStage.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  map3d.camera.aspect = width / height;
  map3d.camera.updateProjectionMatrix();
  map3d.renderer.setSize(width, height, false);
}

function renderMap3d({ resetCamera = false } = {}) {
  if (!state.frame || !els.map3dCanvas) return;
  initMap3d();
  clearGroup(map3d.root);
  map3d.labels.forEach((label) => label.material.map?.dispose());
  map3d.labels = [];
  const scene = state.frame.scene || {};
  const drone = scene.drone || {};
  const dronePosition = point3(drone.position_local_ned_m) || [0, 0, 0];
  const droneQuaternion = quat4(drone.attitude_quaternion) || [1, 0, 0, 0];
  map3d.originNed = dronePosition;
  const points = [dronePosition];
  addGridLayer();
  addWorldAxes();
  addTrail(scene, points);
  if (!scene.planned_path_is_test_path) {
    addPathLayer(scene.planned_path, points, {
      enabled: state.settings.show3dPlannedPath,
      color: 0xff2f2f,
      tubeRadius: 0.04
    });
  }
  addPathLayer(scene.test_path, points, {
    enabled: state.settings.show3dTestPath,
    color: 0xffffff,
    tubeRadius: 0.0225,
    colorByCurvature: true
  });
  addGateMap(scene.gate_map || [], dronePosition, points);
  addTargetLine(scene, dronePosition, points);
  addTargetGates(scene, dronePosition, points);
  addObservationGates(scene, dronePosition, droneQuaternion, points);
  addDrone(dronePosition, droneQuaternion);
  addLookaheadVector(dronePosition, points);
  addVelocityVector(dronePosition, points);
  addDesiredAccelerationVectors(dronePosition, points);
  fitCameraToPoints(points, resetCamera);
  const obsCount = Array.isArray(scene.observation_gates) ? scene.observation_gates.length : 0;
  const mapCount = Array.isArray(scene.gate_map) ? scene.gate_map.length : 0;
  const targetCount = selectedGateEntries(scene).length;
  const testPathCount = scene.test_path?.points_local_ned_m?.length || 0;
  const plannedPathCount = scene.planned_path_is_test_path ? 0 : (scene.planned_path?.points_local_ned_m?.length || 0);
  const hasSceneContent = Boolean(point3(drone.position_local_ned_m) || obsCount || mapCount || targetCount || testPathCount || plannedPathCount);
  els.map3dEmpty.hidden = Boolean(state.frame);
  els.map3dEmpty.textContent = hasSceneContent
    ? ''
    : 'No aligned telemetry, gates, or path data for this frame.';
  els.map3dMeta.textContent = `frame ${state.frame.index + 1}/${state.frame.count} | cycle ${scene.cycle ?? 'n/a'} | local NED / body FRD`;
  els.map3dCounts.textContent = `${obsCount} observation gates | ${mapCount} mapped gates | ${targetCount} selected gates | ${testPathCount} test path points | ${plannedPathCount} planned path points`;
  resizeMap3d();
}

function applyDefault3dPathVisibility() {
  if (state.plannedPathVisibilityUserSet || !els.show3dPlannedPath) return;
  const testPathPoints = state.frame?.scene?.test_path?.points_local_ned_m;
  const hasTestPath = Array.isArray(testPathPoints) && testPathPoints.map(point3).filter(Boolean).length >= 2;
  state.settings.show3dPlannedPath = !hasTestPath;
  els.show3dPlannedPath.checked = state.settings.show3dPlannedPath;
}

function addTrail(scene, points) {
  if (!state.settings.show3dTrail) return;
  const series = state.frame?.telemetry_series;
  const positions = Array.isArray(series?.position) ? series.position.map(point3).filter(Boolean) : [];
  if (positions.length < 2) return;
  points.push(...positions);
  map3d.root.add(makeLine(positions, 0x64748b, 0.55));
}

function addGridLayer() {
  if (!state.settings.show3dGrid) return;
  const grid = new THREE.GridHelper(80, 40, 0x3a4650, 0x1b232a);
  grid.material.transparent = true;
  grid.material.opacity = 0.42;
  map3d.root.add(grid);
}

function addPathLayer(plannedPath, points, { enabled, color, tubeRadius, colorByCurvature = false }) {
  if (!enabled) return;
  const pathPoints = Array.isArray(plannedPath?.points_local_ned_m)
    ? plannedPath.points_local_ned_m.map(point3).filter(Boolean)
    : [];
  if (pathPoints.length < 2) return;
  points.push(...pathPoints);
  if (colorByCurvature) {
    addCurvatureColoredPath(pathPoints, tubeRadius);
    return;
  }
  map3d.root.add(makeLine(pathPoints, color, 1));
  const tube = makeTube(pathPoints, color, tubeRadius);
  if (tube) map3d.root.add(tube);
}

function addCurvatureColoredPath(pathPoints, tubeRadius) {
  const segmentCurvatures = curvatureBySegment(pathPoints);
  for (let i = 0; i < pathPoints.length - 1; i += 1) {
    const segment = [pathPoints[i], pathPoints[i + 1]];
    const color = curvatureColor(segmentCurvatures[i]);
    map3d.root.add(makeLine(segment, color, 1));
    const tube = makeTube(segment, color, tubeRadius);
    if (tube) map3d.root.add(tube);
  }
}

function addLookaheadVector(dronePosition, points) {
  if (!state.settings.show3dLookahead) return;
  const lookahead = lookaheadPoint();
  if (!lookahead) return;
  const offset = subVec3(lookahead, dronePosition);
  if (lengthVec3(offset) < 1e-6) return;
  points.push(lookahead);
  map3d.root.add(makeDashedLine([dronePosition, lookahead], 0xffd45a, 0.96));
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 16, 10),
    new THREE.MeshBasicMaterial({ color: 0xffd45a })
  );
  marker.position.copy(nedToThree(lookahead));
  map3d.root.add(marker);
  addLabel('lookahead', addVec3(lookahead, [0, 0, -0.35]), 0xffd45a);
}

function addTargetLine(scene, dronePosition, points) {
  if (!state.settings.show3dTargetLine) return;
  const targetPosition = selectedGatePosition(scene?.target_gate, dronePosition);
  if (!targetPosition) return;
  const offset = subVec3(targetPosition, dronePosition);
  if (lengthVec3(offset) < 1e-6) return;
  points.push(targetPosition);
  map3d.root.add(makeDashedLine([dronePosition, targetPosition], 0xffd45a, 0.9));
  addLabel('target', midpointVec3(dronePosition, targetPosition), 0xffd45a);
}

function lookaheadPoint() {
  const follower = state.frame?.telemetry?.geometric_path_follower || {};
  return point3(follower.path_follower?.preview_position_local_ned_m)
    || point3(follower.preview_position_local_ned_m)
    || point3(state.frame?.telemetry?.carrot?.position_local_ned_m);
}

function addDesiredAccelerationVectors(dronePosition, points) {
  addDesiredAccelerationVector(
    dronePosition,
    points,
    currentControllerDesiredAccelerationVector(),
    {
      enabled: state.settings.show3dCurrentDesiredAcceleration,
      color: 0xff8a3d,
      label: 'a_cur',
    },
  );
  addDesiredAccelerationVector(
    dronePosition,
    points,
    autipilotDesiredAccelerationVector(),
    {
      enabled: state.settings.show3dAutipilotDesiredAcceleration,
      color: 0xc084fc,
      label: 'a_auti',
    },
  );
}

function addDesiredAccelerationVector(dronePosition, points, desiredAcceleration, options) {
  if (!options.enabled) return;
  if (!desiredAcceleration) return;
  const accelerationNorm = lengthVec3(desiredAcceleration);
  if (accelerationNorm < 1e-6) return;
  const direction = normalizeVec3(desiredAcceleration);
  const length = clamp(accelerationNorm * 0.14, 0.35, 4.0);
  const tip = addVec3(dronePosition, scaleVec3(direction, length));
  points.push(tip);
  map3d.root.add(makeArrowFromLocal(dronePosition, direction, length, options.color, options.label));
}

function currentControllerDesiredAccelerationVector() {
  return point3(state.frame?.telemetry?.geometric_path_follower?.desired_acceleration_local_ned_mps2);
}

function autipilotDesiredAccelerationVector() {
  return point3(state.frame?.telemetry?.autipilot?.desired_acceleration_local_ned_mps2);
}

function addVelocityVector(dronePosition, points) {
  if (!state.settings.show3dVelocity) return;
  const velocity = point3(state.frame?.telemetry?.telemetry?.velocity_local_ned_mps);
  if (!velocity) return;
  const speed = lengthVec3(velocity);
  if (speed < 1e-6) return;
  const direction = normalizeVec3(velocity);
  const length = clamp(speed * 0.18, 0.35, 5.0);
  const tip = addVec3(dronePosition, scaleVec3(direction, length));
  points.push(tip);
  map3d.root.add(makeArrowFromLocal(dronePosition, direction, length, 0x5cf2ff, 'v'));
}

function addObservationGates(scene, dronePosition, droneQuaternion, points) {
  if (!state.settings.show3dObservations) return;
  const gates = Array.isArray(scene.observation_gates) ? scene.observation_gates : [];
  const bodyToLocal = rotationMatrixFromQuaternion(droneQuaternion);
  gates.forEach((gate, index) => {
    const center = observationGateLocalPosition(gate, dronePosition, bodyToLocal);
    if (!center) return;
    const color = gateColor(index);
    const orientation = point3(gate.orientation_xyz);
    const orientationQuaternion = quat4(gate.orientation_local_ned_quat || gate.orientation_quat);
    let normal = normalizeVec3(subVec3(center, dronePosition));
    let horizontal = null;
    let vertical = null;
    if (orientation) {
      const gateToLocal = multiplyMatrix3(
        multiplyMatrix3(bodyToLocal, cameraFrdToBodyFrdMatrix()),
        rotationMatrixFromRpyDeg(orientation[0], orientation[1], orientation[2])
      );
      normal = normalizeVec3(column3(gateToLocal, 0));
      horizontal = normalizeVec3(column3(gateToLocal, 1));
      vertical = normalizeVec3(scaleVec3(column3(gateToLocal, 2), -1));
    } else if (orientationQuaternion) {
      const gateToLocal = rotationMatrixFromQuaternion(orientationQuaternion);
      normal = normalizeVec3(column3(gateToLocal, 0));
      horizontal = normalizeVec3(column3(gateToLocal, 1));
      vertical = normalizeVec3(scaleVec3(column3(gateToLocal, 2), -1));
    }
    points.push(center);
    if (orientation || orientationQuaternion) {
      addGateFrame(center, normal, color, gate.id || `obs-${index + 1}`, 2.7, 1.5, horizontal, vertical, { highlighted: state.hoveredGateIndex === index });
    } else {
      addGateCenter(center, color, gate.id || `obs-${index + 1}`, { highlighted: state.hoveredGateIndex === index });
    }
  });
}

function addGateMap(gates, dronePosition, points) {
  if (!state.settings.show3dGateMap) return;
  gates.forEach((gate, index) => {
    let position = point3(gate.position_local_ned_m);
    const relative = point3(gate.position_relative_ned_m);
    if (!position && relative) position = addVec3(dronePosition, relative);
    if (!position) return;
    points.push(position);
    const quaternion = quat4(gate.quaternion);
    const label = gate.id || `map-${index + 1}`;
    if (!quaternion) {
      addGateCenter(position, '#ffffff', label);
      return;
    }
    const rotation = rotationMatrixFromQuaternion(quaternion);
    const normal = normalizeVec3(column3(rotation, 0));
    const horizontal = normalizeVec3(column3(rotation, 1));
    const vertical = normalizeVec3(scaleVec3(column3(rotation, 2), -1));
    addGateFrame(position, normal, '#ffffff', label, gate.outer_width_m || 2.7, gate.inner_width_m || 1.5, horizontal, vertical);
  });
}

function addTargetGates(scene, dronePosition, points) {
  if (!state.settings.show3dTargets) return;
  selectedGateEntries(scene).forEach(({ role, gate }) => {
    let position = selectedGatePosition(gate, dronePosition);
    if (!position) return;
    points.push(position);
    const color = role === 'target' ? '#ffd45a' : '#5cf2ff';
    const label = `${role}:${gate.id || gate.sequence || 'gate'}`;
    const quaternion = quat4(gate.quaternion);
    if (!quaternion) {
      addGateCenter(position, color, label, { highlighted: true, ring: true });
      return;
    }
    const rotation = rotationMatrixFromQuaternion(quaternion);
    const normal = normalizeVec3(column3(rotation, 0));
    const horizontal = normalizeVec3(column3(rotation, 1));
    const vertical = normalizeVec3(scaleVec3(column3(rotation, 2), -1));
    addGateFrame(position, normal, color, label, gate.outer_width_m || 2.7, gate.inner_width_m || 1.5, horizontal, vertical, { highlighted: true });
  });
}

function selectedGatePosition(gate, dronePosition) {
  if (!gate) return null;
  const position = point3(gate.position_local_ned_m);
  if (position) return position;
  const relative = point3(gate.position_relative_ned_m);
  return relative ? addVec3(dronePosition, relative) : null;
}

function addDrone(position, quaternion) {
  if (!state.settings.show3dDrone) return;
  const color = 0xffd45a;
  const bodyToLocal = rotationMatrixFromQuaternion(quaternion);
  const center = nedToThree(position);
  const body = new THREE.Group();
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 20, 12),
    new THREE.MeshStandardMaterial({ color, emissive: 0x332600, roughness: 0.35 })
  );
  body.add(sphere);
  body.add(makeArrowFromLocal(position, applyMatrix3(bodyToLocal, [1, 0, 0]), 1.25, 0xffd45a, 'F'));
  body.add(makeArrowFromLocal(position, applyMatrix3(bodyToLocal, [0, 1, 0]), 0.75, 0xff6048, 'R'));
  body.add(makeArrowFromLocal(position, applyMatrix3(bodyToLocal, [0, 0, 1]), 0.75, 0x71e989, 'D'));
  body.position.copy(center);
  body.children.forEach((child) => {
    if (child.userData.worldAnchored) child.position.sub(center);
  });
  map3d.root.add(body);
}

function addGateFrame(center, normal, colorCss, label, outerSize, innerSize, horizontalAxis = null, verticalAxis = null, options = {}) {
  const color = new THREE.Color(colorCss);
  const normalVec = normalizeVec3(normal);
  let vertical = verticalAxis ? normalizeVec3(verticalAxis) : rejectVector([0, 0, -1], normalVec);
  if (lengthVec3(vertical) < 1e-6) vertical = rejectVector([0, 1, 0], normalVec);
  vertical = normalizeVec3(vertical);
  const horizontal = horizontalAxis ? normalizeVec3(horizontalAxis) : normalizeVec3(crossVec3(vertical, normalVec));
  const outer = squarePoints(center, horizontal, vertical, outerSize);
  const inner = squarePoints(center, horizontal, vertical, innerSize);
  if (options.highlighted) {
    const highlight = squarePoints(center, horizontal, vertical, outerSize * 1.08);
    map3d.root.add(makeLine(highlight, 0xffd45a, 0.92));
  }
  map3d.root.add(makeLine(outer, color.getHex(), 1));
  map3d.root.add(makeLine(inner, color.getHex(), 0.72));
  map3d.root.add(makeArrowFromLocal(center, normalVec, 1.4, 0xffd45a, 'dir'));
  map3d.root.add(makeArrowFromLocal(center, vertical, 0.9, 0x71e989, 'up'));
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(options.highlighted ? 0.13 : 0.08, 12, 8),
    new THREE.MeshBasicMaterial({ color: options.highlighted ? 0xffd45a : color })
  );
  marker.position.copy(nedToThree(center));
  map3d.root.add(marker);
  addLabel(label, addVec3(center, scaleVec3(vertical, outerSize * 0.58)), color.getHex());
}

function addGateCenter(center, colorCss, label, options = {}) {
  const color = new THREE.Color(colorCss);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(options.highlighted ? 0.18 : 0.11, 16, 10),
    new THREE.MeshBasicMaterial({ color: options.highlighted ? 0xffd45a : color })
  );
  marker.position.copy(nedToThree(center));
  map3d.root.add(marker);
  if (options.ring) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.18, 0.24, 24),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.88 })
    );
    ring.position.copy(nedToThree(center));
    map3d.root.add(ring);
  }
  addLabel(label, addVec3(center, [0, 0, -0.35]), options.highlighted ? 0xffd45a : color.getHex());
}

function addWorldAxes() {
  if (!state.settings.show3dAxes) return;
  const origin = [0, 0, 0];
  map3d.root.add(makeArrowFromLocal(origin, [1, 0, 0], 4, 0x5cf2ff, 'N'));
  map3d.root.add(makeArrowFromLocal(origin, [0, 1, 0], 4, 0xff6048, 'E'));
  map3d.root.add(makeArrowFromLocal(origin, [0, 0, 1], 4, 0x71e989, 'D'));
}

function makeLine(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(nedToThree));
  const material = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  return new THREE.Line(geometry, material);
}

function makeDashedLine(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(nedToThree));
  const material = new THREE.LineDashedMaterial({
    color,
    dashSize: 0.32,
    gapSize: 0.18,
    transparent: opacity < 1,
    opacity
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  return line;
}

function makeTube(points, color, radius) {
  const curvePoints = points.map(nedToThree);
  if (curvePoints.length < 2) return null;
  const curve = new THREE.CatmullRomCurve3(curvePoints);
  const geometry = new THREE.TubeGeometry(curve, Math.max(8, curvePoints.length * 2), radius, 8, false);
  const material = new THREE.MeshStandardMaterial({ color, emissive: 0x111111, roughness: 0.3 });
  return new THREE.Mesh(geometry, material);
}

function makeArrowFromLocal(origin, direction, length, color, label) {
  const dir = normalizeVec3(direction);
  const arrow = new THREE.ArrowHelper(
    normalizeThree(nedVectorToThree(dir)),
    nedToThree(origin),
    length,
    color,
    Math.min(0.28, length * 0.24),
    Math.min(0.13, length * 0.1)
  );
  arrow.userData.worldAnchored = true;
  if (label) addLabel(label, addVec3(origin, scaleVec3(dir, length + 0.28)), color);
  return arrow;
}

function addLabel(text, position, color) {
  if (!state.settings.show3dLabels) return;
  if (!map3d.scene) return;
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = 256;
  canvas.height = 64;
  context.font = '24px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  context.fillStyle = 'rgba(8, 11, 14, .78)';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = `#${new THREE.Color(color).getHexString()}`;
  context.fillText(String(text).slice(0, 18), 12, 40);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.position.copy(nedToThree(position));
  sprite.scale.set(1.9, 0.48, 1);
  map3d.root.add(sprite);
  map3d.labels.push(sprite);
}

function fitCameraToPoints(points, force = false) {
  if (!force && map3d.controls?.userData?.fitted) return;
  const valid = points.map(point3).filter(Boolean);
  if (!valid.length) return;
  const box = new THREE.Box3();
  valid.forEach((point) => box.expandByPoint(nedToThree(point)));
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 6);
  map3d.controls.target.copy(center);
  map3d.camera.position.copy(center.clone().add(new THREE.Vector3(-radius * 0.9, radius * 0.65, radius * 1.15)));
  map3d.camera.near = Math.max(0.01, radius / 1000);
  map3d.camera.far = Math.max(1000, radius * 120);
  map3d.camera.updateProjectionMatrix();
  map3d.referenceDistance = Math.max(1, map3d.camera.position.distanceTo(map3d.controls.target));
  configureMap3dControls();
  map3d.controls.userData.fitted = true;
  map3d.controls.update();
}

function configureMap3dControls() {
  if (!map3d.controls) return;
  map3d.controls.rotateSpeed = 0.85;
  map3d.controls.panSpeed = 1.0;
  map3d.controls.zoomSpeed = 0.9;
  map3d.controls.minDistance = 1.0;
  map3d.controls.maxDistance = 350.0;
  if ('zoomToCursor' in map3d.controls) {
    map3d.controls.zoomToCursor = true;
  }
  updateMap3dControlSensitivity();
}

function updateMap3dControlSensitivity() {
  if (!map3d.controls || !map3d.camera) return;
  const currentDistance = Math.max(0.001, map3d.camera.position.distanceTo(map3d.controls.target));
  const referenceDistance = Math.max(1, map3d.referenceDistance || 20);
  const speedScale = clamp(referenceDistance / currentDistance, 0.12, 8.0);
  map3d.controls.panSpeed = speedScale;
  map3d.controls.zoomSpeed = clamp(speedScale, 0.35, 3.0);
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.traverse?.((node) => {
      node.geometry?.dispose?.();
      node.material?.dispose?.();
    });
  }
}

function nedToThree(point) {
  const origin = point3(map3d.originNed) || [0, 0, 0];
  const local = subVec3(point, origin);
  return new THREE.Vector3(local[0], -local[2], local[1]);
}

function nedVectorToThree(vector) {
  return new THREE.Vector3(vector[0], -vector[2], vector[1]);
}

function normalizeThree(vector) {
  return vector.lengthSq() > 1e-12 ? vector.normalize() : new THREE.Vector3(1, 0, 0);
}

function curvatureBySegment(points) {
  const vertexCurvatures = points.map(() => 0);
  for (let i = 1; i < points.length - 1; i += 1) {
    const previous = subVec3(points[i], points[i - 1]);
    const next = subVec3(points[i + 1], points[i]);
    const previousLength = lengthVec3(previous);
    const nextLength = lengthVec3(next);
    if (previousLength < 1e-6 || nextLength < 1e-6) continue;
    const cosine = clamp(dotVec3(previous, next) / (previousLength * nextLength), -1, 1);
    const turnAngleRad = Math.acos(cosine);
    vertexCurvatures[i] = turnAngleRad / Math.max((previousLength + nextLength) * 0.5, 1e-6);
  }
  const segmentCurvatures = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    segmentCurvatures.push(Math.max(vertexCurvatures[i], vertexCurvatures[i + 1]));
  }
  return segmentCurvatures;
}

function curvatureColor(curvature) {
  const tightness = clamp(Number(curvature) / 0.55, 0, 1);
  const stops = [
    new THREE.Color(0x71e989),
    new THREE.Color(0xffd45a),
    new THREE.Color(0xff6048)
  ];
  const scaled = tightness * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const localT = scaled - index;
  return stops[index].clone().lerp(stops[index + 1], localT).getHex();
}

function cameraOpticalToBodyFrd(vector) {
  const opticalBody = [vector[2], vector[0], -vector[1]];
  const tilt = 20 * Math.PI / 180;
  const cos = Math.cos(tilt);
  const sin = Math.sin(tilt);
  return [
    opticalBody[0] * cos + opticalBody[2] * sin,
    opticalBody[1],
    -opticalBody[0] * sin + opticalBody[2] * cos
  ];
}

function bodyFrdToCameraOptical(vector) {
  return applyMatrix3(transposeMatrix3(cameraOpticalToBodyFrdMatrix()), vector);
}

function cameraFrdToBodyFrdMatrix() {
  const cameraOpticalFromCameraFrd = [
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0]
  ];
  return multiplyMatrix3(cameraOpticalToBodyFrdMatrix(), cameraOpticalFromCameraFrd);
}

function cameraOpticalToBodyFrdMatrix() {
  const tilt = 20 * Math.PI / 180;
  const cos = Math.cos(tilt);
  const sin = Math.sin(tilt);
  const opticalToBody = [
    [0, 0, 1],
    [1, 0, 0],
    [0, -1, 0]
  ];
  const pitchUp = [
    [cos, 0, sin],
    [0, 1, 0],
    [-sin, 0, cos]
  ];
  return multiplyMatrix3(pitchUp, opticalToBody);
}

function rotationMatrixFromRpyDeg(rollDeg, pitchDeg, yawDeg) {
  return rotationMatrixFromQuaternion(quaternionFromRpyDeg(rollDeg, pitchDeg, yawDeg));
}

function quaternionFromRpyDeg(rollDeg, pitchDeg, yawDeg) {
  const roll = Number(rollDeg) * Math.PI / 180;
  const pitch = Number(pitchDeg) * Math.PI / 180;
  const yaw = Number(yawDeg) * Math.PI / 180;
  const cr = Math.cos(roll / 2);
  const sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2);
  const sp = Math.sin(pitch / 2);
  const cy = Math.cos(yaw / 2);
  const sy = Math.sin(yaw / 2);
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy
  ];
}

function rotationMatrixFromQuaternion(quaternion) {
  const q = quat4(quaternion) || [1, 0, 0, 0];
  const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  const [qw, qx, qy, qz] = q.map((value) => value / norm);
  return [
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
    [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
    [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
  ];
}

function applyMatrix3(matrix, vector) {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
  ];
}

function transposeMatrix3(matrix) {
  return [
    [matrix[0][0], matrix[1][0], matrix[2][0]],
    [matrix[0][1], matrix[1][1], matrix[2][1]],
    [matrix[0][2], matrix[1][2], matrix[2][2]]
  ];
}

function multiplyMatrix3(a, b) {
  return [
    [
      a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
      a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
      a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]
    ],
    [
      a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
      a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
      a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]
    ],
    [
      a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
      a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
      a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]
    ]
  ];
}

function column3(matrix, index) {
  return [matrix[0][index], matrix[1][index], matrix[2][index]];
}

function squarePoints(center, horizontal, vertical, size) {
  const half = Number(size) / 2;
  return [
    addVec3(center, addVec3(scaleVec3(horizontal, -half), scaleVec3(vertical, -half))),
    addVec3(center, addVec3(scaleVec3(horizontal, half), scaleVec3(vertical, -half))),
    addVec3(center, addVec3(scaleVec3(horizontal, half), scaleVec3(vertical, half))),
    addVec3(center, addVec3(scaleVec3(horizontal, -half), scaleVec3(vertical, half))),
    addVec3(center, addVec3(scaleVec3(horizontal, -half), scaleVec3(vertical, -half)))
  ];
}

function point3(value) {
  return Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every((item) => Number.isFinite(Number(item)))
    ? [Number(value[0]), Number(value[1]), Number(value[2])]
    : null;
}

function quat4(value) {
  return Array.isArray(value) && value.length >= 4 && value.slice(0, 4).every((item) => Number.isFinite(Number(item)))
    ? [Number(value[0]), Number(value[1]), Number(value[2]), Number(value[3])]
    : null;
}

function addVec3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subVec3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scaleVec3(vector, scale) {
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}

function lengthVec3(vector) {
  return Math.hypot(vector[0], vector[1], vector[2]);
}

function normalizeVec3(vector) {
  const length = lengthVec3(vector);
  return length > 1e-12 ? scaleVec3(vector, 1 / length) : [1, 0, 0];
}

function crossVec3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function dotVec3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function rejectVector(vector, normal) {
  return addVec3(vector, scaleVec3(normal, -dotVec3(vector, normal)));
}

function drawGateObservation(ctx, canvas, gate, index, alpha) {
  const cameraPosition = observationGateCameraPosition(gate);
  const projected = projectPoint(cameraPosition, canvas.width, canvas.height);
  if (!projected) return;
  const color = gateColor(index);
  const z = Number(cameraPosition?.[2]);
  const intrinsics = scaledCameraIntrinsics(canvas.width, canvas.height);
  const halfX = (intrinsics.fx * state.settings.gateSizeM / Math.max(z, 0.001)) / 2;
  const halfY = (intrinsics.fy * state.settings.gateSizeM / Math.max(z, 0.001)) / 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  const highlighted = state.hoveredGateIndex === index;
  ctx.lineWidth = highlighted ? 4 : 2;
  ctx.strokeStyle = highlighted ? '#ffd45a' : color;
  ctx.fillStyle = highlighted ? '#ffd45a' : color;
  const hasOrientation = gateHasOrientation(gate);
  if (hasOrientation && state.settings.showGateBoxes) {
    ctx.strokeRect(projected.x - halfX, projected.y - halfY, halfX * 2, halfY * 2);
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(projected.x - halfX * 0.56, projected.y - halfY * 0.56, halfX * 1.12, halfY * 1.12);
    ctx.setLineDash([]);
  }
  if (state.settings.showCenters || !hasOrientation) {
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(projected.x - 14, projected.y);
    ctx.lineTo(projected.x + 14, projected.y);
    ctx.moveTo(projected.x, projected.y - 14);
    ctx.lineTo(projected.x, projected.y + 14);
    ctx.stroke();
  }
  if (hasOrientation && highlighted && state.settings.showGateBoxes) {
    ctx.globalAlpha = Math.min(1, alpha + 0.08);
    ctx.setLineDash([10, 6]);
    ctx.strokeRect(projected.x - halfX * 1.08, projected.y - halfY * 1.08, halfX * 2.16, halfY * 2.16);
    ctx.setLineDash([]);
  }
  if (state.settings.showLabels) {
    const text = `${gate.id || `gate-${index + 1}`} ${formatNumber(gate.position_confidence)} z=${formatNumber(z)}m`;
    ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    const metrics = ctx.measureText(text);
    const labelX = Math.min(Math.max(projected.x + 9, 4), canvas.width - metrics.width - 12);
    const labelY = Math.max(projected.y - halfY - 10, 18);
    ctx.fillStyle = 'rgba(0,0,0,.72)';
    ctx.fillRect(labelX - 4, labelY - 14, metrics.width + 8, 19);
    ctx.fillStyle = color;
    ctx.fillText(text, labelX, labelY);
  }
  ctx.restore();
}

function drawSelectedGateOverlay(ctx, canvas, entry, alpha) {
  const cameraPosition = observationGateCameraPosition(entry.gate);
  const projected = projectPoint(cameraPosition, canvas.width, canvas.height);
  if (!projected) return;
  const color = entry.role === 'target' ? '#ffd45a' : '#5cf2ff';
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineWidth = entry.role === 'target' ? 3 : 2;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(projected.x, projected.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(projected.x - 14, projected.y);
  ctx.lineTo(projected.x + 14, projected.y);
  ctx.moveTo(projected.x, projected.y - 14);
  ctx.lineTo(projected.x, projected.y + 14);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(projected.x, projected.y, entry.role === 'target' ? 22 : 18, 0, Math.PI * 2);
  ctx.stroke();
  if (state.settings.showLabels) {
    const text = `${entry.role} ${entry.gate.id || entry.gate.sequence || ''}`.trim();
    ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    const metrics = ctx.measureText(text);
    const labelX = Math.min(Math.max(projected.x + 12, 4), canvas.width - metrics.width - 12);
    const labelY = Math.max(projected.y - 24, 18);
    ctx.fillStyle = 'rgba(0,0,0,.72)';
    ctx.fillRect(labelX - 4, labelY - 14, metrics.width + 8, 19);
    ctx.fillStyle = color;
    ctx.fillText(text, labelX, labelY);
  }
  ctx.restore();
}

function selectedGateEntries(scene = state.frame?.scene) {
  const source = scene || {};
  const frame = state.frame || {};
  const entries = [];
  const targetGate = source.target_gate || frame.target_gate;
  const nextGate = source.next_gate || frame.next_gate;
  if (targetGate) entries.push({ role: 'target', gate: targetGate });
  if (nextGate) entries.push({ role: 'next', gate: nextGate });
  return entries;
}

function midpointVec3(a, b) {
  return [
    (a[0] + b[0]) / 2,
    (a[1] + b[1]) / 2,
    (a[2] + b[2]) / 2
  ];
}

function gateHasOrientation(gate) {
  return Boolean(point3(gate?.orientation_xyz) || quat4(gate?.orientation_local_ned_quat || gate?.orientation_quat));
}

function observationGateCameraPosition(gate) {
  const cameraPosition = point3(gate?.position_xyz);
  if (cameraPosition) return cameraPosition;
  const scene = state.frame?.scene || {};
  const drone = scene.drone || {};
  const dronePosition = point3(drone.position_local_ned_m) || [0, 0, 0];
  const droneQuaternion = quat4(drone.attitude_quaternion) || [1, 0, 0, 0];
  const bodyToLocal = rotationMatrixFromQuaternion(droneQuaternion);
  const localPosition = point3(gate?.position_local_ned_m || gate?.position_local_ned);
  const localRelative = localPosition
    ? subVec3(localPosition, dronePosition)
    : point3(gate?.position_relative_ned_m);
  if (!localRelative) return null;
  const bodyRelative = applyMatrix3(transposeMatrix3(bodyToLocal), localRelative);
  return bodyFrdToCameraOptical(bodyRelative);
}

function observationGateLocalPosition(gate, dronePosition, bodyToLocal) {
  const localPosition = point3(gate?.position_local_ned_m || gate?.position_local_ned);
  if (localPosition) return localPosition;
  const localRelative = point3(gate?.position_relative_ned_m);
  if (localRelative) return addVec3(dronePosition, localRelative);
  const cameraPosition = point3(gate?.position_xyz);
  if (!cameraPosition) return null;
  return addVec3(dronePosition, applyMatrix3(bodyToLocal, cameraOpticalToBodyFrd(cameraPosition)));
}

function projectPoint(position, width, height) {
  if (!Array.isArray(position) || position.length < 3) return null;
  const right = Number(position[0]);
  const up = Number(position[1]);
  const forward = Number(position[2]);
  if (!Number.isFinite(right) || !Number.isFinite(up) || !Number.isFinite(forward) || forward <= 0.001) return null;
  const intrinsics = scaledCameraIntrinsics(width, height);
  return {
    x: intrinsics.cx + intrinsics.fx * right / forward,
    y: intrinsics.cy - intrinsics.fy * up / forward
  };
}

function scaledCameraIntrinsics(width, height) {
  const scaleX = width / VISION_CAMERA.widthPx;
  const scaleY = height / VISION_CAMERA.heightPx;
  const fy = height / (2 * Math.tan((state.settings.verticalFovDeg * Math.PI / 180) / 2));
  return {
    fx: VISION_CAMERA.fx * scaleX,
    fy,
    cx: VISION_CAMERA.cx * scaleX,
    cy: VISION_CAMERA.cy * scaleY
  };
}

function gateColor(index) {
  return ['#5cf2ff', '#ffd45a', '#71e989', '#ff6048', '#b68cff'][index % 5];
}

function formatVec(value, suffix) {
  if (!Array.isArray(value) || value.length < 3) return 'n/a';
  return `${formatNumber(value[0])}, ${formatNumber(value[1])}, ${formatNumber(value[2])}${suffix ? ` ${suffix}` : ''}`;
}

function formatQuat(value) {
  if (!Array.isArray(value) || value.length < 4) return 'n/a';
  return `${formatNumber(value[0])}, ${formatNumber(value[1])}, ${formatNumber(value[2])}, ${formatNumber(value[3])}`;
}

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : 'n/a';
}

function map(value, inMin, inMax, outMin, outMax) {
  if (Math.abs(inMax - inMin) < 1e-12) return (outMin + outMax) / 2;
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, Number(value)));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

function installEvents() {
  syncFovControl();
  syncZoomControl();
  els.viewTabs.forEach((button) => {
    button.addEventListener('click', () => {
      setViewMode(button.dataset.view);
    });
  });
  els.runSelect.addEventListener('change', async () => {
    state.runPath = els.runSelect.value;
    state.frameIndex = 0;
    await loadRun();
  });
  els.refreshButton.addEventListener('click', () => loadRuns());
  els.frameSlider.addEventListener('input', () => {
    els.frameReadout.textContent = `Frame ${Number(els.frameSlider.value) + 1}/${state.frame?.count || '-'}`;
  });
  els.frameSlider.addEventListener('change', () => loadFrame(Number(els.frameSlider.value)));
  els.prevButton.addEventListener('click', () => loadFrame(state.frameIndex - 1));
  els.nextButton.addEventListener('click', () => loadFrame(state.frameIndex + 1));
  els.playButton.addEventListener('click', togglePlayback);
  document.querySelectorAll('.modeButton').forEach((button) => {
    button.addEventListener('click', () => {
      setInspectorMode(button.dataset.mode);
    });
  });
  document.querySelectorAll('.plotTab').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.plotTab').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      state.plotMode = button.dataset.plot;
      renderTelemetryPlot();
    });
  });
  for (const [element, key] of [
    [els.showObservations, 'showObservations'],
    [els.showGateBoxes, 'showGateBoxes'],
    [els.showCenters, 'showCenters'],
    [els.showLabels, 'showLabels']
  ]) {
    element.addEventListener('change', () => {
      state.settings[key] = Boolean(element.checked);
      renderOverlay();
    });
  }
  els.overlayOpacity.addEventListener('input', () => {
    state.settings.overlayOpacity = Number(els.overlayOpacity.value) / 100;
    els.overlayOpacityValue.textContent = `${els.overlayOpacity.value}%`;
    renderOverlay();
  });
  els.gateSize.addEventListener('input', () => {
    state.settings.gateSizeM = Number(els.gateSize.value);
    els.gateSizeValue.textContent = `${state.settings.gateSizeM.toFixed(2)}m`;
    renderOverlay();
  });
  els.verticalFov.addEventListener('input', () => {
    state.settings.verticalFovDeg = Number(els.verticalFov.value);
    els.verticalFovValue.textContent = `${state.settings.verticalFovDeg.toFixed(1)}deg`;
    renderOverlay();
  });
  els.zoomSlider.addEventListener('input', () => {
    setZoom(Number(els.zoomSlider.value) / 100);
  });
  els.zoomOutButton.addEventListener('click', () => {
    setZoom(state.settings.zoom - 0.1);
  });
  els.zoomInButton.addEventListener('click', () => {
    setZoom(state.settings.zoom + 0.1);
  });
  els.zoomFitButton.addEventListener('click', () => {
    setZoom(1);
  });
  els.controlsHelpClose.addEventListener('click', () => {
    els.controlsHelp.hidden = true;
  });
  els.jsonPopoutButton.addEventListener('click', () => {
    renderInspector();
    els.jsonPopout.hidden = false;
  });
  els.jsonPopoutClose.addEventListener('click', closeJsonPopout);
  els.jsonPopout.addEventListener('click', (event) => {
    if (event.target === els.jsonPopout) closeJsonPopout();
  });
  els.map3dResetButton.addEventListener('click', () => {
    if (map3d.controls) map3d.controls.userData.fitted = false;
    renderMap3d({ resetCamera: true });
  });
  for (const [element, key] of [
    [els.show3dGrid, 'show3dGrid'],
    [els.show3dAxes, 'show3dAxes'],
    [els.show3dDrone, 'show3dDrone'],
    [els.show3dObservations, 'show3dObservations'],
    [els.show3dGateMap, 'show3dGateMap'],
    [els.show3dTargets, 'show3dTargets'],
    [els.show3dTargetLine, 'show3dTargetLine'],
    [els.show3dTestPath, 'show3dTestPath'],
    [els.show3dPlannedPath, 'show3dPlannedPath'],
    [els.show3dTrail, 'show3dTrail'],
    [els.show3dLookahead, 'show3dLookahead'],
    [els.show3dVelocity, 'show3dVelocity'],
    [els.show3dCurrentDesiredAcceleration, 'show3dCurrentDesiredAcceleration'],
    [els.show3dAutipilotDesiredAcceleration, 'show3dAutipilotDesiredAcceleration'],
    [els.show3dLabels, 'show3dLabels']
  ]) {
    element.addEventListener('change', () => {
      if (key === 'show3dPlannedPath') state.plannedPathVisibilityUserSet = true;
      state.settings[key] = Boolean(element.checked);
      renderMap3d();
    });
  }
  for (const [element, key] of [
    [els.showTelemetryActual, 'showTelemetryActual'],
    [els.showTelemetryTruth, 'showTelemetryTruth']
  ]) {
    element.addEventListener('change', () => {
      state.settings[key] = Boolean(element.checked);
      renderTelemetryDashboard();
    });
  }
  els.frameStage.addEventListener('wheel', (event) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    setZoom(state.settings.zoom + (event.deltaY < 0 ? 0.1 : -0.1));
  }, { passive: false });
  els.frameStage.addEventListener('scroll', saveFrameViewport);
  els.timelineCanvas.addEventListener('click', (event) => {
    const rect = els.timelineCanvas.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    const maxFrame = Math.max((state.frame?.count || 1) - 1, 0);
    loadFrame(Math.round(ratio * maxFrame));
  });
  els.overlayCanvas.addEventListener('mousemove', updateHoverReadout);
  els.telemetryPlot.addEventListener('pointermove', (event) => {
    els.telemetryPlot._compactTelemetryPointer = canvasPixelPointer(els.telemetryPlot, event);
    renderTelemetryPlot();
  });
  els.telemetryPlot.addEventListener('pointerleave', () => {
    els.telemetryPlot._compactTelemetryPointer = null;
    renderTelemetryPlot();
  });
  window.addEventListener('resize', () => {
    syncCanvasToImage();
    renderOverlay();
    resizeMap3d();
    renderTelemetryDashboard();
  });
  window.addEventListener('keydown', (event) => {
    if (event.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) jumpObservedFrame(-1);
      else loadFrame(state.frameIndex - 1);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) jumpObservedFrame(1);
      else loadFrame(state.frameIndex + 1);
    } else if (event.key === ' ') {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      setZoom(state.settings.zoom + 0.1);
    } else if (event.key === '-' || event.key === '_') {
      event.preventDefault();
      setZoom(state.settings.zoom - 0.1);
    } else if (event.key === 'Escape' && !els.jsonPopout.hidden) {
      event.preventDefault();
      closeJsonPopout();
    }
  });
}

function updateHoverReadout(event) {
  const rect = els.overlayCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * els.overlayCanvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * els.overlayCanvas.height;
  const intrinsics = scaledCameraIntrinsics(els.overlayCanvas.width, els.overlayCanvas.height);
  const z = 10;
  const right = ((x - intrinsics.cx) / intrinsics.fx) * z;
  const up = -((y - intrinsics.cy) / intrinsics.fy) * z;
  els.hoverReadout.textContent = `px ${x.toFixed(0)}, ${y.toFixed(0)} | ray @10m right=${right.toFixed(2)} up=${up.toFixed(2)}`;
}

function syncFovControl() {
  if (!els.verticalFov || !els.verticalFovValue) return;
  const value = Number(state.settings.verticalFovDeg.toFixed(1));
  els.verticalFov.value = String(value);
  els.verticalFovValue.textContent = `${value.toFixed(1)}deg`;
  els.verticalFov.title = `Derived from deterministic vision intrinsics: fy=${VISION_CAMERA.fy}px at ${VISION_CAMERA.widthPx}x${VISION_CAMERA.heightPx}.`;
}

function setZoom(value) {
  const next = Math.max(0.5, Math.min(4, Number(value) || 1));
  const before = viewportCenterRatios();
  state.settings.zoom = Number(next.toFixed(2));
  syncZoomControl();
  syncCanvasToImage();
  restoreViewportCenter(before);
  renderOverlay();
}

function syncZoomControl() {
  if (!els.zoomSlider || !els.zoomValue) return;
  const percent = Math.round(state.settings.zoom * 100);
  els.zoomSlider.value = String(percent);
  els.zoomValue.textContent = `${percent}%`;
}

function setInspectorMode(mode) {
  state.inspectorMode = ['telemetry', 'observation', 'raw'].includes(mode) ? mode : 'telemetry';
  document.querySelectorAll('.modeButton').forEach((button) => {
    button.classList.toggle('active', button.dataset.mode === state.inspectorMode);
  });
  renderInspector();
}

function closeJsonPopout() {
  els.jsonPopout.hidden = true;
}

function isFrameViewVisible() {
  return state.viewMode === 'frame' && els.frameStage.clientWidth > 0 && els.frameStage.clientHeight > 0;
}

function saveFrameViewport() {
  if (!els.frameStage) return;
  state.frameViewportState = {
    scrollLeft: els.frameStage.scrollLeft,
    scrollTop: els.frameStage.scrollTop
  };
}

function restoreFrameViewport() {
  const saved = state.frameViewportState || { scrollLeft: 0, scrollTop: 0 };
  const maxLeft = Math.max(0, els.frameStage.scrollWidth - els.frameStage.clientWidth);
  const maxTop = Math.max(0, els.frameStage.scrollHeight - els.frameStage.clientHeight);
  els.frameStage.scrollLeft = Math.max(0, Math.min(maxLeft, saved.scrollLeft));
  els.frameStage.scrollTop = Math.max(0, Math.min(maxTop, saved.scrollTop));
}

function setViewMode(mode) {
  if (state.viewMode === 'frame') saveFrameViewport();
  state.viewMode = ['frame', 'map3d', 'telemetry'].includes(mode) ? mode : 'frame';
  els.viewTabs.forEach((button) => {
    button.classList.toggle('active', button.dataset.view === state.viewMode);
  });
  els.viewPanes.forEach((pane) => {
    pane.classList.toggle('active', pane.dataset.viewPane === state.viewMode);
  });
  if (state.viewMode === 'map3d') {
    initMap3d();
    resizeMap3d();
    renderMap3d();
  } else if (state.viewMode === 'telemetry') {
    renderTelemetryDashboard();
  } else {
    syncCanvasToImage();
    restoreFrameViewport();
    renderOverlay();
  }
}

function viewportCenterRatios() {
  const maxLeft = els.frameStage.scrollWidth - els.frameStage.clientWidth;
  const maxTop = els.frameStage.scrollHeight - els.frameStage.clientHeight;
  return {
    x: maxLeft > 0 ? (els.frameStage.scrollLeft + els.frameStage.clientWidth / 2) / maxLeft : 0.5,
    y: maxTop > 0 ? (els.frameStage.scrollTop + els.frameStage.clientHeight / 2) / maxTop : 0.5
  };
}

function restoreViewportCenter(ratios) {
  const maxLeft = Math.max(0, els.frameStage.scrollWidth - els.frameStage.clientWidth);
  const maxTop = Math.max(0, els.frameStage.scrollHeight - els.frameStage.clientHeight);
  els.frameStage.scrollLeft = Math.max(0, Math.min(maxLeft, ratios.x * Math.max(1, maxLeft) - els.frameStage.clientWidth / 2));
  els.frameStage.scrollTop = Math.max(0, Math.min(maxTop, ratios.y * Math.max(1, maxTop) - els.frameStage.clientHeight / 2));
}

function togglePlayback() {
  state.playing = !state.playing;
  els.playButton.textContent = state.playing ? 'Pause' : 'Play';
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = 0;
  }
  if (state.playing) {
    state.playTimer = setInterval(() => {
      const maxFrame = Math.max((state.frame?.count || 1) - 1, 0);
      loadFrame(state.frameIndex >= maxFrame ? 0 : state.frameIndex + 1);
    }, 120);
  }
}

installEvents();
loadRuns().catch((error) => {
  console.error(error);
  setStatus(`startup failed: ${error.message}`);
});
