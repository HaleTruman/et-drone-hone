import {
  preferredRunId,
  rememberRun,
} from './frame_navigation_state.js?v=1';

const runSelect = document.getElementById('run-select');
const acceptedSelect = document.getElementById('accepted-select');
const previousBatchButton = document.getElementById('previous-batch');
const nextBatchButton = document.getElementById('next-batch');
const batchStatus = document.getElementById('batch-status');
const summary = document.getElementById('standard-gate-summary');
const grid = document.getElementById('layer-grid');
const message = document.getElementById('frame-message');
const RESULT_BATCH_SIZE = 200;

const state = {
  runs: [],
  catalog: null,
  loadVersion: 0,
  imagePromises: new Map(),
  batchIndex: 0,
};

runSelect.addEventListener('change', () => void selectRun(runSelect.value));
acceptedSelect.addEventListener('change', () => {
  state.batchIndex = 0;
  renderSelection();
});
previousBatchButton.addEventListener('click', () => changeBatch(-1));
nextBatchButton.addEventListener('click', () => changeBatch(1));

async function init() {
  try {
    const response = await fetch('/api/runs', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Run catalog fetch failed: ${response.status}`);
    const payload = await response.json();
    state.runs = (Array.isArray(payload.runs) ? payload.runs : [])
      .filter((run) => run.has_logged_frames && run.review_json_count > 0);
    runSelect.replaceChildren(...state.runs.map((run) => new Option(
      `${run.id} · ${run.review_json_count} review JSON`, run.id)));
    if (!state.runs.length) {
      return showMessage('No associated review JSON is available.');
    }
    const newest = state.runs.reduce((latest, run) =>
      run.id.localeCompare(latest.id) > 0 ? run : latest);
    await selectRun(preferredRunId(state.runs, newest.id));
  } catch (error) {
    showMessage(error.message || 'Unable to load StandardGateResult records.');
  }
}

async function selectRun(runId) {
  const loadVersion = ++state.loadVersion;
  state.imagePromises.clear();
  showMessage(`Loading source.run_id=${runId} StandardGateResult records…`);
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/standard-gate-results`,
      { cache: 'no-store' },
    );
    if (!response.ok) {
      throw new Error(`StandardGateResult catalog failed: ${response.status}`);
    }
    const catalog = await response.json();
    if (loadVersion !== state.loadVersion) return;
    if (catalog.version !== 'deterministic-v3.standard-gate-result-catalog.v1'
        || catalog.id !== runId || !Array.isArray(catalog.results)) {
      throw new Error('StandardGateResult catalog contract is invalid.');
    }
    state.catalog = catalog;
    state.batchIndex = 0;
    runSelect.value = runId;
    rememberRun(runId);
    renderSelection();
  } catch (error) {
    if (loadVersion === state.loadVersion) {
      showMessage(error.message || `Unable to load source.run_id=${runId}.`);
    }
  }
}

function renderSelection() {
  const catalog = state.catalog;
  if (!catalog) return;
  const selected = acceptedSelect.value;
  const results = (Array.isArray(catalog.results) ? catalog.results : [])
    .filter((item) => selected === 'all'
      || String(item.StandardGateResult?.accepted) === selected);
  const batchCount = Math.max(1, Math.ceil(results.length / RESULT_BATCH_SIZE));
  state.batchIndex = Math.min(state.batchIndex, batchCount - 1);
  const start = state.batchIndex * RESULT_BATCH_SIZE;
  const end = Math.min(start + RESULT_BATCH_SIZE, results.length);
  const batch = results.slice(start, end);
  const cards = batch.map(buildCard);
  grid.replaceChildren(...cards);
  cards.forEach((card, index) => void populateCard(card, batch[index]));
  grid.hidden = false;
  message.hidden = true;
  previousBatchButton.disabled = state.batchIndex === 0;
  nextBatchButton.disabled = end >= results.length;
  batchStatus.textContent = results.length
    ? `${start + 1}-${end} of ${results.length}` : '0 of 0';
  summary.textContent = [
    `version=${catalog.version}`,
    `rendered=${batch.length}`,
    `matched=${results.length}`,
    `standard_gate_result_count=${catalog.standard_gate_result_count || 0}`,
    `accepted_standard_gate_result_count=${catalog.accepted_standard_gate_result_count || 0}`,
    `high_confidence_standard_gate_result_count=${catalog.high_confidence_standard_gate_result_count || 0}`,
    catalog.missing_frame_observation_count
      ? `missing_frame_observation_count=${catalog.missing_frame_observation_count}` : '',
    catalog.missing_component_observation_count
      ? `missing_component_observation_count=${catalog.missing_component_observation_count}` : '',
  ].filter(Boolean).join(' · ');
  if (!results.length) {
    showMessage(
      `No StandardGateResult.accepted=${selected} records are available.`);
  }
}

function changeBatch(offset) {
  state.batchIndex += offset;
  renderSelection();
  grid.scrollIntoView({ block: 'start' });
}

function buildCard(result) {
  const standard = result.StandardGateResult || {};
  const component = result.ComponentObservation || {};
  const card = document.createElement('article');
  card.className = 'standard-gate-result-card';
  card.dataset.accepted = String(standard.accepted === true);

  const identity = document.createElement('div');
  identity.className = 'standard-gate-result-identity';
  identity.append(
    textElement('strong', [
      `source.run_id=${state.catalog?.id}`,
      `StandardGateResult.frame_id=${standard.frame_id}`,
      `StandardGateResult.sim_time_ns=${standard.sim_time_ns}`,
      `StandardGateResult.component_id=${standard.component_id}`,
    ].join(' · ')),
    textElement('small', [
      `ComponentObservation.bbox_xywh=${formatValue(component.bbox_xywh)}`,
      `source.relative_path=${formatValue(result.source?.relative_path)}`,
      `StandardGateResult.route=${formatValue(standard.route)}`,
      `StandardGateResult.fitter=${formatValue(standard.fitter)}`,
      `StandardGateResult.configuration_version=${formatValue(standard.configuration_version)}`,
    ].join(' · ')),
  );

  const evidence = document.createElement('div');
  evidence.className = 'standard-gate-result-evidence';
  [
    ['StandardGateResult.fit_confidence', standard.fit_confidence],
    ['StandardGateResult.high_confidence_threshold',
      standard.high_confidence_threshold],
    ['StandardGateResult.accepted', standard.accepted],
    ['StandardGateResult.rejection_reason', standard.rejection_reason],
  ].forEach(([field, value]) => evidence.append(fieldElement(field, value)));

  const heading = document.createElement('header');
  heading.className = 'standard-gate-result-heading';
  const headingBody = document.createElement('div');
  headingBody.append(identity, evidence);
  const status = textElement(
    'div', `StandardGateResult.accepted=${formatValue(standard.accepted)}`);
  status.className = 'standard-gate-status';
  heading.append(headingBody, status);

  const layerGrid = document.createElement('div');
  layerGrid.className = 'standard-gate-evidence-grid';
  const layers = layerDefinitions(result);
  layerGrid.replaceChildren(...layers.map(buildLayer));
  card.append(heading, layerGrid);
  return card;
}

function layerDefinitions(result) {
  const standard = result.StandardGateResult || {};
  const layers = result.layers || {};
  return [
    {
      id: 'FrameObservation.closed_mask',
      detail: [
        'owner=src/preprocessing.py::preprocess_frame',
        'isolation=ui/backend/serve_review_ui.py::render_standard_gate_component_layer',
        'ComponentObservation.component_id',
      ].join(' · '),
      url: layers['FrameObservation.closed_mask'],
    },
    {
      id: 'StandardGateResult.fitted_corners_uv',
      detail: [
        'owner=src/standard_gate_processing/quadrilateral_fitter.py::fit_standard_quadrilateral',
        'source=ui/backend/replay_historic_run.py::HistoricRunSource',
        'render=ui/frontend/standard_gate_app.js::drawComponentCrop',
        `StandardGateResult.fit_confidence=${formatValue(standard.fit_confidence)}`,
      ].join(' · '),
      canvas: 'fit',
    },
  ];
}

function buildLayer(layer) {
  const panel = document.createElement('figure');
  panel.className = 'standard-gate-layer';
  panel.dataset.layerId = layer.id;
  if (layer.canvas) {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    canvas.dataset.canvasKind = layer.canvas;
    panel.append(canvas);
  } else if (layer.url) {
    const image = new Image();
    image.alt = layer.id;
    image.decoding = 'async';
    image.addEventListener('error', () => panel.classList.add('load-error'));
    image.addEventListener('load', () => panel.classList.remove('load-error'));
    image.src = layer.url;
    panel.append(image);
  } else {
    panel.classList.add('unavailable');
  }
  const label = document.createElement('figcaption');
  label.className = 'standard-gate-layer-label';
  const detail = textElement('small', layer.detail);
  detail.title = layer.detail;
  label.append(
    textElement('strong', layer.id),
    detail,
  );
  panel.append(label);
  return panel;
}

async function populateCard(card, result) {
  const fitCanvas = card.querySelector('canvas[data-canvas-kind="fit"]');
  const corners = result.StandardGateResult?.fitted_corners_uv;
  if (!fitCanvas || !validCorners(corners)) {
    if (fitCanvas) fitCanvas.parentElement.classList.add('unavailable');
    return;
  }
  const sourceUrl = result.frame?.image_url;
  const bbox = result.ComponentObservation?.bbox_xywh;
  if (!sourceUrl || !validBbox(bbox)) {
    fitCanvas.parentElement.classList.add('unavailable');
    return;
  }
  try {
    const image = await loadImage(sourceUrl);
    drawComponentCrop(fitCanvas, image, bbox, corners);
  } catch (_error) {
    fitCanvas.parentElement.classList.add('load-error');
  }
}

function drawComponentCrop(canvas, image, bbox, corners = null) {
  const [x, y, width, height] = bbox.map(Number);
  const context = canvas.getContext('2d', { alpha: false });
  const scale = Math.min(canvas.width / width, canvas.height / height);
  const targetWidth = width * scale;
  const targetHeight = height * scale;
  const targetX = (canvas.width - targetWidth) / 2;
  const targetY = (canvas.height - targetHeight) / 2;
  context.imageSmoothingEnabled = false;
  context.fillStyle = '#000';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(
    image, x, y, width, height,
    targetX, targetY, targetWidth, targetHeight,
  );
  if (!validCorners(corners)) {
    if (corners !== null) canvas.parentElement.classList.add('load-error');
    return;
  }
  context.beginPath();
  corners.forEach(([u, v], index) => {
    const displayX = targetX + (Number(u) - x) * scale;
    const displayY = targetY + (Number(v) - y) * scale;
    if (index === 0) context.moveTo(displayX, displayY);
    else context.lineTo(displayX, displayY);
  });
  context.closePath();
  context.lineWidth = Math.max(2, Math.min(5, scale));
  context.strokeStyle = '#00f2ff';
  context.stroke();
  context.fillStyle = '#ff2dc6';
  corners.forEach(([u, v]) => {
    const displayX = targetX + (Number(u) - x) * scale;
    const displayY = targetY + (Number(v) - y) * scale;
    context.beginPath();
    context.arc(displayX, displayY, 3, 0, Math.PI * 2);
    context.fill();
  });
}

function loadImage(url) {
  if (!state.imagePromises.has(url)) {
    state.imagePromises.set(url, new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = url;
    }));
  }
  return state.imagePromises.get(url);
}

function validBbox(value) {
  return Array.isArray(value) && value.length === 4
    && value.every(Number.isFinite)
    && value[2] > 0 && value[3] > 0;
}

function validCorners(value) {
  return Array.isArray(value) && value.length === 4
    && value.every((point) => Array.isArray(point) && point.length === 2
      && point.every(Number.isFinite));
}

function fieldElement(field, value) {
  const element = document.createElement('div');
  element.append(
    document.createTextNode(`${field}=`),
    Object.assign(document.createElement('code'), {
      textContent: formatValue(value),
    }),
  );
  return element;
}

function textElement(tagName, value) {
  return Object.assign(document.createElement(tagName), { textContent: value });
}

function formatValue(value) {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

function showMessage(text) {
  grid.hidden = true;
  message.hidden = false;
  message.textContent = text;
}

void init();
