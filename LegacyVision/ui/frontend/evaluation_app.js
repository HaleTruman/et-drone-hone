const evaluationSelect = document.getElementById('evaluation-select');
const overlayLayerSelect = document.getElementById('overlay-layer-select');
const frameSlider = document.getElementById('frame-slider');
const frameMeta = document.getElementById('frame-meta');
const previousButton = document.getElementById('previous-frame');
const nextButton = document.getElementById('next-frame');
const changedOnly = document.getElementById('changed-only');
const frameMessage = document.getElementById('frame-message');
const summaryPanel = document.getElementById('evaluation-summary');
const grid = document.getElementById('evaluation-grid');

const backendNames = ['deterministic_v3', 'deterministic_v3_2'];
const state = {
  evaluations: [],
  evaluation: null,
  frames: [],
  filteredIndices: [],
  filteredPosition: 0,
  selectedLayer: null,
};

evaluationSelect.addEventListener('change', () => void selectEvaluation(evaluationSelect.value));
overlayLayerSelect.addEventListener('change', () => {
  state.selectedLayer = overlayLayerSelect.value;
  renderFrame();
});
changedOnly.addEventListener('change', () => {
  applyFilter();
  selectFilteredPosition(0);
});
frameSlider.addEventListener('input', () => selectFilteredPosition(Number(frameSlider.value)));
previousButton.addEventListener('click', () => selectFilteredPosition(state.filteredPosition - 1));
nextButton.addEventListener('click', () => selectFilteredPosition(state.filteredPosition + 1));

async function init() {
  try {
    const response = await fetch('/api/evaluations', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Evaluation catalog fetch failed: ${response.status}`);
    const payload = await response.json();
    state.evaluations = (Array.isArray(payload.evaluations) ? payload.evaluations : [])
      .filter((evaluation) => evaluation.overlays_enabled);
    evaluationSelect.replaceChildren(...state.evaluations.map((evaluation) => new Option(
      [
        evaluation.id,
        evaluation.source_run_id,
        `${evaluation.processed_frame_count || 0} frames`,
        `${evaluation.frames_changed || 0} changed`,
      ].filter(Boolean).join(' - '),
      evaluation.id,
    )));
    if (!state.evaluations.length) {
      return showMessage(
        'No evaluations with overlays are available. Run the comparison with --render-overlays.',
      );
    }
    await selectEvaluation(state.evaluations[state.evaluations.length - 1].id);
  } catch (error) {
    showMessage(error.message || 'Unable to load backend evaluations.');
  }
}

async function selectEvaluation(evaluationId) {
  showMessage(`Loading ${evaluationId}...`);
  const response = await fetch(
    `/api/evaluations/${encodeURIComponent(evaluationId)}/frames`,
    { cache: 'no-store' },
  );
  if (!response.ok) return showMessage(`Unable to load ${evaluationId}.`);
  state.evaluation = await response.json();
  state.frames = Array.isArray(state.evaluation.frames) ? state.evaluation.frames : [];
  evaluationSelect.value = evaluationId;
  updateLayerOptions();
  renderSummary();
  applyFilter();
  if (!state.filteredIndices.length) return showMessage('No evaluation frames match the active filter.');
  selectFilteredPosition(0);
}

function updateLayerOptions() {
  const layers = new Set();
  for (const frame of state.frames) {
    for (const backendName of backendNames) {
      Object.keys(frame.overlays?.[backendName] || {}).forEach((layer) => layers.add(layer));
    }
  }
  const ordered = [...layers].sort((left, right) => {
    if (left === 'composite') return -1;
    if (right === 'composite') return 1;
    return left.localeCompare(right);
  });
  overlayLayerSelect.replaceChildren(...ordered.map((layer) => new Option(layer, layer)));
  state.selectedLayer = ordered.includes(state.selectedLayer) ? state.selectedLayer : ordered[0] || null;
  overlayLayerSelect.value = state.selectedLayer || '';
  overlayLayerSelect.disabled = ordered.length === 0;
}

function renderSummary() {
  const summary = state.evaluation.summary || {};
  summaryPanel.hidden = false;
  summaryPanel.replaceChildren(
    metric('processed_frame_count', summary.processed_frame_count),
    metric('frames_with_vehicle_state', summary.frames_with_vehicle_state),
    metric('frames_changed', summary.frames_changed),
    metric('frames_with_errors', summary.frames_with_errors),
    metric('max_position_delta_m', formatNumber(summary.max_position_delta_m)),
  );
}

function metric(label, value) {
  const item = document.createElement('div');
  item.className = 'evaluation-metric';
  item.append(
    Object.assign(document.createElement('strong'), { textContent: String(value ?? 'n/a') }),
    Object.assign(document.createElement('small'), { textContent: label }),
  );
  return item;
}

function applyFilter() {
  state.filteredIndices = state.frames
    .map((frame, index) => ({ frame, index }))
    .filter(({ frame }) => !changedOnly.checked || frame.comparison?.changed)
    .map(({ index }) => index);
  frameSlider.max = String(Math.max(state.filteredIndices.length - 1, 0));
  frameSlider.disabled = state.filteredIndices.length === 0;
}

function selectFilteredPosition(position) {
  if (!state.filteredIndices.length) return;
  state.filteredPosition = Math.min(Math.max(position, 0), state.filteredIndices.length - 1);
  frameSlider.value = String(state.filteredPosition);
  previousButton.disabled = state.filteredPosition === 0;
  nextButton.disabled = state.filteredPosition === state.filteredIndices.length - 1;
  renderFrame();
}

function renderFrame() {
  if (!state.filteredIndices.length) return;
  const frameIndex = state.filteredIndices[state.filteredPosition];
  const frame = state.frames[frameIndex];
  const comparison = frame.comparison || {};
  frameMeta.textContent = [
    `evaluation frame ${state.filteredPosition} / ${state.filteredIndices.length - 1}`,
    `source.frame_id=${frame.source?.frame_id}`,
    `comparison.changed=${Boolean(comparison.changed)}`,
    `gate_count_delta=${comparison.gate_count_delta ?? 'n/a'}`,
    `max_position_delta_m=${formatNumber(comparison.max_position_delta_m)}`,
  ].join(' - ');
  grid.replaceChildren(
    imageCard('source.relative_path', frame.source_image_url, frame.source?.relative_path),
    imageCard(
      `deterministic_v3.${state.selectedLayer}`,
      frame.overlays?.deterministic_v3?.[state.selectedLayer],
      backendReference(frame, 'deterministic_v3'),
    ),
    imageCard(
      `deterministic_v3_2.${state.selectedLayer}`,
      frame.overlays?.deterministic_v3_2?.[state.selectedLayer],
      backendReference(frame, 'deterministic_v3_2'),
    ),
  );
  frameMessage.hidden = true;
  grid.hidden = false;
}

function imageCard(labelText, imageUrl, referenceText) {
  const card = document.createElement('article');
  card.className = 'layer-card evaluation-card';
  const image = document.createElement('img');
  image.alt = labelText;
  if (imageUrl) {
    image.src = imageUrl;
  } else {
    card.classList.add('load-error');
  }
  const label = document.createElement('div');
  label.className = 'layer-label';
  label.append(
    Object.assign(document.createElement('strong'), { textContent: labelText }),
    Object.assign(document.createElement('small'), { textContent: referenceText || 'unavailable' }),
  );
  card.append(image, label);
  return card;
}

function backendReference(frame, backendName) {
  const backend = frame.comparison?.gate_pairs_by_index || [];
  const overlay = frame.overlays?.[backendName]?.[state.selectedLayer];
  const gateCount = backendName === 'deterministic_v3'
    ? frame.comparison?.left_gate_count
    : frame.comparison?.right_gate_count;
  return [
    `overlay=${state.selectedLayer}`,
    `gate_count=${gateCount ?? 'n/a'}`,
    overlay ? 'overlay_status=available' : 'overlay_status=missing',
    `${backend.length} gate_pairs_by_index`,
  ].join(' - ');
}

function formatNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(6)
    : 'n/a';
}

function showMessage(message) {
  frameMessage.hidden = false;
  frameMessage.textContent = message;
  grid.hidden = true;
  summaryPanel.hidden = true;
  frameSlider.disabled = true;
  previousButton.disabled = true;
  nextButton.disabled = true;
}

init();
