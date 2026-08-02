import {
  preloadReviewJson,
  selectReviewFrame,
} from './schema_json_cache.js?v=2';
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
const layerGrid = document.getElementById('layer-grid');
const frameMessage = document.getElementById('frame-message');

const state = { runs: [], run: null, frameIndex: 0 };
const layers = [{
  id: 'source.relative_path',
  label: 'source.relative_path',
  reference: 'source.run_id · source.frame_id · source.sim_time_ns',
  evidenceRoot: 'source',
}];

runSelect.addEventListener('change', () => void selectRun(runSelect.value));
frameSlider.addEventListener('input', () => selectFrame(Number(frameSlider.value)));
previousButton.addEventListener('click', () => selectFrame(state.frameIndex - 1));
nextButton.addEventListener('click', () => selectFrame(state.frameIndex + 1));

async function init() {
  try {
    const response = await fetch('/api/runs', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Run catalog fetch failed: ${response.status}`);
    const payload = await response.json();
    state.runs = (Array.isArray(payload.runs) ? payload.runs : [])
      .filter((run) => run.has_logged_frames && run.frame_count > 0);
    buildLayerGrid();
    runSelect.replaceChildren(...state.runs.map((run) => new Option(
      `${run.id} · ${run.frame_count} source.relative_path values`, run.id)));
    if (!state.runs.length) {
      return showMessage('No source.relative_path values are available.');
    }
    const reviewed = state.runs.filter((run) =>
      run.review_json_count > 0 || run.geometry_frame_result_count > 0);
    const candidates = reviewed.length ? reviewed : state.runs;
    const newest = candidates.reduce((latest, run) =>
      run.id.localeCompare(latest.id) > 0 ? run : latest);
    await selectRun(preferredRunId(state.runs, newest.id));
  } catch (error) {
    showMessage(error.message || 'Unable to load source.relative_path review.');
  }
}

function buildLayerGrid() {
  layerGrid.replaceChildren(...layers.map((layer) => {
    const card = document.createElement('article');
    card.className = 'layer-card';
    card.dataset.evidenceRoot = layer.evidenceRoot;
    const image = document.createElement('img');
    image.dataset.layerId = layer.id;
    image.alt = layer.label;
    image.addEventListener('error', () => card.classList.add('load-error'));
    image.addEventListener('load', () => card.classList.remove('load-error'));
    const label = document.createElement('div');
    label.className = 'layer-label';
    label.append(
      Object.assign(document.createElement('strong'), { textContent: layer.label }),
      Object.assign(document.createElement('small'), { textContent: layer.reference }),
    );
    card.append(image, label);
    return card;
  }));
}

async function selectRun(runId) {
  const summary = state.runs.find((run) => run.id === runId) || null;
  if (!summary) return showMessage('Unknown source.run_id.');
  preloadReviewJson(summary.initial_review_url);
  showMessage(`Loading source.run_id=${runId}…`);
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/frames`, { cache: 'no-store' });
  if (!response.ok) return showMessage(`Unable to load source.run_id=${runId}.`);
  state.run = { ...summary, ...await response.json() };
  runSelect.value = state.run.id;
  const count = state.run.frames?.length || 0;
  frameSlider.max = String(Math.max(count - 1, 0));
  frameSlider.disabled = count === 0;
  if (!count) return showMessage('This source.run_id has no source.relative_path values.');
  selectFrame(rememberedFrameIndex(state.run.id, state.run.frames));
}

function selectFrame(index) {
  const frames = state.run?.frames || [];
  if (!frames.length) return;
  state.frameIndex = Math.min(Math.max(index, 0), frames.length - 1);
  frameSlider.value = String(state.frameIndex);
  previousButton.disabled = state.frameIndex === 0;
  nextButton.disabled = state.frameIndex === frames.length - 1;
  const frame = frames[state.frameIndex];
  rememberFrame(state.run.id, state.frameIndex, frame);
  selectReviewFrame(state.run.id, frame);
  frameMeta.textContent =
    `UI frame ${state.frameIndex} / ${frames.length - 1} · ` +
    `source.relative_path=${frame.filename}`;
  layerGrid.querySelectorAll('img[data-layer-id]').forEach((image) => {
    image.src = frame.layers?.[image.dataset.layerId] || frame.image_url;
  });
  const reference = layerGrid.querySelector('.layer-label small');
  if (reference) reference.textContent =
    `source.run_id=${state.run.id} · source.relative_path=${frame.filename}`;
  frameMessage.hidden = true;
  layerGrid.hidden = false;
}

function showMessage(message) {
  layerGrid.hidden = true;
  frameMessage.hidden = false;
  frameMessage.textContent = message;
  frameSlider.disabled = true;
  previousButton.disabled = true;
  nextButton.disabled = true;
}

init();
