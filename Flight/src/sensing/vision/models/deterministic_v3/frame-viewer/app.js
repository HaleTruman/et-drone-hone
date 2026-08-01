const runSelect = document.getElementById('run-select');
const frameSlider = document.getElementById('frame-slider');
const frameMeta = document.getElementById('frame-meta');
const previousButton = document.getElementById('previous-frame');
const nextButton = document.getElementById('next-frame');
const layerGrid = document.getElementById('layer-grid');
const frameMessage = document.getElementById('frame-message');
const sweepControl = document.getElementById('density-sweep-control');
const sweepSlider = document.getElementById('density-sweep-slider');
const sweepOutput = document.getElementById('density-sweep-output');

const state = { layers: [], sweeps: [], runs: [], run: null, frameIndex: 0 };

runSelect.addEventListener('change', () => selectRun(runSelect.value));
frameSlider.addEventListener('input', () => selectFrame(Number(frameSlider.value)));
previousButton.addEventListener('click', () => selectFrame(state.frameIndex - 1));
nextButton.addEventListener('click', () => selectFrame(state.frameIndex + 1));
sweepSlider.addEventListener('input', applySweep);

async function init() {
  try {
    const response = await fetch('data/runs-manifest.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Manifest fetch failed: ${response.status}`);
    const payload = await response.json();
    state.layers = Array.isArray(payload.layers) ? payload.layers : [];
    state.sweeps = Array.isArray(payload.sweeps) ? payload.sweeps : [];
    state.runs = Array.isArray(payload.runs) ? payload.runs : [];
    if (!state.layers.length) return showMessage('No component layers are defined.');
    buildLayerGrid();
    configureSweep();
    runSelect.replaceChildren();
    state.runs.forEach((run) => runSelect.append(new Option(run.name || run.id, run.id)));
    if (!state.runs.length) return showMessage('No generated mask runs are available.');
    const newestRun = state.runs.reduce((latest, run) =>
      run.id.localeCompare(latest.id) > 0 ? run : latest);
    selectRun(newestRun.id);
  } catch (error) {
    showMessage(error.message || 'Unable to load mask review manifest.');
  }
}

function configureSweep() {
  const sweep = state.sweeps[0];
  if (!sweep?.presets?.length) return;
  const defaultIndex = Math.max(0, sweep.presets.findIndex(
    (preset) => preset.id === sweep.default_preset));
  sweepSlider.max = String(sweep.presets.length - 1);
  sweepSlider.value = String(defaultIndex);
  sweepControl.hidden = false;
}

function applySweep() {
  const sweep = state.sweeps[0];
  const frame = state.run?.frames?.[state.frameIndex];
  const preset = sweep?.presets?.[Number(sweepSlider.value)];
  if (!frame || !preset) return;
  sweepOutput.textContent = preset.label;
  const image = layerGrid.querySelector(`img[data-layer-id="${sweep.layer_id}"]`);
  const url = frame.sweeps?.[sweep.layer_id]?.[preset.id];
  if (!image || !url) return;
  image.src = url;
  const reference = image.parentElement.querySelector('.layer-label small');
  if (reference) reference.textContent =
    `void_detection.py::legacy_inverse_mask_density_layer · ${preset.label}`;
}

function buildLayerGrid() {
  layerGrid.replaceChildren(...state.layers.map((layer) => {
    const card = document.createElement('article');
    card.className = 'layer-card';
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

function selectRun(runId) {
  state.run = state.runs.find((run) => run.id === runId) || null;
  state.frameIndex = 0;
  runSelect.value = state.run?.id || '';
  const count = state.run?.frames?.length || 0;
  frameSlider.max = String(Math.max(count - 1, 0));
  frameSlider.disabled = count === 0;
  if (!count) return showMessage('This run has no generated mask frames.');
  selectFrame(0);
}

function selectFrame(index) {
  const frames = state.run?.frames || [];
  if (!frames.length) return;
  state.frameIndex = Math.min(Math.max(index, 0), frames.length - 1);
  frameSlider.value = String(state.frameIndex);
  previousButton.disabled = state.frameIndex === 0;
  nextButton.disabled = state.frameIndex === frames.length - 1;
  const frame = frames[state.frameIndex];
  frameMeta.textContent = `Frame ${frame.id} (UI ${state.frameIndex + 1}) / ${frames.length}`;
  layerGrid.querySelectorAll('img[data-layer-id]').forEach((image) => {
    image.src = frame.layers[image.dataset.layerId] || '';
  });
  applySweep();
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
