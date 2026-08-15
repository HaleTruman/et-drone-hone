const runSelect = document.getElementById('run-select');
const frameSlider = document.getElementById('frame-slider');
const frameIndex = document.getElementById('frame-index');
const frameOutput = document.getElementById('frame-output');
const assignmentMode = document.getElementById('assignment-mode');
const includeClipped = document.getElementById('include-clipped');
const showOverlapCandidates = document.getElementById('show-overlap-candidates');
const resetButton = document.getElementById('reset');
const frameIdentity = document.getElementById('frame-identity');
const computeStatus = document.getElementById('compute-status');
const message = document.getElementById('message');
const analysisImage = document.getElementById('analysis-image');
const controlLabels = [...document.querySelectorAll('label[data-control]')];

const state = {
  catalog: null,
  runId: null,
  frameOffset: 0,
  settings: {},
  debounce: null,
  controller: null,
  generation: 0,
};

function records() {
  return state.catalog.runs[state.runId];
}

function record() {
  return records()[state.frameOffset];
}

function configureControl(label, value) {
  const name = label.dataset.control;
  const bounds = state.catalog.control_bounds[name];
  const input = label.querySelector('input');
  input.min = bounds.minimum;
  input.max = bounds.maximum;
  input.step = bounds.step;
  input.value = value;
  label.querySelector('output').value = value;
  state.settings[name] = value;
}

function configureDefaults() {
  const defaults = state.catalog.default;
  assignmentMode.value = defaults.assignment_mode;
  includeClipped.checked = defaults.include_clipped_components;
  showOverlapCandidates.checked = defaults.show_overlap_candidate_flags;
  for (const label of controlLabels) {
    configureControl(label, defaults[label.dataset.control]);
  }
}

function selectFrame(offset) {
  const available = records();
  state.frameOffset = Math.min(
    available.length - 1,
    Math.max(0, Number.parseInt(offset, 10) || 0),
  );
  frameSlider.value = state.frameOffset;
  frameIndex.value = state.frameOffset;
  const selected = record();
  frameOutput.value = selected.frame_id;
  frameIdentity.textContent = [
    `run_id=${state.runId}`,
    `frame_id=${selected.frame_id}`,
    `sim_time_ns=${selected.sim_time_ns}`,
    `source=${selected.source_relative_path}`,
  ].join(' · ');
  scheduleRender();
}

function configureRun(runId, preferredFrameId = null) {
  state.runId = runId;
  runSelect.value = runId;
  const available = records();
  frameSlider.min = 0;
  frameSlider.max = available.length - 1;
  frameSlider.step = 1;
  frameIndex.min = 0;
  frameIndex.max = available.length - 1;
  frameIndex.step = 1;
  frameSlider.disabled = false;
  frameIndex.disabled = false;
  const preferred = available.findIndex(
    (item) => item.frame_id === preferredFrameId,
  );
  selectFrame(preferred >= 0 ? preferred : 0);
}

function query() {
  const selected = record();
  return new URLSearchParams({
    run_id: state.runId,
    frame_id: selected.frame_id,
    assignment_mode: assignmentMode.value,
    include_clipped_components: includeClipped.checked ? '1' : '0',
    show_overlap_candidate_flags: showOverlapCandidates.checked ? '1' : '0',
    ...state.settings,
  });
}

function scheduleRender() {
  window.clearTimeout(state.debounce);
  state.debounce = window.setTimeout(() => void render(), 240);
}

async function render() {
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const generation = ++state.generation;
  computeStatus.textContent = 'computing whole-frame evidence…';
  message.hidden = false;
  message.textContent = analysisImage.src
    ? 'Updating experimental maps…'
    : 'Computing experimental maps…';
  try {
    const response = await fetch(`/api/analysis.png?${query()}`, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `analysis failed: ${response.status}`);
    }
    const metrics = JSON.parse(
      response.headers.get('X-Thickness-Metrics') || '{}',
    );
    const blob = await response.blob();
    if (generation !== state.generation) return;
    if (analysisImage.dataset.objectUrl) {
      URL.revokeObjectURL(analysisImage.dataset.objectUrl);
    }
    analysisImage.dataset.objectUrl = URL.createObjectURL(blob);
    analysisImage.src = analysisImage.dataset.objectUrl;
    analysisImage.hidden = false;
    message.hidden = true;
    computeStatus.textContent = [
      `compute=${Number(metrics.compute_ms || 0).toFixed(1)} ms`,
      `components=${metrics.component_count}`,
      `thickness P50/P90/max=${Number(metrics.local_thickness_p50_px).toFixed(1)}/${Number(metrics.local_thickness_p90_px).toFixed(1)}/${Number(metrics.local_thickness_max_px).toFixed(1)} px`,
      `unsupported=${metrics.unsupported_px}`,
      `clipped=${metrics.clipped_component_count}`,
      `overlap_candidate_count=${metrics.overlap_candidate_count ?? 0}`,
      `clipped_candidates=${metrics.clipped_overlap_candidate_count ?? 0}`,
      `two_population_ready=${metrics.thickness_split_ready_count ?? 0}`,
      ...(showOverlapCandidates.checked && Array.isArray(metrics.overlap_candidates)
        && metrics.overlap_candidates.length
        ? [`overlap_candidates=${metrics.overlap_candidates.map((candidate) => {
          const componentId = candidate.component_id ?? 'unknown';
          const reason = candidate.reason ?? candidate.classification_reason ?? 'unspecified';
          return `${componentId}:${reason}`;
        }).join(',')}`]
        : []),
    ].join(' · ');
  } catch (error) {
    if (error.name !== 'AbortError') {
      message.hidden = false;
      message.textContent = error.message || String(error);
      computeStatus.textContent = 'analysis error';
    }
  }
}

runSelect.addEventListener('change', () => configureRun(runSelect.value));
frameSlider.addEventListener('input', () => selectFrame(frameSlider.value));
frameIndex.addEventListener('change', () => selectFrame(frameIndex.value));
assignmentMode.addEventListener('change', scheduleRender);
includeClipped.addEventListener('change', scheduleRender);
showOverlapCandidates.addEventListener('change', scheduleRender);

for (const label of controlLabels) {
  const input = label.querySelector('input');
  input.addEventListener('input', () => {
    const value = Number(input.value);
    state.settings[label.dataset.control] = value;
    label.querySelector('output').value = value;
    scheduleRender();
  });
}

resetButton.addEventListener('click', () => {
  configureDefaults();
  configureRun(
    state.catalog.default.run_id,
    state.catalog.default.frame_id,
  );
});

async function init() {
  try {
    const response = await fetch('/api/catalog', { cache: 'no-store' });
    if (!response.ok) throw new Error(`catalog failed: ${response.status}`);
    state.catalog = await response.json();
    runSelect.replaceChildren(...Object.keys(state.catalog.runs).map(
      (runId) => new Option(
        `${runId} · ${state.catalog.runs[runId].length} frames`, runId,
      ),
    ));
    assignmentMode.replaceChildren(...state.catalog.assignment_modes.map(
      (mode) => new Option(mode.label, mode.value),
    ));
    runSelect.disabled = false;
    assignmentMode.disabled = false;
    configureDefaults();
    configureRun(
      state.catalog.default.run_id,
      state.catalog.default.frame_id,
    );
  } catch (error) {
    message.textContent = error.message || String(error);
  }
}

void init();
