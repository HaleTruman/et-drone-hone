import { applySchemaProvenance } from './schema_provenance.js?v=2';
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
const profileSummary = document.getElementById('profile-summary');

const sharedLayers = [
  {
    id: 'source.relative_path',
    label: 'source.relative_path',
    reference: 'source.run_id · source.frame_id · source.sim_time_ns',
    evidenceRoot: 'source',
  },
  {
    id: 'FrameObservation.base_mask',
    label: 'FrameObservation.base_mask',
    reference: 'schema_records[FrameObservation].base_mask',
    evidenceRoot: 'FrameObservation',
  },
  {
    id: 'FrameObservation.size_filtered_mask',
    label: 'FrameObservation.size_filtered_mask',
    reference: 'schema_records[FrameObservation].size_filtered_mask',
    evidenceRoot: 'FrameObservation',
  },
  {
    id: 'FrameObservation.closed_mask',
    label: 'FrameObservation.closed_mask',
    reference: 'schema_records[FrameObservation].closed_mask',
    evidenceRoot: 'FrameObservation',
  },
];

function reviewLayers(run) {
  const configuration = run.review_contract?.DensityBankConfiguration || {};
  const profiles = Array.isArray(configuration.profiles)
    ? configuration.profiles
    : (run.density_profile_ids || []).map((profile_id) => ({ profile_id }));
  const clipLayer = {
    id: 'ComponentObservation.touches_frame',
    label: 'ComponentObservation.touches_frame',
    reference: [
      'FrameObservation.component_labels',
      'ComponentObservation.component_id',
      'ComponentObservation.touches_frame',
      `DensityBankConfiguration.ignore_frame_edge_clipped=${configuration.ignore_frame_edge_clipped}`,
    ].join(' · '),
    evidenceRoot: 'ComponentObservation',
  };
  const densityLayers = profiles.flatMap((profile) => {
    const profileId = profile.profile_id;
    const settings = Object.entries(profile)
      .filter(([field]) => field !== 'profile_id' && field !== 'calibration_version')
      .map(([field, value]) => `${field}=${value}`)
      .join(' · ');
    return ['final_field', 'p70_mask', 'p80_mask', 'p90_mask'].map((fieldName) => ({
      id: `DensityEvidence[${profileId}].${fieldName}`,
      label: `DensityEvidence.${fieldName}`,
      reference: [
        'schema_records[DensityEvidence]',
        `profile.profile_id=${profileId}`,
        settings,
      ].filter(Boolean).join(' · '),
      evidenceRoot: 'DensityEvidence',
      batch: 'DensityEvidence',
      provenance: {
        fieldPath: 'profile.profile_id',
        fieldValue: profileId,
        evidencePath: `DensityEvidence.${fieldName}`,
      },
    }));
  });
  const preprocessingLayers = [
    ...sharedLayers,
    clipLayer,
    ...Array.from({ length: 3 }, () => ({
      placeholder: true,
      batch: 'preprocessing',
    })),
  ].map((layer) => ({ batch: 'preprocessing', ...layer }));
  const cShapeLayers = [{
    id: 'CShapeResult.refined_mask',
    label: 'CShapeResult.refined_mask',
    reference: [
      'schema_records[CShapeResult]',
      'topology_label=c_shape',
      'visible_lines',
      'completed_quadrilateral_uv',
      'refined_mask_origin_uv',
    ].join(' · '),
    evidenceRoot: 'CShapeResult',
    batch: 'CShapeResult',
    provenance: {
      fieldPath: 'topology_label',
      fieldValue: 'c_shape',
      evidencePath: 'CShapeResult.refined_mask',
    },
  }];
  return [...preprocessingLayers, ...cShapeLayers, ...densityLayers];
}

const state = { runs: [], run: null, frameIndex: 0, layers: [] };

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
      .filter((run) => run.has_logged_frames && run.review_json_count > 0);
    runSelect.replaceChildren(...state.runs.map((run) => new Option(
      `${run.id} · ${run.frame_count} frames · ${run.review_json_count} schema_records`,
      run.id)));
    if (!state.runs.length) {
      return showMessage('No runs containing DensityEvidence records are available.');
    }
    const newest = state.runs.reduce((latest, run) =>
      run.id.localeCompare(latest.id) > 0 ? run : latest);
    await selectRun(preferredRunId(state.runs, newest.id));
  } catch (error) {
    showMessage(error.message || 'Unable to load DensityEvidence records.');
  }
}

function buildLayerGrid() {
  layerGrid.replaceChildren(...state.layers.map((layer) => {
    const card = document.createElement('article');
    card.className = 'layer-card';
    card.dataset.batch = layer.batch;
    if (layer.placeholder) {
      card.classList.add('layer-card-blank');
      card.ariaHidden = 'true';
      return card;
    }
    card.dataset.evidenceRoot = layer.evidenceRoot;
    applySchemaProvenance(card, layer.provenance);
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
  showMessage(`Loading ${runId}…`);
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/frames`, { cache: 'no-store' });
  if (!response.ok) return showMessage(`Unable to load ${runId}.`);
  state.run = { ...summary, ...await response.json() };
  state.layers = reviewLayers(state.run);
  buildLayerGrid();
  const configuration = state.run.review_contract?.DensityBankConfiguration || {};
  profileSummary.textContent = [
    `DensityBankConfiguration.calibration_version=${configuration.calibration_version}`,
    `DensityBankConfiguration.ignore_frame_edge_clipped=${configuration.ignore_frame_edge_clipped}`,
    `DensityBankConfiguration.profiles=${configuration.profiles?.length || 0}`,
  ].join(' · ');
  runSelect.value = state.run.id;
  const count = state.run.frames?.length || 0;
  frameSlider.max = String(Math.max(count - 1, 0));
  frameSlider.disabled = count === 0;
  if (!count) return showMessage('This run has no associated mask frames.');
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
    image.src = frame.layers?.[image.dataset.layerId] || '';
  });
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
