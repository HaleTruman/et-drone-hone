import {
  preferredRunId,
  rememberRun,
} from './frame_navigation_state.js?v=1';

const runSelect = document.getElementById('run-select');
const labelSelect = document.getElementById('topology-label-select');
const summary = document.getElementById('topology-summary');
const grid = document.getElementById('topology-grid');
const message = document.getElementById('frame-message');

const labelColors = Object.freeze({
  standard: '#28e27d',
  multi_void: '#ff9f1c',
  c_shape: '#d878ff',
  unknown: '#45c7ff',
  clipped: '#ff4057',
});
const state = { runs: [], catalog: null, loadVersion: 0, imagePromises: new Map() };
const pendingCrops = new WeakMap();
const cropObserver = new IntersectionObserver((entries) => {
  entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
    cropObserver.unobserve(entry.target);
    const pending = pendingCrops.get(entry.target);
    if (!pending) return;
    renderCrop(pending.canvas, pending.component)
      .catch(() => entry.target.querySelector('.topology-crop-frame')
        ?.classList.add('load-error'));
  });
}, { rootMargin: '300px' });

runSelect.addEventListener('change', () => void selectRun(runSelect.value));
labelSelect.addEventListener('change', renderSelection);

async function init() {
  try {
    const response = await fetch('/api/runs', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Run catalog fetch failed: ${response.status}`);
    const payload = await response.json();
    state.runs = (Array.isArray(payload.runs) ? payload.runs : [])
      .filter((run) => run.has_logged_frames && run.review_json_count > 0);
    runSelect.replaceChildren(...state.runs.map((run) => new Option(
      `${run.id} · ${run.review_json_count} review JSON`, run.id)));
    if (!state.runs.length) return showMessage('No associated review JSON is available.');
    const newest = state.runs.reduce((latest, run) =>
      run.id.localeCompare(latest.id) > 0 ? run : latest);
    await selectRun(preferredRunId(state.runs, newest.id));
  } catch (error) {
    showMessage(error.message || 'Unable to load TopologyDecision.topology_label.');
  }
}

async function selectRun(runId) {
  const loadVersion = ++state.loadVersion;
  state.imagePromises.clear();
  showMessage(`Loading source.run_id=${runId} TopologyDecision records…`);
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/topology-components`,
      { cache: 'no-store' },
    );
    if (!response.ok) throw new Error(`TopologyDecision catalog failed: ${response.status}`);
    const catalog = await response.json();
    if (loadVersion !== state.loadVersion) return;
    state.catalog = catalog;
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
  const missing = Number(catalog.missing_topology_label_count || 0);
  const labeled = Number(catalog.labeled_decision_count || 0);
  if (!labeled && missing) {
    summary.textContent = `${missing} TopologyDecision records omit topology_label`;
    return showMessage(
      'TopologyDecision.topology_label is unavailable in this review JSON. ' +
      'Regenerate the review run with the current schema.');
  }
  const selected = labelSelect.value;
  const components = (catalog.components || []).filter((component) =>
    selected === 'all' || component.topology_label === selected);
  cropObserver.disconnect();
  grid.replaceChildren(...components.map(buildCard));
  grid.hidden = false;
  message.hidden = true;
  const incomplete = [
    missing ? `missing topology_label=${missing}` : '',
    catalog.missing_bbox_xywh_count
      ? `missing bbox_xywh=${catalog.missing_bbox_xywh_count}` : '',
  ].filter(Boolean).join(' · ');
  summary.textContent = [
    `shown=${components.length}`,
    `labeled=${labeled}`,
    incomplete,
  ].filter(Boolean).join(' · ');
  if (!components.length) {
    showMessage(`No TopologyDecision.topology_label=${selected} components are available.`);
    summary.textContent = [`shown=0`, `labeled=${labeled}`, incomplete]
      .filter(Boolean).join(' · ');
  }
}

function buildCard(component) {
  const card = document.createElement('article');
  card.className = 'topology-component-card';
  card.dataset.topologyLabel = component.topology_label;
  card.style.setProperty(
    '--topology-label-color', labelColors[component.topology_label] || '#ffffff');
  const canvas = document.createElement('canvas');
  canvas.width = 250;
  canvas.height = 250;
  canvas.setAttribute('aria-label',
    `TopologyDecision.topology_label=${component.topology_label}`);
  const cropFrame = document.createElement('div');
  cropFrame.className = 'topology-crop-frame';
  cropFrame.append(canvas);
  const label = document.createElement('div');
  label.className = 'topology-component-label';
  label.append(
    Object.assign(document.createElement('strong'), {
      textContent: `TopologyDecision.topology_label=${component.topology_label}`,
    }),
    Object.assign(document.createElement('small'), {
      textContent: `component_id=${component.component_id} · ` +
        `bbox_xywh=${component.bbox_xywh.join(',')} · ` +
        `source.frame_id=${component.source?.frame_id}`,
    }),
  );
  card.append(cropFrame, label);
  pendingCrops.set(card, { canvas, component });
  cropObserver.observe(card);
  return card;
}

async function renderCrop(canvas, component) {
  const image = await loadImage(component.frame.image_url);
  const [x, y, width, height] = component.bbox_xywh.map(Number);
  const context = canvas.getContext('2d', { alpha: false });
  const scale = Math.min(250 / width, 250 / height);
  const targetWidth = width * scale;
  const targetHeight = height * scale;
  const targetX = (250 - targetWidth) / 2;
  const targetY = (250 - targetHeight) / 2;
  context.imageSmoothingEnabled = false;
  context.fillStyle = '#000';
  context.fillRect(0, 0, 250, 250);
  context.drawImage(
    image, x, y, width, height,
    targetX, targetY, targetWidth, targetHeight,
  );
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

function showMessage(text) {
  grid.hidden = true;
  message.hidden = false;
  message.textContent = text;
}

init();
