const BATCH_SIZE = 24;
const profileSelect = document.getElementById('profile-select');
const includeClipped = document.getElementById('include-clipped');
const baselineSummary = document.getElementById('baseline-summary');
const corpusSummary = document.getElementById('corpus-summary');
const pageSummary = document.getElementById('page-summary');
const computeSummary = document.getElementById('compute-summary');
const resetCandidate = document.getElementById('reset-candidate');
const message = document.getElementById('message');
const grid = document.getElementById('component-grid');
const collectionSentinel = document.getElementById('collection-sentinel');
const controlLabels = [...document.querySelectorAll('#candidate-controls label[data-field]')];

const state = {
  manifest: null,
  profile: null,
  records: [],
  recordsById: new Map(),
  total: 0,
  settings: {},
  debounce: null,
  generation: 0,
  collectionRevision: 0,
  collectionController: null,
  loadingCollection: false,
  collectionComplete: false,
  renderControllers: new Map(),
  renderTimes: new Map(),
};

function areaLabel(profile) {
  const maximum = profile.maximum_component_area_px;
  return maximum === null
    ? `${profile.minimum_component_area_px}+ px`
    : `${profile.minimum_component_area_px}–${maximum} px`;
}

function profileCounts(profileId) {
  return state.manifest.counts.by_profile[profileId];
}

function baselineSettings(profile) {
  return {
    density_radius_px: profile.density_radius_px,
    ridge_radius_px: profile.ridge_radius_px,
    inverse_gamma: profile.inverse_gamma,
    ridge_gamma: profile.ridge_gamma,
  };
}

function configureControls() {
  state.settings = baselineSettings(state.profile);
  for (const label of controlLabels) {
    const field = label.dataset.field;
    const bounds = state.manifest.control_bounds[field];
    const slider = label.querySelector('.slider');
    const number = label.querySelector('.number');
    for (const input of [slider, number]) {
      input.min = bounds.minimum;
      input.max = bounds.maximum;
      input.step = bounds.step;
      input.value = state.settings[field];
    }
    label.querySelector('output').value = state.settings[field];
  }
  baselineSummary.textContent = [
    `baseline ${state.profile.profile_id}`,
    `maximum_component_area_px=${state.profile.maximum_component_area_px}`,
    `relative_cap=${state.profile.relative_cap}`,
    `calibration_version=${state.profile.calibration_version}`,
  ].join(' · ');
}

function syncControl(label, source) {
  const field = label.dataset.field;
  const bounds = state.manifest.control_bounds[field];
  const parsed = field.endsWith('_px')
    ? Number.parseInt(source.value, 10)
    : Number(source.value);
  if (!Number.isFinite(parsed)) return;
  const value = Math.min(bounds.maximum, Math.max(bounds.minimum, parsed));
  const slider = label.querySelector('.slider');
  const number = label.querySelector('.number');
  slider.value = value;
  number.value = value;
  label.querySelector('output').value = value;
  state.settings[field] = value;
  scheduleRender();
}

function candidateQuery() {
  return new URLSearchParams(Object.entries(state.settings)).toString();
}

function abortAllRenders() {
  for (const controller of state.renderControllers.values()) controller.abort();
  state.renderControllers.clear();
  for (const article of grid.querySelectorAll('.component-card')) {
    delete article.dataset.loadingGeneration;
  }
}

function revokeGridImages() {
  for (const image of grid.querySelectorAll('img[data-object-url]')) {
    URL.revokeObjectURL(image.dataset.objectUrl);
  }
}

function updateCollectionSummary() {
  const loaded = state.records.length;
  pageSummary.textContent = `${loaded} / ${state.total} loaded`;
  if (state.loadingCollection) {
    collectionSentinel.textContent = 'Loading more components…';
  } else if (state.collectionComplete) {
    collectionSentinel.textContent = state.total
      ? `Complete collection · ${state.total} components`
      : 'No components in this grouping.';
  } else {
    collectionSentinel.textContent = 'Scroll down to continue through this grouping';
  }
}

function updateComputeSummary() {
  const times = [...state.renderTimes.values()];
  const mean = times.length
    ? times.reduce((sum, value) => sum + value, 0) / times.length
    : 0;
  computeSummary.textContent = [
    `candidate=${JSON.stringify(state.settings)}`,
    `server mean=${mean.toFixed(2)} ms/component`,
    `${times.length}/${state.records.length} loaded cards current`,
  ].join(' · ');
}

function card(record) {
  const article = document.createElement('article');
  article.className = 'component-card';
  article.dataset.assetId = record.asset_id;
  article.dataset.clipped = String(Boolean(record.touches_frame));
  article.dataset.stale = 'true';
  const image = document.createElement('img');
  image.alt = `DensityEvidence comparison for asset ${record.asset_id}`;
  const meta = document.createElement('div');
  meta.className = 'component-meta';
  meta.innerHTML = [
    `<strong>${record.run_id}/${record.frame_id}/${record.component_id}</strong>`,
    `sim_time_ns=${record.sim_time_ns}`,
    `bbox_xywh=(${record.bbox_x},${record.bbox_y},${record.bbox_width},${record.bbox_height})`,
    `area_px=${record.area_px}`,
    `touches_frame=${Boolean(record.touches_frame)}`,
    `density_eligible=${Boolean(record.density_eligible)}`,
    `profile.profile_id=${record.profile_id}`,
  ].join(' · ');
  article.append(image, meta);
  return article;
}

async function renderCard(article) {
  const assetId = Number(article.dataset.assetId);
  const generation = state.generation;
  const generationKey = String(generation);
  if (
    article.dataset.renderGeneration === generationKey
    || article.dataset.loadingGeneration === generationKey
  ) return;
  const record = state.recordsById.get(assetId);
  if (!record) return;

  state.renderControllers.get(assetId)?.abort();
  const controller = new AbortController();
  state.renderControllers.set(assetId, controller);
  article.dataset.loadingGeneration = generationKey;
  const query = candidateQuery();
  try {
    const response = await fetch(
      `/api/component/${assetId}/comparison.png?${query}`,
      { cache: 'no-store', signal: controller.signal },
    );
    if (!response.ok) throw new Error(`candidate render failed: ${response.status}`);
    const computeMs = Number(response.headers.get('X-Profile-Compute-Ms'));
    const blob = await response.blob();
    if (generation !== state.generation || !article.isConnected) return;
    const image = article.querySelector('img');
    if (image.dataset.objectUrl) URL.revokeObjectURL(image.dataset.objectUrl);
    image.dataset.objectUrl = URL.createObjectURL(blob);
    image.src = image.dataset.objectUrl;
    article.dataset.renderGeneration = generationKey;
    delete article.dataset.stale;
    if (Number.isFinite(computeMs)) state.renderTimes.set(assetId, computeMs);
    updateComputeSummary();
  } catch (error) {
    if (error.name !== 'AbortError') computeSummary.textContent = error.message;
  } finally {
    if (state.renderControllers.get(assetId) === controller) {
      state.renderControllers.delete(assetId);
      delete article.dataset.loadingGeneration;
    }
  }
}

const cardObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const article = entry.target;
    article.dataset.visible = String(entry.isIntersecting);
    const assetId = Number(article.dataset.assetId);
    if (entry.isIntersecting) {
      void renderCard(article);
    } else {
      state.renderControllers.get(assetId)?.abort();
      state.renderControllers.delete(assetId);
      delete article.dataset.loadingGeneration;
    }
  }
}, { rootMargin: '600px 0px' });

async function loadNextBatch() {
  if (
    !state.profile
    || state.loadingCollection
    || state.collectionComplete
  ) return;
  const revision = state.collectionRevision;
  const controller = new AbortController();
  state.collectionController = controller;
  state.loadingCollection = true;
  if (!state.records.length) {
    message.hidden = false;
    message.textContent = 'Loading ComponentObservation bounding boxes…';
  }
  updateCollectionSummary();
  const query = new URLSearchParams({
    profile_id: state.profile.profile_id,
    include_clipped: includeClipped.checked ? '1' : '0',
    offset: state.records.length,
    limit: BATCH_SIZE,
  });
  try {
    const response = await fetch(`/api/components?${query}`, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`component catalog request failed: ${response.status}`);
    const payload = await response.json();
    if (revision !== state.collectionRevision) return;
    state.total = payload.total;
    const cards = payload.records.map((record) => {
      state.records.push(record);
      state.recordsById.set(record.asset_id, record);
      return card(record);
    });
    grid.append(...cards);
    for (const article of cards) cardObserver.observe(article);
    state.collectionComplete = state.records.length >= state.total;
    message.hidden = Boolean(state.records.length);
    message.textContent = state.records.length ? '' : 'No components in this grouping.';
    grid.hidden = !state.records.length;
  } catch (error) {
    if (error.name !== 'AbortError' && revision === state.collectionRevision) {
      state.collectionComplete = true;
      message.hidden = false;
      message.textContent = error.message || String(error);
    }
  } finally {
    if (state.collectionController === controller) {
      state.collectionController = null;
      state.loadingCollection = false;
      updateCollectionSummary();
      window.requestAnimationFrame(() => {
        const nearViewport = collectionSentinel.getBoundingClientRect().top
          < window.innerHeight + 1000;
        if (nearViewport) void loadNextBatch();
      });
    }
  }
}

const collectionObserver = new IntersectionObserver((entries) => {
  if (entries.some((entry) => entry.isIntersecting)) void loadNextBatch();
}, { rootMargin: '1000px 0px' });

async function resetCollection() {
  state.collectionRevision += 1;
  state.collectionController?.abort();
  state.collectionController = null;
  state.loadingCollection = false;
  state.collectionComplete = false;
  abortAllRenders();
  state.generation += 1;
  state.renderTimes.clear();
  cardObserver.disconnect();
  revokeGridImages();
  grid.replaceChildren();
  grid.hidden = true;
  state.records = [];
  state.recordsById.clear();
  state.total = 0;
  updateComputeSummary();
  updateCollectionSummary();
  await loadNextBatch();
}

function refreshCandidate() {
  abortAllRenders();
  state.generation += 1;
  state.renderTimes.clear();
  for (const article of grid.querySelectorAll('.component-card')) {
    article.dataset.stale = 'true';
    if (article.dataset.visible === 'true') void renderCard(article);
  }
  updateComputeSummary();
}

function scheduleRender() {
  window.clearTimeout(state.debounce);
  state.debounce = window.setTimeout(refreshCandidate, 180);
}

for (const label of controlLabels) {
  label.querySelector('.slider').addEventListener(
    'input', (event) => syncControl(label, event.target),
  );
  label.querySelector('.number').addEventListener(
    'change', (event) => syncControl(label, event.target),
  );
}

profileSelect.addEventListener('change', async () => {
  state.profile = state.manifest.profiles.find(
    (profile) => profile.profile_id === profileSelect.value,
  );
  configureControls();
  await resetCollection();
});

includeClipped.addEventListener('change', async () => {
  await resetCollection();
});

resetCandidate.addEventListener('click', () => {
  configureControls();
  scheduleRender();
});

async function init() {
  try {
    const response = await fetch('/api/manifest', { cache: 'no-store' });
    if (!response.ok) throw new Error(`manifest request failed: ${response.status}`);
    state.manifest = await response.json();
    corpusSummary.textContent = [
      `runs=${state.manifest.source_runs.join(',')}`,
      `frames=${state.manifest.counts.frames}`,
      `components=${state.manifest.counts.components}`,
      `density_eligible=${state.manifest.counts.density_eligible}`,
      `frame_edge_clipped=${state.manifest.counts.frame_edge_clipped}`,
    ].join(' · ');
    profileSelect.replaceChildren(...state.manifest.profiles.map((profile) => {
      const counts = profileCounts(profile.profile_id);
      return new Option(
        `${profile.profile_id} · area ${areaLabel(profile)} · ${counts.density_eligible} eligible / ${counts.all} all`,
        profile.profile_id,
      );
    }));
    profileSelect.disabled = false;
    state.profile = state.manifest.profiles[0];
    profileSelect.value = state.profile.profile_id;
    configureControls();
    collectionObserver.observe(collectionSentinel);
    await resetCollection();
  } catch (error) {
    message.hidden = false;
    message.textContent = error.message || String(error);
  }
}

void init();
