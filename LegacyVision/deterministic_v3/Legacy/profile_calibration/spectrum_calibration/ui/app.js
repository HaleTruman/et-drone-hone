const FIELD_NAMES = [
  'density_radius_px',
  'ridge_radius_px',
  'inverse_gamma',
  'ridge_gamma',
  'relative_cap',
  'open_kernel_radius_px',
  'close_kernel_radius_px',
];
const INTEGER_FIELDS = new Set([
  'density_radius_px',
  'ridge_radius_px',
  'open_kernel_radius_px',
  'close_kernel_radius_px',
]);
const MORPHOLOGY_RADIUS_FIELDS = new Set([
  'open_kernel_radius_px',
  'close_kernel_radius_px',
]);
const ENDPOINT_NAMES = ['large', 'small'];
const BATCH_SIZE = 100;
const PANEL_FIELDS = [
  'source_png',
  'closed_mask_png',
  'candidate_input_mask',
  'final_field',
  'p70_mask',
  'p80_mask',
  'p90_mask',
];

const controls = [...document.querySelectorAll('.endpoint-controls label[data-field]')];
const resetButton = document.getElementById('reset-endpoints');
const candidateStatus = document.getElementById('candidate-status');
const columnHeaders = document.getElementById('column-headers');
const componentColumns = document.getElementById('component-columns');
const pageMessage = document.getElementById('page-message');
const collectionSentinel = document.getElementById('collection-sentinel');

const state = {
  manifest: null,
  bins: [],
  endpoints: { large: {}, small: {} },
  endpointDefaults: { large: {}, small: {} },
  resolved: new Map(),
  columns: new Map(),
  loadedByBin: new Map(),
  totalByBin: new Map(),
  offset: 0,
  collectionComplete: false,
  collectionLoading: false,
  collectionController: null,
  resolveController: null,
  renderControllers: new Map(),
  renderTimes: new Map(),
  generation: 0,
  debounce: null,
};

function numberOr(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function binId(value) {
  return String(value.bin_id ?? value.profile_id ?? value.id ?? value.bin_index);
}

function binIndex(value) {
  return numberOr(value.bin_index, numberOr(String(binId(value)).match(/\d+/)?.[0], 0));
}

function normalizeBins(manifest) {
  let values = manifest.bins ?? manifest.profiles ?? [];
  if (!Array.isArray(values)) values = Object.values(values);
  return values.map((value) => {
    const id = binId(value);
    const profileCount = manifest.counts?.by_profile?.[id];
    const count = numberOr(
      value.total ?? value.count ?? value.component_count,
      numberOr(profileCount?.density_eligible ?? profileCount?.all, 0),
    );
    return {
      ...value,
      bin_id: id,
      bin_index: binIndex(value),
      min_area: value.min_area
        ?? value.minimum_area_px
        ?? value.minimum_component_area_px
        ?? null,
      max_area: value.max_area
        ?? value.maximum_area_px
        ?? value.maximum_component_area_px
        ?? null,
      total: count,
    };
  }).sort((left, right) => {
    // The persisted zero-based bin_index is the authoritative rank assignment:
    // 0 is the largest-area distribution and 9 is the smallest.
    return left.bin_index - right.bin_index;
  }).slice(0, 10);
}

function endpointSource(manifest, endpoint) {
  const explicit = manifest.endpoint_defaults?.[endpoint]
    ?? manifest[`${endpoint}_endpoint`]
    ?? manifest.endpoints?.[endpoint];
  if (explicit) return explicit;
  const flat = Object.fromEntries(FIELD_NAMES.map((field) => [
    field,
    manifest.endpoint_defaults?.[`${endpoint}_${field}`],
  ]));
  if (FIELD_NAMES.every((field) => flat[field] != null)) return flat;
  const bin = endpoint === 'large' ? state.bins[0] : state.bins.at(-1);
  return bin?.settings ?? bin?.profile ?? bin ?? {};
}

function boundsFor(field) {
  const bounds = state.manifest.endpoint_bounds?.[field]
    ?? state.manifest.control_bounds?.[field]
    ?? {};
  const integer = INTEGER_FIELDS.has(field);
  const morphologyRadius = MORPHOLOGY_RADIUS_FIELDS.has(field);
  return {
    minimum: numberOr(
      bounds.minimum ?? bounds.min,
      morphologyRadius ? 0 : (integer ? 1 : 0.01),
    ),
    maximum: numberOr(bounds.maximum ?? bounds.max, integer ? 64 : 100),
    step: numberOr(bounds.step, integer ? 1 : 0.01),
  };
}

function resolvedSettings(value) {
  const source = value.settings ?? value.resolved_settings ?? value.profile ?? value;
  return Object.fromEntries(FIELD_NAMES.map((field) => [field, numberOr(source[field], 0)]));
}

function formatValue(field, value) {
  if (!Number.isFinite(Number(value))) return 'unavailable';
  return INTEGER_FIELDS.has(field)
    ? String(Math.round(Number(value)))
    : Number(value).toFixed(2);
}

function areaRange(bin) {
  const minimum = bin.min_area;
  const maximum = bin.max_area;
  if (minimum == null && maximum == null) return 'area_px range unavailable';
  if (maximum == null) return `area_px>=${minimum}`;
  if (minimum == null) return `area_px<=${maximum}`;
  return `area_px=${minimum}–${maximum}`;
}

function configureControls() {
  for (const endpoint of ENDPOINT_NAMES) {
    const defaults = endpointSource(state.manifest, endpoint);
    state.endpointDefaults[endpoint] = Object.fromEntries(FIELD_NAMES.map((field) => [
      field,
      numberOr(
        defaults[field],
        MORPHOLOGY_RADIUS_FIELDS.has(field)
          ? 0
          : (INTEGER_FIELDS.has(field) ? 1 : 1.0),
      ),
    ]));
    state.endpoints[endpoint] = { ...state.endpointDefaults[endpoint] };
  }
  for (const label of controls) {
    const endpoint = label.closest('[data-endpoint]').dataset.endpoint;
    const field = label.dataset.field;
    const bounds = boundsFor(field);
    for (const input of label.querySelectorAll('input')) {
      input.min = bounds.minimum;
      input.max = bounds.maximum;
      input.step = bounds.step;
      input.value = state.endpoints[endpoint][field];
      input.disabled = false;
    }
  }
  resetButton.disabled = false;
}

function syncControl(label, input) {
  const endpoint = label.closest('[data-endpoint]').dataset.endpoint;
  const field = label.dataset.field;
  const bounds = boundsFor(field);
  let value = numberOr(input.value);
  if (value == null) return;
  value = Math.min(bounds.maximum, Math.max(bounds.minimum, value));
  if (INTEGER_FIELDS.has(field)) value = Math.round(value);
  state.endpoints[endpoint][field] = value;
  for (const peer of label.querySelectorAll('input')) peer.value = value;
  scheduleCandidateRefresh();
}

function endpointQuery() {
  const query = new URLSearchParams();
  for (const endpoint of ENDPOINT_NAMES) {
    for (const field of FIELD_NAMES) {
      query.set(`${endpoint}_${field}`, state.endpoints[endpoint][field]);
    }
  }
  return query;
}

function createBinHeader(bin) {
  const header = document.createElement('article');
  header.className = 'bin-header';
  header.dataset.binId = bin.bin_id;
  header.dataset.binIndex = bin.bin_index;
  header.title = `${bin.bin_id} · ${areaRange(bin)} · count=${bin.total}`;
  const settingLines = FIELD_NAMES.map((field) => {
    const code = document.createElement('code');
    code.dataset.field = field;
    code.textContent = `${field}=…`;
    return code;
  });
  header.append(
    Object.assign(document.createElement('strong'), {
      textContent: `bin_index=${bin.bin_index}`,
    }),
    Object.assign(document.createElement('span'), {
      className: 'bin-id',
      textContent: `default_profile_id=${bin.default_profile_id ?? 'unavailable'}`,
    }),
    Object.assign(document.createElement('span'), {
      className: 'bin-range',
      textContent: areaRange(bin),
    }),
    Object.assign(document.createElement('span'), {
      className: 'bin-count',
      textContent: `count=${bin.total}`,
    }),
    ...settingLines,
  );
  return header;
}

function buildColumns() {
  columnHeaders.replaceChildren(...state.bins.map(createBinHeader));
  componentColumns.replaceChildren(...state.bins.map((bin) => {
    const column = document.createElement('section');
    column.className = 'component-column';
    column.dataset.binId = bin.bin_id;
    column.dataset.binIndex = bin.bin_index;
    column.ariaLabel = `${bin.bin_id} descending ComponentObservation.area_px distribution`;
    state.columns.set(bin.bin_id, column);
    state.loadedByBin.set(bin.bin_id, 0);
    state.totalByBin.set(bin.bin_id, bin.total);
    return column;
  }));
}

function updateResolvedHeaders() {
  for (const bin of state.bins) {
    const settings = state.resolved.get(bin.bin_id)
      ?? state.resolved.get(String(bin.bin_index))
      ?? resolvedSettings(bin);
    const header = columnHeaders.querySelector(`[data-bin-id="${CSS.escape(bin.bin_id)}"]`);
    if (!header) continue;
    for (const field of FIELD_NAMES) {
      const code = header.querySelector(`code[data-field="${field}"]`);
      code.textContent = `${field}=${formatValue(field, settings[field])}`;
    }
    header.title = [
      `bin_index=${bin.bin_index}`,
      `default_profile_id=${bin.default_profile_id ?? 'unavailable'}`,
      areaRange(bin),
      `count=${state.totalByBin.get(bin.bin_id) ?? bin.total}`,
      ...FIELD_NAMES.map((field) => `${field}=${formatValue(field, settings[field])}`),
    ].join(' · ');
  }
}

function normalizeResolved(payload) {
  let values = payload.resolved_bins ?? payload.bins ?? payload.profiles ?? [];
  if (!Array.isArray(values)) values = Object.values(values);
  const result = new Map();
  for (const value of values) {
    const settings = resolvedSettings(value);
    const id = value.bin_id ?? value.profile_id ?? value.id;
    if (id != null) result.set(String(id), settings);
    if (value.bin_index != null) result.set(String(value.bin_index), settings);
  }
  return result;
}

async function resolveColumns(generation = state.generation) {
  state.resolveController?.abort();
  const controller = new AbortController();
  state.resolveController = controller;
  try {
    const response = await fetch(`/api/resolve?${endpointQuery()}`, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`/api/resolve failed: ${response.status}`);
    const payload = await response.json();
    if (generation !== state.generation) return;
    state.resolved = normalizeResolved(payload);
    updateResolvedHeaders();
    const signature = payload.signature ?? payload.candidate_signature ?? 'resolved';
    candidateStatus.textContent = `${signature} · generation=${generation}`;
  } catch (error) {
    if (error.name !== 'AbortError') candidateStatus.textContent = error.message;
  } finally {
    if (state.resolveController === controller) state.resolveController = null;
  }
}

function recordBinId(record, payloadBin) {
  const explicit = record.bin_id ?? record.profile_id;
  if (explicit != null) return String(explicit);
  const index = record.bin_index ?? payloadBin?.bin_index;
  const matched = state.bins.find((bin) => bin.bin_index === Number(index));
  return matched?.bin_id ?? String(index);
}

function componentTitle(record) {
  const bbox = record.bbox_xywh ?? [
    record.bbox_x,
    record.bbox_y,
    record.bbox_width,
    record.bbox_height,
  ];
  return [
    `asset_id=${record.asset_id}`,
    `run_id=${record.run_id}`,
    `frame_id=${record.frame_id}`,
    `sim_time_ns=${record.sim_time_ns}`,
    `component_id=${record.component_id}`,
    `area_px=${record.area_px}`,
    `bbox_xywh=(${bbox.join(',')})`,
    `touches_frame=${Boolean(record.touches_frame)}`,
    `panels=${PANEL_FIELDS.join(',')}`,
  ].join(' · ');
}

function createCard(record, bin) {
  const card = document.createElement('article');
  card.className = 'component-card';
  card.dataset.assetId = record.asset_id;
  card.dataset.binId = bin.bin_id;
  card.dataset.visible = 'false';
  card.dataset.stale = 'true';
  card.title = componentTitle(record);

  const meta = document.createElement('div');
  meta.className = 'component-meta';
  meta.append(
    Object.assign(document.createElement('strong'), {
      textContent: `area_px=${record.area_px} · asset_id=${record.asset_id}`,
    }),
    Object.assign(document.createElement('span'), {
      textContent: `run_id=${record.run_id}`,
    }),
    Object.assign(document.createElement('span'), {
      textContent: `frame_id=${record.frame_id} · component_id=${record.component_id}`,
    }),
    Object.assign(document.createElement('span'), {
      textContent: `touches_frame=${Boolean(record.touches_frame)}`,
    }),
  );
  const image = document.createElement('img');
  image.className = 'candidate-image';
  image.alt = `Candidate ${PANEL_FIELDS.join(', ')} for asset_id=${record.asset_id}`;
  card.append(meta, image);
  imageObserver.observe(card);
  return card;
}

function payloadBins(payload) {
  const bins = payload.bins ?? payload.columns;
  if (Array.isArray(bins)) return bins;
  if (bins && typeof bins === 'object') {
    return Object.entries(bins).map(([id, value]) => ({
      bin_id: id,
      ...(Array.isArray(value) ? { records: value } : value),
    }));
  }
  const records = Array.isArray(payload.records) ? payload.records : [];
  const grouped = new Map();
  for (const record of records) {
    const id = recordBinId(record);
    if (!grouped.has(id)) grouped.set(id, []);
    grouped.get(id).push(record);
  }
  return [...grouped].map(([id, values]) => ({ bin_id: id, records: values }));
}

function appendBatch(payload) {
  const returned = new Map();
  for (const payloadBin of payloadBins(payload)) {
    const id = recordBinId(payloadBin, payloadBin);
    const bin = state.bins.find((value) =>
      value.bin_id === id || value.bin_index === Number(payloadBin.bin_index));
    if (!bin) continue;
    const records = Array.isArray(payloadBin.records) ? payloadBin.records : [];
    returned.set(bin.bin_id, records.length);
    const total = numberOr(payloadBin.total ?? payloadBin.count);
    if (total != null) {
      state.totalByBin.set(bin.bin_id, total);
      const header = columnHeaders.querySelector(`[data-bin-id="${CSS.escape(bin.bin_id)}"] .bin-count`);
      if (header) header.textContent = `count=${total}`;
    }
    const column = state.columns.get(bin.bin_id);
    column.append(...records.map((record) => createCard(record, bin)));
    state.loadedByBin.set(
      bin.bin_id,
      (state.loadedByBin.get(bin.bin_id) ?? 0) + records.length,
    );
  }
  return returned;
}

function loadedSummary() {
  const loaded = [...state.loadedByBin.values()].reduce((sum, value) => sum + value, 0);
  const total = [...state.totalByBin.values()].reduce((sum, value) => sum + value, 0);
  return `${loaded}/${total || '?'} components loaded`;
}

function updateSentinel() {
  if (state.collectionLoading) {
    collectionSentinel.textContent = `Loading offset=${state.offset}, limit=${BATCH_SIZE} for every bin…`;
  } else if (state.collectionComplete) {
    collectionSentinel.textContent = `Complete exhaustive distributions · ${loadedSummary()}`;
  } else {
    collectionSentinel.textContent = `${loadedSummary()} · scroll to load the next synchronized batch`;
  }
}

async function loadNextBatch() {
  if (state.collectionLoading || state.collectionComplete || !state.bins.length) return;
  state.collectionLoading = true;
  updateSentinel();
  const controller = new AbortController();
  state.collectionController = controller;
  try {
    const query = new URLSearchParams({ offset: state.offset, limit: BATCH_SIZE });
    const response = await fetch(`/api/components?${query}`, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`/api/components failed: ${response.status}`);
    const payload = await response.json();
    const returned = appendBatch(payload);
    state.offset += BATCH_SIZE;
    const knownComplete = state.bins.every((bin) => {
      const total = state.totalByBin.get(bin.bin_id);
      return Number.isFinite(total) && (state.loadedByBin.get(bin.bin_id) ?? 0) >= total;
    });
    const noRecords = state.bins.every((bin) => (returned.get(bin.bin_id) ?? 0) === 0);
    state.collectionComplete = payload.complete === true || payload.has_more === false
      || knownComplete || noRecords;
    componentColumns.hidden = false;
    pageMessage.hidden = true;
  } catch (error) {
    if (error.name !== 'AbortError') {
      pageMessage.hidden = false;
      pageMessage.textContent = error.message;
      collectionSentinel.textContent = 'Component loading stopped.';
    }
  } finally {
    if (state.collectionController === controller) {
      state.collectionController = null;
      state.collectionLoading = false;
      updateSentinel();
    }
  }
}

function revokeImage(image) {
  if (image.dataset.objectUrl) {
    URL.revokeObjectURL(image.dataset.objectUrl);
    delete image.dataset.objectUrl;
  }
}

async function renderCard(card) {
  const generation = state.generation;
  const generationKey = String(generation);
  if (card.dataset.renderGeneration === generationKey
      || card.dataset.loadingGeneration === generationKey) return;
  const assetId = card.dataset.assetId;
  state.renderControllers.get(assetId)?.abort();
  const controller = new AbortController();
  state.renderControllers.set(assetId, controller);
  card.dataset.loadingGeneration = generationKey;
  card.dataset.loading = 'true';
  delete card.dataset.error;
  const started = performance.now();
  try {
    const response = await fetch(
      `/api/component/${encodeURIComponent(assetId)}/candidate.png?${endpointQuery()}`,
      { cache: 'no-store', signal: controller.signal },
    );
    if (!response.ok) throw new Error(`candidate.png failed: ${response.status}`);
    const blob = await response.blob();
    if (generation !== state.generation || !card.isConnected) return;
    const image = card.querySelector('.candidate-image');
    revokeImage(image);
    const objectUrl = URL.createObjectURL(blob);
    image.dataset.objectUrl = objectUrl;
    image.src = objectUrl;
    card.dataset.renderGeneration = generationKey;
    card.dataset.stale = 'false';
    const serverMs = numberOr(response.headers.get('X-Density-Compute-Ms'));
    state.renderTimes.set(assetId, serverMs ?? performance.now() - started);
    const times = [...state.renderTimes.values()];
    const mean = times.reduce((sum, value) => sum + value, 0) / Math.max(times.length, 1);
    candidateStatus.textContent = `generation=${generation} · visible mean=${mean.toFixed(2)} ms · ${times.length} rendered`;
  } catch (error) {
    if (error.name !== 'AbortError') {
      card.dataset.error = 'true';
      card.title = `${card.title} · ${error.message}`;
    }
  } finally {
    if (state.renderControllers.get(assetId) === controller) {
      state.renderControllers.delete(assetId);
      delete card.dataset.loadingGeneration;
      delete card.dataset.loading;
    }
  }
}

function abortRenders() {
  for (const controller of state.renderControllers.values()) controller.abort();
  state.renderControllers.clear();
  for (const card of componentColumns.querySelectorAll('.component-card')) {
    delete card.dataset.loadingGeneration;
    delete card.dataset.loading;
  }
}

function refreshCandidate() {
  state.generation += 1;
  abortRenders();
  state.renderTimes.clear();
  for (const card of componentColumns.querySelectorAll('.component-card')) {
    card.dataset.stale = 'true';
    if (card.dataset.visible === 'true') void renderCard(card);
  }
  candidateStatus.textContent = `resolving generation=${state.generation}…`;
  void resolveColumns(state.generation);
}

function scheduleCandidateRefresh() {
  window.clearTimeout(state.debounce);
  state.debounce = window.setTimeout(refreshCandidate, 220);
}

const imageObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const card = entry.target;
    card.dataset.visible = String(entry.isIntersecting);
    if (entry.isIntersecting) {
      void renderCard(card);
    } else {
      const assetId = card.dataset.assetId;
      state.renderControllers.get(assetId)?.abort();
      state.renderControllers.delete(assetId);
      delete card.dataset.loadingGeneration;
      delete card.dataset.loading;
    }
  }
}, { rootMargin: '1200px 0px' });

const collectionObserver = new IntersectionObserver((entries) => {
  if (entries.some((entry) => entry.isIntersecting)) void loadNextBatch();
}, { rootMargin: '1800px 0px' });

for (const label of controls) {
  label.querySelector('.range').addEventListener('input', (event) => {
    syncControl(label, event.target);
  });
  label.querySelector('.number').addEventListener('change', (event) => {
    syncControl(label, event.target);
  });
}

resetButton.addEventListener('click', () => {
  state.endpoints = {
    large: { ...state.endpointDefaults.large },
    small: { ...state.endpointDefaults.small },
  };
  for (const label of controls) {
    const endpoint = label.closest('[data-endpoint]').dataset.endpoint;
    const value = state.endpoints[endpoint][label.dataset.field];
    for (const input of label.querySelectorAll('input')) input.value = value;
  }
  scheduleCandidateRefresh();
});

window.addEventListener('beforeunload', () => {
  abortRenders();
  state.collectionController?.abort();
  state.resolveController?.abort();
  for (const image of componentColumns.querySelectorAll('.candidate-image')) revokeImage(image);
});

async function init() {
  try {
    const response = await fetch('/api/manifest', { cache: 'no-store' });
    if (!response.ok) throw new Error(`/api/manifest failed: ${response.status}`);
    state.manifest = await response.json();
    state.bins = normalizeBins(state.manifest);
    if (state.bins.length !== 10) {
      throw new Error(`/api/manifest must expose exactly 10 bins; received ${state.bins.length}`);
    }
    configureControls();
    buildColumns();
    updateResolvedHeaders();
    collectionObserver.observe(collectionSentinel);
    await resolveColumns();
    await loadNextBatch();
  } catch (error) {
    pageMessage.hidden = false;
    pageMessage.textContent = error.message || String(error);
    candidateStatus.textContent = 'Initialization failed';
  }
}

void init();
