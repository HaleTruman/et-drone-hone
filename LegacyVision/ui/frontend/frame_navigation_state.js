const STORAGE_KEY = 'deterministic-v3.review-navigation.v1';

function readState() {
  try {
    const value = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '{}');
    return value && typeof value === 'object' ? value : {};
  } catch (_error) {
    return {};
  }
}

function writeState(value) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch (_error) {
    // Navigation remains functional when browser storage is unavailable.
  }
}

export function preferredRunId(runs, fallbackRunId) {
  const available = new Set(runs.map((run) => run.id));
  const remembered = readState().lastRunId;
  return available.has(remembered) ? remembered : fallbackRunId;
}

export function rememberedFrameIndex(runId, frames) {
  const remembered = readState().framesByRun?.[runId];
  if (!remembered || !frames.length) return 0;
  const exactIndex = frames.findIndex((frame) =>
    frame.filename === remembered.filename);
  const index = exactIndex >= 0 ? exactIndex : Number(remembered.frameIndex);
  return Number.isInteger(index)
    ? Math.min(Math.max(index, 0), frames.length - 1)
    : 0;
}

export function rememberRun(runId) {
  const state = readState();
  state.lastRunId = runId;
  writeState(state);
}

export function rememberFrame(runId, frameIndex, frame) {
  const state = readState();
  state.lastRunId = runId;
  state.framesByRun = state.framesByRun || {};
  state.framesByRun[runId] = {
    frameIndex,
    filename: frame?.filename || null,
  };
  writeState(state);
}
