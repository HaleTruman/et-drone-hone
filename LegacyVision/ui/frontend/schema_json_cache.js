const MAX_CACHED_FRAMES = 4;
const cache = new Map();
const pending = new Map();
let currentFrame = null;

function remember(reviewUrl, document) {
  cache.delete(reviewUrl);
  cache.set(reviewUrl, document);
  while (cache.size > MAX_CACHED_FRAMES) {
    cache.delete(cache.keys().next().value);
  }
  return document;
}

export async function loadReviewDocument(reviewUrl) {
  if (!reviewUrl) throw new Error('This frame has no associated schema_records JSON');
  if (cache.has(reviewUrl)) return remember(reviewUrl, cache.get(reviewUrl));
  if (pending.has(reviewUrl)) return pending.get(reviewUrl);

  const request = fetch(reviewUrl, { cache: 'no-store' })
    .then((response) => {
      if (!response.ok) throw new Error(`JSON fetch failed: ${response.status}`);
      return response.json();
    })
    .then((payload) => remember(reviewUrl, {
      payload,
      formatted: JSON.stringify(payload, null, 2),
    }))
    .finally(() => pending.delete(reviewUrl));
  pending.set(reviewUrl, request);
  return request;
}

export function preloadReviewJson(reviewUrl) {
  if (reviewUrl) void loadReviewDocument(reviewUrl).catch(() => {});
}

export function selectReviewFrame(runId, frame) {
  currentFrame = {
    runId,
    frame,
    reviewUrl: frame?.review_urls?.[0] || null,
  };
  preloadReviewJson(currentFrame.reviewUrl);
}

export function getSelectedReviewFrame() {
  return currentFrame;
}
