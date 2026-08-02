import {
  getSelectedReviewFrame,
  loadReviewDocument,
} from './schema_json_cache.js?v=2';

const style = document.createElement('style');
style.textContent = `
  .schema-json-info {
    position: absolute;
    z-index: 8;
    top: 6px;
    right: 6px;
    width: 20px;
    height: 20px;
    padding: 0;
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 50%;
    color: #fff;
    background: rgba(0, 0, 0, 0.72);
    font: 600 11px/18px ui-monospace, monospace;
    cursor: pointer;
    opacity: 0;
    transition: opacity 100ms ease;
  }
  .layer-card:hover .schema-json-info,
  .layer-card:focus-within .schema-json-info { opacity: 1; }
  .schema-json-dialog {
    width: min(92vw, 1040px);
    height: min(88vh, 820px);
    max-width: none;
    max-height: none;
    padding: 0;
    border: 1px solid #1d2733;
    color: #dce6ef;
    background: #0b1016;
  }
  .schema-json-dialog[open] {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
  }
  .schema-json-dialog::backdrop { background: rgba(5, 9, 13, 0.72); }
  .schema-json-header {
    display: flex;
    gap: 12px;
    align-items: start;
    justify-content: space-between;
    padding: 10px 12px;
    border-bottom: 1px solid #263443;
  }
  .schema-json-header h2 {
    margin: 0;
    font: 600 12px/1.3 ui-monospace, monospace;
  }
  .schema-json-meta {
    margin-top: 3px;
    color: #92a5b7;
    font: 9px/1.3 ui-monospace, monospace;
  }
  .schema-json-close {
    border: 0;
    color: #dce6ef;
    background: transparent;
    font: 18px/1 sans-serif;
    cursor: pointer;
  }
  .schema-json-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    min-height: 28px;
    padding: 5px 12px;
    border-bottom: 1px solid #263443;
    font: 9px/1.2 ui-monospace, monospace;
  }
  .schema-json-controls select {
    max-width: 220px;
    color: #dce6ef;
    border: 1px solid #3b4d60;
    background: #111a23;
    font: inherit;
  }
  .schema-json-content {
    min-height: 0;
    margin: 0;
    padding: 12px;
    overflow: auto;
    color: #cbd7e2;
    background: #0b1016;
    font: 10px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
    white-space: pre;
  }
  .schema-json-content mark {
    display: inline-block;
    width: 100%;
    color: #0b1016;
    background: #ffe37a;
  }
`;
document.head.append(style);

const dialog = document.createElement('dialog');
dialog.className = 'schema-json-dialog';
const header = document.createElement('header');
header.className = 'schema-json-header';
const headingWrap = document.createElement('div');
const heading = document.createElement('h2');
const metadata = document.createElement('div');
metadata.className = 'schema-json-meta';
headingWrap.append(heading, metadata);
const closeButton = document.createElement('button');
closeButton.className = 'schema-json-close';
closeButton.type = 'button';
closeButton.ariaLabel = 'Close JSON inspector';
closeButton.textContent = '×';
header.append(headingWrap, closeButton);
const controls = document.createElement('div');
controls.className = 'schema-json-controls';
const targetLabel = document.createElement('span');
const componentSelect = document.createElement('select');
componentSelect.ariaLabel = 'component_id';
componentSelect.hidden = true;
controls.append(targetLabel, componentSelect);
const content = document.createElement('pre');
content.className = 'schema-json-content';
dialog.append(header, controls, content);
document.body.append(dialog);

closeButton.addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});

function recordStarts(lines, recordName) {
  const needle = `"name": "${recordName}"`;
  return lines.flatMap((line, index) => line.includes(needle) ? [index] : []);
}

function findTargets(lines, layerId) {
  const density = /^DensityEvidence\[([^\]]+)]\.([A-Za-z0-9_]+)$/.exec(layerId);
  if (density) {
    const [, profileId, fieldName] = density;
    const starts = recordStarts(lines, 'DensityEvidence');
    return starts.flatMap((start, index) => {
      const end = starts[index + 1] ?? lines.length;
      const profileLine = lines.findIndex((line, lineIndex) =>
        lineIndex >= start && lineIndex < end &&
        line.includes(`"profile_id": "${profileId}"`));
      const fieldLine = lines.findIndex((line, lineIndex) =>
        lineIndex > profileLine && lineIndex < end &&
        line.includes(`"${fieldName}":`));
      if (profileLine < 0 || fieldLine < 0) return [];
      const componentLine = lines.find((line, lineIndex) =>
        lineIndex >= start && lineIndex < profileLine &&
        line.includes('"component_id":'));
      const componentId = /"component_id":\s*([^,]+)/.exec(componentLine || '')?.[1];
      return [{ line: fieldLine, componentId }];
    });
  }

  const componentField = /^ComponentObservation\.([A-Za-z0-9_]+)$/.exec(layerId);
  if (componentField) {
    const fieldName = componentField[1];
    const starts = recordStarts(lines, 'ComponentObservation');
    return starts.flatMap((start, index) => {
      const end = starts[index + 1] ?? lines.length;
      const fieldLine = lines.findIndex((line, lineIndex) =>
        lineIndex >= start && lineIndex < end &&
        line.includes(`"${fieldName}":`));
      if (fieldLine < 0) return [];
      const componentLine = lines.find((line, lineIndex) =>
        lineIndex >= start && lineIndex < end &&
        line.includes('"component_id":'));
      const componentId = /"component_id":\s*([^,]+)/.exec(componentLine || '')?.[1];
      return [{ line: fieldLine, componentId }];
    });
  }

  const schemaField = /^([A-Za-z][A-Za-z0-9_]*|source)\.([A-Za-z0-9_]+)$/.exec(layerId);
  if (!schemaField) return [];
  const [, recordName, fieldName] = schemaField;
  const starts = recordName === 'source'
    ? [lines.findIndex((line) => line.includes('"source": {'))]
    : recordStarts(lines, recordName);
  return starts.flatMap((start, index) => {
    if (start < 0) return [];
    const end = starts[index + 1] ?? lines.length;
    const line = lines.findIndex((value, lineIndex) =>
      lineIndex >= start && lineIndex < end && value.includes(`"${fieldName}":`));
    if (line < 0) return [];
    const componentLine = lines.find((value, lineIndex) =>
      lineIndex >= start && lineIndex < end && value.includes('"component_id":'));
    const componentId = /"component_id":\s*([^,]+)/.exec(componentLine || '')?.[1];
    return [{ line, componentId }];
  });
}

function renderFullJson(payload, formatted, layerId) {
  const lines = formatted.split('\n');
  const targets = findTargets(lines, layerId);
  const marks = [];
  content.replaceChildren();
  let cursor = 0;
  for (const target of targets) {
    if (target.line > cursor) {
      content.append(document.createTextNode(
        `${lines.slice(cursor, target.line).join('\n')}\n`));
    }
    const mark = document.createElement('mark');
    mark.textContent = lines[target.line];
    mark.dataset.componentId = target.componentId || '';
    content.append(mark, document.createTextNode('\n'));
    marks.push(mark);
    cursor = target.line + 1;
  }
  if (cursor < lines.length) {
    content.append(document.createTextNode(lines.slice(cursor).join('\n')));
  }

  componentSelect.replaceChildren(...targets.map((target, index) => new Option(
    target.componentId ? `component_id=${target.componentId}` : `match=${index + 1}`,
    String(index))));
  componentSelect.hidden = targets.length < 2;
  componentSelect.onchange = () => marks[Number(componentSelect.value)]
    ?.scrollIntoView({ block: 'center' });
  requestAnimationFrame(() => marks[0]?.scrollIntoView({ block: 'center' }));
  return targets.length;
}

async function currentFrameRecord() {
  const selected = getSelectedReviewFrame();
  if (!selected?.runId) throw new Error('source.run_id is unavailable');
  const document = await loadReviewDocument(selected.reviewUrl);
  return {
    runId: selected.runId,
    frame: selected.frame,
    payload: document.payload,
    formatted: document.formatted,
  };
}

async function openInspector(card) {
  const layerId = card.querySelector('img[data-layer-id]')?.dataset.layerId;
  if (!layerId) return;
  heading.textContent = layerId;
  metadata.textContent = 'Loading schema_records…';
  targetLabel.textContent = layerId;
  componentSelect.hidden = true;
  content.textContent = 'Loading…';
  if (!dialog.open) dialog.showModal();
  try {
    const { runId, frame, payload, formatted } = await currentFrameRecord();
    const matchCount = renderFullJson(payload, formatted, layerId);
    metadata.textContent = [
      `source.run_id=${runId}`,
      `source.frame_id=${payload.source?.frame_id ?? 'unavailable'}`,
      `source.relative_path=${frame.filename}`,
      `matches=${matchCount}`,
    ].join(' · ');
  } catch (error) {
    metadata.textContent = 'JSON unavailable';
    content.textContent = error.message || String(error);
  }
}

function decorateCards(root = document) {
  root.querySelectorAll?.(
    '.layer-card:not(.layer-card-blank):not([data-json-inspector])',
  ).forEach((card) => {
    card.dataset.jsonInspector = 'true';
    const button = document.createElement('button');
    button.className = 'schema-json-info';
    button.type = 'button';
    button.textContent = 'i';
    const layerId = card.querySelector('img[data-layer-id]')?.dataset.layerId || 'field';
    button.ariaLabel = `Inspect ${layerId} JSON`;
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      void openInspector(card);
    });
    card.append(button);
  });
}

decorateCards();
new MutationObserver(() => decorateCards()).observe(document.body, {
  childList: true,
  subtree: true,
});
