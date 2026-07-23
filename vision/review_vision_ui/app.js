const STAGES = [
  {
    stage: "color_masking",
    finalKey: "color_masking",
    jsonId: "colorJson",
    panelId: "colorPanel",
    renderer: "color_masking",
  },
  {
    stage: "bboxing",
    finalKey: "bboxing",
    jsonId: "bboxJson",
    panelId: "bboxPanel",
    renderer: "bboxing",
  },
  {
    stage: "clipping",
    finalKey: "clipping",
    jsonId: "clippingJson",
    panelId: "clippingPanel",
    renderer: "clipping",
  },
  {
    stage: "contouring",
    finalKey: "contouring",
    jsonId: "contourJson",
    panelId: "contourPanel",
    renderer: "contouring",
  },
  {
    stage: "pose_estimation",
    finalKey: "pose_estimation",
    jsonId: "poseJson",
    panelId: "posePanel",
    renderer: "pose_estimation",
  },
  {
    stage: "instance_tracking",
    finalKey: "instance_tracking",
    jsonId: "instanceJson",
    panelId: "instancePanel",
    renderer: "instance_tracking",
  },
];

const state = {
  manifest: null,
  frameIndex: [],
  current: 0,
  playing: false,
  timer: null,
  speed: 1,
  renderInFlight: false,
  pendingFrameRequest: null,
  renderVersion: 0,
  launchPollTimer: null,
};

const $ = (id) => document.getElementById(id);

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    payload = { error: response.statusText || "Expected JSON response" };
  }
  if (!response.ok) {
    const error = new Error(payload.error || response.statusText);
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function postJson(url, payload) {
  return fetchJson(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function fetchMaskBuffer(frame) {
  try {
    const response = await fetch(`/api/frame/${frame.frame_ordinal}/mask`);
    if (!response.ok) {
      let message = "mask artifact missing";
      try {
        const payload = await response.json();
        message = payload.error || message;
      } catch (error) {
        message = response.statusText || message;
      }
      return { buffer: null, error: message };
    }
    return { buffer: await response.arrayBuffer(), error: null };
  } catch (error) {
    return { buffer: null, error: error.message || "unable to load mask artifact" };
  }
}

function finalStage(finalPayload, stageKey) {
  return finalPayload && finalPayload[stageKey] ? finalPayload[stageKey] : null;
}

async function fetchStagePayload(frame, finalPayload, config) {
  const finalPayloadForStage = finalStage(finalPayload, config.finalKey);
  let missingMessage = "No debug artifact for this stage; showing final frame data.";
  try {
    const envelope = await fetchJson(`/api/frame/${frame.frame_ordinal}/debug/${config.stage}`);
    return {
      stage: config.stage,
      source: "debug",
      note: null,
      envelope,
      payload: envelope && Object.prototype.hasOwnProperty.call(envelope, "payload") ? envelope.payload : envelope,
    };
  } catch (error) {
    if (error.payload && error.payload.error) {
      missingMessage = `${error.payload.error}; showing final frame data.`;
    }
  }
  return {
    stage: config.stage,
    source: "final",
    note: missingMessage,
    envelope: null,
    payload: finalPayloadForStage,
  };
}

function renderJson(elementId, stageResult) {
  const wrapped = {
    source: stageResult.source,
    note: stageResult.note || undefined,
    data: stageResult.payload,
  };
  $(elementId).textContent = pretty(wrapped);
}

function formatNumber(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return number.toFixed(digits);
}

function formatTuple3(value, emptyLabel = "-") {
  if (!Array.isArray(value) || value.length !== 3) {
    return emptyLabel;
  }
  return value.map((item) => formatNumber(item)).join(", ");
}

function formatConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${Math.round(number * 100)}%`;
}

function renderResultsFrameList() {
  const root = $("resultsFrameList");
  if (!root) {
    return;
  }
  root.replaceChildren();
  if (!state.frameIndex.length) {
    const empty = document.createElement("div");
    empty.className = "results-empty";
    empty.textContent = "No frames loaded.";
    root.append(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  state.frameIndex.forEach((frame, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `results-frame-row${index === state.current ? " active" : ""}`;
    button.addEventListener("click", () => requestFrame(index));

    const ordinal = document.createElement("span");
    ordinal.className = "results-frame-ordinal";
    ordinal.textContent = String(index + 1).padStart(3, "0");

    const frameId = document.createElement("span");
    frameId.className = "results-frame-id";
    frameId.textContent = frame.frame_id || `frame_${String(index).padStart(6, "0")}`;

    button.append(ordinal, frameId);
    fragment.append(button);
  });
  root.append(fragment);
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "results-metric";
  const name = document.createElement("span");
  name.textContent = label;
  const data = document.createElement("strong");
  data.textContent = value;
  item.append(name, data);
  return item;
}

function renderResultsSummary(results, frame) {
  const root = $("resultsFrameSummary");
  if (!root) {
    return;
  }
  root.replaceChildren();
  if (!results) {
    const empty = document.createElement("div");
    empty.className = "results-empty";
    empty.textContent = "No vision_results object found in this frame.";
    root.append(empty);
    return;
  }
  root.append(
    metric("Schema", results.schema || "vision-results.v1"),
    metric("Frame", results.frame_id || frame.frame_id || "-"),
    metric("Coordinate Frame", results.coordinate_frame || "-"),
    metric("Image", `${results.image_width || frame.image_width || "-"} x ${results.image_height || frame.image_height || "-"}`),
    metric("Gates", String(Array.isArray(results.gates) ? results.gates.length : 0))
  );
}

function gateRow(label, value) {
  const row = document.createElement("div");
  row.className = "gate-row";
  const name = document.createElement("span");
  name.textContent = label;
  const data = document.createElement("strong");
  data.textContent = value;
  row.append(name, data);
  return row;
}

function renderGateList(results) {
  const root = $("resultsGateList");
  if (!root) {
    return;
  }
  root.replaceChildren();
  const gates = results && Array.isArray(results.gates) ? results.gates : [];
  if (!gates.length) {
    const empty = document.createElement("div");
    empty.className = "results-empty";
    empty.textContent = "No gates reported for this frame.";
    root.append(empty);
    return;
  }
  gates.forEach((gate) => {
    const card = document.createElement("article");
    card.className = "gate-card";

    const title = document.createElement("div");
    title.className = "gate-title";
    title.textContent = gate.gate_id || "unnamed-gate";

    const grid = document.createElement("div");
    grid.className = "gate-grid";
    grid.append(
      gateRow("Position camera m", formatTuple3(gate.position_camera_m)),
      gateRow("Position confidence", formatConfidence(gate.position_confidence)),
      gateRow("Orientation camera", formatTuple3(gate.orientation_camera, "none")),
      gateRow("Orientation confidence", formatConfidence(gate.orientation_confidence))
    );

    const trace = document.createElement("details");
    trace.className = "trace-toggle";
    const summary = document.createElement("summary");
    summary.textContent = "Trace";
    const pre = document.createElement("pre");
    pre.textContent = pretty(gate.trace || {});
    trace.append(summary, pre);

    card.append(title, grid, trace);
    root.append(card);
  });
}

function renderVisionResults(finalPayload, frame) {
  const results = finalPayload && finalPayload.vision_results ? finalPayload.vision_results : null;
  renderResultsFrameList();
  renderResultsSummary(results, frame);
  renderGateList(results);
  $("visionResultsJson").textContent = pretty(results || {});
}

function imageDimensions(finalPayload, frame) {
  const presetImage =
    state.manifest && state.manifest.preset && state.manifest.preset.image ? state.manifest.preset.image : {};
  return {
    imageWidth: Number(
      finalPayload.image_width ||
        (finalPayload.source && finalPayload.source.image_width) ||
        frame.image_width ||
        presetImage.width ||
        640
    ),
    imageHeight: Number(
      finalPayload.image_height ||
        (finalPayload.source && finalPayload.source.image_height) ||
        frame.image_height ||
        presetImage.height ||
        360
    ),
  };
}

function setSourceImages(sourceUrl) {
  document.querySelectorAll(".stage-source").forEach((image) => {
    image.src = sourceUrl;
  });
}

function clearReviewPanels() {
  document.querySelectorAll(".stage-source").forEach((image) => {
    image.removeAttribute("src");
  });
  document.querySelectorAll(".stage-overlay").forEach((canvas) => {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
  });
  document.querySelectorAll(".json-view").forEach((element) => {
    element.textContent = "";
  });
  document.querySelectorAll(".stage-note").forEach((element) => {
    element.textContent = "";
    element.classList.add("hidden");
  });
  ["resultsFrameList", "resultsFrameSummary", "resultsGateList"].forEach((id) => {
    const element = $(id);
    if (element) {
      element.replaceChildren();
    }
  });
}

function panelFor(config) {
  if (!config.panelId) {
    return null;
  }
  const root = $(config.panelId);
  if (!root) {
    return null;
  }
  return {
    root,
    image: root.querySelector(".stage-source"),
    canvas: root.querySelector(".stage-overlay"),
    note: root.querySelector(".stage-note"),
  };
}

function renderVisualStage(config, context) {
  const panel = panelFor(config);
  if (!panel || !panel.canvas) {
    return;
  }
  const renderers = window.VisionStageRenderers || {};
  const renderer = renderers[config.renderer];
  const shared = window.VisionReview && window.VisionReview.shared;
  if (!renderer || typeof renderer.render !== "function") {
    if (shared) {
      shared.clearCanvas(context, panel);
      shared.showNote(panel, `No renderer is registered for ${config.stage}.`);
    }
    return;
  }
  try {
    renderer.render(context, panel);
  } catch (error) {
    console.error(error);
    if (shared) {
      shared.clearCanvas(context, panel);
      shared.showNote(panel, error.message || `Unable to render ${config.stage}.`);
    }
  }
}

function updateReadout() {
  const total = state.frameIndex.length;
  $("frameSlider").max = String(Math.max(0, total - 1));
  $("frameSlider").value = String(state.current);
  $("frameReadout").textContent = total ? `Frame ${state.current + 1} / ${total}` : "Frame 0 / 0";
  renderResultsFrameList();
}

function setFrameControlsEnabled(enabled) {
  ["prevButton", "playButton", "nextButton", "speedSelect", "frameSlider"].forEach((id) => {
    $(id).disabled = !enabled;
  });
}

function clampedFrameIndex(index) {
  return Math.max(0, Math.min(index, state.frameIndex.length - 1));
}

function isActiveRender(index, version) {
  return version === state.renderVersion && index === state.current;
}

function requestFrame(index) {
  if (!state.frameIndex.length) {
    return;
  }
  state.current = clampedFrameIndex(index);
  state.renderVersion += 1;
  updateReadout();
  state.pendingFrameRequest = {
    index: state.current,
    version: state.renderVersion,
  };
  if (!state.renderInFlight) {
    drainFrameRequests();
  }
}

async function drainFrameRequests() {
  state.renderInFlight = true;
  try {
    while (state.pendingFrameRequest) {
      const request = state.pendingFrameRequest;
      state.pendingFrameRequest = null;
      await renderFrame(request.index, request.version);
    }
  } finally {
    state.renderInFlight = false;
    if (state.pendingFrameRequest) {
      drainFrameRequests();
    }
  }
}

async function renderFrame(index, version) {
  const frame = state.frameIndex[index];
  if (!frame || !isActiveRender(index, version)) {
    return;
  }

  const sourceUrl = `/api/frame/${frame.frame_ordinal}/source`;
  setSourceImages(sourceUrl);

  let finalPayload;
  try {
    finalPayload = await fetchJson(`/api/frame/${frame.frame_ordinal}/final`);
  } catch (error) {
    finalPayload = {};
  }
  if (!isActiveRender(index, version)) {
    return;
  }

  $("sourceJson").textContent = pretty(finalPayload.source || frame);
  renderVisionResults(finalPayload, frame);
  const stagePromise = Promise.all(STAGES.map((config) => fetchStagePayload(frame, finalPayload, config)));
  const maskPromise = fetchMaskBuffer(frame);
  const [stageResults, maskResult] = await Promise.all([stagePromise, maskPromise]);
  if (!isActiveRender(index, version)) {
    return;
  }

  const stagePayloads = {};
  stageResults.forEach((stageResult) => {
    stagePayloads[stageResult.stage] = stageResult;
  });

  STAGES.forEach((config) => {
    renderJson(config.jsonId, stagePayloads[config.stage]);
  });

  const dimensions = imageDimensions(finalPayload, frame);
  const context = {
    frame,
    frameOrdinal: frame.frame_ordinal,
    sourceUrl,
    manifest: state.manifest,
    finalPayload,
    stagePayloads,
    maskBuffer: maskResult.buffer,
    maskError: maskResult.error,
    imageWidth: dimensions.imageWidth,
    imageHeight: dimensions.imageHeight,
  };

  STAGES.filter((config) => config.renderer).forEach((config) => renderVisualStage(config, context));
}

function setPlaying(next) {
  state.playing = next && state.frameIndex.length > 0;
  $("playButton").textContent = state.playing ? "Pause" : "Play";
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  if (state.playing) {
    const intervalMs = 1000 / (30 * state.speed);
    state.timer = setInterval(() => {
      const nextFrame = (state.current + 1) % Math.max(1, state.frameIndex.length);
      requestFrame(nextFrame);
    }, intervalMs);
  }
}

function setLauncherStatus(message, tone) {
  const element = $("launcherStatus");
  element.textContent = message;
  element.classList.toggle("error", tone === "error");
  element.classList.toggle("complete", tone === "complete");
}

function applyDebugStatus(status) {
  const runDebugButton = $("runDebugButton");
  const isRunning = status && status.status === "running";
  runDebugButton.disabled = isRunning;
  if (!status || status.status === "idle") {
    setLauncherStatus("Enter a run folder to generate debug JSON.");
    return;
  }
  if (isRunning) {
    const count = Number(status.frameCount || 0);
    setLauncherStatus(`Running debug: ${count} frames written.`);
    return;
  }
  if (status.status === "complete") {
    setLauncherStatus(`Complete: ${status.runRoot}`, "complete");
    return;
  }
  if (status.status === "failed") {
    setLauncherStatus(status.error || "Debug run failed.", "error");
    return;
  }
  setLauncherStatus(String(status.status || "Idle"));
}

function stopLaunchPolling() {
  if (state.launchPollTimer) {
    clearInterval(state.launchPollTimer);
    state.launchPollTimer = null;
  }
}

async function pollDebugStatus() {
  try {
    const status = await fetchJson("/api/run-debug/status");
    applyDebugStatus(status);
    if (status.status === "running") {
      return;
    }
    stopLaunchPolling();
    if (status.status === "complete") {
      await loadManifest();
    }
  } catch (error) {
    stopLaunchPolling();
    setLauncherStatus(error.message || "Unable to read debug run status.", "error");
  }
}

function startLaunchPolling() {
  stopLaunchPolling();
  state.launchPollTimer = setInterval(pollDebugStatus, 1000);
  pollDebugStatus();
}

async function startDebugRun() {
  const runFolder = $("runFolderInput").value.trim();
  if (!runFolder) {
    setLauncherStatus("Enter a run folder path first.", "error");
    return;
  }
  setPlaying(false);
  setLauncherStatus("Starting debug run...");
  $("runDebugButton").disabled = true;
  try {
    const status = await postJson("/api/run-debug", { runFolder });
    applyDebugStatus(status);
    startLaunchPolling();
  } catch (error) {
    $("runDebugButton").disabled = false;
    setLauncherStatus(error.message || "Unable to start debug run.", "error");
  }
}

function wireControls() {
  $("prevButton").addEventListener("click", () => requestFrame(state.current - 1));
  $("nextButton").addEventListener("click", () => requestFrame(state.current + 1));
  $("playButton").addEventListener("click", () => setPlaying(!state.playing));
  $("speedSelect").addEventListener("change", (event) => {
    state.speed = Number(event.target.value) || 1;
    if (state.playing) {
      setPlaying(true);
    }
  });
  $("frameSlider").addEventListener("input", (event) => requestFrame(Number(event.target.value)));
  $("runDebugButton").addEventListener("click", startDebugRun);
  $("runFolderInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      startDebugRun();
    }
  });
}

async function loadManifest() {
  try {
    state.manifest = await fetchJson("/api/manifest");
    state.frameIndex = Array.isArray(state.manifest.frame_index) ? state.manifest.frame_index : [];
    state.current = 0;
    $("runMeta").textContent = `${state.manifest.run_id} - ${state.frameIndex.length} frames - debug ${
      state.manifest.debug ? "on" : "off"
    }`;
    setFrameControlsEnabled(state.frameIndex.length > 0);
    updateReadout();
    if (state.frameIndex.length) {
      requestFrame(0);
    } else {
      clearReviewPanels();
    }
    return true;
  } catch (error) {
    state.manifest = null;
    state.frameIndex = [];
    state.current = 0;
    setPlaying(false);
    setFrameControlsEnabled(false);
    clearReviewPanels();
    updateReadout();
    $("runMeta").textContent = error.message || "No run loaded.";
    return false;
  }
}

async function init() {
  wireControls();
  setFrameControlsEnabled(false);
  await loadManifest();
  try {
    const status = await fetchJson("/api/run-debug/status");
    applyDebugStatus(status);
    if (status.status === "running") {
      startLaunchPolling();
    }
  } catch (error) {
    setLauncherStatus("Run launcher is unavailable.", "error");
  }
}

init();
