const state = {
  source: "datasets",
  datasets: [],
  evaluations: [],
  validations: [],
  datasetRunName: null,
  evaluationName: null,
  validationName: null,
  frames: [],
  metrics: null,
  frameIndex: 0,
  evaluationImageMode: "source",
  image: new Image(),
};

const colors = [
  "#00e5ff",
  "#ffca28",
  "#66bb6a",
  "#ef5350",
  "#ab47bc",
  "#ffa726",
  "#26c6da",
  "#ec407a",
];

const cornerColors = {
  Corner_outer_TL: "#ff5252",
  Corner_outer_TR: "#ff8a80",
  Corner_outer_BL: "#40c4ff",
  Corner_outer_BR: "#82b1ff",
  Corner_inner_TL: "#69f0ae",
  Corner_inner_TR: "#b9f6ca",
  Corner_inner_BL: "#ffd740",
  Corner_inner_BR: "#ffff8d",
};

const el = {
  status: document.getElementById("status"),
  sourceSelect: document.getElementById("sourceSelect"),
  runSelect: document.getElementById("runSelect"),
  evaluationSelect: document.getElementById("evaluationSelect"),
  validationSelect: document.getElementById("validationSelect"),
  frameSlider: document.getElementById("frameSlider"),
  frameLabel: document.getElementById("frameLabel"),
  canvas: document.getElementById("canvas"),
  showBbox: document.getElementById("showBbox"),
  showCorners: document.getElementById("showCorners"),
  onlyVisibleGates: document.getElementById("onlyVisibleGates"),
  showEvalTruthBbox: document.getElementById("showEvalTruthBbox"),
  showEvalTruthCorners: document.getElementById("showEvalTruthCorners"),
  showEvalPredBbox: document.getElementById("showEvalPredBbox"),
  showEvalPredCorners: document.getElementById("showEvalPredCorners"),
  showEvalLabels: document.getElementById("showEvalLabels"),
  cornerLegend: document.getElementById("cornerLegend"),
  evaluationPanel: document.getElementById("evaluationPanel"),
  evaluationInfo: document.getElementById("evaluationInfo"),
  frameInfo: document.getElementById("frameInfo"),
  gateList: document.getElementById("gateList"),
};

const ctx = el.canvas.getContext("2d");

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function frameUrl(datasetRunName, frame) {
  const file = frame.frame_path.split(/[\\/]/).pop();
  const [datasetName, runName] = datasetRunName.split("/");
  return `/datasets/${encodeURIComponent(datasetName)}/runs/${encodeURIComponent(runName)}/frames/${encodeURIComponent(file)}`;
}

function evaluationOverlayUrl(evaluationName, frame) {
  const match = frame.sample_id.match(/^(.*)\/frame_(\d+)$/);
  if (!match) return null;
  const file = `${match[1].replaceAll("/", "__")}_frame_${match[2]}.jpg`;
  return `/artifacts/${encodeURIComponent(evaluationName)}/evaluation/overlays/${encodeURIComponent(file)}`;
}

async function loadInitialData() {
  state.datasets = await fetchJson("/api/datasets");
  state.evaluations = await fetchJson("/api/evaluations");
  state.validations = await fetchJson("/api/validations");
  populateSelects();

  if (state.validations.length) {
    el.sourceSelect.value = "validations";
    await setSource("validations");
    return;
  }

  if (state.evaluations.length) {
    el.sourceSelect.value = "evaluations";
    await setSource("evaluations");
    return;
  }

  await setSource("datasets");
}

function populateSelects() {
  el.runSelect.innerHTML = "";
  for (const run of state.datasets) {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = `${run.name} (${run.frame_count} frames)`;
    el.runSelect.appendChild(option);
  }

  el.evaluationSelect.innerHTML = "";
  for (const evaluation of state.evaluations) {
    const option = document.createElement("option");
    option.value = evaluation.name;
    option.textContent = `${evaluation.name} (${evaluation.frame_count} frames)`;
    el.evaluationSelect.appendChild(option);
  }

  el.validationSelect.innerHTML = "";
  for (const validation of state.validations) {
    const option = document.createElement("option");
    option.value = validation.name;
    option.textContent = `${validation.name} (${validation.frame_count} frames)`;
    el.validationSelect.appendChild(option);
  }
}

async function setSource(source) {
  state.source = source;
  el.sourceSelect.value = source;
  document.body.dataset.source = source;
  const evaluationMode = source === "evaluations";
  const validationMode = source === "validations";
  const predictionMode = evaluationMode || validationMode;
  el.runSelect.closest("label").hidden = predictionMode;
  el.evaluationSelect.closest("label").hidden = !evaluationMode;
  el.validationSelect.closest("label").hidden = !validationMode;
  el.evaluationPanel.hidden = !predictionMode;

  if (evaluationMode) {
    if (!state.evaluations.length) {
      el.status.textContent = "No evaluation artifacts found.";
      clearCanvas();
      return;
    }
    const latest = [...state.evaluations].sort((a, b) => b.modified - a.modified)[0];
    el.evaluationSelect.value = state.evaluationName || latest.name;
    await loadEvaluation(el.evaluationSelect.value);
    return;
  }

  if (validationMode) {
    if (!state.validations.length) {
      el.status.textContent = "No validation runs found.";
      clearCanvas();
      return;
    }
    const latest = [...state.validations].sort((a, b) => b.modified - a.modified)[0];
    el.validationSelect.value = state.validationName || latest.name;
    await loadValidation(el.validationSelect.value);
    return;
  }

  if (!state.datasets.length) {
    el.status.textContent = "No datasets found.";
    clearCanvas();
    return;
  }
  const latest = [...state.datasets].sort((a, b) => b.modified - a.modified)[0];
  el.runSelect.value = state.datasetRunName || latest.name;
  await loadRun(el.runSelect.value);
}

async function loadRun(datasetRunName) {
  state.datasetRunName = datasetRunName;
  state.metrics = null;
  const [datasetName, runName] = datasetRunName.split("/");
  state.frames = await fetchJson(
    `/api/datasets/${encodeURIComponent(datasetName)}/runs/${encodeURIComponent(runName)}/metadata`,
  );
  state.frameIndex = 0;
  updateSlider();
  el.status.textContent = `Loaded ${datasetRunName}`;
  await loadFrame(0);
}

async function loadEvaluation(evaluationName) {
  state.evaluationName = evaluationName;
  state.frames = await fetchJson(
    `/api/evaluations/${encodeURIComponent(evaluationName)}/predictions`,
  );
  state.metrics = await fetchJson(
    `/api/evaluations/${encodeURIComponent(evaluationName)}/metrics`,
  );
  state.frameIndex = 0;
  updateSlider();
  el.status.textContent = `Loaded evaluation ${evaluationName}`;
  updateEvaluationInfo();
  await loadFrame(0);
}

async function loadValidation(validationName) {
  state.validationName = validationName;
  state.frames = await fetchJson(
    `/api/validations/${encodeURIComponent(validationName)}/predictions`,
  );
  state.metrics = await fetchJson(
    `/api/validations/${encodeURIComponent(validationName)}/metrics`,
  );
  state.frameIndex = 0;
  updateSlider();
  el.status.textContent = `Loaded validation ${validationName}`;
  updateEvaluationInfo();
  await loadFrame(0);
}

function updateSlider() {
  el.frameSlider.min = "1";
  el.frameSlider.max = String(Math.max(1, state.frames.length));
  el.frameSlider.value = "1";
  el.frameLabel.textContent = `0 / ${state.frames.length}`;
}

async function loadFrame(index) {
  state.frameIndex = Math.max(0, Math.min(index, state.frames.length - 1));
  const frame = state.frames[state.frameIndex];

  if (!frame) {
    clearCanvas();
    return;
  }

  el.frameSlider.value = String(state.frameIndex + 1);
  el.frameLabel.textContent = `${state.frameIndex + 1} / ${state.frames.length}`;

  if (state.source === "evaluations") {
    await loadEvaluationFrame(frame);
  } else if (state.source === "validations") {
    await loadValidationFrame(frame);
  } else {
    await loadImage(frameUrl(state.datasetRunName, frame));
    drawRunFrame();
  }
}

async function loadEvaluationFrame(frame) {
  try {
    const frameUrl = evaluationFrameUrl(frame);
    if (!frameUrl) throw new Error("No evaluation frame URL");
    await loadImage(frameUrl);
    state.evaluationImageMode = "source";
  } catch {
    const overlayUrl = evaluationOverlayUrl(state.evaluationName, frame);
    if (!overlayUrl) throw new Error("No evaluation frame image");
    await loadImage(overlayUrl);
    state.evaluationImageMode = "baked-overlay";
  }
  drawEvaluationFrame();
}

async function loadValidationFrame(frame) {
  await loadImage(validationFrameUrl(state.validationName, frame));
  state.evaluationImageMode = "baked-overlay";
  drawEvaluationFrame();
}

function evaluationFrameUrl(frame) {
  const datasetRelativePath = frame.image_path.split(/datasets[\\/]/).pop();
  if (datasetRelativePath) return `/datasets/${datasetRelativePath.replaceAll("\\", "/")}`;

  const runRelativePath = frame.image_path.split(/runs[\\/]/).pop();
  if (!runRelativePath) return null;
  return `/datasets/${runRelativePath.replaceAll("\\", "/")}`;
}

function validationFrameUrl(validationName, frame) {
  const file = frame.frame_file || frame.frame_path.split(/[\\/]/).pop();
  return `/validation/${encodeURIComponent(validationName)}/frames/${encodeURIComponent(file)}`;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    state.image.onload = () => {
      if (state.image.naturalWidth && state.image.naturalHeight) {
        el.canvas.width = state.image.naturalWidth;
        el.canvas.height = state.image.naturalHeight;
      }
      resolve();
    };
    state.image.onerror = reject;
    state.image.src = `${src}?t=${Date.now()}`;
  });
}

function clearCanvas() {
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
}

function drawBaseImage() {
  clearCanvas();
  ctx.drawImage(state.image, 0, 0, el.canvas.width, el.canvas.height);
}

function drawRunFrame() {
  const frame = state.frames[state.frameIndex];
  drawBaseImage();

  const gates = frame.gates || [];
  gates.forEach((gate, index) => {
    if (el.onlyVisibleGates.checked && !gate.visible_in_frame) return;
    const color = colors[index % colors.length];

    if (el.showBbox.checked) {
      drawBox(gate.bbox_2d_px || gate.bbox_2d, color, [], gate.label);
    }

    if (el.showCorners.checked) {
      drawCorners(gate.corners || {});
    }
  });

  updateRunFrameInfo(frame);
  updateRunGateList(gates);
}

function drawEvaluationFrame() {
  const frame = state.frames[state.frameIndex];
  drawBaseImage();
  updateEvaluationLayerControls();
  drawEvaluationLayers(frame);
  updateEvaluationFrameInfo(frame);
  updateEvaluationGateList(frame);
}

function updateEvaluationLayerControls() {
  const disabled = state.evaluationImageMode !== "source";
  const title = disabled
    ? "This artifact points to a source frame that is not available locally, so the baked overlay image is being shown."
    : "";
  for (const checkbox of evaluationLayerCheckboxes()) {
    checkbox.disabled = disabled;
    checkbox.closest("label").title = title;
  }
}

function drawEvaluationLayers(frame) {
  if (!frame) return;
  if (state.evaluationImageMode !== "source") return;

  const scaleX = frame.frame_width_px ? el.canvas.width / frame.frame_width_px : 1;
  const scaleY = frame.frame_height_px ? el.canvas.height / frame.frame_height_px : 1;
  ctx.save();
  ctx.scale(scaleX, scaleY);

  if (el.showEvalTruthBbox.checked || el.showEvalTruthCorners.checked) {
    (frame.targets || []).forEach((target, index) => {
      const color = "#50ff64";
      if (el.showEvalTruthCorners.checked) {
        drawQuad(target.outer_corners_px, color, 2);
      }
      if (el.showEvalTruthBbox.checked) {
        drawXyxyBox(target.bbox_xyxy_px, color, 1, el.showEvalLabels.checked ? target.gate_label : "");
      }
      if (el.showEvalTruthCorners.checked) {
        drawPointSet(target.outer_corners_px, color, 2.5);
      }
    });
  }

  if (el.showEvalPredBbox.checked || el.showEvalPredCorners.checked) {
    (frame.detections || []).forEach((detection, index) => {
      const color = "#00dcff";
      if (el.showEvalPredCorners.checked) {
        drawQuad(detection.outer_corners_px, color, 3);
      }
      if (el.showEvalPredBbox.checked) {
        const label = el.showEvalLabels.checked ? predictionLabel(detection, index) : "";
        drawXyxyBox(detection.bbox_xyxy_px, color, 2, label);
      }
      if (el.showEvalPredCorners.checked) {
        drawPointSet(detection.outer_corners_px, color, 3);
      }
    });
  }

  if (el.showEvalLabels.checked) {
    drawEvaluationLegend(frame);
  }
  ctx.restore();
}

function drawBox(box, color, dash, label) {
  const x = box && (box.x_min_px ?? box.x_min);
  const y = box && (box.y_min_px ?? box.y_min);
  const width = box && (box.width_px ?? box.width);
  const height = box && (box.height_px ?? box.height);
  if (!box || width <= 0 || height <= 0) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = label === "visible" ? 2 : 1;
  ctx.setLineDash(dash);
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = color;
  ctx.font = "12px Arial";
  ctx.fillText(label, x + 4, Math.max(12, y - 4));
  ctx.restore();
}

function drawXyxyBox(box, color, width, label) {
  if (!box || box.length < 4) return;
  const x = box[0];
  const y = box[1];
  const w = box[2] - box[0];
  const h = box[3] - box[1];
  if (w <= 0 || h <= 0) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(x, y, w, h);
  if (label) {
    drawCanvasLabel(label, x + 4, Math.max(12, y - 4), color);
  }
  ctx.restore();
}

function drawQuad(points, color, width) {
  if (!points || points.length < 4) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  ctx.lineTo(points[1][0], points[1][1]);
  ctx.lineTo(points[3][0], points[3][1]);
  ctx.lineTo(points[2][0], points[2][1]);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawPointSet(points, color, radius) {
  if (!points) return;
  for (const point of points) {
    if (!point || point.length < 2) continue;
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fill();
    ctx.restore();
  }
}

function drawCanvasLabel(text, x, y, color) {
  ctx.save();
  ctx.font = "12px Arial";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#000";
  ctx.fillStyle = color;
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawEvaluationLegend(frame) {
  drawCanvasLabel(
    `GT gates: ${(frame.targets || []).length} (green)  Predicted: ${(frame.detections || []).length} (cyan)`,
    8,
    14,
    "#ffffff",
  );
}

function predictionLabel(detection, index) {
  let label = `#${index + 1} ${formatNumber(detection.score, 2)}`;
  const position = detection.relative_position_camera_frame_m || detection.relative_position_camera_frame_cm;
  const orientation = detection.relative_orientation_euler_deg;
  if (position && orientation) {
    const suffix = detection.relative_position_camera_frame_m ? "m" : "cm";
    const decimals = detection.relative_position_camera_frame_m ? 2 : 0;
    label +=
      ` R/U/F ${formatNumber(position.right, decimals)}/${formatNumber(position.up, decimals)}/${formatNumber(position.forward, decimals)}${suffix}` +
      ` Y180/P/R ${formatNumber(orientation.yaw_mod_180_deg, 1)}/${formatNumber(orientation.pitch_deg, 1)}/${formatNumber(orientation.roll_deg, 1)}`;
  }
  return label;
}

function drawCorners(corners) {
  for (const [name, corner] of cornerEntries(corners)) {
    const pixel = corner && (corner.pixel_px || corner.pixel);
    if (!corner || !pixel || !corner.inside_frame) continue;
    const visible = corner.visibility && corner.visibility.visible;
    const color = cornerColors[name] || "#ffffff";

    ctx.save();
    ctx.fillStyle = visible ? color : "rgba(255,255,255,0.35)";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, 2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fill();
    ctx.restore();
  }
}

function cornerEntries(corners) {
  if (!corners) return [];
  if (corners.outer || corners.inner) {
    const entries = [];
    for (const [ring, group] of Object.entries(corners)) {
      if (!group || typeof group !== "object") continue;
      for (const [location, corner] of Object.entries(group)) {
        entries.push([`Corner_${ring}_${location.toUpperCase()}`, corner]);
      }
    }
    return entries;
  }
  return Object.entries(corners);
}

function updateRunFrameInfo(frame) {
  writeDefinitionList(el.frameInfo, [
    ["Dataset Run", state.datasetRunName],
    ["Frame", frame.frame_number],
    ["Target", frame.target_gate],
    ["Saved", String(frame.frame_saved)],
    ["Camera", frame.camera && frame.camera.label],
  ]);
}

function updateEvaluationInfo() {
  const metrics = state.metrics && state.metrics.metrics;
  if (!metrics) {
    writeDefinitionList(el.evaluationInfo, [["Metrics", "Not found"]]);
    return;
  }

  if (state.source === "validations") {
    writeDefinitionList(el.evaluationInfo, [
      ["Run", state.validationName],
      ["Checkpoint epoch", state.metrics.checkpoint_epoch],
      ["Score threshold", formatNumber(state.metrics.score_threshold, 2)],
      ["Frames", metrics.frame_count],
      ["Predicted gates", metrics.predicted_gate_count],
      ["Empty frames", metrics.empty_frame_count],
      ["Detections / frame", formatNumber(metrics.detections_per_frame_mean, 2)],
      ["Score mean", formatNumber(metrics.score_mean, 3)],
      ["Pose solved", metrics.pose_solved_count],
      ["Model ms / frame", formatNumber(metrics.model_milliseconds_per_frame_mean, 1)],
    ]);
    return;
  }

  writeDefinitionList(el.evaluationInfo, [
    ["Artifact", state.evaluationName],
    ["Checkpoint epoch", state.metrics.checkpoint_epoch],
    ["Score threshold", formatNumber(state.metrics.score_threshold, 2)],
    ["Precision", formatNumber(metrics.detection_precision, 3)],
    ["Recall", formatNumber(metrics.detection_recall, 3)],
    ["F1", formatNumber(metrics.detection_f1, 3)],
    ["Pred / target", `${metrics.predicted_gate_count} / ${metrics.target_gate_count}`],
  ]);
}

function updateEvaluationFrameInfo(frame) {
  if (state.source === "validations") {
    writeDefinitionList(el.frameInfo, [
      ["Sample", frame.sample_id],
      ["Image", "annotated validation frame"],
      ["Source", frame.image_path && frame.image_path.split(/[\\/]/).pop()],
      ["Detections", (frame.detections || []).length],
    ]);
    return;
  }

  const matches = matchFrameDetections(frame);
  writeDefinitionList(el.frameInfo, [
    ["Sample", frame.sample_id],
    ["Image", state.evaluationImageMode === "source" ? "source frame" : "baked overlay"],
    ["Targets", (frame.targets || []).length],
    ["Detections", (frame.detections || []).length],
    ["Matched", matches.matched.length],
    ["False positives", matches.unmatchedDetections.length],
    ["Missed", matches.unmatchedTargets.length],
  ]);
}

function updateRunGateList(gates) {
  el.gateList.innerHTML = "";
  gates.forEach((gate, index) => {
    if (el.onlyVisibleGates.checked && !gate.visible_in_frame) return;

    const item = document.createElement("div");
    item.className = "gate";
    const visible = gate.visible_in_frame ? "visible" : "not visible";
    const visibilityTag = getVisibilityTag(gate);
    item.innerHTML = `
      <strong style="color:${colors[index % colors.length]}">${escapeHtml(gate.label)}</strong>
      <span>${visible}</span>
      ${renderVisibilityTag(visibilityTag)}
    `;
    el.gateList.appendChild(item);
  });
}

function updateEvaluationGateList(frame) {
  el.gateList.innerHTML = "";

  if (state.source === "validations") {
    const detections = frame.detections || [];
    detections.forEach((detection, index) => {
      const item = document.createElement("div");
      item.className = "gate gate-match";
      item.innerHTML = `
        <strong>Prediction ${index + 1} score ${formatNumber(detection.score, 3)}</strong>
        ${renderPoseBlock("Estimate", detection.relative_position_camera_frame_m || detection.relative_position_camera_frame_cm, detection.relative_orientation_euler_deg, detection.relative_position_camera_frame_m ? "m" : "cm")}
        <span>Reprojection error: ${formatNumber(detection.pose_reprojection_error_px, 2)} px</span>
      `;
      el.gateList.appendChild(item);
    });
    if (!detections.length) {
      const item = document.createElement("div");
      item.className = "gate";
      item.innerHTML = "<strong>No predicted gates</strong>";
      el.gateList.appendChild(item);
    }
    return;
  }

  const matches = matchFrameDetections(frame);

  for (const match of matches.matched) {
    el.gateList.appendChild(renderMatch(frame, match));
  }

  for (const targetIndex of matches.unmatchedTargets) {
    const item = document.createElement("div");
    item.className = "gate gate-missed";
    const target = frame.targets[targetIndex];
    item.innerHTML = `
      <strong>Missed truth: ${escapeHtml(target.gate_label)}</strong>
      ${renderVisibilityTag(getVisibilityTag(target))}
      ${renderPoseBlock("Truth", target.relative_position_camera_frame_m || target.relative_position_camera_frame_cm, target.relative_orientation_euler_deg, target.relative_position_camera_frame_m ? "m" : "cm")}
    `;
    el.gateList.appendChild(item);
  }

  for (const detectionIndex of matches.unmatchedDetections) {
    const item = document.createElement("div");
    item.className = "gate gate-false-positive";
    const detection = frame.detections[detectionIndex];
    item.innerHTML = `
      <strong>False positive ${detectionIndex + 1} score ${formatNumber(detection.score, 3)}</strong>
      ${renderVisibilityTag(getVisibilityTag(detection))}
      ${renderPoseBlock("Estimate", detection.relative_position_camera_frame_m || detection.relative_position_camera_frame_cm, detection.relative_orientation_euler_deg, detection.relative_position_camera_frame_m ? "m" : "cm")}
      <span>Reprojection error: ${formatNumber(detection.pose_reprojection_error_px, 2)} px</span>
    `;
    el.gateList.appendChild(item);
  }
}

function renderMatch(frame, match) {
  const target = frame.targets[match.targetIndex];
  const detection = frame.detections[match.detectionIndex];
  const positionError = vectorError(
    detection.relative_position_camera_frame_m || detection.relative_position_camera_frame_cm,
    target.relative_position_camera_frame_m || target.relative_position_camera_frame_cm,
    ["right", "up", "forward"],
  );
  const orientationError = orientationErrors(
    detection.relative_orientation_euler_deg,
    target.relative_orientation_euler_deg,
  );
  const item = document.createElement("div");
  item.className = "gate gate-match";
  item.innerHTML = `
    <strong>${escapeHtml(target.gate_label)} -> detection ${match.detectionIndex + 1}</strong>
    <span>IoU ${formatNumber(match.iou, 3)} · score ${formatNumber(detection.score, 3)} · reproj ${formatNumber(detection.pose_reprojection_error_px, 2)} px</span>
    <div class="tag-row">
      ${renderVisibilityTag(getVisibilityTag(target), "Truth")}
      ${renderVisibilityTag(getVisibilityTag(detection), "Estimate")}
    </div>
    ${renderPoseBlock("Truth", target.relative_position_camera_frame_m || target.relative_position_camera_frame_cm, target.relative_orientation_euler_deg, target.relative_position_camera_frame_m ? "m" : "cm")}
    ${renderPoseBlock("Estimate", detection.relative_position_camera_frame_m || detection.relative_position_camera_frame_cm, detection.relative_orientation_euler_deg, detection.relative_position_camera_frame_m ? "m" : "cm")}
    <div class="metric-grid">
      <span>Position error</span><b>${formatVector(positionError.values, 2)} m</b>
      <span>Position L2</span><b>${formatNumber(positionError.l2, 2)} m</b>
      <span>Orientation error</span><b>${formatVector(orientationError, 1)} deg</b>
    </div>
  `;
  return item;
}

function getVisibilityTag(item) {
  const direct =
    item.visibility_class ??
    item.visibility_label ??
    item.visibility_tag ??
    item.visibility ??
    item.visible_state ??
    item.gate_visibility ??
    item.predicted_visibility ??
    item.target_visibility;

  if (typeof direct === "string") return normalizeVisibilityTag(direct);
  if (direct && typeof direct === "object") {
    const nested =
      direct.class ??
      direct.label ??
      direct.tag ??
      direct.state ??
      direct.name ??
      direct.value;
    if (typeof nested === "string") return normalizeVisibilityTag(nested);
  }

  if (item.visible_in_frame === false) return "off-screen";
  if (item.visible === false) return "occluded";

  const corners = item.corners && cornerEntries(item.corners).map(([, corner]) => corner);
  if (corners && corners.length) {
    const outerCorners = corners.filter((corner) => corner && (corner.pixel_px || corner.pixel));
    const insideCount = outerCorners.filter((corner) => corner.inside_frame).length;
    const visibleCount = outerCorners.filter(
      (corner) => corner.visibility && corner.visibility.visible,
    ).length;
    if (insideCount > 0 && insideCount < outerCorners.length) return "partial";
    if (insideCount === 0) return "off-screen";
    if (visibleCount > 0 && visibleCount < outerCorners.length) return "occluded";
    if (visibleCount === 0 && outerCorners.length) return "occluded";
    return "visible";
  }

  if (item.outer_corners_px) {
    const points = item.outer_corners_px;
    const width = el.canvas.width || state.image.naturalWidth || 1920;
    const height = el.canvas.height || state.image.naturalHeight || 1080;
    const insideCount = points.filter(([x, y]) => x >= 0 && x < width && y >= 0 && y < height).length;
    if (insideCount > 0 && insideCount < points.length) return "partial";
    if (insideCount === 0) return "off-screen";
  }

  return null;
}

function normalizeVisibilityTag(value) {
  const normalized = value.toLowerCase().replaceAll("_", "-").trim();
  if (["partially-visible", "partially-visible-gate", "partial-visible"].includes(normalized)) {
    return "partial";
  }
  if (["fully-visible", "in-frame"].includes(normalized)) return "visible";
  if (["not-visible", "offscreen", "off-screen", "outside-frame"].includes(normalized)) {
    return "off-screen";
  }
  return normalized;
}

function renderVisibilityTag(tag, prefix = "") {
  if (!tag) return "";
  const label = prefix ? `${prefix}: ${tag}` : tag;
  const className = `visibility-tag visibility-${tag.replaceAll(" ", "-")}`;
  return `<span class="${className}">${escapeHtml(label)}</span>`;
}

function renderPoseBlock(label, position, orientation, unit = "m") {
  if (!position || !orientation) return `<span>${label}: no pose</span>`;
  const decimals = unit === "m" ? 2 : 1;
  return `
    <div class="pose-block">
      <span>${label}</span>
      <code>R/U/F ${formatNumber(position.right, decimals)} / ${formatNumber(position.up, decimals)} / ${formatNumber(position.forward, decimals)} ${unit}</code>
      <code>Y/P/R ${formatNumber(orientation.yaw_mod_180_deg, 1)} / ${formatNumber(orientation.pitch_deg, 1)} / ${formatNumber(orientation.roll_deg, 1)} deg</code>
    </div>
  `;
}

function matchFrameDetections(frame) {
  const candidates = [];
  const detections = frame.detections || [];
  const targets = frame.targets || [];
  for (let detectionIndex = 0; detectionIndex < detections.length; detectionIndex += 1) {
    for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
      const iou = bboxIou(
        detections[detectionIndex].bbox_xyxy_px,
        targets[targetIndex].bbox_xyxy_px,
      );
      if (iou >= 0.5) candidates.push({ detectionIndex, targetIndex, iou });
    }
  }

  candidates.sort((a, b) => b.iou - a.iou);
  const usedDetections = new Set();
  const usedTargets = new Set();
  const matched = [];

  for (const candidate of candidates) {
    if (usedDetections.has(candidate.detectionIndex) || usedTargets.has(candidate.targetIndex)) {
      continue;
    }
    matched.push(candidate);
    usedDetections.add(candidate.detectionIndex);
    usedTargets.add(candidate.targetIndex);
  }

  return {
    matched,
    unmatchedDetections: detections
      .map((_, index) => index)
      .filter((index) => !usedDetections.has(index)),
    unmatchedTargets: targets
      .map((_, index) => index)
      .filter((index) => !usedTargets.has(index)),
  };
}

function bboxIou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function vectorError(estimated, truth, keys) {
  if (!estimated || !truth) return { values: [null, null, null], l2: null };
  const values = keys.map((key) => Math.abs(estimated[key] - truth[key]));
  const l2 = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  return { values, l2 };
}

function orientationErrors(estimated, truth) {
  if (!estimated || !truth) return [null, null, null];
  return [
    yaw180Error(estimated.yaw_mod_180_deg, truth.yaw_mod_180_deg),
    circularError(estimated.pitch_deg, truth.pitch_deg),
    circularError(estimated.roll_deg, truth.roll_deg),
  ];
}

function circularError(a, b) {
  return Math.abs((((a - b + 180) % 360) + 360) % 360 - 180);
}

function yaw180Error(a, b) {
  return Math.abs((((a - b + 90) % 180) + 180) % 180 - 90);
}

function writeDefinitionList(list, rows) {
  list.innerHTML = "";
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value ?? "";
    list.append(dt, dd);
  }
}

function formatVector(values, digits) {
  return values.map((value) => formatNumber(value, digits)).join(" / ");
}

function formatNumber(value, digits) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "n/a";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildCornerLegend() {
  el.cornerLegend.innerHTML = "";
  for (const [name, color] of Object.entries(cornerColors)) {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `
      <span class="legend-swatch" style="background:${color}"></span>
      <span class="legend-label">${escapeHtml(name.replace("Corner_", ""))}</span>
    `;
    el.cornerLegend.appendChild(item);
  }
}

el.sourceSelect.addEventListener("change", () => setSource(el.sourceSelect.value));
el.runSelect.addEventListener("change", () => loadRun(el.runSelect.value));
el.evaluationSelect.addEventListener("change", () => loadEvaluation(el.evaluationSelect.value));
el.validationSelect.addEventListener("change", () => loadValidation(el.validationSelect.value));
el.frameSlider.addEventListener("input", () => loadFrame(Number(el.frameSlider.value) - 1));

function evaluationLayerCheckboxes() {
  return [
  el.showEvalTruthBbox,
  el.showEvalTruthCorners,
  el.showEvalPredBbox,
  el.showEvalPredCorners,
  el.showEvalLabels,
  ];
}

for (const checkbox of [
  el.showBbox,
  el.showCorners,
  el.onlyVisibleGates,
  ...evaluationLayerCheckboxes(),
]) {
  checkbox.addEventListener("change", () => {
    if (state.source === "evaluations" || state.source === "validations") {
      drawEvaluationFrame();
    } else {
      drawRunFrame();
    }
  });
}

window.addEventListener("keydown", (event) => {
  const keyActions = {
    ArrowLeft: -1,
    ArrowRight: 1,
    ArrowDown: -10,
    ArrowUp: 10,
  };

  if (!(event.key in keyActions)) return;

  event.preventDefault();
  loadFrame(state.frameIndex + keyActions[event.key]);
});

buildCornerLegend();
loadInitialData().catch((error) => {
  console.error(error);
  el.status.textContent = error.message;
});
