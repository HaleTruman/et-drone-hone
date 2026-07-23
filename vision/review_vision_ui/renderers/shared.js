(function registerShared(global) {
  const review = (global.VisionReview = global.VisionReview || {});
  global.VisionStageRenderers = global.VisionStageRenderers || {};

  const fallbackPalette = [
    "#ff40ff",
    "#ffe600",
    "#00d0ff",
    "#ff9300",
    "#24b76c",
    "#e65041",
    "#7458d6",
    "#f5f5f5",
  ];

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatNumber(value, digits = 2) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return "n/a";
    }
    return parsed.toFixed(digits);
  }

  function normalizeHex(value, fallback = "#ffffff") {
    if (typeof value !== "string" || value.length === 0) {
      return fallback;
    }
    return value.startsWith("#") ? value : `#${value}`;
  }

  function rgbFromHex(hex, fallback = [255, 255, 255]) {
    const clean = normalizeHex(hex).replace("#", "");
    if (clean.length === 3) {
      return clean.split("").map((part) => parseInt(`${part}${part}`, 16));
    }
    if (clean.length !== 6) {
      return fallback;
    }
    const red = parseInt(clean.slice(0, 2), 16);
    const green = parseInt(clean.slice(2, 4), 16);
    const blue = parseInt(clean.slice(4, 6), 16);
    if ([red, green, blue].some((part) => !Number.isFinite(part))) {
      return fallback;
    }
    return [red, green, blue];
  }

  function rgba(hex, alpha = 1) {
    const [red, green, blue] = rgbFromHex(hex);
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function prepareCanvas(context, panel) {
    const canvas = panel.canvas;
    const width = number(context.imageWidth, 640);
    const height = number(context.imageHeight, 360);
    if (canvas.width !== width) {
      canvas.width = width;
    }
    if (canvas.height !== height) {
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    return { canvas, ctx, width, height };
  }

  function clearCanvas(context, panel) {
    return prepareCanvas(context, panel);
  }

  function showNote(panel, message) {
    if (!panel.note) {
      return;
    }
    panel.note.textContent = message || "";
    panel.note.classList.toggle("hidden", !message);
  }

  function hideNote(panel) {
    showNote(panel, "");
  }

  function stagePayload(context, stageName) {
    const stage = context.stagePayloads && context.stagePayloads[stageName];
    return stage ? stage.payload : null;
  }

  function observations(payload, fallbackKey) {
    if (!payload || typeof payload !== "object") {
      return [];
    }
    if (Array.isArray(payload.observations)) {
      return payload.observations;
    }
    if (fallbackKey && Array.isArray(payload[fallbackKey])) {
      return payload[fallbackKey];
    }
    return [];
  }

  function classColor(context, prefix, fallback) {
    const classes =
      context &&
      context.manifest &&
      context.manifest.preset &&
      Array.isArray(context.manifest.preset.classes)
        ? context.manifest.preset.classes
        : [];
    const record = classes.find((entry) => String(entry.prefix) === String(prefix));
    return normalizeHex(record && record.color, fallback || "#ffffff");
  }

  function colorForIndex(index) {
    return fallbackPalette[Math.abs(index) % fallbackPalette.length];
  }

  function colorForBit(context, bitIndex) {
    const layers =
      context &&
      context.manifest &&
      context.manifest.preset &&
      Array.isArray(context.manifest.preset.maskLayers)
        ? context.manifest.preset.maskLayers
        : [];
    const layer = layers.find((entry) => Number(entry.bit) === Number(bitIndex));
    if (layer && layer.prefix) {
      return classColor(context, layer.prefix, colorForIndex(bitIndex));
    }
    return colorForIndex(bitIndex);
  }

  function hashColor(value) {
    const text = String(value || "");
    let hash = 0;
    for (let index = 0; index < text.length; index += 1) {
      hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
    }
    const hue = hash % 360;
    return `hsl(${hue}, 74%, 51%)`;
  }

  function qualityColor(quality) {
    const value = number(quality, 0);
    if (value >= 0.85) {
      return "#24b76c";
    }
    if (value >= 0.55) {
      return "#f0a62a";
    }
    return "#e65041";
  }

  function validPoint(point) {
    return (
      Array.isArray(point) &&
      point.length >= 2 &&
      Number.isFinite(Number(point[0])) &&
      Number.isFinite(Number(point[1]))
    );
  }

  function validPoints(points) {
    return Array.isArray(points) ? points.filter(validPoint) : [];
  }

  function drawPath(ctx, points, options = {}) {
    const clean = validPoints(points);
    if (clean.length < 2) {
      return;
    }
    ctx.save();
    ctx.beginPath();
    clean.forEach((point, index) => {
      const x = number(point[0]);
      const y = number(point[1]);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (options.close) {
      ctx.closePath();
    }
    if (options.dash) {
      ctx.setLineDash(options.dash);
    }
    if (options.fill) {
      ctx.fillStyle = options.fill;
      ctx.fill();
    }
    ctx.strokeStyle = options.stroke || "#ffffff";
    ctx.lineWidth = options.width || 2;
    ctx.stroke();
    ctx.restore();
  }

  function drawBox(ctx, bbox, options = {}) {
    if (!Array.isArray(bbox) || bbox.length < 4) {
      return;
    }
    const x0 = number(bbox[0]);
    const y0 = number(bbox[1]);
    const x1 = number(bbox[2]);
    const y1 = number(bbox[3]);
    ctx.save();
    if (options.dash) {
      ctx.setLineDash(options.dash);
    }
    if (options.fill) {
      ctx.fillStyle = options.fill;
      ctx.fillRect(x0, y0, Math.max(1, x1 - x0 + 1), Math.max(1, y1 - y0 + 1));
    }
    ctx.strokeStyle = options.stroke || "#ffffff";
    ctx.lineWidth = options.width || 2;
    ctx.strokeRect(x0, y0, Math.max(1, x1 - x0 + 1), Math.max(1, y1 - y0 + 1));
    ctx.restore();
  }

  function drawPoint(ctx, point, options = {}) {
    if (!validPoint(point)) {
      return;
    }
    const radius = options.radius || 3;
    ctx.save();
    ctx.beginPath();
    ctx.arc(number(point[0]), number(point[1]), radius, 0, Math.PI * 2);
    ctx.fillStyle = options.fill || "#ffffff";
    ctx.fill();
    if (options.stroke) {
      ctx.strokeStyle = options.stroke;
      ctx.lineWidth = options.width || 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawCross(ctx, point, options = {}) {
    if (!validPoint(point)) {
      return;
    }
    const x = number(point[0]);
    const y = number(point[1]);
    const size = options.size || 5;
    ctx.save();
    ctx.strokeStyle = options.stroke || "#ffffff";
    ctx.lineWidth = options.width || 1.5;
    ctx.beginPath();
    ctx.moveTo(x - size, y);
    ctx.lineTo(x + size, y);
    ctx.moveTo(x, y - size);
    ctx.lineTo(x, y + size);
    ctx.stroke();
    ctx.restore();
  }

  function centerOfPoints(points) {
    const clean = validPoints(points);
    if (!clean.length) {
      return null;
    }
    const sums = clean.reduce(
      (accumulator, point) => [accumulator[0] + number(point[0]), accumulator[1] + number(point[1])],
      [0, 0]
    );
    return [sums[0] / clean.length, sums[1] / clean.length];
  }

  function bboxAnchor(bbox) {
    if (!Array.isArray(bbox) || bbox.length < 4) {
      return [4, 16];
    }
    return [number(bbox[0]) + 3, number(bbox[1]) - 4];
  }

  function drawLabel(ctx, text, x, y, options = {}) {
    const lines = String(text || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (!lines.length) {
      return;
    }
    ctx.save();
    ctx.font = options.font || "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    const paddingX = 5;
    const paddingY = 3;
    const lineHeight = 14;
    const textWidth = Math.max(...lines.map((line) => ctx.measureText(line).width));
    const boxWidth = textWidth + paddingX * 2;
    const boxHeight = lines.length * lineHeight + paddingY * 2;
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    let boxX = clamp(number(x), 0, Math.max(0, canvasWidth - boxWidth));
    let boxY = number(y) - boxHeight;
    if (boxY < 0) {
      boxY = number(y) + 6;
    }
    boxY = clamp(boxY, 0, Math.max(0, canvasHeight - boxHeight));
    ctx.fillStyle = options.background || "rgba(16, 20, 24, 0.78)";
    ctx.fillRect(boxX, boxY, boxWidth, boxHeight);
    ctx.strokeStyle = options.border || "rgba(255, 255, 255, 0.18)";
    ctx.lineWidth = 1;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
    ctx.fillStyle = options.color || "#ffffff";
    lines.forEach((line, index) => {
      ctx.fillText(line, boxX + paddingX, boxY + paddingY + 10 + index * lineHeight);
    });
    ctx.restore();
  }

  function observationMap(items, idField = "bbox_id") {
    const map = new Map();
    items.forEach((item) => {
      if (item && item[idField]) {
        map.set(String(item[idField]), item);
      }
    });
    return map;
  }

  review.shared = {
    bboxAnchor,
    centerOfPoints,
    classColor,
    clearCanvas,
    clamp,
    colorForBit,
    colorForIndex,
    drawBox,
    drawCross,
    drawLabel,
    drawPath,
    drawPoint,
    formatNumber,
    hashColor,
    hideNote,
    observationMap,
    observations,
    prepareCanvas,
    qualityColor,
    rgbFromHex,
    rgba,
    showNote,
    stagePayload,
    validPoints,
  };
})(window);
