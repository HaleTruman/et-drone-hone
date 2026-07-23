(function registerClippingRenderer(global) {
  const Shared = global.VisionReview.shared;

  function drawMargin(ctx, width, height, margin) {
    if (!margin) {
      return;
    }
    ctx.save();
    ctx.fillStyle = "rgba(240, 166, 42, 0.08)";
    ctx.fillRect(0, 0, width, margin);
    ctx.fillRect(0, Math.max(0, height - margin), width, margin);
    ctx.fillRect(0, 0, margin, height);
    ctx.fillRect(Math.max(0, width - margin), 0, margin, height);
    ctx.strokeStyle = "rgba(240, 166, 42, 0.75)";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(0, margin);
    ctx.lineTo(width, margin);
    ctx.moveTo(0, height - margin);
    ctx.lineTo(width, height - margin);
    ctx.moveTo(margin, 0);
    ctx.lineTo(margin, height);
    ctx.moveTo(width - margin, 0);
    ctx.lineTo(width - margin, height);
    ctx.stroke();
    ctx.restore();
  }

  function drawSide(ctx, bbox, side, color, dash) {
    if (!Array.isArray(bbox) || bbox.length < 4) {
      return;
    }
    const x0 = Number(bbox[0]);
    const y0 = Number(bbox[1]);
    const x1 = Number(bbox[2]) + 1;
    const y1 = Number(bbox[3]) + 1;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    if (dash) {
      ctx.setLineDash(dash);
    }
    ctx.beginPath();
    if (side === "left") {
      ctx.moveTo(x0, y0);
      ctx.lineTo(x0, y1);
    } else if (side === "right") {
      ctx.moveTo(x1, y0);
      ctx.lineTo(x1, y1);
    } else if (side === "top") {
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y0);
    } else if (side === "bottom") {
      ctx.moveTo(x0, y1);
      ctx.lineTo(x1, y1);
    }
    ctx.stroke();
    ctx.restore();
  }

  function render(context, panel) {
    const { ctx, width, height } = Shared.prepareCanvas(context, panel);
    const payload = Shared.stagePayload(context, "clipping");
    const bboxPayload = Shared.stagePayload(context, "bboxing");
    const bboxes = Shared.observationMap(Shared.observations(bboxPayload), "bbox_id");
    const observations = Shared.observations(payload);
    const settings = payload && payload.settings ? payload.settings : {};
    const margin = Number(settings.marginPx || 0);

    drawMargin(ctx, width, height, margin);
    if (!payload) {
      Shared.showNote(panel, "No clipping payload is available for this frame.");
      return;
    }

    observations.forEach((observation) => {
      const bboxObservation = bboxes.get(String(observation.bbox_id));
      const bbox = bboxObservation && bboxObservation.bbox_px;
      if (!bbox) {
        return;
      }
      const severity = Number(observation.severity || 0);
      const sides = Array.isArray(observation.sides) ? observation.sides : [];
      const nearSides = Array.isArray(observation.near_sides) ? observation.near_sides : [];
      const color = sides.length ? "#e65041" : nearSides.length ? "#f0a62a" : "#687076";
      Shared.drawBox(ctx, bbox, {
        stroke: color,
        width: sides.length || nearSides.length ? 2.5 : 1.5,
        dash: observation.enabled === false ? [4, 4] : null,
      });
      nearSides.forEach((side) => drawSide(ctx, bbox, side, "#f0a62a", [5, 4]));
      sides.forEach((side) => drawSide(ctx, bbox, side, "#e65041"));
      const [labelX, labelY] = Shared.bboxAnchor(bbox);
      Shared.drawLabel(
        ctx,
        `${observation.bbox_id}\n${observation.status || "ok"} sev ${Shared.formatNumber(severity, 2)}`,
        labelX,
        labelY,
        { border: Shared.rgba(color, 0.55) }
      );
    });

    if (settings.enabled === false) {
      Shared.showNote(panel, "Clipping is disabled for this run; showing configured margin and bbox status.");
    } else {
      Shared.hideNote(panel);
    }
  }

  global.VisionStageRenderers.clipping = { render };
})(window);
