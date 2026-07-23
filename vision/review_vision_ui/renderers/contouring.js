(function registerContouringRenderer(global) {
  const Shared = global.VisionReview.shared;

  function drawContour(ctx, contour, color, options = {}) {
    if (!contour || !Array.isArray(contour.pointsPx)) {
      return;
    }
    Shared.drawPath(ctx, contour.pointsPx, {
      stroke: color,
      width: options.width || 2,
      close: true,
      dash: options.dash || null,
      fill: options.fill || null,
    });
    if (Array.isArray(contour.centerPx)) {
      Shared.drawPoint(ctx, contour.centerPx, { fill: color, radius: options.radius || 2.5 });
    }
  }

  function render(context, panel) {
    const { ctx } = Shared.prepareCanvas(context, panel);
    const payload = Shared.stagePayload(context, "contouring");
    const observations = Shared.observations(payload);
    if (!observations.length) {
      Shared.showNote(panel, "No contours were emitted for this frame.");
      return;
    }

    observations.forEach((observation, index) => {
      const color = Shared.colorForIndex(index);
      Shared.drawBox(ctx, observation.bbox_px, {
        stroke: Shared.rgba(color, 0.55),
        width: 1,
        dash: [3, 4],
      });
      drawContour(ctx, observation.outer, color, { fill: Shared.rgba(color, 0.05), width: 2.5 });
      (observation.voids || []).forEach((voidContour) => {
        drawContour(ctx, voidContour, "#e65041", { width: 1.8, dash: [6, 3], radius: 2 });
      });
      (observation.additional_outers || []).forEach((outer) => {
        drawContour(ctx, outer, "#7458d6", { width: 1.8, dash: [3, 3], radius: 2 });
      });
      if (Array.isArray(observation.quad_points_px)) {
        Shared.drawPath(ctx, observation.quad_points_px, {
          stroke: "#ffffff",
          width: 1.3,
          close: true,
          dash: [5, 4],
        });
      }
      const anchor =
        observation.outer && Array.isArray(observation.outer.centerPx)
          ? observation.outer.centerPx
          : Shared.bboxAnchor(observation.bbox_px);
      const metrics = observation.metrics || {};
      Shared.drawLabel(
        ctx,
        `${observation.bbox_id}\nouter ${metrics.outerCount || 0} void ${metrics.voidCount || 0}`,
        anchor[0],
        anchor[1],
        { border: Shared.rgba(color, 0.55) }
      );
    });
    Shared.hideNote(panel);
  }

  global.VisionStageRenderers.contouring = { render };
})(window);
