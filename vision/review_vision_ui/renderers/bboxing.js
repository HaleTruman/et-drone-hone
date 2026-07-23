(function registerBBoxingRenderer(global) {
  const Shared = global.VisionReview.shared;

  function render(context, panel) {
    const { ctx } = Shared.prepareCanvas(context, panel);
    const payload = Shared.stagePayload(context, "bboxing");
    const observations = Shared.observations(payload);
    if (!observations.length) {
      Shared.showNote(panel, "No bounding boxes were emitted for this frame.");
      return;
    }

    observations.forEach((observation, index) => {
      const color = Shared.classColor(context, observation.dominant_prefix, Shared.colorForIndex(index));
      Shared.drawBox(ctx, observation.bbox_px, {
        stroke: color,
        fill: Shared.rgba(color, 0.08),
        width: 2,
      });
      if (observation.quad_fit && Array.isArray(observation.quad_fit.pointsPx)) {
        Shared.drawPath(ctx, observation.quad_fit.pointsPx, {
          stroke: "#ffffff",
          width: 1.5,
          close: true,
          dash: [5, 4],
        });
      }
      Shared.drawCross(ctx, observation.center_px, { stroke: color, size: 5, width: 2 });
      const [labelX, labelY] = Shared.bboxAnchor(observation.bbox_px);
      const label = [
        `${observation.bbox_id || `bbox-${index + 1}`} ${observation.dominant_prefix || ""}`,
        `px ${observation.pixel_count || 0} fill ${Shared.formatNumber(observation.quality && observation.quality.fillRatio, 2)}`,
      ].join("\n");
      Shared.drawLabel(ctx, label, labelX, labelY, { border: Shared.rgba(color, 0.55) });
    });
    Shared.hideNote(panel);
  }

  global.VisionStageRenderers.bboxing = { render };
})(window);
