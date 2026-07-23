(function registerInstanceRenderer(global) {
  const Shared = global.VisionReview.shared;

  function render(context, panel) {
    const { ctx } = Shared.prepareCanvas(context, panel);
    const payload = Shared.stagePayload(context, "instance_tracking");
    const observations = Shared.observations(payload, "instances");
    if (!observations.length) {
      Shared.showNote(panel, "No tracked instances were emitted for this frame.");
      return;
    }

    observations.forEach((observation) => {
      const color = Shared.hashColor(observation.instance_id || observation.observation_id || observation.bbox_id);
      const bbox = observation.bbox && observation.bbox.bbox_px;
      const pose = observation.pose || {};
      if (bbox) {
        Shared.drawBox(ctx, bbox, {
          stroke: color,
          fill: "rgba(255, 255, 255, 0.03)",
          width: 2.2,
        });
        Shared.drawCross(ctx, observation.bbox.center_px, { stroke: color, size: 5, width: 2 });
      }
      if (Array.isArray(pose.imagePointsPx) && pose.imagePointsPx.length >= 4) {
        Shared.drawPath(ctx, pose.imagePointsPx, {
          stroke: "#ffffff",
          width: 1.4,
          close: true,
          dash: [4, 4],
        });
      }
      const anchor = bbox ? Shared.bboxAnchor(bbox) : Shared.centerOfPoints(pose.imagePointsPx) || [4, 16];
      const label = [
        `${observation.instance_id || "untracked"} ${observation.tracking_status || ""} age ${observation.age_frames || 0}`,
        `q ${Shared.formatNumber(observation.observationQuality, 2)} assoc ${Shared.formatNumber(
          observation.association_score,
          2
        )}`,
      ].join("\n");
      Shared.drawLabel(ctx, label, anchor[0], anchor[1], { border: color });
    });
    Shared.hideNote(panel);
  }

  global.VisionStageRenderers.instance_tracking = { render };
})(window);
