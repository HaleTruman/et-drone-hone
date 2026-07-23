(function registerPoseRenderer(global) {
  const Shared = global.VisionReview.shared;

  function fallbackBbox(context, bboxId) {
    const bboxPayload = Shared.stagePayload(context, "bboxing");
    const contourPayload = Shared.stagePayload(context, "contouring");
    const bboxMap = Shared.observationMap(Shared.observations(bboxPayload), "bbox_id");
    const contourMap = Shared.observationMap(Shared.observations(contourPayload), "bbox_id");
    const bboxObservation = bboxMap.get(String(bboxId));
    if (bboxObservation && bboxObservation.bbox_px) {
      return bboxObservation.bbox_px;
    }
    const contourObservation = contourMap.get(String(bboxId));
    return contourObservation && contourObservation.bbox_px;
  }

  function render(context, panel) {
    const { ctx } = Shared.prepareCanvas(context, panel);
    const payload = Shared.stagePayload(context, "pose_estimation");
    const observations = Shared.observations(payload);
    if (!observations.length) {
      Shared.showNote(panel, "No pose observations were emitted for this frame.");
      return;
    }

    observations.forEach((observation) => {
      const quality = observation.fitQuality && Number(observation.fitQuality.overall);
      const color = observation.available ? Shared.qualityColor(quality) : "#e65041";
      const points = Array.isArray(observation.imagePointsPx) ? observation.imagePointsPx : [];
      const center = Shared.centerOfPoints(points);
      if (points.length >= 4) {
        Shared.drawPath(ctx, points, {
          stroke: color,
          width: 2.5,
          close: true,
          fill: Shared.rgba(color, 0.06),
        });
        points.forEach((point) => Shared.drawPoint(ctx, point, { fill: color, stroke: "#101418", radius: 3 }));
      } else {
        Shared.drawBox(ctx, fallbackBbox(context, observation.bbox_id), {
          stroke: color,
          width: 2,
          dash: [5, 4],
        });
      }

      const xyz = Array.isArray(observation.xyzCameraM) ? observation.xyzCameraM : [];
      const rpy = Array.isArray(observation.rpyCameraDeg) ? observation.rpyCameraDeg : [];
      const labelPoint = center || Shared.bboxAnchor(fallbackBbox(context, observation.bbox_id));
      const label = observation.available
        ? [
            `${observation.bbox_id} z ${Shared.formatNumber(observation.depthM, 2)}m q ${Shared.formatNumber(quality, 2)}`,
            `xyz ${xyz.map((value) => Shared.formatNumber(value, 1)).join(",")} rpy ${rpy
              .map((value) => Shared.formatNumber(value, 1))
              .join(",")}`,
          ].join("\n")
        : `${observation.bbox_id} unavailable\n${observation.reason || "pose could not be solved"}`;
      Shared.drawLabel(ctx, label, labelPoint[0], labelPoint[1], { border: Shared.rgba(color, 0.55) });
    });
    Shared.hideNote(panel);
  }

  global.VisionStageRenderers.pose_estimation = { render };
})(window);
