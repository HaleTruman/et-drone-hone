"""Build the public VisionResults contract from tracked instances."""

from __future__ import annotations

import math
from typing import Any

try:  # pragma: no cover - exercised by package imports
    from .schema import InstanceFrame, PipelinePreset, VisionGateResult, VisionResults, utc_now
except ImportError:  # pragma: no cover
    from schema import InstanceFrame, PipelinePreset, VisionGateResult, VisionResults, utc_now


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, min(1.0, number))


def _float3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def opencv_camera_to_result_camera(value: list[float]) -> list[float]:
    x_right, y_down, z_forward = [float(item) for item in value]
    return [x_right, -y_down, z_forward]


class VisionResultsBuilder:
    def __init__(self, preset: PipelinePreset):
        self.settings = dict(preset.flightBridge)

    def process(self, instance_frame: InstanceFrame) -> VisionResults:
        gates: list[VisionGateResult] = []
        for instance in instance_frame.instances:
            pose = instance.get("pose") if isinstance(instance.get("pose"), dict) else {}
            if not pose.get("available"):
                continue
            xyz = _float3(pose.get("xyzCameraM"))
            if xyz is None:
                continue
            try:
                depth = float(pose.get("depthM", xyz[2]))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(depth) or depth <= 0.0:
                continue
            fit_quality = pose.get("fitQuality") if isinstance(pose.get("fitQuality"), dict) else {}
            orientation = _float3(pose.get("rpyCameraDeg"))
            bbox = instance.get("bbox") if isinstance(instance.get("bbox"), dict) else {}
            gate_id = str(instance.get("instance_id") or instance.get("instanceId") or instance.get("observation_id") or "")
            trace = {
                "source_stage": "instance_tracking",
                "source_instance_id": instance.get("instance_id") or instance.get("instanceId"),
                "source_observation_id": instance.get("observation_id"),
                "source_bbox_id": instance.get("bbox_id") or bbox.get("bbox_id") or pose.get("bbox_id"),
                "tracking_status": instance.get("tracking_status"),
                "association_score": instance.get("association_score"),
                "observation_quality": instance.get("observationQuality"),
                "pose_bbox_id": pose.get("bbox_id"),
                "opencv_position_camera_m": xyz,
                "coordinate_transform": "opencv-camera-to-result-camera-flip-y",
            }
            gates.append(
                VisionGateResult(
                    gate_id=gate_id,
                    position_camera_m=opencv_camera_to_result_camera(xyz),
                    position_confidence=_clamp01(instance.get("observationQuality")),
                    orientation_camera=orientation,
                    orientation_confidence=_clamp01(fit_quality.get("overall")),
                    trace=trace,
                )
            )
        return VisionResults(
            run_id=instance_frame.run_id,
            frame_ordinal=instance_frame.frame_ordinal,
            frame_id=instance_frame.frame_id,
            source_path=instance_frame.source_path,
            image_width=instance_frame.image_width,
            image_height=instance_frame.image_height,
            created_at=utc_now(),
            timing_ms={},
            gates=gates,
            obstacles=[],
            trace={
                "source_stage": "instance_tracking",
                "coordinate_transform": "opencv-camera-to-result-camera-flip-y",
            },
        )
