"""Rolling in-memory instance tracking stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised by script-mode imports
    from .schema import BBoxFrame, InstanceFrame, InstanceObservation, PipelinePreset, PoseFrame, utc_now
except ImportError:  # pragma: no cover
    from schema import BBoxFrame, InstanceFrame, InstanceObservation, PipelinePreset, PoseFrame, utc_now


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    try:
        return math.sqrt(sum((float(left) - float(right)) ** 2 for left, right in zip(a, b)))
    except (TypeError, ValueError):
        return None


def _iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = [int(value) for value in a]
    bx0, by0, bx1, by1 = [int(value) for value in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    intersection = iw * ih
    area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
    area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
    return float(intersection) / float(max(1, area_a + area_b - intersection))


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, min(1.0, number))


@dataclass
class _Track:
    instance_id: str
    bbox_px: list[int]
    center_px: list[float]
    xyzCameraM: list[float] | None
    rpyCameraDeg: list[float] | None
    last_frame_ordinal: int
    age_frames: int = 1


class InstanceTracker:
    def __init__(self, preset: PipelinePreset):
        self.settings = dict(preset.instanceTracking)
        self.tracks: dict[str, _Track] = {}
        self.next_index = 1

    def _new_id(self) -> str:
        value = f"gate-{self.next_index:04d}"
        self.next_index += 1
        return value

    def _score(self, bbox: dict[str, Any], pose: dict[str, Any], track: _Track) -> dict[str, Any]:
        center_distance = _distance(bbox.get("center_px"), track.center_px)
        iou = _iou(bbox.get("bbox_px", [0, 0, 0, 0]), track.bbox_px)
        pose_distance = _distance(pose.get("xyzCameraM"), track.xyzCameraM) if pose.get("available") else None
        rpy_distance = _distance(pose.get("rpyCameraDeg"), track.rpyCameraDeg) if pose.get("available") else None
        max_2d = float(self.settings.get("max2dDistancePx", 90.0))
        max_3d = float(self.settings.get("max3dDistanceM", 3.0))
        max_rpy = float(self.settings.get("maxRpyDeltaDeg", 45.0))
        center_score = 0.0 if center_distance is None else max(0.0, 1.0 - center_distance / max(1.0, max_2d))
        pose_score = 0.0 if pose_distance is None else max(0.0, 1.0 - pose_distance / max(0.01, max_3d))
        rpy_score = 0.0 if rpy_distance is None else max(0.0, 1.0 - rpy_distance / max(1.0, max_rpy))
        iou_score = min(1.0, iou / max(0.001, float(self.settings.get("minIou", 0.02))))
        score = max(center_score * 0.65 + iou_score * 0.25 + pose_score * 0.10, pose_score * 0.75 + rpy_score * 0.25)
        return {
            "instanceId": track.instance_id,
            "score": clean_float(score, 6),
            "signals": {
                "centerDistancePx": None if center_distance is None else clean_float(center_distance, 4),
                "center2d": clean_float(center_score, 6),
                "iou": clean_float(iou, 6),
                "iouScore": clean_float(iou_score, 6),
                "poseDistanceM": None if pose_distance is None else clean_float(pose_distance, 6),
                "pose3d": clean_float(pose_score, 6),
                "rpyDeltaDeg": None if rpy_distance is None else clean_float(rpy_distance, 4),
                "rpy": clean_float(rpy_score, 6),
            },
        }

    def process(self, bbox_frame: BBoxFrame, pose_frame: PoseFrame) -> InstanceFrame:
        pose_by_bbox = {item.bbox_id: item.to_dict() for item in pose_frame.observations}
        max_gap = int(self.settings.get("maxFrameGap", 5))
        min_score = float(self.settings.get("minAssociationScore", 0.45))
        active_tracks = [
            track
            for track in self.tracks.values()
            if bbox_frame.frame_ordinal - track.last_frame_ordinal <= max_gap
        ]
        claimed: set[str] = set()
        observations: list[InstanceObservation] = []
        for index, bbox in enumerate(bbox_frame.observations):
            bbox_payload = bbox.to_dict()
            pose_payload = pose_by_bbox.get(bbox.bbox_id, {"bbox_id": bbox.bbox_id, "available": False, "reason": "missing-pose"})
            candidate_scores = sorted(
                [self._score(bbox_payload, pose_payload, track) for track in active_tracks if track.instance_id not in claimed],
                key=lambda item: float(item["score"]),
                reverse=True,
            )
            chosen = candidate_scores[0] if candidate_scores and float(candidate_scores[0]["score"]) >= min_score else None
            if chosen is None:
                instance_id = self._new_id()
                tracking_status = "new"
                association_score = 0.0
                age = 1
                gap = 0
            else:
                instance_id = str(chosen["instanceId"])
                claimed.add(instance_id)
                tracking_status = "matched"
                association_score = float(chosen["score"])
                previous = self.tracks[instance_id]
                age = previous.age_frames + 1
                gap = bbox_frame.frame_ordinal - previous.last_frame_ordinal
            fit_quality = pose_payload.get("fitQuality") if isinstance(pose_payload.get("fitQuality"), dict) else {}
            pose_quality = _clamp01(fit_quality.get("overall"), 0.0) if pose_payload.get("available") else 0.0
            fill_ratio = _clamp01((bbox_payload.get("quality") or {}).get("fillRatio"), 0.0)
            observation_quality = max(0.1, 0.35 * fill_ratio + 0.65 * pose_quality) if pose_payload.get("available") else 0.25 * fill_ratio
            observation = InstanceObservation(
                observation_id=f"{bbox_frame.frame_id}-obs-{index + 1:03d}",
                instance_id=instance_id,
                bbox_id=bbox.bbox_id,
                bbox=bbox_payload,
                pose=pose_payload,
                association_score=clean_float(association_score, 6),
                observationQuality=clean_float(observation_quality, 6),
                tracking_status=tracking_status,
                age_frames=age,
                missed_frame_gap=gap,
                candidate_scores=candidate_scores[:5],
            )
            observations.append(observation)
            self.tracks[instance_id] = _Track(
                instance_id=instance_id,
                bbox_px=list(bbox.bbox_px),
                center_px=list(bbox.center_px),
                xyzCameraM=pose_payload.get("xyzCameraM") if pose_payload.get("available") else None,
                rpyCameraDeg=pose_payload.get("rpyCameraDeg") if pose_payload.get("available") else None,
                last_frame_ordinal=bbox_frame.frame_ordinal,
                age_frames=age,
            )
        stale = [
            instance_id
            for instance_id, track in self.tracks.items()
            if bbox_frame.frame_ordinal - track.last_frame_ordinal > max_gap
        ]
        for instance_id in stale:
            del self.tracks[instance_id]
        instances = [item.to_dict() for item in observations]
        return InstanceFrame(
            run_id=bbox_frame.run_id,
            frame_ordinal=bbox_frame.frame_ordinal,
            frame_id=bbox_frame.frame_id,
            source_path=bbox_frame.source_path,
            image_width=bbox_frame.image_width,
            image_height=bbox_frame.image_height,
            created_at=utc_now(),
            timing_ms={},
            observations=observations,
            instances=instances,
            active_track_count=len(self.tracks),
            settings=dict(self.settings),
        )

