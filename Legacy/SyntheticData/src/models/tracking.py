"""Lightweight constant-velocity tracking for all detected gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    velocity: np.ndarray
    missed_frames: int = 0

    @property
    def predicted_bbox(self) -> np.ndarray:
        return self.bbox + self.velocity


class GateTracker:
    """Assign stable IDs to detections in an ordered image sequence."""

    def __init__(self, minimum_iou: float = 0.1, maximum_missed_frames: int = 3):
        self.minimum_iou = minimum_iou
        self.maximum_missed_frames = maximum_missed_frames
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[dict[str, object]]) -> list[dict[str, object]]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                iou = _bbox_iou(
                    track.predicted_bbox,
                    np.asarray(detection["bbox_xyxy"], dtype=np.float32),
                )
                if iou >= self.minimum_iou:
                    candidates.append((iou, track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
            track = self._tracks[track_id]
            track.velocity = 0.5 * track.velocity + 0.5 * (box - track.bbox)
            track.bbox = box.copy()
            track.missed_frames = 0
            detection["track_id"] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.bbox = track.predicted_bbox
                track.missed_frames += 1
                if track.missed_frames > self.maximum_missed_frames:
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
            self._tracks[track_id] = _Track(
                track_id=track_id,
                bbox=box.copy(),
                velocity=np.zeros((4,), dtype=np.float32),
            )
            detection["track_id"] = track_id
        return detections


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
