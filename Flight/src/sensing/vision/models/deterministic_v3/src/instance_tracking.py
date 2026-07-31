import argparse
import json
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from .schema import EllipseEstimate, EllipseTrackRecord, TrackedVoidDetection

def read_frames(path):
    rows = json.loads(path.read_text())
    frames = [[] for _ in range(rows[0]["frame_count"])] if rows else []
    for row in rows:
        for key in ("frame_size_px", "center_px", "semi_axes_px"):
            row[key] = tuple(row[key])
        item = EllipseEstimate(**row)
        frames[item.frame_index].append(item)
    return frames

def ellipse_mask(item):
    width, height = item.frame_size_px
    mask = np.zeros((height, width), np.uint8)
    center = tuple(round(value) for value in item.center_px)
    axes = tuple(round(value) for value in item.semi_axes_px)
    cv2.ellipse(mask, center, axes, item.angle_degrees, 0, 360, 255, cv2.FILLED)
    return mask

def ellipse_iou(left, right):
    a, b = ellipse_mask(left), ellipse_mask(right)
    union = cv2.countNonZero(cv2.bitwise_or(a, b))
    return cv2.countNonZero(cv2.bitwise_and(a, b)) / union if union else 0.0

class InstanceTracker:
    def __init__(self, min_iou=0.1):
        self.min_iou = min_iou
        self.previous = []
        self.next_id = 0

    def update(self, frame):
        candidates = sorted(
            (ellipse_iou(old[0], new), old_i, new_i)
            for old_i, old in enumerate(self.previous)
            for new_i, new in enumerate(frame)
        )
        matches, old_used, new_used = {}, set(), set()
        for score, old_i, new_i in reversed(candidates):
            if score < self.min_iou or old_i in old_used or new_i in new_used:
                continue
            matches[new_i] = (self.previous[old_i], score)
            old_used.add(old_i)
            new_used.add(new_i)
        current, records = [], []
        for index, item in enumerate(frame):
            prior, score = matches.get(index, (None, 0.0))
            track_id = prior[1] if prior else \
                f"{item.run_id}-track-{self.next_id:05d}"
            count = prior[2] + 1 if prior else 1
            self.next_id += not prior
            records.append(EllipseTrackRecord(
                track_id, item.frame_index, count, item.ellipse_id, item.center_px,
                prior[0].ellipse_id if prior else None,
                prior[0].center_px if prior else None, round(score, 6)))
            current.append((item, track_id, count))
        self.previous = current
        return records

    def update_geometry(self, detections):
        tracks = self.update([item.ellipse for item in detections])
        return [TrackedVoidDetection(track, geometry)
                for track, geometry in zip(tracks, detections)]

def track_frames(frames, min_iou):
    tracker = InstanceTracker(min_iou)
    return [record for frame in frames for record in tracker.update(frame)]

def tracking_payload(records):
    matched = [record.iou_score for record in records
               if record.previous_ellipse_id is not None]
    return {"mean_iou": round(sum(matched) / len(matched), 6) if matched else 0.0,
            "records": [asdict(record) for record in records]}

def publish_tracked(records, output):
    tracks = [record.track for record in records]
    payload = tracking_payload(tracks)
    (output / "instance_tracks.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (output / "tracked_void_detections.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n")
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimates", type=Path)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    records = track_frames(read_frames(args.estimates), args.min_iou)
    payload = tracking_payload(records)
    output = args.output or args.estimates.with_name("instance_tracks.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Published {len(records)} records to {output} (mean IoU {payload['mean_iou']}).")

if __name__ == "__main__":
    main()
