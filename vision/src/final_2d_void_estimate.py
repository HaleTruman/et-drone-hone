"""Judges each tracked void into one bounded, stable final_void_id slot with its own
confidence-scored position — voids only, independent of instance_tracking.py. Recreates its own
camera_odometry.py/track_estimate.py usage rather than reading instance_tracking.py's
motion_trail/forecast; instance_tracking.py stays untouched. Runs live, once per frame, in
parallel with instance_tracking.py's own update() — see void_north_star.html."""

from __future__ import annotations

from math import hypot

import camera_odometry as codo
from schema import FinalVoidEstimate, VoidAnalysis
from track_estimate import TrackEstimate, WEIGHT_DECAY_RATE

MAX_FINAL_VOID_SLOTS = 5  # the bounded threshold -- own constant, independent of instance_tracking.MAX_TRACKED_VOIDS
FORECAST_HORIZON_FRAMES = 15  # both mc and te predict this far ahead; also the gap-tolerance window before eviction
MC_WEIGHT = 0.5  # "nearly equal weight" between cv and mc -- v1, flagged for calibration
TIER_BASE = {"weak": 0.3, "moderate": 0.6, "strong": 0.9}  # forecast-tier base, mirrors final_2d_estimate.py

# Ghost pool: safely re-merging a track whose path went disjointed. Deliberately self-contained --
# not reusing any instance_tracking.py tolerance, since those were calibrated for a different job
# (matching within one frame) and may not be the right shape for bridging a gap. Tune manually.
GHOST_POOL_SIZE = MAX_FINAL_VOID_SLOTS  # never more ghosts than there are slots to reclaim
GHOST_MATCH_TOLERANCE_PX = 40.0  # max distance from a ghost's current projection to reclaim its slot
GHOST_AREA_AGREEMENT_MIN = 0.5  # min pixel-count agreement (min/max ratio) required to reclaim a slot


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _new_entry() -> dict:
    return {
        "last_center": None, "last_frame_path": None, "missing": 0,
        "reliable": False, "estimate": TrackEstimate(),
        "mc_trail": None, "last_confidence": None, "last_pixel_count": None,
    }


class FinalVoidEstimator:
    def __init__(self) -> None:
        self._telemetry_checked = False
        self._telemetry = None
        self._pool: dict[str, dict] = {}
        self._slots: dict[str, int] = {}
        self._ghosts: list[dict] = []

    def _ensure_telemetry(self, frame_path: str) -> None:
        if self._telemetry_checked:
            return
        self._telemetry_checked = True
        path = codo.find_telemetry_for_frame(frame_path)
        if path is not None:
            self._telemetry = codo.load_telemetry(path)

    def _build_mc_trail(self, center, frame_path, upcoming_frame_paths):
        if self._telemetry is None or not upcoming_frame_paths:
            return None
        points = []
        for future_path in upcoming_frame_paths[:FORECAST_HORIZON_FRAMES]:
            predicted = codo.predict_center(self._telemetry, frame_path, center, future_path)
            if predicted is not None:
                points.append((future_path, predicted))
        return points or None

    def _observe(self, entry, void, frame_path, t, upcoming_frame_paths):
        mc_point, mc_term = None, 0.5
        if self._telemetry is not None and entry["last_frame_path"] is not None:
            result = codo.compensated_match(self._telemetry, entry["last_frame_path"], entry["last_center"], frame_path, void.center)
            if result is not None:
                compensated_distance, predicted_center, _ = result
                raw_distance = hypot(void.center[0] - entry["last_center"][0], void.center[1] - entry["last_center"][1])
                ratio = compensated_distance / max(raw_distance, 1e-6)
                mc_term = _clamp01(1.0 - min(ratio, 1.0))
                mc_point = predicted_center

        cv_term = void.track_match.area_agreement if void.track_match is not None else 0.5
        confidence = _clamp01(MC_WEIGHT * mc_term + (1.0 - MC_WEIGHT) * cv_term)

        entry["estimate"].observe(t, void.center)
        if mc_point is not None:
            entry["estimate"].observe(t, mc_point)
        entry["mc_trail"] = self._build_mc_trail(void.center, frame_path, upcoming_frame_paths)
        entry["last_center"] = void.center
        entry["last_frame_path"] = frame_path
        entry["last_confidence"] = confidence
        entry["last_pixel_count"] = void.pixel_count
        entry["missing"] = 0
        entry["reliable"] = void.reliable
        return confidence

    def _gap_estimate(self, track_id, entry, t):
        if entry["mc_trail"] and entry["missing"] <= len(entry["mc_trail"]):
            point = entry["mc_trail"][entry["missing"] - 1][1]
            confidence = _clamp01((entry["last_confidence"] or 0.0) * WEIGHT_DECAY_RATE ** entry["missing"])
            return FinalVoidEstimate(0, point, "mc", confidence, False, track_id)

        tier = entry["estimate"].confidence()
        if tier is not None:
            point = entry["estimate"].project(t)
            if point is not None:
                trail_len = len(entry["mc_trail"]) if entry["mc_trail"] else 0
                steps_past = max(0, entry["missing"] - trail_len)
                confidence = _clamp01(TIER_BASE[tier] * WEIGHT_DECAY_RATE ** steps_past)
                return FinalVoidEstimate(0, point, "forecast", confidence, False, track_id)
        return None

    def _ghost(self, track_id, entry, slot_number) -> None:
        """A slot's occupant just left tracking (clipped or aged past the horizon). Keep enough
        of its state to keep projecting it briefly, so a genuine continuation can reclaim the
        slot instead of spending a new one."""
        self._ghosts.append({
            "track_id": track_id, "slot": slot_number, "pixel_count": entry["last_pixel_count"],
            "missing": entry["missing"], "estimate": entry["estimate"],
            "mc_trail": entry["mc_trail"], "last_confidence": entry["last_confidence"],
        })
        self._ghosts = self._ghosts[-GHOST_POOL_SIZE:]

    def _check_ghosts(self, track_id, center, pixel_count, t):
        """A newly-eligible track_id, checked against recently-vacated slots' own continued
        projections -- never a silent identity merge, just a reclaimed slot number. Closest match
        within tolerance wins and is consumed so it can't be claimed twice."""
        best_index, best_slot, best_distance = None, None, None
        for i, ghost in enumerate(self._ghosts):
            estimate = self._gap_estimate(ghost["track_id"], ghost, t)
            if estimate is None:
                continue
            distance = hypot(center[0] - estimate.center[0], center[1] - estimate.center[1])
            if distance > GHOST_MATCH_TOLERANCE_PX:
                continue
            if not ghost["pixel_count"] or not pixel_count:
                continue
            area_agreement = min(pixel_count, ghost["pixel_count"]) / max(pixel_count, ghost["pixel_count"])
            if area_agreement < GHOST_AREA_AGREEMENT_MIN:
                continue
            if best_distance is None or distance < best_distance:
                best_index, best_slot, best_distance = i, ghost["slot"], distance
        if best_index is not None:
            self._ghosts.pop(best_index)
        return best_slot

    def _assign_slots(self, candidates: dict[str, FinalVoidEstimate]) -> None:
        for track_id in list(self._slots):
            if track_id not in candidates:
                del self._slots[track_id]
        for track_id in candidates:
            if track_id in self._slots or len(self._slots) >= MAX_FINAL_VOID_SLOTS:
                continue
            used = set(self._slots.values())
            free_number = next(n for n in range(1, MAX_FINAL_VOID_SLOTS + 1) if n not in used)
            self._slots[track_id] = free_number

    def update(self, analysis: VoidAnalysis, frame_path: str, upcoming_frame_paths=()) -> VoidAnalysis:
        self._ensure_telemetry(frame_path)
        t = codo.frame_epoch_seconds(frame_path)
        seen = set()
        candidates: dict[str, FinalVoidEstimate] = {}

        for ghost in self._ghosts:
            ghost["missing"] += 1
        self._ghosts = [g for g in self._ghosts if g["missing"] <= FORECAST_HORIZON_FRAMES]

        for region in analysis.regions:
            for void in region.voids:
                track_id = void.track_id
                if not track_id:
                    continue
                if void.mask_clipping:
                    # clipping is an immediate, hard exit -- the void leaves tracking entirely
                    entry = self._pool.pop(track_id, None)
                    slot_number = self._slots.pop(track_id, None)
                    if entry is not None and slot_number is not None:
                        self._ghost(track_id, entry, slot_number)
                    continue
                seen.add(track_id)
                entry = self._pool.setdefault(track_id, _new_entry())
                confidence = self._observe(entry, void, frame_path, t, upcoming_frame_paths)
                if void.reliable:
                    candidates[track_id] = FinalVoidEstimate(0, void.center, "cv", confidence, False, track_id)
                    if track_id not in self._slots:
                        reclaimed_slot = self._check_ghosts(track_id, void.center, void.pixel_count, t)
                        if reclaimed_slot is not None:
                            self._slots[track_id] = reclaimed_slot

        for track_id in list(self._pool):
            if track_id in seen:
                continue
            entry = self._pool[track_id]
            entry["missing"] += 1
            if entry["missing"] > FORECAST_HORIZON_FRAMES:
                slot_number = self._slots.pop(track_id, None)
                if slot_number is not None:
                    self._ghost(track_id, entry, slot_number)
                del self._pool[track_id]
                continue
            if not entry["reliable"]:
                continue
            estimate = self._gap_estimate(track_id, entry, t)
            if estimate is not None:
                candidates[track_id] = estimate

        self._assign_slots(candidates)
        for track_id, slot_number in self._slots.items():
            estimate = candidates[track_id]
            estimate.final_void_id = slot_number
            analysis.final_void_estimates.append(estimate)

        return analysis
