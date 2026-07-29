"""staleish Judges each tracked void's per-frame position from CV/MC/TE signals into one
confidence-scored FinalEstimate — the mechanism behind void track persistence and
reacquisition. Instances reuse the same tiers as supporting infrastructure; voids
are the mission."""

from __future__ import annotations

from schema import FinalEstimate
from track_estimate import WEIGHT_DECAY_RATE

MAX_FINAL_GAP_FRAMES = 10  # separate from VOID_MAX_MISSING even though it starts equal —
                           # "still tracked internally" is a weaker bar than "still worth reporting"
TIER_BASE = {"weak": 0.3, "moderate": 0.6, "strong": 0.9}  # forecast confidence base, v1 placeholder
MC_AGREEMENT_FLOOR = 0.5  # disagreement alone can't zero out CV's own evidence
CONFIDENCE_SMOOTHING = 0.7  # EMA blend rate — raw per-frame score is noisy, this is the fix


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def smooth_confidence(previous: float | None, raw: float) -> float:
    """Turns a jittery per-frame score into a stable trust signal. A continuously-tracked
    object's raw IOU-driven score can swing 0.5-0.9 frame to frame; this is the fix."""
    if previous is None:
        return raw
    return CONFIDENCE_SMOOTHING * previous + (1.0 - CONFIDENCE_SMOOTHING) * raw


def cv_confidence(track_match) -> float:
    """CV's own match quality, primed (not overridden) by MC's agreement when it exists."""
    if track_match is None:
        return 0.5  # first sighting — real CV data, just not yet corroborated by a match history
    ceiling = 3.0 if track_match.iou is not None else 2.0
    base = _clamp01(track_match.score / ceiling)
    mc = track_match.motion_compensation
    if mc is None:
        return base
    ratio = mc.compensated_distance_px / max(track_match.raw_distance_px, 1e-6)
    multiplier = 1.0 - (1.0 - MC_AGREEMENT_FLOOR) * min(ratio, 1.0)
    return _clamp01(base * multiplier)


def estimate_from_match(track_id, center, confidence, reliable=True) -> FinalEstimate | None:
    """Called only when a real detection existed this frame — tier "cv" if reliable.
    Not-yet-reliable sightings publish nothing (instances default to reliable=True — unaffected).
    Takes the already-smoothed confidence — the caller owns computing + smoothing it."""
    if not reliable:
        return None
    return FinalEstimate(track_id, center, "cv", confidence, 0)


def estimate_from_gap(track_id, missing, last_cv_confidence, motion_trail, estimate, now) -> FinalEstimate | None:
    """Called for a track not detected this frame. Walks mc-continuation, then forecast."""
    if missing > MAX_FINAL_GAP_FRAMES:
        return None

    if motion_trail is not None and missing <= len(motion_trail.points):
        point = motion_trail.points[missing - 1].predicted_center
        confidence = _clamp01((last_cv_confidence or 0.0) * WEIGHT_DECAY_RATE ** missing)
        return FinalEstimate(track_id, point, "mc", confidence, missing)

    if estimate is not None and estimate.confidence() is not None:
        point = estimate.project(now)
        if point is not None:
            trail_len = len(motion_trail.points) if motion_trail is not None else 0
            steps_past_trail = max(0, missing - trail_len)
            confidence = _clamp01(TIER_BASE[estimate.confidence()] * WEIGHT_DECAY_RATE ** steps_past_trail)
            return FinalEstimate(track_id, point, "forecast", confidence, missing)

    return None
