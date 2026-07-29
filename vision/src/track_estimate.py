from __future__ import annotations

import numpy as np

MIN_HISTORY = 3
MAX_HISTORY = 12
FRAME_DT_S = 1.0 / 30.0  # nominal camera frame interval
WEIGHT_GRACE_FRAMES = 2  # this many most-recent frames' worth of history get full weight
WEIGHT_DECAY_RATE = 0.8  # falloff per additional frame-equivalent of age past the grace window


def _recency_weights(ts: np.ndarray) -> np.ndarray:
    """1.0 for the most recent WEIGHT_GRACE_FRAMES worth of history, then geometric falloff by
    age in frame-equivalents (real elapsed time / FRAME_DT_S, so a raw+mc pair sharing one
    timestamp gets equal weight instead of one looking older than the other)."""
    age_frames = (ts[-1] - ts) / FRAME_DT_S
    return WEIGHT_DECAY_RATE ** np.maximum(0.0, age_frames - (WEIGHT_GRACE_FRAMES - 1))


class TrackEstimate:
    """A void's own history (raw + mc points), fit to a loose next-point guess. Not a real track."""

    def __init__(self) -> None:
        self.history: list[tuple[float, float, float]] = []

    def observe(self, t: float, point: tuple[float, float]) -> None:
        self.history.append((t, point[0], point[1]))
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)

    def confidence(self) -> str | None:
        n = len(self.history)
        if n >= 10:
            return "strong"
        if n >= 5:
            return "moderate"
        if n >= MIN_HISTORY:
            return "weak"
        return None

    def project(self, t: float) -> tuple[float, float] | None:
        if self.confidence() is None:
            return None
        ts, xs, ys = (np.array(c) for c in zip(*self.history))
        weights = _recency_weights(ts)
        ax, bx = np.polyfit(ts, xs, 1, w=weights)
        ay, by = np.polyfit(ts, ys, 1, w=weights)
        return ax * t + bx, ay * t + by
