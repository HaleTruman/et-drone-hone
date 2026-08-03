from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .target_mapper import LocalVisionTarget


@dataclass(frozen=True)
class TargetFilterResult:
    accepted: bool
    target: LocalVisionTarget | None
    reason: str
    jump_m: float | None = None


class TargetFilter:
    """Rejects implausible target jumps and smooths accepted local-NED targets."""

    def __init__(self, *, max_jump_m: float = 15.0, smoothing_alpha: float = 0.35):
        self.max_jump_m = float(max_jump_m)
        self.smoothing_alpha = float(smoothing_alpha)
        self._latest_position: np.ndarray | None = None

    def reset(self) -> None:
        self._latest_position = None

    def update(self, target: LocalVisionTarget) -> TargetFilterResult:
        current = np.asarray(target.position_local_ned_m, dtype=float)
        if self._latest_position is None:
            self._latest_position = current
            return TargetFilterResult(True, target, "accepted_initial", 0.0)
        jump = float(np.linalg.norm(current - self._latest_position))
        if jump > self.max_jump_m:
            return TargetFilterResult(False, None, "rejected_target_jump", jump)
        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        smoothed = alpha * current + (1.0 - alpha) * self._latest_position
        self._latest_position = smoothed
        return TargetFilterResult(
            True,
            replace(target, position_local_ned_m=tuple(float(value) for value in smoothed)),
            "accepted_smoothed",
            jump,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "max_jump_m": self.max_jump_m,
            "smoothing_alpha": self.smoothing_alpha,
            "latest_position_local_ned_m": None if self._latest_position is None else self._latest_position.tolist(),
        }
