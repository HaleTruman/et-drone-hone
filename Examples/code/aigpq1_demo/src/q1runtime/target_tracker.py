from __future__ import annotations

from dataclasses import asdict, dataclass

from .target_mapper import LocalVisionTarget


@dataclass(frozen=True)
class TrackedTarget:
    target: LocalVisionTarget
    updated_monotonic_s: float
    update_count: int


class TargetTracker:
    """Keeps the last usable vision target alive across short perception gaps."""

    def __init__(self, hold_s: float = 3.0):
        self.hold_s = float(hold_s)
        self._latest: TrackedTarget | None = None
        self.update_count = 0

    def update(self, target: LocalVisionTarget, *, now_s: float) -> TrackedTarget:
        self.update_count += 1
        self._latest = TrackedTarget(
            target=target,
            updated_monotonic_s=float(now_s),
            update_count=self.update_count,
        )
        return self._latest

    def clear(self) -> None:
        self._latest = None

    def latest(self, *, now_s: float) -> TrackedTarget | None:
        if self._latest is None:
            return None
        if self.age_s(now_s=now_s) > self.hold_s:
            return None
        return self._latest

    def age_s(self, *, now_s: float) -> float | None:
        if self._latest is None:
            return None
        return max(0.0, float(now_s) - self._latest.updated_monotonic_s)

    def snapshot(self, *, now_s: float | None = None) -> dict[str, object]:
        age = None if now_s is None else self.age_s(now_s=now_s)
        return {
            "hold_s": self.hold_s,
            "update_count": self.update_count,
            "age_s": age,
            "active": False if now_s is None else self.latest(now_s=now_s) is not None,
            "latest": None if self._latest is None else asdict(self._latest),
        }
