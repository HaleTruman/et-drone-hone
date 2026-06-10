from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    max_horizontal_radius_m: float = 25.0
    min_z_ned_m: float = -8.0
    max_z_ned_m: float = 3.0
    max_velocity_mps: float = 8.0
    telemetry_timeout_s: float = 0.5
    stop_on_reset: bool = True


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    reason: str
    details: dict[str, Any]

    @property
    def should_stop(self) -> bool:
        return self.action in ("stop", "stop_and_disarm")


class SafetySupervisor:
    """Live-flight safety checks for q1runtime command output."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.last_reset_count: int | None = None
        self.last_telemetry_monotonic_s: float | None = None
        self.trip_count = 0
        self.last_decision = SafetyDecision("continue", "safety_disabled" if not self.config.enabled else "ok", {})

    def evaluate(self, telemetry: Any | None, *, now_s: float) -> SafetyDecision:
        if not self.config.enabled:
            return self._decision("continue", "safety_disabled", {})
        if telemetry is None:
            if self.last_telemetry_monotonic_s is None:
                return self._decision("stop", "missing_telemetry", {})
            age = float(now_s) - self.last_telemetry_monotonic_s
            if age > self.config.telemetry_timeout_s:
                return self._decision("stop", "stale_telemetry", {"age_s": age})
            return self._decision("continue", "ok", {"telemetry_age_s": age})

        self.last_telemetry_monotonic_s = float(now_s)
        reset_count = getattr(telemetry, "reset_count", None)
        if reset_count is not None:
            reset_count = int(reset_count)
            if self.last_reset_count is None:
                self.last_reset_count = reset_count
            elif reset_count != self.last_reset_count:
                previous = self.last_reset_count
                self.last_reset_count = reset_count
                if self.config.stop_on_reset:
                    return self._decision(
                        "stop",
                        "simulator_reset_detected",
                        {"previous_reset_count": previous, "reset_count": reset_count},
                    )

        position = getattr(telemetry, "position_local_ned_m", None)
        if position is not None:
            pos = np.asarray(position, dtype=float)
            horizontal_radius = float(np.linalg.norm(pos[0:2]))
            if horizontal_radius > self.config.max_horizontal_radius_m:
                return self._decision("stop_and_disarm", "horizontal_radius_exceeded", {"radius_m": horizontal_radius})
            z = float(pos[2])
            if z < self.config.min_z_ned_m or z > self.config.max_z_ned_m:
                return self._decision("stop_and_disarm", "z_bounds_exceeded", {"z_ned_m": z})

        velocity = getattr(telemetry, "velocity_local_ned_mps", None)
        if velocity is not None:
            speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
            if speed > self.config.max_velocity_mps or not math.isfinite(speed):
                return self._decision("stop_and_disarm", "velocity_limit_exceeded", {"speed_mps": speed})

        return self._decision("continue", "ok", {})

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                "enabled": self.config.enabled,
                "max_horizontal_radius_m": self.config.max_horizontal_radius_m,
                "min_z_ned_m": self.config.min_z_ned_m,
                "max_z_ned_m": self.config.max_z_ned_m,
                "max_velocity_mps": self.config.max_velocity_mps,
                "telemetry_timeout_s": self.config.telemetry_timeout_s,
                "stop_on_reset": self.config.stop_on_reset,
            },
            "last_reset_count": self.last_reset_count,
            "last_telemetry_monotonic_s": self.last_telemetry_monotonic_s,
            "trip_count": self.trip_count,
            "last_decision": {
                "action": self.last_decision.action,
                "reason": self.last_decision.reason,
                "details": self.last_decision.details,
            },
        }

    def _decision(self, action: str, reason: str, details: dict[str, Any]) -> SafetyDecision:
        decision = SafetyDecision(action, reason, details)
        if decision.should_stop and (self.last_decision.reason != reason or self.last_decision.action != action):
            self.trip_count += 1
        self.last_decision = decision
        return decision
