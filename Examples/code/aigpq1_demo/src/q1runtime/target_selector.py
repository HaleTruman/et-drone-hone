from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .target_mapper import LocalVisionTarget


@dataclass(frozen=True)
class TargetSelectionConfig:
    mode: str = "forward_progress"
    min_camera_forward_m: float = 0.10
    min_forward_m: float = 0.50
    velocity_min_mps: float = 0.50
    allow_camera_fallback_when_routed: bool = False


@dataclass(frozen=True)
class TargetSelectionResult:
    target: LocalVisionTarget | None
    reason: str
    candidate_count: int
    eligible_count: int
    route_direction_xy: tuple[float, float] | None = None
    selected_gate_id: str | None = None
    selected_ahead_m: float | None = None
    selected_lateral_m: float | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class TargetSelector:
    """Select exactly one control target from current-frame regressor candidates."""

    def __init__(self, config: TargetSelectionConfig | None = None):
        self.config = config or TargetSelectionConfig()
        self._route_direction_xy: np.ndarray | None = None

    def reset(self) -> None:
        self._route_direction_xy = None

    def select(
        self,
        targets: list[LocalVisionTarget],
        telemetry: Any,
        *,
        route_velocity_local_ned_mps: tuple[float, float, float] | list[float] | None = None,
    ) -> TargetSelectionResult:
        if not targets:
            return TargetSelectionResult(None, "no_candidate_targets", 0, 0)
        if self.config.mode == "camera_distance":
            target = min(targets, key=self._camera_distance_m)
            self._update_route_direction(telemetry, route_velocity_local_ned_mps=route_velocity_local_ned_mps)
            return TargetSelectionResult(
                target=target,
                reason="selected_nearest_camera_distance",
                candidate_count=len(targets),
                eligible_count=len(targets),
                selected_gate_id=target.gate_id,
            )
        if self.config.mode != "forward_progress":
            raise ValueError("target_selection_mode must be 'forward_progress' or 'camera_distance'")

        self._update_route_direction(telemetry, route_velocity_local_ned_mps=route_velocity_local_ned_mps)
        route_direction = self._route_direction_xy
        current_xy = self._current_xy(telemetry)
        scored: list[dict[str, Any]] = []
        for target in targets:
            camera_forward_m = float(target.position_camera_m[2])
            if camera_forward_m < float(self.config.min_camera_forward_m):
                continue
            local_xy = np.asarray(target.position_local_ned_m[:2], dtype=float)
            delta_xy = local_xy - current_xy
            if route_direction is None:
                ahead_m = None
                lateral_m = float(np.linalg.norm(delta_xy))
            else:
                ahead_m = float(np.dot(delta_xy, route_direction))
                lateral_vector = delta_xy - ahead_m * route_direction
                lateral_m = float(np.linalg.norm(lateral_vector))
            scored.append(
                {
                    "target": target,
                    "ahead_m": ahead_m,
                    "lateral_m": lateral_m,
                    "camera_distance_m": self._camera_distance_m(target),
                    "confidence": float(target.position_confidence),
                }
            )

        if not scored:
            return TargetSelectionResult(
                None,
                "no_candidate_in_front_of_camera",
                len(targets),
                0,
                route_direction_xy=self._route_payload(),
            )

        if route_direction is not None:
            eligible = [
                item
                for item in scored
                if item["ahead_m"] is not None and float(item["ahead_m"]) >= float(self.config.min_forward_m)
            ]
            if eligible:
                selected = min(
                    eligible,
                    key=lambda item: (
                        float(item["ahead_m"]),
                        float(item["lateral_m"]),
                        float(item["camera_distance_m"]),
                        -float(item["confidence"]),
                    ),
                )
                return self._result(
                    selected,
                    reason="selected_nearest_forward_progress",
                    candidate_count=len(targets),
                    eligible_count=len(eligible),
                )
            if not self.config.allow_camera_fallback_when_routed:
                return TargetSelectionResult(
                    None,
                    "no_forward_candidate_for_route",
                    len(targets),
                    0,
                    route_direction_xy=self._route_payload(),
                )

        selected = min(scored, key=lambda item: (float(item["camera_distance_m"]), -float(item["confidence"])))
        return self._result(
            selected,
            reason="selected_nearest_camera_distance_no_route",
            candidate_count=len(targets),
            eligible_count=len(scored),
        )

    def snapshot(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "route_direction_xy": self._route_payload()}

    def _update_route_direction(
        self,
        telemetry: Any,
        *,
        route_velocity_local_ned_mps: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        velocity = route_velocity_local_ned_mps
        if velocity is None:
            velocity = getattr(telemetry, "velocity_local_ned_mps", None)
        if velocity is None:
            return
        velocity_xy = np.asarray(velocity[:2], dtype=float)
        speed = float(np.linalg.norm(velocity_xy))
        if speed >= float(self.config.velocity_min_mps):
            self._route_direction_xy = velocity_xy / speed

    def _result(
        self,
        selected: dict[str, Any],
        *,
        reason: str,
        candidate_count: int,
        eligible_count: int,
    ) -> TargetSelectionResult:
        target = selected["target"]
        return TargetSelectionResult(
            target=target,
            reason=reason,
            candidate_count=candidate_count,
            eligible_count=eligible_count,
            route_direction_xy=self._route_payload(),
            selected_gate_id=target.gate_id,
            selected_ahead_m=None if selected["ahead_m"] is None else float(selected["ahead_m"]),
            selected_lateral_m=None if selected["lateral_m"] is None else float(selected["lateral_m"]),
        )

    def _route_payload(self) -> tuple[float, float] | None:
        if self._route_direction_xy is None:
            return None
        return tuple(float(value) for value in self._route_direction_xy)

    @staticmethod
    def _camera_distance_m(target: LocalVisionTarget) -> float:
        return float(np.linalg.norm(np.asarray(target.position_camera_m, dtype=float)))

    @staticmethod
    def _current_xy(telemetry: Any) -> np.ndarray:
        position = getattr(telemetry, "position_local_ned_m", None)
        if position is None:
            raise ValueError("Telemetry must include position_local_ned_m for target selection.")
        return np.asarray(position[:2], dtype=float)
