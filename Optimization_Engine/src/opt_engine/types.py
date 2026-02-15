from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Frame = Literal["internal", "unreal"]
Units = Literal["m", "cm"]


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class Scenario:
    name: str
    frame: Frame
    units: Units
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class Constraints:
    v_max: float = 6.0
    a_fwd_max: float = 2.0
    a_brake_max: float = 3.0
    a_lat_max: float = 4.0
    kappa_epsilon: float = 1e-6
    v_min: float = 1e-3


@dataclass(frozen=True)
class SamplingConfig:
    samples_per_segment: int = 50


@dataclass(frozen=True)
class SampledPath:
    points_m: np.ndarray  # shape: (N, 3)
    s_m: np.ndarray  # shape: (N,)
    ds_m: np.ndarray  # shape: (N-1,)
    waypoint_indices: tuple[int, ...]


@dataclass(frozen=True)
class SpeedProfile:
    v_mps: np.ndarray  # shape: (N,)
    v_cap_mps: np.ndarray  # shape: (N,)
    v_kappa_mps: np.ndarray  # shape: (N,)
    time_s: float


@dataclass(frozen=True)
class OptimizationResult:
    scenario_name: str
    best_lambda: float
    time_s: float
    sampled_path: SampledPath
    curvature_1pm: np.ndarray  # shape: (N,)
    speed_profile: SpeedProfile
    diagnostics: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "best_lambda": float(self.best_lambda),
            "time_s": float(self.time_s),
            "path": {
                "points_m": self.sampled_path.points_m.tolist(),
                "s_m": self.sampled_path.s_m.tolist(),
                "ds_m": self.sampled_path.ds_m.tolist(),
                "waypoint_indices": list(self.sampled_path.waypoint_indices),
            },
            "curvature_1pm": self.curvature_1pm.tolist(),
            "speed_profile": {
                "v_mps": self.speed_profile.v_mps.tolist(),
                "v_cap_mps": self.speed_profile.v_cap_mps.tolist(),
                "v_kappa_mps": self.speed_profile.v_kappa_mps.tolist(),
                "time_s": float(self.speed_profile.time_s),
            },
            "diagnostics": self.diagnostics,
        }
