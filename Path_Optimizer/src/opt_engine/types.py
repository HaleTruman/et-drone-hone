from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import math

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
class Vec3:
    x: float
    y: float
    z: float

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class Target:
    actor_label: str
    actor_path: str
    position: Waypoint
    axis_x: Vec3
    axis_y: Vec3
    axis_z: Vec3


@dataclass(frozen=True)
class Scenario:
    name: str
    frame: Frame
    units: Units
    waypoints: tuple[Waypoint, ...]
    targets: tuple[Target, ...] | None = None
    origin_target: Target | None = None
    level: str | None = None
    mesh: str | None = None
    generated_at: str | None = None


@dataclass(frozen=True)
class Constraints:
    v_max: float = 6.0
    a_fwd_max: float = 2.0
    a_brake_max: float = 3.0
    a_lat_max: float = 4.0

    # Optional: hard geometry limit. If enabled, paths with κ(s) > κ_max are infeasible.
    r_min_m: float | None = None
    kappa_max_1pm: float | None = None

    # Optional: derive a_lat_max from tilt limit via a_lat_max = g*tan(theta_max).
    use_theta_max: bool = False
    theta_max_deg: float = 35.0
    g_mps2: float = 9.81

    # Optional: yaw-rate speed cap when heading is constrained to follow direction of travel.
    heading_constrained: bool = False
    yaw_rate_max_rps: float = 2.0

    kappa_epsilon: float = 1e-6
    v_min: float = 1e-3
    yaw_epsilon_radpm: float = 1e-6

    def effective_a_lat_max(self) -> float:
        if not self.use_theta_max:
            return float(self.a_lat_max)
        theta_rad = math.radians(float(self.theta_max_deg))
        return float(self.g_mps2) * math.tan(theta_rad)

    def effective_kappa_max_1pm(self) -> float | None:
        if self.r_min_m is not None and self.kappa_max_1pm is not None:
            raise ValueError("Specify only one of r_min_m or kappa_max_1pm")
        if self.r_min_m is not None:
            r = float(self.r_min_m)
            if r <= 0:
                raise ValueError("r_min_m must be > 0")
            return 1.0 / r
        if self.kappa_max_1pm is not None:
            k = float(self.kappa_max_1pm)
            if k <= 0:
                raise ValueError("kappa_max_1pm must be > 0")
            return k
        return None


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
    v_yaw_mps: np.ndarray | None
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
        v_yaw = self.speed_profile.v_yaw_mps.tolist() if self.speed_profile.v_yaw_mps is not None else None
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
                "v_yaw_mps": v_yaw,
                "time_s": float(self.speed_profile.time_s),
            },
            "diagnostics": self.diagnostics,
        }
