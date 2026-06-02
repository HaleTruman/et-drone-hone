from __future__ import annotations

import math

import numpy as np

from autonomy.opt_engine.types import Constraints, SpeedProfile


def speed_caps_from_curvature(curvature_1pm: np.ndarray, constraints: Constraints) -> tuple[np.ndarray, np.ndarray]:
    curvature = np.asarray(curvature_1pm, dtype=float)
    a_lat_max = float(constraints.effective_a_lat_max())
    if a_lat_max <= 0:
        raise ValueError("effective a_lat_max must be > 0")
    v_kappa = np.sqrt(a_lat_max / (np.abs(curvature) + constraints.kappa_epsilon))
    v_cap = np.minimum(constraints.v_max, v_kappa)
    return v_cap, v_kappa


def apply_yaw_rate_speed_cap(
    *, v_cap_mps: np.ndarray, dpsi_ds_radpm: np.ndarray, constraints: Constraints
) -> tuple[np.ndarray, np.ndarray]:
    if not constraints.heading_constrained:
        raise ValueError("apply_yaw_rate_speed_cap called but heading_constrained=False")
    if constraints.yaw_rate_max_rps <= 0:
        raise ValueError("yaw_rate_max_rps must be > 0 when heading_constrained=True")

    v_cap = np.asarray(v_cap_mps, dtype=float)
    dpsi_ds = np.asarray(dpsi_ds_radpm, dtype=float)
    if dpsi_ds.shape != v_cap.shape:
        raise ValueError("dpsi_ds_radpm must have the same shape as v_cap_mps")

    v_yaw = float(constraints.yaw_rate_max_rps) / (np.abs(dpsi_ds) + float(constraints.yaw_epsilon_radpm))
    v_cap2 = np.minimum(v_cap, v_yaw)
    return v_cap2, v_yaw


def solve_speed_profile(
    *,
    ds_m: np.ndarray,
    v_cap_mps: np.ndarray,
    v_kappa_mps: np.ndarray,
    constraints: Constraints,
    v_yaw_mps: np.ndarray | None = None,
) -> SpeedProfile:
    ds = np.asarray(ds_m, dtype=float)
    v_cap = np.asarray(v_cap_mps, dtype=float)
    v_kappa = np.asarray(v_kappa_mps, dtype=float)
    v_yaw = np.asarray(v_yaw_mps, dtype=float) if v_yaw_mps is not None else None

    if v_cap.ndim != 1:
        raise ValueError("v_cap_mps must be shape (N,)")
    if v_kappa.shape != v_cap.shape:
        raise ValueError("v_kappa_mps must match v_cap_mps shape")
    if v_yaw is not None and v_yaw.shape != v_cap.shape:
        raise ValueError("v_yaw_mps must match v_cap_mps shape when provided")
    if ds.shape != (v_cap.shape[0] - 1,):
        raise ValueError("ds_m must be shape (N-1,) for N=len(v_cap_mps)")

    if constraints.v_max <= 0:
        raise ValueError("v_max must be > 0")
    if constraints.a_fwd_max <= 0:
        raise ValueError("a_fwd_max must be > 0")
    if constraints.a_brake_max <= 0:
        raise ValueError("a_brake_max must be > 0")
    if constraints.effective_a_lat_max() <= 0:
        raise ValueError("effective a_lat_max must be > 0")

    v = np.array(v_cap, copy=True)

    # Forward pass (free start speed => start at cap)
    for k in range(v.shape[0] - 1):
        ds_k = float(ds[k])
        if ds_k <= 0:
            continue
        v_next_max = math.sqrt(max(0.0, v[k] * v[k] + 2.0 * constraints.a_fwd_max * ds_k))
        v[k + 1] = min(v[k + 1], v_next_max)

    # Backward pass (free end speed => end at forward result)
    for k in range(v.shape[0] - 2, -1, -1):
        ds_k = float(ds[k])
        if ds_k <= 0:
            continue
        v_prev_max = math.sqrt(max(0.0, v[k + 1] * v[k + 1] + 2.0 * constraints.a_brake_max * ds_k))
        v[k] = min(v[k], v_prev_max)

    v_mid = 0.5 * (v[:-1] + v[1:])
    v_mid = np.maximum(v_mid, constraints.v_min)
    time_s = float(np.sum(ds / v_mid))

    return SpeedProfile(v_mps=v, v_cap_mps=v_cap, v_kappa_mps=v_kappa, v_yaw_mps=v_yaw, time_s=time_s)
