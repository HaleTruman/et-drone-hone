from __future__ import annotations

import numpy as np

from opt_engine.types import Constraints, SpeedProfile


def summarize_bindings(
    *, constraints: Constraints, curvature_1pm: np.ndarray, speed_profile: SpeedProfile
) -> dict[str, object]:
    v = np.asarray(speed_profile.v_mps, dtype=float)
    v_kappa = np.asarray(speed_profile.v_kappa_mps, dtype=float)
    v_cap = np.asarray(speed_profile.v_cap_mps, dtype=float)
    v_yaw = np.asarray(speed_profile.v_yaw_mps, dtype=float) if speed_profile.v_yaw_mps is not None else None

    at_cap = np.abs(v - v_cap) <= 1e-3
    cap_vmax = np.abs(v_cap - float(constraints.v_max)) <= 1e-6
    cap_curv = np.abs(v_cap - v_kappa) <= 1e-6
    cap_yaw = (np.abs(v_cap - v_yaw) <= 1e-6) if v_yaw is not None else np.zeros_like(at_cap, dtype=bool)

    bind_vmax = at_cap & cap_vmax
    bind_curv = at_cap & cap_curv
    bind_yaw = at_cap & cap_yaw

    return {
        "n_points": int(v.shape[0]),
        "bind_vmax_count": int(np.count_nonzero(bind_vmax)),
        "bind_curvature_count": int(np.count_nonzero(bind_curv)),
        "bind_yaw_count": int(np.count_nonzero(bind_yaw)),
        "max_curvature_1pm": float(np.max(np.asarray(curvature_1pm, dtype=float))) if v.shape[0] else 0.0,
        "max_speed_mps": float(np.max(v)) if v.shape[0] else 0.0,
        "min_speed_mps": float(np.min(v)) if v.shape[0] else 0.0,
    }
