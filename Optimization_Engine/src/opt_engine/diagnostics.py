from __future__ import annotations

import numpy as np

from opt_engine.types import Constraints, SpeedProfile


def summarize_bindings(
    *, constraints: Constraints, curvature_1pm: np.ndarray, speed_profile: SpeedProfile
) -> dict[str, object]:
    tol = 1e-6
    v = np.asarray(speed_profile.v_mps, dtype=float)
    v_kappa = np.asarray(speed_profile.v_kappa_mps, dtype=float)

    bind_vmax = (v_kappa > constraints.v_max + tol) & (np.abs(v - constraints.v_max) <= 1e-3)
    bind_curv = (v_kappa <= constraints.v_max + tol) & (np.abs(v - v_kappa) <= 1e-3)

    return {
        "n_points": int(v.shape[0]),
        "bind_vmax_count": int(np.count_nonzero(bind_vmax)),
        "bind_curvature_count": int(np.count_nonzero(bind_curv)),
        "max_curvature_1pm": float(np.max(np.asarray(curvature_1pm, dtype=float))) if v.shape[0] else 0.0,
        "max_speed_mps": float(np.max(v)) if v.shape[0] else 0.0,
        "min_speed_mps": float(np.min(v)) if v.shape[0] else 0.0,
    }

