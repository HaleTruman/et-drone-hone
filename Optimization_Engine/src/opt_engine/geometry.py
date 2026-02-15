from __future__ import annotations

import numpy as np


def arc_length(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] < 2:
        s = np.zeros((points.shape[0],), dtype=float)
        ds = np.zeros((0,), dtype=float)
        return s, ds

    deltas = points[1:] - points[:-1]
    ds = np.linalg.norm(deltas, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, ds


def curvature_from_polyline(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n = points.shape[0]
    kappa = np.zeros((n,), dtype=float)
    if n < 3:
        return kappa

    for k in range(1, n - 1):
        a = points[k - 1]
        b = points[k]
        c = points[k + 1]
        ab = b - a
        ac = c - a
        bc = c - b
        lab = float(np.linalg.norm(ab))
        lbc = float(np.linalg.norm(bc))
        lac = float(np.linalg.norm(ac))
        denom = lab * lbc * lac
        if denom <= 0.0:
            kappa[k] = 0.0
            continue
        cross_mag = float(np.linalg.norm(np.cross(ab, ac)))
        kappa[k] = 2.0 * cross_mag / denom

    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa

