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


def heading_and_dpsi_ds_xy(
    *, points: np.ndarray, s_m: np.ndarray, eps_xy: float = 1e-9, eps_s: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - psi_rad: unwrapped heading angle in XY plane (radians), shape (N,)
      - dpsi_ds_radpm: derivative dψ/ds (radians per meter), shape (N,)
      - valid_xy: whether heading was defined from XY tangent at each sample, shape (N,)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    s = np.asarray(s_m, dtype=float)
    if s.ndim != 1 or s.shape[0] != points.shape[0]:
        raise ValueError("s_m must have shape (N,) matching points")

    n = points.shape[0]
    psi = np.zeros((n,), dtype=float)
    dpsi_ds = np.zeros((n,), dtype=float)
    valid_xy = np.zeros((n,), dtype=bool)
    if n < 2:
        return psi, dpsi_ds, valid_xy

    t = np.zeros_like(points, dtype=float)
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    if n > 2:
        t[1:-1] = points[2:] - points[:-2]

    tx = t[:, 0]
    ty = t[:, 1]
    norm_xy = np.hypot(tx, ty)
    valid_xy = norm_xy > float(eps_xy)
    psi[valid_xy] = np.arctan2(ty[valid_xy], tx[valid_xy])

    # Fill undefined headings with nearest defined value (so unwrap/derivative stay stable).
    if np.any(valid_xy):
        first = int(np.argmax(valid_xy))
        psi[:first] = psi[first]
        last_val = float(psi[first])
        for i in range(first + 1, n):
            if valid_xy[i]:
                last_val = float(psi[i])
            else:
                psi[i] = last_val

    psi = np.unwrap(psi)

    if n == 2:
        denom = float(s[1] - s[0])
        dpsi = float(psi[1] - psi[0])
        dpsi_ds[:] = dpsi / denom if abs(denom) > eps_s else 0.0
        return psi, dpsi_ds, valid_xy

    for k in range(1, n - 1):
        denom = float(s[k + 1] - s[k - 1])
        if abs(denom) <= eps_s:
            dpsi_ds[k] = 0.0
        else:
            dpsi_ds[k] = float(psi[k + 1] - psi[k - 1]) / denom

    denom0 = float(s[1] - s[0])
    dpsi_ds[0] = float(psi[1] - psi[0]) / denom0 if abs(denom0) > eps_s else 0.0
    denom1 = float(s[-1] - s[-2])
    dpsi_ds[-1] = float(psi[-1] - psi[-2]) / denom1 if abs(denom1) > eps_s else 0.0

    return psi, dpsi_ds, valid_xy
