from __future__ import annotations

import numpy as np

from autonomy.opt_engine.types import SamplingConfig


def sample_hermite_spline(
    waypoints_m: np.ndarray, *, lambda_: float, sampling: SamplingConfig
) -> tuple[np.ndarray, tuple[int, ...]]:
    if waypoints_m.ndim != 2 or waypoints_m.shape[1] != 3:
        raise ValueError("waypoints_m must have shape (M, 3)")
    if waypoints_m.shape[0] < 2:
        raise ValueError("Need at least 2 waypoints")

    samples_per_segment = int(sampling.samples_per_segment)
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be >= 2")

    tangents = _compute_base_tangents(waypoints_m) * float(lambda_)
    u = np.linspace(0.0, 1.0, num=samples_per_segment, endpoint=False, dtype=float)

    segments: list[np.ndarray] = []
    for i in range(waypoints_m.shape[0] - 1):
        p0 = waypoints_m[i]
        p1 = waypoints_m[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]
        segments.append(_hermite_segment(p0=p0, p1=p1, m0=m0, m1=m1, u=u))

    points = np.concatenate([*segments, waypoints_m[-1][None, :]], axis=0)
    waypoint_indices = tuple(i * samples_per_segment for i in range(waypoints_m.shape[0]))
    return points, waypoint_indices


def _compute_base_tangents(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    tangents = np.zeros_like(pts, dtype=float)
    if n < 2:
        return tangents

    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if n > 2:
        tangents[1:-1] = 0.5 * (pts[2:] - pts[:-2])
    return tangents


def _hermite_segment(
    *, p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, u: np.ndarray
) -> np.ndarray:
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    return (
        h00[:, None] * p0[None, :]
        + h10[:, None] * m0[None, :]
        + h01[:, None] * p1[None, :]
        + h11[:, None] * m1[None, :]
    )

