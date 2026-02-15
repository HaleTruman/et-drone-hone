import numpy as np
import pytest

from opt_engine.geometry import arc_length, curvature_from_polyline, heading_and_dpsi_ds_xy


def test_curvature_straight_line_is_near_zero():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    kappa = curvature_from_polyline(points)
    assert np.all(kappa < 1e-6)


def test_heading_derivative_on_unit_circle_is_near_one():
    t = np.linspace(0.0, np.pi / 2.0, num=80, dtype=float)
    points = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    s, _ds = arc_length(points)
    _psi, dpsi_ds, valid = heading_and_dpsi_ds_xy(points=points, s_m=s)

    assert np.all(valid)
    mid = dpsi_ds[10:-10]
    assert float(np.median(mid)) == pytest.approx(1.0, abs=0.15)
