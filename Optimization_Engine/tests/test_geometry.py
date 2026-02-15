import numpy as np

from opt_engine.geometry import curvature_from_polyline


def test_curvature_straight_line_is_near_zero():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    kappa = curvature_from_polyline(points)
    assert np.all(kappa < 1e-6)

