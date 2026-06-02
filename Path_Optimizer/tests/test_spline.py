import numpy as np

from autonomy.opt_engine.spline import sample_hermite_spline
from autonomy.opt_engine.types import SamplingConfig


def test_spline_interpolates_waypoints_exactly():
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    sampling = SamplingConfig(samples_per_segment=10)
    points, waypoint_indices = sample_hermite_spline(waypoints, lambda_=1.0, sampling=sampling)

    assert len(waypoint_indices) == waypoints.shape[0]
    for i, idx in enumerate(waypoint_indices):
        assert np.allclose(points[idx], waypoints[i])

