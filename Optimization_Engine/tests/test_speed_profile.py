import numpy as np

from opt_engine.speed_profile import solve_speed_profile
from opt_engine.types import Constraints


def test_speed_profile_respects_caps():
    constraints = Constraints(v_max=5.0, a_fwd_max=2.0, a_brake_max=2.5, a_lat_max=4.0)
    ds = np.array([1.0, 1.0, 1.0], dtype=float)
    v_cap = np.array([5.0, 3.0, 5.0, 5.0], dtype=float)
    v_kappa = v_cap.copy()
    profile = solve_speed_profile(ds_m=ds, v_cap_mps=v_cap, v_kappa_mps=v_kappa, constraints=constraints)
    assert np.all(profile.v_mps <= v_cap + 1e-9)

