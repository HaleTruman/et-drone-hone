import numpy as np
import pytest

from opt_engine.optimize import evaluate_lambda
from opt_engine.speed_profile import apply_yaw_rate_speed_cap
from opt_engine.types import Constraints, SamplingConfig, Scenario, Waypoint


def test_theta_max_coupling_derives_a_lat_max():
    c = Constraints(use_theta_max=True, theta_max_deg=45.0, g_mps2=9.81)
    assert c.effective_a_lat_max() == pytest.approx(9.81, abs=1e-6)


def test_hard_turn_limit_marks_infeasible():
    scenario = Scenario(
        name="tight_turn",
        frame="internal",
        units="m",
        waypoints=(
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(1.0, 1.0, 0.0),
        ),
    )
    constraints = Constraints(v_max=10.0, a_fwd_max=5.0, a_brake_max=5.0, a_lat_max=10.0, r_min_m=1e6)
    sampling = SamplingConfig(samples_per_segment=30)
    res = evaluate_lambda(scenario=scenario, constraints=constraints, sampling=sampling, lambda_=0.5)
    assert res.diagnostics["feasible"] is False
    assert int(res.diagnostics["kappa_violation_count"]) > 0


def test_yaw_rate_cap_applies_as_speed_cap():
    constraints = Constraints(heading_constrained=True, yaw_rate_max_rps=1.0, yaw_epsilon_radpm=1e-9)
    v_cap = np.full((10,), 10.0, dtype=float)
    dpsi_ds = np.full((10,), 1.0, dtype=float)  # rad/m
    v_cap2, v_yaw = apply_yaw_rate_speed_cap(v_cap_mps=v_cap, dpsi_ds_radpm=dpsi_ds, constraints=constraints)
    assert np.allclose(v_yaw, 1.0, atol=1e-6)
    assert np.allclose(v_cap2, 1.0, atol=1e-6)
