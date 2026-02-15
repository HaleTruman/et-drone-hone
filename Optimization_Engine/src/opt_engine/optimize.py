from __future__ import annotations

import numpy as np

from opt_engine.coords_unreal import scenario_waypoints_to_internal_m
from opt_engine.diagnostics import summarize_bindings
from opt_engine.geometry import arc_length, curvature_from_polyline
from opt_engine.speed_profile import solve_speed_profile, speed_caps_from_curvature
from opt_engine.spline import sample_hermite_spline
from opt_engine.types import Constraints, OptimizationResult, SamplingConfig, SampledPath, Scenario


def evaluate_lambda(
    *,
    scenario: Scenario,
    constraints: Constraints,
    sampling: SamplingConfig,
    lambda_: float,
) -> OptimizationResult:
    waypoints_m = scenario_waypoints_to_internal_m(scenario)
    points_m, waypoint_indices = sample_hermite_spline(waypoints_m, lambda_=lambda_, sampling=sampling)
    s_m, ds_m = arc_length(points_m)
    sampled_path = SampledPath(points_m=points_m, s_m=s_m, ds_m=ds_m, waypoint_indices=waypoint_indices)

    curvature_1pm = curvature_from_polyline(points_m)
    v_cap, v_kappa = speed_caps_from_curvature(curvature_1pm, constraints)
    speed_profile = solve_speed_profile(ds_m=ds_m, v_cap_mps=v_cap, v_kappa_mps=v_kappa, constraints=constraints)

    diagnostics = summarize_bindings(constraints=constraints, curvature_1pm=curvature_1pm, speed_profile=speed_profile)
    return OptimizationResult(
        scenario_name=scenario.name,
        best_lambda=float(lambda_),
        time_s=float(speed_profile.time_s),
        sampled_path=sampled_path,
        curvature_1pm=curvature_1pm,
        speed_profile=speed_profile,
        diagnostics=diagnostics,
    )


def optimize_lambda_grid(
    *,
    scenario: Scenario,
    constraints: Constraints,
    sampling: SamplingConfig,
    lambda_min: float,
    lambda_max: float,
    lambda_steps: int,
) -> OptimizationResult:
    if lambda_steps < 1:
        raise ValueError("lambda_steps must be >= 1")

    if lambda_steps == 1:
        return evaluate_lambda(scenario=scenario, constraints=constraints, sampling=sampling, lambda_=lambda_min)

    lambdas = np.linspace(float(lambda_min), float(lambda_max), num=int(lambda_steps), dtype=float)
    best: OptimizationResult | None = None
    for lam in lambdas:
        res = evaluate_lambda(scenario=scenario, constraints=constraints, sampling=sampling, lambda_=float(lam))
        if best is None or res.time_s < best.time_s:
            best = res
    assert best is not None
    return best

