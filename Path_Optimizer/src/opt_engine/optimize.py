from __future__ import annotations

import numpy as np

from opt_engine.coords_unreal import scenario_waypoints_to_internal_m
from opt_engine.diagnostics import summarize_bindings
from opt_engine.geometry import arc_length, curvature_from_polyline, heading_and_dpsi_ds_xy
from opt_engine.speed_profile import apply_yaw_rate_speed_cap, solve_speed_profile, speed_caps_from_curvature
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

    psi_rad: np.ndarray | None = None
    dpsi_ds_radpm: np.ndarray | None = None
    v_yaw: np.ndarray | None = None
    if constraints.heading_constrained:
        psi_rad, dpsi_ds_radpm, _valid_xy = heading_and_dpsi_ds_xy(points=points_m, s_m=s_m)
        v_cap, v_yaw = apply_yaw_rate_speed_cap(v_cap_mps=v_cap, dpsi_ds_radpm=dpsi_ds_radpm, constraints=constraints)

    speed_profile = solve_speed_profile(
        ds_m=ds_m, v_cap_mps=v_cap, v_kappa_mps=v_kappa, v_yaw_mps=v_yaw, constraints=constraints
    )

    diagnostics = summarize_bindings(constraints=constraints, curvature_1pm=curvature_1pm, speed_profile=speed_profile)
    diagnostics["a_lat_max_effective_mps2"] = float(constraints.effective_a_lat_max())
    diagnostics["use_theta_max"] = bool(constraints.use_theta_max)
    diagnostics["heading_constrained"] = bool(constraints.heading_constrained)
    if constraints.heading_constrained:
        diagnostics["yaw_rate_max_rps"] = float(constraints.yaw_rate_max_rps)
        diagnostics["psi_rad"] = psi_rad.tolist() if psi_rad is not None else None
        diagnostics["dpsi_ds_radpm"] = dpsi_ds_radpm.tolist() if dpsi_ds_radpm is not None else None

    kappa_max_1pm = constraints.effective_kappa_max_1pm()
    if kappa_max_1pm is not None:
        violations = curvature_1pm > float(kappa_max_1pm)
        diagnostics["kappa_max_1pm"] = float(kappa_max_1pm)
        diagnostics["r_min_m"] = float(constraints.r_min_m) if constraints.r_min_m is not None else None
        diagnostics["kappa_violation_count"] = int(np.count_nonzero(violations))
        diagnostics["kappa_violation_max_1pm"] = float(np.max(curvature_1pm[violations])) if np.any(violations) else 0.0
        diagnostics["feasible"] = bool(not np.any(violations))
    else:
        diagnostics["kappa_max_1pm"] = None
        diagnostics["r_min_m"] = None
        diagnostics["kappa_violation_count"] = 0
        diagnostics["kappa_violation_max_1pm"] = 0.0
        diagnostics["feasible"] = True
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
    best_infeasible: OptimizationResult | None = None
    for lam in lambdas:
        res = evaluate_lambda(scenario=scenario, constraints=constraints, sampling=sampling, lambda_=float(lam))
        feasible = bool(res.diagnostics.get("feasible", True))
        if feasible:
            if best is None or res.time_s < best.time_s:
                best = res
        else:
            if best_infeasible is None:
                best_infeasible = res
            else:
                # Prefer "least violating": minimal max curvature, then minimal violation count.
                curv = float(res.diagnostics.get("max_curvature_1pm", 0.0))
                best_curv = float(best_infeasible.diagnostics.get("max_curvature_1pm", 0.0))
                if curv < best_curv:
                    best_infeasible = res
                elif curv == best_curv:
                    vcnt = int(res.diagnostics.get("kappa_violation_count", 0))
                    best_vcnt = int(best_infeasible.diagnostics.get("kappa_violation_count", 0))
                    if vcnt < best_vcnt:
                        best_infeasible = res

    if best is not None:
        return best
    if best_infeasible is not None:
        best_infeasible.diagnostics["status"] = "infeasible_all_candidates"
        return best_infeasible
    raise RuntimeError("No candidates evaluated")
