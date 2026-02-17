from __future__ import annotations

import math
from pathlib import Path

from dash import Dash, Input, Output, callback

from opt_engine.coords_unreal import scenario_waypoints_to_internal_m
from opt_engine.optimize import evaluate_lambda, optimize_lambda_grid
from opt_engine.plotting import make_3d_figure, make_curvature_figure, make_profile_figure
from opt_engine.scenario_io import load_scenario
from opt_engine.types import Constraints, SamplingConfig


def register_callbacks(app: Dash) -> None:
    help_style_base: dict[str, object] = {
        "marginTop": "6px",
        "marginBottom": "8px",
        "padding": "8px",
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "background": "#fafafa",
        "fontSize": "12px",
        "color": "#333",
        "whiteSpace": "pre-wrap",
    }

    def _help_style(n_clicks: int | None) -> dict[str, object]:
        is_open = (int(n_clicks or 0) % 2) == 1
        return {**help_style_base, "display": "block" if is_open else "none"}

    @callback(
        Output("help_scenario", "style"),
        Output("help_mode", "style"),
        Output("help_lambda_manual", "style"),
        Output("help_lambda_min", "style"),
        Output("help_lambda_max", "style"),
        Output("help_lambda_steps", "style"),
        Output("help_v_max", "style"),
        Output("help_a_fwd_max", "style"),
        Output("help_a_brake_max", "style"),
        Output("help_a_lat_max", "style"),
        Output("help_use_r_min", "style"),
        Output("help_r_min_m", "style"),
        Output("help_use_theta_max", "style"),
        Output("help_theta_max_deg", "style"),
        Output("help_heading_constrained", "style"),
        Output("help_yaw_rate_max_dps", "style"),
        Output("help_samples_per_segment", "style"),
        Input("info_scenario", "n_clicks"),
        Input("info_mode", "n_clicks"),
        Input("info_lambda_manual", "n_clicks"),
        Input("info_lambda_min", "n_clicks"),
        Input("info_lambda_max", "n_clicks"),
        Input("info_lambda_steps", "n_clicks"),
        Input("info_v_max", "n_clicks"),
        Input("info_a_fwd_max", "n_clicks"),
        Input("info_a_brake_max", "n_clicks"),
        Input("info_a_lat_max", "n_clicks"),
        Input("info_use_r_min", "n_clicks"),
        Input("info_r_min_m", "n_clicks"),
        Input("info_use_theta_max", "n_clicks"),
        Input("info_theta_max_deg", "n_clicks"),
        Input("info_heading_constrained", "n_clicks"),
        Input("info_yaw_rate_max_dps", "n_clicks"),
        Input("info_samples_per_segment", "n_clicks"),
    )
    def _toggle_help(
        info_scenario: int | None,
        info_mode: int | None,
        info_lambda_manual: int | None,
        info_lambda_min: int | None,
        info_lambda_max: int | None,
        info_lambda_steps: int | None,
        info_v_max: int | None,
        info_a_fwd_max: int | None,
        info_a_brake_max: int | None,
        info_a_lat_max: int | None,
        info_use_r_min: int | None,
        info_r_min_m: int | None,
        info_use_theta_max: int | None,
        info_theta_max_deg: int | None,
        info_heading_constrained: int | None,
        info_yaw_rate_max_dps: int | None,
        info_samples_per_segment: int | None,
    ):
        return (
            _help_style(info_scenario),
            _help_style(info_mode),
            _help_style(info_lambda_manual),
            _help_style(info_lambda_min),
            _help_style(info_lambda_max),
            _help_style(info_lambda_steps),
            _help_style(info_v_max),
            _help_style(info_a_fwd_max),
            _help_style(info_a_brake_max),
            _help_style(info_a_lat_max),
            _help_style(info_use_r_min),
            _help_style(info_r_min_m),
            _help_style(info_use_theta_max),
            _help_style(info_theta_max_deg),
            _help_style(info_heading_constrained),
            _help_style(info_yaw_rate_max_dps),
            _help_style(info_samples_per_segment),
        )

    @callback(
        Output("fig_3d", "figure"),
        Output("fig_speed", "figure"),
        Output("fig_kappa", "figure"),
        Output("summary", "children"),
        Input("scenario_path", "value"),
        Input("lambda_mode", "value"),
        Input("lambda_manual", "value"),
        Input("lambda_min", "value"),
        Input("lambda_max", "value"),
        Input("lambda_steps", "value"),
        Input("v_max", "value"),
        Input("a_fwd_max", "value"),
        Input("a_brake_max", "value"),
        Input("a_lat_max", "value"),
        Input("use_r_min", "value"),
        Input("r_min_m", "value"),
        Input("use_theta_max", "value"),
        Input("theta_max_deg", "value"),
        Input("heading_constrained", "value"),
        Input("yaw_rate_max_dps", "value"),
        Input("samples_per_segment", "value"),
    )
    def _recompute(
        scenario_path: str | None,
        lambda_mode: str,
        lambda_manual: float,
        lambda_min: float,
        lambda_max: float,
        lambda_steps: int,
        v_max: float,
        a_fwd_max: float,
        a_brake_max: float,
        a_lat_max: float,
        use_r_min: list[str],
        r_min_m: float,
        use_theta_max: list[str],
        theta_max_deg: float,
        heading_constrained: list[str],
        yaw_rate_max_dps: float,
        samples_per_segment: int,
    ):
        import plotly.graph_objects as go

        empty = go.Figure()
        if not scenario_path:
            return empty, empty, empty, "No course selected."

        scenario_file = Path(scenario_path)
        if not scenario_file.is_file():
            return empty, empty, empty, f"Course file not found: {scenario_path}"

        try:
            scenario = load_scenario(scenario_file)
            use_r_min_enabled = "on" in (use_r_min or [])
            use_theta_enabled = "on" in (use_theta_max or [])
            heading_enabled = "on" in (heading_constrained or [])
            constraints = Constraints(
                v_max=float(v_max),
                a_fwd_max=float(a_fwd_max),
                a_brake_max=float(a_brake_max),
                a_lat_max=float(a_lat_max),
                r_min_m=float(r_min_m) if use_r_min_enabled else None,
                use_theta_max=bool(use_theta_enabled),
                theta_max_deg=float(theta_max_deg),
                heading_constrained=bool(heading_enabled),
                yaw_rate_max_rps=float(yaw_rate_max_dps) * math.pi / 180.0,
            )
            sampling = SamplingConfig(samples_per_segment=int(samples_per_segment))

            if lambda_mode == "manual":
                result = evaluate_lambda(
                    scenario=scenario, constraints=constraints, sampling=sampling, lambda_=float(lambda_manual)
                )
            else:
                result = optimize_lambda_grid(
                    scenario=scenario,
                    constraints=constraints,
                    sampling=sampling,
                    lambda_min=float(lambda_min),
                    lambda_max=float(lambda_max),
                    lambda_steps=int(lambda_steps),
                )

            waypoints_m = scenario_waypoints_to_internal_m(scenario)
            kappa_max_1pm = result.diagnostics.get("kappa_max_1pm")
            kappa_violation_mask = None
            if kappa_max_1pm is not None:
                try:
                    km = float(kappa_max_1pm)
                    kappa_violation_mask = result.curvature_1pm > km
                except Exception:  # noqa: BLE001
                    kappa_violation_mask = None
            fig_3d = make_3d_figure(
                waypoints_m=waypoints_m,
                path_points_m=result.sampled_path.points_m,
                speed_mps=result.speed_profile.v_mps,
                kappa_violation_mask=kappa_violation_mask,
            )
            fig_speed = make_profile_figure(
                s_m=result.sampled_path.s_m,
                v_mps=result.speed_profile.v_mps,
                v_cap_mps=result.speed_profile.v_cap_mps,
                v_kappa_mps=result.speed_profile.v_kappa_mps,
                v_yaw_mps=result.speed_profile.v_yaw_mps,
            )
            fig_kappa = make_curvature_figure(
                s_m=result.sampled_path.s_m, kappa_1pm=result.curvature_1pm, kappa_max_1pm=kappa_max_1pm
            )

            d = result.diagnostics
            feasible = bool(d.get("feasible", True))
            status = "OK" if feasible else f"INFEASIBLE ({d.get('status') or 'kappa_max'})"
            a_lat_eff = float(d.get("a_lat_max_effective_mps2", 0.0))
            summary = (
                f"scenario: {result.scenario_name}\n"
                f"best_lambda: {result.best_lambda:.6g}\n"
                f"status: {status}\n"
                f"time_s: {result.time_s:.6g}\n"
                f"max_curvature_1pm: {d.get('max_curvature_1pm'):.6g}\n"
                f"speed_range_mps: [{d.get('min_speed_mps'):.6g}, {d.get('max_speed_mps'):.6g}]\n"
                f"a_lat_max_effective_mps2: {a_lat_eff:.6g}\n"
                f"bind_vmax_count: {d.get('bind_vmax_count')}\n"
                f"bind_curvature_count: {d.get('bind_curvature_count')}\n"
                f"bind_yaw_count: {d.get('bind_yaw_count')}\n"
            )
            if d.get("kappa_max_1pm") is not None:
                summary += (
                    f"kappa_max_1pm: {d.get('kappa_max_1pm'):.6g}\n"
                    f"r_min_m: {d.get('r_min_m')}\n"
                    f"kappa_violation_count: {d.get('kappa_violation_count')}\n"
                )
            if d.get("heading_constrained"):
                summary += f"yaw_rate_max_rps: {d.get('yaw_rate_max_rps'):.6g}\n"
            return fig_3d, fig_speed, fig_kappa, summary
        except Exception as e:  # noqa: BLE001
            return empty, empty, empty, f"ERROR: {e}"
