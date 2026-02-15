from __future__ import annotations

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
        Input("samples_per_segment", "value"),
    )
    def _recompute(
        scenario_path: str,
        lambda_mode: str,
        lambda_manual: float,
        lambda_min: float,
        lambda_max: float,
        lambda_steps: int,
        v_max: float,
        a_fwd_max: float,
        a_brake_max: float,
        a_lat_max: float,
        samples_per_segment: int,
    ):
        try:
            scenario = load_scenario(Path(scenario_path))
            constraints = Constraints(
                v_max=float(v_max),
                a_fwd_max=float(a_fwd_max),
                a_brake_max=float(a_brake_max),
                a_lat_max=float(a_lat_max),
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
            fig_3d = make_3d_figure(
                waypoints_m=waypoints_m, path_points_m=result.sampled_path.points_m, speed_mps=result.speed_profile.v_mps
            )
            fig_speed = make_profile_figure(s_m=result.sampled_path.s_m, v_mps=result.speed_profile.v_mps)
            fig_kappa = make_curvature_figure(s_m=result.sampled_path.s_m, kappa_1pm=result.curvature_1pm)

            d = result.diagnostics
            summary = (
                f"scenario: {result.scenario_name}\n"
                f"best_lambda: {result.best_lambda:.6g}\n"
                f"time_s: {result.time_s:.6g}\n"
                f"max_curvature_1pm: {d.get('max_curvature_1pm'):.6g}\n"
                f"speed_range_mps: [{d.get('min_speed_mps'):.6g}, {d.get('max_speed_mps'):.6g}]\n"
                f"bind_vmax_count: {d.get('bind_vmax_count')}\n"
                f"bind_curvature_count: {d.get('bind_curvature_count')}\n"
            )
            return fig_3d, fig_speed, fig_kappa, summary
        except Exception as e:  # noqa: BLE001
            import plotly.graph_objects as go

            empty = go.Figure()
            return empty, empty, empty, f"ERROR: {e}"
