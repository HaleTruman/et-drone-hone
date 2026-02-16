from __future__ import annotations

from dash import dcc, html


_LABEL_ROW_STYLE: dict[str, object] = {
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "marginTop": "8px",
}

_INFO_ICON_STYLE: dict[str, object] = {
    "display": "inline-block",
    "cursor": "pointer",
    "border": "1px solid #999",
    "borderRadius": "999px",
    "width": "16px",
    "height": "16px",
    "lineHeight": "14px",
    "textAlign": "center",
    "fontSize": "12px",
    "color": "#555",
    "userSelect": "none",
}

_HELP_TEXT_STYLE_HIDDEN: dict[str, object] = {
    "display": "none",
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


def _label_with_info(*, label: str, info_id: str, help_id: str, help_text: str) -> list[html.Div]:
    icon = html.Span(
        "i",
        id=info_id,
        n_clicks=0,
        title=help_text,
        role="button",
        tabIndex=0,
        style=_INFO_ICON_STYLE,
    )

    return [
        html.Div([html.Span(label), icon], style=_LABEL_ROW_STYLE),
        html.Div(help_text, id=help_id, style=_HELP_TEXT_STYLE_HIDDEN),
    ]


def build_layout(*, scenario_options: list[dict[str, str]]) -> html.Div:
    default_scenario_value = scenario_options[0]["value"] if scenario_options else ""

    slider_common = {"updatemode": "mouseup"}

    return html.Div(
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "padding": "12px"},
        children=[
            html.H2("Optimization Engine (Waypoints + Kinematic Constraints)"),
            html.Div(
                style={"display": "flex", "gap": "16px"},
                children=[
                    html.Div(
                        style={"flex": "0 0 340px"},
                        children=[
                            *_label_with_info(
                                label="Scenario",
                                info_id="info_scenario",
                                help_id="help_scenario",
                                help_text="Scenario: Selects an ordered set of 3D waypoints to optimize through.",
                            ),
                            dcc.Dropdown(
                                id="scenario_path",
                                options=scenario_options,
                                value=default_scenario_value,
                                clearable=False,
                            ),
                            html.Hr(),
                            *_label_with_info(
                                label="Mode",
                                info_id="info_mode",
                                help_id="help_mode",
                                help_text="Mode: Choose whether to grid-search λ (auto) or set λ manually.",
                            ),
                            dcc.RadioItems(
                                id="lambda_mode",
                                options=[
                                    {"label": "Optimize λ (grid search)", "value": "auto"},
                                    {"label": "Manual λ", "value": "manual"},
                                ],
                                value="auto",
                            ),
                            *_label_with_info(
                                label="λ (manual)",
                                info_id="info_lambda_manual",
                                help_id="help_lambda_manual",
                                help_text="λ (manual): Sets spline smoothness directly. Lower λ is more angular; higher λ is smoother.",
                            ),
                            dcc.Slider(id="lambda_manual", min=0.05, max=3.0, step=0.05, value=1.0, **slider_common),
                            *_label_with_info(
                                label="λ min (auto)",
                                info_id="info_lambda_min",
                                help_id="help_lambda_min",
                                help_text="λ min (auto): Lower bound of λ values considered in grid search.",
                            ),
                            dcc.Slider(id="lambda_min", min=0.05, max=3.0, step=0.05, value=0.2, **slider_common),
                            *_label_with_info(
                                label="λ max (auto)",
                                info_id="info_lambda_max",
                                help_id="help_lambda_max",
                                help_text="λ max (auto): Upper bound of λ values considered in grid search.",
                            ),
                            dcc.Slider(id="lambda_max", min=0.05, max=3.0, step=0.05, value=2.0, **slider_common),
                            *_label_with_info(
                                label="λ steps (auto)",
                                info_id="info_lambda_steps",
                                help_id="help_lambda_steps",
                                help_text="λ steps (auto): Number of λ samples between min and max (higher = slower, potentially better).",
                            ),
                            dcc.Slider(id="lambda_steps", min=1, max=60, step=1, value=25, **slider_common),
                            html.Hr(),
                            *_label_with_info(
                                label="v_max (m/s)",
                                info_id="info_v_max",
                                help_id="help_v_max",
                                help_text="v_max: Global maximum speed cap (m/s). The speed profile never exceeds this value.",
                            ),
                            dcc.Slider(id="v_max", min=0.1, max=30.0, step=0.1, value=6.0, **slider_common),
                            *_label_with_info(
                                label="a_fwd_max (m/s²)",
                                info_id="info_a_fwd_max",
                                help_id="help_a_fwd_max",
                                help_text="a_fwd_max: Maximum forward acceleration (m/s²). Limits how quickly speed can increase along the path.",
                            ),
                            dcc.Slider(id="a_fwd_max", min=0.1, max=20.0, step=0.1, value=2.0, **slider_common),
                            *_label_with_info(
                                label="a_brake_max (m/s²)",
                                info_id="info_a_brake_max",
                                help_id="help_a_brake_max",
                                help_text="a_brake_max: Maximum braking deceleration magnitude (m/s²). Limits how quickly speed can decrease approaching turns.",
                            ),
                            dcc.Slider(id="a_brake_max", min=0.1, max=30.0, step=0.1, value=3.0, **slider_common),
                            *_label_with_info(
                                label="a_lat_max (m/s²)",
                                info_id="info_a_lat_max",
                                help_id="help_a_lat_max",
                                help_text="a_lat_max: Maximum lateral acceleration (m/s²). Sets the curvature-based speed limit v ≤ √(a_lat_max/(|κ|+ε)).",
                            ),
                            dcc.Slider(id="a_lat_max", min=0.1, max=30.0, step=0.1, value=4.0, **slider_common),
                            html.Hr(),
                            html.H4("Optional constraints", style={"margin": "8px 0 0 0"}),
                            *_label_with_info(
                                label="Hard turn limit",
                                info_id="info_use_r_min",
                                help_id="help_use_r_min",
                                help_text="Hard turn limit: If enabled, the path is infeasible if curvature κ(s) exceeds κ_max anywhere (equivalently turning radius r(s)=1/κ(s) drops below r_min).",
                            ),
                            dcc.Checklist(
                                id="use_r_min",
                                options=[{"label": "Enabled", "value": "on"}],
                                value=[],
                            ),
                            *_label_with_info(
                                label="r_min (m)",
                                info_id="info_r_min_m",
                                help_id="help_r_min_m",
                                help_text="r_min: Minimum turning radius (meters). Enables κ_max = 1/r_min feasibility check when Hard turn limit is enabled.",
                            ),
                            dcc.Slider(id="r_min_m", min=0.5, max=200.0, step=0.5, value=10.0, **slider_common),
                            *_label_with_info(
                                label="Tilt coupling (θ_max → a_lat_max)",
                                info_id="info_use_theta_max",
                                help_id="help_use_theta_max",
                                help_text="Tilt coupling: If enabled, derive a_lat_max from θ_max via a_lat_max = g*tan(θ_max) instead of using the a_lat_max slider.",
                            ),
                            dcc.Checklist(
                                id="use_theta_max",
                                options=[{"label": "Enabled", "value": "on"}],
                                value=[],
                            ),
                            *_label_with_info(
                                label="θ_max (deg)",
                                info_id="info_theta_max_deg",
                                help_id="help_theta_max_deg",
                                help_text="θ_max: Maximum tilt angle (degrees). When enabled, sets lateral acceleration limit to a_lat_max = 9.81*tan(θ_max).",
                            ),
                            dcc.Slider(id="theta_max_deg", min=1.0, max=89.0, step=0.5, value=35.0, **slider_common),
                            *_label_with_info(
                                label="Heading constrained (yaw-rate cap)",
                                info_id="info_heading_constrained",
                                help_id="help_heading_constrained",
                                help_text="Heading constrained: If enabled, add a yaw-rate cap assuming yaw follows direction of travel in XY: v(s) ≤ ψ̇_max / |dψ/ds|.",
                            ),
                            dcc.Checklist(
                                id="heading_constrained",
                                options=[{"label": "Enabled", "value": "on"}],
                                value=[],
                            ),
                            *_label_with_info(
                                label="ψ̇_max (deg/s)",
                                info_id="info_yaw_rate_max_dps",
                                help_id="help_yaw_rate_max_dps",
                                help_text="ψ̇_max: Maximum yaw rate (degrees per second). When Heading constrained is enabled, this becomes a speed cap via v ≤ ψ̇_max/|dψ/ds|.",
                            ),
                            dcc.Slider(
                                id="yaw_rate_max_dps", min=1.0, max=720.0, step=1.0, value=120.0, **slider_common
                            ),
                            html.Hr(),
                            *_label_with_info(
                                label="samples_per_segment",
                                info_id="info_samples_per_segment",
                                help_id="help_samples_per_segment",
                                help_text="samples_per_segment: Number of points sampled per waypoint-to-waypoint segment. Higher = smoother plots/curvature, slower recompute.",
                            ),
                            dcc.Slider(id="samples_per_segment", min=10, max=250, step=1, value=50, **slider_common),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 auto"},
                        children=[
                            html.Div(id="summary", style={"whiteSpace": "pre-wrap", "marginBottom": "8px"}),
                            dcc.Graph(id="fig_3d", style={"height": "520px"}),
                            html.Div(
                                style={"display": "flex", "gap": "12px"},
                                children=[
                                    dcc.Graph(id="fig_speed", style={"flex": "1 1 50%", "height": "280px"}),
                                    dcc.Graph(id="fig_kappa", style={"flex": "1 1 50%", "height": "280px"}),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
