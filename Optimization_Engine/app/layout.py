from __future__ import annotations

from dash import dcc, html


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
                            html.Label("Scenario"),
                            dcc.Dropdown(
                                id="scenario_path",
                                options=scenario_options,
                                value=default_scenario_value,
                                clearable=False,
                            ),
                            html.Hr(),
                            html.Label("Mode"),
                            dcc.RadioItems(
                                id="lambda_mode",
                                options=[
                                    {"label": "Optimize λ (grid search)", "value": "auto"},
                                    {"label": "Manual λ", "value": "manual"},
                                ],
                                value="auto",
                            ),
                            html.Label("λ (manual)"),
                            dcc.Slider(id="lambda_manual", min=0.05, max=3.0, step=0.05, value=1.0, **slider_common),
                            html.Label("λ min / max / steps (auto)"),
                            dcc.Slider(id="lambda_min", min=0.05, max=3.0, step=0.05, value=0.2, **slider_common),
                            dcc.Slider(id="lambda_max", min=0.05, max=3.0, step=0.05, value=2.0, **slider_common),
                            dcc.Slider(id="lambda_steps", min=1, max=60, step=1, value=25, **slider_common),
                            html.Hr(),
                            html.Label("v_max (m/s)"),
                            dcc.Slider(id="v_max", min=0.1, max=30.0, step=0.1, value=6.0, **slider_common),
                            html.Label("a_fwd_max (m/s²)"),
                            dcc.Slider(id="a_fwd_max", min=0.1, max=20.0, step=0.1, value=2.0, **slider_common),
                            html.Label("a_brake_max (m/s²)"),
                            dcc.Slider(id="a_brake_max", min=0.1, max=30.0, step=0.1, value=3.0, **slider_common),
                            html.Label("a_lat_max (m/s²)"),
                            dcc.Slider(id="a_lat_max", min=0.1, max=30.0, step=0.1, value=4.0, **slider_common),
                            html.Hr(),
                            html.Label("samples_per_segment"),
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

