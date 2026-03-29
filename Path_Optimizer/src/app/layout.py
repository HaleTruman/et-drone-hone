from __future__ import annotations

from dash import dcc, html


_ROOT_STYLE: dict[str, object] = {
    "fontFamily": "Avenir Next, Avenir, Segoe UI, sans-serif",
    "padding": "24px",
    "color": "#0f172a",
    "background": "#f8fafc",
    "minHeight": "100vh",
    "boxSizing": "border-box",
}

_PANEL_STYLE: dict[str, object] = {
    "background": "#ffffff",
    "border": "1px solid #cbd5e1",
    "borderRadius": "12px",
    "padding": "16px",
}


def build_layout(*, scenario_options: list[dict[str, str]]) -> html.Div:
    default_scenario_value = scenario_options[0]["value"] if scenario_options else ""

    return html.Div(
        style=_ROOT_STYLE,
        children=[
            dcc.Interval(id="planning-session-poll", interval=1000, n_intervals=0),
            dcc.Store(id="scene-signature"),
            html.Div(
                style={"width": "min(100%, 1600px)", "margin": "0 auto"},
                children=[
                    html.H1("Reference Trajectory Viewer", style={"marginBottom": "6px"}),
                    html.P(
                        "Visualize the saved planner reference trajectory on top of the course model.",
                        style={"marginTop": "0", "marginBottom": "18px", "color": "#334155"},
                    ),
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "16px"},
                        children=[
                            html.Div(
                                style=_PANEL_STYLE,
                                children=[
                                    html.Label("Course JSON", style={"display": "block", "marginBottom": "8px"}),
                                    dcc.Dropdown(
                                        id="scenario_path",
                                        options=scenario_options,
                                        value=default_scenario_value,
                                        clearable=False,
                                    ),
                                    html.H3("Summary", style={"marginTop": "16px", "marginBottom": "10px"}),
                                    html.Pre(
                                        id="scene-summary",
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "fontFamily": "SFMono-Regular, Menlo, monospace",
                                            "fontSize": "13px",
                                            "lineHeight": "1.5",
                                            "margin": 0,
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                style=_PANEL_STYLE,
                                children=[
                                    dcc.Graph(
                                        id="course-3d",
                                        style={"height": "80vh"},
                                        config={"displaylogo": False},
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
