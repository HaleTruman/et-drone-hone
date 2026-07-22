from dash import dash_table, dcc, html


COLORS = {
    "ink": "#172033",
    "muted": "#60708a",
    "line": "#dbe3ef",
    "panel": "#ffffff",
    "canvas": "#f3f6fb",
    "accent": "#2563eb",
}

PANEL_STYLE = {
    "background": COLORS["panel"],
    "border": f"1px solid {COLORS['line']}",
    "borderRadius": "14px",
    "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.05)",
    "padding": "18px",
}


def build_layout(*, live_run_options: list[dict[str, str]]) -> html.Div:
    return html.Div(
        style={
            "background": COLORS["canvas"],
            "color": COLORS["ink"],
            "fontFamily": "Inter, Segoe UI, sans-serif",
            "minHeight": "100vh",
            "padding": "26px",
            "boxSizing": "border-box",
        },
        children=[
            dcc.Interval(id="run-poll", interval=2_000, n_intervals=0),
            html.Div(
                style={"margin": "0 auto", "maxWidth": "1720px"},
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "20px", "justifyContent": "space-between", "alignItems": "end"},
                        children=[
                            html.Div(
                                children=[
                                    html.H1("Live Run Explorer", style={"fontSize": "30px", "margin": "0 0 4px"}),
                                    html.P(
                                        "Inspect captured MAVLink telemetry and FPV frame runs.",
                                        style={"color": COLORS["muted"], "margin": 0},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    _live_runs_page(live_run_options),
                ],
            ),
        ],
    )


def _live_runs_page(live_run_options: list[dict[str, str]]) -> html.Div:
    selected_run = live_run_options[0]["value"] if live_run_options else None
    return html.Div(
        style={"marginTop": "16px"},
        children=[
            html.Div(
                style={**PANEL_STYLE, "display": "grid", "gap": "14px", "gridTemplateColumns": "minmax(320px, 1fr) auto", "alignItems": "end"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Captured run", style={"display": "block", "fontSize": "13px", "fontWeight": 700, "marginBottom": "6px"}),
                            dcc.Dropdown(
                                id="live-run-path",
                                options=live_run_options,
                                value=selected_run,
                                clearable=False,
                                placeholder="No captured runs found in logs/runs",
                            ),
                        ]
                    ),
                    html.Div("Source: run.json, telemetry.json, gate_map.json, frames.jsonl, and vision_observations.jsonl in logs/runs/", style={"color": COLORS["muted"], "fontSize": "13px", "paddingBottom": "9px"}),
                ],
            ),
            html.Div(id="live-load-error", style={"color": "#b91c1c", "marginTop": "16px"}),
            html.Div(
                id="live-summary-cards",
                style={"display": "grid", "gap": "12px", "gridTemplateColumns": "repeat(auto-fit, minmax(145px, 1fr))", "margin": "16px 0"},
            ),
            html.Div(
                style={"display": "grid", "gap": "16px", "gridTemplateColumns": "minmax(560px, 1.6fr) minmax(360px, 1fr)"},
                children=[
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.H3("FPV Frame Viewer", style={"margin": "0 0 12px"}),
                            html.Img(id="live-frame-image", style={"display": "block", "width": "100%", "minHeight": "300px", "objectFit": "contain", "background": "#111827", "borderRadius": "8px"}),
                            html.Div(
                                style={"display": "grid", "gap": "10px", "gridTemplateColumns": "auto 1fr auto", "alignItems": "center", "marginTop": "14px"},
                                children=[
                                    html.Button("Previous", id="live-frame-previous", n_clicks=0),
                                    dcc.Slider(id="live-frame-index", min=0, max=0, step=1, value=0, marks={0: "0"}, tooltip={"placement": "bottom", "always_visible": False}),
                                    html.Button("Next", id="live-frame-next", n_clicks=0),
                                ],
                            ),
                            html.Div(id="live-frame-caption", style={"color": COLORS["muted"], "fontFamily": "SFMono-Regular, Consolas, monospace", "fontSize": "12px", "marginTop": "14px"}),
                        ],
                    ),
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.H3("Frame-Aligned Telemetry", style={"margin": "0 0 12px"}),
                            html.Pre(id="live-frame-telemetry", style=_pre_style("520px")),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={**PANEL_STYLE, "marginTop": "16px"},
                children=[
                    html.H3("Frame-Aligned Vision Observation", style={"margin": "0 0 12px"}),
                    html.Pre(id="live-frame-observation", style=_pre_style("420px")),
                ],
            ),
            html.Div(
                style={**PANEL_STYLE, "marginTop": "16px"},
                children=[
                    html.H3("Vision Observation Gates At Frame", style={"margin": "0 0 12px"}),
                    dcc.Graph(id="live-gate-map-3d", style={"height": "620px"}, config={"displaylogo": False}),
                ],
            ),
            html.Div(
                style={**PANEL_STYLE, "marginTop": "16px"},
                children=[dcc.Graph(id="live-trajectory-3d", style={"height": "620px"}, config={"displaylogo": False})],
            ),
            _graph_grid(
                [
                    ("Position", "live-position-plot", "360px"),
                    ("Velocity", "live-velocity-plot", "360px"),
                    ("Acceleration", "live-acceleration-plot", "360px"),
                    ("Body Rates", "live-rates-plot", "360px"),
                    ("System Mode", "live-system-mode-plot", "360px"),
                ]
            ),
            dcc.Tabs(
                value="representative-gates",
                style={"marginTop": "16px"},
                children=[
                    dcc.Tab(
                        label="Representative Gate Map",
                        value="representative-gates",
                        children=[
                            html.Div(
                                style={**PANEL_STYLE, "marginTop": "16px"},
                                children=[
                                    html.H3("Representative Gate Map", style={"margin": "0 0 12px"}),
                                    html.Div(id="representative-gate-summary", style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "12px"}),
                                    dcc.Graph(id="representative-gate-map-3d", style={"height": "720px"}, config={"displaylogo": False}),
                                ],
                            ),
                            html.Div(
                                style={**PANEL_STYLE, "marginTop": "16px"},
                                children=[
                                    html.H3("Representative Gates", style={"margin": "0 0 12px"}),
                                    dash_table.DataTable(
                                        id="representative-gate-table",
                                        page_action="none",
                                        sort_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_header={"backgroundColor": "#eaf0f8", "fontWeight": "700", "whiteSpace": "normal"},
                                        style_cell={
                                            "border": f"1px solid {COLORS['line']}",
                                            "fontFamily": "SFMono-Regular, Consolas, monospace",
                                            "fontSize": "12px",
                                            "padding": "7px",
                                            "textAlign": "left",
                                            "whiteSpace": "nowrap",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(label="Lifecycle Events", value="live-events", children=[_table_panel("Lifecycle Events", "live-events-table")]),
                    dcc.Tab(label="Telemetry Cycles", value="live-cycles", children=[_table_panel("Telemetry Cycles", "live-cycles-table")]),
                    dcc.Tab(
                        label="Raw JSON",
                        value="live-raw",
                        children=[html.Div(style={**PANEL_STYLE, "marginTop": "16px"}, children=[html.Pre(id="live-raw-json", style=_pre_style("78vh"))])],
                    ),
                ],
            ),
        ],
    )


def _pre_style(max_height: str) -> dict[str, str | float | int]:
    return {
        "fontFamily": "SFMono-Regular, Consolas, monospace",
        "fontSize": "12px",
        "lineHeight": 1.45,
        "margin": 0,
        "maxHeight": max_height,
        "overflow": "auto",
        "whiteSpace": "pre-wrap",
        "wordBreak": "break-word",
    }


def _graph_grid(graphs: list[tuple[str, str, str]]) -> html.Div:
    return html.Div(
        style={"display": "grid", "gap": "16px", "gridTemplateColumns": "repeat(auto-fit, minmax(540px, 1fr))", "marginTop": "16px"},
        children=[
            html.Div(
                style=PANEL_STYLE,
                children=[dcc.Graph(id=graph_id, style={"height": height}, config={"displaylogo": False})],
            )
            for _title, graph_id, height in graphs
        ],
    )


def _table_panel(title: str, table_id: str) -> html.Div:
    return html.Div(
        style={**PANEL_STYLE, "marginTop": "16px"},
        children=[
            html.H3(title, style={"margin": "0 0 12px"}),
            dash_table.DataTable(
                id=table_id,
                page_action="none",
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto", "overflowY": "visible"},
                style_header={"backgroundColor": "#eaf0f8", "fontWeight": "700", "whiteSpace": "normal"},
                style_cell={
                    "border": f"1px solid {COLORS['line']}",
                    "fontFamily": "SFMono-Regular, Consolas, monospace",
                    "fontSize": "12px",
                    "maxWidth": "280px",
                    "minWidth": "100px",
                    "overflow": "hidden",
                    "padding": "7px",
                    "textAlign": "left",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                },
                css=[
                    {
                        "selector": ".dash-table-tooltip",
                        "rule": "max-width: 720px; white-space: pre-wrap; word-break: break-word;",
                    }
                ],
                tooltip_delay=0,
                tooltip_duration=None,
            ),
        ],
    )
