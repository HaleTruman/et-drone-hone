from __future__ import annotations

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


def build_layout(*, run_options: list[dict[str, str]]) -> html.Div:
    selected_run = run_options[0]["value"] if run_options else None
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
                                    html.H1("Racing Stack Run Explorer", style={"fontSize": "30px", "margin": "0 0 4px"}),
                                    html.P(
                                        "Inspect saved simulation output from logs/.",
                                        style={"color": COLORS["muted"], "margin": 0},
                                    ),
                                ]
                            ),
                            html.Div(
                                style={"minWidth": "410px"},
                                children=[
                                    html.Label("Run", style={"display": "block", "fontSize": "13px", "fontWeight": 700, "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="run-path",
                                        options=run_options,
                                        value=selected_run,
                                        clearable=False,
                                        placeholder="No run logs found",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(id="load-error", style={"color": "#b91c1c", "marginTop": "16px"}),
                    html.Div(
                        id="summary-cards",
                        style={"display": "grid", "gap": "12px", "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))", "margin": "20px 0"},
                    ),
                    dcc.Tabs(
                        value="overview",
                        children=[
                            dcc.Tab(
                                label="Overview",
                                value="overview",
                                children=[
                                    html.Div(
                                        style={**PANEL_STYLE, "marginTop": "16px"},
                                        children=[dcc.Graph(id="trajectory-3d", style={"height": "620px"}, config={"displaylogo": False})],
                                    ),
                                    _graph_grid(
                                        [
                                            ("Position", "position-plot", "360px"),
                                            ("Velocity", "velocity-plot", "360px"),
                                            ("Controls", "controls-plot", "360px"),
                                            ("Attitude Quaternion", "attitude-plot", "360px"),
                                            ("Body Rates", "rates-plot", "360px"),
                                            ("Loop Timing", "timing-plot", "360px"),
                                        ]
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Metadata",
                                value="metadata",
                                children=[_table_panel("Run Metadata", "metadata-table")],
                            ),
                            dcc.Tab(
                                label="Events",
                                value="events",
                                children=[_table_panel("Lifecycle Events", "events-table")],
                            ),
                            dcc.Tab(
                                label="Cycles",
                                value="cycles",
                                children=[_table_panel("Cycle Records", "cycles-table")],
                            ),
                            dcc.Tab(
                                label="Raw JSON",
                                value="raw",
                                children=[
                                    html.Div(
                                        style={**PANEL_STYLE, "marginTop": "16px"},
                                        children=[
                                            html.Pre(
                                                id="raw-json",
                                                style={
                                                    "fontFamily": "SFMono-Regular, Consolas, monospace",
                                                    "fontSize": "12px",
                                                    "lineHeight": 1.45,
                                                    "margin": 0,
                                                    "maxHeight": "78vh",
                                                    "overflow": "auto",
                                                    "whiteSpace": "pre",
                                                },
                                            )
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


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
