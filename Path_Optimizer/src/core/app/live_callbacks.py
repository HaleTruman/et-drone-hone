from __future__ import annotations

import json
import os
from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, html, no_update

from core.app.callbacks import _columns, _empty_figure, _header_tooltips, _layout, _rows, _tooltips, _trajectory_axis_ranges
from core.app.data import flatten_record, value_at
from core.app.live_data import (
    LiveRun,
    discover_live_run_dirs,
    frame_data_uri,
    load_live_run,
    nearest_cycle_for_frame,
    telemetry_times_s,
)


LIVE_GRAPH_IDS = (
    "live-trajectory-3d",
    "live-position-plot",
    "live-velocity-plot",
    "live-acceleration-plot",
    "live-rates-plot",
    "live-system-mode-plot",
)
AXIS_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")


def register_live_callbacks(app: Dash, *, root_dir: str) -> None:
    @callback(
        Output("live-run-path", "options"),
        Output("live-run-path", "value"),
        Input("run-poll", "n_intervals"),
        State("live-run-path", "value"),
    )
    def _refresh_live_runs(_n_intervals: int, selected_path: str | None):
        paths = discover_live_run_dirs(root_dir)
        options = []
        for path in paths:
            try:
                run = load_live_run(path)
                label = f"{run.label}  |  {run.name}"
            except Exception:  # noqa: BLE001
                label = os.path.basename(path)
            options.append({"label": label, "value": path})
        values = {option["value"] for option in options}
        selected_value = no_update if selected_path in values else (paths[0] if paths else None)
        return options, selected_value

    @callback(
        Output("live-load-error", "children"),
        Output("live-summary-cards", "children"),
        *(Output(graph_id, "figure") for graph_id in LIVE_GRAPH_IDS),
        Output("live-events-table", "data"),
        Output("live-events-table", "columns"),
        Output("live-events-table", "tooltip_header"),
        Output("live-events-table", "tooltip_data"),
        Output("live-cycles-table", "data"),
        Output("live-cycles-table", "columns"),
        Output("live-cycles-table", "tooltip_header"),
        Output("live-cycles-table", "tooltip_data"),
        Output("live-raw-json", "children"),
        Output("live-frame-index", "max"),
        Output("live-frame-index", "marks"),
        Input("live-run-path", "value"),
        Input("run-poll", "n_intervals"),
    )
    def _render_live_run(run_path: str | None, _n_intervals: int):
        if not run_path:
            return _empty_live_dashboard("No captured runs were found in data/live_runs/.")
        try:
            run = load_live_run(run_path)
        except Exception as exc:  # noqa: BLE001
            return _empty_live_dashboard(f"Unable to load {run_path}: {exc}")

        event_rows = _rows(run.events)
        cycle_rows = _rows(run.cycles)
        frame_max = max(len(run.frames) - 1, 0)
        marks = _slider_marks(len(run.frames))
        return (
            "",
            _summary_cards(run),
            _trajectory_figure(run),
            _vector_figure(run, "Position", "position_local_ned_m", "Position (m)"),
            _vector_figure(run, "Velocity", "velocity_local_ned_mps", "Velocity (m/s)"),
            _vector_figure(run, "Acceleration", "acceleration_local_ned_mps2", "Acceleration (m/s^2)"),
            _vector_figure(run, "Body Rates", "body_rates_rps", "Rate (rad/s)"),
            _system_mode_figure(run),
            event_rows,
            _columns(event_rows),
            _header_tooltips(event_rows),
            _tooltips(event_rows),
            cycle_rows,
            _columns(cycle_rows),
            _header_tooltips(cycle_rows),
            _tooltips(cycle_rows),
            json.dumps(run.raw, indent=2),
            frame_max,
            marks,
        )

    @callback(
        Output("live-frame-index", "value"),
        Input("live-run-path", "value"),
        Input("live-frame-previous", "n_clicks"),
        Input("live-frame-next", "n_clicks"),
        State("live-frame-index", "value"),
        State("live-frame-index", "max"),
    )
    def _change_frame(_run_path: str | None, _previous: int, _next: int, frame_index: int | None, frame_max: int | None):
        if ctx.triggered_id == "live-run-path":
            return 0
        value = frame_index or 0
        if ctx.triggered_id == "live-frame-previous":
            return max(value - 1, 0)
        if ctx.triggered_id == "live-frame-next":
            return min(value + 1, frame_max or 0)
        return min(value, frame_max or 0)

    @callback(
        Output("live-frame-image", "src"),
        Output("live-frame-caption", "children"),
        Output("live-frame-telemetry", "children"),
        Input("live-run-path", "value"),
        Input("live-frame-index", "value"),
    )
    def _render_frame(run_path: str | None, frame_index: int | None):
        if not run_path:
            return "", "No captured run selected.", ""
        try:
            run = load_live_run(run_path)
            if not run.frames:
                return "", "This run does not contain saved FPV frames.", ""
            index = min(max(frame_index or 0, 0), len(run.frames) - 1)
            frame = run.frames[index]
            sync = nearest_cycle_for_frame(run, frame)
            details = {
                "frame": {
                    "index": index,
                    "count": len(run.frames),
                    "frame_id": frame.frame_id,
                    "sim_time_ns": frame.sim_time_ns,
                    "jpeg_size": frame.jpeg_size,
                    "path": frame.path,
                },
                "synchronized_cycle": {
                    "cycle_index": sync.cycle_index,
                    "target_telemetry_sim_time_ns": sync.target_sim_time_ns,
                    "alignment_error_ms": sync.error_ms,
                    "record": sync.cycle,
                },
            }
            caption = (
                f"Frame {index + 1}/{len(run.frames)} | id={frame.frame_id} | "
                f"timestamp={frame.sim_time_ns} ns | {frame.jpeg_size:,} bytes"
            )
            return frame_data_uri(run, frame), caption, json.dumps(details, indent=2)
        except Exception as exc:  # noqa: BLE001
            return "", f"Unable to render frame: {exc}", ""


def _empty_live_dashboard(message: str):
    empty = _empty_figure()
    return (message, [], *(empty for _ in LIVE_GRAPH_IDS), [], [], {}, [], [], [], {}, [], "", 0, {0: "0"})


def _summary_cards(run: LiveRun) -> list[html.Div]:
    times = telemetry_times_s(run.cycles)
    telemetry = [cycle.get("telemetry") for cycle in run.cycles if isinstance(cycle.get("telemetry"), dict)]
    modes = sorted({str(cycle.get("system_mode", "unknown")) for cycle in run.cycles})
    positions = [sample.get("position_local_ned_m") for sample in telemetry]
    positions = [position for position in positions if isinstance(position, list) and len(position) >= 3]
    displacement = _distance(positions[0], positions[-1]) if len(positions) >= 2 else None
    collisions = max((len(cycle.get("collisions", [])) for cycle in run.cycles if isinstance(cycle.get("collisions"), list)), default=0)
    values = [
        ("Scenario", run.metadata.get("scenario", "n/a")),
        ("Run Timestamp", run.label),
        ("Telemetry Cycles", len(run.cycles)),
        ("Telemetry Span", f"{times[-1]:.3f} s" if times else "n/a"),
        ("FPV Frames", len(run.frames)),
        ("Lifecycle Events", len(run.events)),
        ("System Modes", ", ".join(modes) or "n/a"),
        ("Displacement", f"{displacement:.3f} m" if displacement is not None else "n/a"),
        ("Collisions", collisions),
    ]
    return [
        html.Div(
            style={"background": "#fff", "border": "1px solid #dbe3ef", "borderRadius": "12px", "padding": "13px 15px"},
            children=[
                html.Div(label, style={"color": "#60708a", "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.06em", "textTransform": "uppercase"}),
                html.Div(str(value), style={"fontSize": "15px", "fontWeight": 700, "marginTop": "5px"}),
            ],
        )
        for label, value in values
    ]


def _trajectory_figure(run: LiveRun) -> go.Figure:
    fig = go.Figure()
    points = [value_at(cycle, "telemetry", "position_local_ned_m") for cycle in run.cycles]
    points = [point for point in points if isinstance(point, list) and len(point) >= 3]
    if points:
        fig.add_trace(go.Scatter3d(x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points], mode="lines+markers", name="ODOMETRY", line={"color": "#2563eb", "width": 5}, marker={"size": 2}))
    axis_ranges = _trajectory_axis_ranges(points)
    ui_revision = f"live-trajectory:{run.path}"
    fig.update_layout(
        **_layout("Live Trajectory (Local NED)"),
        dragmode="orbit",
        uirevision=ui_revision,
        scene={
            "xaxis": {"title": "North (m)", "range": axis_ranges[0]},
            "yaxis": {"title": "East (m)", "range": axis_ranges[1]},
            "zaxis": {"title": "Down (m)", "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
            "dragmode": "orbit",
            "uirevision": ui_revision,
        },
    )
    return fig


def _vector_figure(run: LiveRun, title: str, field: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    times = telemetry_times_s(run.cycles)
    for index, axis in enumerate(("x", "y", "z")):
        values = [value_at(cycle, "telemetry", field, index) for cycle in run.cycles]
        if any(value is not None for value in values):
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name=axis, line={"color": AXIS_COLORS[index]}))
    fig.update_layout(**_layout(title), xaxis_title="Telemetry time (s)", yaxis_title=y_title)
    return fig


def _system_mode_figure(run: LiveRun) -> go.Figure:
    fig = go.Figure()
    times = telemetry_times_s(run.cycles)
    modes = [str(cycle.get("system_mode", "unknown")) for cycle in run.cycles]
    fig.add_trace(go.Scatter(x=times, y=modes, mode="lines+markers", name="System mode", line={"color": "#9333ea", "shape": "hv"}))
    fig.update_layout(**_layout("System Mode"), xaxis_title="Telemetry time (s)", yaxis_title="System mode")
    return fig


def _slider_marks(frame_count: int) -> dict[int, str]:
    if frame_count <= 1:
        return {0: "0"}
    maximum = frame_count - 1
    return {value: str(value + 1) for value in sorted({0, maximum // 2, maximum})}


def _distance(start: list[float], end: list[float]) -> float:
    return sum((float(end[index]) - float(start[index])) ** 2 for index in range(3)) ** 0.5
