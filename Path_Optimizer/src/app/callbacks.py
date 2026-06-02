import json
import os
from functools import lru_cache
from typing import Any

import plotly.graph_objects as go
import numpy as np
from dash import Dash, Input, Output, State, callback, html, no_update

from app.data import RunLog, cycle_times_s, discover_simulation_run_files, flatten_record, load_run, value_at


GRAPH_IDS = (
    "trajectory-3d",
    "position-plot",
    "velocity-plot",
    "controls-plot",
    "attitude-plot",
    "rates-plot",
    "timing-plot",
    "modes-plot",
    "disturbance-plot",
)
AXIS_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")


def register_callbacks(app: Dash, *, root_dir: str) -> None:
    @callback(
        Output("run-path", "options"),
        Output("run-path", "value"),
        Input("run-poll", "n_intervals"),
        State("run-path", "value"),
    )
    def _refresh_runs(_n_intervals: int, selected_path: str | None):
        paths = discover_simulation_run_files(root_dir)
        options = []
        for path in paths:
            try:
                run = _load_run(path)
                label = f"{run.label}  |  {run.name}"
            except Exception:  # noqa: BLE001
                label = os.path.basename(path)
            options.append({"label": label, "value": path})
        values = {option["value"] for option in options}
        selected_value = no_update if selected_path in values else (paths[0] if paths else None)
        return options, selected_value

    @callback(
        Output("load-error", "children"),
        Output("summary-cards", "children"),
        *(Output(graph_id, "figure") for graph_id in GRAPH_IDS),
        Output("metadata-table", "data"),
        Output("metadata-table", "columns"),
        Output("events-table", "data"),
        Output("events-table", "columns"),
        Output("events-table", "tooltip_header"),
        Output("events-table", "tooltip_data"),
        Output("cycles-table", "data"),
        Output("cycles-table", "columns"),
        Output("cycles-table", "tooltip_header"),
        Output("cycles-table", "tooltip_data"),
        Output("raw-json", "children"),
        Input("run-path", "value"),
    )
    def _render_run(run_path: str | None):
        if not run_path:
            return _empty_dashboard("No simulation logs were found in logs/sim/.")
        try:
            run = _load_run(run_path)
        except Exception as exc:  # noqa: BLE001
            return _empty_dashboard(f"Unable to load {run_path}: {exc}")

        metadata_rows = [{"field": key, "value": _display(value)} for key, value in flatten_record(run.metadata).items()]
        event_rows = _rows(run.events)
        cycle_rows = _rows(run.cycles)
        return (
            "",
            _summary_cards(run),
            _trajectory_figure(run),
            _vector_figure(run, "Position", "position_local_ned_m", "Position (m)"),
            _vector_figure(run, "Velocity", "velocity_local_ned_mps", "Velocity (m/s)"),
            _controls_figure(run),
            _vector_figure(run, "Attitude Quaternion", "attitude_quaternion", "Quaternion", axes=("w", "x", "y", "z")),
            _vector_figure(run, "Body Rates", "body_rates_rps", "Rate (rad/s)"),
            _timing_figure(run),
            _modes_figure(run),
            _disturbance_figure(run),
            metadata_rows,
            _columns(metadata_rows),
            event_rows,
            _columns(event_rows),
            _header_tooltips(event_rows),
            _tooltips(event_rows),
            cycle_rows,
            _columns(cycle_rows),
            _header_tooltips(cycle_rows),
            _tooltips(cycle_rows),
            json.dumps(run.raw, indent=2),
        )


def _empty_dashboard(message: str):
    empty = _empty_figure()
    return (message, [], *(empty for _ in GRAPH_IDS), [], [], [], [], {}, [], [], [], {}, [], "")


def _summary_cards(run: RunLog) -> list[html.Div]:
    cycles = run.cycles
    elapsed_s = cycle_times_s(cycles)[-1] if cycles else 0.0
    modes = sorted({
        str(cycle.get("system_mode") or value_at(cycle, "modes", "system") or "unknown")
        for cycle in cycles
    })
    lateness = [_number(cycle.get("deadline_lateness_ms")) for cycle in cycles]
    max_lateness = max((value for value in lateness if value is not None), default=None)
    values = [
        ("Scenario", run.metadata.get("scenario", "n/a")),
        ("Run Timestamp", run.label),
        ("Cycles", len(cycles)),
        ("Sim Duration", f"{elapsed_s:.3f} s"),
        ("Events", len(run.events)),
        ("System Modes", ", ".join(modes) or "n/a"),
        ("Max Lateness", f"{max_lateness:.3f} ms" if max_lateness is not None else "n/a"),
        ("Schema", run.schema_version),
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


def _trajectory_figure(run: RunLog) -> go.Figure:
    fig = go.Figure()
    all_points: list[list[float]] = []
    for label, key, color in (("Simulator truth", "simulator_truth", "#2563eb"), ("Estimated", "estimated_state", "#dc2626")):
        points = [value_at(cycle, key, "position_local_ned_m") for cycle in run.cycles]
        points = [point for point in points if isinstance(point, list) and len(point) >= 3]
        if points:
            all_points.extend(points)
            fig.add_trace(go.Scatter3d(x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points], mode="lines", name=label, line={"color": color, "width": 5}))
    target = run.metadata.get("target_position_local_ned_m")
    if isinstance(target, list) and len(target) >= 3:
        all_points.append(target)
        fig.add_trace(go.Scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], mode="markers", name="Target", marker={"color": "#16a34a", "size": 7, "symbol": "diamond"}))
    frames = []
    frame_indices = _playback_frame_indices(run)
    trace_indices = []
    if frame_indices:
        position, quaternion = _quadrotor_pose(run.cycles[frame_indices[0]])
        trace_indices = list(range(len(fig.data), len(fig.data) + 3))
        for trace in _quadrotor_traces(position, quaternion):
            fig.add_trace(trace)
        frames = [
            go.Frame(
                name=str(index),
                data=_quadrotor_traces(*_quadrotor_pose(run.cycles[index])),
                traces=trace_indices,
            )
            for index in frame_indices
        ]
    axis_ranges = _trajectory_axis_ranges(all_points)
    ui_revision = f"trajectory:{run.path}"
    fig.update_layout(
        **_layout("Trajectory (Local NED)"),
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
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.02,
            "y": 1.08,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": _playback_interval_ms(run), "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "currentvalue": {"prefix": "Simulation time: "},
            "pad": {"t": 36},
            "steps": [
                {"label": f"{_cycle_time_s(run.cycles[index]):.2f}s", "method": "animate", "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for index in frame_indices
            ],
        }],
    )
    fig.frames = frames
    return fig


def _quadrotor_traces(position: list[float], quaternion: list[float]) -> list[go.Scatter3d]:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=float)
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])
    center = np.asarray(position, dtype=float)
    traces = []
    for axis, color in (([0.3, 0.3, 0.0], "#111827"), ([-0.3, 0.3, 0.0], "#64748b")):
        arm = rotation @ np.asarray(axis)
        ends = np.vstack((center - arm, center + arm))
        traces.append(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="lines", line={"color": color, "width": 8}, showlegend=False))
    nose = center + rotation @ np.array([0.5, 0.0, 0.0])
    traces.append(go.Scatter3d(x=[center[0], nose[0]], y=[center[1], nose[1]], z=[center[2], nose[2]], mode="lines", line={"color": "#dc2626", "width": 7}, showlegend=False))
    return traces


def _trajectory_axis_ranges(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

    bounds = [(min(float(point[index]) for point in points), max(float(point[index]) for point in points)) for index in range(3)]
    span = max(maximum - minimum for minimum, maximum in bounds)
    half_span = max(span * 0.6, 1.0)
    return [
        [(minimum + maximum) / 2 - half_span, (minimum + maximum) / 2 + half_span]
        for minimum, maximum in bounds
    ]


def _vector_figure(run: RunLog, title: str, field: str, y_title: str, *, axes: tuple[str, ...] = ("x", "y", "z")) -> go.Figure:
    fig = go.Figure()
    times = cycle_times_s(run.cycles)
    for source_label, source_key, dash in (("Truth", "simulator_truth", "solid"), ("Estimate", "estimated_state", "dash")):
        for index, axis in enumerate(axes):
            values = [value_at(cycle, source_key, field, index) for cycle in run.cycles]
            if any(value is not None for value in values):
                fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name=f"{source_label} {axis}", line={"color": AXIS_COLORS[index], "dash": dash}))
    fig.update_layout(**_layout(title), xaxis_title="Simulation time (s)", yaxis_title=y_title)
    return fig


def _controls_figure(run: RunLog) -> go.Figure:
    fig = go.Figure()
    times = cycle_times_s(run.cycles)
    for index in range(4):
        motors = [value_at(cycle, "simulator_truth", "motor_commands", index) for cycle in run.cycles]
        if any(value is not None for value in motors):
            fig.add_trace(go.Scatter(x=times, y=motors, mode="lines", name=f"Motor {index + 1}", line={"color": AXIS_COLORS[index]}))
    thrust = [value_at(cycle, "command", "set_attitude_target", "thrust") for cycle in run.cycles]
    if any(value is not None for value in thrust):
        fig.add_trace(go.Scatter(x=times, y=thrust, mode="lines", name="Command thrust", line={"color": "#111827", "dash": "dot", "width": 3}))
    fig.update_layout(**_layout("Controls"), xaxis_title="Simulation time (s)", yaxis_title="Normalized command")
    return fig


def _timing_figure(run: RunLog) -> go.Figure:
    fig = go.Figure()
    times = cycle_times_s(run.cycles)
    for label, key, color in (("Deadline lateness", "deadline_lateness_ms", "#dc2626"), ("Wall elapsed", "wall_elapsed_ms", "#2563eb")):
        values = [cycle.get(key) for cycle in run.cycles]
        if any(value is not None for value in values):
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines+markers", name=label, line={"color": color}))
    fig.update_layout(**_layout("Loop Timing"), xaxis_title="Simulation time (s)", yaxis_title="Milliseconds")
    return fig


def _modes_figure(run: RunLog) -> go.Figure:
    fig = go.Figure()
    times = cycle_times_s(run.cycles)
    for label, key in (("System", "system"), ("Flight", "flight"), ("Control", "control")):
        values = [value_at(cycle, "modes", key) for cycle in run.cycles]
        if any(value is not None for value in values):
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name=label, line={"shape": "hv"}))
    fig.update_layout(**_layout("Modes"), xaxis_title="Simulation time (s)")
    return fig


def _disturbance_figure(run: RunLog) -> go.Figure:
    fig = go.Figure()
    times = cycle_times_s(run.cycles)
    for index, axis in enumerate(("x", "y", "z")):
        values = [value_at(cycle, "disturbance", "force_i_n", index) for cycle in run.cycles]
        if any(value is not None for value in values):
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name=f"Force {axis}", line={"color": AXIS_COLORS[index]}))
    fig.update_layout(**_layout("Turbulence Force"), xaxis_title="Simulation time (s)", yaxis_title="Force (N)")
    return fig


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{key: _display(value) for key, value in flatten_record(record).items()} for record in records]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _columns(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return [{"name": key, "id": key} for key in keys]


def _tooltips(rows: list[dict[str, Any]]) -> list[dict[str, dict[str, str]]]:
    return [{key: {"value": str(value), "type": "text"} for key, value in row.items()} for row in rows]


def _header_tooltips(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {key: key for key in dict.fromkeys(key for row in rows for key in row)}


def _display(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _layout(title: str) -> dict[str, Any]:
    return {
        "template": "plotly_white",
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "margin": {"l": 54, "r": 22, "t": 54, "b": 46},
        "legend": {"orientation": "h", "y": -0.22},
        "hovermode": "x unified",
    }


def _empty_figure() -> go.Figure:
    figure = go.Figure()
    figure.update_layout(template="plotly_white")
    return figure


def _playback_interval_ms(run: RunLog) -> int:
    times = cycle_times_s(run.cycles)
    if len(times) < 2:
        return 100
    dt_s = max(times[1] - times[0], 0.001)
    return max(33, round(dt_s * 1000))


def _playback_frame_indices(run: RunLog) -> list[int]:
    times = cycle_times_s(run.cycles)
    if not times:
        return []
    if len(times) == 1:
        return [0]
    dt_ms = max((times[1] - times[0]) * 1000, 1.0)
    stride = max(1, round(_playback_interval_ms(run) / dt_ms))
    indices = list(range(0, len(times), stride))
    if indices[-1] != len(times) - 1:
        indices.append(len(times) - 1)
    return indices


def _quadrotor_pose(cycle: dict[str, Any]) -> tuple[list[float], list[float]]:
    return (
        value_at(cycle, "simulator_truth", "position_local_ned_m"),
        value_at(cycle, "simulator_truth", "attitude_quaternion"),
    )


def _cycle_time_s(cycle: dict[str, Any]) -> float:
    return float(cycle.get("sim_time_ns", 0)) / 1_000_000_000


@lru_cache(maxsize=32)
def _load_run(path: str) -> RunLog:
    return load_run(path)
