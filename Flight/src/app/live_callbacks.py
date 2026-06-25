import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, html, no_update

from app.callbacks import _columns, _empty_figure, _header_tooltips, _layout, _rows, _tooltips, _trajectory_axis_ranges
from app.data import flatten_record, value_at
from app.live_data import (
    LiveRun,
    discover_live_run_dirs,
    frame_image_url,
    live_run_option,
    live_run_signature,
    load_live_run_cached,
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
MAX_TABLE_ROWS = 500
MAX_PLOT_POINTS = 5_000
RAW_PREVIEW_LIMIT_BYTES = 500_000


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
                option = live_run_option(path)
            except Exception:  # noqa: BLE001
                option = {"label": os.path.basename(path), "value": path}
            options.append(option)
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
            return _empty_live_dashboard("No captured runs were found in logs/runs/ or legacy data/live_runs/.")
        try:
            return _render_live_run_payload(str(run_path), live_run_signature(str(run_path)))
        except Exception as exc:  # noqa: BLE001
            return _empty_live_dashboard(f"Unable to load {run_path}: {exc}")

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
        Output("live-gate-map-3d", "figure"),
        Input("live-run-path", "value"),
        Input("live-frame-index", "value"),
    )
    def _render_frame(run_path: str | None, frame_index: int | None):
        if not run_path:
            return "", "No captured run selected.", "", _empty_figure()
        try:
            run = load_live_run_cached(run_path)
            if not run.frames:
                return "", "This run does not contain saved FPV frames.", "", _empty_figure()
            index = min(max(frame_index or 0, 0), len(run.frames) - 1)
            frame = run.frames[index]
            sync = nearest_cycle_for_frame(run, frame)
            gate_map_figure = _gate_map_figure(run, frame, sync.cycle, index)
            details = {
                "frame": {
                    "index": index,
                    "count": len(run.frames),
                    "frame_id": frame.frame_id,
                    "cycle": frame.cycle,
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
                f"Frame {index + 1}/{len(run.frames)} | id={frame.frame_id} | cycle={frame.cycle if frame.cycle is not None else 'n/a'} | "
                f"timestamp={frame.sim_time_ns} ns | {frame.jpeg_size:,} bytes"
            )
            return frame_image_url(run, index), caption, json.dumps(details, indent=2), gate_map_figure
        except Exception as exc:  # noqa: BLE001
            return "", f"Unable to render frame: {exc}", "", _empty_figure()

    @callback(
        Output("representative-gate-map-3d", "figure"),
        Output("representative-gate-summary", "children"),
        Output("representative-gate-table", "data"),
        Output("representative-gate-table", "columns"),
        Input("live-run-path", "value"),
        Input("run-poll", "n_intervals"),
    )
    def _render_representative_gate_map(run_path: str | None, _n_intervals: int):
        if not run_path:
            return _empty_figure(), "No captured run selected.", [], []
        try:
            run = load_live_run_cached(run_path)
            records = _stored_gate_map_records(run)
            representatives = _representative_gates(records)
            rows = _representative_gate_rows(representatives)
            summary = (
                f"{len(representatives)} representative gates from {len(records)} stored gate records "
                f"in {run.name}."
            )
            return _representative_gate_map_figure(run, records, representatives), summary, rows, _columns(rows)
        except Exception as exc:  # noqa: BLE001
            return _empty_figure(), f"Unable to render representative gate map: {exc}", [], []


@lru_cache(maxsize=8)
def _render_live_run_payload(run_path: str, _signature: tuple[int, int, int, int]):
    run = load_live_run_cached(run_path)
    event_rows = _rows(_recent_records(run.events, MAX_TABLE_ROWS))
    cycle_rows = _rows(_recent_records(run.cycles, MAX_TABLE_ROWS))
    frame_max = max(len(run.frames) - 1, 0)
    marks = _slider_marks(len(run.frames))
    return (
        "",
        _summary_cards(run),
        _trajectory_figure(run),
        _vector_figure(run, "Position (LOCAL_NED)", "position_local_ned_m", "LOCAL_NED position (m)"),
        _vector_figure(run, "Velocity (LOCAL_NED)", "velocity_local_ned_mps", "LOCAL_NED velocity (m/s)"),
        _vector_figure(run, "Acceleration (LOCAL_NED)", "acceleration_local_ned_mps2", "LOCAL_NED acceleration (m/s^2)"),
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
        _raw_preview(run),
        frame_max,
        marks,
    )


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
    cycles = _plot_cycles(run.cycles)
    points = [value_at(cycle, "telemetry", "position_local_ned_m") for cycle in cycles]
    points = [point for point in points if isinstance(point, list) and len(point) >= 3]
    plot_points = [_ned_point_to_plot(point) for point in points]
    if plot_points:
        fig.add_trace(go.Scatter3d(x=[p[0] for p in plot_points], y=[p[1] for p in plot_points], z=[p[2] for p in plot_points], mode="lines+markers", name="Local-NED odometry", line={"color": "#2563eb", "width": 5}, marker={"size": 2}))
    axis_ranges = _trajectory_axis_ranges(plot_points)
    ui_revision = f"live-trajectory:{run.path}"
    fig.update_layout(
        **_layout("Live Trajectory (LOCAL_NED)"),
        dragmode="orbit",
        uirevision=ui_revision,
        scene={
            "xaxis": {"title": "North (m)", "range": axis_ranges[0]},
            "yaxis": {"title": "West (-East) (m)", "range": axis_ranges[1]},
            "zaxis": {"title": "Up (-Down) (m)", "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
            "dragmode": "orbit",
            "uirevision": ui_revision,
        },
    )
    return fig


def _gate_map_figure(run: LiveRun, frame: Any, cycle: dict[str, Any] | None, frame_index: int) -> go.Figure:
    fig = go.Figure()
    cycle_number = frame.cycle
    if cycle_number is None and isinstance(cycle, dict) and isinstance(cycle.get("cycle"), int):
        cycle_number = cycle["cycle"]
    gates = _gate_map_records_for_cycle(run, cycle_number)
    if not gates:
        gates = _mapped_gates_for_frame(run, frame.frame_id)
    if not gates:
        gates = _cycle_gates(cycle)

    points: list[list[float]] = []
    planned_path = _planned_path_for_cycle(run, cycle_number)
    planned_plot_points = _planned_path_plot_points(planned_path)
    if planned_plot_points:
        points.extend(planned_plot_points)
        fig.add_trace(
            go.Scatter3d(
                x=[point[0] for point in planned_plot_points],
                y=[point[1] for point in planned_plot_points],
                z=[point[2] for point in planned_plot_points],
                mode="lines",
                name="Hot-start path",
                line={"color": "#0f766e", "width": 7},
                showlegend=True,
            )
        )
    telemetry = cycle.get("telemetry") if isinstance(cycle, dict) else None
    drone_position = telemetry.get("position_local_ned_m") if isinstance(telemetry, dict) else None
    for gate in gates:
        position = _gate_frame_position(gate, origin_local_ned_m=drone_position)
        quaternion = _gate_quaternion(gate)
        if not _point3(position):
            continue
        plot_position = _ned_point_to_plot(position)
        points.append(plot_position)
        fig.add_trace(
            go.Scatter3d(
                x=[plot_position[0]],
                y=[plot_position[1]],
                z=[plot_position[2]],
                mode="markers+text",
                name=str(gate.get("id", "gate")),
                text=[str(gate.get("id", "gate"))],
                textposition="top center",
                marker={"size": 5, "color": "#7c3aed"},
                showlegend=False,
            )
        )
        if _quat4(quaternion):
            for trace in _gate_traces(position, quaternion, str(gate.get("id", "gate"))):
                fig.add_trace(trace)

    if isinstance(telemetry, dict):
        drone_quaternion = telemetry.get("attitude_quaternion") or telemetry.get("attitude")
        if _point3(drone_position):
            drone_relative_position = [0.0, 0.0, 0.0]
            points.append(_ned_point_to_plot(drone_relative_position))
            for trace in _drone_traces(drone_relative_position, drone_quaternion):
                fig.add_trace(trace)

    title = f"Drone-Relative Gate Map At Frame {frame_index + 1} (id={frame.frame_id})"
    axis_ranges = _trajectory_axis_ranges(points)
    ui_revision = f"live-gate-map:{run.path}:{frame_index}"
    fig.update_layout(
        **_layout(title),
        dragmode="orbit",
        uirevision=ui_revision,
        scene={
            "xaxis": {"title": "Relative North (m)", "range": axis_ranges[0]},
            "yaxis": {"title": "Relative West (-East) (m)", "range": axis_ranges[1]},
            "zaxis": {"title": "Relative Up (-Down) (m)", "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
            "dragmode": "orbit",
            "uirevision": ui_revision,
        },
        annotations=[
            {
                "text": f"gates={len(gates)} | cycle={cycle.get('cycle', 'n/a') if isinstance(cycle, dict) else 'n/a'}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.98,
                "showarrow": False,
                "font": {"size": 12, "color": "#60708a"},
            }
        ],
    )
    return fig


def _vector_figure(run: LiveRun, title: str, field: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    cycles = _plot_cycles(run.cycles)
    times = telemetry_times_s(cycles)
    for index, axis in enumerate(("x", "y", "z")):
        values = [value_at(cycle, "telemetry", field, index) for cycle in cycles]
        if any(value is not None for value in values):
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name=axis, line={"color": AXIS_COLORS[index]}))
    fig.update_layout(**_layout(title), xaxis_title="Telemetry time (s)", yaxis_title=y_title)
    return fig


def _system_mode_figure(run: LiveRun) -> go.Figure:
    fig = go.Figure()
    cycles = _plot_cycles(run.cycles)
    times = telemetry_times_s(cycles)
    modes = [str(cycle.get("system_mode", "unknown")) for cycle in cycles]
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


def _cycle_gates(cycle: dict[str, Any] | None) -> list[dict[str, Any]]:
    gates = value_at(cycle or {}, "vision", "gates")
    return gates if isinstance(gates, list) else []


def _mapped_gates_for_frame(run: LiveRun, frame_id: int) -> list[dict[str, Any]]:
    records = run.raw.get("vision_frames")
    if not isinstance(records, list):
        return []
    for record in records:
        frame = record.get("frame") if isinstance(record, dict) else None
        if not isinstance(frame, dict) or int(frame.get("frame_id", -1)) != int(frame_id):
            continue
        gates = frame.get("mapped_gates")
        if isinstance(gates, list):
            return gates
    return []


def _stored_gate_map_records(run: LiveRun) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cycle in run.gate_map_cycles:
        gates = cycle.get("gate_map")
        if isinstance(gates, list):
            records.extend(gate for gate in gates if isinstance(gate, dict) and _point3(_gate_position(gate)))
    if records:
        return records

    for event in reversed(run.events):
        gate_map = event.get("gate_map") if isinstance(event, dict) else None
        if isinstance(gate_map, list):
            return [gate for gate in gate_map if isinstance(gate, dict) and _point3(_gate_position(gate))]

    records = run.raw.get("vision_frames")
    if isinstance(records, list):
        for record in reversed(records):
            frame = record.get("frame") if isinstance(record, dict) else None
            gates = frame.get("mapped_gates") if isinstance(frame, dict) else None
            if isinstance(gates, list):
                return [gate for gate in gates if isinstance(gate, dict) and _point3(_gate_position(gate))]
    return []


def _gate_map_records_for_cycle(run: LiveRun, cycle_number: int | None) -> list[dict[str, Any]]:
    if cycle_number is None:
        return []
    selected: dict[str, Any] | None = None
    for cycle in run.gate_map_cycles:
        if cycle.get("cycle") == cycle_number:
            selected = cycle
    if selected is None:
        return []
    gate_map = selected.get("gate_map")
    if not isinstance(gate_map, list):
        return []
    return [gate for gate in gate_map if isinstance(gate, dict)]


def _planned_path_for_cycle(run: LiveRun, cycle_number: int | None) -> dict[str, Any] | None:
    if cycle_number is None:
        return None
    planned_paths = run.raw.get("planned_paths")
    if not isinstance(planned_paths, list):
        return None
    selected: dict[str, Any] | None = None
    for record in planned_paths:
        if not isinstance(record, dict):
            continue
        record_cycle = record.get("cycle")
        if not isinstance(record_cycle, int) or record_cycle > cycle_number:
            continue
        planned_path = record.get("planned_path")
        if isinstance(planned_path, dict):
            selected = planned_path
    return selected


def _planned_path_plot_points(planned_path: dict[str, Any] | None) -> list[list[float]]:
    if not isinstance(planned_path, dict):
        return []
    relative_points = planned_path.get("points_relative_ned_m")
    if not isinstance(relative_points, list):
        return []
    plot_points: list[list[float]] = []
    for point in relative_points:
        if _point3(point):
            plot_points.append(_ned_point_to_plot(point))
    return plot_points


def _representative_gates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, gate in enumerate(records):
        sequence = gate.get("sequence")
        if not isinstance(sequence, int):
            sequence = index
        grouped.setdefault(sequence, []).append(gate)

    representatives: list[dict[str, Any]] = []
    for sequence, gates in sorted(grouped.items()):
        positions = np.asarray([_gate_position(gate) for gate in gates], dtype=float)
        relative_positions = [_gate_relative_position(gate) for gate in gates]
        relative_positions = [position for position in relative_positions if _point3(position)]
        quaternions = np.asarray(
            [_gate_quaternion(gate) or [1.0, 0.0, 0.0, 0.0] for gate in gates],
            dtype=float,
        )
        quaternion = np.median(quaternions, axis=0)
        quaternion = quaternion / max(np.linalg.norm(quaternion), 1e-12)
        confidences = [float(gate.get("confidence", 0.0)) for gate in gates if isinstance(gate.get("confidence"), (int, float))]
        representatives.append(
            {
                "id": f"sequence-{sequence}",
                "sequence": sequence,
                "position_local_ned_m": np.median(positions, axis=0).tolist(),
                "position_relative_ned_m": np.median(np.asarray(relative_positions, dtype=float), axis=0).tolist()
                if relative_positions
                else None,
                "quaternion": quaternion.tolist(),
                "confidence": float(np.median(confidences)) if confidences else 0.0,
                "observation_count": len(gates),
            }
        )
    return representatives


def _representative_gate_map_figure(
    run: LiveRun,
    records: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()
    raw_points = [_gate_position(gate) for gate in records if _point3(_gate_position(gate))]
    plot_raw_points = [_ned_point_to_plot(point) for point in raw_points]
    points: list[list[float]] = list(plot_raw_points)
    if plot_raw_points:
        fig.add_trace(
            go.Scatter3d(
                x=[point[0] for point in plot_raw_points],
                y=[point[1] for point in plot_raw_points],
                z=[point[2] for point in plot_raw_points],
                mode="markers",
                name="Stored records",
                marker={"size": 2, "color": "#94a3b8", "opacity": 0.28},
                showlegend=True,
            )
        )

    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b", "#0f766e")
    for index, gate in enumerate(representatives):
        position = _gate_position(gate)
        quaternion = _gate_quaternion(gate)
        if not _point3(position):
            continue
        plot_position = _ned_point_to_plot(position)
        points.append(plot_position)
        gate_id = str(gate.get("id", f"gate-{index}"))
        color = colors[index % len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=[plot_position[0]],
                y=[plot_position[1]],
                z=[plot_position[2]],
                mode="markers+text",
                name=gate_id,
                text=[gate_id],
                textposition="top center",
                marker={"size": 7, "color": color},
                showlegend=True,
            )
        )
        if _quat4(quaternion):
            for trace in _gate_traces(position, quaternion, gate_id, color=color):
                fig.add_trace(trace)

    axis_ranges = _trajectory_axis_ranges(points)
    fig.update_layout(
        **_layout("Representative Gate Map"),
        dragmode="orbit",
        uirevision=f"representative-gates:{run.path}",
        scene={
            "xaxis": {"title": "North (m)", "range": axis_ranges[0]},
            "yaxis": {"title": "West (-East) (m)", "range": axis_ranges[1]},
            "zaxis": {"title": "Up (-Down) (m)", "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
            "dragmode": "orbit",
        },
    )
    return fig


def _representative_gate_rows(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in gates:
        position = _gate_position(gate) or [None, None, None]
        relative_position = _gate_relative_position(gate) or [None, None, None]
        quaternion = _gate_quaternion(gate) or [None, None, None, None]
        rows.append(
            {
                "id": gate.get("id", ""),
                "sequence": gate.get("sequence", ""),
                "observations": gate.get("observation_count", ""),
                "north_m": _round(position[0]),
                "east_m": _round(position[1]),
                "down_m": _round(position[2]),
                "relative_north_m": _round(relative_position[0]),
                "relative_east_m": _round(relative_position[1]),
                "relative_down_m": _round(relative_position[2]),
                "qw": _round(quaternion[0]),
                "qx": _round(quaternion[1]),
                "qy": _round(quaternion[2]),
                "qz": _round(quaternion[3]),
                "confidence": _round(gate.get("confidence")),
            }
        )
    return rows


def _recent_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    return records[-limit:]


def _plot_cycles(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(cycles) <= MAX_PLOT_POINTS:
        return cycles
    stride = max(1, int(np.ceil(len(cycles) / MAX_PLOT_POINTS)))
    sampled = cycles[::stride]
    if sampled[-1] is not cycles[-1]:
        sampled.append(cycles[-1])
    return sampled


def _raw_preview(run: LiveRun) -> str:
    raw = run.raw
    raw_size = _run_json_size(run)
    if raw_size <= RAW_PREVIEW_LIMIT_BYTES:
        return json.dumps(raw, indent=2)
    preview = {
        "path": run.path,
        "name": run.name,
        "schema_version": run.schema_version,
        "metadata": run.metadata,
        "counts": {
            "events": len(run.events),
            "cycles": len(run.cycles),
            "frames": len(run.frames),
            "gate_map_cycles": len(run.gate_map_cycles),
            "vision_frames": len(raw.get("vision_frames", [])) if isinstance(raw.get("vision_frames"), list) else 0,
            "planned_paths": len(raw.get("planned_paths", [])) if isinstance(raw.get("planned_paths"), list) else 0,
        },
        "events_tail": _recent_records(run.events, 50),
        "cycles_tail": _recent_records(run.cycles, 20),
        "note": f"Raw JSON is {raw_size:,} bytes, so this tab shows a bounded preview for dashboard responsiveness.",
    }
    return json.dumps(preview, indent=2)


def _run_json_size(run: LiveRun) -> int:
    try:
        return os.path.getsize(os.path.join(run.path, "run.json"))
    except OSError:
        return RAW_PREVIEW_LIMIT_BYTES + 1


def _gate_traces(position: list[float], quaternion: list[float], gate_id: str, *, color: str = "#7c3aed") -> list[go.Scatter3d]:
    center = _ned_array_to_plot(np.asarray(position, dtype=float))
    rotation = _ned_rotation_to_plot(_rotation_matrix(quaternion))
    outer = _gate_square_points(center, rotation, size=2.7)
    inner = _gate_square_points(center, rotation, size=1.5)
    normal = center + rotation[:, 1] * 1.6
    traces = [
        go.Scatter3d(x=outer[:, 0], y=outer[:, 1], z=outer[:, 2], mode="lines", name=f"{gate_id} outer", line={"color": color, "width": 5}, showlegend=False),
        go.Scatter3d(x=inner[:, 0], y=inner[:, 1], z=inner[:, 2], mode="lines", name=f"{gate_id} inner", line={"color": color, "width": 3}, showlegend=False, opacity=0.65),
        go.Scatter3d(x=[center[0], normal[0]], y=[center[1], normal[1]], z=[center[2], normal[2]], mode="lines", name=f"{gate_id} normal", line={"color": "#f59e0b", "width": 5}, showlegend=False),
    ]
    return traces


def _drone_traces(position: list[float], quaternion: list[float] | None) -> list[go.Scatter3d]:
    center = _ned_array_to_plot(np.asarray(position, dtype=float))
    rotation = _ned_rotation_to_plot(_rotation_matrix(quaternion if _quat4(quaternion) else [1.0, 0.0, 0.0, 0.0]))
    nose = center + rotation[:, 0] * 1.2
    right = center + rotation[:, 1] * 0.55
    left = center - rotation[:, 1] * 0.55
    return [
        go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode="markers+text", name="Drone", text=["drone"], textposition="bottom center", marker={"size": 7, "color": "#dc2626"}, showlegend=False),
        go.Scatter3d(x=[center[0], nose[0]], y=[center[1], nose[1]], z=[center[2], nose[2]], mode="lines", name="Drone heading", line={"color": "#dc2626", "width": 7}, showlegend=False),
        go.Scatter3d(x=[left[0], right[0]], y=[left[1], right[1]], z=[left[2], right[2]], mode="lines", name="Drone body", line={"color": "#111827", "width": 5}, showlegend=False),
    ]


def _gate_square_points(center: np.ndarray, rotation: np.ndarray, *, size: float) -> np.ndarray:
    half = float(size) / 2.0
    local = np.array(
        [
            [-half, 0.0, -half],
            [half, 0.0, -half],
            [half, 0.0, half],
            [-half, 0.0, half],
            [-half, 0.0, -half],
        ],
        dtype=float,
    )
    return center + local @ rotation.T


def _ned_point_to_plot(point: list[float]) -> list[float]:
    return [float(point[0]), -float(point[1]), -float(point[2])]


def _ned_array_to_plot(point: np.ndarray) -> np.ndarray:
    return np.asarray([float(point[0]), -float(point[1]), -float(point[2])], dtype=float)


def _ned_rotation_to_plot(rotation: np.ndarray) -> np.ndarray:
    return np.diag([1.0, -1.0, -1.0]) @ np.asarray(rotation, dtype=float)


def _rotation_matrix(quaternion: list[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _point3(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 3 and all(isinstance(item, (int, float)) for item in value[:3])


def _quat4(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4])


def _gate_position(gate: dict[str, Any]) -> list[float] | None:
    position = gate.get("position_local_ned_m") or gate.get("pos")
    return position if _point3(position) else None


def _gate_frame_position(gate: dict[str, Any], *, origin_local_ned_m: list[float] | None = None) -> list[float] | None:
    relative_position = gate.get("position_relative_ned_m")
    if _point3(relative_position):
        return relative_position
    local_position = gate.get("position_local_ned_m") or gate.get("pos")
    if _point3(local_position) and _point3(origin_local_ned_m):
        relative = np.asarray(local_position, dtype=float) - np.asarray(origin_local_ned_m, dtype=float)
        return [float(value) for value in relative]
    return local_position if _point3(local_position) else None


def _gate_relative_position(gate: dict[str, Any]) -> list[float] | None:
    position = gate.get("position_relative_ned_m")
    return position if _point3(position) else None


def _gate_quaternion(gate: dict[str, Any]) -> list[float] | None:
    quaternion = gate.get("quaternion") or gate.get("quat")
    return quaternion if _quat4(quaternion) else None


def _round(value: Any, digits: int = 3) -> float | str:
    if not isinstance(value, (int, float)):
        return ""
    return round(float(value), digits)
