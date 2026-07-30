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
    vision_observation_for_frame,
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
GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
GATE_DEPTH_M = 0.26
GATE_NORMAL_INDICATOR_M = 1.2
CAMERA_VERTICAL_FOV_DEG = 90.0
CAMERA_ASPECT_RATIO = 16.0 / 9.0


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
        Output("live-frame-observation", "children"),
        Output("live-gate-map-3d", "figure"),
        Input("live-run-path", "value"),
        Input("live-frame-index", "value"),
    )
    def _render_frame(run_path: str | None, frame_index: int | None):
        if not run_path:
            return "", "No captured run selected.", "", "", _empty_figure()
        try:
            run = load_live_run_cached(run_path)
            if not run.frames:
                return "", "This run does not contain saved FPV frames.", "", "", _empty_figure()
            index = min(max(frame_index or 0, 0), len(run.frames) - 1)
            frame = run.frames[index]
            sync = nearest_cycle_for_frame(run, frame)
            observation = vision_observation_for_frame(run, frame)
            gate_map_figure = _vision_observation_figure(run, frame, observation, index)
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
            observation_details = {
                "frame": {
                    "index": index,
                    "count": len(run.frames),
                    "frame_id": frame.frame_id,
                    "sim_time_ns": frame.sim_time_ns,
                },
                "vision_observation": observation,
            }
            if observation is None:
                observation_details["note"] = "No vision observation record matched this frame_id."
            caption = (
                f"Frame {index + 1}/{len(run.frames)} | id={frame.frame_id} | cycle={frame.cycle if frame.cycle is not None else 'n/a'} | "
                f"timestamp={frame.sim_time_ns} ns | {frame.jpeg_size:,} bytes"
            )
            return frame_image_url(run, index), caption, json.dumps(details, indent=2), json.dumps(observation_details, indent=2), gate_map_figure
        except Exception as exc:  # noqa: BLE001
            return "", f"Unable to render frame: {exc}", "", "", _empty_figure()

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
def _render_live_run_payload(run_path: str, _signature: tuple[int, int, int, int, int]):
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
        ("Vision Observations", len(run.vision_observations)),
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
    for path_record, name, color in (
        (_path_for_cycle(run, "test_paths", "test_path", cycle_number), "Test path", "#ffffff"),
        (_path_for_cycle(run, "planned_paths", "planned_path", cycle_number), "Planned path", "#dc2626"),
    ):
        plot_points = _planned_path_plot_points(path_record)
        if not plot_points:
            continue
        points.extend(plot_points)
        fig.add_trace(
            go.Scatter3d(
                x=[point[0] for point in plot_points],
                y=[point[1] for point in plot_points],
                z=[point[2] for point in plot_points],
                mode="lines",
                name=name,
                line={"color": color, "width": 7},
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


def _vision_observation_figure(
    run: LiveRun,
    frame: Any,
    observation_record: dict[str, Any] | None,
    frame_index: int,
) -> go.Figure:
    fig = go.Figure()
    gates = _vision_observation_gates(observation_record)
    points: list[list[float]] = [[0.0, 0.0, 0.0]]
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            name="Camera",
            text=["camera"],
            textposition="bottom center",
            marker={"size": 7, "color": "#dc2626"},
            showlegend=False,
        )
    )
    max_forward_m = _max_observation_forward_m(gates)
    for trace, trace_points in _camera_frustum_traces(max_forward_m):
        points.extend(trace_points)
        fig.add_trace(trace)
    sync = nearest_cycle_for_frame(run, frame)
    telemetry = sync.cycle.get("telemetry") if isinstance(sync.cycle, dict) and isinstance(sync.cycle.get("telemetry"), dict) else {}
    attitude = _point4_value(telemetry.get("attitude_quaternion") or telemetry.get("attitude"))
    drone_position = _point3_value(telemetry.get("position_local_ned_m"))
    for gate in gates:
        camera_position = _observation_camera_position(gate, attitude, drone_position)
        if camera_position is None:
            continue
        plot_position = _camera_observation_point_to_plot(camera_position)
        points.append(plot_position)
        gate_id = str(gate.get("id", "gate"))
        confidence = _round(gate.get("position_confidence"))
        normal = _observation_camera_normal(gate, attitude)
        if normal is not None:
            for trace, trace_points in _camera_gate_wireframe_traces(plot_position, normal, gate_id):
                points.extend(trace_points)
                fig.add_trace(trace)
        fig.add_trace(
            go.Scatter3d(
                x=[plot_position[0]],
                y=[plot_position[1]],
                z=[plot_position[2]],
                mode="markers+text",
                name=gate_id,
                text=[gate_id],
                textposition="top center",
                marker={"size": 6, "color": "#7c3aed"},
                customdata=[[confidence]],
                hovertemplate=(
                    "gate=%{text}<br>"
                    "forward=%{x:.3f} m<br>"
                    "left=%{y:.3f} m<br>"
                    "up=%{z:.3f} m<br>"
                    "confidence=%{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    title = f"Vision Observation Gates At Frame {frame_index + 1} (id={frame.frame_id})"
    axis_ranges = _trajectory_axis_ranges(points)
    ui_revision = f"live-vision-observation:{run.path}:{frame_index}"
    fig.update_layout(
        **_layout(title),
        dragmode="orbit",
        uirevision=ui_revision,
        scene={
            "xaxis": {"title": "Forward (m)", "range": axis_ranges[0]},
            "yaxis": {"title": "Left (-Right) (m)", "range": axis_ranges[1]},
            "zaxis": {"title": "Up (m)", "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {
                "eye": {"x": -0.001, "y": 0.0, "z": 0.0},
                "center": {"x": 1.0, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "projection": {"type": "perspective"},
            },
            "dragmode": "orbit",
            "uirevision": ui_revision,
        },
        annotations=[
            {
                "text": f"gates={len(gates)} | source={observation_record.get('source', 'n/a') if isinstance(observation_record, dict) else 'n/a'}",
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


def _vision_observation_gates(observation_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    observation = observation_record.get("observation") if isinstance(observation_record, dict) else None
    gates = observation.get("gates") if isinstance(observation, dict) else None
    return [_vision_observation_gate_payload(gate) for gate in gates if isinstance(gate, dict)] if isinstance(gates, list) else []


def _vision_observation_gate_payload(gate: dict[str, Any]) -> dict[str, Any]:
    position_confidence = gate.get("position_confidence") if gate.get("position_confidence") is not None else gate.get("confidence")
    position_local_ned = _point3_value(gate.get("position_local_ned") or gate.get("position_local_ned_m"))
    return {
        **gate,
        "id": gate.get("id") or gate.get("gate_id"),
        "position_xyz": _point3_value(gate.get("position_xyz")),
        "position_local_ned": position_local_ned,
        "position_local_ned_m": position_local_ned,
        "position_relative_ned_m": _point3_value(gate.get("position_relative_ned_m")),
        "position_confidence": position_confidence,
        "orientation_xyz": _point3_value(gate.get("orientation_xyz")),
        "orientation_local_ned_quat": _point4_value(
            gate.get("orientation_local_ned_quat")
            or gate.get("orientation_quat")
            or gate.get("quaternion")
            or gate.get("quat")
        ),
    }


def _observation_camera_position(
    gate: dict[str, Any],
    attitude_quat: list[float] | None,
    drone_position_local_ned_m: list[float] | None,
) -> list[float] | None:
    position_xyz = _point3_value(gate.get("position_xyz"))
    if position_xyz:
        return position_xyz
    local_ned = _point3_value(gate.get("position_local_ned_m") or gate.get("position_local_ned"))
    drone_position = _point3_value(drone_position_local_ned_m)
    relative_ned = (
        (np.asarray(local_ned, dtype=float) - np.asarray(drone_position, dtype=float)).tolist()
        if local_ned is not None and drone_position is not None
        else _point3_value(gate.get("position_relative_ned_m"))
    )
    if relative_ned is None or attitude_quat is None:
        return None
    body_relative = _quat_to_matrix(attitude_quat).T @ np.asarray(relative_ned, dtype=float)
    return _body_frd_to_camera_optical(body_relative).tolist()


def _observation_camera_normal(gate: dict[str, Any], attitude_quat: list[float] | None) -> np.ndarray | None:
    orientation_xyz = _point3_value(gate.get("orientation_xyz"))
    if orientation_xyz:
        return _camera_observation_vector_to_plot(orientation_xyz)
    orientation_quat = _point4_value(gate.get("orientation_local_ned_quat"))
    if orientation_quat is None or attitude_quat is None:
        return None
    local_normal = _quat_to_matrix(orientation_quat)[:, 0]
    body_normal = _quat_to_matrix(attitude_quat).T @ local_normal
    camera_normal = _body_frd_to_camera_optical(body_normal)
    return _camera_observation_vector_to_plot(camera_normal.tolist())


def _camera_observation_point_to_plot(point: list[float]) -> list[float]:
    right, up, forward = point[:3]
    return [float(forward), -float(right), float(up)]


def _camera_observation_vector_to_plot(vector: list[float]) -> np.ndarray:
    right, up, forward = vector[:3]
    return _normalize_vector(np.asarray([float(forward), -float(right), float(up)], dtype=float))


def _point3_value(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _point4_value(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _body_frd_to_camera_optical(vector: np.ndarray) -> np.ndarray:
    return _camera_optical_to_body_frd_matrix().T @ np.asarray(vector, dtype=float)


def _camera_optical_to_body_frd_matrix() -> np.ndarray:
    tilt = np.deg2rad(20.0)
    cos_t = np.cos(tilt)
    sin_t = np.sin(tilt)
    optical_to_body = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    pitch_up = np.asarray(
        [
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t],
        ],
        dtype=float,
    )
    return pitch_up @ optical_to_body


def _quat_to_matrix(quaternion: list[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _camera_gate_wireframe_traces(
    center: list[float],
    normal: np.ndarray,
    gate_id: str,
    *,
    color: str = "#7c3aed",
) -> list[tuple[go.Scatter3d, list[list[float]]]]:
    center_array = np.asarray(center, dtype=float)
    normal = _normalize_vector(normal)
    if np.linalg.norm(normal) < 1e-9:
        normal = np.asarray([1.0, 0.0, 0.0], dtype=float)
    horizontal, vertical = _gate_plane_axes(normal)
    depth_offsets = (-GATE_DEPTH_M / 2.0, GATE_DEPTH_M / 2.0)
    traces: list[tuple[go.Scatter3d, list[list[float]]]] = []

    for depth in depth_offsets:
        face_center = center_array + normal * depth
        traces.append(_line_trace(_square_points(face_center, horizontal, vertical, GATE_OUTER_M), f"{gate_id} outer", color, width=5))
        traces.append(_line_trace(_square_points(face_center, horizontal, vertical, GATE_INNER_M), f"{gate_id} inner", color, width=3, opacity=0.68))

    for size in (GATE_OUTER_M, GATE_INNER_M):
        half = size / 2.0
        for h, v in ((-half, -half), (half, -half), (half, half), (-half, half)):
            edge = np.vstack(
                [
                    center_array + normal * depth_offsets[0] + horizontal * h + vertical * v,
                    center_array + normal * depth_offsets[1] + horizontal * h + vertical * v,
                ]
            )
            traces.append(_line_trace(edge, f"{gate_id} depth", color, width=2, opacity=0.5))

    normal_line = np.vstack([center_array, center_array + normal * GATE_NORMAL_INDICATOR_M])
    traces.append(_line_trace(normal_line, f"{gate_id} normal", "#f59e0b", width=6))
    up_line = np.vstack([center_array, center_array + vertical * (GATE_INNER_M / 2.0)])
    traces.append(_line_trace(up_line, f"{gate_id} up", "#16a34a", width=5))
    return traces


def _camera_frustum_traces(max_forward_m: float) -> list[tuple[go.Scatter3d, list[list[float]]]]:
    forward_m = max(3.0, float(max_forward_m))
    half_up = np.tan(np.deg2rad(CAMERA_VERTICAL_FOV_DEG) / 2.0) * forward_m
    half_left = half_up * CAMERA_ASPECT_RATIO
    origin = np.asarray([0.0, 0.0, 0.0], dtype=float)
    corners = np.asarray(
        [
            [forward_m, -half_left, -half_up],
            [forward_m, half_left, -half_up],
            [forward_m, half_left, half_up],
            [forward_m, -half_left, half_up],
            [forward_m, -half_left, -half_up],
        ],
        dtype=float,
    )
    traces = [_line_trace(corners, "camera frustum", "#64748b", width=2, opacity=0.5)]
    for corner in corners[:4]:
        traces.append(_line_trace(np.vstack([origin, corner]), "camera frustum edge", "#64748b", width=2, opacity=0.35))
    return traces


def _max_observation_forward_m(gates: list[dict[str, Any]]) -> float:
    forward_values = [
        float(position[2])
        for gate in gates
        if _point3(position := gate.get("position_xyz"))
    ]
    return max(forward_values, default=3.0)


def _gate_plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up_reference = np.asarray([0.0, 0.0, 1.0], dtype=float)
    vertical = up_reference - normal * float(np.dot(up_reference, normal))
    if np.linalg.norm(vertical) < 1e-6:
        up_reference = np.asarray([0.0, 1.0, 0.0], dtype=float)
        vertical = up_reference - normal * float(np.dot(up_reference, normal))
    vertical = _normalize_vector(vertical)
    horizontal = _normalize_vector(np.cross(vertical, normal))
    return horizontal, vertical


def _square_points(center: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, size: float) -> np.ndarray:
    half = float(size) / 2.0
    offsets = ((-half, -half), (half, -half), (half, half), (-half, half), (-half, -half))
    return np.asarray([center + horizontal * h + vertical * v for h, v in offsets], dtype=float)


def _line_trace(
    points: np.ndarray,
    name: str,
    color: str,
    *,
    width: int,
    opacity: float = 1.0,
) -> tuple[go.Scatter3d, list[list[float]]]:
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        name=name,
        line={"color": color, "width": width},
        opacity=opacity,
        showlegend=False,
    )
    return trace, points.tolist()


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=float)
    return np.asarray(vector, dtype=float) / norm


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
    return _path_for_cycle(run, "planned_paths", "planned_path", cycle_number)


def _test_path_for_cycle(run: LiveRun, cycle_number: int | None) -> dict[str, Any] | None:
    return _path_for_cycle(run, "test_paths", "test_path", cycle_number)


def _path_for_cycle(run: LiveRun, collection_key: str, path_key: str, cycle_number: int | None) -> dict[str, Any] | None:
    if cycle_number is None:
        return None
    path_records = run.raw.get(collection_key)
    if not isinstance(path_records, list):
        return None
    selected: dict[str, Any] | None = None
    for record in path_records:
        if not isinstance(record, dict):
            continue
        record_cycle = record.get("cycle")
        if not isinstance(record_cycle, int) or record_cycle > cycle_number:
            continue
        planned_path = record.get(path_key)
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
            "vision_observations": len(run.vision_observations),
            "planned_paths": len(raw.get("planned_paths", [])) if isinstance(raw.get("planned_paths"), list) else 0,
            "test_paths": len(raw.get("test_paths", [])) if isinstance(raw.get("test_paths"), list) else 0,
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
