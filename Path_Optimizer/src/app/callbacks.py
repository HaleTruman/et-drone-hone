from __future__ import annotations

import os

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, no_update

from app.data import Course, ReferenceTrajectorySnapshot, load_course, load_reference_trajectory


REFERENCE_TRAJECTORY_PATH = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + "/artifacts/reference_trajectory.json"
)


def register_callbacks(app: Dash) -> None:
    @callback(
        Output("scene-summary", "children"),
        Output("course-3d", "figure"),
        Output("scene-signature", "data"),
        Input("scenario_path", "value"),
        Input("planning-session-poll", "n_intervals"),
        State("scene-signature", "data"),
    )
    def _update_course_view(scenario_path: str | None, _n_intervals: int, current_signature: str | None):
        empty = go.Figure()
        if not scenario_path:
            return "Select a course file to visualize.", empty, None

        if not os.path.isfile(scenario_path):
            return f"Course file not found: {scenario_path}", empty, None

        try:
            course = load_course(scenario_path)
        except Exception as exc:  # noqa: BLE001
            return f"Unable to load course: {exc}", empty, None

        snapshot = _load_matching_reference_trajectory(scenario_path)
        next_signature = _make_scene_signature(scenario_path, snapshot)
        if next_signature == current_signature:
            return no_update, no_update, current_signature

        return _build_summary(course, snapshot), _make_3d_figure(course, snapshot), next_signature


def _build_summary(course: Course, snapshot: ReferenceTrajectorySnapshot | None) -> str:
    bounds = course.bounds_m
    summary = (
        f"name: {course.name}\n"
        f"targets: {len(course.targets)}\n"
        f"path_length_m: {course.path_length_m:.2f}\n"
        f"width_m: {bounds['width']:.2f}\n"
        f"depth_m: {bounds['depth']:.2f}\n"
        f"height_m: {bounds['height']:.2f}\n"
        f"frame: {course.frame}\n"
        f"source_units: {course.units}\n"
        f"level: {course.level or 'n/a'}\n"
        f"generated_at: {course.generated_at or 'n/a'}"
    )
    if snapshot is None:
        return f"{summary}\nreference_trajectory: not loaded"
    return (
        f"{summary}\n"
        f"reference_trajectory: loaded\n"
        f"reference_points: {len(snapshot.trajectory.pos_m)}\n"
        f"drone_position_m: ({snapshot.drone_position.x_m:.2f}, "
        f"{snapshot.drone_position.y_m:.2f}, {snapshot.drone_position.z_m:.2f})"
    )


def _make_3d_figure(course: Course, snapshot: ReferenceTrajectorySnapshot | None) -> go.Figure:
    ui_revision = f"course-3d:{course.name}"
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=[point.x_m for point in course.targets],
            y=[point.y_m for point in course.targets],
            z=[point.z_m for point in course.targets],
            mode="markers+text",
            text=[point.label for point in course.targets],
            textposition="top center",
            marker={"size": 7, "color": "#1d4ed8"},
            name="Gates",
        )
    )

    if course.origin is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[course.origin.x_m],
                y=[course.origin.y_m],
                z=[course.origin.z_m],
                mode="markers+text",
                text=[course.origin.label],
                textposition="bottom center",
                marker={"size": 8, "color": "#16a34a", "symbol": "diamond"},
                name="Origin",
            )
        )

    if snapshot is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[point.x_m for point in snapshot.trajectory.pos_m],
                y=[point.y_m for point in snapshot.trajectory.pos_m],
                z=[point.z_m for point in snapshot.trajectory.pos_m],
                mode="lines",
                line={"color": "#dc2626", "width": 3, "dash": "dash"},
                name="Reference trajectory",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[snapshot.drone_position.x_m],
                y=[snapshot.drone_position.y_m],
                z=[snapshot.drone_position.z_m],
                mode="markers+text",
                text=[snapshot.drone_position.label],
                textposition="bottom center",
                marker={"size": 10, "color": "#e11d48", "symbol": "circle"},
                name="Drone",
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 72, "b": 56},
        title={"text": "3D Course View", "x": 0.02, "xanchor": "left", "y": 0.98},
        uirevision=ui_revision,
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
            "uirevision": ui_revision,
        },
        legend={"orientation": "h", "y": -0.12, "x": 0.0},
    )
    return fig


def _load_matching_reference_trajectory(course_path: str) -> ReferenceTrajectorySnapshot | None:
    if not os.path.isfile(REFERENCE_TRAJECTORY_PATH):
        return None
    try:
        snapshot = load_reference_trajectory(REFERENCE_TRAJECTORY_PATH)
    except Exception:  # noqa: BLE001
        return None
    if snapshot.course_path != os.path.abspath(course_path):
        return None
    return snapshot


def _make_scene_signature(course_path: str, snapshot: ReferenceTrajectorySnapshot | None) -> str:
    scenario_mtime = os.path.getmtime(course_path)
    reference_mtime = os.path.getmtime(REFERENCE_TRAJECTORY_PATH) if snapshot is not None else -1.0
    return f"{os.path.abspath(course_path)}:{scenario_mtime}:{reference_mtime}"
