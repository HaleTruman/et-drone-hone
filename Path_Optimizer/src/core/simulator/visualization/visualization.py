from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from core.quadrotor.model import Quadrotor
from core.simulator.core.simulator import SimulationResult, load_quadrotor_params, run_configured_simulation
from core.simulator.utils.utils import body_axes_from_quaternion

DRONE_TRACE_COUNT = 7


def create_app(config_path: str | Path | None = None) -> Dash:
    result = run_configured_simulation(config_path)
    quadrotor = Quadrotor(load_quadrotor_params())
    tracking_error = np.linalg.norm(result.position - result.target_position, axis=1)
    title = "Drone Track Simulator" if result.gates else "Drone Hover Simulator"
    planner_info = result.planner_info or {}
    planner_time = planner_info.get("gate_crossing_time_s", planner_info.get("horizon_s"))

    app = Dash(__name__)
    app.title = title
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(title),
                    html.Div(
                        [
                            _metric("Duration", f"{result.t[-1]:.1f} s"),
                            _metric("Gate Time", "n/a" if planner_time is None else f"{planner_time:.2f} s"),
                            _metric("Final Speed", f"{result.speed[-1]:.3f} m/s"),
                            _metric("Max Error", f"{np.max(tracking_error):.2f} m"),
                            _metric("Max Tilt", f"{np.rad2deg(np.max(np.abs(result.euler[:, :2]))):.2f} deg"),
                            _metric("Avg Thrust", f"{np.mean(result.thrust):.2f} N"),
                        ],
                        className="metrics",
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="sim-3d",
                        figure=build_scene_figure(result, quadrotor),
                        config={"displayModeBar": True},
                    ),
                ],
                className="grid full",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="kinematics",
                        figure=build_kinematics_figure(result),
                        config={"displayModeBar": True},
                    ),
                ],
                className="grid two",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="attitude",
                        figure=build_attitude_figure(result),
                        config={"displayModeBar": True},
                    ),
                    dcc.Graph(
                        id="controls",
                        figure=build_controls_figure(result),
                        config={"displayModeBar": True},
                    ),
                ],
                className="grid two",
            ),
        ],
        className="page",
    )

    app.index_string = _index_template()
    return app


def _metric(label: str, value: str) -> html.Div:
    return html.Div([html.Span(label), html.Strong(value)], className="metric")


def build_scene_figure(result: SimulationResult, quadrotor: Quadrotor) -> go.Figure:
    pos = result.position
    target = result.target_position
    stride = max(1, len(result.t) // 120)
    frame_indices = list(range(0, len(result.t), stride))
    if frame_indices[-1] != len(result.t) - 1:
        frame_indices.append(len(result.t) - 1)
    frame_duration_ms = _real_time_frame_duration_ms(result.t, frame_indices)

    fig = go.Figure()
    static_trace_count = 0
    if result.reference_path is not None:
        ref = result.reference_path
        fig.add_trace(
            go.Scatter3d(
                x=ref[:, 0],
                y=ref[:, 1],
                z=ref[:, 2],
                mode="lines",
                line=dict(color="#7b61ff", width=4, dash="dot"),
                name="MPCC reference",
            )
        )
        static_trace_count += 1
    if result.planned_path is not None:
        planned = result.planned_path
        fig.add_trace(
            go.Scatter3d(
                x=planned[:, 0],
                y=planned[:, 1],
                z=planned[:, 2],
                mode="lines",
                line=dict(color="#ff7f0e", width=5),
                name="MPCC plan",
            )
        )
        static_trace_count += 1
    for gate_i, wire in enumerate(result.gate_wireframes or []):
        fig.add_trace(
            go.Scatter3d(
                x=wire[:, 0],
                y=wire[:, 1],
                z=wire[:, 2],
                mode="lines",
                line=dict(color="#111111", width=4),
                name="gates" if gate_i == 0 else None,
                showlegend=gate_i == 0,
            )
        )
        static_trace_count += 1
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            line=dict(color="#1f77b4", width=5),
            name="flight path",
        )
    )
    static_trace_count += 1
    fig.add_trace(
        go.Scatter3d(
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            mode="lines",
            line=dict(color="#2ca02c", width=3, dash="dash"),
            name="active target",
        )
    )
    static_trace_count += 1
    for trace in _drone_traces(result, quadrotor, 0):
        fig.add_trace(trace)

    fig.frames = [
        go.Frame(
            data=_drone_traces(result, quadrotor, idx),
            traces=list(range(static_trace_count, static_trace_count + DRONE_TRACE_COUNT)),
            name=f"{idx}",
        )
        for idx in frame_indices
    ]

    fig.update_layout(
        template="plotly_white",
        height=760,
        margin=dict(l=0, r=0, t=40, b=0),
        title="3D Track, Planned Path, and Drone Attitude",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.98,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue={"prefix": "step "},
                pad={"t": 32},
                steps=[
                    dict(
                        method="animate",
                        args=[[f"{idx}"], {"mode": "immediate", "frame": {"duration": 0}}],
                        label=f"{result.t[idx]:.1f}s",
                    )
                    for idx in frame_indices[:: max(1, len(frame_indices) // 12)]
                ],
            )
        ],
    )
    return fig


def _real_time_frame_duration_ms(t: np.ndarray, frame_indices: list[int]) -> int:
    if len(frame_indices) < 2:
        return 1
    frame_times = np.asarray(t, dtype=float)[frame_indices]
    dt = float(np.median(np.diff(frame_times)))
    return max(1, int(round(dt * 1000.0)))


def _drone_traces(result: SimulationResult, quadrotor: Quadrotor, idx: int) -> list[go.Scatter3d | go.Mesh3d]:
    center = result.position[idx]
    quaternion = result.quaternion[idx] if result.display_quaternion is None else result.display_quaternion[idx]
    rotation = body_axes_from_quaternion(quadrotor, quaternion)
    arm = float(np.mean(quadrotor.l))
    x_axis = _display_vector(rotation @ np.array([arm, arm, 0.0]))
    y_axis = _display_vector(rotation @ np.array([-arm, arm, 0.0]))
    nose_axis = _display_vector(rotation @ np.array([arm * 1.65, 0.0, 0.0]))
    thrust_axis = _display_vector(rotation @ np.array([0.0, 0.0, -0.18]))

    rotor_points = np.vstack(
        [
            center + x_axis,
            center + y_axis,
            center - x_axis,
            center - y_axis,
        ]
    )
    body_plane = np.vstack(
        [
            center + _display_vector(rotation @ np.array([arm * 1.15, 0.0, 0.0])),
            center + _display_vector(rotation @ np.array([0.0, arm * 1.15, 0.0])),
            center + _display_vector(rotation @ np.array([-arm * 1.15, 0.0, 0.0])),
            center + _display_vector(rotation @ np.array([0.0, -arm * 1.15, 0.0])),
        ]
    )
    motor = result.motor_commands[idx]
    motor_colors = [f"rgba({int(70 + 170 * m)}, {int(190 - 80 * m)}, 70, 0.95)" for m in motor]
    motor_sizes = 11 + 12 * motor

    return [
        go.Mesh3d(
            x=body_plane[:, 0],
            y=body_plane[:, 1],
            z=body_plane[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#1f77b4",
            opacity=0.22,
            name="body plane",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[center[0] - x_axis[0], center[0] + x_axis[0]],
            y=[center[1] - x_axis[1], center[1] + x_axis[1]],
            z=[center[2] - x_axis[2], center[2] + x_axis[2]],
            mode="lines",
            line=dict(color="#222222", width=8),
            name="body arm A",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[center[0] - y_axis[0], center[0] + y_axis[0]],
            y=[center[1] - y_axis[1], center[1] + y_axis[1]],
            z=[center[2] - y_axis[2], center[2] + y_axis[2]],
            mode="lines",
            line=dict(color="#555555", width=8),
            name="body arm B",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[center[0], center[0] + nose_axis[0]],
            y=[center[1], center[1] + nose_axis[1]],
            z=[center[2], center[2] + nose_axis[2]],
            mode="lines",
            line=dict(color="#d62728", width=7),
            name="nose",
            showlegend=False,
        ),
        go.Scatter3d(
            x=rotor_points[:, 0],
            y=rotor_points[:, 1],
            z=rotor_points[:, 2],
            mode="markers",
            marker=dict(size=motor_sizes, color=motor_colors, line=dict(color="#111111", width=2)),
            name="motors",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[center[0], center[0] + thrust_axis[0]],
            y=[center[1], center[1] + thrust_axis[1]],
            z=[center[2], center[2] + thrust_axis[2]],
            mode="lines",
            line=dict(color="#d62728", width=5),
            name="thrust axis",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers",
            marker=dict(size=6, color="#111111"),
            name="center of mass",
            showlegend=False,
        ),
    ]


def _display_vector(vector: np.ndarray) -> np.ndarray:
    """Render NED-style model vectors in the plot's z-up view."""
    out = np.asarray(vector, dtype=float).copy()
    out[2] *= -1.0
    return out


def build_kinematics_figure(result: SimulationResult) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Position", "Velocity and Speed", "Acceleration"),
        vertical_spacing=0.08,
    )
    labels = ("x", "y", "z")
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(x=result.t, y=result.position[:, i], name=f"pos {label}"), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=result.t,
                y=result.target_position[:, i],
                name=f"target {label}",
                line=dict(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(go.Scatter(x=result.t, y=result.velocity[:, i], name=f"vel {label}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.t, y=result.acceleration[:, i], name=f"accel {label}"), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=result.t, y=result.speed, name="speed", line=dict(color="#111111", width=3)),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="m/s", row=2, col=1)
    fig.update_yaxes(title_text="m/s^2", row=3, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_layout(template="plotly_white", height=670, margin=dict(l=60, r=20, t=50, b=50))
    return fig


def build_attitude_figure(result: SimulationResult) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Attitude", "Body Rates"),
        vertical_spacing=0.12,
    )
    labels = ("roll", "pitch", "yaw")
    for i, label in enumerate(labels):
        fig.add_trace(
            go.Scatter(x=result.t, y=np.rad2deg(result.euler[:, i]), name=label),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.t,
                y=np.rad2deg(result.desired_euler[:, i]),
                name=f"desired {label}",
                line=dict(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=result.t, y=np.rad2deg(result.rates[:, i]), name=f"{label} rate"),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="deg", row=1, col=1)
    fig.update_yaxes(title_text="deg/s", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=60, r=20, t=50, b=50))
    return fig


def build_controls_figure(result: SimulationResult) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Motor Commands", "Total Thrust"),
        vertical_spacing=0.12,
    )
    for i in range(4):
        fig.add_trace(
            go.Scatter(x=result.t, y=result.motor_commands[:, i], name=f"motor {i + 1}"),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=result.t, y=result.thrust, name="total thrust", line=dict(color="#111111", width=3)),
        row=2,
        col=1,
    )
    if result.planned_controls is not None and result.phase is not None:
        track_indices = np.flatnonzero(result.phase == "track")
        if len(track_indices):
            track_t = result.t[track_indices]
            plan_u = result.planned_controls
            plan_t = np.linspace(track_t[0], track_t[-1], plan_u.shape[1])
            plan_collective = np.mean(plan_u, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=plan_t,
                    y=plan_collective,
                    name="MPCC collective",
                    line=dict(color="#ff7f0e", dash="dot", width=3),
                ),
                row=1,
                col=1,
            )
    fig.update_yaxes(title_text="command", range=[0.0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="N", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=60, r=20, t=50, b=50))
    return fig


def _index_template() -> str:
    return """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body { margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }
      .page { max-width: 1600px; margin: 0 auto; padding: 24px; }
      .header { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 18px; }
      h1 { font-size: 30px; line-height: 1.1; margin: 0; font-weight: 760; }
      .metrics { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; min-width: min(960px, 100%); }
      .metric { background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 10px 12px; }
      .metric span { display: block; color: #667085; font-size: 12px; }
      .metric strong { display: block; margin-top: 4px; font-size: 18px; }
      .grid { display: grid; gap: 16px; margin-bottom: 16px; }
      .grid.full { grid-template-columns: 1fr; }
      .grid.two { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
      .dash-graph { background: white; border: 1px solid #d9dee5; border-radius: 8px; overflow: hidden; }
      @media (max-width: 1100px) {
        .header { display: block; }
        .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 14px; }
        .grid.two { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""


def run() -> None:
    app = create_app()
    app.run(debug=False, port=8051)


if __name__ == "__main__":
    run()
