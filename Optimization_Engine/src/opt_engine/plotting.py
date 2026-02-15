from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def make_3d_figure(
    *,
    waypoints_m: np.ndarray,
    path_points_m: np.ndarray,
    speed_mps: np.ndarray | None = None,
    kappa_violation_mask: np.ndarray | None = None,
) -> go.Figure:
    wps = np.asarray(waypoints_m, dtype=float)
    pts = np.asarray(path_points_m, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=wps[:, 0],
            y=wps[:, 1],
            z=wps[:, 2],
            mode="markers+text",
            text=[str(i) for i in range(wps.shape[0])],
            textposition="top center",
            marker={"size": 5, "color": "red"},
            name="waypoints",
        )
    )

    line: dict[str, object] = {"width": 4, "color": "rgba(40, 120, 220, 0.9)"}
    if speed_mps is not None and len(speed_mps) == pts.shape[0]:
        line = {
            "width": 5,
            "color": np.asarray(speed_mps, dtype=float),
            "colorscale": "Viridis",
            "colorbar": {"title": "m/s"},
        }

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=line,
            name="path",
        )
    )

    if kappa_violation_mask is not None:
        mask = np.asarray(kappa_violation_mask, dtype=bool)
        if mask.shape == (pts.shape[0],) and np.any(mask):
            bad = pts[mask]
            fig.add_trace(
                go.Scatter3d(
                    x=bad[:, 0],
                    y=bad[:, 1],
                    z=bad[:, 2],
                    mode="markers",
                    marker={"size": 4, "color": "orange"},
                    name="κ limit violated",
                )
            )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={"aspectmode": "data"},
        legend={"orientation": "h"},
    )
    return fig


def make_profile_figure(
    *,
    s_m: np.ndarray,
    v_mps: np.ndarray,
    v_cap_mps: np.ndarray | None = None,
    v_kappa_mps: np.ndarray | None = None,
    v_yaw_mps: np.ndarray | None = None,
) -> go.Figure:
    s = np.asarray(s_m, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=v, mode="lines", name="v(s)", line={"width": 3}))
    if v_cap_mps is not None:
        fig.add_trace(go.Scatter(x=s, y=np.asarray(v_cap_mps, dtype=float), mode="lines", name="v_cap(s)"))
    if v_kappa_mps is not None:
        fig.add_trace(
            go.Scatter(
                x=s,
                y=np.asarray(v_kappa_mps, dtype=float),
                mode="lines",
                name="v_kappa(s)",
                line={"dash": "dot"},
            )
        )
    if v_yaw_mps is not None:
        fig.add_trace(
            go.Scatter(
                x=s,
                y=np.asarray(v_yaw_mps, dtype=float),
                mode="lines",
                name="v_yaw(s)",
                line={"dash": "dash"},
            )
        )
    fig.update_layout(margin={"l": 40, "r": 10, "t": 20, "b": 40}, xaxis_title="s (m)", yaxis_title="v (m/s)")
    return fig


def make_curvature_figure(*, s_m: np.ndarray, kappa_1pm: np.ndarray, kappa_max_1pm: float | None = None) -> go.Figure:
    s = np.asarray(s_m, dtype=float)
    k = np.asarray(kappa_1pm, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=k, mode="lines", name="kappa(s)"))
    if kappa_max_1pm is not None:
        km = float(kappa_max_1pm)
        fig.add_trace(
            go.Scatter(
                x=[float(s[0]), float(s[-1])] if s.shape[0] else [0.0, 1.0],
                y=[km, km],
                mode="lines",
                name="kappa_max",
                line={"dash": "dash", "color": "orange"},
            )
        )
    fig.update_layout(
        margin={"l": 40, "r": 10, "t": 20, "b": 40}, xaxis_title="s (m)", yaxis_title="κ (1/m)"
    )
    return fig
