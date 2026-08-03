"""3D view of the gate records held by GateMap."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.coordinates import rotation_matrix_from_quaternion
from mapping.gates import GateMap
from core.schema import VisionGateObservation, VisionObservation


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return vector / norm


def draw_arrow(ax, origin, vector, color, label, length_scale: float = 1.0) -> None:
    vector = np.asarray(vector, dtype=float) * float(length_scale)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        arrow_length_ratio=0.12,
        linewidth=2.0,
    )
    end = np.asarray(origin, dtype=float) + vector
    ax.text(end[0], end[1], end[2], label, color=color)


def draw_gate_plane(
    ax,
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    width_m: float,
    height_m: float,
    *,
    color: str = "black",
) -> None:
    right = unit(right)
    up = unit(up)
    corners = np.array(
        [
            center - right * width_m / 2.0 - up * height_m / 2.0,
            center + right * width_m / 2.0 - up * height_m / 2.0,
            center + right * width_m / 2.0 + up * height_m / 2.0,
            center - right * width_m / 2.0 + up * height_m / 2.0,
            center - right * width_m / 2.0 - up * height_m / 2.0,
        ],
        dtype=float,
    )
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=2.0)
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=28)


def set_equal_axes(ax, points: list[np.ndarray]) -> None:
    stacked = np.vstack(points)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] + radius, center[1] - radius)
    ax.set_zlim(center[2] + radius, center[2] - radius)


def show_gate_map_orientation_view(
    observation: VisionObservation,
) -> None:
    gate_map = GateMap()
    if not observation.gates:
        raise ValueError("VisionObservation must include at least one gate.")
    gate_map.update(observation)
    gate_records = gate_map.gates

    origin = np.zeros(3, dtype=float)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Gate Map View")
    ax.set_xlabel("North +")
    ax.set_ylabel("East +")
    ax.set_zlabel("Down +")

    draw_arrow(ax, origin, (1.0, 0.0, 0.0), "tab:blue", "N", 2.5)
    draw_arrow(ax, origin, (0.0, 1.0, 0.0), "tab:orange", "E", 2.5)
    draw_arrow(ax, origin, (0.0, 0.0, 1.0), "tab:green", "D", 2.5)

    gate_axis_points: list[np.ndarray] = []
    gate_colors = ["black", "tab:gray", "tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"]
    for index, record in enumerate(gate_records):
        color = gate_colors[index % len(gate_colors)]
        gate_center = np.asarray(record.position_local_ned_m, dtype=float)
        gate_to_local = rotation_matrix_from_quaternion(record.quaternion)
        gate_forward_local = unit(gate_to_local[:, 0])
        gate_right_local = unit(gate_to_local[:, 1])
        gate_down_local = unit(gate_to_local[:, 2])
        gate_up_local = -gate_down_local

        ax.text(gate_center[0], gate_center[1], gate_center[2], record.gate_id, color=color)
        draw_arrow(ax, gate_center, gate_right_local, color, f"{record.gate_id} right", 1.6)
        draw_arrow(ax, gate_center, gate_up_local, color, f"{record.gate_id} up", 1.6)
        draw_arrow(ax, gate_center, gate_forward_local, color, f"{record.gate_id} forward", 2.0)
        draw_gate_plane(
            ax,
            gate_center,
            gate_right_local,
            gate_up_local,
            width_m=record.outer_width_m,
            height_m=record.outer_height_m,
            color=color,
        )
        gate_axis_points.extend(
            [
                gate_center,
                gate_center + gate_forward_local * 2.0,
                gate_center + gate_right_local * 1.6,
                gate_center + gate_up_local * 1.6,
            ]
        )

    set_equal_axes(
        ax,
        [
            origin,
            *gate_axis_points,
            np.array([3.0, 3.0, 3.0], dtype=float),
            np.array([-1.0, -1.0, -1.0], dtype=float),
        ],
    )
    ax.view_init(elev=24, azim=-58)
    ax.legend(
        [
            "gate orientation",
            "gate plane",
        ],
        loc="upper right",
    )
    fig.tight_layout()

    print("Opening gate orientation view. Close the plot window to finish the validation script.")
    for record in gate_records:
        gate_to_local = rotation_matrix_from_quaternion(record.quaternion)
        gate_forward_local = unit(gate_to_local[:, 0])
        gate_right_local = unit(gate_to_local[:, 1])
        gate_up_local = -unit(gate_to_local[:, 2])
        print(f"{record.gate_id}.gate_right_local_ned = {tuple(float(value) for value in gate_right_local)}")
        print(f"{record.gate_id}.gate_up_local_ned = {tuple(float(value) for value in gate_up_local)}")
        print(f"{record.gate_id}.gate_forward_local_ned = {tuple(float(value) for value in gate_forward_local)}")
    plt.show()


def main() -> int:
    observation = VisionObservation(
        frame_id=1,
        sim_time_ns=0,
        gates=[
            VisionGateObservation(
                gate_id="view_gate",
                position_local_ned=(10.0, 2.0, 0.0),
                position_confidence=1.0,
                orientation_local_ned_quat=(1.0, 0.0, 0.0, 0.0),
                orientation_confidence=1.0,
            )
        ],
    )
    show_gate_map_orientation_view(observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
