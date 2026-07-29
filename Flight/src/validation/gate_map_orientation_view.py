"""3D view of gate map camera/body/local transform results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.coordinates import rotation_matrix_from_quaternion
from core.schema import VehicleState
from mapping.gates import GateMap
from core.schema import VisionGateObservation, VisionObservation


def vehicle_state_at_origin() -> VehicleState:
    return VehicleState(
        sim_time_ns=0,
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        body_rates_frd_rps=(0.0, 0.0, 0.0),
        acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
    )


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
    vehicle_state: VehicleState,
    *,
    camera_tilt_deg: float = 20.0,
) -> None:
    gate_map = GateMap(camera_tilt_deg=camera_tilt_deg)
    if not observation.gates:
        raise ValueError("VisionObservation must include at least one gate.")
    gate_map.update_from_observation(observation, vehicle_state)
    gate_records = gate_map.gates

    origin = np.zeros(3, dtype=float)
    drone_position = np.asarray(vehicle_state.position_local_ned_m, dtype=float)

    body_to_local = rotation_matrix_from_quaternion(vehicle_state.attitude_quaternion)
    camera_rotation = gate_map.camera_frd_to_body_frd_rotation()
    body_forward_local = body_to_local @ np.array([1.0, 0.0, 0.0], dtype=float)
    body_right_local = body_to_local @ np.array([0.0, 1.0, 0.0], dtype=float)
    body_down_local = body_to_local @ np.array([0.0, 0.0, 1.0], dtype=float)
    camera_forward_local = body_to_local @ camera_rotation @ np.array([1.0, 0.0, 0.0], dtype=float)
    camera_right_local = body_to_local @ camera_rotation @ np.array([0.0, 1.0, 0.0], dtype=float)
    camera_down_local = body_to_local @ camera_rotation @ np.array([0.0, 0.0, 1.0], dtype=float)
    camera_up_local = -camera_down_local

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Gate Map Transform View")
    ax.set_xlabel("North / body forward +")
    ax.set_ylabel("East / body right +")
    ax.set_zlabel("Down +")

    draw_arrow(ax, origin, (1.0, 0.0, 0.0), "tab:blue", "N", 2.5)
    draw_arrow(ax, origin, (0.0, 1.0, 0.0), "tab:orange", "E", 2.5)
    draw_arrow(ax, origin, (0.0, 0.0, 1.0), "tab:green", "D", 2.5)
    ax.scatter([drone_position[0]], [drone_position[1]], [drone_position[2]], color="tab:cyan", s=40)
    ax.text(drone_position[0], drone_position[1], drone_position[2], "drone", color="tab:cyan")
    draw_arrow(ax, drone_position, body_forward_local, "tab:cyan", "body F", 2.0)
    draw_arrow(ax, drone_position, body_right_local, "tab:pink", "body R", 2.0)
    draw_arrow(ax, drone_position, body_down_local, "tab:olive", "body D", 2.0)
    draw_arrow(ax, drone_position, camera_forward_local, "tab:purple", "camera forward", 2.0)
    draw_arrow(ax, drone_position, camera_right_local, "tab:red", "camera right", 2.0)
    draw_arrow(ax, drone_position, camera_up_local, "tab:brown", "camera up", 2.0)

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
            drone_position,
            drone_position + body_forward_local * 2.0,
            drone_position + body_right_local * 2.0,
            drone_position + body_down_local * 2.0,
            drone_position + camera_forward_local * 2.0,
            drone_position + camera_right_local * 2.0,
            drone_position + camera_up_local * 2.0,
            *gate_axis_points,
            np.array([3.0, 3.0, 3.0], dtype=float),
            np.array([-1.0, -1.0, -1.0], dtype=float),
        ],
    )
    ax.view_init(elev=24, azim=-58)
    ax.legend(
        [
            "local/body axes",
            "camera axes",
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
    vehicle_state = vehicle_state_at_origin()
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
    show_gate_map_orientation_view(observation, vehicle_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
