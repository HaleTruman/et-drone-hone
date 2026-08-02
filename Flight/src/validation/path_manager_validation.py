"""Interactive PathManager validation with mock gates.

Edit ``MOCK_GATES`` below, or pass a JSON file with ``--gates-json``.

Example JSON:
[
  {"id": "gate-001", "position_local_ned_m": [10.0, 0.0, 0.0]},
  {"id": "gate-002", "position_local_ned_m": [20.0, 8.0, -1.0]}
]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy.pathing import PathManager
from core.initialization import initialization as init
from core.schema import VehicleState
from mapping.gates import GateRecord


MOCK_DRONE_POSITION_LOCAL_NED_M = (0.0, 0.0, 0.0)

MOCK_GATES: list[dict[str, Any]] = [
    {"id": "gate-001", "position_local_ned_m": (11.0, 0.0, 0.0)},
    {"id": "gate-002", "position_local_ned_m": (27.0, 8.0, -2.2)},
    {"id": "gate-003", "position_local_ned_m": (36.0, 9.0, -1.8)},
    {"id": "gate-004", "position_local_ned_m": (38.0, 1.0, -0.8)},
]


def main() -> int:
    args = _parse_args()
    gate_payloads = _load_gate_payloads(args.gates_json)
    gates = [_gate_record(payload, index) for index, payload in enumerate(gate_payloads)]
    vehicle_state = _vehicle_state(tuple(float(value) for value in args.drone_position))
    path_manager = _path_manager(args.mode)

    planned_path = path_manager.plan(gates=gates, vehicle_state=vehicle_state)
    _print_summary(planned_path, gates)
    _show_path(planned_path, gates, vehicle_state.position_local_ned_m)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View PathManager output for mock gates.")
    parser.add_argument(
        "--gates-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of mock gate objects.",
    )
    parser.add_argument(
        "--mode",
        choices=("gate",),
        default=init.PLANNING_MODE,
        help="PathManager planning mode to test.",
    )
    parser.add_argument(
        "--drone-position",
        nargs=3,
        type=float,
        default=MOCK_DRONE_POSITION_LOCAL_NED_M,
        metavar=("N", "E", "D"),
        help="Mock vehicle position in local NED meters.",
    )
    return parser.parse_args()


def _load_gate_payloads(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(MOCK_GATES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--gates-json must contain a JSON list.")
    return [gate for gate in payload if isinstance(gate, dict)]


def _path_manager(mode: str) -> PathManager:
    return PathManager(
        spline_corner_tightness=init.SPLINE_CORNER_TIGHTNESS,
        adaptive_spline_tightness=init.ADAPTIVE_SPLINE_TIGHTNESS,
        distant_spline_corner_tightness=init.DISTANT_SPLINE_CORNER_TIGHTNESS,
        min_spline_corner_tightness=init.MIN_SPLINE_CORNER_TIGHTNESS,
        max_spline_corner_tightness=init.MAX_SPLINE_CORNER_TIGHTNESS,
        gentle_turn_angle_deg=init.GENTLE_TURN_ANGLE_DEG,
        sharp_turn_angle_deg=init.SHARP_TURN_ANGLE_DEG,
        short_segment_reference_m=init.SHORT_SEGMENT_REFERENCE_M,
        long_segment_reference_m=init.LONG_SEGMENT_REFERENCE_M,
        planning_mode=mode,
        path_update_mode=init.PATH_UPDATE_MODE,
        path_crossed_gate_persist_distance_m=init.PATH_CROSSED_GATE_PERSIST_DISTANCE_M,
        path_crossed_gate_curvature_preserve_m=init.PATH_CROSSED_GATE_CURVATURE_PRESERVE_M,
        spacing_m=init.PATH_SPACING_M,
        path_tail_length_m=init.PATH_TAIL_LENGTH_M,
        path_splice_lookahead_gain_s=init.PATH_SPLICE_LOOKAHEAD_GAIN_S,
        max_points=init.PATH_MAX_POINTS,
    )


def _gate_record(payload: dict[str, Any], index: int) -> GateRecord:
    position = (
        payload.get("position_local_ned_m")
        or payload.get("position_local_ned")
        or payload.get("pos")
        or payload.get("position")
    )
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"mock gate {index} must include a 3-value local-NED position.")
    return GateRecord(
        gate_id=str(payload.get("id") or payload.get("gate_id") or f"gate-{index + 1:03d}"),
        position_local_ned_m=tuple(float(value) for value in position),
        quaternion=None,
        position_confidence=float(payload.get("position_confidence", 1.0)),
        quaternion_confidence=None,
        crossed=bool(payload.get("crossed", False)),
        sequence=index,
        observation_count=int(payload.get("observation_count", 1)),
        source="mock",
    )


def _vehicle_state(position_local_ned_m: tuple[float, float, float]) -> VehicleState:
    return VehicleState(
        sim_time_ns=0,
        position_local_ned_m=position_local_ned_m,
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        body_rates_frd_rps=(0.0, 0.0, 0.0),
        acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
    )


def _print_summary(planned_path, gates: list[GateRecord]) -> None:
    print("Gates:")
    for gate in gates:
        print(f"  {gate.gate_id}: {gate.position_local_ned_m} crossed={gate.crossed}")
    print("Planned path:")
    print(f"  source: {planned_path.source}")
    print(f"  gate_ids: {planned_path.gate_ids}")
    print(f"  points: {len(planned_path.points_relative_ned_m)}")
    print(f"  anchors: {planned_path.anchors_relative_ned_m}")
    print(f"  gate_center_errors_m: {planned_path.gate_center_errors_m}")
    print(f"  computation_ms: {planned_path.computation_ms:.3f}")


def _show_path(
    planned_path,
    gates: list[GateRecord],
    drone_position_local_ned_m: tuple[float, float, float],
) -> None:
    points = np.asarray(planned_path.points_relative_ned_m, dtype=float)
    anchors = np.asarray(planned_path.anchors_relative_ned_m, dtype=float)
    drone = np.asarray(drone_position_local_ned_m, dtype=float)
    gate_points = np.asarray([gate.position_local_ned_m for gate in gates], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"PathManager Validation ({planned_path.source})")
    ax.set_xlabel("North +")
    ax.set_ylabel("East +")
    ax.set_zlabel("Down +")

    if len(points):
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", linewidth=2.5, label="planned path")
        ax.scatter(points[0, 0], points[0, 1], points[0, 2], color="tab:green", s=45, label="path start")
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="tab:red", s=45, label="path end")

    if len(anchors):
        ax.plot(anchors[:, 0], anchors[:, 1], anchors[:, 2], "--", color="tab:gray", label="anchors")
        ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], color="tab:gray", s=28)

    ax.scatter([drone[0]], [drone[1]], [drone[2]], color="black", s=70, marker="x", label="drone")
    for gate in gates:
        center = np.asarray(gate.position_local_ned_m, dtype=float)
        color = "tab:orange" if not gate.crossed else "tab:purple"
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=55)
        ax.text(center[0], center[1], center[2], f" {gate.gate_id}", color=color)

    _set_equal_axes(ax, [points, anchors, gate_points, drone.reshape(1, 3)])
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    print("Opening path view. Close the plot window to finish.")
    plt.show()


def _set_equal_axes(ax, arrays: list[np.ndarray]) -> None:
    valid = [array.reshape(-1, 3) for array in arrays if array.size]
    if not valid:
        return
    stacked = np.vstack(valid)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    raise SystemExit(main())
