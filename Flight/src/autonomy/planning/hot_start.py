from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from sensing.gates import GateMap, GateRecord


@dataclass(frozen=True)
class HotStartTrajectory:
    points_relative_ned_m: list[list[float]]
    anchors_relative_ned_m: list[list[float]]
    gate_ids: list[str]
    spacing_m: float
    computation_ms: float
    source: str = "hot_start"

    def to_log_dict(self, *, origin_local_ned_m: Any | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["origin_local_ned_m"] = None if origin_local_ned_m is None else [float(value) for value in origin_local_ned_m]
        return payload


class HotStartPlanner:
    """Fast relative-NED trajectory seed through currently visible gates."""

    def __init__(
        self,
        *,
        spacing_m: float = 0.75,
        gate_axis_offset_m: float = 1.5,
        max_points: int = 240,
        max_gates: int = 8,
    ) -> None:
        self.spacing_m = max(0.1, float(spacing_m))
        self.gate_axis_offset_m = max(0.0, float(gate_axis_offset_m))
        self.max_points = max(2, int(max_points))
        self.max_gates = max(1, int(max_gates))

    def plan_from_gate_map(self, gate_map: GateMap) -> HotStartTrajectory:
        started = perf_counter()
        gates = [
            gate
            for gate in gate_map.get_next_gates(self.max_gates)
            if gate.position_relative_ned_m is not None
        ]
        anchors = self._anchors_for_gates(gates)
        points = self._sample_polyline(anchors)
        return HotStartTrajectory(
            points_relative_ned_m=points,
            anchors_relative_ned_m=anchors.astype(float).tolist(),
            gate_ids=[gate.gate_id for gate in gates],
            spacing_m=float(self.spacing_m),
            computation_ms=(perf_counter() - started) * 1000.0,
        )

    def _anchors_for_gates(self, gates: list[GateRecord]) -> np.ndarray:
        anchors = [np.zeros(3, dtype=float)]
        previous = anchors[0]
        for gate in gates:
            center = np.asarray(gate.position_relative_ned_m, dtype=float)
            axis = self._gate_through_axis(gate)
            if float(np.dot(axis, center - previous)) < 0.0:
                axis = -axis
            offset = min(self.gate_axis_offset_m, max(0.0, float(np.linalg.norm(center - previous)) * 0.4))
            if offset > 0.0:
                anchors.append(center - axis * offset)
            anchors.append(center)
            if offset > 0.0:
                anchors.append(center + axis * offset)
                previous = center + axis * offset
            else:
                previous = center
        return self._dedupe_points(np.asarray(anchors, dtype=float))

    def _gate_through_axis(self, gate: GateRecord) -> np.ndarray:
        rotation = _rotation_matrix(gate.quaternion)
        axis = rotation[:, 1]
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return axis / norm

    def _sample_polyline(self, anchors: np.ndarray) -> list[list[float]]:
        if len(anchors) <= 1:
            return anchors.astype(float).tolist()
        samples = [anchors[0]]
        for start, end in zip(anchors[:-1], anchors[1:]):
            delta = end - start
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                continue
            steps = max(1, int(np.ceil(distance / self.spacing_m)))
            for step in range(1, steps + 1):
                samples.append(start + delta * (step / steps))
                if len(samples) >= self.max_points:
                    return np.asarray(samples, dtype=float).tolist()
        return np.asarray(samples, dtype=float).tolist()

    @staticmethod
    def _dedupe_points(points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return points
        deduped = [points[0]]
        for point in points[1:]:
            if float(np.linalg.norm(point - deduped[-1])) > 1e-6:
                deduped.append(point)
        return np.asarray(deduped, dtype=float)


def _rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
