import numpy as np

from autonomy.planning import HotStartPlanner
from sensing.perception import GateMap, GateRecord


def gate(gate_id: str, relative, quaternion=(1.0, 0.0, 0.0, 0.0)) -> GateRecord:
    return GateRecord(
        gate_id=gate_id,
        position_local_ned_m=tuple(float(value) for value in relative),
        position_relative_ned_m=tuple(float(value) for value in relative),
        quaternion=quaternion,
        confidence=1.0,
        sequence=int(gate_id),
    )


def test_hot_start_planner_samples_relative_ned_path_through_visible_gates() -> None:
    gate_map = GateMap()
    gate_map.add_or_update_gate(gate("0", (5.0, 0.0, 0.0)))
    gate_map.add_or_update_gate(gate("1", (20.0, 1.0, -1.0)))

    trajectory = HotStartPlanner(spacing_m=1.0).plan_from_gate_map(gate_map)

    assert trajectory.gate_ids == ["0", "1"]
    np.testing.assert_allclose(trajectory.points_relative_ned_m[0], [0.0, 0.0, 0.0])
    assert np.linalg.norm(np.asarray(trajectory.points_relative_ned_m[-1]) - np.array([20.0, 1.0, -1.0])) < 2.0
    assert trajectory.computation_ms >= 0.0


def test_hot_start_planner_uses_gate_orientation_for_approach_axis() -> None:
    gate_map = GateMap()
    yaw_90_quaternion = (0.70710678, 0.0, 0.0, 0.70710678)
    gate_map.add_or_update_gate(gate("0", (-10.0, 0.0, 0.0), quaternion=yaw_90_quaternion))

    trajectory = HotStartPlanner(spacing_m=10.0, gate_axis_offset_m=1.5).plan_from_gate_map(gate_map)

    np.testing.assert_allclose(trajectory.anchors_relative_ned_m[1], [-8.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(trajectory.anchors_relative_ned_m[2], [-10.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(trajectory.anchors_relative_ned_m[3], [-11.5, 0.0, 0.0], atol=1e-6)


def test_hot_start_planner_ignores_gates_without_relative_position() -> None:
    gate_map = GateMap()
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="0",
            position_local_ned_m=(100.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=1.0,
            sequence=0,
        )
    )

    trajectory = HotStartPlanner().plan_from_gate_map(gate_map)

    assert trajectory.gate_ids == []
    np.testing.assert_allclose(trajectory.points_relative_ned_m, [[0.0, 0.0, 0.0]])
