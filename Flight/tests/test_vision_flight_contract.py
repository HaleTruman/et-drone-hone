import numpy as np

from sensing.perception import GateMap, GatePoseEstimator, GateRecord, VisionObservation
from sensing.telemetry import DataSynchronizer, TelemetrySample


def telemetry_sample(sim_time_ns: int) -> TelemetrySample:
    return TelemetrySample(
        sim_time_ns=sim_time_ns,
        attitude=(1.0, 0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        body_rates_rps=(0.0, 0.0, 0.0),
        position_local_ned_m=(1.0, 2.0, -3.0),
    )


def test_controller_payload_becomes_typed_vision_observation() -> None:
    observation = VisionObservation.from_controller_payload(
        {
            "run": {"cycle": 7, "sim_time_ns": 123},
            "gates": [
                {
                    "id": "gate-001w",
                    "position_xyz": [10.0, 1.0, 2.0],
                    "position_confidence": 0.8,
                    "orientation_xyz": [1.0, 0.0, 0.0],
                    "orientation_confidence": 0.7,
                }
            ],
            "obstacles": [],
        }
    )

    assert observation.frame_id == 7
    assert observation.sim_time_ns == 123
    assert observation.gates[0].gate_id == "gate-001w"
    assert observation.gates[0].position_camera_m == (10.0, 1.0, 2.0)


def test_gate_pose_estimator_maps_camera_observation_to_gate_record() -> None:
    observation = VisionObservation.from_controller_payload(
        {
            "run": {"cycle": 1, "sim_time_ns": 99},
            "gates": [
                {
                    "id": "gate-001w",
                    "position_xyz": [0.0, 0.0, 10.0],
                    "position_confidence": 0.5,
                    "orientation_xyz": [0.0, 0.0, 1.0],
                    "orientation_confidence": 0.5,
                }
            ],
        }
    )
    estimator = GatePoseEstimator(camera_tilt_deg=0.0)

    record = estimator.estimate_gate_pose(
        observation.gates[0],
        vehicle_position_local_ned_m=np.array([1.0, 2.0, -3.0]),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        sequence=3,
    )

    assert record.gate_id == "gate-001w"
    assert record.position_local_ned_m == (11.0, 2.0, -3.0)
    assert record.position_relative_ned_m == (10.0, 0.0, 0.0)
    assert record.confidence == 0.5
    assert record.sequence == 3
    assert np.isclose(np.linalg.norm(record.quaternion), 1.0)


def test_gate_pose_estimator_updates_gate_map_from_observation_and_telemetry() -> None:
    observation = VisionObservation.from_controller_payload(
        {
            "run": {"cycle": 1, "sim_time_ns": 99},
            "gates": [
                {
                    "id": "gate-001w",
                    "position_xyz": [0.0, 0.0, 10.0],
                    "position_confidence": 0.5,
                }
            ],
        }
    )
    gate_map = GateMap()

    records = GatePoseEstimator(camera_tilt_deg=0.0).update_gate_map_from_observation(
        observation,
        telemetry=telemetry_sample(99),
        gate_map=gate_map,
    )

    assert records[0] == gate_map.get_gate("gate-001w")
    assert records[0].last_observed_cycle == 1
    assert records[0].position_relative_ned_m == (10.0, 0.0, 0.0)
    assert gate_map.get_reference_path().tolist() == [[11.0, 2.0, -3.0]]


def test_gate_map_associates_nearby_observations_with_existing_gate() -> None:
    gate_map = GateMap(association_distance_m=2.0)

    first = gate_map.add_or_update_gate(
        GateRecord(
            gate_id="frame-1-gate-a",
            position_local_ned_m=(10.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
            sequence=0,
        )
    )
    second = gate_map.add_or_update_gate(
        GateRecord(
            gate_id="frame-2-gate-b",
            position_local_ned_m=(11.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
            sequence=1,
        )
    )

    gates = gate_map.get_next_gates(10)
    assert len(gates) == 1
    assert second.gate_id == first.gate_id
    assert gates[0].gate_id == "frame-1-gate-a"
    assert gates[0].position_local_ned_m == (10.5, 0.0, 0.0)
    assert gates[0].sequence == 0


def test_gate_map_keeps_far_observations_as_unique_gates() -> None:
    gate_map = GateMap(association_distance_m=2.0)

    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="gate-a",
            position_local_ned_m=(10.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
        )
    )
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="gate-b",
            position_local_ned_m=(13.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
        )
    )

    assert [gate.gate_id for gate in gate_map.get_next_gates(10)] == ["gate-a", "gate-b"]


def test_gate_map_anchors_noisy_vision_to_authoritative_track_gate() -> None:
    gate_map = GateMap(association_distance_m=2.0, authoritative_association_distance_m=18.0)
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="0",
            position_local_ned_m=(-23.0, -0.4, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=1.0,
            sequence=0,
            source="track",
        )
    )

    mapped = gate_map.add_or_update_gate(
        GateRecord(
            gate_id="gate-a264",
            position_local_ned_m=(-30.0, -0.7, -3.5),
            position_relative_ned_m=(-8.0, -0.3, -3.5),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
            last_observed_cycle=10,
        ),
        allow_new=False,
    )

    gates = gate_map.get_next_gates(10)
    assert mapped is not None
    assert mapped.gate_id == "0"
    assert len(gates) == 1
    assert gates[0].position_local_ned_m == (-23.0, -0.4, 0.0)
    assert gates[0].position_relative_ned_m == (-8.0, -0.3, -3.5)
    assert gates[0].observation_count == 2


def test_gate_map_rejects_new_vision_gate_when_authoritative_map_exists() -> None:
    gate_map = GateMap(association_distance_m=2.0, authoritative_association_distance_m=10.0)
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="0",
            position_local_ned_m=(0.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=1.0,
            sequence=0,
            source="track",
        )
    )

    mapped = gate_map.add_or_update_gate(
        GateRecord(
            gate_id="spurious",
            position_local_ned_m=(100.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.8,
        ),
        allow_new=False,
    )

    assert mapped is None
    assert [gate.gate_id for gate in gate_map.get_next_gates(10)] == ["0"]


def test_gate_map_treats_track_gate_as_confirmed_with_single_observation() -> None:
    gate_map = GateMap(min_observations=2)

    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="0",
            position_local_ned_m=(0.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=1.0,
            sequence=0,
            source="track",
        )
    )

    assert [gate.gate_id for gate in gate_map.get_next_gates(10)] == ["0"]


def test_gate_map_coalesces_existing_duplicate_when_tracks_converge() -> None:
    gate_map = GateMap(association_distance_m=6.0)
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="gate-b-old",
            position_local_ned_m=(-45.0, -2.0, 4.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.9,
            observation_count=8,
        )
    )
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="gate-c-old",
            position_local_ned_m=(-52.0, -2.0, 4.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.9,
            observation_count=4,
        )
    )

    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="gate-c-old",
            position_local_ned_m=(-49.0, -2.0, 4.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.9,
            last_observed_cycle=30,
        )
    )

    gates = gate_map.get_next_gates(10)
    assert len(gates) == 1
    assert gates[0].observation_count == 13
    assert gates[0].last_observed_cycle == 30


def test_gate_map_requires_distinct_cycle_observations_for_confirmed_gates() -> None:
    gate_map = GateMap(association_distance_m=2.0, min_observations=2)

    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="same-cycle-a",
            position_local_ned_m=(10.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
            last_observed_cycle=1,
        )
    )
    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="same-cycle-b",
            position_local_ned_m=(10.5, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
            last_observed_cycle=1,
        )
    )

    assert gate_map.get_next_gates(10) == []

    gate_map.add_or_update_gate(
        GateRecord(
            gate_id="next-cycle",
            position_local_ned_m=(10.25, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            confidence=0.75,
            last_observed_cycle=2,
        )
    )

    gates = gate_map.get_next_gates(10)
    assert len(gates) == 1
    assert gates[0].observation_count == 2
    assert gates[0].last_observed_cycle == 2


def test_gate_pose_estimator_applies_upward_camera_tilt() -> None:
    observation = VisionObservation.from_controller_payload(
        {
            "run": {"cycle": 1, "sim_time_ns": 99},
            "gates": [
                {
                    "id": "gate-001w",
                    "position_xyz": [0.0, 0.0, 10.0],
                    "position_confidence": 0.5,
                }
            ],
        }
    )

    record = GatePoseEstimator(camera_tilt_deg=20.0).estimate_gate_pose(
        observation.gates[0],
        vehicle_position_local_ned_m=np.array([0.0, 0.0, 0.0]),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
    )

    assert np.allclose(record.position_local_ned_m, (9.396926207859085, 0.0, -3.420201433256687), atol=1e-9)


def test_synchronizer_aligns_vision_observation_with_telemetry() -> None:
    sync = DataSynchronizer(max_delta_ns=10)
    observation = VisionObservation(frame_id=1, sim_time_ns=100, gates=())

    pair = sync.align_observation_with_telemetry(
        observation,
        [telemetry_sample(80), telemetry_sample(104), telemetry_sample(130)],
    )

    assert pair is not None
    assert pair[0] is observation
    assert pair[1].sim_time_ns == 104
    assert sync.get_synchronized_data() == pair
