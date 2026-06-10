import numpy as np

from sensing.perception import GateMap, GatePoseEstimator, VisionObservation
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
    assert gate_map.get_reference_path().tolist() == [[11.0, 2.0, -3.0]]


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
