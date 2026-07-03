import numpy as np

from core.control.body_rate_guidance import BodyRateGuidanceController
from autonomy.planning.path_manager import PathManager
from sensing.perception import GateRecord, GateTargetTracker, select_guidance_gate
from core.schemas import OdometryState
from sensing.telemetry import MavlinkTelemetry


def telemetry(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> MavlinkTelemetry:
    return MavlinkTelemetry(
        sim_time_ns=1,
        odometry=OdometryState(
            sim_time_ns=1,
            attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
            position_local_ned_m=position,
            velocity_local_ned_mps=velocity,
            body_rates_frd_rps=(0.0, 0.0, 0.0),
            acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
        ),
        reset_count=1,
    )


def gate(gate_id: str, position, confidence=0.9, crossed=False, sequence=None, relative=None) -> GateRecord:
    return GateRecord(
        gate_id=gate_id,
        position_local_ned_m=tuple(float(value) for value in position),
        position_relative_ned_m=tuple(float(value) for value in (relative if relative is not None else position)),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        confidence=float(confidence),
        crossed=crossed,
        sequence=sequence,
    )


def test_select_guidance_gate_uses_nearest_confident_uncrossed_gate() -> None:
    selected = select_guidance_gate(
        [
            gate("far", (10.0, 0.0, 0.0), confidence=0.9),
            gate("low", (1.0, 0.0, 0.0), confidence=0.05),
            gate("crossed", (0.5, 0.0, 0.0), confidence=0.9, crossed=True),
            gate("near", (2.0, 0.0, 0.0), confidence=0.9),
        ],
        telemetry=telemetry(),
        min_confidence=0.1,
    )

    assert selected is not None
    assert selected.gate_id == "near"


def test_select_guidance_gate_prefers_relative_ned_distance_over_local_map_distance() -> None:
    selected = select_guidance_gate(
        [
            gate("local-near-relative-far", (1.0, 0.0, 0.0), relative=(20.0, 0.0, 0.0)),
            gate("local-far-relative-near", (100.0, 0.0, 0.0), relative=(2.0, 0.0, 0.0)),
        ],
        telemetry=telemetry(),
        min_confidence=0.1,
    )

    assert selected is not None
    assert selected.gate_id == "local-far-relative-near"


def test_body_rate_guidance_builds_attitude_target_toward_gate() -> None:
    controller = BodyRateGuidanceController()
    tracker = GateTargetTracker()
    target = tracker.update(gate("gate-1", (1.0, 0.0, -0.4)), now_s=0.0, frame_id=12)

    payload = controller.build_guidance_command(telemetry=telemetry(), target=target)

    assert payload["source"] == "main_body_rate_guidance"
    assert payload["attitude_type_mask"] == 128
    assert payload["thrust"] > controller.config.base_thrust
    assert payload["body_rates_rps"][1] > 0.0
    assert payload["target_roll_pitch_deg"] == [0.0, 16.0]
    assert payload["gate_id"] == "gate-1"
    assert payload["vision_frame_id"] == 12
    np.testing.assert_allclose(payload["target_control"]["target_position_local_ned_m"], [1.0, 0.0, -0.4])
    np.testing.assert_allclose(tracker.latest(now_s=0.0).position_relative_ned_m, [1.0, 0.0, -0.4])


def test_body_rate_guidance_uses_relative_ned_target_not_local_map_delta() -> None:
    controller = BodyRateGuidanceController()
    tracker = GateTargetTracker()
    target = tracker.update(
        gate("gate-1", (100.0, 50.0, -10.0), relative=(1.0, 0.0, -0.4)),
        now_s=0.0,
        frame_id=12,
    )

    payload = controller.build_guidance_command(telemetry=telemetry(position=(10.0, 0.0, 0.0)), target=target)

    np.testing.assert_allclose(payload["target_control"]["raw_delta_local_ned_m"], [1.0, 0.0, -0.4])
    np.testing.assert_allclose(payload["target_control"]["target_position_local_ned_m"], [11.0, 0.0, -0.4])
    np.testing.assert_allclose(payload["raw_target_position_local_ned_m"], [100.0, 50.0, -10.0])


def test_path_manager_uses_relative_ned_waypoints_when_available() -> None:
    from sensing.perception import GateMap

    gate_map = GateMap()
    gate_map.add_or_update_gate(gate("gate-1", (100.0, 50.0, -10.0), relative=(1.0, 0.0, -0.4)))

    np.testing.assert_allclose(PathManager().update_from_gate_map(gate_map), [[1.0, 0.0, -0.4]])

