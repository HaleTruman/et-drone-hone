import numpy as np

from core.control.body_rate_guidance import BodyRateGuidanceController
from sensing.perception import GateRecord, GateTargetTracker, select_guidance_gate
from sensing.telemetry import MavlinkClient, TelemetrySample


def telemetry(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> TelemetrySample:
    return TelemetrySample(
        sim_time_ns=1,
        position_local_ned_m=position,
        velocity_local_ned_mps=velocity,
        attitude=(1.0, 0.0, 0.0, 0.0),
        body_rates_rps=(0.0, 0.0, 0.0),
        reset_count=1,
    )


def gate(gate_id: str, position, confidence=0.9, crossed=False, sequence=None) -> GateRecord:
    return GateRecord(
        gate_id=gate_id,
        position_local_ned_m=tuple(float(value) for value in position),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        confidence=float(confidence),
        crossed=crossed,
        sequence=sequence,
    )


class FakeMavlinkClient(MavlinkClient):
    def __init__(self, sample: TelemetrySample | None = None) -> None:
        super().__init__("offline")
        self.sample = sample or telemetry()
        self.position_targets = []
        self.attitude_targets = []

    def get_latest_telemetry(self):
        return self.sample

    def send_position_target(self, payload):
        self.position_targets.append(payload)
        self.latest_position_target = payload

    def send_attitude_target(self, payload):
        self.attitude_targets.append(payload)
        self.latest_attitude_target = payload


class FakeLogger:
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event, **payload):
        self.events.append({"event": event, **payload})


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


def test_body_rate_guidance_builds_attitude_target_toward_gate() -> None:
    controller = BodyRateGuidanceController()
    tracker = GateTargetTracker()
    target = tracker.update(gate("gate-1", (1.0, 0.0, -0.4)), now_s=0.0, frame_id=12)

    payload = controller.build_guidance_command(telemetry=telemetry(), target=target)

    assert payload["source"] == "main_body_rate_guidance"
    assert payload["attitude_type_mask"] == 128
    assert payload["thrust"] > controller.config.base_thrust
    assert payload["body_rates_rps"][1] > 0.0
    assert payload["gate_id"] == "gate-1"
    assert payload["vision_frame_id"] == 12
    np.testing.assert_allclose(payload["target_control"]["target_position_local_ned_m"], [1.0, 0.0, -0.4])


def test_stream_body_rate_command_continues_after_missing_frame_then_holds_after_expiry() -> None:
    client = FakeMavlinkClient(telemetry())
    controller = BodyRateGuidanceController()
    tracker = GateTargetTracker(hold_s=0.75)
    tracker.update(gate("gate-1", (10.0, 0.0, 0.0)), now_s=1.0, frame_id=5)

    first = client.stream_gate_body_rate_command(tracker, client.sample, controller, now_s=1.0)
    repeated_without_new_frame = client.stream_gate_body_rate_command(tracker, client.sample, controller, now_s=1.5)
    expired = client.stream_gate_body_rate_command(tracker, client.sample, controller, now_s=2.0)

    assert first["reason"] == "streamed_body_rate_guidance"
    assert repeated_without_new_frame["reason"] == "streamed_body_rate_guidance"
    assert expired["reason"] == "streamed_body_rate_hold_no_target"
    assert len(client.attitude_targets) == 3
    assert client.attitude_targets[-1]["target_control"]["mode"] == "hold_current_position"
    assert client.attitude_targets[-1]["thrust"] == controller.config.base_thrust


def test_run_prelevel_uses_body_rate_guidance_controller_when_provided() -> None:
    client = FakeMavlinkClient(telemetry())
    controller = BodyRateGuidanceController()
    logger = FakeLogger()

    client.run_prelevel(
        guidance_controller=controller,
        duration_s=0.01,
        thrust=0.2,
        hz=1000.0,
        log_event=logger.log_event,
    )

    assert client.attitude_targets
    payload = client.attitude_targets[0]
    assert payload["source"] == "main_body_rate_prelevel"
    assert payload["attitude_type_mask"] == 128
    assert payload["phase"] == "prelevel"
    assert payload["thrust"] == 0.2
    assert [event["event"] for event in logger.events] == ["prelevel_started", "prelevel_finished"]
