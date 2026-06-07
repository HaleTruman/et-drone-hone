import json
import time

from test import LiveHoverTestConfig, LiveHoverTestRunner
from sensing.telemetry.mavlink_bridge import TelemetrySample


class FakeBridge:
    def __init__(self) -> None:
        self.connected = False
        self.is_live = True
        self.armed = False
        self.collisions = []
        self.race_status = None
        self.last_heartbeat_monotonic_s = time.monotonic()
        self.attitude_targets = []
        self.disarmed = False
        self.telemetry = TelemetrySample(
            sim_time_ns=1,
            position_local_ned_m=(2.0, 3.0, -4.0),
            attitude=(1.0, 0.0, 0.0, 0.0),
            velocity_local_ned_mps=(0.0, 0.0, 0.0),
            body_rates_rps=(0.0, 0.0, 0.0),
            acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
            reset_count=0,
        )

    def connect(self, heartbeat_timeout_s: float) -> None:
        self.connected = True

    def start_heartbeat(self) -> None:
        pass

    def subscribe_telemetry(self) -> None:
        pass

    def get_latest_telemetry(self) -> TelemetrySample:
        return self.telemetry

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.disarmed = True

    def send_attitude_target(self, target) -> None:
        self.attitude_targets.append(target)

    def snapshot(self) -> dict:
        return {"connected": self.connected, "armed": self.armed}

    def shutdown(self) -> None:
        self.connected = False


class FakeVisionStream:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start_listener(self) -> None:
        self.started = True

    def snapshot(self) -> dict:
        return {"started": self.started}

    def shutdown(self) -> None:
        self.stopped = True


def test_live_runner_transitions_to_hover_and_sends_attitude_target(tmp_path) -> None:
    bridge = FakeBridge()
    vision = FakeVisionStream()
    config = LiveHoverTestConfig(hover_s=0.01, control_hz=1000.0)

    log_path = LiveHoverTestRunner(config, data_dir=tmp_path, bridge=bridge, vision_stream=vision).run()

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert [event["event"] for event in payload["events"]] == [
        "simulator_started",
        "hover_started",
        "shutdown",
    ]
    assert payload["events"][0]["target_position_local_ned_m"] == [2.0, 3.0, -5.0]
    assert bridge.attitude_targets
    assert "quaternion" in bridge.attitude_targets[0]
    assert "thrust" in bridge.attitude_targets[0]
    assert bridge.disarmed is True
    assert vision.stopped is True
