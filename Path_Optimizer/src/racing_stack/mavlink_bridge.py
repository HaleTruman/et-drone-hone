from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TelemetrySample:
    sim_time_ns: int
    attitude: tuple[float, float, float, float]
    velocity_local_ned_mps: tuple[float, float, float]
    body_rates_rps: tuple[float, float, float]
    system_status: str | None = None
    raw: dict[str, Any] | None = None


class MavlinkBridge:
    """Transport-neutral MAVLink state cache until the UDP client is selected."""

    def __init__(self, endpoint: str, heartbeat_hz: float = 2.0):
        self.endpoint = endpoint
        self.heartbeat_hz = heartbeat_hz
        self._latest_telemetry: TelemetrySample | None = None
        self.connected = False
        self.heartbeat_started = False
        self.telemetry_subscribed = False
        self.latest_position_target: dict[str, Any] | None = None
        self.latest_attitude_target: dict[str, Any] | None = None

    def connect(self) -> None:
        self.connected = True

    def start_heartbeat(self) -> None:
        self.heartbeat_started = True

    def subscribe_telemetry(self) -> None:
        self.telemetry_subscribed = True

    def send_position_target(self, target: dict[str, Any]) -> None:
        self.latest_position_target = target

    def send_attitude_target(self, target: dict[str, Any]) -> None:
        self.latest_attitude_target = target

    def get_latest_telemetry(self) -> TelemetrySample | None:
        return self._latest_telemetry

    def update_latest_telemetry(self, sample: TelemetrySample) -> None:
        self._latest_telemetry = sample

    def shutdown(self) -> None:
        self.connected = False
