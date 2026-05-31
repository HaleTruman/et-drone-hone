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
    def __init__(self, endpoint: str, heartbeat_hz: float = 2.0):
        self.endpoint = endpoint
        self.heartbeat_hz = heartbeat_hz
        self._latest_telemetry: TelemetrySample | None = None

    def connect(self) -> None:
        raise NotImplementedError("Select and configure the MAVLink client library.")

    def start_heartbeat(self) -> None:
        raise NotImplementedError("Implement the MAVLink heartbeat worker.")

    def subscribe_telemetry(self) -> None:
        raise NotImplementedError("Subscribe to ATTITUDE, HIGHRES_IMU, and TIMESYNC.")

    def send_position_target(self, target: dict[str, Any]) -> None:
        raise NotImplementedError("Map the payload to SET_POSITION_TARGET_LOCAL_NED.")

    def send_attitude_target(self, target: dict[str, Any]) -> None:
        raise NotImplementedError("Map the payload to SET_ATTITUDE_TARGET.")

    def get_latest_telemetry(self) -> TelemetrySample | None:
        return self._latest_telemetry

    def update_latest_telemetry(self, sample: TelemetrySample) -> None:
        self._latest_telemetry = sample

    def shutdown(self) -> None:
        pass
