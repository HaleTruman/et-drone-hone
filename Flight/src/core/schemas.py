"""Canonical runtime schemas for the Flight stack."""

from dataclasses import dataclass, field
from typing import Any


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class OdometryState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3


@dataclass(frozen=True)
class RuntimeStatus:
    connected: bool
    running: bool
    heartbeat_started: bool
    telemetry_subscribed: bool
    armed: bool
    latest_message_age_s: float | None = None
    latest_heartbeat_age_s: float | None = None


@dataclass(frozen=True)
class RaceStatus:
    sim_boot_time_ms: int
    race_start_boot_time_ms: int
    race_finish_time_ns: int
    active_gate_index: int
    last_gate_race_time: int


@dataclass(frozen=True)
class TrackGate:
    gate_id: int
    position_local_ned_m: Vec3
    quaternion: QuatWxyz
    width_m: float
    height_m: float


@dataclass(frozen=True)
class CollisionEvent:
    collision_id: int
    threat_level: int
    impact_kg_mps: float


@dataclass(frozen=True)
class MavlinkHeartbeat:
    type: Any
    autopilot: Any
    base_mode: int
    custom_mode: Any
    system_status: Any
    mavlink_version: Any


@dataclass(frozen=True)
class MavlinkTimesync:
    ts1: int
    tc1: int


@dataclass(frozen=True)
class MavlinkHighresImu:
    time_boot_us: int
    acceleration_body_frd_mps2: Vec3
    gyro_body_frd_rps: Vec3
    magnetic_field_gauss: Vec3 | None = None
    absolute_pressure_hpa: float | None = None
    differential_pressure_hpa: float | None = None
    pressure_altitude_m: float | None = None
    temperature_c: float | None = None
    fields_updated: int | None = None
    id: int | None = None


@dataclass(frozen=True, init=False)
class MavlinkTelemetry:
    sim_time_ns: int
    odometry: OdometryState | None = None
    imu: MavlinkHighresImu | None = None
    system_status: str | None = None
    reset_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        sim_time_ns: int,
        odometry: OdometryState | None = None,
        *,
        imu: MavlinkHighresImu | None = None,
        system_status: str | None = None,
        reset_count: int | None = None,
        raw: dict[str, Any] | None = None,
    ):
        object.__setattr__(self, "sim_time_ns", int(sim_time_ns))
        object.__setattr__(self, "odometry", odometry)
        object.__setattr__(self, "imu", imu)
        object.__setattr__(self, "system_status", system_status)
        object.__setattr__(self, "reset_count", reset_count)
        object.__setattr__(self, "raw", raw or {})

    @property
    def acceleration_body_frd_mps2(self) -> Vec3 | None:
        if self.imu is None:
            return None
        return self.imu.acceleration_body_frd_mps2

    @property
    def gyro_body_frd_rps(self) -> Vec3 | None:
        if self.imu is None:
            return None
        return self.imu.gyro_body_frd_rps


@dataclass(frozen=True)
class MavlinkActuatorOutputStatus:
    time_boot_us: int
    active: int
    actuator: tuple[float, ...]
