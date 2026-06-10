"""Canonical runtime schemas for the Path Optimizer stack."""

from dataclasses import dataclass, field
from typing import Any


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImuSample:
    sim_time_ns: int
    acceleration_local_ned_mps2: Vec3
    gyro_frd_rps: Vec3
    velocity_local_ned_mps: Vec3
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttitudeSample:
    sim_time_ns: int
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    euler_rad: Vec3 | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OdometryState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3


@dataclass(frozen=True, init=False)
class TelemetrySample:
    sim_time_ns: int
    odometry: OdometryState
    imu: ImuSample | None = None
    attitude_sample: AttitudeSample | None = None
    system_status: str | None = None
    reset_count: int | None = None
    diagnostic_odometry: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        sim_time_ns: int,
        odometry: OdometryState | None = None,
        *,
        imu: ImuSample | None = None,
        attitude_sample: AttitudeSample | None = None,
        attitude: QuatWxyz | None = None,
        velocity_local_ned_mps: Vec3 | None = None,
        body_rates_rps: Vec3 | None = None,
        body_rates_frd_rps: Vec3 | None = None,
        position_local_ned_m: Vec3 | None = None,
        acceleration_local_ned_mps2: Vec3 | None = None,
        system_status: str | None = None,
        reset_count: int | None = None,
        diagnostic_odometry: dict[str, Any] | None = None,
        raw: dict[str, Any] | None = None,
    ):
        if odometry is None:
            odometry = OdometryState(
                sim_time_ns=int(sim_time_ns),
                position_local_ned_m=position_local_ned_m or (0.0, 0.0, 0.0),
                velocity_local_ned_mps=velocity_local_ned_mps or (0.0, 0.0, 0.0),
                attitude_quaternion=attitude or (1.0, 0.0, 0.0, 0.0),
                body_rates_frd_rps=body_rates_frd_rps or body_rates_rps or (0.0, 0.0, 0.0),
                acceleration_local_ned_mps2=acceleration_local_ned_mps2 or (0.0, 0.0, 0.0),
            )
        object.__setattr__(self, "sim_time_ns", int(sim_time_ns))
        object.__setattr__(self, "odometry", odometry)
        object.__setattr__(self, "imu", imu)
        object.__setattr__(self, "attitude_sample", attitude_sample)
        object.__setattr__(self, "system_status", system_status)
        object.__setattr__(self, "reset_count", reset_count)
        object.__setattr__(self, "diagnostic_odometry", diagnostic_odometry)
        object.__setattr__(self, "raw", raw or {})

    @property
    def position_local_ned_m(self) -> Vec3:
        return self.odometry.position_local_ned_m

    @property
    def velocity_local_ned_mps(self) -> Vec3:
        return self.odometry.velocity_local_ned_mps

    @property
    def attitude_quaternion(self) -> QuatWxyz:
        return self.odometry.attitude_quaternion

    @property
    def attitude(self) -> QuatWxyz:
        return self.odometry.attitude_quaternion

    @property
    def body_rates_frd_rps(self) -> Vec3:
        return self.odometry.body_rates_frd_rps

    @property
    def body_rates_rps(self) -> Vec3:
        return self.odometry.body_rates_frd_rps

    @property
    def acceleration_local_ned_mps2(self) -> Vec3:
        return self.odometry.acceleration_local_ned_mps2


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
class MavlinkAttitude:
    time_boot_ms: int
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    angular_velocity_body_frd_rps: Vec3


@dataclass(frozen=True)
class MavlinkLocalPositionNed:
    time_boot_ms: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3


@dataclass(frozen=True)
class MavlinkOdometry:
    time_usec: int
    frame_id: int
    child_frame_id: int
    attitude_quaternion: QuatWxyz
    pose_covariance: tuple[float, ...]
    velocity_covariance: tuple[float, ...]
    reset_count: int
    estimator_type: int
    position_local_ned_m: Vec3 | None = None
    velocity_local_ned_mps: Vec3 | None = None
    position_m: Vec3 | None = None
    velocity_mps: Vec3 | None = None
    angular_velocity_body_frd_rps: Vec3 | None = None
    angular_velocity_rps: Vec3 | None = None
    source: str = "ODOMETRY"


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


@dataclass(frozen=True)
class MavlinkActuatorOutputStatus:
    time_boot_us: int
    active: int
    actuator: tuple[float, ...]
