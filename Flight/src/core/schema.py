"""Canonical runtime schemas for the Flight stack."""

import copy
from dataclasses import dataclass, field
from typing import Any


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class VehicleState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    image: Any | None = None
    saved_path: str | None = None


@dataclass(frozen=True)
class VisionGateObservation:
    gate_id: str
    position_local_ned: Vec3
    position_confidence: float
    orientation_local_ned_quat: QuatWxyz | None = None
    orientation_confidence: float | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VisionGateObservation":
        position = payload.get("position_local_ned") or payload.get("position_xyz")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("Vision gate payload must include position_local_ned with three values.")
        orientation = payload.get("orientation_local_ned_quat")
        if orientation is None:
            orientation = payload.get("orientation_quat")
        if orientation is not None and (not isinstance(orientation, (list, tuple)) or len(orientation) != 4):
            raise ValueError("Vision gate payload orientation_local_ned_quat must include four values.")
        return cls(
            gate_id=str(payload.get("id", payload.get("gate_id", ""))),
            position_local_ned=tuple(float(value) for value in position),
            position_confidence=float(payload.get("position_confidence", 0.0)),
            orientation_local_ned_quat=None
            if orientation is None
            else tuple(float(value) for value in orientation),
            orientation_confidence=None
            if payload.get("orientation_confidence") is None
            else float(payload.get("orientation_confidence")),
            trace=dict(payload.get("trace") or {}) if isinstance(payload.get("trace"), dict) else {},
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.gate_id,
            "position_local_ned": [float(value) for value in self.position_local_ned],
            "position_confidence": float(self.position_confidence),
            "orientation_local_ned_quat": None
            if self.orientation_local_ned_quat is None
            else [float(value) for value in self.orientation_local_ned_quat],
            "orientation_confidence": self.orientation_confidence,
        }
        if self.trace:
            payload["trace"] = self.trace
        return payload


@dataclass(frozen=True)
class VisionObservation:
    frame_id: int
    sim_time_ns: int
    gates: list[VisionGateObservation]
    source: str = "vision"
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_controller_payload(cls, payload: dict[str, Any], *, source: str = "vision") -> "VisionObservation":
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        cycle = int(run.get("cycle", 0))
        sim_time_ns = int(run.get("sim_time_ns", 0))
        gates = [
            VisionGateObservation.from_payload(gate)
            for gate in payload.get("gates", [])
            if isinstance(gate, dict)
        ]
        trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
        return cls(frame_id=cycle, sim_time_ns=sim_time_ns, gates=gates, source=source, trace=dict(trace))

    def to_controller_payload(self, *, output_dir: str = "memory") -> dict[str, Any]:
        payload = {
            "run": {
                "output_dir": output_dir,
                "cycle": int(self.frame_id),
                "frame_id": f"frame_{int(self.frame_id):06d}",
                "sim_time_ns": int(self.sim_time_ns),
            },
            "gates": [gate.to_payload() for gate in self.gates],
            "obstacles": [],
        }
        if self.trace:
            payload["trace"] = self.trace
        return payload


@dataclass(frozen=True)
class VioCorrection:
    """Deferred visual-odometry correction queued for the next state update.

    - measurement: VIO pose/velocity measurement to blend into the estimator.
    - frame_id: Vision frame that produced the measurement.
    - frame_sim_time_ns: Simulator timestamp from that vision frame.
    - queued_inner_cycle: Inner-loop cycle when the correction was queued.
    - queued_outer_cycle: Outer-loop cycle when the correction was queued.
    - source: Human-readable source label for logging/debugging.
    """

    measurement: Any
    frame_id: int
    frame_sim_time_ns: int
    queued_inner_cycle: int
    queued_outer_cycle: int
    source: str = "vio"

    def to_log_dict(self) -> dict[str, Any]:
        measurement_payload = self.measurement
        if hasattr(self.measurement, "to_log_dict"):
            measurement_payload = self.measurement.to_log_dict()
        return {
            "frame_id": int(self.frame_id),
            "frame_sim_time_ns": int(self.frame_sim_time_ns),
            "queued_inner_cycle": int(self.queued_inner_cycle),
            "queued_outer_cycle": int(self.queued_outer_cycle),
            "source": self.source,
            "measurement": measurement_payload,
        }


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
    """
    HIGHRES_IMU sample normalized for the flight stack.
    - time_boot_us: Sensor sample time in microseconds since vehicle boot.
    - acceleration_body_frd_mps2: Specific force in the body FRD frame, in m/s^2.
    - gyro_body_frd_rps: Angular velocity in the body FRD frame, in rad/s.
    - magnetic_field_gauss: Magnetic field vector in gauss, when provided.
    - absolute_pressure_hpa: Static absolute pressure in hectopascals, when provided.
    - differential_pressure_hpa: Differential pressure in hectopascals, when provided.
    - pressure_altitude_m: Barometric pressure altitude in meters, when provided.
    - temperature_c: IMU temperature in degrees Celsius, when provided.
    - fields_updated: MAVLink bitmask indicating which HIGHRES_IMU fields changed.
    - id: Sensor instance identifier, when provided by the sender.
    """

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
    vehicle_state: VehicleState | None = None
    imu: MavlinkHighresImu | None = None
    system_status: str | None = None
    reset_count: int | None = None
    sim_truth: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        sim_time_ns: int,
        vehicle_state: VehicleState | None = None,
        *,
        odometry: VehicleState | None = None,
        imu: MavlinkHighresImu | None = None,
        system_status: str | None = None,
        reset_count: int | None = None,
        sim_truth: dict[str, Any] | None = None,
        raw: dict[str, Any] | None = None,
    ):
        object.__setattr__(self, "sim_time_ns", int(sim_time_ns))
        object.__setattr__(self, "vehicle_state", vehicle_state if vehicle_state is not None else odometry)
        object.__setattr__(self, "imu", imu)
        object.__setattr__(self, "system_status", system_status)
        object.__setattr__(self, "reset_count", reset_count)
        object.__setattr__(self, "sim_truth", copy.deepcopy(sim_truth) if sim_truth is not None else None)
        object.__setattr__(self, "raw", raw or {})

    @property
    def odometry(self) -> VehicleState | None:
        return self.vehicle_state

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
class StateRecord:
    """Log-focused snapshot of raw telemetry plus the fused estimator update."""

    sim_time_ns: int
    vehicle_state: VehicleState | None = None
    imu: MavlinkHighresImu | None = None
    system_status: str | None = None
    reset_count: int | None = None
    sim_truth: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    state_update: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_telemetry(
        cls,
        telemetry: MavlinkTelemetry,
        *,
        vehicle_state: VehicleState | None,
        vio_correction: VioCorrection | None = None,
        vio_status: str | None = None,
        vio_residual: dict[str, object] | None = None,
        kalman_status: str | None = None,
    ) -> "StateRecord":
        sim_time_ns = vehicle_state.sim_time_ns if vehicle_state is not None else telemetry.sim_time_ns
        vehicle_state_source = (
            "vehicle_state_estimator_highres_imu_vio_kalman"
            if vio_correction is not None and vio_status == "accepted" and kalman_status == "vio_update"
            else "vehicle_state_estimator_highres_imu_vio"
            if vio_correction is not None and vio_status == "accepted"
            else "vehicle_state_estimator_highres_imu_kalman"
            if kalman_status is not None
            else "vehicle_state_estimator_highres_imu"
        )
        vio_correction_payload = None if vio_correction is None else vio_correction.to_log_dict()
        state_update = {
            "vehicle_state_source": vehicle_state_source,
            "vio_status": vio_status,
            "vio_residual": vio_residual,
            "vio_correction": vio_correction_payload,
            "kalman_status": kalman_status,
        }
        return cls(
            sim_time_ns=int(sim_time_ns),
            vehicle_state=vehicle_state,
            imu=telemetry.imu,
            system_status=telemetry.system_status,
            reset_count=telemetry.reset_count,
            sim_truth=copy.deepcopy(telemetry.sim_truth),
            raw={
                **telemetry.raw,
                **state_update,
            },
            state_update=state_update,
        )


@dataclass(frozen=True)
class MavlinkActuatorOutputStatus:
    time_boot_us: int
    active: int
    actuator: tuple[float, ...]
