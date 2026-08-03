"""MAVLink transport and state cache for the AI GP simulator."""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2
MAVLINK_CMD_SIM_RESET = 31000


@dataclass(frozen=True)
class TelemetrySample:
    sim_time_ns: int
    attitude: tuple[float, float, float, float]
    velocity_local_ned_mps: tuple[float, float, float]
    body_rates_rps: tuple[float, float, float]
    position_local_ned_m: tuple[float, float, float] | None = None
    acceleration_local_ned_mps2: tuple[float, float, float] | None = None
    system_status: str | None = None
    reset_count: int | None = None
    raw: dict[str, Any] | None = None


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
    position_local_ned_m: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    width_m: float
    height_m: float


@dataclass(frozen=True)
class CollisionEvent:
    collision_id: int
    threat_level: int
    impact_kg_mps: float


class MavlinkBridge:
    """Live MAVLink UDP client with a transport-neutral offline fallback."""

    def __init__(
        self,
        endpoint: str,
        heartbeat_hz: float = 2.0,
        timesync_hz: float = 10.0,
        connection_factory: Callable[[str], Any] | None = None,
    ):
        self.endpoint = endpoint
        self.heartbeat_hz = float(heartbeat_hz)
        self.timesync_hz = float(timesync_hz)
        self._connection_factory = connection_factory
        self._connection: Any | None = None
        self._receiver_thread: threading.Thread | None = None
        self._timesync_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._expected_track_chunks: dict[int, int] = {}

        self._latest_telemetry: TelemetrySample | None = None
        self.starting_telemetry: TelemetrySample | None = None
        self.connected = False
        self.heartbeat_started = False
        self.telemetry_subscribed = False
        self.armed = False
        self.last_heartbeat_monotonic_s: float | None = None
        self.latest_timesync: dict[str, int] | None = None
        self.latest_attitude: dict[str, Any] | None = None
        self.latest_local_position: dict[str, Any] | None = None
        self.latest_imu: dict[str, Any] | None = None
        self.latest_actuator_output: dict[str, Any] | None = None
        self.race_status: RaceStatus | None = None
        self.track_gates: list[TrackGate] = []
        self.collisions: list[CollisionEvent] = []
        self.latest_position_target: dict[str, Any] | None = None
        self.latest_attitude_target: dict[str, Any] | None = None

    @property
    def is_live(self) -> bool:
        return self._connection_factory is not None or ":" in self.endpoint

    def connect(self, heartbeat_timeout_s: float = 10.0) -> None:
        if not self.is_live:
            self.connected = True
            return
        factory = self._connection_factory
        if factory is None:
            from pymavlink import mavutil

            factory = mavutil.mavlink_connection
        self._connection = factory(self.endpoint)
        heartbeat = self._connection.wait_heartbeat(timeout=heartbeat_timeout_s)
        if heartbeat is None:
            raise TimeoutError(f"No simulator heartbeat received from {self.endpoint}")
        self._on_heartbeat(heartbeat)
        self.connected = True

    def start_heartbeat(self) -> None:
        """Start periodic TIMESYNC requests used to align client and simulator time."""

        self.heartbeat_started = True
        if not self.is_live or self._timesync_thread is not None:
            return
        self._running.set()
        self._timesync_thread = threading.Thread(target=self._timesync_loop, name="mavlink-timesync", daemon=True)
        self._timesync_thread.start()

    def subscribe_telemetry(self) -> None:
        self.telemetry_subscribed = True
        if not self.is_live or self._receiver_thread is not None:
            return
        self._running.set()
        self._receiver_thread = threading.Thread(target=self._receive_loop, name="mavlink-rx", daemon=True)
        self._receiver_thread.start()

    def arm(self) -> None:
        mavutil = self._require_mavutil()
        connection = self._require_connection()
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def disarm(self) -> None:
        mavutil = self._require_mavutil()
        connection = self._require_connection()
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_sim_reset_command(self) -> None:
        connection = self._require_connection()
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_position_target(self, target: dict[str, Any]) -> None:
        self.latest_position_target = target
        if not self.is_live:
            return
        mavutil = self._require_mavutil()
        connection = self._require_connection()
        position = target.get("position_local_ned_m")
        position_axes = target.get("position_axes")
        velocity = target["velocity_local_ned_mps"]
        yaw = target.get("yaw_rad")
        mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        if position is None:
            position = (0.0, 0.0, 0.0)
            mask |= (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            )
        elif position_axes is not None:
            axes = [bool(value) for value in position_axes]
            if len(axes) != 3:
                raise ValueError("position_axes must contain [use_x, use_y, use_z]")
            if not axes[0]:
                mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            if not axes[1]:
                mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            if not axes[2]:
                mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        if yaw is None:
            yaw = 0.0
            mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        connection.mav.set_position_target_local_ned_send(
            self._time_boot_ms(),
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask,
            *position,
            *velocity,
            0.0,
            0.0,
            0.0,
            yaw,
            0.0,
        )

    def send_attitude_target(self, target: dict[str, Any]) -> None:
        self.latest_attitude_target = target
        if not self.is_live:
            return
        mavutil = self._require_mavutil()
        connection = self._require_connection()
        body_rates = target.get("body_rates_rps", (0.0, 0.0, 0.0))
        type_mask = int(target.get("attitude_type_mask", 0 if "body_rates_rps" in target else 7))
        connection.mav.set_attitude_target_send(
            self._time_boot_ms(),
            connection.target_system,
            connection.target_component,
            type_mask,
            target["quaternion"],
            *body_rates,
            target["thrust"],
        )

    def send_motor_target(self, motor_commands: list[float] | tuple[float, ...]) -> None:
        connection = self._require_connection()
        commands = [float(value) for value in motor_commands]
        if len(commands) != 4:
            raise ValueError("Expected four normalized motor commands")
        connection.mav.set_actuator_control_target_send(
            int(time.time() * 1e6),
            connection.target_system,
            connection.target_component,
            0,
            commands + [0.0, 0.0, 0.0, 0.0],
        )

    def get_latest_telemetry(self) -> TelemetrySample | None:
        with self._lock:
            return self._latest_telemetry

    def update_latest_telemetry(self, sample: TelemetrySample) -> None:
        with self._lock:
            self._latest_telemetry = sample
            if self.starting_telemetry is None:
                self.starting_telemetry = sample

    def snapshot(self) -> dict[str, Any]:
        telemetry = self.get_latest_telemetry()
        return {
            "endpoint": self.endpoint,
            "connected": self.connected,
            "armed": self.armed,
            "latest_telemetry": asdict(telemetry) if telemetry else None,
            "starting_telemetry": asdict(self.starting_telemetry) if self.starting_telemetry else None,
            "latest_timesync": self.latest_timesync,
            "latest_attitude": self.latest_attitude,
            "latest_local_position": self.latest_local_position,
            "latest_imu": self.latest_imu,
            "latest_actuator_output": self.latest_actuator_output,
            "race_status": asdict(self.race_status) if self.race_status else None,
            "track_gates": [asdict(gate) for gate in self.track_gates],
            "collisions": [asdict(collision) for collision in self.collisions],
        }

    def populate_gate_map(self, gate_map: Any) -> None:
        from Examples.aigpq1_demo.src.sensing.perception.gate_map import GateRecord

        for sequence, gate in enumerate(self.track_gates):
            gate_map.add_or_update_gate(
                GateRecord(
                    gate_id=str(gate.gate_id),
                    position_local_ned_m=gate.position_local_ned_m,
                    quaternion=gate.quaternion,
                    confidence=1.0,
                    sequence=sequence,
                )
            )

    def handle_message(self, msg: Any) -> None:
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            self._on_heartbeat(msg)
        elif msg_type == "TIMESYNC":
            self._on_timesync(msg)
        elif msg_type == "ATTITUDE":
            self._on_attitude(msg)
        elif msg_type == "LOCAL_POSITION_NED":
            self._on_local_position_ned(msg)
        elif msg_type == "ODOMETRY":
            self._on_odometry(msg)
        elif msg_type == "HIGHRES_IMU":
            self._on_highres_imu(msg)
        elif msg_type == "ENCAPSULATED_DATA":
            self._on_encapsulated_data(msg)
        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            self._on_actuator_output_status(msg)
        elif msg_type == "COLLISION":
            self._on_collision(msg)
        elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            transfer_id = int(msg.width)
            self._track_chunks[transfer_id] = {}
            self._expected_track_chunks[transfer_id] = int(msg.packets)

    def shutdown(self) -> None:
        self._running.clear()
        for thread in (self._receiver_thread, self._timesync_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        self.connected = False

    def _receive_loop(self) -> None:
        connection = self._require_connection()
        while self._running.is_set():
            try:
                msg = connection.recv_match(blocking=False)
            except ConnectionResetError:
                self.connected = False
                return
            if msg is None:
                time.sleep(0.001)
            elif msg.get_type() != "BAD_DATA":
                self.handle_message(msg)

    def _timesync_loop(self) -> None:
        connection = self._require_connection()
        while self._running.is_set():
            connection.mav.timesync_send(time.time_ns(), 0)
            time.sleep(1.0 / self.timesync_hz)

    def _on_heartbeat(self, msg: Any) -> None:
        mavutil = self._require_mavutil()
        self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        self.last_heartbeat_monotonic_s = time.monotonic()

    def _on_timesync(self, msg: Any) -> None:
        self.latest_timesync = {"request_time_ns": int(msg.ts1), "response_time_ns": int(msg.tc1)}

    def _on_attitude(self, msg: Any) -> None:
        self.latest_attitude = {
            "time_boot_ms": int(msg.time_boot_ms),
            "euler_rad": (float(msg.roll), float(msg.pitch), float(msg.yaw)),
            "body_rates_rps": (float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)),
        }

    def _on_local_position_ned(self, msg: Any) -> None:
        self.latest_local_position = {
            "time_boot_ms": int(msg.time_boot_ms),
            "position_local_ned_m": (float(msg.x), float(msg.y), float(msg.z)),
            "velocity_local_ned_mps": (float(msg.vx), float(msg.vy), float(msg.vz)),
        }

    def _on_odometry(self, msg: Any) -> None:
        acceleration = None if self.latest_imu is None else self.latest_imu["acceleration_local_ned_mps2"]
        sample = TelemetrySample(
            sim_time_ns=int(msg.time_usec) * 1_000,
            position_local_ned_m=(float(msg.x), float(msg.y), float(msg.z)),
            attitude=tuple(float(value) for value in msg.q),
            velocity_local_ned_mps=(float(msg.vx), float(msg.vy), float(msg.vz)),
            body_rates_rps=(float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)),
            acceleration_local_ned_mps2=acceleration,
            reset_count=int(msg.reset_counter),
            raw={"source": "ODOMETRY"},
        )
        self.update_latest_telemetry(sample)

    def _on_highres_imu(self, msg: Any) -> None:
        self.latest_imu = {
            "time_boot_us": int(msg.time_usec),
            "acceleration_local_ned_mps2": (float(msg.xacc), float(msg.yacc), float(msg.zacc)),
            "gyro_rps": (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro)),
        }

    def _on_encapsulated_data(self, msg: Any) -> None:
        raw_payload = bytes(msg.data)
        if not raw_payload:
            return
        if raw_payload[0] == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self._on_race_status(raw_payload)
        elif raw_payload[0] == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self._on_track_data_packet(msg, raw_payload)

    def _on_race_status(self, raw_payload: bytes) -> None:
        _, sim_boot_ms, race_start_ms, race_finish_ns, gate_index, last_gate_time = struct.unpack_from(
            "<BQqqIq", raw_payload
        )
        self.race_status = RaceStatus(sim_boot_ms, race_start_ms, race_finish_ns, gate_index, last_gate_time)

    def _on_track_data_packet(self, msg: Any, raw_payload: bytes) -> None:
        _, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self._expected_track_chunks:
            return
        chunks = self._track_chunks[transfer_id]
        chunks[int(msg.seqnr)] = raw_payload[3:]
        expected = self._expected_track_chunks[transfer_id]
        if len(chunks) == expected and all(index in chunks for index in range(expected)):
            payload = b"".join(chunks[index] for index in range(expected))
            del self._track_chunks[transfer_id]
            del self._expected_track_chunks[transfer_id]
            self._on_track_data(payload)

    def _on_track_data(self, payload: bytes) -> None:
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            values = struct.unpack_from("<Hfffffffff", payload)
            payload = payload[38:]
            gates.append(
                TrackGate(
                    gate_id=int(values[0]),
                    position_local_ned_m=tuple(float(value) for value in values[1:4]),
                    quaternion=tuple(float(value) for value in values[4:8]),
                    width_m=float(values[8]),
                    height_m=float(values[9]),
                )
            )
        self.track_gates = gates

    def _on_actuator_output_status(self, msg: Any) -> None:
        self.latest_actuator_output = {
            "time_boot_us": int(msg.time_usec),
            "motor_commands": tuple(float(value) for value in msg.actuator[:4]),
        }

    def _on_collision(self, msg: Any) -> None:
        self.collisions.append(CollisionEvent(int(msg.id), int(msg.threat_level), float(msg.horizontal_minimum_delta)))

    def _time_boot_ms(self) -> int:
        if self._latest_telemetry is not None:
            return self._latest_telemetry.sim_time_ns // 1_000_000
        return 0

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("MAVLink bridge is not connected")
        return self._connection

    @staticmethod
    def _require_mavutil() -> Any:
        from pymavlink import mavutil

        return mavutil
