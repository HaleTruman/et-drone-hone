"""MAVLink client and raw message cache for the AI GP simulator."""

import struct
import threading
import time
from dataclasses import asdict
from typing import Any, Callable

from core.coordinates import vec3
from core.schemas import (
    CollisionEvent,
    MavlinkActuatorOutputStatus,
    MavlinkHeartbeat,
    MavlinkHighresImu,
    MavlinkTelemetry,
    MavlinkTimesync,
    RaceStatus,
    RuntimeStatus,
    TrackGate,
)
from sensing.perception.track_gates import TrackGateReceiver

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2
MAVLINK_CMD_SIM_RESET = 31000
MAV_FRAME_LOCAL_NED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
ZERO_VEC3 = (0.0, 0.0, 0.0)


class MavlinkClient:
    """Live MAVLink UDP client with a transport-neutral offline fallback."""

    def __init__(
        self,
        endpoint: str = "udpin:127.0.0.1:14550",
        heartbeat_hz: float = 2.0,
        timesync_hz: float = 10.0,
        connection_factory: Callable[[str], Any] | None = None,
        track_gate_receiver: TrackGateReceiver | None = None,
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
        self.track_gate_receiver = track_gate_receiver or TrackGateReceiver()

        self.connected = False
        self.heartbeat_started = False
        self.telemetry_subscribed = False
        self.armed = False
        self.last_heartbeat_monotonic_s: float | None = None
        self.latest_timesync: MavlinkTimesync | None = None
        self.latest_heartbeat: MavlinkHeartbeat | None = None
        self.latest_imu: MavlinkHighresImu | None = None
        self.latest_actuator_output: MavlinkActuatorOutputStatus | None = None
        self.race_status: RaceStatus | None = None
        self.collisions: list[CollisionEvent] = []
        self.latest_position_target: dict[str, Any] | None = None
        self.latest_attitude_target: dict[str, Any] | None = None
        self._latest_message_monotonic_s: float | None = None

    @property
    def is_live(self) -> bool:
        return self._connection_factory is not None or ":" in self.endpoint

    @property
    def track_gates(self) -> list[TrackGate]:
        return self.track_gate_receiver.track_gates

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
        print("MAVLink client connected...")

    def start_heartbeat(self) -> None:
        """Start periodic TIMESYNC requests used to align client and simulator time."""

        self.heartbeat_started = True
        if not self.is_live or self._timesync_thread is not None:
            return
        self._running.set()
        self._timesync_thread = threading.Thread(target=self._timesync_loop, name="mavlink-timesync", daemon=True)
        self._timesync_thread.start()
        print("MAVLink heartbeat started...")

    def subscribe_telemetry(self) -> None:
        self.telemetry_subscribed = True
        if not self.is_live or self._receiver_thread is not None:
            return
        self._running.set()
        self._receiver_thread = threading.Thread(target=self._receive_loop, name="mavlink-rx", daemon=True)
        self._receiver_thread.start()
        print("MAVLink subscribed to telemetry...")

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
        velocity = target.get("velocity_local_ned_mps")
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
        if velocity is None:
            velocity = ZERO_VEC3
            mask |= (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            )
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

    def send_control_outputs(self, control_outputs: dict[str, Any]) -> None:
        if "quaternion" in control_outputs and "thrust" in control_outputs:
            self.send_attitude_target(control_outputs)
            return
        if "velocity_local_ned_mps" in control_outputs:
            self.send_position_target(control_outputs)
            return
        raise ValueError(f"Unsupported control output keys: {sorted(control_outputs)}")

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

    def wait_for_imu_telemetry(self, *, timeout_s: float, idle_sleep_s: float = 0.02) -> MavlinkTelemetry:
        deadline_s = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline_s:
            telemetry = self.get_telemetry()
            if telemetry is not None and telemetry.imu is not None:
                return telemetry
            time.sleep(idle_sleep_s)
        raise TimeoutError("No HIGHRES_IMU telemetry received before timeout.")

    def wait_until_receiving(self, *, timeout_s: float, idle_sleep_s: float = 0.02) -> MavlinkTelemetry:
        if not self.connected:
            raise RuntimeError("MAVLink client is not connected")
        if not self.telemetry_subscribed:
            raise RuntimeError("MAVLink telemetry has not been subscribed")
        return self.wait_for_imu_telemetry(timeout_s=timeout_s, idle_sleep_s=idle_sleep_s)

    def arm_and_wait(self, *, timeout_s: float = 5.0) -> None:
        self.arm()
        deadline_s = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline_s:
            if self.armed:
                return
            time.sleep(0.02)
        raise TimeoutError("Simulator did not confirm armed state before timeout.")

    def get_telemetry(self) -> MavlinkTelemetry | None:
        """Build a raw telemetry bundle from cached MAVLink messages."""

        if self.latest_imu is None:
            return None

        sim_time_ns = self._latest_sample_time_ns()
        return MavlinkTelemetry(
            sim_time_ns=sim_time_ns,
            odometry=None,
            imu=self.latest_imu,
            system_status=self._latest_system_status(),
            reset_count=None,
            raw={"source": "mavlink_client"},
        )

    def status(self) -> RuntimeStatus:
        now = time.monotonic()
        message_age_s = None if self._latest_message_monotonic_s is None else now - self._latest_message_monotonic_s
        heartbeat_age_s = None if self.last_heartbeat_monotonic_s is None else now - self.last_heartbeat_monotonic_s
        return RuntimeStatus(
            connected=self.connected,
            running=self._running.is_set(),
            heartbeat_started=self.heartbeat_started,
            telemetry_subscribed=self.telemetry_subscribed,
            armed=self.armed,
            latest_message_age_s=message_age_s,
            latest_heartbeat_age_s=heartbeat_age_s,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "connected": self.connected,
            "armed": self.armed,
            "status": asdict(self.status()),
            "latest_heartbeat": self._snapshot_value(self.latest_heartbeat),
            "latest_timesync": self._snapshot_value(self.latest_timesync),
            "latest_imu": self._snapshot_value(self.latest_imu),
            "latest_actuator_output": self._snapshot_value(self.latest_actuator_output),
            "race_status": asdict(self.race_status) if self.race_status else None,
            "track_gates": [asdict(gate) for gate in self.track_gates],
            "collisions": [asdict(collision) for collision in self.collisions],
            "latest_position_target": self.latest_position_target,
            "latest_attitude_target": self.latest_attitude_target,
        }

    def handle_message(self, msg: Any) -> None:
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            self._on_heartbeat(msg)
        elif msg_type == "TIMESYNC":
            self._on_timesync(msg)
        elif msg_type == "HIGHRES_IMU":
            self._on_highres_imu(msg)
        elif msg_type == "ENCAPSULATED_DATA":
            self._on_encapsulated_data(msg)
        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            self._on_actuator_output_status(msg)
        elif msg_type == "COLLISION":
            self._on_collision(msg)
        elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            self.track_gate_receiver.handle_handshake(transfer_id=int(msg.width), packets=int(msg.packets))

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
        self.armed = bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
        self.last_heartbeat_monotonic_s = time.monotonic()
        self.latest_heartbeat = MavlinkHeartbeat(
            type=getattr(msg, "type", None),
            autopilot=getattr(msg, "autopilot", None),
            base_mode=int(msg.base_mode),
            custom_mode=getattr(msg, "custom_mode", None),
            system_status=getattr(msg, "system_status", None),
            mavlink_version=getattr(msg, "mavlink_version", None),
        )
        self._mark_message_received()

    def _on_timesync(self, msg: Any) -> None:
        self.latest_timesync = MavlinkTimesync(ts1=int(msg.ts1), tc1=int(msg.tc1))
        self._mark_message_received()

    def _on_highres_imu(self, msg: Any) -> None:
        magnetic_field_gauss = None
        if all(hasattr(msg, axis) for axis in ("xmag", "ymag", "zmag")):
            magnetic_field_gauss = vec3((float(msg.xmag), float(msg.ymag), float(msg.zmag)))
        self.latest_imu = MavlinkHighresImu(
            time_boot_us=int(msg.time_usec),
            acceleration_body_frd_mps2=vec3((float(msg.xacc), float(msg.yacc), float(msg.zacc))),
            gyro_body_frd_rps=vec3((float(msg.xgyro), float(msg.ygyro), float(msg.zgyro))),
            magnetic_field_gauss=magnetic_field_gauss,
            absolute_pressure_hpa=None if not hasattr(msg, "abs_pressure") else float(msg.abs_pressure),
            differential_pressure_hpa=None if not hasattr(msg, "diff_pressure") else float(msg.diff_pressure),
            pressure_altitude_m=None if not hasattr(msg, "pressure_alt") else float(msg.pressure_alt),
            temperature_c=None if not hasattr(msg, "temperature") else float(msg.temperature),
            fields_updated=None if not hasattr(msg, "fields_updated") else int(msg.fields_updated),
            id=None if not hasattr(msg, "id") else int(msg.id),
        )
        self._mark_message_received()

    def _on_encapsulated_data(self, msg: Any) -> None:
        raw_payload = bytes(msg.data)
        if not raw_payload:
            return
        if raw_payload[0] == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self._on_race_status(raw_payload)
        elif raw_payload[0] == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self.track_gate_receiver.handle_packet(seqnr=int(msg.seqnr), raw_payload=raw_payload)

    def _on_race_status(self, raw_payload: bytes) -> None:
        _, sim_boot_ms, race_start_ms, race_finish_ns, gate_index, last_gate_time = struct.unpack_from(
            "<BQqqIq", raw_payload
        )
        self.race_status = RaceStatus(sim_boot_ms, race_start_ms, race_finish_ns, gate_index, last_gate_time)

    def _on_actuator_output_status(self, msg: Any) -> None:
        self.latest_actuator_output = MavlinkActuatorOutputStatus(
            time_boot_us=int(msg.time_usec),
            active=int(msg.active),
            actuator=tuple(float(value) for value in msg.actuator),
        )
        self._mark_message_received()

    def _on_collision(self, msg: Any) -> None:
        self.collisions.append(CollisionEvent(int(msg.id), int(msg.threat_level), float(msg.horizontal_minimum_delta)))
        self._mark_message_received()

    def _time_boot_ms(self) -> int:
        if self.latest_imu is not None:
            return self.latest_imu.time_boot_us // 1_000
        return 0

    def _latest_sample_time_ns(self) -> int:
        if self.latest_imu is not None:
            return self.latest_imu.time_boot_us * 1_000
        return 0

    def _latest_system_status(self) -> str | None:
        if self.latest_heartbeat is None or self.latest_heartbeat.system_status is None:
            return None
        return str(self.latest_heartbeat.system_status)

    def _mark_message_received(self) -> None:
        self._latest_message_monotonic_s = time.monotonic()

    @staticmethod
    def _snapshot_value(value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return value

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("MAVLink client is not connected")
        return self._connection

    @staticmethod
    def _require_mavutil() -> Any:
        from pymavlink import mavutil

        return mavutil
