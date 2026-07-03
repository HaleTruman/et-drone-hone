import struct
from types import SimpleNamespace

from sensing.perception.track_gates import TrackGateReceiver
from sensing.telemetry.mavlink_client import MavlinkClient


class Message(SimpleNamespace):
    def get_type(self) -> str:
        return self.message_type


def test_highres_imu_is_cached_without_building_estimated_state() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    client.handle_message(
        Message(
            message_type="HIGHRES_IMU",
            time_usec=1_000_000,
            xacc=1.0,
            yacc=0.0,
            zacc=-9.80665,
            xgyro=0.1,
            ygyro=0.2,
            zgyro=0.3,
            velocity_local_ned_mps=(4.0, 5.0, 6.0),
        )
    )

    assert client.latest_imu is not None
    assert client.latest_imu.acceleration_body_frd_mps2 == (1.0, 0.0, -9.80665)
    assert client.latest_imu.gyro_body_frd_rps == (0.1, 0.2, 0.3)
    assert not hasattr(client.latest_imu, "velocity_local_ned_mps")
    assert not hasattr(client, "latest_estimated_state")

    telemetry = client.get_telemetry()
    assert telemetry is not None
    assert telemetry.sim_time_ns == 1_000_000_000
    assert telemetry.odometry is None
    assert telemetry.imu is client.latest_imu
    assert telemetry.acceleration_body_frd_mps2 == (1.0, 0.0, -9.80665)
    assert telemetry.gyro_body_frd_rps == (0.1, 0.2, 0.3)


def test_wait_until_receiving_returns_imu_telemetry() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    client.connected = True
    client.subscribe_telemetry()
    client.handle_message(
        Message(
            message_type="HIGHRES_IMU",
            time_usec=1_000_000,
            xacc=0.0,
            yacc=0.0,
            zacc=-9.80665,
            xgyro=0.0,
            ygyro=0.0,
            zgyro=0.0,
        )
    )

    telemetry = client.wait_until_receiving(timeout_s=0.01)

    assert telemetry.imu is client.latest_imu


def test_highres_imu_optional_fields_are_cached_when_present() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    client.handle_message(
        Message(
            message_type="HIGHRES_IMU",
            time_usec=1234,
            xacc=0.0,
            yacc=0.0,
            zacc=9.8,
            xgyro=0.1,
            ygyro=0.2,
            zgyro=0.3,
            xmag=1.0,
            ymag=2.0,
            zmag=3.0,
            abs_pressure=4.0,
            diff_pressure=5.0,
            pressure_alt=6.0,
            temperature=7.0,
            fields_updated=8,
            id=9,
        )
    )

    assert client.latest_imu.magnetic_field_gauss == (1.0, 2.0, 3.0)
    assert client.latest_imu.absolute_pressure_hpa == 4.0
    assert client.latest_imu.differential_pressure_hpa == 5.0
    assert client.latest_imu.pressure_altitude_m == 6.0
    assert client.latest_imu.temperature_c == 7.0
    assert client.latest_imu.fields_updated == 8
    assert client.latest_imu.id == 9


def test_send_control_outputs_dispatches_attitude_targets_offline() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    target = {"quaternion": [1.0, 0.0, 0.0, 0.0], "thrust": 0.5}

    client.send_control_outputs(target)

    assert client.latest_attitude_target == target


def test_send_control_outputs_rejects_unknown_output_shape() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")

    try:
        client.send_control_outputs({"motor_commands": [0.1, 0.1, 0.1, 0.1]})
    except ValueError as exc:
        assert "Unsupported control output keys" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_track_messages_are_forwarded_to_track_gate_receiver() -> None:
    receiver = TrackGateReceiver()
    client = MavlinkClient(endpoint="telemetry-simulator", track_gate_receiver=receiver)

    client.handle_message(Message(message_type="DATA_TRANSMISSION_HANDSHAKE", width=44, packets=1))
    client.handle_message(
        Message(
            message_type="ENCAPSULATED_DATA",
            seqnr=0,
            data=bytes([2]) + struct.pack("<H", 44) + struct.pack("<H", 0),
        )
    )

    assert client.track_gates is receiver.track_gates


def test_race_status_and_collision_are_retained() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    client.handle_message(
        Message(message_type="ENCAPSULATED_DATA", data=struct.pack("<BQqqIq", 1, 100, 50, -1, 2, 7))
    )
    client.handle_message(
        Message(message_type="COLLISION", id=1002, threat_level=2, horizontal_minimum_delta=1.25)
    )

    assert client.race_status.active_gate_index == 2
    assert client.collisions[0].collision_id == 1002
