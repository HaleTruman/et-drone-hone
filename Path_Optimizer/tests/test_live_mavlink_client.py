import struct
from types import SimpleNamespace

import numpy as np

from sensing.perception.gate_map import GateMap
from sensing.telemetry.mavlink_client import MAV_FRAME_BODY_FRD, MAV_FRAME_LOCAL_NED, MavlinkClient


class Message(SimpleNamespace):
    def get_type(self) -> str:
        return self.message_type


def test_raw_messages_do_not_inject_cross_message_fields() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    client.handle_message(
        Message(
            message_type="ATTITUDE",
            time_boot_ms=1,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            rollspeed=0.1,
            pitchspeed=0.2,
            yawspeed=0.3,
        )
    )
    client.handle_message(
        Message(
            message_type="ODOMETRY",
            time_usec=1234,
            frame_id=MAV_FRAME_LOCAL_NED,
            child_frame_id=MAV_FRAME_BODY_FRD,
            x=1.0,
            y=2.0,
            z=-3.0,
            q=[1.0, 0.0, 0.0, 0.0],
            vx=4.0,
            vy=5.0,
            vz=6.0,
            rollspeed=0.1,
            pitchspeed=0.2,
            yawspeed=0.3,
            pose_covariance=[0.0] * 21,
            velocity_covariance=[0.0] * 21,
            reset_counter=7,
            estimator_type=1,
        )
    )
    client.handle_message(
        Message(
            message_type="LOCAL_POSITION_NED",
            time_boot_ms=1,
            x=10.0,
            y=20.0,
            z=-30.0,
            vx=40.0,
            vy=50.0,
            vz=60.0,
        )
    )
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
            velocity_local_ned_mps=(4.0, 5.0, 6.0),
        )
    )

    assert client.latest_imu is not None
    assert client.latest_imu.acceleration_body_frd_mps2 == (0.0, 0.0, 9.8)
    assert client.latest_imu.gyro_body_frd_rps == (0.1, 0.2, 0.3)
    assert not hasattr(client.latest_imu, "velocity_local_ned_mps")

    assert client.latest_odometry is not None
    assert client.latest_odometry.position_local_ned_m == (1.0, 2.0, -3.0)
    assert client.latest_odometry.velocity_local_ned_mps == (4.0, 5.0, 6.0)
    assert client.latest_odometry.angular_velocity_body_frd_rps == (0.1, 0.2, 0.3)
    assert not hasattr(client.latest_odometry, "acceleration_local_ned_mps2")
    assert not hasattr(client.latest_odometry, "acceleration_body_frd_mps2")

    assert client.latest_attitude is not None
    assert not hasattr(client.latest_attitude, "attitude_quaternion")

    telemetry = client.get_latest_telemetry()
    assert telemetry is not None
    assert telemetry.sim_time_ns == 1_234_000
    assert telemetry.position_local_ned_m == (1.0, 2.0, -3.0)
    assert telemetry.diagnostic_odometry["position_local_ned_m"] == client.latest_odometry.position_local_ned_m


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


def test_attitude_cache_stores_raw_message_fields_only() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")

    client.handle_message(
        Message(
            message_type="ATTITUDE",
            time_boot_ms=1,
            roll=0.2,
            pitch=-0.3,
            yaw=0.4,
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
        )
    )

    assert client.latest_attitude.time_boot_ms == 1
    assert client.latest_attitude.roll_rad == 0.2
    assert client.latest_attitude.pitch_rad == -0.3
    assert client.latest_attitude.yaw_rad == 0.4
    assert client.latest_attitude.angular_velocity_body_frd_rps == (0.0, 0.0, 0.0)
    assert client.get_latest_telemetry() is None


def test_track_chunks_are_reassembled_and_can_seed_gate_map() -> None:
    client = MavlinkClient(endpoint="telemetry-simulator")
    track_payload = struct.pack(
        "<HHfffffffff",
        1,
        9,
        1.0,
        2.0,
        -3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        2.7,
        1.5,
    )
    client.handle_message(Message(message_type="DATA_TRANSMISSION_HANDSHAKE", width=44, packets=2))
    split = len(track_payload) // 2
    client.handle_message(
        Message(message_type="ENCAPSULATED_DATA", seqnr=0, data=bytes([2]) + struct.pack("<H", 44) + track_payload[:split])
    )
    client.handle_message(
        Message(message_type="ENCAPSULATED_DATA", seqnr=1, data=bytes([2]) + struct.pack("<H", 44) + track_payload[split:])
    )

    assert len(client.track_gates) == 1
    assert client.track_gates[0].gate_id == 9
    gate_map = GateMap()
    client.populate_gate_map(gate_map)
    np.testing.assert_allclose(gate_map.get_gate("9").position_local_ned_m, [1.0, 2.0, -3.0])


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
