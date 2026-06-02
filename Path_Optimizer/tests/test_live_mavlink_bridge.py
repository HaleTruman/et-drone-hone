import struct
from types import SimpleNamespace

import numpy as np

from sensing.perception.gate_map import GateMap
from sensing.telemetry.mavlink_bridge import MavlinkBridge


class Message(SimpleNamespace):
    def get_type(self) -> str:
        return self.message_type


def test_odometry_is_stored_as_live_telemetry_and_starting_attitude() -> None:
    bridge = MavlinkBridge(endpoint="telemetry-simulator")
    bridge.handle_message(
        Message(
            message_type="ODOMETRY",
            time_usec=1234,
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
            reset_counter=7,
        )
    )

    telemetry = bridge.get_latest_telemetry()
    assert telemetry is not None
    assert telemetry.sim_time_ns == 1_234_000
    assert telemetry.position_local_ned_m == (1.0, 2.0, -3.0)
    assert telemetry.attitude == (1.0, 0.0, 0.0, 0.0)
    assert bridge.starting_telemetry == telemetry


def test_track_chunks_are_reassembled_and_can_seed_gate_map() -> None:
    bridge = MavlinkBridge(endpoint="telemetry-simulator")
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
    bridge.handle_message(Message(message_type="DATA_TRANSMISSION_HANDSHAKE", width=44, packets=2))
    split = len(track_payload) // 2
    bridge.handle_message(
        Message(message_type="ENCAPSULATED_DATA", seqnr=0, data=bytes([2]) + struct.pack("<H", 44) + track_payload[:split])
    )
    bridge.handle_message(
        Message(message_type="ENCAPSULATED_DATA", seqnr=1, data=bytes([2]) + struct.pack("<H", 44) + track_payload[split:])
    )

    assert len(bridge.track_gates) == 1
    assert bridge.track_gates[0].gate_id == 9
    gate_map = GateMap()
    bridge.populate_gate_map(gate_map)
    np.testing.assert_allclose(gate_map.get_gate("9").position_local_ned_m, [1.0, 2.0, -3.0])


def test_race_status_and_collision_are_retained() -> None:
    bridge = MavlinkBridge(endpoint="telemetry-simulator")
    bridge.handle_message(
        Message(message_type="ENCAPSULATED_DATA", data=struct.pack("<BQqqIq", 1, 100, 50, -1, 2, 7))
    )
    bridge.handle_message(
        Message(message_type="COLLISION", id=1002, threat_level=2, horizontal_minimum_delta=1.25)
    )

    assert bridge.race_status.active_gate_index == 2
    assert bridge.collisions[0].collision_id == 1002
