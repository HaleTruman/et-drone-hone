import struct

import numpy as np

from sensing.perception.gate_map import GateMap
from sensing.perception.track_gates import TrackGateReceiver


def test_track_chunks_are_reassembled_and_can_seed_gate_map() -> None:
    receiver = TrackGateReceiver()
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
    receiver.handle_handshake(transfer_id=44, packets=2)
    split = len(track_payload) // 2
    assert receiver.handle_packet(seqnr=0, raw_payload=bytes([2]) + struct.pack("<H", 44) + track_payload[:split]) is None
    gates = receiver.handle_packet(seqnr=1, raw_payload=bytes([2]) + struct.pack("<H", 44) + track_payload[split:])

    assert gates is not None
    assert len(receiver.track_gates) == 1
    assert receiver.track_gates[0].gate_id == 9
    gate_map = GateMap()
    assert gate_map.seed_from_track_gates(receiver.track_gates)
    np.testing.assert_allclose(gate_map.get_gate("9").position_local_ned_m, [1.0, 2.0, -3.0])
    assert gate_map.get_gate("9").source == "track"


def test_wait_for_track_gates_returns_empty_on_timeout() -> None:
    receiver = TrackGateReceiver()

    assert receiver.wait_for_track_gates(timeout_s=0.0) == []
