"""Authoritative track-gate packet assembly and parsing."""

import struct
import time

from core.schemas import TrackGate


class TrackGateReceiver:
    """Reassemble simulator track-gate packets into TrackGate records."""

    def __init__(self) -> None:
        self.track_gates: list[TrackGate] = []
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._expected_track_chunks: dict[int, int] = {}

    def handle_handshake(self, *, transfer_id: int, packets: int) -> None:
        self._track_chunks[int(transfer_id)] = {}
        self._expected_track_chunks[int(transfer_id)] = int(packets)

    def handle_packet(self, *, seqnr: int, raw_payload: bytes) -> list[TrackGate] | None:
        _, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self._expected_track_chunks:
            return None
        chunks = self._track_chunks[transfer_id]
        chunks[int(seqnr)] = raw_payload[3:]
        expected = self._expected_track_chunks[transfer_id]
        if len(chunks) != expected or not all(index in chunks for index in range(expected)):
            return None

        payload = b"".join(chunks[index] for index in range(expected))
        del self._track_chunks[transfer_id]
        del self._expected_track_chunks[transfer_id]
        self.track_gates = self.parse_track_data(payload)
        return list(self.track_gates)

    def wait_for_track_gates(self, *, timeout_s: float, idle_sleep_s: float = 0.02) -> list[TrackGate]:
        deadline_s = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline_s:
            if self.track_gates:
                return list(self.track_gates)
            time.sleep(idle_sleep_s)
        return []

    @staticmethod
    def parse_track_data(payload: bytes) -> list[TrackGate]:
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
        return gates
