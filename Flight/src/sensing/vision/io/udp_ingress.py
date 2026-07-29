"""Receive and reassemble UDP JPEG frames from the vision stream."""

from __future__ import annotations

import socket
from contextlib import closing
from dataclasses import dataclass
from typing import Iterator

from core.schema import VisionFrame
from sensing.vision.io.udp_protocol import DEFAULT_HOST, DEFAULT_PORT, VisionPacket, unpack_packet


@dataclass(frozen=True)
class IngressConfig:
    bind_host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_s: float | None = None
    max_frames: int = 0
    recv_bytes: int = 65535
    max_pending_frames: int = 64


class FrameAccumulator:
    def __init__(self, max_pending_frames: int = 64) -> None:
        self.max_pending_frames = max(1, int(max_pending_frames))
        self._pending: dict[int, dict[str, object]] = {}

    def add_packet(self, packet: VisionPacket) -> VisionFrame | None:
        frame = self._pending.get(packet.frame_id)
        if frame is None:
            if len(self._pending) >= self.max_pending_frames:
                oldest_frame_id = next(iter(self._pending))
                self._pending.pop(oldest_frame_id, None)
            frame = {
                "jpeg_size": packet.jpeg_size,
                "sim_time_ns": packet.sim_time_ns,
                "chunks": [None] * packet.total_chunks,
            }
            self._pending[packet.frame_id] = frame

        if int(frame["jpeg_size"]) != packet.jpeg_size or int(frame["sim_time_ns"]) != packet.sim_time_ns:
            self._pending.pop(packet.frame_id, None)
            raise ValueError(f"Inconsistent metadata for frame {packet.frame_id}.")

        chunks = frame["chunks"]
        if not isinstance(chunks, list) or len(chunks) != packet.total_chunks:
            self._pending.pop(packet.frame_id, None)
            raise ValueError(f"Inconsistent chunk count for frame {packet.frame_id}.")
        chunks[packet.chunk_id] = packet.payload
        if any(chunk is None for chunk in chunks):
            return None

        jpeg_bytes = b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))
        self._pending.pop(packet.frame_id, None)
        if len(jpeg_bytes) != packet.jpeg_size:
            raise ValueError(f"Reconstructed JPEG size mismatch for frame {packet.frame_id}.")
        return VisionFrame(frame_id=packet.frame_id, sim_time_ns=packet.sim_time_ns, jpeg_bytes=jpeg_bytes)


def open_ingress_socket(config: IngressConfig) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((str(config.bind_host), int(config.port)))
    if config.timeout_s is not None:
        sock.settimeout(float(config.timeout_s))
    return sock


def iter_jpeg_frames_from_socket(sock: socket.socket, config: IngressConfig) -> Iterator[VisionFrame]:
    accumulator = FrameAccumulator(config.max_pending_frames)
    yielded = 0
    while config.max_frames <= 0 or yielded < config.max_frames:
        try:
            data, _address = sock.recvfrom(int(config.recv_bytes))
        except socket.timeout:
            return
        packet = unpack_packet(data)
        frame = accumulator.add_packet(packet)
        if frame is None:
            continue
        yielded += 1
        yield frame


def iter_jpeg_frames(config: IngressConfig) -> Iterator[VisionFrame]:
    with closing(open_ingress_socket(config)) as sock:
        yield from iter_jpeg_frames_from_socket(sock, config)
