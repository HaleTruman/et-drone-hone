"""Replay local JPEG frames as the official UDP vision stream."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path

from vision.src.io.udp_protocol import (
    DEFAULT_CHUNK_PAYLOAD_BYTES,
    DEFAULT_FPS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    chunk_jpeg,
    frame_id_from_path,
    pack_packet,
    sorted_jpeg_paths,
)


@dataclass(frozen=True)
class ShimConfig:
    frames_dir: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    fps: float = DEFAULT_FPS
    chunk_payload_bytes: int = DEFAULT_CHUNK_PAYLOAD_BYTES
    max_frames: int = 0


@dataclass(frozen=True)
class ShimStats:
    frames_sent: int
    packets_sent: int
    bytes_sent: int


def send_sample_frames(config: ShimConfig) -> ShimStats:
    frames_dir = Path(config.frames_dir).expanduser()
    paths = sorted_jpeg_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png frames found in {frames_dir}")
    if config.max_frames > 0:
        paths = paths[: config.max_frames]

    period_s = 1.0 / float(config.fps) if float(config.fps) > 0.0 else 0.0
    sim_period_ns = int(round(1_000_000_000.0 * period_s)) if period_s > 0.0 else 0
    packets_sent = 0
    bytes_sent = 0
    start_time = time.monotonic()
    address = (str(config.host), int(config.port))

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for sequence_index, path in enumerate(paths):
            if period_s > 0.0:
                target_time = start_time + (sequence_index * period_s)
                sleep_s = target_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            jpeg_bytes = path.read_bytes()
            frame_id = frame_id_from_path(path)
            sim_time_ns = sequence_index * sim_period_ns
            for packet in chunk_jpeg(frame_id, jpeg_bytes, sim_time_ns, config.chunk_payload_bytes):
                data = pack_packet(packet)
                bytes_sent += sock.sendto(data, address)
                packets_sent += 1
    return ShimStats(frames_sent=len(paths), packets_sent=packets_sent, bytes_sent=bytes_sent)
