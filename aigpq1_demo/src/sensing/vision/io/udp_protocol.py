"""Shared packet helpers for the 4.6 UDP JPEG vision stream."""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


VISION_HEADER = struct.Struct("<IHHIIQ")
VISION_HEADER_SIZE = VISION_HEADER.size
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5600
DEFAULT_FPS = 30.0
DEFAULT_CHUNK_PAYLOAD_BYTES = 1200
JPEG_FRAME_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class VisionPacket:
    frame_id: int
    chunk_id: int
    total_chunks: int
    jpeg_size: int
    payload_size: int
    sim_time_ns: int
    payload: bytes


def frame_id_from_path(path: Path) -> int:
    match = re.search(r"(?:^|_)frame_(\d+)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|_)frame_(\d+)", path.parent.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot derive frame id from filename: {path}")


def sorted_jpeg_paths(frames_dir: Path) -> list[Path]:
    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = []
        for frame in manifest.get("frames", []):
            rgb_path = frame.get("rgb_path") or frame.get("rgb")
            if not rgb_path:
                continue
            path = frames_dir / rgb_path
            if path.exists():
                paths.append(path)
        return sorted(paths, key=lambda path: (frame_id_from_path(path), path.name))

    paths = [
        path
        for path in frames_dir.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in JPEG_FRAME_SUFFIXES
            or (path.suffix.lower() == ".png" and path.stem.endswith("_rgb"))
        )
    ]
    return sorted(paths, key=lambda path: (frame_id_from_path(path), path.name))


def pack_packet(packet: VisionPacket) -> bytes:
    header = VISION_HEADER.pack(
        packet.frame_id,
        packet.chunk_id,
        packet.total_chunks,
        packet.jpeg_size,
        packet.payload_size,
        packet.sim_time_ns,
    )
    return header + packet.payload


def unpack_packet(data: bytes) -> VisionPacket:
    if len(data) < VISION_HEADER_SIZE:
        raise ValueError(f"UDP packet is shorter than {VISION_HEADER_SIZE} byte vision header.")
    frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = VISION_HEADER.unpack_from(data)
    payload = data[VISION_HEADER_SIZE:]
    if payload_size != len(payload):
        raise ValueError(f"Payload size mismatch: header={payload_size} actual={len(payload)}.")
    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive.")
    if chunk_id >= total_chunks:
        raise ValueError(f"chunk_id {chunk_id} is outside total_chunks {total_chunks}.")
    if jpeg_size <= 0:
        raise ValueError("jpeg_size must be positive.")
    return VisionPacket(
        frame_id=frame_id,
        chunk_id=chunk_id,
        total_chunks=total_chunks,
        jpeg_size=jpeg_size,
        payload_size=payload_size,
        sim_time_ns=sim_time_ns,
        payload=payload,
    )


def chunk_jpeg(
    frame_id: int,
    jpeg_bytes: bytes,
    sim_time_ns: int,
    chunk_payload_bytes: int = DEFAULT_CHUNK_PAYLOAD_BYTES,
) -> list[VisionPacket]:
    if not jpeg_bytes:
        raise ValueError("JPEG bytes are empty.")
    if chunk_payload_bytes <= 0:
        raise ValueError("chunk_payload_bytes must be positive.")
    total_chunks = int(math.ceil(len(jpeg_bytes) / float(chunk_payload_bytes)))
    if total_chunks > 65535:
        raise ValueError(f"JPEG requires {total_chunks} chunks, which exceeds uint16.")
    packets: list[VisionPacket] = []
    for chunk_id in range(total_chunks):
        start = chunk_id * chunk_payload_bytes
        payload = jpeg_bytes[start : start + chunk_payload_bytes]
        packets.append(
            VisionPacket(
                frame_id=int(frame_id),
                chunk_id=int(chunk_id),
                total_chunks=total_chunks,
                jpeg_size=len(jpeg_bytes),
                payload_size=len(payload),
                sim_time_ns=int(sim_time_ns),
                payload=payload,
            )
        )
    return packets
