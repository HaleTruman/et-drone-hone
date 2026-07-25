"""UDP FPV frame receiver with bounded buffering and raw JPEG persistence."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io.udp_protocol import VISION_HEADER, VISION_HEADER_SIZE, unpack_packet

VISION_HEADER_FORMAT = VISION_HEADER.format


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    image: Any | None = None
    saved_path: str | None = None


class VisionStreamReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5600,
        output_dir: str | Path | None = None,
        max_buffered_frames: int = 120,
        max_partial_frames: int = 30,
    ):
        self.host = host
        self.port = int(port)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.max_partial_frames = int(max_partial_frames)
        self._frames: deque[VisionFrame] = deque(maxlen=max_buffered_frames)
        self._partial_frames: dict[int, dict[str, Any]] = {}
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._running = threading.Event()
        self._saving_frames = threading.Event()
        self._lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._saved_frame_paths: dict[int, str] = {}
        self.saved_frame_count = 0
        self.invalid_packet_count = 0
        self.dropped_partial_frame_count = 0

    def start_listener(self) -> None:
        if self._thread is not None:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.25)
        self._socket.bind((self.host, self.port))
        self._running.set()
        self._thread = threading.Thread(target=self._vision_loop, name="vision-rx", daemon=True)
        self._thread.start()
        print("Vision receiver started...")

    def begin_saving_frames(self) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._saving_frames.set()

    def process_packet(self, packet: bytes) -> VisionFrame | None:
        if len(packet) < VISION_HEADER_SIZE:
            self.invalid_packet_count += 1
            return None
        try:
            unpacked = unpack_packet(packet)
        except ValueError:
            self.invalid_packet_count += 1
            return None
        frame_id = unpacked.frame_id
        chunk_id = unpacked.chunk_id
        total_chunks = unpacked.total_chunks
        jpeg_size = unpacked.jpeg_size
        sim_time_ns = unpacked.sim_time_ns
        payload = unpacked.payload
        partial = self._partial_frames.setdefault(
            frame_id,
            {"chunks": {}, "total_chunks": total_chunks, "jpeg_size": jpeg_size, "sim_time_ns": sim_time_ns},
        )
        if partial["total_chunks"] != total_chunks or partial["jpeg_size"] != jpeg_size:
            self.invalid_packet_count += 1
            del self._partial_frames[frame_id]
            return None
        partial["chunks"][chunk_id] = payload
        self._trim_partial_frames()
        if len(partial["chunks"]) != total_chunks:
            return None
        jpeg_bytes = self.reassemble_frame(partial["chunks"][index] for index in range(total_chunks))
        del self._partial_frames[frame_id]
        if len(jpeg_bytes) != jpeg_size:
            self.invalid_packet_count += 1
            return None
        saved_path = self._save_frame(frame_id, sim_time_ns, jpeg_bytes)
        frame = VisionFrame(frame_id, sim_time_ns, jpeg_bytes, saved_path=saved_path)
        self.queue_frame(frame)
        return frame

    def reassemble_frame(self, chunks: Iterable[bytes]) -> bytes:
        return b"".join(chunks)

    def decode_jpeg(self, jpeg_bytes: bytes) -> Any:
        import cv2
        import numpy as np

        return cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    def get_latest_frame(self) -> VisionFrame | None:
        with self._lock:
            if not self._frames:
                return None
            latest = self._frames.pop()

            self._frames.clear()
            return latest

    def get_next_frame(self) -> VisionFrame | None:
        with self._lock:
            return self._frames.popleft() if self._frames else None

    def wait_until_receiving(self, *, timeout_s: float, idle_sleep_s: float = 0.02) -> VisionFrame:
        deadline_s = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline_s:
            with self._lock:
                if self._frames:
                    return self._frames[0]
            time.sleep(idle_sleep_s)
        raise TimeoutError("No vision frame received before timeout.")

    def queue_frame(self, frame: VisionFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def clear_buffer(self) -> None:
        with self._lock:
            self._frames.clear()
            self._partial_frames.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "saving_frames": self._saving_frames.is_set(),
            "saved_frame_count": self.saved_frame_count,
            "buffered_frame_count": len(self._frames),
            "partial_frame_count": len(self._partial_frames),
            "invalid_packet_count": self.invalid_packet_count,
            "dropped_partial_frame_count": self.dropped_partial_frame_count,
        }

    def record_frame_cycle(self, frame_id: int, cycle: int) -> None:
        if self.output_dir is None:
            return
        manifest_path = self._manifest_path()
        if not manifest_path.is_file():
            return
        with self._manifest_lock:
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            updated_lines: list[str] = []
            changed = False
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get("frame_id", -1)) == int(frame_id):
                    record["cycle"] = int(cycle)
                    changed = True
                updated_lines.append(json.dumps(record, separators=(",", ":")))
            if changed:
                manifest_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    def shutdown(self) -> None:
        self._running.clear()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _vision_loop(self) -> None:
        assert self._socket is not None
        while self._running.is_set():
            try:
                packet, _ = self._socket.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            self.process_packet(packet)

    def _save_frame(self, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes) -> str | None:
        if self.output_dir is None or not self._saving_frames.is_set():
            return None
        if frame_id in self._saved_frame_paths:
            return self._saved_frame_paths[frame_id]
        filename = f"frame-{frame_id:08d}-{sim_time_ns}.jpg"
        path = self.output_dir / filename
        path.write_bytes(jpeg_bytes)
        manifest_path = self._manifest_path()
        manifest_record_path = self._manifest_record_path(path, manifest_path)
        metadata = {
            "frame_id": frame_id,
            "sim_time_ns": sim_time_ns,
            "jpeg_size": len(jpeg_bytes),
            "path": manifest_record_path,
        }
        with self._manifest_lock:
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        self._saved_frame_paths[frame_id] = str(path)
        self.saved_frame_count += 1
        return str(path)

    def _manifest_path(self) -> Path:
        assert self.output_dir is not None
        if self.output_dir.name in {"frames", "vision_frames"}:
            return self.output_dir.parent / "frames.jsonl"
        return self.output_dir / "frames.jsonl"

    def _manifest_record_path(self, image_path: Path, manifest_path: Path) -> str:
        try:
            return image_path.relative_to(manifest_path.parent).as_posix()
        except ValueError:
            return image_path.name

    def _trim_partial_frames(self) -> None:
        while len(self._partial_frames) > self.max_partial_frames:
            oldest_frame_id = next(iter(self._partial_frames))
            del self._partial_frames[oldest_frame_id]
            self.dropped_partial_frame_count += 1
