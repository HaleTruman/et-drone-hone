from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    image: Any | None = None


class VisionStreamReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5600):
        self.host = host
        self.port = port
        self._frames: deque[VisionFrame] = deque()

    def start_listener(self) -> None:
        raise NotImplementedError("Implement UDP packet reception and frame buffering.")

    def reassemble_frame(self, chunks: Iterable[bytes]) -> bytes:
        return b"".join(chunks)

    def decode_jpeg(self, jpeg_bytes: bytes) -> Any:
        raise NotImplementedError("Choose the JPEG decoder used by the perception pipeline.")

    def get_next_frame(self) -> VisionFrame | None:
        return self._frames.popleft() if self._frames else None

    def queue_frame(self, frame: VisionFrame) -> None:
        self._frames.append(frame)

    def shutdown(self) -> None:
        pass
