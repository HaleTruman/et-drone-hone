from collections import deque
from typing import Iterable

from .mavlink_bridge import TelemetrySample
from .vision_stream import VisionFrame


class DataSynchronizer:
    def __init__(self, max_delta_ns: int = 50_000_000):
        self.max_delta_ns = max_delta_ns
        self._synchronized: deque[tuple[VisionFrame, TelemetrySample]] = deque()

    def align_frame_with_telemetry(
        self,
        frame: VisionFrame,
        telemetry: Iterable[TelemetrySample],
    ) -> tuple[VisionFrame, TelemetrySample] | None:
        samples = list(telemetry)
        if not samples:
            return None
        closest = min(samples, key=lambda sample: abs(sample.sim_time_ns - frame.sim_time_ns))
        if abs(closest.sim_time_ns - frame.sim_time_ns) > self.max_delta_ns:
            return None
        pair = (frame, closest)
        self._synchronized.append(pair)
        return pair

    def get_synchronized_data(self) -> tuple[VisionFrame, TelemetrySample] | None:
        return self._synchronized.popleft() if self._synchronized else None
