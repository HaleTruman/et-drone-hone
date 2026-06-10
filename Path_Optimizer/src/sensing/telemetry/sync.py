from collections import deque
from typing import Any, Iterable

from core.schemas import TelemetrySample
from sensing.perception import VisionObservation
from sensing.vision.vision_stream import VisionFrame


class DataSynchronizer:
    def __init__(self, max_delta_ns: int = 50_000_000):
        self.max_delta_ns = max_delta_ns
        self._synchronized: deque[tuple[Any, TelemetrySample]] = deque()

    def align_frame_with_telemetry(
        self,
        frame: VisionFrame,
        telemetry: Iterable[TelemetrySample],
    ) -> tuple[VisionFrame, TelemetrySample] | None:
        return self._align_by_sim_time(frame, telemetry)

    def get_synchronized_data(self) -> tuple[Any, TelemetrySample] | None:
        return self._synchronized.popleft() if self._synchronized else None

    def align_observation_with_telemetry(
        self,
        observation: VisionObservation,
        telemetry: Iterable[TelemetrySample],
    ) -> tuple[VisionObservation, TelemetrySample] | None:
        return self._align_by_sim_time(observation, telemetry)

    def _align_by_sim_time(
        self,
        item: Any,
        telemetry: Iterable[TelemetrySample],
    ) -> tuple[Any, TelemetrySample] | None:
        samples = list(telemetry)
        if not samples:
            return None
        closest = min(samples, key=lambda sample: abs(sample.sim_time_ns - item.sim_time_ns))
        if abs(closest.sim_time_ns - item.sim_time_ns) > self.max_delta_ns:
            return None
        pair = (item, closest)
        self._synchronized.append(pair)
        return pair
