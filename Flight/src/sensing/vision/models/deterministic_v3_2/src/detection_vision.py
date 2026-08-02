"""The live entrypoint: one VisionFrame in, one VisionObservation out.

Holds the per-stream state the service cannot keep for us -- the colour LUT, the
instance tracker, and gate identity -- and reuses the same VoidDetector and
GatePublisher the offline path uses, so live and replay share one implementation.

Mirrors deterministic_v3_2's DeterministicVision so the service treats every
backend identically.
"""
from .here_gate import GatePublisher
from .void_detection import LUT_PATH, VoidDetector

class DetectionVision:
    def __init__(self, lut_path: str = LUT_PATH, run_id: str = "live") -> None:
        self._detector = VoidDetector(run_id=run_id, lut_path=lut_path)
        self._publisher = GatePublisher()

    def process_frame(self, frame, vehicle_state=None):
        detections = self._detector.process_frame(frame, vehicle_state)
        return self._publisher.observe(detections, vehicle_state)
