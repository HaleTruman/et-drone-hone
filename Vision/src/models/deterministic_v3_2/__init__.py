"""Public interface for the deterministic_v3_2 vision backend."""

__all__ = ["DeterministicVision", "DetectionVision"]


def __getattr__(name):
    if name in {"DeterministicVision", "DetectionVision"}:
        from .src.detection_vision import DetectionVision
        return DetectionVision
    raise AttributeError(name)
