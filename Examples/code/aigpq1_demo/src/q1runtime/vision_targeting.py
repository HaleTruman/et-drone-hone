from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from Examples.aigpq1_demo.src.sensing.perception import VisionGateObservation, VisionObservation
from Examples.aigpq1_demo.src.sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService
from Examples.aigpq1_demo.src.sensing.vision.vision_stream import VisionFrame

from .config import Q1RuntimeConfig


@dataclass
class Q1VisionTargeter:
    """Runs the vision stack and selects the nearest usable top-1 target."""

    service: VisionPerceptionService
    min_position_confidence: float = 0.50

    @classmethod
    def from_config(cls, config: Q1RuntimeConfig) -> "Q1VisionTargeter":
        service = VisionPerceptionService(
            VisionPerceptionConfig(
                device=config.vision_device,
                run_landmarker=config.run_landmarker,
                top_k=config.vision_candidate_top_k,
                passthrough_regressor_targets=config.passthrough_regressor_targets,
                gate_threshold=config.gate_threshold,
                confidence_threshold=config.confidence_threshold,
                min_component_area=config.min_component_area,
                max_candidates=config.max_candidates,
            )
        )
        return cls(service=service, min_position_confidence=config.min_position_confidence)

    def process_frame(self, frame: VisionFrame) -> VisionObservation:
        return self.service.process_frame(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            jpeg_bytes=frame.jpeg_bytes,
        )

    def select_top_gate(self, observation: VisionObservation) -> VisionGateObservation | None:
        candidates = self.candidate_gates(observation)
        if not candidates:
            return None
        candidates.sort(key=self._camera_distance_m)
        return candidates[0]

    def candidate_gates(self, observation: VisionObservation) -> list[VisionGateObservation]:
        return [
            gate
            for gate in observation.gates
            if float(gate.position_confidence) >= float(self.min_position_confidence)
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "min_position_confidence": float(self.min_position_confidence),
            "service": self.service.snapshot(),
        }

    @staticmethod
    def _camera_distance_m(gate: VisionGateObservation) -> float:
        return float(np.linalg.norm(np.asarray(gate.position_camera_m, dtype=float)))
