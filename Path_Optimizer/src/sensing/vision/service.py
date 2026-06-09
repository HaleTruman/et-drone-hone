from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sensing.perception import VisionGateObservation, VisionObservation
from sensing.vision.cnn.rgb_inference import DEFAULT_CHECKPOINT, LightmaskInference
from sensing.vision.cnn.rgb_normalizer import jpeg_bytes_to_tensor
from sensing.vision.landmarker.landmark_output import build_controller_payload, build_passthrough_controller_payload
from sensing.vision.landmarker.landmarker import Landmarker, new_landmarker_state
from sensing.vision.regressor.cnn_ingress import RawLogitsFrame
from sensing.vision.regressor.logit_inference import DEFAULT_REGRESSOR_CHECKPOINT, LogitRegressor
from sensing.vision.regressor.regression_output import build_surveyer_payload


@dataclass(frozen=True)
class VisionPerceptionConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    regressor_checkpoint: Path = DEFAULT_REGRESSOR_CHECKPOINT
    device: str = "auto"
    run_landmarker: bool = True
    top_k: int = 5
    passthrough_regressor_targets: bool = False
    gate_threshold: float = 0.50
    confidence_threshold: float = 0.50
    min_component_area: int = 3
    max_candidates: int = 32


class VisionPerceptionService:
    """In-memory CNN -> regressor -> optional landmarker composition for flight."""

    def __init__(
        self,
        config: VisionPerceptionConfig | None = None,
        *,
        cnn: LightmaskInference | None = None,
        regressor: LogitRegressor | None = None,
        landmarker: Landmarker | None = None,
    ) -> None:
        self.config = config or VisionPerceptionConfig()
        self._cnn = cnn
        self._regressor = regressor
        self._landmarker = landmarker

    def process_frame(self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes) -> VisionObservation:
        tensor = jpeg_bytes_to_tensor(jpeg_bytes)
        cnn_result = self.cnn.run_frame(tensor)
        raw_logits = RawLogitsFrame(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            mask_logits=cnn_result.mask_logits,
            depth_logits=cnn_result.depth_logits,
        )
        regression = self.regressor.run_frame(raw_logits)
        regressor_payload = build_surveyer_payload(regression, output_dir=Path("memory"))

        if self.config.run_landmarker:
            if self.config.passthrough_regressor_targets:
                # Passthrough preserves controller output shape while skipping landmark state/matching.
                controller_payload = build_passthrough_controller_payload(
                    regressor_payload,
                    output_dir=Path("memory"),
                    top_k=int(self.config.top_k),
                )
                return VisionObservation.from_controller_payload(controller_payload, source="cnn_regressor_passthrough")

            update_result = self.landmarker.update_frame(regressor_payload)
            controller_payload = build_controller_payload(
                self.landmarker.state,
                run=regressor_payload.get("run") or {},
                output_dir=Path("memory"),
                top_k=int(self.config.top_k),
                current_gates=update_result["current_gates"],
            )
            return VisionObservation.from_controller_payload(controller_payload, source="cnn_regressor_landmarker")

        gates = tuple(
            VisionGateObservation.from_payload(gate)
            for gate in regressor_payload.get("gates", [])
            if isinstance(gate, dict)
        )
        return VisionObservation(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            gates=gates,
            source="cnn_regressor",
        )

    @property
    def cnn(self) -> LightmaskInference:
        if self._cnn is None:
            self._cnn = LightmaskInference(Path(self.config.checkpoint), device=self.config.device)
        return self._cnn

    @property
    def regressor(self) -> LogitRegressor:
        if self._regressor is None:
            self._regressor = LogitRegressor(
                Path(self.config.regressor_checkpoint),
                device=self.config.device,
                gate_threshold=float(self.config.gate_threshold),
                confidence_threshold=float(self.config.confidence_threshold),
                min_component_area=int(self.config.min_component_area),
                max_candidates=int(self.config.max_candidates),
            )
        return self._regressor

    @property
    def landmarker(self) -> Landmarker:
        if self._landmarker is None:
            self._landmarker = Landmarker(new_landmarker_state())
        return self._landmarker

    def snapshot(self) -> dict[str, Any]:
        return {
            "device": self.config.device,
            "run_landmarker": self.config.run_landmarker,
            "top_k": self.config.top_k,
            "passthrough_regressor_targets": self.config.passthrough_regressor_targets,
            "cnn_loaded": self._cnn is not None,
            "regressor_loaded": self._regressor is not None,
            "landmarker_loaded": self._landmarker is not None,
        }
