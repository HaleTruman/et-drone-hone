from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core.schema import VehicleState, VisionFrame, VisionGateObservation, VisionObservation
from sensing.vision.models.cnn.rgb_inference import DEFAULT_CHECKPOINT, LightmaskInference
from sensing.vision.models.cnn.rgb_normalizer import jpeg_bytes_to_tensor
from sensing.vision.models.cnn.landmarker.landmark_output import build_controller_payload, build_passthrough_controller_payload
from sensing.vision.models.cnn.landmarker.landmarker import Landmarker, new_landmarker_state
from sensing.vision.models.cnn.regressor.cnn_ingress import RawLogitsFrame
from sensing.vision.models.cnn.regressor.logit_inference import DEFAULT_REGRESSOR_CHECKPOINT, LogitRegressor
from sensing.vision.models.cnn.regressor.regression_output import build_surveyer_payload
from sensing.vision.models.deterministic import DeterministicVisionConfig
from sensing.vision.models.deterministic_v2 import DeterministicVisionV2Config


DETERMINISTIC_V3_BACKEND_DIR = Path(__file__).resolve().parent / "models" / "deterministic_v3"


@dataclass(frozen=True)
class DeterministicVisionV3Config:
    lut_path: Path = DETERMINISTIC_V3_BACKEND_DIR / "assets" / "color_lut_v1.npz"


@dataclass(frozen=True)
class VisionPerceptionConfig:
    backend: str = "cnn_regressor"
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
    deterministic: DeterministicVisionConfig = DeterministicVisionConfig()
    deterministic_v2: DeterministicVisionV2Config = DeterministicVisionV2Config()
    deterministic_v3: DeterministicVisionV3Config = DeterministicVisionV3Config()


class VisionPerceptionService:
    """Flight-facing vision service with selectable perception backends."""

    def __init__(
        self,
        config: VisionPerceptionConfig | None = None,
        *,
        cnn: LightmaskInference | None = None,
        regressor: LogitRegressor | None = None,
        landmarker: Landmarker | None = None,
        deterministic: Any | None = None,
        deterministic_v2: Any | None = None,
        deterministic_v3: Any | None = None,
    ) -> None:
        self.config = config or VisionPerceptionConfig()
        if self.config.backend not in {
            "cnn_regressor",
            "deterministic_0721",
            "deterministic_0721_v2",
            "deterministic_v3",
        }:
            raise ValueError(
                "VisionPerceptionConfig.backend must be 'cnn_regressor', 'deterministic_0721', "
                "'deterministic_0721_v2', or 'deterministic_v3'."
            )
        self._cnn = cnn
        self._regressor = regressor
        self._landmarker = landmarker
        self._deterministic = deterministic
        self._deterministic_v2 = deterministic_v2
        self._deterministic_v3 = deterministic_v3

    def process_frame(
        self,
        *,
        frame_id: int,
        sim_time_ns: int,
        jpeg_bytes: bytes,
        vehicle_state: VehicleState | None = None,
    ) -> VisionObservation:
        if self.config.backend == "deterministic_0721":
            return self.deterministic.process_frame(frame_id=frame_id, sim_time_ns=sim_time_ns, jpeg_bytes=jpeg_bytes)
        if self.config.backend == "deterministic_0721_v2":
            return self.deterministic_v2.process_frame(frame_id=frame_id, sim_time_ns=sim_time_ns, jpeg_bytes=jpeg_bytes)
        if self.config.backend == "deterministic_v3":
            frame = VisionFrame(frame_id=int(frame_id), sim_time_ns=int(sim_time_ns), jpeg_bytes=jpeg_bytes)
            return self.deterministic_v3.process_frame(frame, self._vehicle_state_for_frame(frame, vehicle_state))
        return self._process_cnn_frame(frame_id=frame_id, sim_time_ns=sim_time_ns, jpeg_bytes=jpeg_bytes)

    @staticmethod
    def _vehicle_state_for_frame(frame: VisionFrame, vehicle_state: VehicleState | None) -> VehicleState | None:
        if vehicle_state is None:
            return None
        if vehicle_state.sim_time_ns == frame.sim_time_ns:
            return vehicle_state
        return replace(vehicle_state, sim_time_ns=int(frame.sim_time_ns))

    def _process_cnn_frame(self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes) -> VisionObservation:
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

    def process_vision_frame(
        self,
        frame: VisionFrame,
        *,
        vehicle_state: VehicleState | None = None,
    ) -> VisionObservation:
        if self.config.backend == "deterministic_v3":
            return self.deterministic_v3.process_frame(frame, self._vehicle_state_for_frame(frame, vehicle_state))
        return self.process_frame(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            jpeg_bytes=frame.jpeg_bytes,
            vehicle_state=vehicle_state,
        )

    def detect_gates(
        self,
        frame: VisionFrame,
        *,
        vehicle_state: VehicleState | None = None,
    ) -> tuple[VisionGateObservation, ...]:
        return tuple(self.process_vision_frame(frame, vehicle_state=vehicle_state).gates)

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

    @property
    def deterministic(self) -> Any:
        if self._deterministic is None:
            from sensing.vision.models.deterministic.service import DeterministicVisionBackend

            self._deterministic = DeterministicVisionBackend(self.config.deterministic)
        return self._deterministic

    @property
    def deterministic_v2(self) -> Any:
        if self._deterministic_v2 is None:
            from sensing.vision.models.deterministic_v2.service import DeterministicVisionV2Backend

            self._deterministic_v2 = DeterministicVisionV2Backend(self.config.deterministic_v2)
        return self._deterministic_v2

    @property
    def deterministic_v3(self) -> Any:
        if self._deterministic_v3 is None:
            from sensing.vision.models.deterministic_v3 import DeterministicVision

            self._deterministic_v3 = DeterministicVision(lut_path=str(self.config.deterministic_v3.lut_path))
        return self._deterministic_v3

    def shutdown(self) -> None:
        if self._deterministic is not None:
            self._deterministic.shutdown()
        if self._deterministic_v2 is not None:
            self._deterministic_v2.shutdown()

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "backend": self.config.backend,
            "device": self.config.device,
            "run_landmarker": self.config.run_landmarker,
            "top_k": self.config.top_k,
            "passthrough_regressor_targets": self.config.passthrough_regressor_targets,
            "cnn_loaded": self._cnn is not None,
            "regressor_loaded": self._regressor is not None,
            "landmarker_loaded": self._landmarker is not None,
        }
        if self._deterministic is not None:
            snapshot["deterministic"] = self._deterministic.snapshot()
        if self._deterministic_v2 is not None:
            snapshot["deterministic_v2"] = self._deterministic_v2.snapshot()
        if self._deterministic_v3 is not None:
            snapshot["deterministic_v3"] = {
                "backend": "deterministic_v3",
                "lut_path": str(self.config.deterministic_v3.lut_path),
            }
        return snapshot
