# Deterministic v2

This gate projection backend decodes each JPEG once and runs profiles `A`, `B`,
and `C` independently. Its service maps gates across the frames processed so
far. Keep one service instance for an ordered stream.

## Call

The standalone CLI is retired. After installing `aigp-vision`, select
`deterministic_0721_v2` and a writable scratch directory:

```python
from pathlib import Path
from aigp_vision import VisionPerceptionConfig, VisionPerceptionService
from aigp_vision.models.deterministic_v2 import DeterministicVisionV2Config

vision = VisionPerceptionService(VisionPerceptionConfig(
    backend="deterministic_0721_v2",
    deterministic_v2=DeterministicVisionV2Config(
        scratch_root=Path("logs/vision/deterministic_v2"),
    ),
))
try:
    observation = vision.process_vision_frame(frame)  # Caller-owned JPEG frame.
finally:
    vision.shutdown()
```

Scratch files are cleaned up by default; `keep_scratch=True` retains them.

## Contract

The callable service returns `aigp_vision.VisionObservation`, containing
`VisionGateObservation` gates and the caller's frame ID and simulation timestamp.
Use `observation.to_controller_payload()` for its JSON-compatible payload.

The pipeline's separate `VisionGateResult` and `VisionResults` types live in
`aigp_vision.models.deterministic_v2.src.schema` and describe camera-coordinate
projection results. The service preserves its existing mapping into the common
observation fields; this backend does not use `vehicle_state` to transform
camera estimates. Pipeline JSON files and manifests are internal artifacts,
not the service's return contract.
