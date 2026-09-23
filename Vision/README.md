# Vision

A callable copy of Flight's existing perception service and models, extracted
from remote `dev` at `b8821ebe1ce5c13b752933f108f5377c5c8555c4`.
Flight owns frame acquisition and supplies frames, timestamps, and optional
vehicle state. This package performs perception and returns `VisionObservation`.
Flight's existing files and imports are unchanged.

## Install

From the repository root:

```sh
python -m pip install ./Vision
```

Dependencies retain the corresponding pins from `Flight/requirements.txt`.
For development in an environment where those dependencies are already managed,
use `python -m pip install --no-deps -e ./Vision`.

## Call

```python
from aigp_vision import VisionPerceptionConfig, VisionPerceptionService

vision = VisionPerceptionService(
    VisionPerceptionConfig(backend="deterministic_v3_2")
)
try:
    # frame and vehicle_state come from the caller, such as Flight.
    observation = vision.process_vision_frame(frame, vehicle_state=vehicle_state)
finally:
    vision.shutdown()
```

The existing `VisionFrame` and `VehicleState` field layouts are also exported by
`aigp_vision`. Flight's existing objects can be passed directly; the copied
dataclasses have separate Python class identities but the same fields and
payload methods. Alternatively, call
`vision.process_frame(frame_id=frame_id, sim_time_ns=sim_time_ns,
jpeg_bytes=jpeg_bytes, vehicle_state=vehicle_state)`.

Keep one service instance for an ordered stream to retain tracking state. Frame
IDs and timestamps are supplied by the caller. Backend names, model weights,
thresholds, coordinate conventions, and outputs are unchanged. The service's
default backend remains `cnn_regressor`; the example explicitly selects Flight's
current configured backend.

## Existing limitations

The older `deterministic_0721` backend lacks its default LUT/review assets in
`dev`. The archived `detection_v2` tool references an absent `reports` module.
These are preserved source limitations, not repaired by this extraction.
Offline review/CLI workflows are not part of the callable interface.

For the older deterministic backends, use their existing configuration's
`scratch_root` option to choose a writable working directory when needed.
No receiver, clock, flight initialization, telemetry, or controller is included.
Installing this package does not switch Flight to it.

## Verify

```sh
cd Vision
python -m pytest
```

Parity tests compare this copy with the original Flight code in the checkout.
