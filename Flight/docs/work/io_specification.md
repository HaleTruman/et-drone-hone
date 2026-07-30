# Vision Library I/O Specification

This document specifies the public input and output contracts for a standalone
deterministic vision library. The library should use a conventional Python
package structure and expose a single central entrypoint:

```python
observation = DeterministicVision().process_frame(frame)
```

`frame` must be a `VisionFrame`-compatible dataclass. The return value must be a
`VisionObservation`-compatible dataclass.

## Package Shape

The library should be importable as a normal Python package and should keep
public contracts separate from implementation internals.

Recommended structure:

```text
deterministic_vision/
  README.md
  src/
    vision
      __init__.py
      deterministic.py
      schema.py
      config.py
      ...whatever_else_you_want.py
```

Public imports should be available from the package root or stable schema
module:

```python
from deterministic_vision import DeterministicVision
from deterministic_vision.schema import VisionFrame, VisionObservation, VisionGateObservation
```

The implementation may contain additional modules for detection, segmentation,
pose estimation, debugging, calibration, and visualization, but downstream
systems should not need to import those internals to run the vision pipeline.

## Public Entrypoint

The library must provide a `DeterministicVision` class. This class owns the
configuration and all deterministic vision methods used by the pipeline.

Required method:

```python
class DeterministicVision:
    def process_frame(self, frame: VisionFrame, vehicle_state: VehicleState | None) -> VisionObservation:
        ...
```

Behavioral requirements:

- `process_frame` accepts exactly one frame object that conforms to the
  `VisionFrame` schema.
- `process_frame` returns a `VisionObservation`.
- The returned observation must preserve `frame.frame_id` as
  `VisionObservation.frame_id`.
- The returned observation must preserve `frame.sim_time_ns` as
  `VisionObservation.sim_time_ns`.
- If no gates are detected, the method must return a valid `VisionObservation`
  with `gates=[]`, not `None`.
- The method should be deterministic for the same input frame and same
  configuration.
- Runtime diagnostics, intermediate detections, thresholds, timings, and other
  debug data should be placed in `trace` or in documented optional extension
  fields.

## Input Schema: VisionFrame

The input frame schema must match the current Flight contract defined at
`Flight/src/core/schema.py`.

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    image: Any | None = None
    saved_path: str | None = None
```

Additionally, the ``DeterministicVision().process_frame(frame, vehicle_state)`` method takes a VehicleState optionally, which will be imported from the existing schema. The definition of a VehicleState dataclass will be defined at ``Flight\src\core\schema.py``

```python
Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class VehicleState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3
```

Fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `frame_id` | `int` | Yes | Monotonic frame identifier from the producing stream. |
| `sim_time_ns` | `int` | Yes | Simulation timestamp in nanoseconds. |
| `jpeg_bytes` | `bytes` | Yes | Raw encoded JPEG frame bytes. |
| `image` | `Any \| None` | No | Optional decoded image object, normally an OpenCV/Numpy image. |
| `saved_path` | `str \| None` | No | Optional path to the persisted source frame. |

The library may decode `jpeg_bytes` itself or use `image` when supplied. If both
are present, implementations should document which one is authoritative. For
Flight compatibility, `jpeg_bytes` must remain accepted even when `image` is
`None`.

## Minimum Output Schema: VisionObservation

The output observation schema must hold at least the current Flight contract
defined at `Flight/src/core/schema.py`.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class VisionGateObservation:
    gate_id: str
    position_local_ned: tuple[float, float, float]
    position_confidence: float
    orientation_local_ned_quat: tuple[float, float, float, float] | None = None
    orientation_confidence: float | None = None
    trace: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class VisionObservation:
    frame_id: int
    sim_time_ns: int
    gates: list[VisionGateObservation]
    source: str = "vision"
    trace: dict[str, Any] = field(default_factory=dict)
```

### VisionObservation Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `frame_id` | `int` | Yes | Frame identifier copied from the input `VisionFrame`. |
| `sim_time_ns` | `int` | Yes | Simulation timestamp copied from the input `VisionFrame`. |
| `gates` | `list[VisionGateObservation]` | Yes | Gate observations in local-NED coordinates. |
| `source` | `str` | No | Producer identifier. Default should be `"vision"` unless a more specific backend name is useful. |
| `trace` | `dict[str, Any]` | No | JSON-compatible diagnostic metadata. |

### VisionGateObservation Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `gate_id` | `str` | Yes | Stable identifier for the observed gate when known. Temporary IDs are acceptable when identity is not known. |
| `position_local_ned` | `tuple[float, float, float]` | Yes | Gate position in local-NED coordinates, in meters. |
| `position_confidence` | `float` | Yes | Confidence score for the position estimate. |
| `orientation_local_ned_quat` | `tuple[float, float, float, float] \| None` | No | Gate orientation in local-NED quaternion form when available. |
| `orientation_confidence` | `float \| None` | No | Confidence score for the orientation estimate. |
| `trace` | `dict[str, Any]` | No | Per-gate JSON-compatible diagnostics. |

## Optional Rich Result Fields

`VisionObservation` must include the minimum fields above, but the library may
also expose richer output data similar to the current deterministic-v2
`VisionResults` schema at
`Flight/src/sensing/vision/models/deterministic_v2/src/schema.py`.

Optional result metadata may include:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | `str` | Result schema identifier, for example `"vision-results.v1"`. |
| `run_id` | `str` | Identifier for a processing run. |
| `frame_ordinal` | `int` | Sequential frame number within a run. |
| `source_path` | `str` | Source image path when the frame came from disk or was persisted. |
| `image_width` | `int` | Width of the decoded image in pixels. |
| `image_height` | `int` | Height of the decoded image in pixels. |
| `created_at` | `str` | UTC creation timestamp. |
| `timing_ms` | `dict[str, float]` | Timing breakdown by stage. |
| `coordinate_frame` | `str` | Name of the coordinate convention used by result fields. |
| `camera_position_local_ned_m` | `tuple[float, float, float]` | Camera position in local-NED coordinates when available. |
| `camera_orientation_local_ned_quat` | `tuple[float, float, float, float]` | Camera orientation in local-NED quaternion form when available. |
| `obstacles` | `list[dict[str, Any]]` | Optional obstacle detections or estimates. |
| `trace` | `dict[str, Any]` | JSON-compatible diagnostic metadata. |

There are two acceptable approaches for richer data:

1. Add optional fields to `VisionObservation` while keeping the minimum Flight
   fields unchanged.
2. Return a `VisionObservation` as the primary controller-facing object and
   include richer deterministic-v2-style details under `VisionObservation.trace`
   or through a separately documented `VisionResults` object.

The first approach is convenient for a standalone package. The second approach
is safer when integrating with existing Flight code that expects the current
minimal dataclass.


## Validation Rules

The library should validate or normalize the public output before returning it:

- `position_local_ned` must contain exactly three numeric values.
- `orientation_local_ned_quat`, when present, must contain exactly four numeric values.
- Confidence fields must be numeric floats.
- `gate_id` must be a string.
- Missing detections must be represented by an empty `gates` list.
- Invalid input frames should raise a clear exception rather than returning a
  partial or malformed observation.

## Minimal Example

```python
from vision_library import DeterministicVision
from vision_library.schema import VisionFrame

Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]

@dataclass(frozen=True)
class VehicleState:
    sim_time_ns: int
    position_local_ned_m: Vec3
    velocity_local_ned_mps: Vec3
    attitude_quaternion: QuatWxyz
    body_rates_frd_rps: Vec3
    acceleration_local_ned_mps2: Vec3


vision = DeterministicVision()

frame = VisionFrame(
    frame_id=1,
    sim_time_ns=123456789,
    jpeg_bytes=jpeg_bytes,
)

vehicle_state = VehicleState(
  sim_time_ns: 123456789,
  position_local_ned_m = Vec3([0, 0, 0]),
  velocity_local_ned_mps = Vec3([0, 0, 0]),
  attitude_quaternion = QuatWxyz([1, 0, 0, 0]),
  body_rates_frd_rps Vec3([0, 0, 0]),
  acceleration_local_ned_mps2 Vec3([0, 0, 0])
)

observation = vision.process_frame(frame, vehicle_state)

assert observation.frame_id == 1
assert observation.sim_time_ns == 123456789
assert isinstance(observation.gates, list)
```

## Compatibility Requirement

For Flight integration, the required compatibility target is:

```python
from core.schema import VisionFrame, VisionObservation

observation: VisionObservation = DeterministicVision().process_frame(frame)
```

Any richer data model must preserve this minimum behavior so existing mapping,
planning, logging, and controller payload consumers can continue to operate.
