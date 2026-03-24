## Image Capture Contract

This document describes the current live image-capture implementation first.

It is not an ideal-state draft anymore. If a behavior is not implemented in the
live repo, it should be called out under **Future Work / Target-State
Extensions** at the bottom instead of being mixed into the live contract.

## Purpose

Image capture is the clearest end-to-end proof path in the current repo because
it crosses all of these boundaries:

1. authored config
2. compiled runtime config
3. top-level build handoff
4. top-level runtime launch and config apply
5. runtime capture dispatch inside Unreal
6. returned `OBS` payload
7. raw and accepted artifact persistence

If this path is deterministic and traceable, the same pattern can later be
extended to richer telemetry, control-linked observations, and multi-action
runtime episodes.

## Source Of Truth

When this document conflicts with implementation details, use these live code
paths as the source of truth:

1. `UE_Tooling/run_unreal_build.py`
2. `UE_Tooling/run_unreal_io.py`
3. `UE_Tooling/Config/RunConfig.py`
4. `UE_Tooling/Data_Interface/Sampler_Manager.py`
5. `UE_Tooling/WebSocket/protocol.py`
6. `UE_Tooling/WebSocket/ws_bridge.py`
7. `UE_Drone_Env/Source/UE_Drone_Env/SetDataConfigRuntimeActor.cpp`
8. `UE_Drone_Env/Source/UE_Drone_Env/SampleManagerRuntimeActor.cpp`
9. `UE_Drone_Env/Source/UE_Drone_Env/DroneSensorsRuntimeComponent.cpp`

## Status Legend

- `ACTIVE`: emitted, consumed, or enforced by live code today
- `COMPAT`: accepted or emitted for compatibility during transition, but not the preferred canonical shape
- `LOOSE`: generated or mirrored contract surface exists today, but it is not the exact live smoke-path call boundary
- `FUTURE`: desired contract direction, not required for the current live image smoke path

## Current Live End-To-End Flow

The live image-capture path currently works like this:

1. `UE_Tooling/run_unreal_build.py` orchestrates the WebSocket, Course, Drone,
   and IO build stages and writes a canonical build manifest under
   `UE_Tooling/Artifacts/build/<run_name>.json`.
2. That build manifest carries the authoritative level handoff plus IO placement
   validation, including canonical singleton placement for:
   - `BP_SetDataConfig_Main`
   - `BP_SampleManager_Main`
   - `BP_DronePawn_Startup_Main`
3. `UE_Tooling/run_unreal_io.py` consumes the build manifest, starts the
   bridge, launches Unreal, compiles or loads `SET_CONFIG`, applies config,
   validates startup topology, and then runs the sampler smoke capture.
4. `UE_Tooling/Data_Interface/Sampler_Manager.py` emits a minimal
   `CAPTURE_NOW` action with top-level `capture_id` and empty payload.
5. `ASetDataConfigRuntimeActor::HandleCaptureNow(...)` resolves
   `BP_SampleManager_Main` and dispatches capture into the runtime sample
   manager instead of emitting a placeholder image directly.
6. `ASampleManagerRuntimeActor` resolves the hosted
   `UDroneSensorsRuntimeComponent`, captures a real rendered PNG, queries a real
   viewpoint snapshot, resolves the active config reference, and assembles the
   observation payload.
7. `ASetDataConfigRuntimeActor` sends the assembled `OBS` envelope back over
   the WebSocket.
8. `ws_bridge.py` optionally persists raw transport artifacts.
9. `Sampler_Manager.py` validates the returned image against active config
   identity and expected viewpoint metadata before writing accepted sample
   artifacts.

## Runtime Preconditions For Image Capture

Image capture is not a standalone runtime action. The current live repo expects
these preconditions to already be true:

### Build-Layer Preconditions

Owner:

- `UE_Tooling/run_unreal_build.py`

Required live outputs:

| Output | Status | Meaning |
| --- | --- | --- |
| top-level build manifest under `UE_Tooling/Artifacts/build/<run_name>.json` | `ACTIVE` | canonical build handoff for runtime |
| `authoritative_course_level_path` | `ACTIVE` | exact map package path runtime must launch |
| IO placement summary for `BP_SetDataConfig_Main` | `ACTIVE` | proves config ingress is preplaced |
| IO placement summary for `BP_SampleManager_Main` | `ACTIVE` | proves capture owner is preplaced |
| IO placement summary for `BP_DronePawn_Startup_Main` | `ACTIVE` | proves canonical startup drone exists for smoke capture |

### Runtime-Layer Preconditions

Owner:

- `UE_Tooling/run_unreal_io.py`

Required live runtime stages for smoke capture:

| Runtime stage | Status | Meaning |
| --- | --- | --- |
| build handoff ingest | `ACTIVE` | validates build manifest and required placements |
| bridge supervision | `ACTIVE` | starts websocket bridge and captures bridge artifacts |
| Unreal runtime launch | `ACTIVE` | launches the authoritative level in a render-enabled runtime mode |
| config apply | `ACTIVE` | sends `SET_CONFIG`, observes `ACK` and `CONFIG_READY` |
| startup topology validation | `ACTIVE` | proves startup pawn preplacement and post-config readiness |
| sampler smoke capture | `ACTIVE` | emits one truthful sampler-owned capture request and validates returned `OBS` |

The current document is about the contract inside and after that smoke-capture
stage, not about the entire runtime orchestrator.

## Stage 1: Source Config Contract

Primary source file:

- `UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml`

Current live fields:

| YAML path | Status | Meaning |
| --- | --- | --- |
| `schema_version` | `ACTIVE` | local YAML schema version only |
| `viewpoints.capture_interval_seconds` | `ACTIVE` | tooling-side capture cadence consumed by `Sampler_Manager.py` |
| `viewpoints.resolution.width` | `ACTIVE` | runtime image width |
| `viewpoints.resolution.height` | `ACTIVE` | runtime image height |
| `viewpoints.resolution.fov_deg` | `ACTIVE` | runtime image FOV |
| `viewpoints.capture_rig.offset_cm.x` | `ACTIVE` | camera rig offset X |
| `viewpoints.capture_rig.offset_cm.y` | `ACTIVE` | camera rig offset Y |
| `viewpoints.capture_rig.offset_cm.z` | `ACTIVE` | camera rig offset Z |
| `viewpoints.capture_rig.rotation_deg.pitch` | `ACTIVE` | camera rig pitch |
| `viewpoints.capture_rig.rotation_deg.roll` | `ACTIVE` | camera rig roll |
| `viewpoints.capture_rig.rotation_deg.yaw` | `ACTIVE` | camera rig yaw |

Current live behavior:

- `RunConfig.py` treats the runtime image fields as required compile inputs.
- `Sampler_Manager.py` reads only `capture_interval_seconds` from this file.
- The runtime image fields are not optional suggestion fields. Missing or
  malformed values fail the compile path.

Canonical rule for the live repo:

- optical settings and rig transform come from this viewpoint YAML
- per-capture behavior does not override them
- capture cadence is tooling-owned, not Unreal-config-owned

## Stage 2: Compiled Runtime Config Contract

Current compiler:

- `UE_Tooling/Config/RunConfig.py`

Current live image-related `SET_CONFIG` fields:

| `SET_CONFIG` path | Status | Meaning |
| --- | --- | --- |
| `payload.config_id` | `ACTIVE` | effective runtime config identity |
| `payload.config_hash` | `ACTIVE` | effective runtime config hash |
| `payload.sensor_rig.active_viewpoint` | `ACTIVE` | current active viewpoint name |
| `payload.sensor_rig.front_fov_deg` | `ACTIVE` | effective FOV |
| `payload.sensor_rig.capture_width` | `ACTIVE` | effective width |
| `payload.sensor_rig.capture_height` | `ACTIVE` | effective height |
| `payload.sensor_rig.front_offset_cm.x` | `ACTIVE` | effective rig offset X |
| `payload.sensor_rig.front_offset_cm.y` | `ACTIVE` | effective rig offset Y |
| `payload.sensor_rig.front_offset_cm.z` | `ACTIVE` | effective rig offset Z |
| `payload.sensor_rig.front_rotation_deg.pitch` | `ACTIVE` | effective rig pitch |
| `payload.sensor_rig.front_rotation_deg.roll` | `ACTIVE` | effective rig roll |
| `payload.sensor_rig.front_rotation_deg.yaw` | `ACTIVE` | effective rig yaw |

Current live runtime behavior:

- `ASetDataConfigRuntimeActor` validates the incoming `sensor_rig` block and
  applies it onto live sensor owners.
- `UDroneSensorsRuntimeComponent` reads back the active values from the live
  owner when capture is requested.
- `DA_SensorRigProfileDefault` remains bootstrap-only state and is expected to
  be overwritten on configured runs.

Canonical rule for the live repo:

- `SET_CONFIG.payload.sensor_rig` is the authoritative runtime image config
- bootstrap defaults are not the source of truth for a configured run

## Stage 3: Capture Request Contract

Tooling owner:

- `UE_Tooling/Data_Interface/Sampler_Manager.py`

Current live `CAPTURE_NOW` contract:

| Path | Status | Meaning |
| --- | --- | --- |
| top-level `type` | `ACTIVE` | `CAPTURE_NOW` |
| top-level `run_id` | `ACTIVE` | run trace key |
| top-level `schema_version` | `ACTIVE` | wire contract version |
| top-level `seq` | `ACTIVE` | message ordering key |
| top-level `timestamp` | `ACTIVE` | action timestamp |
| top-level `drone_id` | `ACTIVE` | target drone |
| top-level `capture_id` | `ACTIVE` | traceable capture key |
| `payload` | `ACTIVE` | empty object for the current smoke path |
| `payload.capture_id` | `COMPAT` | accepted as fallback by UE ingress if top-level `capture_id` is absent |

Current live behavior:

1. wait for bridge connection
2. wait for config-ready state
3. generate next `capture_id`
4. send one minimal `CAPTURE_NOW`
5. wait for one matching `OBS`
6. validate returned image against active config identity and expected viewpoint
7. log one terminal status: `received` or `failed`

Canonical rule for the live repo:

- `CAPTURE_NOW` is a trigger, not a second config surface
- the live capture behavior must come from already-applied config
- capture correlation is keyed by top-level `capture_id`

## Stage 4: Unreal Runtime Resolution Contract

Current live Unreal-side owners:

| System | Live role in image capture | Status |
| --- | --- | --- |
| `ASetDataConfigRuntimeActor` / `BP_SetDataConfig` | receive `CAPTURE_NOW`, resolve sample manager, validate returned observation, emit `OBS` | `ACTIVE` |
| `ASampleManagerRuntimeActor` / `BP_SampleManager` | own atomic image capture orchestration and observation assembly | `ACTIVE` |
| `UDroneSensorsRuntimeComponent` / `BP_DroneSensors` | own active image state, capture the rendered PNG, expose viewpoint snapshot | `ACTIVE` |
| `ST_DroneViewpointSnapshot` | generated schema contract for the intended viewpoint surface | `LOOSE` |
| `BPI_DroneViewpointProvider` | generated interface contract for viewpoint ownership symmetry | `LOOSE` |

Important current implementation truth:

- the live path does **not** emit a placeholder PNG anymore
- the live path does **not** bypass `BP_SampleManager` anymore
- the live path currently uses the native hosted sensor runtime component
  directly for the smoke capture path
- that means the generated `BPI_DroneViewpointProvider` and
  `ST_DroneViewpointSnapshot` assets still matter as contract surfaces, but they
  are not the exact live C++ call boundary used by the current smoke path

Current live viewpoint snapshot content:

| Field | Status | Meaning |
| --- | --- | --- |
| `timestamp_utc` | `ACTIVE` | snapshot timestamp |
| `run_id` | `ACTIVE` | run trace key |
| `capture_id` | `ACTIVE` | capture trace key |
| `viewpoint_name` | `ACTIVE` | active viewpoint used |
| `fov_deg` | `ACTIVE` | effective FOV |
| `width` | `ACTIVE` | effective width |
| `height` | `ACTIVE` | effective height |
| `rig_offset_from_drone_body_cm` | `ACTIVE` | effective rig offset |
| `rig_rotation_from_drone_body_deg.pitch` | `ACTIVE` | effective rig pitch |
| `rig_rotation_from_drone_body_deg.roll` | `ACTIVE` | effective rig roll |
| `rig_rotation_from_drone_body_deg.yaw` | `ACTIVE` | effective rig yaw |

## Stage 5: Live `OBS` Image Contract

Current live returned `OBS` payload fields for image capture:

| Path | Status | Meaning |
| --- | --- | --- |
| top-level `type` | `ACTIVE` | `OBS` |
| top-level `run_id` | `ACTIVE` | run trace key |
| top-level `drone_id` | `ACTIVE` | drone trace key |
| top-level `capture_id` | `ACTIVE` | capture trace key |
| `payload.timestamp_utc` | `ACTIVE` | observation timestamp |
| `payload.run_id` | `ACTIVE` | duplicated run identity inside payload |
| `payload.drone_id` | `ACTIVE` | duplicated drone identity inside payload |
| `payload.capture_id` | `ACTIVE` | duplicated capture identity inside payload |
| `payload.config_ref.config_id` | `ACTIVE` | applied runtime config identity |
| `payload.config_ref.config_hash` | `ACTIVE` | applied runtime config hash |
| `payload.viewpoint.timestamp_utc` | `ACTIVE` | viewpoint snapshot timestamp |
| `payload.viewpoint.run_id` | `ACTIVE` | viewpoint trace link |
| `payload.viewpoint.capture_id` | `ACTIVE` | viewpoint trace link |
| `payload.viewpoint.viewpoint_name` | `ACTIVE` | active viewpoint name |
| `payload.viewpoint.fov_deg` | `ACTIVE` | effective FOV |
| `payload.viewpoint.width` | `ACTIVE` | effective width |
| `payload.viewpoint.height` | `ACTIVE` | effective height |
| `payload.viewpoint.rig_offset_from_drone_body_cm` | `ACTIVE` | effective rig offset |
| `payload.viewpoint.rig_rotation_from_drone_body_deg` | `ACTIVE` | effective rig rotation |
| `payload.image.encoding` | `ACTIVE` | currently `png_base64` |
| `payload.image.bytes_b64` | `ACTIVE` | canonical image byte field |
| `payload.image.width` | `ACTIVE` | image width |
| `payload.image.height` | `ACTIVE` | image height |
| `payload.image_bytes_b64` | `COMPAT` | legacy top-level payload alias retained for compatibility |

Current live omissions for the image smoke path:

| Path | Status | Meaning |
| --- | --- | --- |
| `payload.pose.*` | `FUTURE` | not part of the current live image smoke payload |
| `payload.telemetry.*` | `FUTURE` | not part of the current live image smoke payload |

Current live validation behavior:

- `ASetDataConfigRuntimeActor` rejects observation payloads that are missing
  `config_ref`, `viewpoint`, or image bytes
- `Sampler_Manager.py` requires:
  - matching `run_id`
  - matching `drone_id`
  - non-empty `capture_id`
  - matching `config_ref.config_id`
  - matching `config_ref.config_hash`
  - matching `viewpoint.width`
  - matching `viewpoint.height`
  - matching `viewpoint.fov_deg`
  - PNG dimensions that match viewpoint metadata

Canonical rule for the live repo:

- one accepted image must be self-describing enough to prove which config and
  viewpoint produced it
- compatibility aliasing can remain temporarily, but canonical nested `payload.image`
  is already live and should be treated as preferred

## Stage 6: Artifact Persistence Contract

Current live artifact owners:

- `UE_Tooling/run_unreal_io.py` owns the runtime session root
- `UE_Tooling/WebSocket/ws_bridge.py` owns optional raw transport persistence
- `UE_Tooling/Data_Interface/Sampler_Manager.py` owns accepted image persistence

Current live runtime wrapper paths:

| Artifact path | Status | Owner | Meaning |
| --- | --- | --- | --- |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/runtime_session_manifest.json` | `ACTIVE` | `run_unreal_io.py` | top-level runtime phase evidence |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/bridge/<runtime_run_id>/summary.json` | `ACTIVE` | `ws_bridge.py` | bridge session summary |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/bridge/<runtime_run_id>/raw_messages.jsonl` | `ACTIVE` | `ws_bridge.py` | raw message audit trail |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/bridge/<runtime_run_id>/raw_observations.jsonl` | `ACTIVE` | `ws_bridge.py` | raw observation audit trail |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/bridge/<runtime_run_id>/raw_captures/<capture_id>/raw__<drone_id>__<seq>.png` | `ACTIVE` | `ws_bridge.py` | raw decoded PNG |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/bridge/<runtime_run_id>/samples/images/<capture_id>/<drone_id>__<seq>__<width-height>__<interval>.png` | `ACTIVE` | `Sampler_Manager.py` | accepted sample image |
| `UE_Tooling/Artifacts/runtime/<runtime_run_id>/bridge/<runtime_run_id>/samples/accepted_samples.jsonl` | `ACTIVE` | `Sampler_Manager.py` | accepted sample metadata |

Current accepted sample record content:

| Field | Status | Meaning |
| --- | --- | --- |
| `config_ref.config_id` | `ACTIVE` | accepted sample config identity |
| `config_ref.config_hash` | `ACTIVE` | accepted sample config hash |
| `viewpoint.width` | `ACTIVE` | accepted sample width |
| `viewpoint.height` | `ACTIVE` | accepted sample height |
| `viewpoint.fov_deg` | `ACTIVE` | accepted sample FOV |
| `image.sample_file` | `ACTIVE` | relative path to accepted PNG |
| `image.raw_image_file` | `ACTIVE` | relative path back to raw PNG when present |

Canonical rule for the live repo:

- raw artifacts remain transport-owned
- accepted sample artifacts remain sampler-owned
- runtime wrapper artifacts remain runtime-orchestrator-owned

## End-To-End Traceability Requirements

For one accepted image sample, the current live repo can trace:

1. build manifest path
2. authoritative level path
3. runtime session manifest path
4. bridge summary path
5. run-level `config_id`
6. run-level `config_hash`
7. `capture_id`
8. `drone_id`
9. returned viewpoint width, height, FOV, rig offset, and rig rotation
10. raw decoded PNG path
11. accepted PNG path

That is the current minimum truthful image-capture trace chain.

## Current Live Truth Summary

The live repo now satisfies these image-capture statements:

- real rendered PNG capture is used for the smoke path
- `CAPTURE_NOW` dispatch goes through `BP_SampleManager`
- viewpoint metadata is returned with the observation
- config reference is returned with the observation
- canonical nested `payload.image` exists on the wire
- legacy `payload.image_bytes_b64` still exists as a compatibility alias
- strict tooling-side validation uses config identity plus viewpoint metadata
- accepted sample persistence happens only after validation succeeds

The live repo does **not** currently claim these stronger image-capture statements:

- pose is included in the image smoke payload
- telemetry is included in the image smoke payload
- the generated `BPI_*` and `ST_*` assets are the exact live smoke-path call boundary
- multi-viewpoint or per-capture viewpoint override behavior exists

## Future Work / Target-State Extensions

The items below are intentionally future-oriented. They are not required to
describe the current live image smoke path truthfully.

### 1. Tighten provider symmetry

Desired direction:

- the generated `BPI_DroneViewpointProvider` and `ST_DroneViewpointSnapshot`
  should become the exact live runtime query boundary, not just the mirrored
  contract surface around the native smoke-path implementation

### 2. Expand `OBS` beyond the current image-only smoke payload

Desired direction:

- add canonical pose block if image+pose atomic observations are required
- add canonical telemetry block if image+telemetry atomic observations are required
- keep those additions separate from the current minimal image smoke truth

### 3. Remove compatibility duplication once downstream paths are fully aligned

Desired direction:

- eventually retire `payload.image_bytes_b64`
- eventually reduce duplicate trace fields inside payload where envelope fields
  already exist, if that can be done without breaking current tooling

### 4. Richer capture semantics

Desired direction:

- explicit viewpoint selection beyond the current single active viewpoint model
- multi-viewpoint capture
- stronger linkage between capture, spawn, and command context when the runtime
  expands beyond the current smoke-proof scope
