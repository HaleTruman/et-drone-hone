## Image Capture Contract

Note: this document currently describes an ideal target-state image capture contract and should not be read as proof that every described behavior is implemented end to end.

This section defines the detailed reference contract for image capture during runtime runs. It is intended to become the model for how pose and telemetry contracts should also be documented:

1. source config
2. compiled runtime config
3. capture request
4. Unreal runtime resolution
5. observation payload
6. artifact persistence
7. traceability and open issues

### Why This Section Exists

Image capture is the clearest end-to-end example of the contract problem this repo is trying to solve.

One captured image should be traceable all the way through:

- the authored viewpoint config
- the compiled `SET_CONFIG` payload
- the capture request initiated by `Sampler_Manager.py`
- the Unreal-side capture systems and viewpoint provider contract
- the returned `OBS` payload
- the decoded artifact written to disk

If this flow is clean and deterministic, the same shape can be mirrored for pose and telemetry.

The image-capture config surface should stay focused on the fields that actually vary per run:

- `fov_deg`
- `width`
- `height`
- camera rig offset from the drone body
- camera rig rotation relative to the drone body

### Image Capture Flow

Current intended ownership flow:

1. `UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml` defines image-capture preferences at authoring time.
2. `UE_Tooling/Config/RunConfig.py` compiles runtime image settings from that file into the `payload.sensor_rig` block of `SET_CONFIG`.
3. `Sampler_Manager.py` reads tooling-side capture cadence from that same file and owns runtime image capture requests.
4. `AWSConfigHandshakeActor` / `BP_SetDataConfig` gate runtime actions after config is ready.
5. `BP_SampleManager` should own atomic image capture orchestration inside Unreal.
6. `BPI_DroneViewpointProvider` should expose the viewpoint chosen for capture.
7. `BP_DroneSensors` should own the live viewpoint settings used to satisfy that request.
8. Unreal should emit `OBS` containing both image data and viewpoint metadata. # need more detail for this (dont removed unless human arrpoeved)
9. `ws_bridge.py` should persist the raw `OBS` plus decoded image artifacts.
10. `Sampler_Manager.py` should own any downstream normalized sample placement.

### Stage 1: Source Config Contract

Primary source file:

- `UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml`

Current file shape:

| YAML path | Meaning | Status | Canonical expectation |
| --- | --- | --- | --- |
| `schema_version` | authoring-file schema version | `ACTIVE` | must stay distinct from websocket `schema_version` |
| `viewpoints.capture_interval_seconds` | tooling-side capture cadence in seconds | `ACTIVE` | should be consumed by `Sampler_Manager.py`, not compiled into `SET_CONFIG` |
| `viewpoints.resolution.width` | image width for the active run | `ACTIVE` | should define capture width in pixels |
| `viewpoints.resolution.height` | image height for the active run | `ACTIVE` | should define capture height in pixels |
| `viewpoints.resolution.fov_deg` | field of view for the active run | `ACTIVE` | should define capture FOV |
| `viewpoints.capture_rig.offset_cm.x` | camera offset X relative to drone body | `ACTIVE` | should be the canonical authored source for camera placement |
| `viewpoints.capture_rig.offset_cm.y` | camera offset Y relative to drone body | `ACTIVE` | same expectation |
| `viewpoints.capture_rig.offset_cm.z` | camera offset Z relative to drone body | `ACTIVE` | same expectation |
| `viewpoints.capture_rig.rotation_deg.pitch` | camera pitch relative to drone body | `ACTIVE` | should be the canonical authored source for camera aim |
| `viewpoints.capture_rig.rotation_deg.roll` | camera roll relative to drone body | `ACTIVE` | same expectation |
| `viewpoints.capture_rig.rotation_deg.yaw` | camera yaw relative to drone body | `ACTIVE` | same expectation |

Current compiler/runtime status against the live file shape:

- `RunConfig.py` consumes `resolution.*`, `capture_rig.offset_cm.*`, and `capture_rig.rotation_deg.*` from this file
- `Sampler_Manager.py` consumes `viewpoints.capture_interval_seconds` from this file as tooling-side sampling cadence
- those runtime image fields are required inputs, not fallback-backed inputs
- missing or malformed runtime image fields should fail fast during `RunConfig.py` compile

Canonical rule for this stage:

- width, height, FOV, rig offset, and rig rotation are the actual image-capture config inputs that should be tracked per run
- `capture_interval_seconds` is a tooling-side sampling input and should not be treated as runtime UE config
- the authored image config should answer two things clearly:
  1. what optical settings should be used
  2. where the camera is mounted and what angle it is pointed at

### Stage 2: Compiled Runtime Config Contract

Current compiler:

- `UE_Tooling/Config/RunConfig.py`

Current image-related compiled fields:

| `SET_CONFIG` path | Source | Status | Meaning |
| --- | --- | --- | --- |
| `payload.sensor_rig.active_viewpoint` | compiler policy default | `ACTIVE` | selected runtime viewpoint name |
| `payload.sensor_rig.front_fov_deg` | `viewpoints.resolution.fov_deg` | `ACTIVE` | effective capture FOV |
| `payload.sensor_rig.capture_width` | `viewpoints.resolution.width` | `ACTIVE` | effective image width |
| `payload.sensor_rig.capture_height` | `viewpoints.resolution.height` | `ACTIVE` | effective image height |
| `payload.sensor_rig.front_offset_cm.x` | `viewpoints.capture_rig.offset_cm.x` | `ACTIVE` | effective rig offset X |
| `payload.sensor_rig.front_offset_cm.y` | `viewpoints.capture_rig.offset_cm.y` | `ACTIVE` | effective rig offset Y |
| `payload.sensor_rig.front_offset_cm.z` | `viewpoints.capture_rig.offset_cm.z` | `ACTIVE` | effective rig offset Z |
| `payload.sensor_rig.front_rotation_deg.pitch` | `viewpoints.capture_rig.rotation_deg.pitch` | `ACTIVE` | effective camera pitch relative to drone body |
| `payload.sensor_rig.front_rotation_deg.roll` | `viewpoints.capture_rig.rotation_deg.roll` | `ACTIVE` | effective camera roll relative to drone body |
| `payload.sensor_rig.front_rotation_deg.yaw` | `viewpoints.capture_rig.rotation_deg.yaw` | `ACTIVE` | effective camera yaw relative to drone body |

Current downstream concern:

- UE/runtime consumers and generated sensor-rig assets do not yet clearly expose or apply `front_rotation_deg.*`
- the current compiler intentionally fails fast if the runtime image fields are missing or malformed, so upstream config shape must stay aligned

Canonical rule for this stage:

- `SET_CONFIG` must carry the effective runtime image settings that Unreal will actually use, not just the authored source values
- the compiled runtime image block should fully describe the runtime camera that Unreal will use without requiring inference from any pose config side channel
- generated UE sensor-rig defaults may exist before config apply, but they are bootstrap values only and should be overwritten by the `SET_CONFIG` values on every configured run

### Stage 3: Capture Request Contract

Tooling owner:

- `Sampler_Manager.py`

Current and intended `CAPTURE_NOW` contract is minimal:

| Path | Status | Meaning |
| --- | --- | --- |
| top-level `type` | `ACTIVE` | message type `CAPTURE_NOW` |
| top-level `run_id` | `ACTIVE` | run trace key |
| top-level `schema_version` | `ACTIVE` | wire-contract version |
| top-level `seq` | `ACTIVE` | message ordering key |
| top-level `timestamp` | `ACTIVE` | capture request timestamp |
| top-level `drone_id` | `ACTIVE` | target drone for capture |
| top-level `capture_id` | `ACTIVE` | traceable capture identifier |
| `payload` | `ACTIVE` | empty object for v1 minimal capture trigger |

Current `Sampler_Manager.py` behavior on the tooling side:

1. read `viewpoints.capture_interval_seconds` from the viewpoint YAML, default `15.0` seconds
2. wait for bridge connection to UE
3. wait for bridge config-ready state owned elsewhere
4. generate the next `capture_id`
5. send one minimal `CAPTURE_NOW`
6. log `sent`
7. wait for matching `OBS`
8. log `received` or `failed`
9. repeat every configured interval

Canonical rule for this stage:

- `CAPTURE_NOW` is a trigger and correlation message, not a second config surface
- image behavior should come from the compiled runtime config already applied through `SET_CONFIG`
- `Sampler_Manager.py` should own capture cadence, `capture_id` generation, and `sent` / `received` / `failed` sample logging
- `Sampler_Manager.py` should not compile config or own the `SET_CONFIG` startup flow
- `Sampler_Manager.py` is the most practical tooling-side place to perform capture conformance validation after `OBS` is received, because it already owns capture correlation and sample acceptance for each `capture_id`

Recommended capture conformance validation for `Sampler_Manager.py`:

1. treat a returned sample as valid only if it can be tied to the active applied config identity for the run
2. compare returned image metadata against the expected applied runtime image settings for that run
3. flag the sample as `failed` if the returned observation does not carry enough config-linked metadata to prove conformance

This validation should stay narrow:

- `ws_bridge.py` should keep owning raw transport and raw persistence
- Unreal runtime should keep owning the actual capture behavior # need more detail for this (dont removed unless human arrpoeved)
- `Sampler_Manager.py` should only validate that the received sample can be trusted as having come from the applied config already in force for the run

### Stage 4: Unreal Runtime Resolution Contract

Current intended Unreal-side ownership:

| System | Role in image capture | Status |
| --- | --- | --- |
| `AWSConfigHandshakeActor` / `BP_SetDataConfig` | gate runtime actions on config-ready and forward capture intent into runtime systems | `ACTIVE` |
| `BP_SampleManager` | own atomic image capture orchestration | `LOOSE` |
| `BPI_DroneViewpointProvider.ListViewpoints` | enumerate available viewpoints | `LOOSE` |
| `BPI_DroneViewpointProvider.GetViewpointSnapshot` | return viewpoint metadata for the requested capture | `LOOSE` |
| `BP_DroneSensors` | own active viewpoint, FOV, width, height, and rig offsets used for capture | `LOOSE` |
| `ST_DroneViewpointSnapshot` | typed UE-side viewpoint snapshot schema | `LOOSE` |

Current intended `ST_DroneViewpointSnapshot` fields:

| Field | Meaning |
| --- | --- |
| `timestamp_utc` | snapshot timestamp |
| `run_id` | run trace key |
| `capture_id` | capture trace key |
| `viewpoint_name` | runtime viewpoint used |
| `fov_deg` | effective FOV |
| `width` | effective width |
| `height` | effective height |
| `rig_offset_from_drone_body_cm` | viewpoint rig offset |

Recommended canonical addition to `ST_DroneViewpointSnapshot`:

| Field | Status | Meaning |
| --- | --- | --- |
| `rig_rotation_from_drone_body_deg.pitch` | `OPEN` | effective camera pitch relative to the drone body |
| `rig_rotation_from_drone_body_deg.roll` | `OPEN` | effective camera roll relative to the drone body |
| `rig_rotation_from_drone_body_deg.yaw` | `OPEN` | effective camera yaw relative to the drone body |

Critical current gap:

- the live plugin path in `AWSConfigHandshakeActor::HandleCaptureNow(...)` currently bypasses `BP_SampleManager` and `BPI_DroneViewpointProvider`, emits a placeholder PNG, and does not include a viewpoint block in `OBS`

That means the current live runtime does not yet satisfy the intended image capture contract.

### Stage 5: Canonical `OBS` Image Contract

Current live image-related `OBS` fields:

| Path | Status | Meaning |
| --- | --- | --- |
| top-level `run_id` | `ACTIVE` | run trace key |
| top-level `drone_id` | `ACTIVE` | drone trace key |
| top-level `capture_id` | `ACTIVE` | capture trace key |
| `payload.timestamp_utc` | `ACTIVE` | payload timestamp |
| `payload.image_bytes_b64` | `ACTIVE` | base64 PNG bytes |
| `payload.pose.*` | `ACTIVE` | pose snapshot currently included |

Strict tooling-side validation now expects these additional `OBS` fields before an image sample can be accepted:

- `payload.config_ref.config_id`
- `payload.config_ref.config_hash`
- `payload.viewpoint.width`
- `payload.viewpoint.height`
- `payload.viewpoint.fov_deg`

Those are now part of the tooling-side image contract even though the current UE runtime does not yet emit them end to end.

Recommended canonical image-focused `OBS` shape:

| Path | Status | Purpose | Notes |
| --- | --- | --- | --- |
| `payload.viewpoint.fov_deg` | `OPEN` | effective FOV for this image | required by strict tooling validation; should reflect actual capture settings |
| `payload.viewpoint.width` | `OPEN` | effective width | required by strict tooling validation; should reflect actual capture settings |
| `payload.viewpoint.height` | `OPEN` | effective height | required by strict tooling validation; should reflect actual capture settings |
| `payload.viewpoint.rig_offset_from_drone_body_cm` | `OPEN` | viewpoint offset used | should reflect actual capture settings |
| `payload.viewpoint.rig_rotation_from_drone_body_deg.pitch` | `OPEN` | viewpoint pitch used | should reflect actual capture settings |
| `payload.viewpoint.rig_rotation_from_drone_body_deg.roll` | `OPEN` | viewpoint roll used | should reflect actual capture settings |
| `payload.viewpoint.rig_rotation_from_drone_body_deg.yaw` | `OPEN` | viewpoint yaw used | should reflect actual capture settings |
| `payload.image.bytes_b64` | `OPEN` | canonical image byte field | `payload.image_bytes_b64` should be treated as compatibility alias during transition |
| `payload.image.width` | `OPEN` | image width written to bytes | should match `viewpoint.width` |
| `payload.image.height` | `OPEN` | image height written to bytes | should match `viewpoint.height` |
| `payload.config_ref.config_id` | `OPEN` | effective config reference | required by strict tooling validation; should tie image directly to config identity |
| `payload.config_ref.config_hash` | `OPEN` | effective config reference | required by strict tooling validation; same reason |

Canonical rule for this stage:

- one returned image must be self-describing enough to answer which viewpoint, resolution, FOV, and config identity produced it
- that self-description is what allows `Sampler_Manager.py` to validate image-to-config conformance before accepting a sample as good

### Stage 6: Artifact Persistence Contract

Current tooling-side persistence behavior:

| Artifact | Current behavior | Status |
| --- | --- | --- |
| raw bridge image file | optionally written as `raw_captures/<capture_id>/raw__<drone_id>__<seq>.png` | `ACTIVE` |
| raw bridge observation record | optionally written to `raw_observations.jsonl` with `run_id`, `drone_id`, `capture_id`, `seq`, `raw_image_file`, and full raw message | `ACTIVE` |
| accepted sample image file | written by `Sampler_Manager.py` as `samples/images/<capture_id>/<drone_id>__<seq>__<width-height>__<interval>.png` after validation | `ACTIVE` |
| accepted sample record | written by `Sampler_Manager.py` to `samples/accepted_samples.jsonl` with config/viewpoint/image metadata and accepted sample path | `ACTIVE` |

Recommended canonical image artifact expectations:

| Artifact path or field | Status | Purpose | Notes |
| --- | --- | --- | --- |
| `raw_captures/<capture_id>/raw__<drone_id>__<seq>.png` | `ACTIVE` | raw bridge image artifact | transport-owned, optional, and directly traceable back to the raw observation record |
| `raw_observations.jsonl` -> `raw_image_file` | `ACTIVE` | direct lookup from raw observation record to raw image | bridge-owned audit/debug path |
| `samples/images/<capture_id>/<drone_id>__<seq>__<width-height>__<interval>.png` | `ACTIVE` | accepted image sample artifact | sampler-owned validated image output |
| `samples/accepted_samples.jsonl` -> `config_ref.config_id` / `config_ref.config_hash` | `ACTIVE` | direct config trace linkage for accepted samples | avoids losing runtime config context at the accepted sample layer |
| `samples/accepted_samples.jsonl` -> `viewpoint.width` / `viewpoint.height` / `viewpoint.fov_deg` | `ACTIVE` | accepted sample validation summary | lets downstream users inspect accepted image settings without re-parsing the raw message |

Canonical rule for this stage:

- `ws_bridge.py` should stay transport-focused and only own optional raw artifact persistence
- `Sampler_Manager.py` should own accepted image persistence and should only write accepted samples after config-ref and viewpoint validation passes
- the accepted sample filename should use the canonical pattern `<drone_id>__<seq>__<width-height>__<interval>.png`
- the artifact layer should preserve enough summarized image metadata that a dataset or audit tool can identify the image without re-parsing the full websocket message every time

### End-To-End Traceability Requirements For One Image

For one image capture, the contract should preserve this chain without ambiguity:

1. source viewpoint YAML path and its local `schema_version`
2. source viewpoint YAML content hash
3. compiled `SET_CONFIG.payload.sensor_rig` values actually sent to Unreal
4. `config_id` and `config_hash`
5. `capture_id`
6. `drone_id`
7. actual FOV, width, height, rig offset, and rig rotation used by Unreal
8. returned `OBS` image metadata and bytes
9. decoded image file path in runtime artifacts

If any one of those links is missing, image reproducibility becomes weaker.

### What Is Actively Defined Versus Missing

Actively defined today:

- viewpoint YAML exists
- `RunConfig.py` compiles width, height, and FOV into `SET_CONFIG`
- `Sampler_Manager.py` is the intended tooling owner of capture requests
- `Sampler_Manager.py` now validates returned image samples against `config_ref` plus viewpoint width, height, and FOV before accepting them
- `BPI_DroneViewpointProvider` and `ST_DroneViewpointSnapshot` define the intended Unreal viewpoint-query surface
- `ws_bridge.py` can persist raw `OBS` messages and decoded raw PNG files with raw-prefixed naming

Missing or ambiguous today:

- runtime use of the viewpoint-config rig offset values
- current UE runtime does not yet emit the required `payload.config_ref.*` and `payload.viewpoint.*` fields needed by strict tooling-side image validation
- explicit camera rig rotation contract from config through runtime and back into `OBS`
- a canonical `OBS.viewpoint` block on the live wire
- a canonical nested `OBS.image` block on the live wire
- runtime use of `BP_SampleManager` and `BPI_DroneViewpointProvider` in the live plugin capture path
- artifact summaries that expose viewpoint and image metadata directly

### Reusable Template For Pose And Telemetry

Pose and telemetry sections should mirror this same structure:

1. source config
2. compiled runtime config
3. capture request semantics
4. Unreal runtime resolution surfaces
5. canonical `OBS` payload block
6. artifact persistence contract
7. end-to-end traceability requirements
