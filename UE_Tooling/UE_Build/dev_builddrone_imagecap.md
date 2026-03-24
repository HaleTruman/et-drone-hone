# Dev Build Drone Image Capture

## Purpose

This brief defines the UE build-side implementation plan for image capture only.

The goal is to make the image capture path explicit from authored config to runtime capture to `OBS`, while keeping script responsibilities narrow and non-overlapping.

Paths in this document are relative to `UE_Tooling/UE_Build`.

## Scope

Included:

- generated UE content for drone/image/runtime IO setup
- websocket plugin behavior only where it affects image capture dispatch
- runtime config application required before a real image can be returned in `OBS`
- image-only `OBS` requirements needed by tooling validation

Not included yet:

- telemetry runtime graph-body completion outside the image path
- downstream tooling persistence details beyond what UE must emit

## Core Boundary Decisions

These boundaries should be treated as the working implementation contract.

- `run_*` scripts are orchestration only. They rebuild/apply generator outputs and write manifests. They do not define runtime image policy.
- shared helper scripts assemble assets and shared structure. They may define field names and asset relationships, but they should not own active run values.
- `DA_SensorRigProfileDefault` owns bootstrap defaults only. It exists so headless/editor startup has a workable sensor state before `SET_CONFIG`.
- `BPI_*` assets define query contracts only. They do not own state, defaults, or runtime apply behavior.
- `BP_DroneSensors` owns active camera-related runtime state and the actual capture hardware/software path for the drone.
- `BP_SetDataConfig` owns config ingress, validation, apply dispatch, and active config identity.
- `BP_SampleManager` owns atomic capture orchestration and `OBS` assembly.
- `WSConfigHandshakeActor.cpp` owns websocket transport ingress/egress and thin dispatch only. It must not assemble image `OBS` inline once the real runtime path exists.
- `BP_DronePawn` is the composition root. It should expose or host the runtime modules, not duplicate their logic.

## Canonical Symmetric Runtime Module Model

The top-level runtime contract should be symmetric:

- `ST_DroneCommandNormalized` <-> `BPI_DroneCommandReceiver` <-> `BP_DroneMovement_6DOF`
- `ST_DroneViewpointSnapshot` <-> `BPI_DroneViewpointProvider` <-> `BP_DroneSensors`
- `ST_DroneTelemetrySnapshot` <-> `BPI_DroneTelemetryProvider` <-> `BP_DroneTelemetrySampler`

`BP_DronePawn` remains the composition root only.

## Target Runtime Chain

The image capture path should behave like this:

1. tooling compiles image config into `SET_CONFIG`
2. `BP_SetDataConfig` receives `SET_CONFIG`
3. `BP_SetDataConfig` validates required runtime fields and stores active `config_id` / `config_hash`
4. `BP_SetDataConfig` applies image settings onto the live runtime sensor owner
5. `BP_DroneSensors` starts from `DA_SensorRigProfileDefault`, then overwrites those values with the applied run config
6. `CAPTURE_NOW` is routed into `BP_SampleManager`
7. `BP_SampleManager` resolves the target drone and its sensor/viewpoint provider path
8. `BP_SampleManager` asks the runtime sensor owner for:
   - a real image capture
   - the current viewpoint snapshot
9. `BP_SampleManager` assembles canonical `OBS`
10. returned `OBS` includes enough proof for tooling-side validation:
   - `payload.config_ref.config_id`
   - `payload.config_ref.config_hash`
   - `payload.viewpoint.width`
   - `payload.viewpoint.height`
   - `payload.viewpoint.fov_deg`
   - image bytes

## Current Live Problem

The current live plugin path does not satisfy the image capture contract.

Today:

- `AWSConfigHandshakeActor::HandleCaptureNow(...)` handles capture directly
- it resolves pawn transform directly
- it emits a hardcoded placeholder PNG
- it does not call `BP_SampleManager`
- it does not query `BPI_DroneViewpointProvider`
- it does not include `config_ref` or `viewpoint` metadata in `OBS`

That means current runtime behavior proves websocket transport only. It does not prove real image capture or config-conformant observation output.

## Role Split By Script

### Build Orchestration

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `Drone/run_drone_build.py` | Build run for drone-side generated assets | Run the ordered generator list, apply outputs to the UE project, and emit a clear generation artifact/manifest. | Owning image settings, default values, or runtime capture logic. | Keep this as a runner only. It should report build provenance and execution status, but should not embed the canonical runtime contract. |
| `IO/run_io_build.py` | Build run for IO-side generated assets and required map integration | Rebuild config/sample IO assets, place required IO ingress actors into the target map deterministically, and emit a build manifest. | Owning capture logic or image policy. | This is the right place for `BP_SetDataConfig` placement because placement is an IO build integration step, not a reusable asset generator concern. It should also record IO build provenance the same way the other runners do. |
| `WebSocket/run_websocket_build.py` | Plugin rebuild/validate run | Rebuild/version/validate websocket plugin build state after plugin-side changes. | Owning image-capture behavior. | No direct image-capture feature work is expected here right now. Keep it as the runner used after plugin changes. |

### Shared Build Helper

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `Drone/assemble_drone_assets.py` | Shared assembly helper for generated drone assets | Hold shared generation utilities, asset relationships, metadata stamping, and validation helpers. | Owning active sensor defaults, runtime image policy, or per-asset schema ownership. | Reduce this to shared mechanics only. Struct contracts should be owned by explicit `Structs/gen_st_*` generators, not by the helper. |

### Shared Struct Contracts

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `Drone/Structs/gen_st_droneviewpointsnapshot.py` | `ST_DroneViewpointSnapshot.uasset` | Own the canonical viewpoint snapshot schema used by interfaces and runtime consumers. | Owning runtime camera state or capture behavior. | Keep the full image-validation field set here, including rig offset and rig rotation, and make other UE build scripts consume this struct rather than redefining it. |
| `Drone/Structs/gen_st_dronecommandnormalized.py` | `ST_DroneCommandNormalized.uasset` | Own the normalized command schema. | Owning controller logic. | Keep it explicit for symmetry and runner-order clarity. |
| `Drone/Structs/gen_st_dronetelemetrysnapshot.py` | `ST_DroneTelemetrySnapshot.uasset` | Own the canonical top-level telemetry snapshot schema. | Owning runtime telemetry capture behavior. | Keep pose-aligned fields and telemetry detail records together here so the top-level `ST` / `BPI` / `BP` module boundary stays symmetric. |

### Bootstrap Runtime Defaults

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `Drone_Content/Data/gen_da_sensorrigprofiledefault.py` | `DA_SensorRigProfileDefault.uasset` | Generate the bootstrap/default sensor profile that exists before `SET_CONFIG` is applied. | Defining the authoritative run config. | Add a full bootstrap image shape, including rig rotation. Use clearly wrong-but-workable sentinels so bad startup/config wiring is obvious during validation. |

### Runtime Query Contracts

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `Drone_Content/Interfaces/gen_bpi_droneviewpointprovider.py` | `BPI_DroneViewpointProvider.uasset` | Define the query contract for viewpoint state and point at the explicit viewpoint snapshot struct. | Owning defaults, runtime image policy, applying config, or capturing images itself. | Ensure the snapshot contract includes all image-validation fields needed by tooling. This interface should expose function surfaces plus struct dependency only. |
| `Drone_Content/Interfaces/gen_bpi_dronetelemetryprovider.py` | `BPI_DroneTelemetryProvider.uasset` | Define the canonical top-level query contract for telemetry state. | Owning defaults, runtime telemetry policy, or image state. | Keep this aligned with `ST_DroneTelemetrySnapshot` so telemetry remains one top-level module contract. |

### Runtime State Owners

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `Drone_Content/Drone_Blueprints/gen_bp_dronesensors.py` | `BP_DroneSensors.uasset` | Own active camera state, scene capture resources, viewpoint naming, and the runtime functions that expose current image/viewpoint state for the drone. | Receiving websocket messages directly or acting as the top-level config ingress actor. | This is the key runtime state owner for image capture. It should initialize from `DA_SensorRigProfileDefault`, expose a clear apply function for runtime sensor rig config, own `SceneCaptureComponent2D`-style capture resources, and return the active snapshot values used in `OBS`. |
| `Drone_Content/Drone_Blueprints/gen_bp_dronepawn.py` | `BP_DronePawn.uasset` | Serve as the composition root that hosts the movement, sensors, and telemetry modules as real Blueprint component templates. | Re-implementing sensor logic, config ingress, or sample orchestration. | Keep this as a development step and composition concern. It should make the sensor owner reachable in a stable way through hosted module components, but it should not absorb image logic from `BP_DroneSensors` or `BP_SampleManager`. |
| `Drone_Content/Drone_Blueprints/gen_bp_dronetelemetrysampler.py` | `BP_DroneTelemetrySampler.uasset` | Own the canonical top-level telemetry runtime module, including pose-aligned telemetry output. | Blocking the first image-only milestone or remaining split across separate permanent runtime modules. | Keep `GetTelemetrySnapshot` as the canonical top-level function and carry pose-aligned telemetry fields inside `ST_DroneTelemetrySnapshot` rather than as a separate top-level runtime function. |

### Config Ingress

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `IO_Content/Data_Config/gen_st_runconfig.py` | `ST_RunConfig.uasset` or equivalent typed config surface | Define the typed UE-side shape for runtime config data used by apply logic. | Applying config itself. | Ensure the image-related `sensor_rig` fields match the canonical runtime contract so downstream Blueprints can apply them without ad hoc JSON field digging. |
| `IO_Content/Data_Config/gen_bp_setdataconfig.py` | `BP_SetDataConfig.uasset` | Own `SET_CONFIG` ingress, validation of required runtime fields, storage of active `config_id` / `config_hash`, and apply dispatch into runtime owners. | Owning capture logic or long-lived sensor state itself. | This is the owner of config ingress. It should validate the payload, find the target runtime owners, call their apply functions, and publish config-ready state for `BP_SampleManager`. This should be the only image-config ingress owner on the UE side. |

### Capture Orchestration

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `IO_Content/Data_Interface/gen_bp_samplemanager.py` | `BP_SampleManager.uasset` | Own atomic capture orchestration for `CAPTURE_NOW` and assemble canonical `OBS`. | Owning websocket transport, raw plugin ingress, or long-term config storage. | This is the in-engine owner of image capture orchestration. It should resolve the target drone, request the real image from the runtime sensor owner, request the viewpoint snapshot, stamp `config_ref`, and assemble `OBS` in the canonical shape expected by tooling. |

### Transport

| Script | Generated asset / runtime target | True role | Explicitly not responsible for | Image-capture work needed |
| --- | --- | --- | --- | --- |
| `WebSocket/Plugin_Source/DroneWebSocket/Source/DroneWebSocket/Private/WSConfigHandshakeActor.cpp` | Live websocket ingress/runtime handler | Own websocket transport ingress/egress and thin dispatch into the real UE runtime handlers. | Inline image capture, placeholder image assembly, or direct observation construction once runtime assets exist. | Replace the direct placeholder `HandleCaptureNow(...)` path with dispatch into `BP_SampleManager`. Keep this file thin. |

## What Each Generated UE Asset Should Actually Own

This is the shortest clean ownership model for image capture.

- `DA_SensorRigProfileDefault.uasset`
  - bootstrap image defaults only
  - intentionally wrong-but-workable sentinels allowed

- `BPI_DroneViewpointProvider.uasset`
  - query contract only
  - no stored defaults
  - no config apply logic

- `BP_DroneSensors.uasset`
  - active runtime camera state
  - default initialization from `DA_SensorRigProfileDefault`
  - runtime override application
  - viewpoint snapshot return
  - actual image capture resources and functions

- `ST_DroneTelemetrySnapshot.uasset`
  - canonical top-level telemetry payload contract
  - contains pose-aligned fields and telemetry detail records together

- `BPI_DroneTelemetryProvider.uasset`
  - canonical top-level telemetry query contract
  - no stored defaults
  - no runtime policy ownership

- `BP_DroneTelemetrySampler.uasset`
  - canonical top-level telemetry module
  - live telemetry/pose readback for later `OBS` assembly
  - canonical public surface is `GetTelemetrySnapshot`

- `BP_DronePawn.uasset`
  - composition root only
  - stable access to the runtime sensor owner

- `ST_RunConfig.uasset`
  - typed config shape only

- `BP_SetDataConfig.uasset`
  - config ingress
  - required field validation
  - active config identity
  - apply dispatch to runtime owners
  - config-ready state

- `BP_SampleManager.uasset`
  - `CAPTURE_NOW` orchestration
  - resolve target runtime owners
  - request current image + viewpoint state
  - assemble `OBS`
  - stamp `config_ref`

- `WSConfigHandshakeActor`
  - transport and dispatch only

Temporary compatibility assets retained only until telemetry convergence:


## Unreal Engine Concepts That Must Be Explicitly Covered

These are the UE concepts the implementation needs in order for the OBS image path to be real and maintainable.

1. Bootstrap defaults via a Data Asset
   - `DA_SensorRigProfileDefault` should exist so the project can boot and build headlessly before runtime config is applied.

2. Runtime sensor ownership
   - the live camera state must belong to a runtime owner, not to the transport layer
   - for this project, that owner should be `BP_DroneSensors`

3. Query contracts via Blueprint Interfaces
   - interfaces should define what can be asked of a runtime object
   - the implementing Blueprint should own the real data

4. Real image capture resources
   - the runtime sensor owner needs a real capture path, typically a `SceneCaptureComponent2D`-style setup and a render target or equivalent capture surface

5. Runtime image readback/encoding
   - after capture, the runtime path must read back or export the captured image and place it into the websocket `OBS` path in the agreed shape

6. Config apply lifecycle
   - defaults load first
   - `SET_CONFIG` overwrites them
   - `config_ready` should only be published after the live runtime state reflects the applied values

7. Runtime discovery/placement
   - `BP_SetDataConfig` and `BP_SampleManager` must be discoverable in a deterministic way
   - if either is placed in the level, that placement must be performed and validated by the domain build flow

8. `OBS` assembly ownership
   - the same runtime owner that orchestrates the capture should assemble the return object
   - transport should not invent image metadata on its own

## Detailed Execution Task List

Follow this order to implement the image-only path with minimal overlap.

### Symmetry Refactor Status

Phase 0. Lock the canonical symmetric runtime contract model in docs.
- current status: complete. The canonical top-level runtime model is now explicitly `ST` / `BPI` / `BP` by module for command, viewpoint, and telemetry, with pose and per-mesh proximity retained only as temporary compatibility assets until their consumers converge.

Phase 1. Refactor the canonical struct layer without breaking the current build.
- current status: complete. `ST_DroneTelemetrySnapshot` is the canonical top-level telemetry schema and carries pose-aligned fields plus telemetry detail records directly.

Phase 2. Refactor the BPI layer to match the canonical telemetry contract without breaking the current build.
- current status: complete. `BPI_DroneTelemetryProvider` is explicitly documented and generated as the canonical top-level telemetry query surface and depends only on `ST_DroneTelemetrySnapshot`.

Phase 3. Refactor the telemetry BP module to match the canonical telemetry contract without breaking the current build.
- current status: complete in a compatibility-safe form. `BP_DroneTelemetrySampler` is now generated as the canonical top-level telemetry runtime module, exposes `GetTelemetrySnapshot` as its canonical public function, and carries only the canonical `ST_DroneTelemetrySnapshot` template for schema traceability. Legacy pose-only surfaces still exist elsewhere temporarily until the remaining consumers converge.

1. Narrow the runners.
   - keep `run_drone_build.py`, `run_io_build.py`, and `run_websocket_build.py` orchestration-only
   - remove or demote any embedded contract summaries that look like a second source of truth
   - current status: complete. `run_drone_build.py` no longer embeds the canonical runtime contract, and `run_io_build.py` now emits a proper IO build manifest so the runner layer is symmetric enough for this stage

2. Narrow the shared assembly helper.
   - keep `assemble_drone_assets.py` focused on build assembly utilities, stable field names, and struct/interface shape definitions only
   - remove any value-level rig policy from it
   - current status: complete. `assemble_drone_assets.py` no longer embeds or stamps runtime `SET_CONFIG` field requirements, and the cross-audit of the other shared helpers found one secondary value helper in `assemble_course_assets.py`, which has been pushed back into the course-specific generator that owns that color choice

3. Complete bootstrap defaults.
   - update `gen_da_sensorrigprofiledefault.py`
   - include the full image field set, including rig rotation
   - use obviously wrong-but-valid bootstrap values so bad apply-config behavior is easy to detect
   - current status: complete. `DA_SensorRigProfileDefault` now defines the full bootstrap image field set on its generated class and asset instance, including rig rotation, with valid-but-visibly-wrong defaults so bad config-apply behavior is easier to detect during later runtime validation

4. Lock the viewpoint snapshot shape.
   - ensure the viewpoint snapshot structure contains the image fields tooling must validate against
   - keep this structural definition centralized and reused by the relevant generators
   - phase 1 status: complete. Struct ownership now lives in explicit `Drone/Structs/gen_st_*` generators, `run_drone_build.py` builds structs before interfaces, and the viewpoint struct generator owns the canonical field list including rig rotation
   - phase 2 status: complete. Interface consumers now load explicit struct assets and no longer carry image or telemetry defaults/policy in their interface metadata
   - phase 3 status: bounded. The struct generators and shared helper now stamp the active struct-member authoring mode onto generated struct assets. The local UE 5.7 source shows `FStructureEditorUtils` exists in C++, but the Unreal Python surface available to these scripts does not show a reflected `StructureEditorUtils` binding, so the generated struct assets remain explicit metadata-backed schema owners for now rather than verified real-member-authored structs

4.5. Future work: typed viewpoint interface signatures.
   - probe and, if supported cleanly, add typed signature authoring for `BPI_DroneViewpointProvider`
   - target signature intent:
     - `ListViewpoints -> Array<Name>`
     - `GetViewpointSnapshot(Name) -> ST_DroneViewpointSnapshot`
   - current status: deferred future work. A headless Unreal Python probe confirmed that the current editor surface can create interface graphs and load the struct asset, but does not expose `K2Node_FunctionEntry` / `K2Node_FunctionResult`, and the graph `nodes` property is protected. That means the current Python-only path is not a clean place to author or verify typed interface signatures
   - websocket reference for scope control: this was originally blocked by a pre-step11 transport path that bypassed `BP_SampleManager`. With step 11 complete, transport now dispatches to runtime sample-manager capture, but this typed-interface signature item is still useful for stronger contract enforcement inside Blueprint tooling.
   - recommended closure path later:
     - first preference: a small editor-surface helper or reflected Unreal API that can safely edit interface function signatures
     - second preference: a targeted C++ editor bridge if Unreal Python remains insufficient
   - do not expand this item into step 5 runtime ownership work. Keep it narrowly scoped to interface signature authoring and verification only

5. Lock `BP_DroneSensors` ownership.
   - define the required runtime variables for active image state
   - define a clear apply function for sensor rig config
   - define the runtime capture function(s)
   - define the snapshot-return function(s)
   - current status: bounded. `gen_bp_dronesensors.py` now makes `BP_DroneSensors` the declared runtime owner of active image state, loads bootstrap values from `DA_SensorRigProfileDefault`, carries the full front-view runtime field set including rig rotation, and exposes the public function surfaces `ApplySensorRigConfig`, `CaptureActiveViewpointPngBytes`, `ListViewpoints`, and `GetViewpointSnapshot`
   - bounded limitation: the generator currently creates the function graphs and variable/default ownership model, but not the full node-level graph bodies for real capture-resource behavior yet. That means step 5 now has the correct ownership boundary and runtime state model, but the actual config-apply graph logic and real image-capture graph logic still need to be authored in follow-on work

5.5. Future work: implement `BP_DroneSensors` graph bodies.
   - author the node-level behavior behind the step-5 public function surfaces without changing ownership boundaries
   - required follow-on work:
     - `ApplySensorRigConfig`
       - read the compiled runtime image fields
       - validate width / height / FOV / rig transform values
       - overwrite active in-memory image state
       - mark `runtime_config_applied = true` only on success
     - `CaptureActiveViewpointPngBytes`
       - use the active front-view runtime state, not bootstrap defaults
       - trigger real capture-resource behavior
       - return actual PNG bytes and real image dimensions
     - `GetViewpointSnapshot`
       - return the live runtime camera state derived from the active variables
       - include width, height, FOV, rig offset, and rig rotation
     - `ListViewpoints`
       - return the currently supported runtime viewpoints for v1, which should remain `front` only
   - scope guard:
     - do not move `config_id` / `config_hash` ownership into `BP_DroneSensors`
     - do not move `OBS` assembly into `BP_DroneSensors`
     - do not move sample acceptance or transport concerns into `BP_DroneSensors`
   - success state for 5.5:
     - `BP_DroneSensors` no longer only declares function graphs; it performs real runtime image-state apply, capture, and snapshot work inside those graphs
     - later steps can call into those functions directly instead of bypassing the sensor owner

6. Keep `BP_DronePawn` as composition only.
   - ensure the pawn exposes or hosts the sensor owner cleanly
   - do not move sample/config logic into the pawn
   - current status: complete. `gen_bp_dronepawn.py` now generates `BP_DronePawn` as a true composition root with hosted `BP_DroneMovement_6DOF`, `BP_DroneSensors`, and `BP_DroneTelemetrySampler` component templates authored directly onto the pawn through the Unreal `SubobjectDataSubsystem` editor path. The pawn no longer declares direct command, pose, viewpoint, or telemetry function surfaces, no longer carries image-default fields that belong to `BP_DroneSensors`, and now exposes `primary_sensor_component_name` / `primary_sensor_owner_role_name` instead of Blueprint-path module references

7. Complete the typed run config boundary.
   - update `gen_st_runconfig.py` so the UE-side typed shape matches the image config contract
   - current status: complete. `gen_st_runconfig.py` now stamps an explicit typed `SET_CONFIG` payload contract for `config_id`, `config_hash`, `movement`, `sensor_rig`, and `telemetry`, including required image `sensor_rig` fields and runtime apply-target mappings for `BP_DroneSensors` (`front_fov_deg`, resolution, offset, rotation, and active viewpoint)

8. Complete config ingress.
   - update `gen_bp_setdataconfig.py`
   - validate required image fields
   - store active `config_id` / `config_hash`
   - call into `BP_DroneSensors` apply behavior
   - publish config-ready only after success
   - current status: 8A and 8B complete. `WSConfigHandshakeActor` now routes `SET_CONFIG` through the explicit ingress seam and the default `OnSetConfigReceived` path validates required image fields, stores active `config_id` / `config_hash`, applies sensor-rig values onto live (and newly spawned) sensor owners, and emits `ACK` / `CONFIG_READY` only after successful apply. `gen_bp_setdataconfig.py` now stamps the typed required-path and runtime-apply-target contract metadata from `ST_RunConfig`
   - phase 8C status: complete (minimum truthful apply semantics). `BP_DroneSensors` is now generated on top of a native `DroneSensorsRuntimeComponent` parent that owns `ApplySensorRigConfig` validation semantics; `WSConfigHandshakeActor` no longer force-sets `runtime_config_applied=true` before calling apply. Instead it writes staged sensor-rig values, calls `ApplySensorRigConfig`, and treats apply as success only when `runtime_config_applied` is true after the call. If false, ingress rejects with the surfaced `config_apply_last_error` when available
   - phase 8D status: complete for hard-fail ingress policy. `WSConfigHandshakeActor::OnSetConfigReceived` now rejects `SET_CONFIG` when no live sensor owners are present at ingress apply time, emits `ERROR`, and does not publish `CONFIG_READY` for that run

8.5. Future work: harden ingress usability without weakening 8D.
   - keep the 8D hard-fail policy. Do not reopen the old "accept now, hope cached apply later fixes it" behavior when no live sensor owner exists
   - remaining concerns to close:
     - startup topology mismatch:
       - current ingress is now strict enough to reject `SET_CONFIG` when no live `BP_DroneSensors` owner exists yet
       - that is correct behavior, but it means any runtime flow that depends on post-config `SPAWN_DRONES` is now operationally incompatible with the ingress contract
       - future work must choose one deterministic pre-config owner strategy and build it into the runtime/build path:
         - preferred: a build-managed preplaced canonical `BP_DronePawn` with hosted `BP_DroneSensors`
         - acceptable fallback: an explicit bootstrap spawn path that completes before websocket connect and before `SET_CONFIG`
         - not acceptable: softening ingress so config is accepted with zero owners
     - bridge error lifecycle:
       - `ws_bridge.py` correctly marks the run failed when Unreal emits `ERROR` for rejected `SET_CONFIG`
       - but it still waits for the later `CONFIG_READY` timeout, which creates slower failure feedback and a second terminal error record
       - future hardening should let config-time `ERROR` terminate the run immediately with one clear failure reason
     - scope clarity on truth:
       - 8C and 8D now prove truthful config-apply validation and truthful ingress rejection behavior
       - they do not yet prove real capture-resource state, real viewpoint return values, or truthful `OBS` image payloads
       - that remaining truth boundary belongs to step 9 and step 11 and should not be claimed as already solved by config ingress alone
   - success state for 8.5:
     - hard-fail ingress remains intact
     - at least one live sensor owner exists deterministically before `SET_CONFIG`
     - config rejection causes prompt bridge termination without waiting for a second timeout
     - step 9 can assume ingress is both strict and operationally usable

9. Complete sample orchestration.
   - update `gen_bp_samplemanager.py`
   - resolve target drone/runtime sensor owner
   - request real image capture
   - request current viewpoint snapshot
   - assemble canonical `OBS` with `config_ref`
   - current repo-state review:
     - `BP_SampleManager` is still only a minimal shell. `gen_bp_samplemanager.py` currently creates one `CaptureNow` graph and a few scalar defaults, but no runtime orchestration body, no discovery contract, and no canonical `OBS` assembly logic
     - historical note: this was the pre-step11 behavior. It is now replaced by sample-manager dispatch and manager-owned `OBS` payload output.
     - `BP_DroneSensors` now owns the correct runtime image state variables and public function surfaces, but `CaptureActiveViewpointPngBytes` and `GetViewpointSnapshot` are still graph shells, so step 9 cannot be truthfully completed by wiring `BP_SampleManager` into empty functions
     - tooling-side strict validation already expects `payload.config_ref.config_id`, `payload.config_ref.config_hash`, and `payload.viewpoint.{width,height,fov_deg}`. That means step 9 is now the minimum runtime work required before a returned image observation can be accepted as conformant
     - `run_io_build.py` currently guarantees deterministic placement only for `BP_SetDataConfig`. It does not yet guarantee deterministic `BP_SampleManager` discovery, which is a real dependency for clean step-9 implementation
   - preconditions that must be treated as real:
     - step `8` ingress must stay strict. Step `9` must adapt runtime topology to that strict ingress model rather than weakening it
     - step `9` depends on the minimum callable subset of `5.5` being completed for the sensor owner:
       - `CaptureActiveViewpointPngBytes`
       - `GetViewpointSnapshot`
     - step `9` must stop transport-owned observation invention, but it must not yet perform the final websocket cutover. That remains step `11`
   - implementation stages required to complete step 9 without shortcuts:
     - 9A. Lock runtime discovery before authoring orchestration logic
       - intent:
         - define one deterministic way the runtime can find the authoritative sample orchestrator every run
         - remove ambiguity before any capture logic is authored
       - required development:
         - make `BP_SampleManager` a build-managed runtime actor, not an ad hoc discovery guess
         - preferred model: exactly one placed `BP_SampleManager` instance in the runtime map, created and validated by `run_io_build.py`
         - add an explicit label and stable identity metadata for that actor the same way `BP_SetDataConfig` is currently handled
         - ensure the manager can be discovered without depending on actor order, editor naming accidents, or class-path string scans at runtime
       - file scope expected:
         - `UE_Tooling/UE_Build/Content_Generation/IO/run_io_build.py`
         - `UE_Tooling/UE_Build/Content_Generation/IO/Data_Interface/gen_bp_samplemanager.py`
         - `UE_Tooling/UE_Build/Content_Generation/IO/assemble_io_assets.py` only if small shared placement helpers are needed
       - hard requirements:
         - one and only one authoritative manager instance after the IO build run
         - build manifest and placement result must state whether the manager was created, reused, or deduplicated
         - runtime must have a deterministic failure mode if the manager is missing or duplicated
       - non-goals:
         - do not wire websocket capture dispatch yet
         - do not let the plugin invent a fallback manager if the authoritative one is missing
       - validation required:
         - IO build places the manager deterministically
         - map reload still contains exactly one manager
         - runtime lookup can prove it resolved the expected actor instance
       - 9A status: complete.
         - `run_io_build.py` now owns deterministic singleton placement for both `BP_SetDataConfig` and `BP_SampleManager`
         - placement records now capture per-actor created/reused/deduplicated outcomes under `placements.placements_by_key`
         - `gen_bp_samplemanager.py` now stamps explicit runtime discovery identity metadata (`sample_orchestrator`, deterministic singleton mode, target label `BP_SampleManager_Main`)
         - validation evidence:
           - first validation pass: `IOBuild_v001_20260316_103935.json` created `BP_SampleManager_Main` and confirmed singleton count
           - second validation pass: `IOBuild_v001_20260316_104102.json` reused existing singleton for both required IO actors with `count_after=1`
     - 9B. Close the minimum callable sensor-owner gap that step 9 depends on
       - intent:
         - make the sensor owner callable in a way that returns truthful runtime capture facts rather than placeholder or bootstrap values
       - required development:
         - implement `BP_DroneSensors.CaptureActiveViewpointPngBytes`
           - consume the active in-memory image state already set by config ingress
           - drive real capture-resource behavior
           - return actual PNG bytes from the active front viewpoint
         - implement `BP_DroneSensors.GetViewpointSnapshot`
           - report the live runtime viewpoint state actually used for capture
           - include width, height, FOV, rig offset, and rig rotation
         - optionally implement `ListViewpoints` if the orchestration path needs explicit viewpoint validation before capture
       - file scope expected:
         - `UE_Tooling/UE_Build/Content_Generation/Drone/Drone_Blueprints/gen_bp_dronesensors.py`
         - `UE_Drone_Env/Source/UE_Drone_Env/DroneSensorsRuntimeComponent.h`
         - `UE_Drone_Env/Source/UE_Drone_Env/DroneSensorsRuntimeComponent.cpp`
         - additional narrow native helper files only if Unreal Python cannot author the required node bodies cleanly
       - hard requirements:
         - no bootstrap-default readback after config has been applied
         - no placeholder byte generation
         - no plugin-owned image fabrication
         - the returned viewpoint snapshot must describe the exact capture that just happened
       - design guardrails:
         - `BP_DroneSensors` owns sensor state and sensor capture only
         - it must not assemble `OBS`
         - it must not own `config_id` / `config_hash`
         - it must not own websocket send behavior
       - failure cases that must be explicit:
         - no valid active viewpoint
         - invalid capture dimensions
         - capture resource missing or not initialized
         - image readback/encoding failure
         - snapshot fields inconsistent with active runtime state
       - validation required:
         - positive test: valid config yields real PNG bytes and matching viewpoint snapshot
         - negative test: invalid runtime state yields explicit failure instead of silent fallback
         - decoded PNG dimensions match the returned viewpoint metadata
       - 9B status: complete.
         - `DroneSensorsRuntimeComponent` now owns truthful runtime capture implementation for:
           - `CaptureActiveViewpointPngBytes`
           - `GetViewpointSnapshot`
           - `ListViewpoints`
         - `gen_bp_dronesensors.py` now always resets function graphs (including empty desired sets) so stale Blueprint override shells cannot shadow native capture/snapshot behavior
         - runtime property access in `DroneSensorsRuntimeComponent` now resolves stable Blueprint field names by exact or `<name>_*` prefix, which closes name-stability issues from generated Blueprint variable internals
         - PNG encoding is now explicit via `FImageUtils::PNGCompressImageArray` (not legacy thumbnail compression), so returned image bytes are deterministic PNG payloads for downstream validation
         - native validation coverage now exists in `UE_Drone_Env/Source/UE_Drone_Env/Tests/DroneSensorsRuntimeStep9BTest.cpp` and directly exercises the real contract functions (not probe-only wrappers)
         - validation evidence:
           - content regeneration pass succeeded: `UE_Tooling/Artifacts/drone_content/DroneContentKit_v001_20260316_124453.json`
           - automation test pass (render-enabled run): `UE.Drone.ImageCapture.Step9B.DroneSensorsRuntimeCallable`
           - automation log: `/tmp/ue_step9b_automation_no_nullrhi.log`
           - automation report: `/tmp/ue_step9b_report_no_nullrhi`
         - validation note:
           - the same test fails under `-nullrhi` because render readback is disabled in that mode; 9B capture truth must be validated in a render-enabled headless run
     - 9C. Define the config-reference handoff explicitly
       - intent:
         - guarantee that the config identity stamped into `OBS` is the same identity that ingress actually applied
       - required development:
         - expose one authoritative runtime read seam for active config identity
         - make `BP_SampleManager` query that seam at capture time rather than caching or duplicating config state internally
         - define behavior for capture requests when config identity is unavailable or incomplete
       - file scope expected:
         - `UE_Tooling/UE_Build/Content_Generation/IO/Data_Config/gen_bp_setdataconfig.py`
         - `UE_Tooling/UE_Build/Content_Generation/IO/Data_Interface/gen_bp_samplemanager.py`
         - `UE_Tooling/UE_Build/WebSocket/Plugin_Source/DroneWebSocket/Source/DroneWebSocket/Private/WSConfigHandshakeActor.cpp` only if a narrow read accessor or helper is required for the manager to query active config identity
       - hard requirements:
         - one source of truth for active `config_id`
         - one source of truth for active `config_hash`
         - no shadow copies in `BP_SampleManager`
         - no default or placeholder config reference in `OBS`
       - design guardrails:
         - config ownership remains with ingress
         - `BP_SampleManager` reads config identity but does not own config lifecycle
       - validation required:
         - after `SET_CONFIG`, repeated captures all emit the same active `config_ref`
         - after a rejected config, capture cannot emit stale previous config identity
       - 9C status: complete for config-reference seam and stale-state safety.
         - `WSConfigHandshakeActor` exposes `GetActiveConfigReference(config_id, config_hash)` as the authoritative runtime read seam and clears that reference on new `SET_CONFIG` start and on rejected apply.
         - `gen_bp_setdataconfig.py` stamps ownership/read metadata so ingress remains the single source of truth.
         - `gen_bp_samplemanager.py` now defines non-cached per-capture read contract metadata and explicit `ResolveActiveConfigReference` surface.
       - validation evidence:
         - automation test pass (render-enabled run): `UE.Drone.ImageCapture.Step9C.ConfigReferenceHandoff`
         - automation log: `/tmp/ue_step9c_automation_20260316.log`
         - automation report: `/tmp/ue_step9c_report_20260316`
         - deterministic IO placement singleton reuse on explicit target map:
           - `UE_Tooling/Artifacts/io_build/IOBuild_v001_20260316_132133.json`
           - `UE_Tooling/Artifacts/io_build/IOBuild_v001_20260316_132334.json`
       - validation note:
         - `UE_Build/run_unreal_build_gen.py` still injects a legacy default `--level` unless explicitly overridden; for truthful 9A/9C validation runs, pass the versioned course-map path explicitly.
     - 9D. Author `BP_SampleManager` as orchestrator only
       - intent:
         - make `BP_SampleManager` the in-engine owner of one atomic capture transaction and nothing broader
       - required development:
         - resolve the target pawn from `drone_id`
         - resolve the hosted sensor module from the pawn using composition metadata such as `primary_sensor_component_name`
         - if telemetry is included now, resolve the hosted telemetry module through pawn composition metadata as well
         - call the sensor-owner image capture function
         - call the sensor-owner viewpoint snapshot function
         - if used, call the telemetry-owner snapshot function
         - assemble one canonical in-memory observation object from those returned values
         - ensure the manager treats this as one atomic capture operation:
           - one request
           - one image result
           - one viewpoint result
           - one config reference
           - zero partial success publication
       - file scope expected:
         - `UE_Tooling/UE_Build/Content_Generation/IO/Data_Interface/gen_bp_samplemanager.py`
         - `UE_Tooling/UE_Build/Content_Generation/Drone/Drone_Blueprints/gen_bp_dronepawn.py` only if small additional discovery metadata is required
         - `UE_Tooling/UE_Build/Content_Generation/Drone/Drone_Blueprints/gen_bp_dronetelemetrysampler.py` only if step-9 telemetry inclusion needs a callable runtime surface beyond the current shell
       - hard requirements:
         - manager owns orchestration, not sensor state
         - manager owns `OBS` assembly, not websocket transport
         - manager must fail closed on missing pawn, missing sensor module, missing config identity, or mismatched capture returns
       - design guardrails:
         - do not move capture-resource ownership into `BP_SampleManager`
         - do not reintroduce direct plugin reads of actor pose as the truth source for image observations
         - do not let manager bypass the pawn composition model by directly hardcoding asset paths to child modules
       - validation required:
         - a capture request against a known `drone_id` resolves the expected pawn and hosted sensor owner
         - the manager returns one assembled observation object or one explicit failure, never a partial observation
       - 9D status: complete.
         - native orchestration owner exists at `UE_Drone_Env/Source/UE_Drone_Env/SampleManagerRuntimeActor.{h,cpp}` and `BP_SampleManager` now inherits it.
         - `CaptureNow(...)` now performs one atomic transaction: pawn resolve, sensor resolve, config-ref resolve, image capture, viewpoint snapshot, and canonical observation assembly.
         - failure is fail-closed for missing pawn, missing sensor owner, missing config identity, snapshot/capture mismatch, and empty capture output.
         - `gen_bp_samplemanager.py` is constrained to defaults/metadata and does not author override function graphs (`EXPECTED_FUNCTIONS = []`), preserving native runtime truth.
         - ingress apply-order bug was closed in websocket config apply path by staging `ActiveSensorRigConfig = ParsedConfig` before live target apply.
       - validation evidence:
         - UE build pass after changes:
           - `Build.sh UE_Drone_EnvEditor Mac Development ...` (2026-03-18)
         - IO build pass on explicit versioned map:
           - `/tmp/ue_io_build_step9d_20260318_fix.log`
           - `UE_Tooling/Artifacts/io_build/IOBuild_v001_20260318_073223.json`
         - automation pass (render-enabled run):
           - `UE.Drone.ImageCapture.Step9B.DroneSensorsRuntimeCallable`
             - report: `/tmp/ue_step9b_report_20260318_fix13/index.json`
           - `UE.Drone.ImageCapture.Step9C.ConfigReferenceHandoff`
             - report: `/tmp/ue_step9c_report_20260318_fix13/index.json`
           - `UE.Drone.ImageCapture.Step9D.SampleManagerAtomicOrchestration`
             - report: `/tmp/ue_step9d_report_20260318_fix13/index.json`
       - validation notes:
         - current 9C/9D headless automation warnings about websocket send while disconnected (`cannot send ACK/CONFIG_READY/ERROR`) are expected in this harness and do not indicate runtime truth failure.
         - step 9 remains pre-cutover; live `HandleCaptureNow(...)` transport dispatch is intentionally deferred to step 11.
     - 9E. Define the `OBS` payload shape for the runtime seam, not just the transport seam
       - intent:
         - define the observation payload as a runtime truth contract rather than a plugin convenience object
       - minimum required fields for step 9:
         - `payload.config_ref.config_id`
         - `payload.config_ref.config_hash`
         - `payload.viewpoint.timestamp_utc`
         - `payload.viewpoint.run_id`
         - `payload.viewpoint.capture_id`
         - `payload.viewpoint.viewpoint_name`
         - `payload.viewpoint.fov_deg`
         - `payload.viewpoint.width`
         - `payload.viewpoint.height`
         - `payload.viewpoint.rig_offset_from_drone_body_cm`
         - `payload.viewpoint.rig_rotation_from_drone_body_deg`
         - image bytes
       - compatibility requirements:
         - continue emitting legacy `payload.image_bytes_b64` while also moving toward canonical `payload.image.bytes_b64`
         - if image width and height are emitted in a canonical image block, they must match the viewpoint block and the decoded PNG dimensions
         - do not remove currently emitted trace/pose compatibility fields until tooling and runtime consumers are explicitly audited clear of them
         - if compatibility pose data remains, source it from the telemetry runtime owner or another explicit runtime owner, never from plugin-side fabricated reads
       - hard requirements:
         - every image observation must be self-describing enough for tooling to validate:
           - which config produced it
           - which viewpoint produced it
           - what resolution it used
           - what FOV it used
         - `OBS` must reflect the effective runtime values used for capture, not the authored YAML values in the abstract
       - design guardrails:
         - do not stamp fields the runtime did not actually resolve
         - do not satisfy strict validation by adding guessed metadata disconnected from the real capture
       - validation required:
         - strict schema validation passes
         - `Sampler_Manager.py` accepts the sample without special-case exemptions
         - PNG decode dimensions, viewpoint metadata, and active config identity all agree
      - 9E status: complete for the in-engine runtime seam.
        - canonical runtime observation assembly now lives in `UE_Drone_Env/Source/UE_Drone_Env/SampleManagerRuntimeActor.cpp::AssembleObservationJson(...)`
        - assembled payload now carries:
          - `config_ref.{config_id,config_hash}`
          - `viewpoint.{timestamp_utc,run_id,capture_id,viewpoint_name,fov_deg,width,height,rig_offset_from_drone_body_cm,rig_rotation_from_drone_body_deg}`
          - canonical `image.{encoding,bytes_b64,width,height}`
          - compatibility `image_bytes_b64`
        - payload assembly fails closed if image bytes are empty, dimensions are invalid, or JSON serialization fails
      - validation evidence:
        - automation pass (render-enabled run):
          - `UE.Drone.ImageCapture.Step9D.SampleManagerAtomicOrchestration`
            - report: `/tmp/ue_step9d_report_20260318_fix13/index.json`
        - decoded PNG dimensions matched viewpoint metadata in the 9D automation assertions
     - 9F. Keep the live transport cutover out of step 9 until orchestration is real
       - intent:
         - protect the implementation from a common shortcut: making the plugin look conformant before the runtime actually is conformant
       - required development stance:
         - step `9` should complete the in-engine manager orchestration seam first
         - step `11` should be the transport cutover where `HandleCaptureNow(...)` stops emitting placeholder `OBS` and dispatches into the real manager
       - hard prohibitions during step 9:
         - do not partially rewrite `HandleCaptureNow(...)` to synthesize `config_ref` or `viewpoint` in C++ just to satisfy validators
         - do not duplicate sample assembly logic in both `BP_SampleManager` and `WSConfigHandshakeActor`
         - do not introduce a second observation schema path for "temporary" plugin-owned capture
       - what is acceptable in step 9:
         - narrow plugin/helper changes only when they are required to enable manager discovery or manager invocation later
         - native helper work inside the sensor owner if Unreal Python cannot author truthful capture nodes cleanly
       - validation required before step 11 begins:
         - the in-engine orchestration path can already produce one complete observation object locally
         - any remaining gap to live wire behavior is transport dispatch only, not missing runtime truth
      - 9F status: complete as a historical guardrail phase, now superseded by step 11 cutover.
        - the pre-cutover helper seams were intentionally removed during step 11 implementation.
        - transport now dispatches to the runtime sample manager and no longer emits plugin-owned placeholder image payloads.
   - success state for step 9:
     - `BP_SampleManager` can be resolved deterministically at runtime
     - it obtains image bytes and viewpoint data from real runtime owners
     - it assembles an `OBS` payload that matches the applied runtime config
     - strict tooling-side validation can accept the returned image sample
     - no config, viewpoint, or image facts are invented inside the websocket transport layer

10. Clarify placement/discovery.
    - current status: complete for the IO runtime actors required by this stage.
    - `BP_SampleManager` is now a placed deterministic level singleton, not an ad hoc runtime spawn and not a discovery guess through another owner.
    - `run_io_build.py` is the placement owner for both:
      - `BP_SetDataConfig_Main`
      - `BP_SampleManager_Main`
    - `gen_bp_samplemanager.py` now stamps discovery metadata only and explicitly points placement ownership at `run_io_build.py`; it does not place actors itself.
    - current validation evidence:
      - `IOBuild_v001_20260316_103935.json` created `BP_SampleManager_Main` and confirmed singleton placement
      - `IOBuild_v001_20260316_104102.json` reused existing singleton instances with `count_after=1`
      - `IOBuild_v001_20260318_081000.json` reused both required IO actors with `count_after=1`
    - remaining boundary:
      - this closes placement/discovery for the IO actors only
      - it does not yet solve the broader startup-orchestration question of how a live pre-config sensor owner is guaranteed to exist before strict `SET_CONFIG` ingress

11. Remove gameplay/runtime orchestration from the websocket plugin.
    - current status: complete (Option A implemented on 2026-03-21).
    - why this step is reopened:
      - the previous step `11` cutover solved the placeholder `OBS` problem, but it preserved the wrong architectural owner.
      - `WSConfigHandshakeActor` still owns gameplay/runtime action handling from inside the `DroneWebSocket` plugin, which is what creates pressure toward plugin-to-game coupling and reflection-based dispatch.
      - build order cannot solve that class-boundary issue; the ownership boundary itself must change.
    - approved target boundary:
      - `DroneWebSocket` plugin owns transport only:
        - `UWSClientComponent`
        - `WSProtocolTypes`
        - websocket plugin build/bootstrap/validation tooling
      - gameplay/runtime orchestration moves into the `UE_Drone_Env` game module:
        - `SET_CONFIG` ingress
        - `SPAWN_DRONES` dispatch
        - `CMD` dispatch
        - `CAPTURE_NOW` dispatch into `BP_SampleManager`
      - `BP_SetDataConfig` remains the placed runtime ingress asset, but its parent class must become a new game-module runtime actor rather than `WSConfigHandshakeActor`.
    - file ownership decision:
      - do not move this behavior into `UE_Tooling/UE_Build/WebSocket/assemble_websocket_assets.py`
        - that file is a tooling-side bootstrap/build helper only
        - it should remain responsible for plugin source staging, project build invocation, and plugin startup validation
        - putting live gameplay/runtime orchestration there would collapse build-time and runtime boundaries
      - keep `UE_Tooling/UE_Build/Content_Generation/IO/Data_Config/gen_bp_setdataconfig.py` as the generator owner of `/Game/io/Data_Config/BP_SetDataConfig`
        - this is still the correct tooling-side owner for the placed ingress asset
        - the required change is to point it at a new game-module parent class
      - keep `UE_Tooling/UE_Build/Content_Generation/IO/run_io_build.py` as the placement owner for `BP_SetDataConfig_Main`
      - keep `UE_Tooling/UE_Build/WebSocket/run_websocket_build.py` and related websocket tooling focused on plugin bootstrap/build/validation only
    - required runtime implementation shape:
      - add a new game-module C++ runtime ingress/orchestration actor under `UE_Drone_Env/Source/UE_Drone_Env/`
        - recommended file pair: `SetDataConfigRuntimeActor.h/.cpp`
        - reason for a new class:
          - no existing `UE_Drone_Env` runtime class currently owns websocket ingress plus action dispatch with the correct boundary
          - `SampleManagerRuntimeActor` should remain atomic sample owner only
          - `DroneController`/`DroneSpawner` paths should remain control/spawn owners only
      - the new game-side actor should:
        - own the websocket client component reference from the plugin
        - bind socket connect/close/error/message delegates
        - parse transport envelopes and gate actions on config-ready state
        - apply `SET_CONFIG` to live drone sensor owners
        - resolve live runtime owners and make direct typed calls to:
          - sample manager for `CAPTURE_NOW`
          - spawner/controller runtime owners for spawn/control flows
      - the plugin must stop owning any direct knowledge of:
        - `SampleManagerRuntimeActor`
        - drone pawn resolution
        - sensor component resolution
        - config apply semantics
        - observation assembly semantics
    - expected repo changes for this step:
      - `UE_Drone_Env/Source/UE_Drone_Env/UE_Drone_Env.Build.cs`
        - promote the `DroneWebSocket` dependency out of the editor-only block so the game runtime actor can use plugin transport types in live runs
      - new `UE_Drone_Env/Source/UE_Drone_Env/SetDataConfigRuntimeActor.h/.cpp`
        - migrate gameplay/runtime logic out of `WSConfigHandshakeActor`
        - keep direct typed calls inside the game module only
      - `UE_Tooling/UE_Build/Content_Generation/IO/Data_Config/gen_bp_setdataconfig.py`
        - change the generated parent class from `WSConfigHandshakeActor` to the new game-module runtime ingress actor
        - preserve the existing `/Game/io/Data_Config/BP_SetDataConfig` asset path and placement contract
      - websocket plugin source under `UE_Drone_Env/Plugins/DroneWebSocket/...`
        - thin to transport-only surfaces
        - deprecate or remove `WSConfigHandshakeActor` after all runtime references are migrated
      - step `9C/9D/11` automation tests
        - update includes, parent-class expectations, and dispatch expectations to use the new game-owned ingress path
    - implemented repo state:
      - `UE_Drone_Env/Source/UE_Drone_Env/SetDataConfigRuntimeActor.h/.cpp` now owns runtime websocket ingress, strict `SET_CONFIG` apply, `SPAWN_DRONES`, `CMD`, and `CAPTURE_NOW` dispatch.
      - `UE_Drone_Env/Source/UE_Drone_Env/UE_Drone_Env.Build.cs` now includes runtime `DroneWebSocket` dependency outside editor-only scope.
      - `UE_Tooling/UE_Build/Content_Generation/IO/Data_Config/gen_bp_setdataconfig.py` now parents `BP_SetDataConfig` to `SetDataConfigRuntimeActor`.
      - `UE_Drone_Env/Plugins/DroneWebSocket/.../WSConfigHandshakeActor.*` and `UE_Tooling/UE_Build/WebSocket/Plugin_Source/.../WSConfigHandshakeActor.*` are now transport-only compatibility surfaces (no gameplay/runtime orchestration).
      - `UE_Drone_Env/Source/UE_Drone_Env/Tests/ConfigReferenceStep9CTest.cpp`, `SampleManagerRuntimeStep9DTest.cpp`, and `TransportDispatchStep11Test.cpp` now target `ASetDataConfigRuntimeActor`.
    - explicit phased execution order for this step:
      - phase `11A`: complete
      - phase `11B`: complete
      - phase `11C`: complete
      - phase `11D`: complete
      - phase `11E`: complete
    - validation evidence for step `11` completion:
      - `BP_SetDataConfig_Main` placement remained deterministic via `run_io_build.py` (`IOBuild_v001_20260321_065312.json`).
      - websocket plugin tooling path validated end-to-end via `run_websocket_build.py` (`WebSocketPlugin_v011_step11_optiona_20260321_070049`, status `success`).
      - headless automation pass results:
        - `UE.Drone.ImageCapture.Step9C.ConfigReferenceHandoff` -> `Success`
        - `UE.Drone.ImageCapture.Step9D.SampleManagerAtomicOrchestration` -> `Success`
        - `UE.Drone.ImageCapture.Step11.TransportDispatchToSampleManager` -> `Success`
      - `CAPTURE_NOW` dispatch now stays inside the game module and uses typed `ASampleManagerRuntimeActor` calls (no plugin-side reflection invoke path).
      - websocket plugin handshake actor no longer owns `SET_CONFIG` apply, sensor resolution, pawn commanding, or observation assembly.

12. Rebuild and validate.
    - regenerate drone content
    - regenerate IO build
    - rebuild websocket plugin
    - run a capture that proves:
      - config was applied
      - returned image is real
      - returned `OBS` contains `config_ref` and `viewpoint` fields
    - current status: complete via runtime orchestration flow.
      - runtime operator command:
        - `python3 UE_Tooling/run_unreal_io.py --build-manifest UE_Tooling/Artifacts/build/UEBuild_v010_20260322_184500.json --runtime-run-id phase10_liveproof_20260322_1928 --config-timeout-seconds 120 --unreal-startup-timeout-seconds 120 --bridge-probe-seconds 5`
      - runtime evidence artifact:
        - `UE_Tooling/Artifacts/runtime/phase10_liveproof_20260322_1928/runtime_session_manifest.json`
      - operator-facing summary fields now present in runtime manifest:
        - `operator_summary.phase_status_by_name`
        - `operator_summary.capability_status_by_domain`
        - `operator_summary.evidence_paths`
      - capture proof in same runtime manifest:
        - `phase_results.phase7_sampler_smoke_capture.capture_evidence.validation_passed = true`
        - accepted sample file and raw image file paths populated

## Immediate Next Build Decisions To Lock

These decisions should be locked before implementation expands.

- `BP_DroneSensors` owns active camera state and the real capture resources.
- `BP_SetDataConfig` owns config ingress and apply dispatch.
- `BP_SampleManager` owns atomic image capture orchestration and `OBS` assembly.
- the websocket plugin stays transport-only; the placed game-side ingress actor owns runtime config/action dispatch.
- `assemble_drone_assets.py` stays structural, not policy-owning.
- runner scripts stay orchestration-only.
- `run_io_build.py` should own placement of required IO actors, but should not silently absorb extra runtime responsibilities beyond deterministic map integration.

## Minimum Definition Of Done For Image Capture

The image capture build path is not done until all of the following are true:

- `SET_CONFIG` image fields are applied to live runtime state before capture
- `DA_SensorRigProfileDefault` is only bootstrap state, not the authoritative run state
- `CAPTURE_NOW` is routed into `BP_SampleManager`
- `BP_SampleManager` gets viewpoint state through `BPI_DroneViewpointProvider`
- `BP_SampleManager` gets a real image through the runtime sensor owner
- returned `OBS` contains:
  - `payload.config_ref.config_id`
  - `payload.config_ref.config_hash`
  - `payload.viewpoint.width`
  - `payload.viewpoint.height`
  - `payload.viewpoint.fov_deg`
  - image bytes
- the returned values match the applied runtime config for the run
- the websocket plugin no longer emits the hardcoded placeholder PNG for image capture

## Unreal Implementation Notes

These notes are consistent with the current boundary recommendations and with standard Unreal patterns:

- Data Assets are appropriate for authored/bootstrap default data, not for owning per-run applied runtime state.
- Blueprint Interfaces are appropriate for runtime query contracts, but the implementing object should own the actual state.
- actor-owned runtime functionality should live in the relevant runtime owner rather than the transport layer.
- real image capture should come from a runtime capture component/render target path, not from placeholder transport code.

Official UE references reviewed for this brief:

- Data Assets: <https://dev.epicgames.com/documentation/en-us/unreal-engine/data-assets-in-unreal-engine>
- Blueprint Interfaces: <https://dev.epicgames.com/documentation/en-us/unreal-engine/blueprint-interface-in-unreal-engine>
- Components: <https://dev.epicgames.com/documentation/en-us/unreal-engine/adding-components-to-an-actor-in-unreal-engine>
- `USceneCaptureComponent2D`: <https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Engine/Components/USceneCaptureComponent2D>
- `FImageUtils::GetRenderTargetImage`: <https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Engine/FImageUtils/GetRenderTargetImage/1>

## Comment Restatement And Response

1. Restated comment: `run_drone_build.py` should only apply the generated scripts to the project during build and should not own image capture settings.
   Response: Agreed. The runner now stays focused on orchestration, logging, and build provenance. It should not embed or publish the canonical runtime contract.

2. Restated comment: `assemble_drone_assets.py` is a shared build helper and should not contain rig-specific settings.
   Response: Agreed with one refinement. It can still own structural definitions such as shared field names and struct shape, but it should not contain active values or rig policy.

3. Restated comment: `gen_da_sensorrigprofiledefault.py` should produce workable defaults that are clearly wrong, such as `222x333`, so bad config application is easy to detect.
   Response: Agreed. That is the right role for the bootstrap default asset and it gives the sample-validation path a clear signal when runtime config was not really applied.

4. Restated comment: the viewpoint provider path should default to `DA_SensorRigProfileDefault` values and then be overwritten during config apply pre-flight.
   Response: Agreed on the behavior, but the state should live in the implementing runtime asset, not in the interface itself. In practice that means `BP_DroneSensors` should initialize from the data asset, then be overwritten by `BP_SetDataConfig`.

5. Restated comment: telemetry and pose ownership should converge under the canonical top-level telemetry contract so the runtime module boundaries stay symmetric and predictable.
   Response: Agreed. I marked it as structurally relevant but deferred for the current image-only milestone.

6. Restated comment: `gen_bp_dronesensors.py` is unclear and we need to lock down whether it is responsible for applying config.
   Response: My recommendation is that it should own the active camera state and expose an apply function, but it should not be the top-level config ingress owner. `BP_SetDataConfig` should receive config and call into `BP_DroneSensors` to apply it.

7. Restated comment: `gen_bp_dronepawn.py` should remain a development step rather than becoming the place where image logic gets stuffed.
   Response: Agreed. I kept it as the composition root only.

8. Restated comment: `gen_bp_dronetelemetrysampler.py` should stay acknowledged.
   Response: Agreed. It remains in the plan for structural alignment, but not as first-path image work.

9. Restated comment: `gen_st_runconfig.py` needs to exist as the typed config boundary.
   Response: Agreed. I kept it as the typed shape owner only and not as the place where config is applied.

10. Restated comment: `gen_bp_setdataconfig.py` needs clearer scope because several files currently seem to share responsibilities.
    Response: Agreed. I narrowed it to one job: config ingress, required field validation, active config identity, and apply dispatch. That should remove most of the current ambiguity.

11. Restated comment: `gen_bp_samplemanager.py` is the right place for the atomic sample work.
    Response: Agreed. I made it the in-engine owner of `CAPTURE_NOW` orchestration and `OBS` assembly.

12. Restated comment: `run_io_build.py` is fine conceptually.
    Response: Agreed. I kept it as a build runner and IO build integration owner. It now owns deterministic `BP_SetDataConfig` placement without taking on runtime capture logic.

13. Restated comment: map placement for `BP_SetDataConfig` needs clarification.
    Response: Agreed. The cleanest resolution is to treat placement as part of the IO build flow, not as a separate asset generator and not as a separate wiring layer for this case. `run_io_build.py` should own deterministic placement of `BP_SetDataConfig`, and any future `BP_SampleManager` placement should be an explicit addition to that same IO build flow only if the runtime architecture requires a placed actor.

14. Restated comment: `WSConfigHandshakeActor.cpp` needs to be fixed after the other files are ready to send and receive the right data.
    Response: Agreed. I kept it late in the execution order because it should only thin down to dispatch once the runtime assets behind it are real.

15. Restated comment: `run_websocket_build.py` does not seem to require work right now.
    Response: Agreed. I left it in the brief only as the rebuild/validation runner used after plugin changes, not as an image-capture implementation owner.
