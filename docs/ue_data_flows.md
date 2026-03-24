# UE Drone Runtime Data Flows

This document describes the current live runtime-flow model across four core
scenarios:

1. drone spawning
2. drone config
3. drone controlling
4. data inference / capture

It is a global runtime-flow document, not an image-only note. Image capture is
currently the most proven implementation seam, but the same top-level build,
runtime, transport, and Unreal ingress layers also define how spawn and command
flows should extend.

## Scope

These flows describe runtime behavior and the immediate build/runtime seam that
enables runtime behavior.

That means this document includes:

- the build-layer outputs required before runtime can work
- the runtime wrapper that supervises Unreal launch and websocket startup
- the scenario-specific action and response paths once the runtime session is up

It does not try to re-document every field in the wire contract. For exact
message-level field truth, use [data_contract.md](data/data_contract.md).

## Live Execution Context

All four runtime scenarios depend on the same upstream seam:

1. `UE_Tooling/run_unreal_build.py`
   - orchestrates WebSocket -> Course -> Drone -> IO build stages
   - writes the canonical build manifest under
     `UE_Tooling/Artifacts/build/<run_name>.json`
   - records the authoritative course level path and IO placement truth
2. `UE_Tooling/UE_Build/Content_Generation/IO/run_io_build.py`
   - performs deterministic singleton placement in the authoritative level for:
     - `BP_SetDataConfig_Main`
     - `BP_SampleManager_Main`
     - `BP_DronePawn_Startup_Main`
3. `UE_Tooling/run_unreal_io.py`
   - consumes the build manifest
   - starts the websocket bridge
   - launches Unreal against the authoritative level
   - compiles or loads `SET_CONFIG`
   - applies config and validates startup topology
   - then supervises scenario actions on top of that ready runtime

That shared seam matters because spawn, command, and capture should all extend
the same runtime ownership model instead of bypassing it with ad hoc launch or
placement logic.

## Shared Ownership Model

### Tooling Side

- `run_unreal_build.py` owns top-level build orchestration and build handoff.
- `run_unreal_io.py` owns top-level runtime orchestration and session evidence.
- `RunConfig.py` owns deterministic `SET_CONFIG` compilation from YAML.
- `Drone_Spawner.py` owns `SPAWN_DRONES` action shaping and current spawn-capability probe semantics.
- `Drone_Controller.py` is the intended tooling owner of `CMD` shaping, but its
  higher-level control loop is still pending.
- `Sampler_Manager.py` owns `CAPTURE_NOW`, image validation, and accepted sample persistence.
- `protocol.py` owns the canonical wire envelope and type tokens.
- `ws_bridge.py` owns transport, config gating, raw artifact persistence, and
  in-memory observation/status handoff.

### Unreal Side

- `UWSClientComponent` owns the websocket client connection.
- `ASetDataConfigRuntimeActor` / `BP_SetDataConfig` owns runtime ingress:
  - `SET_CONFIG`
  - config gating
  - dispatch for `SPAWN_DRONES`, `CMD`, and `CAPTURE_NOW`
  - outbound `ACK`, `CONFIG_READY`, `STATUS`, `OBS`, and `ERROR`
- `ASampleManagerRuntimeActor` / `BP_SampleManager` owns the current live
  capture orchestration path.
- `UDroneSensorsRuntimeComponent` / `BP_DroneSensors` owns active image state
  and real PNG capture for the current live capture path.
- `BP_DroneMovement_6DOF`, `BP_DronePawn`, `BP_DroneSensors`, and
  `BP_DroneTelemetrySampler` remain the runtime drone-side ownership surfaces
  that broader command/viewpoint/telemetry contracts should converge around.

## Scenario Status Summary

| Scenario | Live status | Current implementation seam |
| --- | --- | --- |
| Spawn | `ACTIVE` with bounded scope | live action path exists; top-level runtime currently uses it as a supervised capability probe |
| Config | `ACTIVE` | fully part of the current build/runtime startup path |
| Command | `MIXED` | Unreal-side receiver path is live; higher-level tooling/runtime supervision is still a later integration seam |
| Capture | `ACTIVE` | currently the strongest proved end-to-end runtime path |

## Scenario 1: Drone Spawning

### Live runtime flow

```text
Drone_Spawner.py
  -> protocol.py
  -> ws_bridge.py
  -> UWSClientComponent
  -> ASetDataConfigRuntimeActor::HandleSpawnDrones(...)
  -> spawn BP_DronePawn instances
     -> apply active sensor rig config to spawned pawns
  -> STATUS event: SPAWN_DRONES_ACCEPTED
```

### Current live ownership

- Tooling owner: `UE_Tooling/Drone_Controller/Drone_Spawner.py`
- Unreal ingress owner: `ASetDataConfigRuntimeActor`
- Runtime spawn owner: current spawn logic inside `HandleSpawnDrones(...)`

### Current live behavior

- `Drone_Spawner.py` builds canonical `SPAWN_DRONES` actions through
  `protocol.make_action(...)`.
- The current runtime path accepts:
  - `payload.spawn_count`
  - `payload.count` as compatibility alias
  - optional `payload.drone_ids[]`
- `ASetDataConfigRuntimeActor::HandleSpawnDrones(...)` currently:
  - loads the drone pawn class
  - spawns one or more pawns
  - spaces them using fixed runtime spacing
  - tags and tracks the spawned drone IDs
  - applies current sensor-rig config to spawned pawns when available
  - emits `STATUS` with event `SPAWN_DRONES_ACCEPTED`

### Current limits

- authored `spawn_points[]` are not the live path today
- the current runtime uses fixed spacing rather than full transform-aware spawn requests
- top-level runtime orchestration uses this as a capability proof today, not yet as the complete episode-spawn system

### Why this matters globally

Spawn should continue to extend from the existing runtime ingress and status
model. It should not create a second parallel startup or placement path outside
the current build/runtime seam.

## Scenario 2: Drone Config

### Live runtime flow

```text
movement/viewpoint/telemetry YAML
  -> RunConfig.py
  -> ws_bridge.py startup config flow
  -> UWSClientComponent
  -> ASetDataConfigRuntimeActor
  -> apply config to live sensor owners / runtime state
  -> ACK
  -> CONFIG_READY
```

### Current live ownership

- Tooling owner: `UE_Tooling/Config/RunConfig.py`
- Transport owner: `UE_Tooling/WebSocket/ws_bridge.py`
- Unreal ingress/config owner: `ASetDataConfigRuntimeActor`

### Current live behavior

- `RunConfig.py` compiles YAML into one deterministic `SET_CONFIG` payload.
- The build/runtime wrappers treat config apply as a startup prerequisite, not an optional side action.
- `ASetDataConfigRuntimeActor`:
  - validates the incoming config payload
  - records `config_id` and `config_hash`
  - applies `sensor_rig` settings to live sensor owners
  - emits `ACK`
  - emits `CONFIG_READY`
- `run_unreal_io.py` uses config apply and post-config topology validation as the
  foundation for all later runtime actions.

### Current live image/config seam

Image capture is the clearest proof that config is actually taking effect:

- viewpoint YAML defines width, height, FOV, rig offset, and rig rotation
- `RunConfig.py` compiles those into `payload.sensor_rig`
- Unreal applies those onto live sensor owners
- later capture validation checks returned `OBS` against the active applied config identity

That seam should remain the model for future telemetry and broader runtime
observation work.

## Scenario 3: Drone Controlling

### Live runtime flow

```text
Drone_Controller.py or future controller owner
  -> protocol.py
  -> ws_bridge.py
  -> UWSClientComponent
  -> ASetDataConfigRuntimeActor::HandleCmd(...)
  -> resolve target pawn by drone_id
  -> apply translation / rotation delta
  -> STATUS event: CMD_APPLIED
```

### Current live ownership

- Intended tooling owner: `UE_Tooling/Drone_Controller/Drone_Controller.py`
- Unreal ingress owner: `ASetDataConfigRuntimeActor`
- Runtime application owner today: current command logic inside `HandleCmd(...)`

### Current live behavior

- The Unreal-side command receiver path is live.
- `ASetDataConfigRuntimeActor::HandleCmd(...)` currently:
  - resolves `drone_id` from top-level field or payload fallback
  - finds the target pawn
  - reads `pitch`, `roll`, `yaw`, and `throttle`
  - applies world offset and rotation deltas directly
  - emits `STATUS` with event `CMD_APPLIED`

### Current limits

- `Drone_Controller.py` is still a documentation/scaffold surface, not the
  full top-level runtime control loop
- current command application is a bounded direct runtime implementation, not a
  final normalized movement-module pipeline
- top-level runtime supervision for command capability is still a later seam

### Why this matters globally

The command path should be completed by extending the same runtime ingress,
status, and ownership surfaces already used by config/spawn/capture. It should
not introduce a second control transport or bypass the current runtime wrapper.

## Scenario 4: Data Inference / Capture

### Live runtime flow

```text
Sampler_Manager.py
  -> protocol.py
  -> ws_bridge.py
  -> UWSClientComponent
  -> ASetDataConfigRuntimeActor::HandleCaptureNow(...)
  -> resolve BP_SampleManager / ASampleManagerRuntimeActor
  -> resolve UDroneSensorsRuntimeComponent
  -> capture real PNG + viewpoint snapshot + config_ref
  -> OBS
  -> ws_bridge.py raw artifacts
  -> Sampler_Manager.py validation + accepted sample persistence
```

### Current live ownership

- Tooling owner: `UE_Tooling/Data_Interface/Sampler_Manager.py`
- Unreal ingress owner: `ASetDataConfigRuntimeActor`
- Runtime capture owner: `ASampleManagerRuntimeActor`
- Runtime image producer: `UDroneSensorsRuntimeComponent`

### Current live behavior

- `Sampler_Manager.py` sends minimal `CAPTURE_NOW`.
- `ASetDataConfigRuntimeActor::HandleCaptureNow(...)` resolves the sample manager instead of emitting a placeholder image directly.
- `ASampleManagerRuntimeActor`:
  - resolves active config reference
  - captures a real rendered PNG from the active sensor viewpoint
  - reads the active viewpoint snapshot
  - assembles one observation payload
- The returned `OBS` now includes:
  - `config_ref`
  - `viewpoint`
  - canonical nested `image`
  - compatibility `image_bytes_b64`
- `ws_bridge.py` persists raw transport artifacts.
- `Sampler_Manager.py` validates image/config/viewpoint truth and writes accepted sample artifacts.

### Why capture is the current extension seam

Capture is the most complete current proof because it already uses:

- top-level build handoff
- top-level runtime startup and phase supervision
- config truth
- runtime ingress truth
- real returned observation payloads
- raw artifact persistence
- accepted sample persistence

That makes it the best implementation seam for extending broader runtime
behavior without changing the ownership model.

## How The Four Scenarios Fit Together

These scenarios should not be thought of as unrelated flows.

The current repo is building toward one layered model:

1. build produces the authoritative level and required singleton placements
2. runtime launches that level and reaches config-ready truth
3. spawn, command, and capture all dispatch through the same runtime ingress
4. runtime replies come back through the same websocket boundary
5. raw and accepted artifacts are persisted under the same runtime session root

Image capture is currently the strongest end-to-end proof of that model, but it
should remain the seam we extend from, not a special-case system that other
flows bypass.

## Current Practical Guidance

If you are reasoning about the live repo, treat the system like this:

- build layer owns authoritative level creation and singleton placement
- runtime layer owns websocket startup, Unreal launch, config apply, and runtime supervision
- spawn extends from that runtime layer and already has a bounded live path
- command extends from that runtime layer, but its broader tooling integration is still pending
- capture extends from that runtime layer and is currently the strongest
  end-to-end implemented path

## Summary

The current live repo is no longer just a target-state runtime sketch.

It already has:

- a canonical build wrapper
- a canonical runtime wrapper
- a live config path
- a live spawn path
- a live Unreal-side command path
- a live capture path with truthful returned `OBS` and accepted sample persistence

What remains is not to invent a second architecture for the other scenarios,
but to continue extending the same build/IO/runtime seam cleanly across spawn,
command, telemetry, and broader observation behavior.
