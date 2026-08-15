# UE_Drone_Env/AGENTS.md

This is the Unreal Engine simulation project. It intentionally contains only what Unreal must own: assets, runtime Blueprints, and a thin WebSocket client plugin that exposes transport to Blueprints. All “recipe/config” lives outside (UE_Tooling) and is injected at runtime via `SET_CONFIG`.

## Non-negotiables
Blueprint assets (.uasset) must remain under a mounted content root (Project Content or Plugin Content). Do not move runtime Blueprints out of `Content/` expecting Unreal to load them.

Unreal does not parse YAML at runtime. Unreal receives a JSON `SET_CONFIG` message (built by tooling) and applies overrides in memory.

## Project map

### Config/
Standard Unreal project configuration (INI). This is not where YAML capture configs belong.

### Plugins/DroneWebSocket/
This is the minimal C++ bridge that lets Blueprints connect to and use WebSockets. It exists because low-level socket transport is not something you want implemented in Blueprint graphs.

- `DroneWebSocket.uplugin`  
  Plugin manifest. Unreal discovers and loads the plugin through this file.

- `Source/DroneWebSocket/DroneWebSocket.Build.cs`  
  Module build rules. Declares dependency on Unreal’s WebSockets module so the code compiles.

- `Public/WSClientComponent.h`  
  Blueprint-facing `UActorComponent` API: Connect/Send/Close plus events (OnConnected/OnMessage/OnClosed/OnError). This is what Blueprints attach and use.

- `Private/WSClientComponent.cpp`  
  Implementation: creates the socket, binds callbacks, and forwards incoming messages to Blueprint events.

- `WSProtocolTypes.h/.cpp` (optional)  
  Typed structs/enums and helper serialization. Use only if you decide to parse/construct typed messages in C++. If you keep JSON parsing in Blueprints, you can keep this lightweight.
  NOTE: We want to keep parsing lightweight but we dont want the same parcing system scatered accross scripts so we need to reason about this with the human developer before taking any action. 

### Content/
All runtime assets and Blueprints.

#### Course_Content_/
Course/environment assets (materials, meshes, markers). These should be as dependency-free as possible.

#### Drone_Content/ (the “drone kit”)
Reusable drone assets that can be shared across maps and scenarios.
- Meshes/: drone meshes and (optional) collision proxy.
- Materials/: base materials and instances.
- Blueprints/: drone container and capability modules:
  - `BP_DronePawn`: container Pawn that composes modules.
  - `BP_DroneSensors`: sensor rig / mounts (named transforms).
  - `BP_DroneMovement_6DOF`: movement module (ActorComponent) that applies commands.
  - `BP_DroneTelemetrySampler`: telemetry module (ActorComponent) that exposes state.
- Interfaces/: contracts external systems rely on:
  - Pose provider, viewpoint provider, telemetry provider, command receiver.
- Data/: defaults used when no runtime override exists:
  - `DA_DroneMovementDefault`
  - `DA_SensorRigProfileDefault`

The key rule: DA defaults are inert data; runtime objects copy these into variables, then overrides can replace those variables.

#### io/ (runtime wiring for orchestration + data capture)
This folder is where the sim is “driven” and “observed” at runtime.

- `io/Drone_Controller/`
  - `BP_DroneSpawner`: spawns one or many `BP_DronePawn`.
  - `BP_DroneController`: routes command messages to a specific drone (by drone_id/tag) and calls `BPI_DroneCommandReceiver`.

- `io/Data_Config/`
  - `ST_RunConfig`: the UE-side struct that matches the shape of `SET_CONFIG`.
  - `BP_SetDataConfig`: receives `SET_CONFIG` JSON over WebSocket, parses into `ST_RunConfig`, and applies overrides in memory. It must also store the override so future spawns inherit it during that run.

- `io/Data_Interface/`
  - `BP_SampleManager`: orchestrates “atomic snapshots” for traceability:
    - creates `capture_id`
    - queries Pose/Viewpoint/Telemetry via BPIs
    - triggers scene capture (manager-owned SceneCapture/RT pool recommended)
    - emits/records metadata + images
    - logs applied config_id/hash for every run

## Runtime injection model (how config actually applies)
Defaults live in `DA_DroneMovementDefault` and `DA_SensorRigProfileDefault`.

At runtime:
1) Tooling sends `SET_CONFIG` JSON.
2) `BP_SetDataConfig` receives it (via WSClientComponent), parses it into `ST_RunConfig`.
3) `BP_SetDataConfig` applies the override:
   - calls “ApplyMovementConfig” on `BP_DroneMovement_6DOF` (or through the pawn)
   - calls “ApplySensorRigProfile” on `BP_DroneSensors`
4) `BP_SampleManager` records config_id/hash + schema_version for traceability.

You do not mutate DA assets at runtime. You override the live variables those DAs initialize if cofig is provided.
Note: Yaml conflig can match default config and not nessesitate override of live variables. Reason with human developer on this item. 

## Viewpoints (“different perspectives”) and capture
Perspective selection is driven by:
- `BP_DroneSensors` mount transforms (one transform to start forward facing with offset rig)
- runtime config fields inside `ST_RunConfig` that specify mounts to capture and what camera params (FOV/resolution)

The `BP_SampleManager` should treat captures as “config-driven sampling,” not hardcoded viewpoints.

## Scaling to multiple drones
Do not create per-drone sockets. Route by `drone_id` inside messages. Spawner tags/IDs each drone. Controller and SampleManager use IDs to target commands/captures.

If control latency is critical while streaming heavy data, open a second WebSocket connection/channel for bulk data; keep control/config on a clean channel.
Note: this is not a mvp requirment. If streaming latency is impacted by data infrence bandwitch this approch will be added. reason with human developer on this. 

## Practical pitfalls
- If a Blueprint stops loading after moving assets
- Keep naming prefixes consistent (BP_, BPI_, DA_, SM_, M_, MI_) so searching stays sane.