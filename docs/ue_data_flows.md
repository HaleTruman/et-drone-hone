# UE Drone Runtime Data Flows

This document explains the **live runtime message flow** through the current architecture for four core scenarios:

1. Drone spawning
2. Drone config (movement + sensors)
3. Drone controlling
4. Data inference / capture

These flows describe **runtime behavior only**. The `UE_Tooling/UE_Build/` scripts create the `.uasset` and other files ahead of time, but they do **not** directly control the simulation once Unreal is running.

---

## Key idea

At runtime, the system works like this:

* **UE_Tooling** creates commands and config payloads.
* **WebSocket** carries those messages between tooling and Unreal.
* **UE_Drone_Env** receives those messages and executes them through live Blueprint assets.
* **BP_DronePawn** and its modules provide movement, viewpoints, telemetry, and image capture support.
* **BP_SampleManager** assembles atomic snapshots for observation/capture.

---

## Runtime architecture summary

### Tooling side

* `UE_Tooling/Drone_Controller/Drone_Spawner.py` builds spawn requests.
* `UE_Tooling/Drone_Controller/Drone_Controller.py` builds movement/control requests.
* `UE_Tooling/Config/RunConfig.py` merges YAML config into one `SET_CONFIG` payload.
* `UE_Tooling/Data_Interface/Sampler_Manager.py` requests captures and manages observation collection.
* `UE_Tooling/Data_Interface/Sampler.py` decodes and stores observations.
* `UE_Tooling/WebSocket/protocol.py` defines the message envelope and message types.
* `UE_Tooling/WebSocket/ws_bridge.py` sends and receives runtime messages.

### Unreal side

* `BP_DroneSpawner.uasset` spawns drones.
* `BP_DroneController.uasset` routes commands to the correct drone.
* `BP_SetDataConfig.uasset` receives runtime config and applies overrides in memory.
* `BP_SampleManager.uasset` queries pose/viewpoints/telemetry and produces atomic captures.
* `BP_DronePawn.uasset` is the runtime drone container.
* `BP_DroneMovement_6DOF.uasset` applies movement commands.
* `BP_DroneSensors.uasset` provides viewpoint transforms and camera-related properties.
* `BP_DroneTelemetrySampler.uasset` provides telemetry data.
* `BPI_*` interfaces define the contracts that managers/controllers call.
* `DA_*Default` assets provide baseline defaults that can be overridden at runtime 

---

# Scenario 1: Drone Spawning

## Example JSON command

```json
{
  "type": "SPAWN_DRONES",
  "schema_version": "1.0",
  "run_id": "course_v1_20260311_120000",
  "seq": 1,
  "payload": {
    "count": 3,
    "drone_blueprint": "/Game/Drone_Content/Blueprints/BP_DronePawn",
    "spawn_points": [
      {"drone_id": "drone_001", "location": [0, 0, 150], "rotation": [0, 0, 0]},
      {"drone_id": "drone_002", "location": [300, 0, 150], "rotation": [0, 0, 0]},
      {"drone_id": "drone_003", "location": [600, 0, 150], "rotation": [0, 0, 0]}
    ]
  }
}
```

## Simple flow chart

```text
UE_Tooling/Drone_Controller/Drone_Spawner.py
  -> UE_Tooling/WebSocket/protocol.py
  -> UE_Tooling/WebSocket/ws_bridge.py
  -> UE_Drone_Env/Plugins/DroneWebSocket/Public/WSClientComponent.h
  -> UE_Drone_Env/Content/io/Drone_Controller/BP_DroneSpawner.uasset
  -> UE_Drone_Env/Content/Drone_Content/Blueprints/BP_DronePawn.uasset
     -> BP_DroneMovement_6DOF.uasset
     -> BP_DroneSensors.uasset
     -> BP_DroneTelemetrySampler.uasset
```

## Explanation

`Drone_Spawner.py` is the tooling-side entrypoint for spawn requests. It builds the spawn payload and passes it through `protocol.py`, which wraps it in the agreed runtime message shape, and `ws_bridge.py`, which sends it into Unreal.

Inside Unreal, `WSClientComponent.h` exposes the incoming message to the Blueprint runtime. `BP_DroneSpawner.uasset` interprets the payload and creates one or more `BP_DronePawn.uasset` instances in the level. Each pawn becomes a live drone actor and composes the movement, sensor, and telemetry modules so it is immediately controllable and observable.

---

# Scenario 2: Drone Config (movement + sensors)

## Example JSON command

```json
{
  "type": "SET_CONFIG",
  "schema_version": "1.0",
  "run_id": "course_v1_20260311_120000",
  "seq": 2,
  "payload": {
    "config_id": "cfg_001",
    "movement": {
      "max_speed": 1400.0,
      "max_accel": 600.0,
      "yaw_rate_limit": 90.0,
      "pitch_rate_limit": 60.0,
      "roll_rate_limit": 60.0,
      "damping": 0.12
    },
    "sensors": {
      "active_viewpoints": ["front"],
      "front_fov": 90.0,
      "capture_width": 1280,
      "capture_height": 720,
      "telemetry_fields": ["location", "rotation", "velocity"]
    }
  }
}
```

## Simple flow chart

```text
UE_Tooling/Config/Drone_Controller_Config/Config_DroneMovementTuning.yaml
+ UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Pose.yaml
+ UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml
+ UE_Tooling/Config/Data_Interface_Config/SensorRigProfileConfig_Telemetry.yaml
  -> UE_Tooling/Config/RunConfig.py
  -> UE_Tooling/WebSocket/protocol.py
  -> UE_Tooling/WebSocket/ws_bridge.py
  -> UE_Drone_Env/Plugins/DroneWebSocket/Public/WSClientComponent.h
  -> UE_Drone_Env/Content/io/Data_Config/BP_SetDataConfig.uasset
  -> UE_Drone_Env/Content/io/Data_Config/ST_RunConfig.uasset
  -> UE_Drone_Env/Content/Drone_Content/Blueprints/BP_DronePawn.uasset
     -> BP_DroneMovement_6DOF.uasset
     -> BP_DroneSensors.uasset
     -> DA_DroneMovementDefault.uasset
     -> DA_SensorRigProfileDefault.uasset
```

## Explanation

The YAML files in `UE_Tooling/Config/` are the human-editable source configuration. `RunConfig.py` merges them into one `SET_CONFIG` JSON payload, then `protocol.py` and `ws_bridge.py` transport that payload into Unreal.

Inside Unreal, `BP_SetDataConfig.uasset` receives the message and parses it into `ST_RunConfig.uasset`, which acts as the typed runtime config structure. It then applies the override in memory to the live drone systems, specifically `BP_DroneMovement_6DOF.uasset` and `BP_DroneSensors.uasset`. The `DA_DroneMovementDefault.uasset` and `DA_SensorRigProfileDefault.uasset` assets remain the baseline defaults and are not mutated; they are only the starting values that runtime config can override for the current run.

---

# Scenario 3: Drone Controlling

## Example JSON command

```json
{
  "type": "CMD",
  "schema_version": "1.0",
  "run_id": "course_v1_20260311_120000",
  "seq": 25,
  "drone_id": "drone_001",
  "payload": {
    "pitch": -0.15,
    "roll": 0.05,
    "yaw": 0.20,
    "throttle": 0.65
  }
}
```

## Simple flow chart

```text
UE_Tooling/Drone_Controller/Drone_Controller.py
  -> UE_Tooling/WebSocket/protocol.py
  -> UE_Tooling/WebSocket/ws_bridge.py
  -> UE_Drone_Env/Plugins/DroneWebSocket/Public/WSClientComponent.h
  -> UE_Drone_Env/Content/io/Drone_Controller/BP_DroneController.uasset
  -> UE_Drone_Env/Content/Drone_Content/Interfaces/BPI_DroneCommandReceiver.uasset
  -> UE_Drone_Env/Content/Drone_Content/Blueprints/BP_DronePawn.uasset
  -> UE_Drone_Env/Content/Drone_Content/Blueprints/BP_DroneMovement_6DOF.uasset
```

## Explanation

`Drone_Controller.py` is the tooling-side live command sender. It creates a movement/control message for a specific `drone_id`, and that message goes through `protocol.py` and `ws_bridge.py` into Unreal.

`BP_DroneController.uasset` is the in-engine command router. It receives the command, finds the correct drone, and calls `BPI_DroneCommandReceiver.uasset`, which is implemented by `BP_DronePawn.uasset`. The pawn then forwards the command to `BP_DroneMovement_6DOF.uasset`, which is the actual movement module that updates translation and rotation in the world.

---

# Scenario 4: Data Inference / Capture

## Example JSON command

```json
{
  "type": "CAPTURE_NOW",
  "schema_version": "1.0",
  "run_id": "course_v1_20260311_120000",
  "seq": 40,
  "drone_id": "drone_001",
  "payload": {
    "capture_id": "cap_000040",
    "viewpoints": ["front"],
    "include_pose": true,
    "include_telemetry": true,
    "include_image": true
  }
}
```

## Simple flow chart

```text
UE_Tooling/Data_Interface/Sampler_Manager.py
  -> UE_Tooling/WebSocket/protocol.py
  -> UE_Tooling/WebSocket/ws_bridge.py
  -> UE_Drone_Env/Plugins/DroneWebSocket/Public/WSClientComponent.h
  -> UE_Drone_Env/Content/io/Data_Interface/BP_SampleManager.uasset
     -> UE_Drone_Env/Content/Drone_Content/Interfaces/BPI_DronePoseProvider.uasset
     -> UE_Drone_Env/Content/Drone_Content/Interfaces/BPI_DroneViewpointProvider.uasset
     -> UE_Drone_Env/Content/Drone_Content/Interfaces/BPI_DroneTelemetryProvider.uasset
     -> UE_Drone_Env/Content/Drone_Content/Blueprints/BP_DronePawn.uasset
        -> BP_DroneSensors.uasset
        -> BP_DroneTelemetrySampler.uasset
  -> UE_Drone_Env/Plugins/DroneWebSocket/Public/WSClientComponent.h
  -> UE_Tooling/WebSocket/ws_bridge.py
  -> UE_Tooling/Data_Interface/Sampler.py
```

## Explanation

`Sampler_Manager.py` is the tooling-side initiator for data capture and observation collection. It sends a `CAPTURE_NOW` request through `protocol.py` and `ws_bridge.py`, which reaches `BP_SampleManager.uasset` inside Unreal.

`BP_SampleManager.uasset` is the runtime data inference and capture orchestrator. It queries the live drone through `BPI_DronePoseProvider.uasset`, `BPI_DroneViewpointProvider.uasset`, and `BPI_DroneTelemetryProvider.uasset`. `BP_DroneSensors.uasset` provides the viewpoint transform and camera-related properties, while `BP_DroneTelemetrySampler.uasset` provides telemetry values, and the pawn itself provides the live object context. The manager combines those values with the rendered image into one atomic observation identified by `capture_id`, then sends it back through the WebSocket layer to `Sampler.py`, which decodes and stores it for the dataset.

---

# Final summary

## Four live runtime paths

```text
SPAWN:
Drone_Spawner.py
  -> protocol.py
  -> ws_bridge.py
  -> WSClientComponent
  -> BP_DroneSpawner
  -> BP_DronePawn

SET_CONFIG:
YAMLs
  -> RunConfig.py
  -> protocol.py
  -> ws_bridge.py
  -> WSClientComponent
  -> BP_SetDataConfig
  -> ST_RunConfig
  -> BP_DronePawn / BP_DroneMovement_6DOF / BP_DroneSensors

CMD:
Drone_Controller.py
  -> protocol.py
  -> ws_bridge.py
  -> WSClientComponent
  -> BP_DroneController
  -> BPI_DroneCommandReceiver
  -> BP_DronePawn
  -> BP_DroneMovement_6DOF

CAPTURE / OBS:
Sampler_Manager.py
  -> protocol.py
  -> ws_bridge.py
  -> WSClientComponent
  -> BP_SampleManager
  -> BPI_* providers on BP_DronePawn
  -> image + telemetry + pose
  -> WSClientComponent
  -> ws_bridge.py
  -> Sampler.py
```

The high-level pattern is simple: **tooling builds messages, WebSocket transports them, Unreal Blueprints execute them, and the drone pawn plus its modules provide the actual movement, viewpoints, telemetry, and images.**
