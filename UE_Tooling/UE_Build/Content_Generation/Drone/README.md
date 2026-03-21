# Drone Generation Conventions

This folder is the generation source-of-truth for runtime drone assets under:

- `UE_Drone_Env/Content/Drone_Content/...`

## Core rules

1. Every `gen_*` script rebuilds its target asset from scratch each build run.
2. Shared lifecycle logic lives in `assemble_drone_assets.py`.
3. Ordered orchestration and strict failure logging live in `run_drone_build.py`.
4. Canonical output paths remain stable (`/Game/Drone_Content/...`).

## Script roles

- `Structs/gen_st_*`: rebuild explicit shared struct assets; top-level canonical module contracts are `ST_DroneCommandNormalized`, `ST_DroneViewpointSnapshot`, and `ST_DroneTelemetrySnapshot`.
- `Interfaces/gen_bpi_*`: rebuild v1 BPI surfaces. Canonical top-level module contracts are `BPI_DroneCommandReceiver`, `BPI_DroneViewpointProvider`, and `BPI_DroneTelemetryProvider`. `BPI_DroneTelemetryProvider` depends only on `ST_DroneTelemetrySnapshot` as the canonical telemetry contract.
- `Data/gen_da_*`: rebuild default data assets with sentinel placeholders for runtime-override validation.
- `Materials/gen_*`: rebuild base material and default instance for drone body visuals.
- `Meshes/gen_*`: rebuild body/collision meshes from deterministic primitive templates.
- `Drone_Blueprints/gen_bp_*`: rebuild movement/sensor/telemetry/pawn blueprints and wire canonical references.
- `assemble_drone_assets.py`: shared helper API for creation, metadata stamping, and assertions.
- `run_drone_build.py`: headless runner with strict fail-fast policy and artifact output.

## Canonical output map

- Interfaces:
  - `/Game/Drone_Content/Interfaces/BPI_DroneCommandReceiver`
  - `/Game/Drone_Content/Interfaces/BPI_DroneViewpointProvider`
  - `/Game/Drone_Content/Interfaces/BPI_DroneTelemetryProvider`
- Data:
  - `/Game/Drone_Content/Data/DA_DroneMovementDefault`
  - `/Game/Drone_Content/Data/DA_SensorRigProfileDefault`
- Structs:
  - `/Game/Drone_Content/Structs/ST_DroneCommandNormalized`
  - `/Game/Drone_Content/Structs/ST_DroneViewpointSnapshot`
  - `/Game/Drone_Content/Structs/ST_DroneTelemetrySnapshot`
- Materials:
  - `/Game/Drone_Content/Materials/M_DroneBody_Base`
  - `/Game/Drone_Content/Materials/MI_DroneBody_Default`
- Meshes:
  - `/Game/Drone_Content/Meshes/SM_DroneBody`
  - `/Game/Drone_Content/Meshes/SM_DroneCollisionProxy`
- Blueprints:
  - `/Game/Drone_Content/Blueprints/BP_DroneMovement_6DOF`
  - `/Game/Drone_Content/Blueprints/BP_DroneSensors`
  - `/Game/Drone_Content/Blueprints/BP_DroneTelemetrySampler`
  - `/Game/Drone_Content/Blueprints/BP_DronePawn`

## v1 policy summary

- Movement: minimal deterministic command surface (`pitch`, `roll`, `yaw`, `throttle`), no physics.
- Viewpoints: `front` only; bootstrap values come from `DA_SensorRigProfileDefault` and are expected to be overwritten by `SET_CONFIG` before real runs.
- Telemetry: the canonical top-level telemetry contract is `ST_DroneTelemetrySnapshot`, which carries pose-aligned fields and per-mesh proximity records together.
- DA defaults: explicit sentinel placeholders; runtime config must overwrite during run startup.

## Run order

`run_drone_build.py` executes in this order:

1. Structs
2. Interfaces
3. Data defaults
4. Materials
5. Meshes
6. Module blueprints
7. Pawn blueprint

## Artifacts

Run artifacts are written to:

- `UE_Tooling/Artifacts/drone_content/<kit>_<version>_<timestamp>.json`
- `UE_Tooling/Artifacts/drone_content/logs/<run_name>/*.log`

Each artifact includes:
- per-script version metadata,
- return codes and timing,
- parsed `[DroneContentResult]` records,
- strict failure status,
- ordered generator execution details.

## Canonical symmetric module model

Top-level canonical runtime module contracts are:

- `ST_DroneCommandNormalized` <-> `BPI_DroneCommandReceiver` <-> `BP_DroneMovement_6DOF`
- `ST_DroneViewpointSnapshot` <-> `BPI_DroneViewpointProvider` <-> `BP_DroneSensors`
- `ST_DroneTelemetrySnapshot` <-> `BPI_DroneTelemetryProvider` <-> `BP_DroneTelemetrySampler`

`BP_DronePawn` remains the composition root only.

Within that pawn:

- hosted module component templates identify the movement, sensors, and telemetry modules
- stable role names plus deterministic component names identify those module responsibilities
- `primary_sensor_component_name` plus `primary_sensor_owner_role_name` provide the clean discovery path to the runtime sensor owner
- the pawn should not declare direct command, pose, viewpoint, or telemetry function surfaces of its own

Within that telemetry module:

- `BP_DroneTelemetrySampler` should expose `GetTelemetrySnapshot` as the canonical top-level telemetry function.
- Pose-aligned fields travel inside `ST_DroneTelemetrySnapshot` rather than through a separate canonical top-level function.

## Headless execution

Run full Drone generation:

```bash
python3 UE_Tooling/UE_Build/Content_Generation/Drone/run_drone_build.py
```

Optional overrides:

```bash
python3 UE_Tooling/UE_Build/Content_Generation/Drone/run_drone_build.py \
  --kit-name DroneContentKit \
  --version-id v001 \
  --timestamp 20260312_120000 \
  --level /Game/Course_Content/Maps/L_CourseTorus
```
