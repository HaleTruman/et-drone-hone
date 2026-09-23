# Flight

`Flight/` is the Python flight stack for an autonomous drone-racing project. It connects to an external simulator or vehicle bridge over MAVLink, receives FPV image frames, estimates vehicle state, maintains a persistent gate map, and sends quaternion attitude-target commands for gate-aware racing.

```text
MAVLink HIGHRES_IMU
  -> VehicleStateEstimator
  -> GateMap target / next-gate selection
  -> AutiPilot gate-aware acceleration controller
  -> MAVLink SET_ATTITUDE_TARGET

FPV image packets
  -> VisionStreamReceiver
  -> VisionPerceptionService (aigp_vision)
  -> GateMap
```

Path planning, MPCC, geometric path following, hover control, and other controller workbenches still exist in the repository, but they are not the active command source in `src/main.py`.

## What It Demonstrates

- Real-time inner/outer loop orchestration for a drone flight stack.
- MAVLink telemetry ingestion and attitude-target command output.
- IMU-based state estimation with optional VIO and Kalman correction hooks.
- Vision-driven gate detection and persistent gate-map target selection.
- A gate-aware acceleration controller that outputs quaternion error targets and normalized thrust.
- Structured run logging, frame capture, and a browser-based review UI.

## Quick Start

From the `Flight/` directory, install the dependencies and the sibling Vision
library with the same Python interpreter used to run Flight:

```powershell
python -m pip install -r requirements.txt ../Vision
```

Run the live stack:

```powershell
python src/main.py
```

Run the local log viewer:

```powershell
cd ..\UI
python app.py
```

The simulator is not owned by this repository. It is expected to provide MAVLink telemetry on the configured endpoint and FPV image packets on the configured vision port.

## Configuration

Runtime configuration lives in:

```text
config/flight.yaml
```

`src/core/initialization/initialization.py` loads that YAML file and wires the runtime objects from it. To run with a different config file:

```powershell
$env:FLIGHT_CONFIG_PATH = "C:\path\to\flight-local.yaml"
$env:PYTHONPATH = "src"
python src/main.py
```

Important defaults:

```yaml
simulator:
  mavlink_endpoint: "udpin:127.0.0.1:14550"
  sim_runtime: "VQ_2"

vision:
  host: "0.0.0.0"
  port: 5600

loops:
  inner_loop_hz: 120.0
  outer_loop_hz: 30.0
```

Run logs include the resolved initialization constants, including the config path used for the run.

## Source Layout

```text
src/
  main.py                         # Live runtime entry point

  core/
    initialization/               # YAML config loading and object wiring
    schema.py                     # Shared runtime dataclasses
    coordinates.py                # Quaternion, frame, and vector helpers
    control/
      autipilot/                  # Active gate-aware controller
      attitude/                   # Workbench attitude/body-rate controller
      hover/                      # Workbench hover controller
      path_follower/              # Workbench geometric path follower
      command_mapper.py           # Command payload helpers
    logging/                      # Structured run logging and MP4 generation
    modes/                        # System mode state machine
    quadrotor/                    # Vehicle model/parameter utilities

  sensing/
    telemetry/                    # MAVLink client and telemetry cache
    odometry/                     # State estimator, VIO, and filtering
    vision/
      io/                         # Frame receiver and UDP protocol
      service.py                  # Reexports the aigp_vision service and config

  mapping/
    gates/                        # Persistent gate records and target selection

  autonomy/
    pathing/                      # Path-manager workbench, inactive in main.py
    planning/                     # MPCC/simple planning experiments

../Vision/
  src/aigp_vision/                 # Gate perception models, assets, and service

../UI/
  app.py                          # Local flight log review UI entry point
  ui/
    server.py                     # Flask server and API routes
    data.py                       # Log loading and normalization
    static/                       # Browser UI assets
```

## Runtime Flow

`src/main.py` is an orchestrator. It handles startup, timing, sensor ingestion, command dispatch, logging, and shutdown.

Startup:

1. Connect to MAVLink and wait for heartbeat.
2. Start MAVLink heartbeat/timesync and telemetry subscriptions.
3. Start the vision receiver.
4. Reset the simulator and wait for fresh telemetry and vision.
5. Calibrate stationary IMU bias from `HIGHRES_IMU`.
6. Initialize vehicle state from stationary IMU samples.
7. Arm the vehicle and enter `RACING` mode.
8. Collect an initial gate map from live vision frames.

Main loop:

1. Read MAVLink telemetry and latest IMU.
2. Optionally feed IMU and frames into VIO.
3. Update `VehicleStateEstimator`.
4. Refresh gate crossed/target state in `GateMap`.
5. Process completed vision jobs and update gate records.
6. Compute an `AutiPilot` command from `target_gate` and `next_gate`.
7. Send the command through `MavlinkClient.send_attitude_target()`.
8. Log telemetry, command, gate-map, vision, VIO, and timing state.

Shutdown:

1. Stop recording if enabled.
2. Stop the vision executor and receiver.
3. Preserve final shutdown telemetry.
4. Close MAVLink.
5. Save run logs and optionally generate an MP4.

## Active Controller

The active controller is `core/control/autipilot/controller.py`.

`AutiPilot.compute_control()` consumes:

- `VehicleState`
- current `GateRecord`
- optional next `GateRecord`
- time since takeoff

It outputs a MAVLink-ready payload containing:

- `error_quaternion`
- `error_quaternion_target_converted`
- `error_quaternion_target_scaled`
- `error_quaternion_vector_scales`
- `thrust`
- diagnostic controller details

The MAVLink client sends the scaled error quaternion as `SET_ATTITUDE_TARGET` with body-rate fields ignored.

## Gate Mapping

`GateMap` owns persistent gate records. New observations are merged into candidates or promoted gates. On every update, it:

- marks gates crossed when the vehicle is within `gate_passed_distance_m`
- selects `target_gate` as the closest uncrossed gate
- selects `next_gate` as the second-closest uncrossed gate
- exposes candidate target fields for logging and diagnostics

This target selection does not depend on the path manager.

## Coordinate Conventions

- Inertial frame: local NED, with positive `z` down.
- Body frame: FRD, with positive `z` down.
- Quaternions are scalar-first `[w, x, y, z]`.
- Controller attitude commands use the simulator-observed quaternion error convention prepared by `AutiPilot`.

## Logging and Viewer

Runs are written under the configured `logging.runs_root`, which defaults to:

```text
../Viewer/logs/flight/runs
```

Each run includes structured JSON logs and captured vision frames:

```text
run-<timestamp>/
  run.json
  telemetry.json
  gate_map.json
  frames.jsonl
  vision_frames/
```

The review UI in `../UI/app.py` loads these runs for inspection and plotting.

## Current Boundaries

Active in `main.py`:

- MAVLink telemetry and command transport
- vision frame ingestion and perception
- gate mapping
- vehicle state estimation
- optional VIO/Kalman hooks
- `AutiPilot` command generation
- logging, OBS recording, and video generation

Present but inactive in `main.py`:

- `autonomy/pathing/PathManager`
- `core/control/path_follower/GeometricPathFollower`
- `core/control/hover/HoverController`
- MPCC planning workbenches

## Resume Notes

For review, the most important files are:

- `src/main.py`
- `src/core/initialization/initialization.py`
- `config/flight.yaml`
- `src/core/control/autipilot/controller.py`
- `src/mapping/gates/gate_map.py`
- `src/sensing/telemetry/mavlink_client.py`
- `src/sensing/odometry/state.py`

Together they show the real-time loop, configuration boundary, control law, perception-to-map bridge, MAVLink integration, and estimator design.
