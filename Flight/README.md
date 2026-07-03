# Flight

`Flight/` is the Python flight-stack workspace for the drone racing project. It contains the live MAVLink bridge, vision ingestion and perception pipeline, gate mapping, guidance/control code, run logging, and the Dash run explorer.

The current stack assumes the simulator lives outside this repository and is reached through MAVLink. The old in-repo simulator has been removed. Runtime data now enters `Flight` through two live interfaces:

- MAVLink UDP telemetry and commands, handled by `src/sensing/telemetry/mavlink_client.py`.
- UDP FPV image packets, handled by `src/sensing/vision/vision_stream.py`.

`src/main.py` is the current live runtime entry point. `src/app.py` starts the local run viewer.

## Quick Start

From the `Flight/` directory:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python src/app.py
```

To run the live stack against the external MAVLink simulator:

```powershell
$env:PYTHONPATH = "src"
python src/main.py
```

The runtime currently uses these defaults from `src/main.py`:

```text
MAVLINK_ENDPOINT = udpin:127.0.0.1:14550
VISION_HOST      = 0.0.0.0
VISION_PORT      = 5600
LOOP_HZ          = 30.0
```

The external simulator is expected to emit MAVLink on the configured endpoint and FPV image UDP packets on the configured vision port. The repository does not start or own the simulator process.

## Current Source Layout

```text
src/
  main.py                         # Live MAVLink + vision runtime loop
  app.py                          # Dash run viewer entry point

  core/
    schemas.py                    # Runtime data shapes and MAVLink cache dataclasses
    coordinates.py                # Quaternion/vector/frame helpers
    logging/                      # Structured run logger
    control/
      body_rate_guidance/         # Current live body-rate guidance controller
      forward_velocity/           # Forward velocity controller workbench
      hover/                      # Hover controller
      rate/                       # Rate controller
      command_mapper.py           # Control payload helpers
    modes/                        # Race/safety mode helpers
    quadrotor/                    # Quadrotor parameters and dynamics
    udp_relay/                    # Standalone UDP relay utilities

  sensing/
    telemetry/                    # MAVLink client, bridge alias, telemetry sync
    odometry/                     # VehicleState estimator/state holder
    vision/                       # UDP frame receiver, CNN/regressor/landmarker pipeline
    perception/                   # Vision observations, gate map, gate targeting, gate pose

  autonomy/
    planning/
      hot_start.py                # Gate-map-derived path helper used by live loop/logs
      mpcc/                       # MPCC planner workbench

  app/
    dash_app.py                   # Dash app factory and live-frame file route
    data.py                       # Run-log loading and normalization
    live_data.py                  # Live run discovery/frame helpers
    callbacks.py                  # Historical run dashboard callbacks
    live_callbacks.py             # Live run dashboard callbacks
    layout.py                     # Dash layout

  validation/                     # Validation utilities
```

## Runtime Data Flow

At a high level, `src/main.py` wires these pieces together:

```text
External simulator
  |-- MAVLink UDP: HEARTBEAT, TIMESYNC replies, HIGHRES_IMU, ACTUATOR_OUTPUT_STATUS, race packets
  |     -> MavlinkClient
  |     -> MavlinkTelemetry
  |     -> VehicleState IMU integration / guidance / perception / logs
  |
  |-- FPV image UDP packets
        -> VisionStreamReceiver
        -> VisionPerceptionService
        -> VisionObservation
        -> GatePoseEstimator
        -> GateMap
        -> GateTargetTracker
        -> BodyRateGuidanceController
        -> MAVLink attitude target command
```

The loop in `main.py` runs at `LOOP_HZ` and does the following:

1. Receives raw MAVLink telemetry from `MavlinkClient.get_telemetry()`.
2. Receives at most one queued FPV frame from `VisionStreamReceiver.get_next_frame()`.
3. Seeds or refreshes the authoritative gate map from simulator-provided track packets when available.
4. Updates the internal `VehicleState` estimate from `HIGHRES_IMU`.
5. Initializes attitude from accelerometer gravity direction on the first IMU sample.
6. Logs telemetry and cycle state.
7. If a vision frame is available, runs perception and maps detected gates into local NED.
8. Selects or holds a gate target.
9. Builds a body-rate guidance command.
10. Sends that command back to the simulator through MAVLink.
11. Writes run, telemetry, gate-map, frame, command, and planned-path logs.

## MAVLink Interface

`src/sensing/telemetry/mavlink_client.py` is the only runtime owner of MAVLink transport. It connects to the external simulator using `pymavlink` when the endpoint looks live, or acts as an offline cache in tests.

The current live endpoint is:

```text
udpin:127.0.0.1:14550
```

### MAVLink Messages Consumed

The client currently handles:

- `HEARTBEAT`
  - Updates armed state and system status.
  - Stored as `MavlinkHeartbeat`.
- `TIMESYNC`
  - Stored as `MavlinkTimesync`.
  - A periodic timesync loop sends requests while live.
- `HIGHRES_IMU`
  - Raw body-frame accelerometer/gyro telemetry.
  - Stored as `MavlinkHighresImu`.
  - Included in `MavlinkTelemetry.imu`.
- `ACTUATOR_OUTPUT_STATUS`
  - Stored as `MavlinkActuatorOutputStatus`.
- `COLLISION`
  - Stored as `CollisionEvent`.
- `DATA_TRANSMISSION_HANDSHAKE` and `ENCAPSULATED_DATA`
  - Used to receive simulator track-gate data and race status.
  - Track gates become `TrackGate`.
  - Race status becomes `RaceStatus`.

The client does not depend on MAVLink `ODOMETRY`, `ATTITUDE`, or `LOCAL_POSITION_NED`. `MavlinkClient.get_telemetry()` returns raw MAVLink telemetry; `main.py` passes that into `VehicleState.update(telemetry)` and receives the flight-facing telemetry bundle with `MavlinkTelemetry.odometry = vehicle_state.state`.

### MAVLink Commands Sent

The client can send:

- Arm/disarm commands.
- Simulator reset command using `MAVLINK_CMD_SIM_RESET = 31000`.
- `SET_POSITION_TARGET_LOCAL_NED` payloads via `send_position_target()`.
- `SET_ATTITUDE_TARGET` payloads via `send_attitude_target()`.
- Actuator/motor target commands via `send_motor_target()`.

The live loop currently uses body-rate guidance and sends attitude target commands through `send_attitude_target()`.

### Reset and Startup Sequence

When `RESET_ON_START = True`, `main.py` does this:

1. Connects to MAVLink and waits for a heartbeat.
2. Starts the heartbeat/timesync loop.
3. Starts telemetry subscription.
4. Sends the simulator reset command.
5. Waits for stable post-reset estimated state.
6. Waits briefly for track gates.
7. Seeds the gate map from simulator track data.
8. Arms the simulator if `ARM_ON_START = True`.
9. Runs a short prelevel phase with low thrust before entering the main loop.

The reset readiness test watches the IMU-integrated `MavlinkTelemetry.odometry.velocity_local_ned_mps` estimate and requires speed to remain below `RESET_STABLE_MAX_SPEED_MPS`.

## Runtime Schemas and Boundaries

Canonical dataclasses live in `src/core/schemas.py`.

### MAVLink Transport Shapes

These dataclasses are raw or near-raw MAVLink cache shapes:

- `MavlinkHeartbeat`
- `MavlinkTimesync`
- `MavlinkHighresImu`
- `MavlinkActuatorOutputStatus`
- `RaceStatus`
- `TrackGate`
- `CollisionEvent`

`MavlinkTelemetry` is used in two places:

- `MavlinkClient.get_telemetry()` returns raw simulator telemetry with `imu` populated and `odometry` unset.
- `VehicleState.update(telemetry)` returns flight-facing telemetry with `odometry` set to `vehicle_state.state`.

```text
MavlinkTelemetry
  sim_time_ns
  odometry: OdometryState | None
  imu: MavlinkHighresImu | None
  system_status
  reset_count
  raw
```

`MavlinkClient` does not fabricate normalized fields. Code that needs position, velocity, attitude, or body rates should first pass client telemetry through `VehicleState.update()`, then read the estimate from `telemetry.odometry`.

Examples:

```python
position = telemetry.odometry.position_local_ned_m
velocity = telemetry.odometry.velocity_local_ned_mps
attitude = telemetry.odometry.attitude_quaternion
body_rates = telemetry.odometry.body_rates_frd_rps
imu_accel = telemetry.imu.acceleration_body_frd_mps2 if telemetry.imu else None
```

### Estimator/Flight-State Shape

`OdometryState` is the normalized state emitted by flight code such as `VehicleState`:

```text
OdometryState
  sim_time_ns
  position_local_ned_m
  velocity_local_ned_mps
  attitude_quaternion
  body_rates_frd_rps
  acceleration_local_ned_mps2
```

The boundary is intentional:

- `MavlinkHighresImu` moves raw IMU telemetry from the simulator through the MAVLink client.
- `VehicleState.initialize_from_imu()` infers initial roll/pitch from accelerometer gravity direction.
- `VehicleState.update_from_imu()` integrates acceleration and gyro samples into `OdometryState`.

This keeps simulator transport data separate from estimator output while avoiding any dependency on a simulator pose message.

## Vehicle State

`src/sensing/odometry/state.py` owns `VehicleState`.

`VehicleState` is the mutable state holder used by the live loop. It stores:

- Local NED position and velocity.
- Attitude quaternion.
- Body angular velocity and angular acceleration.
- Body-frame acceleration from IMU.
- Local NED acceleration after attitude rotation and gravity handling.
- Last IMU timestamp.

Important methods:

- `reset(odometry: OdometryState | None = None) -> OdometryState`
  - Clears internal state.
  - Can seed from an already-normalized `OdometryState`.
- `update_odometry(odometry: OdometryState) -> OdometryState`
  - Updates state from normalized estimator data.
- `initialize_from_imu(latest_imu: MavlinkHighresImu) -> OdometryState`
  - Uses accelerometer gravity direction to initialize attitude.
  - Leaves yaw at zero because yaw is not observable from accelerometer alone.
- `update_from_imu(latest_imu: MavlinkHighresImu) -> OdometryState`
  - Updates acceleration and body rates.
  - Integrates acceleration and gyro deltas over time after the first IMU sample.

`main.py` owns the live `VehicleState` instance and updates it from raw client telemetry:

```python
telemetry = mavlink_client.get_telemetry()
telemetry = vehicle_state.update(telemetry)
estimated_state = vehicle_state.state
```

## Vision Input

`src/sensing/vision/vision_stream.py` receives FPV image data over UDP.

The simulator sends frames as chunked UDP packets. The receiver:

- Unpacks packet headers using `sensing.vision.io.udp_protocol`.
- Reassembles chunks by `frame_id`.
- Validates expected chunk count and JPEG size.
- Saves each completed JPEG when `output_dir` is configured.
- Appends frame metadata to `frames.jsonl`.
- Keeps a bounded in-memory queue for the live loop.

Runtime output location:

```text
logs/runs/run-<timestamp>/vision_frames/
logs/runs/run-<timestamp>/frames.jsonl
```

`VisionFrame` contains:

```text
frame_id
sim_time_ns
jpeg_bytes
image
saved_path
```

The current live loop reads one queued frame per cycle with `vision.get_next_frame()`.

## Vision Perception

`src/sensing/vision/service.py` wraps the in-memory perception pipeline:

```text
JPEG bytes
  -> jpeg_bytes_to_tensor()
  -> LightmaskInference
  -> RawLogitsFrame
  -> LogitRegressor
  -> Surveyer/regressor payload
  -> optional Landmarker
  -> VisionObservation
```

Configuration is held by `VisionPerceptionConfig`:

- `checkpoint`
  - CNN checkpoint, default `src/sensing/vision/cnn/cnn_last.pt`.
- `regressor_checkpoint`
  - Regressor checkpoint, default `src/sensing/vision/regressor/regressor_last.pt`.
- `device`
  - `"auto"` by default.
- `run_landmarker`
  - Whether to run the landmarker after regressor output.
- `top_k`
  - Gate targets retained for controller payloads.
- `passthrough_regressor_targets`
  - Preserves controller output shape while skipping landmark state/matching.
- `gate_threshold`, `confidence_threshold`, `min_component_area`, `max_candidates`
  - Regressor filtering parameters.

In `main.py`, live perception is currently initialized as:

```python
VisionPerceptionService(VisionPerceptionConfig(run_landmarker=False))
```

That means the live loop currently runs CNN + regressor and returns `VisionObservation` directly from regressor output, without landmarker state/matching.

## Gate Mapping and Target Selection

Gate-related code lives under `src/sensing/perception/`.

Key pieces:

- `VisionObservation`
  - Typed representation of gates detected in a vision frame.
- `GatePoseEstimator`
  - Converts camera-local gate observations into local NED gate records.
  - Uses vehicle position and attitude from `telemetry.odometry`.
  - Applies camera optical-to-body and body-to-NED transforms.
- `GateMap`
  - Maintains known gates.
  - Can be seeded from authoritative simulator track gates.
  - Can merge vision observations into existing gate records.
- `GateTargetTracker`
  - Holds the most recent usable target across short vision gaps.
- `select_guidance_gate`
  - Selects the nearest confident, not-yet-crossed gate.
  - Prefers `position_relative_ned_m` when available.

The simulator can send authoritative track gates through MAVLink encapsulated data. `GateMap.seed_from_track_gates()` converts those `TrackGate` records into `GateRecord` entries and reseeds the map when track-gate data changes.

When vision frames are processed, `GatePoseEstimator.update_gate_map_from_observation()` maps observed gates into local NED using:

```text
telemetry.odometry.position_local_ned_m
telemetry.odometry.attitude_quaternion
```

The hot-start planner then creates a path from the gate map for logging/inspection.

## Guidance and Control

The live runtime currently uses `BodyRateGuidanceController` from:

```text
src/core/control/body_rate_guidance/controller.py
```

The controller consumes:

- Current position from `telemetry.odometry.position_local_ned_m`.
- Current velocity from `telemetry.odometry.velocity_local_ned_mps`.
- Current attitude from `telemetry.odometry.attitude_quaternion`.
- A selected or held gate target from `GateTargetTracker`.

It outputs a command payload shaped for MAVLink attitude target commands:

```text
quaternion
thrust
attitude_type_mask
body_rates_rps
source
phase / guidance details
```

`main.py` builds the command with `BodyRateGuidanceController` and passes the finished payload to `MavlinkClient.send_attitude_target()`.

If no active target exists, the controller emits a hold/no-target command. During shutdown, `main.py` sends a body-rate stop command before closing the MAVLink connection.

Other controllers exist for workbench/testing:

- `core/control/hover/`
- `core/control/forward_velocity/`
- `core/control/rate/`

They are not the current live-loop control method.

## Planning

Planning code lives under `src/autonomy/planning/`.

Currently active in the live loop:

- `HotStartPlanner`
  - Uses the gate map to generate a simple path representation.
  - Logged as `planned_paths` and used for inspection.

Workbench/development:

- `autonomy/planning/mpcc/`
  - Contains MPCC-related planning code.
  - It is not the current command source in `main.py`.

## Logging

`src/core/logging/logging.py` writes structured run logs.

Each live run writes to:

```text
logs/runs/run-<YYYYMMDDTHHMMSSZ>/
  run.json
  telemetry.json
  gate_map.json
  frames.jsonl
  vision_frames/
    frame-<frame_id>-<sim_time_ns>.jpg
```

### `run.json`

Contains:

- `schema_version`
- `metadata`
- `events`
- `cycles`
- `vision_frames`
- `planned_paths`

Cycle records include timing, telemetry, bridge snapshot, command payload, target tracker snapshot, body-rate guidance snapshot, and vision/perception status.

### `telemetry.json`

Sidecar containing:

```text
schema_version
metadata
samples[]
```

Each telemetry sample contains the cycle/timing context and the serialized `MavlinkTelemetry`. IMU data is logged through:

```text
sample.telemetry.imu
```

Odometry data is logged through:

```text
sample.telemetry.odometry
```

### `gate_map.json`

Sidecar containing gate-map snapshots by cycle.

### `frames.jsonl` and `vision_frames/`

`VisionStreamReceiver` writes every completed JPEG frame to `vision_frames/` and records metadata in `frames.jsonl`. The Dash viewer serves those images through a Flask route instead of embedding all image bytes in the page.

## Dash Run Viewer

Start the viewer with:

```powershell
$env:PYTHONPATH = "src"
python src/app.py
```

The Dash app is built in `src/app/dash_app.py`.

The viewer discovers runs from:

```text
logs/run-*.json
logs/run-*/run.json
logs/runs/run-*/run.json
```

It also loads sidecars:

```text
telemetry.json
gate_map.json
frames.jsonl
```

The app normalizes telemetry for plotting in `src/app/data.py`. That normalization copies fields from `telemetry.odometry` into legacy top-level plotting keys such as `position_local_ned_m`, `velocity_local_ned_mps`, and `attitude_quaternion`. This is a viewer compatibility layer for old and new logs; it is not the runtime telemetry contract.

Frame images are served by:

```text
/live-frame?run=<run_path>&index=<frame_index>
```

The route verifies the requested run is discoverable before serving the JPEG file with Flask `send_file(..., conditional=True)`.

## Tests

Run all tests from `Flight/`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Current test coverage includes:

- MAVLink message parsing/cache behavior.
- Runtime orchestration behavior in `main.py`.
- `VehicleState` IMU integration and accelerometer-based attitude initialization.
- Body-rate guidance, hover, forward velocity, and mode helpers.
- Vision stream packet reassembly and frame persistence.
- Vision/perception/gate-map contract tests.
- Logger sidecar output.
- Dash run viewer and live frame serving behavior.

## Important Current Contracts

- The external simulator is authoritative for MAVLink telemetry.
- The active MAVLink UDP stream provides `HEARTBEAT`, `HIGHRES_IMU`, `ACTUATOR_OUTPUT_STATUS`, race-status `ENCAPSULATED_DATA`, and `TIMESYNC` replies.
- `HIGHRES_IMU` is the authoritative raw IMU telemetry message.
- `MavlinkClient.get_telemetry()` returns raw telemetry once `HIGHRES_IMU` is available; its `odometry` field is unset.
- `VehicleState.update(telemetry)` is the state-estimation boundary and returns telemetry with `odometry` set.
- The flight-facing telemetry used inside `main.py` carries the local `OdometryState` estimate from `VehicleState`, not a MAVLink odometry packet.
- Runtime code should read estimated position/velocity/attitude through `telemetry.odometry`.
- Runtime code should read raw IMU through `telemetry.imu` or `mavlink_client.latest_imu`.
- The Dash viewer may normalize logs for plotting, but that does not define the live runtime contract.

## Removed / Not Present

The following are intentionally no longer part of the active code path:

- The old in-repo/home-built simulator under `src/simulator`.
- Scenario JSON files for the removed local simulator.
- `TelemetrySample`.
- `ImuSample`.
- `AttitudeSample`.
- `diagnostic_odometry`.
- Raw `MavlinkAttitude` and `MavlinkLocalPositionNed` schema classes.
- `MavlinkTelemetry.attitude_sample`.
- `MavlinkTelemetry.local_position`.
- Compatibility construction of `MavlinkTelemetry` from normalized position/velocity/attitude args.
- Helpers that converted `OdometryState` back into `MavlinkTelemetry`.

## Current Limitations and Follow-Up Areas

- The live runtime constants in `main.py` are hardcoded module-level values, not loaded from a runtime config file.
- The vision perception service is wired with `run_landmarker=False` in the live entry point.
- MPCC code exists but is not the active live command source.
- The Dash viewer intentionally keeps compatibility normalization for older logs.
- The simulator protocol is external; any changes to MAVLink message availability or frame IDs should be reflected in `MavlinkClient` and tests.
