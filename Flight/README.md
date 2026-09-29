# Flight

`Flight/` is the live Python flight stack. It connects to a simulator or vehicle bridge over MAVLink, receives FPV image frames over UDP, runs the sibling `Vision` perception service, maintains a persistent gate map, estimates vehicle state, and sends MAVLink attitude-target commands from `src/main.py`.

## What Runs

```text
MAVLink telemetry
  -> sensing.telemetry.MavlinkClient
  -> sensing.odometry.VehicleStateEstimator
  -> mapping.gates.GateMap
  -> core.control.autipilot.AutiPilot
  -> MAVLink SET_ATTITUDE_TARGET

FPV image packets
  -> sensing.vision.io.VisionStreamReceiver
  -> Vision service.py / models
  -> mapping.gates.GateMap
```

Planning, MPCC, path-following, hover, and validation modules are still present for experiments, but `src/main.py` currently uses the gate-aware `AutiPilot` controller as the active command source.

## Setup

From the repository root:

```powershell
cd Flight
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs `../Vision` in editable mode, so changes in the sibling Vision project are picked up by Flight.

Optional OBS screen recording uses `.env` values:

```powershell
Copy-Item .env.sample .env
```

Edit `.env` only if `recording.record_screen` is enabled in `config/flight.yaml`.

## Configuration

Default runtime settings live in:

```text
Flight/config/flight.yaml
```

Important defaults:

```yaml
simulator:
  mavlink_endpoint: "udpin:127.0.0.1:14550"

vision:
  host: "0.0.0.0"
  port: 5600
  perception_backend: "deterministic_v3_2"

loops:
  inner_loop_hz: 120.0
  outer_loop_hz: 30.0

flight:
  allow_flight: true
```

To run with a different config:

```powershell
$env:FLIGHT_CONFIG_PATH = "C:\path\to\flight-local.yaml"
python src/main.py
```

## Run

Start the simulator or vehicle bridge first. Flight expects:

- MAVLink heartbeat and telemetry on the configured `simulator.mavlink_endpoint`.
- FPV image packets on the configured `vision.host` and `vision.port`.

Then run:

```powershell
cd Flight
.\.venv\Scripts\Activate.ps1
python src/main.py
```

`main.py` performs startup, reset, IMU calibration, gate-map initialization, the real-time control loop, logging, and shutdown cleanup.

## Main Runtime Sequence

1. Load `config/flight.yaml` through `core.initialization.initialize()`.
2. Connect to MAVLink and wait for heartbeat.
3. Start heartbeat, timesync, telemetry subscriptions, and vision UDP listening.
4. Reset the simulator and wait for fresh telemetry and vision.
5. Calibrate stationary IMU bias and initialize vehicle state.
6. Arm the vehicle and switch the system mode into racing.
7. Build an initial gate map from live vision observations.
8. Run the inner loop at `inner_loop_hz` and the outer vision loop at `outer_loop_hz`.
9. Send attitude-target commands when `flight.allow_flight` is true.
10. Save logs and optionally generate video on shutdown.

## Source Layout

```text
Flight/
  src/main.py                  Runtime entrypoint.
  config/flight.yaml           Default live-run configuration.
  requirements.txt             Python dependencies plus editable ../Vision.

  src/core/                    Shared schemas, initialization, control,
                               coordinates, modes, logging, and quadrotor helpers.
  src/sensing/                 MAVLink telemetry, odometry, VIO, and vision IO.
  src/mapping/                 Gate map and target selection.
  src/autonomy/                Planning and pathing experiments.
  validation/                  Local validation and visualization scripts.
```

## Logs

Runs are written under `logging.runs_root`, which defaults to:

```text
../Viewer/logs/flight/runs
```

Open them with the Viewer project:

```powershell
cd ..\Viewer
python -m pip install -r requirements.txt
python app.py
```

## Notes

- `Flight/src/sensing/vision/io` owns UDP frame ingestion and stays inside Flight.
- Perception models and assets live in `../Vision/src/models`.
- The active controller implementation is `src/core/control/autipilot/controller.py`.
- Stop a live run with `Ctrl+C`; `main.py` still executes its shutdown and log-save path.
