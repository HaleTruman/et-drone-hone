# Path Optimizer

Path planning, simulation, telemetry, and live-flight tooling for the drone racing stack.

## Source Layout

```text
src/
  sensing/
    vision/               # Vision stream ingestion
    perception/           # Gate observations and gate map
    estimation/           # Vehicle state estimation
    telemetry/            # MAVLink bridge and telemetry synchronization
  autonomy/
    planning/
      mpcc/               # MPCC workbench and planner implementation
  core/
    control/              # Command mapping and flight controllers
    modes/                # Stack orchestration, safety state, live smoke flight
    quadrotor/            # Quadrotor dynamics and parameters
    logging/              # Structured run logging
    udp_relay/            # Standalone UDP relay
  app/                    # Dash run viewer
  simulator/              # Unified deterministic telemetry and hover/track simulation
  app.py                  # Local Dash viewer entry point
  drone.py                # Minimal offline control-loop rig
  main.py                 # Reserved production stack entry point
```

`src/main.py` remains the intended single stack entry point. It is an explicit placeholder until the production runtime is wired.

## Useful Commands

From `Path_Optimizer`, add `src` to `PYTHONPATH` before running modules:

```powershell
$env:PYTHONPATH = "src"
python src/app.py
python src/sim.py --scenario scenarios/armed_hover.json --transport inprocess --accelerated
python -m simulator --scenario scenarios/idle_telemetry.json --transport inprocess --accelerated
python src/test.py --hover-s 8 --hover-altitude-m 1
python -m pytest -q
```

The default simulator scenario, transport, endpoint, pacing, and rates are set
in `src/simulator/config/settings.yaml`. CLI flags override those settings for
one run.

## Active Components

- `autonomy/planning/mpcc/` contains the MPCC planner workbench that will feed the production stack after further development.
- `simulator/` owns deterministic MAVLink-shaped flight simulation, scenarios, and logging.
- `sensing/`, `autonomy/`, and `core/` separate the live stack by responsibility.

## Course Data

Official Unreal exports belong in `course_model/*.json`. Internal computations use meters; centimeter inputs are scaled by `0.01`.

## Known Follow-Up

Wire the domain packages into `src/main.py` when the production control loop is ready.
