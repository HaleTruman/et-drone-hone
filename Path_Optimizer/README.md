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
    control/              # Command mapping and flight controllers
    modes/                # Stack orchestration, safety state, live smoke flight
    opt_engine/           # Active spline and speed-profile optimization library
  core/
    simulator/            # Unified deterministic telemetry and hover/track simulation
    quadrotor/            # Quadrotor dynamics and parameters
    app/                  # Dash run viewer
    logging/              # Structured run logging
    udp_relay/            # Standalone UDP relay
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
python -m core.simulator --duration-s 1
python -m autonomy.modes.live_forward_flight --idle-s 2 --racing-s 8 --forward-speed-mps 2
python -m pytest -q
```

## Active Components

- `autonomy/opt_engine/` loads Unreal target exports, samples Hermite splines, computes curvature, and solves feasible speed profiles.
- `autonomy/planning/mpcc/` contains the MPCC planner workbench that will feed the production stack after further development.
- `core/simulator/` owns both deterministic MAVLink-shaped telemetry generation and the richer hover/track simulation.
- `sensing/`, `autonomy/`, and `core/` separate the live stack by responsibility.

## Course Data

Official Unreal exports belong in `course_model/*.json`. Internal computations use meters; centimeter inputs are scaled by `0.01`.

## Known Follow-Up

Wire the domain packages into `src/main.py` when the production control loop is ready.
