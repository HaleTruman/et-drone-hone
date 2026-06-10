# Flight

Path planning, simulation, telemetry, and live-flight tooling for the drone racing stack.

## Source Layout

```text
src/
  sensing/
    vision/               # Vision UDP ingestion, CNN inference, regressor, and landmarker runtime
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
  main.py                 # Reserved production stack entry point
```

`src/main.py` remains the intended single stack entry point. It is an explicit placeholder until the production runtime is wired.

## Useful Commands

From `Flight`, add `src` to `PYTHONPATH` before running modules:

```powershell
$env:PYTHONPATH = "src"
python src/app.py
python -m pytest -q
```

The default simulator scenario, transport, endpoint, pacing, and rates are set
in `src/simulator/config/settings.yaml`. CLI flags override those settings for
one run.

## Active Components

- `autonomy/planning/mpcc/` contains the MPCC planner workbench that will feed the production stack after further development.
- `simulator/` owns deterministic MAVLink-shaped flight simulation, scenarios, and logging.
- `sensing/vision/` is the canonical flight-stack home for the production vision runtime. The root `vision/` package is kept only for compatibility with existing review/replay tooling during the merge.
- `sensing/`, `autonomy/`, and `core/` separate the live stack by responsibility.

## Known Follow-Up

Wire the domain packages into `src/main.py` when the production control loop is ready.
