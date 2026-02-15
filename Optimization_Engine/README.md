# Optimization Engine (Waypoint + Kinematic Constraints)

Local, self-contained optimization/visualization tool that:
- loads ordered 3D waypoints,
- generates an **exactly interpolating** spline path family (smoothness dial \u03bb),
- computes curvature and a **time-optimal feasible speed profile** along the curve under simple kinematic limits,
- searches \u03bb (optional) to minimize total traversal time,
- visualizes waypoints, path, and diagnostics in a local Dash + Plotly UI.

## Folder structure

```text
Optimization_Engine/
  Development_Breif.md
  README.md
  requirements.txt
  run_app.py
  run_optimize.py
  app/
    dash_app.py
    layout.py
    callbacks.py
  src/
    opt_engine/
      __init__.py
      types.py
      scenario_io.py
      coords_unreal.py
      spline.py
      geometry.py
      speed_profile.py
      optimize.py
      diagnostics.py
      plotting.py
  data/
    scenarios/
      simple_demo.json
  artifacts/
  tests/
    test_geometry.py
    test_speed_profile.py
    test_spline.py
```

## Script / module purpose

### Entrypoints

- `Optimization_Engine/run_app.py` — runs the local UI; adds `Optimization_Engine/src` to `sys.path` so the engine can be imported without packaging.
- `Optimization_Engine/run_optimize.py` — runs optimization headlessly from the CLI and writes a JSON artifact to `Optimization_Engine/artifacts/`.

### UI (`Optimization_Engine/app/`)

- `Optimization_Engine/app/dash_app.py` — creates the Dash app, discovers scenarios, wires layout + callbacks, runs the dev server.
- `Optimization_Engine/app/layout.py` — UI layout only (controls + graphs).
- `Optimization_Engine/app/callbacks.py` — callback graph: reads UI inputs, runs optimization, and updates Plotly figures + summary text.

### Core engine (`Optimization_Engine/src/opt_engine/`)

- `Optimization_Engine/src/opt_engine/types.py` — dataclasses and numpy-backed containers (`Scenario`, `Constraints`, `SampledPath`, `SpeedProfile`, `OptimizationResult`).
- `Optimization_Engine/src/opt_engine/scenario_io.py` — scenario discovery/loading + JSON validation.
- `Optimization_Engine/src/opt_engine/coords_unreal.py` — coordinate/units adapter for Unreal import; MVP applies unit scaling (`cm`→`m`) and reserves axis remaps for later.
- `Optimization_Engine/src/opt_engine/spline.py` — exactly-interpolating Hermite spline sampling with smoothness dial `lambda_`.
- `Optimization_Engine/src/opt_engine/geometry.py` — arc-length accumulation + curvature estimation from sampled points.
- `Optimization_Engine/src/opt_engine/speed_profile.py` — curvature speed cap + forward/backward pass for feasible time-optimal speed profile (**free start/end speeds**).
- `Optimization_Engine/src/opt_engine/optimize.py` — evaluates a single `lambda_` or runs a `lambda_` grid search and returns the best-time result.
- `Optimization_Engine/src/opt_engine/diagnostics.py` — simple binding diagnostics/summary stats for UI display.
- `Optimization_Engine/src/opt_engine/plotting.py` — Plotly figure builders (3D scene + speed/curvature plots).

### Data / outputs

- `Optimization_Engine/data/scenarios/*.json` — waypoint scenarios (inputs).
- `Optimization_Engine/artifacts/` — headless outputs (JSON/CSV/HTML as we add exporters).
- `Optimization_Engine/tests/` — small unit tests for spline/geometry/speed-profile logic.

## Quickstart

From the repo root:

```bash
python3 -m venv Optimization_Engine/.venv
python3 -m pip install -r Optimization_Engine/requirements.txt
python3 Optimization_Engine/run_app.py
```

Then open the printed local URL in your browser.

## Scenarios

Scenarios live in `Optimization_Engine/data/scenarios/*.json`.

Schema (MVP):

```json
{
  "name": "simple_demo",
  "frame": "internal|unreal",
  "units": "m|cm",
  "waypoints": [{"x":0,"y":0,"z":0}, {"x":2,"y":1,"z":0.5}]
}
```

Notes:
- Internal computations use meters; `units="cm"` inputs are scaled by `0.01`.
- `frame="unreal"` is accepted for future Unreal integration; the MVP keeps axis mapping identity and focuses on consistent units.

## Headless run

```bash
python3 Optimization_Engine/run_optimize.py --scenario Optimization_Engine/data/scenarios/simple_demo.json
```

Outputs are written under `Optimization_Engine/artifacts/`.

## Tests

This project is intentionally lightweight (no packaging step yet). Run tests by adding `Optimization_Engine/src` to `PYTHONPATH`:

```bash
PYTHONPATH=Optimization_Engine/src python3 -m pytest Optimization_Engine/tests
```
