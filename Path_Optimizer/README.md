# Path_Optimizer (Waypoint/Target Path + Kinematic Constraints)

Local, self-contained optimization/visualization tool that:
- loads ordered 3D waypoints/targets (Unreal export supported),
- generates an **exactly interpolating** spline path family (smoothness dial \u03bb),
- computes curvature and a **time-optimal feasible speed profile** along the curve under simple kinematic limits,
- searches \u03bb (optional) to minimize total traversal time,
- visualizes waypoints, path, and diagnostics in a local Dash + Plotly UI.

## Folder structure

```text
Path_Optimizer/
  Development_Breif.md
  README.md
  devREADME.md
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
  course_model/
    targets-*.json
  data/
    scenarios/
      simple_demo.json
  artifacts/
  tests/
    test_*.py
```

## Script / module purpose

### Entrypoints

- `Path_Optimizer/run_app.py` — runs the local UI; adds `Path_Optimizer/src` to `sys.path` so the engine can be imported without packaging.
- `Path_Optimizer/run_optimize.py` — runs optimization headlessly from the CLI and writes a JSON artifact to `Path_Optimizer/artifacts/`.

### UI (`Path_Optimizer/app/`)

- `Path_Optimizer/app/dash_app.py` — creates the Dash app, discovers course JSON files, wires layout + callbacks, runs the dev server.
- `Path_Optimizer/app/layout.py` — UI layout only (controls + graphs).
- `Path_Optimizer/app/callbacks.py` — callback graph: reads UI inputs, runs optimization, and updates Plotly figures + summary text.

### Core engine (`Path_Optimizer/src/opt_engine/`)

- `Path_Optimizer/src/opt_engine/types.py` — dataclasses and numpy-backed containers (`Scenario`, `Constraints`, `SampledPath`, `SpeedProfile`, `OptimizationResult`).
- `Path_Optimizer/src/opt_engine/scenario_io.py` — scenario discovery/loading + JSON validation (supports legacy `waypoints[]` and official `targets[]`).
- `Path_Optimizer/src/opt_engine/coords_unreal.py` — coordinate/units adapter for Unreal import; currently applies unit scaling (`cm`→`m`).
- `Path_Optimizer/src/opt_engine/spline.py` — exactly-interpolating Hermite spline sampling with smoothness dial `lambda_`.
- `Path_Optimizer/src/opt_engine/geometry.py` — arc-length accumulation + curvature estimation from sampled points.
- `Path_Optimizer/src/opt_engine/speed_profile.py` — curvature speed cap + forward/backward pass for feasible time-optimal speed profile (**free start/end speeds**).
- `Path_Optimizer/src/opt_engine/optimize.py` — evaluates a single `lambda_` or runs a `lambda_` grid search and returns the best-time result.
- `Path_Optimizer/src/opt_engine/diagnostics.py` — simple binding diagnostics/summary stats for UI display.
- `Path_Optimizer/src/opt_engine/plotting.py` — Plotly figure builders (3D scene + speed/curvature plots).

### Data / outputs

- `Path_Optimizer/course_model/*.json` — official Unreal `targets[]` exports (preferred by the UI dropdown).
- `Path_Optimizer/data/scenarios/*.json` — legacy waypoint scenarios (fallback).
- `Path_Optimizer/artifacts/` — headless outputs (JSON/CSV/HTML as we add exporters).
- `Path_Optimizer/tests/` — unit tests for loader/spline/geometry/speed-profile/constraints.

## Quickstart

From the repo root:

```bash
python3 -m venv Path_Optimizer/.venv
source Path_Optimizer/.venv/bin/activate
python3 -m pip install -r Path_Optimizer/requirements.txt
python3 Path_Optimizer/run_app.py
```

Then open the printed local URL in your browser.

## Scenarios

The UI dropdown looks for course files in:
- `Path_Optimizer/course_model/*.json` (preferred)
- `Path_Optimizer/data/scenarios/*.json` (fallback)

Legacy waypoint schema:

```json
{
  "name": "simple_demo",
  "frame": "internal|unreal",
  "units": "m|cm",
  "waypoints": [{"x":0,"y":0,"z":0}, {"x":2,"y":1,"z":0.5}]
}
```

Official Unreal course-target schema (preferred):

```json
{
  "name": "targets-SimBlank-...",
  "level": "/Game/...",
  "mesh": "/Game/...",
  "frame": "unreal",
  "units": "cm",
  "generated_at": "2026-02-16T19:46:48",
  "targets": [
    {
      "actor_label": "RedSphere_1m_Course_01",
      "actor_path": "/Game/...",
      "position_cm": {"x": 0, "y": 0, "z": 0},
      "axis_x": {"x": 1, "y": 0, "z": 0},
      "axis_y": {"x": 0, "y": 1, "z": 0},
      "axis_z": {"x": 0, "y": 0, "z": 1}
    }
  ]
}
```

Notes:
- Internal computations use meters; `units="cm"` inputs are scaled by `0.01`.
- `frame="unreal"` is accepted for future Unreal integration; the MVP keeps axis mapping identity and focuses on consistent units.

## Headless run

```bash
Path_Optimizer/.venv/bin/python Path_Optimizer/run_optimize.py --scenario Path_Optimizer/data/scenarios/simple_demo.json
```

Outputs are written under `Path_Optimizer/artifacts/`.

## Tests

This project is intentionally lightweight (no packaging step yet). Run tests by adding `Path_Optimizer/src` to `PYTHONPATH`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Path_Optimizer/src \
  python3 -m pytest -q Path_Optimizer/tests
```
