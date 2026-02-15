# Optimization Engine Development Brief (Waypoint + Kinematic Constraints)

## What we’re building

We’re building a small, self-contained “optimization engine” inside an existing repo that takes a set of 3D waypoints (dots the drone must pass through) plus a small set of kinematic/vehicle limits, then computes and visualizes the **fastest feasible** trajectory family. The key output is not only a geometric curve in space, but also the **speed profile** along that curve (because once constraints exist, the time-optimal solution depends on both geometry and timing). The tool must be interactive: changing constraints (sliders) should update the path and show how “optimal” shifts.

This should live in a subfolder so it can evolve independently of Unreal work. It should be runnable locally from VS Code with one command and render an interactive 3D visualization in a browser.

## Core concept in math terms

We treat the route as a curve \(r(s)\) in 3D parameterized by arc length \(s\), with a speed profile \(v(s)\). Waypoints are constraints like \(r(s_i) \approx P_i\) (or exactly equal for the initial version). From the curve we compute curvature \(\kappa(s)\) numerically. A minimal physical model gives a speed limit from turning capability:
\[
v(s) \le \min\left(v_{\max}, \sqrt{\frac{a_{\text{lat,max}}}{|\kappa(s)| + \epsilon}}\right)
\]
Then we apply forward acceleration and braking limits to produce a feasible \(v(s)\) (forward/backward pass). Total traversal time is:
\[
T = \int_0^L \frac{1}{v(s)}\,ds
\]
We then choose a curve family (with a “smooth ↔ angular” dial parameter) and search for parameters that minimize \(T\) under constraints.

## Inputs, outputs, and constraints

### Inputs
Waypoints are a list of 3D points in world coordinates. We assume a fixed visit order for MVP. Vehicle constraints below are user-adjustable (defaults provided).

| Symbol | Meaning | Notes for MVP |
|---|---|---|
| \(v_{\max}\) | maximum speed cap | global speed upper bound |
| \(a_{\text{fwd,max}}\) | maximum forward (longitudinal) acceleration | limits ramp-up of speed |
| \(a_{\text{brake,max}}\) | maximum braking (longitudinal deceleration) | limits slow-down approaching turns |
| \(a_{\text{lat,max}}\) | maximum lateral acceleration (turning capability) | ties speed to curvature |
| \(\kappa_{\max}\) / \(r_{\min}\) | maximum curvature / minimum turning radius | optional hard geometry constraint (may be derived) |
| \(\theta_{\max}\) | maximum tilt angle | usually couples to \(a_{\text{lat,max}}\) if gravity is modeled |
| \(\dot\psi_{\max}\) | maximum yaw rate | only matters if heading must follow motion direction |

### Outputs
The engine must produce (1) a sampled 3D curve, (2) a speed profile over samples, (3) traversal time \(T\), (4) diagnostic curves like curvature \(\kappa(s)\), implied lateral acceleration \(a_{\text{lat}}(s)\), implied yaw rate \(\dot\psi(s)\), and whether/where constraints are active.

## Important modeling choices (avoid conceptual traps)

Tilt angle, lateral acceleration, and gravity are coupled in real flight. If gravity is modeled, a common relationship is \(a_{\text{lat}} \approx g\tan(\theta)\), so \(\theta_{\max}\) implies a max lateral acceleration. For a “vacuum-like” kinematic MVP, we can treat \(a_{\text{lat,max}}\) as primary and use \(\theta_{\max}\) as an optional consistency check (or allow a user toggle: “derive \(a_{\text{lat,max}}\) from \(\theta_{\max}\) using \(g\)”).

Yaw rate matters only if we require the vehicle’s yaw (heading) to align with the direction of travel. The MVP should include a boolean: “heading-constrained.” If off, we ignore yaw rate. If on, we compute heading from the tangent direction in the XY plane: \(\psi(s)=\mathrm{atan2}(t_y,t_x)\) and approximate \(\dot\psi \approx \frac{d\psi}{dt} = \frac{d\psi}{ds}\,v(s)\).

## Approach for MVP

### Curve family (the “dial”)
Implement one simple curve family with a single smoothness dial \(\lambda\) so the user can slide between angular-ish and fluid. MVP recommendation: build a spline through waypoints and scale tangents by \(\lambda\).

A workable choice is a cubic Hermite spline (or Catmull–Rom converted to Hermite). Compute nominal tangents from waypoint neighbors, then scale:
\[
m_i(\lambda) = \lambda \cdot m_i^{\text{base}}
\]
Small \(\lambda\) produces tighter, more corner-like behavior; larger \(\lambda\) produces broader, smoother arcs. This gives the user a direct “fluid ↔ angular” dial without solving a heavy global optimizer.

### Feasible speed profile along a curve
Once a curve is sampled into points \(r_k\) with cumulative arc lengths \(s_k\), compute curvature \(\kappa_k\) numerically (finite differences on the tangent direction). Compute curvature-limited speed \(v^{\kappa}_k\). Then enforce longitudinal acceleration constraints using a standard forward/backward pass:

Forward pass (accel):
\[
v_{k+1} \le \sqrt{v_k^2 + 2 a_{\text{fwd,max}} \Delta s_k}
\]

Backward pass (braking):
\[
v_{k} \le \sqrt{v_{k+1}^2 + 2 a_{\text{brake,max}} \Delta s_k}
\]

Finally clamp by \(v_{\max}\) and curvature-based limits. Then compute total time by summing \(\Delta t_k = \Delta s_k / v_k\).

### “Optimization”
For MVP, do not introduce a full nonlinear optimizer. Use a small parameter search over \(\lambda\) (and optionally a turn-radius clamp) and select the best feasible solution. This is robust, explainable, and easy to visualize.

## Visualization requirement (open source)

Use a lightweight open-source stack that runs locally and renders interactive 3D. Recommended: Python + Plotly + Dash (MIT-licensed) so we can have sliders for constraints and immediate re-render of the curve, waypoints, and diagnostics. The UI should show the path in 3D with waypoints as markers, plus secondary plots for speed vs arc length and curvature vs arc length, and a single number for total time.

## Repo layout (subfolder)

Create a subfolder at repo root, for example:

`optimization_engine/`
- `README.md` (how to run, what it does)
- `pyproject.toml` (or `requirements.txt`) for dependencies
- `src/opt_engine/` (core computation)
- `app/` (Dash app entrypoint)
- `data/` (example waypoint scenarios in JSON)
- `tests/` (small unit tests for curvature + speed-profile pass)
- `artifacts/` (optional output exports: sampled curve JSON/CSV)

The module boundaries should be clean: path generation (spline + sampling), curvature + diagnostics, speed-profile solver, feasibility checks, and visualization app.

## MVP acceptance criteria

When the user provides 5–50 waypoints, the tool should render a 3D path through them and compute a time. Adjusting any of the core constraints should visibly change the chosen path parameters (at least \(\lambda\) and the speed profile) and update the time and diagnostic plots immediately. The tool should clearly show which constraint is “binding” (e.g., curvature-limited here, braking-limited there). The entire system should run with a single command from the repo (documented), and it should be easy to extend later into a more realistic dynamic model if needed.

## Future extensions (explicitly not required for MVP)

After MVP, we can add (1) soft waypoint regions instead of exact points, (2) obstacle avoidance, (3) full dynamic model with thrust/torque bounds, (4) global objective function minimization such as curvature-energy \(\int \kappa^2 ds\), and (5) export into Unreal as a SplineComponent-friendly format.

---

# Detailed implementation brief (MVP)

## Confirmed decisions (locked for MVP)

- **Waypoint interpolation is exact**: the generated curve must pass through every waypoint, in order.
- **Start/end speed are free**: there is no constraint like \(v(0)=0\) or \(v(L)=0\); the solver should maximize feasible speed subject to caps and accel/brake limits.
- **UI responsiveness is not a constraint**: correctness/clarity over speed.

## Scope and non-goals

### In scope (MVP)

- Ordered 3D waypoints \(\{P_i\}\) and a curve family \(r_\lambda(s)\) that interpolates them.
- Kinematic constraints: \(v_{\max}\), \(a_{\text{fwd,max}}\), \(a_{\text{brake,max}}\), \(a_{\text{lat,max}}\).
- Curvature estimation \(\kappa(s)\) from sampled curve points.
- Time-optimal feasible speed profile along a **fixed** curve using a forward/backward pass.
- A simple parameter search over \(\lambda\) to select the fastest curve variant.
- Local interactive UI (Dash + Plotly) to visualize waypoints, the chosen path, and diagnostics.
- A clear coordinate/units adapter so Unreal Engine coordinates can be imported later without rewrites.

### Explicitly out of scope (MVP)

- Obstacle avoidance, no-fly zones, collision checks.
- Full dynamics (thrust/torque, drag), closed-loop control, stochastic effects.
- Global nonlinear optimization over all spline control variables.

## Repository / package layout (inside `Optimization_Engine/`)

```
Optimization_Engine/
  README.md
  requirements.txt
  Development_Breif.md
  run_app.py
  run_optimize.py
  data/
    scenarios/
      *.json
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
  app/
    dash_app.py
    layout.py
    callbacks.py
  tests/
    test_spline.py
    test_geometry.py
    test_speed_profile.py
  artifacts/
```

## Data model

### Waypoints and scenarios

- A **scenario** is a named set of ordered 3D waypoints plus metadata.
- We will store scenarios as JSON under `data/scenarios/`.

#### Scenario JSON schema (MVP)

```json
{
  "name": "simple_demo",
  "frame": "internal|unreal",
  "units": "m|cm",
  "waypoints": [
    {"x": 0.0, "y": 0.0, "z": 0.0},
    {"x": 2.0, "y": 1.0, "z": 0.5}
  ]
}
```

Rules:
- `waypoints` length must be \(\ge 2\).
- Consecutive duplicates are invalid for MVP (they create zero-length segments).
- `frame="unreal"` indicates the coordinates are in Unreal’s world axes conventions; `coords_unreal.py` converts to internal.

### Constraints

Constraints are user-tunable, with sensible defaults:

- `v_max` (m/s)
- `a_fwd_max` (m/s^2) — limits how quickly speed can increase
- `a_brake_max` (m/s^2) — limits how quickly speed can decrease (use a positive value; it will be applied as a decel magnitude)
- `a_lat_max` (m/s^2) — curvature-based turning capability
- `kappa_epsilon` (1/m) — numerical stabilizer in \(v^\kappa\)

## Coordinate and units conventions

### Internal convention (recommended)

- Internal computation uses **meters**.
- Internal axes are simply \((x,y,z)\) with no enforced handedness; Plotly will render whatever we provide.

### Unreal import (adapter)

Unreal typically uses:
- Units: **centimeters**
- Axes: **X forward, Y right, Z up** (left-handed overall convention)

For MVP we implement conversion knobs:
- `units_scale`: default `0.01` to convert cm → m
- optional axis remap (kept identity for now unless needed later)

The adapter exists so that when Unreal exports waypoint positions, we can ingest them by setting `frame="unreal", units="cm"`.

## Curve generation (exact interpolation)

We implement a spline family that **always** satisfies \(r(t_i)=P_i\).

### Base spline

Use piecewise cubic Hermite segments between each \((P_i, P_{i+1})\) with endpoint tangents \((m_i, m_{i+1})\).

For \(u\in[0,1]\) (segment parameter), the Hermite curve is:

\[
r(u) = h_{00}(u)P_i + h_{10}(u)m_i + h_{01}(u)P_{i+1} + h_{11}(u)m_{i+1}
\]

with basis functions:
\[
h_{00}=2u^3-3u^2+1,\quad
h_{10}=u^3-2u^2+u,\quad
h_{01}=-2u^3+3u^2,\quad
h_{11}=u^3-u^2
\]

### Smoothness dial \(\lambda\)

Compute base tangents:
- interior: \(m_i^{base}=0.5(P_{i+1}-P_{i-1})\)
- endpoints: forward/backward difference

Then scale tangents:
\[
m_i(\lambda) = \lambda \cdot m_i^{base}
\]

Changing \(\lambda\) modifies curvature but preserves exact waypoint interpolation.

### Sampling

Sample each segment at `samples_per_segment` points (uniform in \(u\) for MVP). Record:
- sampled positions \(r_k\)
- cumulative arc length \(s_k\)
- segment lengths \(\Delta s_k\)

## Geometry / curvature estimation

Given sampled points \(r_k\) and arc lengths \(s_k\):

- Approximate tangent direction with central differences:
  - \(t_k \approx \mathrm{normalize}(r_{k+1}-r_{k-1})\)
- Approximate curvature magnitude:
  - \(\kappa_k \approx \left\lVert \frac{t_{k+1}-t_{k-1}}{s_{k+1}-s_{k-1}} \right\rVert\)

Endpoints use one-sided approximations.

## Speed profile solver (free boundary conditions)

### Step 1: speed caps

Curvature-based speed cap:
\[
v^\kappa_k = \sqrt{\frac{a_{\text{lat,max}}}{|\kappa_k|+\epsilon}}
\]

Global cap:
\[
v^{cap}_k = \min(v_{\max}, v^\kappa_k)
\]

### Step 2: forward / backward pass

Let \(\Delta s_k = s_{k+1}-s_k\).

Forward (acceleration-limited upper bound), with **free** start:
- initialize \(v_0 = v^{cap}_0\)
- propagate:
\[
v_{k+1} \leftarrow \min\left(v^{cap}_{k+1}, \sqrt{v_k^2 + 2 a_{\text{fwd,max}} \Delta s_k}\right)
\]

Backward (braking-limited), with **free** end:
- initialize at end \(v_{N-1}\) from the forward result
- propagate:
\[
v_{k} \leftarrow \min\left(v_{k}, \sqrt{v_{k+1}^2 + 2 a_{\text{brake,max}} \Delta s_k}\right)
\]

### Step 3: time and implied accelerations

- \(\Delta t_k = \Delta s_k / \max(v_k, v_{min})\) with a small `v_min` to avoid division by zero
- Total time \(T=\sum_k \Delta t_k\)
- Optional diagnostics:
  - implied longitudinal accel \(a_{long,k} \approx (v_{k+1}^2-v_k^2)/(2\Delta s_k)\)
  - implied lateral accel \(a_{lat,k} \approx v_k^2 \kappa_k\)

## Optimization loop (MVP)

Search a small set of \(\lambda\) values over `[lambda_min, lambda_max]` with `lambda_steps`. For each:
- generate curve samples
- compute curvature
- solve speed profile
- compute \(T\)

Pick the \(\lambda\) with minimal \(T\). Report best \(\lambda\) and diagnostics.

## UI requirements (Dash)

### Controls (MVP)

- Scenario dropdown
- `v_max`, `a_fwd_max`, `a_brake_max`, `a_lat_max` sliders
- “Manual \(\lambda\)” vs “Optimize \(\lambda\)” toggle
- \(\lambda\) slider (manual) OR `lambda_min/lambda_max/lambda_steps` (auto)
- `samples_per_segment` slider

### Visualizations

- 3D plot:
  - waypoints as markers (indexed labels)
  - chosen path as a line (optionally color by speed)
- 2D plots:
  - speed \(v(s)\) vs \(s\)
  - curvature \(\kappa(s)\) vs \(s\)
- Summary text:
  - best time \(T\), best \(\lambda\), max curvature, max speed, and where constraints bind (approx)

## CLI utilities

- `run_optimize.py`: run optimizer headless for a scenario and write `artifacts/*.json` (and optionally an HTML plot export).

## Testing plan (lightweight)

- `test_spline.py`: verify exact waypoint interpolation (curve samples include each waypoint at segment boundaries).
- `test_geometry.py`: sanity-check curvature on a straight line (\(\kappa \approx 0\)).
- `test_speed_profile.py`: verify forward/back pass respects accel/brake limits and caps.
