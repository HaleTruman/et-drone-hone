# Path_Optimizer Code Deep Dive

This document explains the `Path_Optimizer` project as if you were rebuilding it by hand. It covers the code paths, data structures, algorithms, control flow, and the current seams between the lightweight path optimizer, the MPCC planner, and the quadrotor simulator.

## Mental Model

`Path_Optimizer` currently contains three related but distinct systems:

1. **Lightweight geometric path optimizer** in `src/opt_engine/`
   - Loads course targets or waypoint JSON.
   - Converts positions to meters.
   - Builds an exactly interpolating Hermite spline through the waypoints.
   - Samples curvature.
   - Computes a feasible speed profile under velocity, acceleration, lateral acceleration, optional yaw-rate, and optional hard curvature limits.
   - Can grid-search the Hermite smoothness parameter `lambda_`.

2. **Reference trajectory viewer** in `src/app/`
   - Dash app that shows course targets, the origin marker, and a saved reference trajectory from `artifacts/reference_trajectory.json`.
   - This app is wired by `src/app.py`.
   - It is not currently the full optimizer UI described in parts of `README.md`; it is a viewer for saved planner output.

3. **Quadrotor / MPCC / simulation stack** in `src/quadrotor/`, `src/controller/`, `src/drone_mpcc_planner/`, and `src/sim/`
   - Defines a 13-state nonlinear quadrotor model.
   - Defines a rate PID and higher-level hover/track controllers.
   - Builds a single-gate MPCC problem using CasADi.
   - Simulates the quadrotor with RK4 and visualizes the resulting flight.

There is also an older/prototype planning notebook path in `notebooks/`, including a complementary-progress-constraint optimizer prototype.

## Repository Layout

Important source paths:

- `src/opt_engine/`: geometric path optimizer.
- `src/app.py`: entrypoint for the Dash reference trajectory viewer.
- `src/app/`: viewer layout, callbacks, and data loading.
- `src/main.py`: script that generates `artifacts/reference_trajectory.json` from course data using `engine.PlanningEngine`.
- `src/engine/planner.py`: simple heuristic initial trajectory generator.
- `src/quadrotor/model.py`: NumPy 13-state quadrotor dynamics.
- `src/controller/rate/controller.py`: PID body-rate controller.
- `src/drone_mpcc_planner/`: CasADi MPCC planner and supporting path/dynamics/visualization helpers.
- `src/sim/`: hover/track simulator and Plotly visualization app.
- `tests/`: unit tests for the geometric optimizer.
- `course_model/`: Unreal-exported course target JSON.
- `artifacts/`: generated JSON outputs used by the viewer.
- `notebooks/`: experimental optimizer notebook and copied planner prototype.

Generated/cached items such as `.venv/`, `.pytest_cache/`, `__pycache__/`, and `.DS_Store` are not part of the hand-written logic.

## Data Formats

### Course Model JSON

The preferred course input is `course_model/targets-SimBlank-20260216_194648.json`. It contains a `targets` array. Each target has:

- `actor_label`: Unreal actor name.
- `actor_path`: Unreal object path.
- `position_cm`: target position in centimeters.
- `axis_x`, `axis_y`, `axis_z`: local target axes.

The loader treats labels ending in `origin` or containing `_origin` as metadata. The origin marker is stored separately and excluded from the waypoint path.

In the current checked-in course file, there is one origin marker and two course targets: `RedSphere_1m_Course_01` and `RedSphere_1m_Course_02`.

### Legacy Waypoint JSON

The older schema uses a top-level `waypoints` array:

```json
{
  "name": "simple_demo",
  "frame": "internal",
  "units": "m",
  "waypoints": [
    {"x": 0, "y": 0, "z": 0},
    {"x": 1, "y": 0, "z": 0}
  ]
}
```

The geometric optimizer supports both schemas.

### Reference Trajectory Artifact

`src/main.py` writes `artifacts/reference_trajectory.json`. The viewer expects:

- `course_path`: absolute or relative path to the course JSON that produced the trajectory.
- `drone_state`: state vector plus decomposed position, velocity, quaternion, and body rates.
- `reference_trajectory.t`: sample times.
- `reference_trajectory.pos`: sample positions in meters.

The viewer only overlays the trajectory if the artifact's `course_path` matches the selected course file.

## Core Types: `src/opt_engine/types.py`

This module defines the shared vocabulary for the geometric optimizer.

### `Waypoint` and `Vec3`

Both are frozen dataclasses with `x`, `y`, and `z` floats plus `as_np()`, which returns a NumPy vector. `Waypoint` is used for positions; `Vec3` is used for axes.

### `Target`

Represents one Unreal course target:

- `actor_label`
- `actor_path`
- `position`
- `axis_x`, `axis_y`, `axis_z`

### `Scenario`

The optimizer's canonical course container:

- `name`
- `frame`: `"internal"` or `"unreal"`
- `units`: `"m"` or `"cm"`
- `waypoints`: ordered path points.
- `targets`: optional rich target metadata.
- `origin_target`: optional origin marker.
- `level`, `mesh`, `generated_at`: optional Unreal metadata.

The key design choice is that optimization uses `waypoints`, while Unreal-specific details stay available as metadata.

### `Constraints`

Holds speed and feasibility limits:

- `v_max`: absolute max speed.
- `a_fwd_max`: max forward acceleration.
- `a_brake_max`: max braking deceleration.
- `a_lat_max`: max lateral acceleration.
- `r_min_m` or `kappa_max_1pm`: optional hard curvature limit.
- `use_theta_max`, `theta_max_deg`, `g_mps2`: optional way to derive lateral acceleration from a tilt limit.
- `heading_constrained`, `yaw_rate_max_rps`: optional yaw-rate speed cap.
- small epsilons for numerical stability.

`effective_a_lat_max()` returns either `a_lat_max` or `g * tan(theta_max)`.

`effective_kappa_max_1pm()` enforces that only one hard curvature limit is specified. If `r_min_m` is set, it converts radius to curvature by `kappa = 1 / radius`.

### `SamplingConfig`

Currently just `samples_per_segment`. Every waypoint-to-waypoint Hermite segment gets this many samples, excluding the endpoint. The final waypoint is appended once at the end.

### `SampledPath`

Stores:

- `points_m`: sampled XYZ points in meters, shape `(N, 3)`.
- `s_m`: cumulative arc length, shape `(N,)`.
- `ds_m`: segment lengths, shape `(N - 1,)`.
- `waypoint_indices`: indices in `points_m` where original waypoints occur exactly.

### `SpeedProfile`

Stores:

- `v_mps`: final feasible speed profile.
- `v_cap_mps`: active pointwise speed caps after all cap sources.
- `v_kappa_mps`: curvature-only speed cap.
- `v_yaw_mps`: yaw-rate cap, or `None`.
- `time_s`: estimated traversal time.

### `OptimizationResult`

The full output of one path evaluation or a lambda grid search:

- scenario name.
- selected lambda.
- total time.
- sampled path.
- curvature array.
- speed profile.
- diagnostics dictionary.

`to_json()` turns NumPy arrays into normal lists so the result can be written as JSON.

## Scenario Loading: `src/opt_engine/scenario_io.py`

This module does discovery, parsing, and validation.

### Discovery

`discover_scenario_files(root_dir)` checks:

1. `root_dir/course_model/*.json`
2. `root_dir/data/scenarios/*.json`

It prefers `course_model` if any JSON files exist there. This matches the intent that Unreal exports are the primary source.

### Loading

`load_scenario(path)` reads JSON, picks defaults, and branches:

- If the JSON has `targets`, it calls `_load_targets_scenario()`.
- Otherwise it calls `_load_waypoints_scenario()`.

For target files, `_load_targets_scenario()`:

1. Reads top-level metadata.
2. Iterates through every target.
3. Chooses the position key using `position_{units}`, falling back to `position_cm`, `position_m`, or `position`.
4. Parses axes as `Vec3`.
5. Builds a `Target`.
6. Sends origin-like labels to `origin_target` instead of the course target list.
7. Appends all non-origin target positions to `waypoints`.

Validation requires:

- at least two waypoints.
- units are `"m"` or `"cm"`.
- frame is `"internal"` or `"unreal"`.
- no consecutive duplicate waypoints.

## Unreal Coordinate Adapter: `src/opt_engine/coords_unreal.py`

`scenario_waypoints_to_internal_m()` converts `Scenario.waypoints` to a NumPy `(M, 3)` array in meters.

The only current coordinate conversion is unit scaling:

- `cm` becomes meters with `0.01`.
- `m` stays as-is.

For `frame="unreal"`, the function currently keeps axes unchanged. The module exists so future handedness or axis remapping can be isolated here.

## Spline Sampling: `src/opt_engine/spline.py`

The optimizer uses a cubic Hermite spline that exactly interpolates the input waypoints.

### Tangents

`_compute_base_tangents()` computes one tangent per waypoint:

- First point tangent: `p1 - p0`.
- Last point tangent: `p_last - p_previous`.
- Interior tangent: `0.5 * (p_next - p_prev)`.

Then every tangent is scaled by `lambda_`.

The intuition:

- `lambda_ = 0` collapses tangents to zero, producing a more stop-and-go curve through each waypoint.
- Larger `lambda_` creates smoother, more flowing transitions but can overshoot or increase curvature in some layouts.

### Segment Formula

`_hermite_segment()` evaluates:

```text
p(u) = h00(u) p0 + h10(u) m0 + h01(u) p1 + h11(u) m1
```

with the standard cubic Hermite basis:

```text
h00 =  2u^3 - 3u^2 + 1
h10 =   u^3 - 2u^2 + u
h01 = -2u^3 + 3u^2
h11 =   u^3 - u^2
```

`sample_hermite_spline()` samples each segment at evenly spaced `u` values from `0` up to but not including `1`. Because each segment excludes its endpoint, adjacent segments do not duplicate shared waypoints. The final waypoint is appended after all segments.

`waypoint_indices = i * samples_per_segment`, so each original waypoint can be recovered exactly from the sampled array.

## Geometry: `src/opt_engine/geometry.py`

### Arc Length

`arc_length(points)` computes:

- `ds`: Euclidean distance between adjacent points.
- `s`: cumulative distance starting at zero.

This arc-length coordinate is the independent variable for speed, curvature, yaw-rate diagnostics, and plotting.

### Polyline Curvature

`curvature_from_polyline(points)` estimates curvature using triples of adjacent points.

For each interior point `b`, with neighbors `a` and `c`, it computes:

```text
kappa = 2 * |(b - a) x (c - a)| / (|b - a| * |c - b| * |c - a|)
```

This is a circumcircle-style curvature estimate for a discrete polyline. Degenerate triples get zero curvature. The first and last curvatures are copied from their nearest interior neighbors.

### Heading and Yaw Derivative

`heading_and_dpsi_ds_xy(points, s_m)` computes heading only in the XY plane:

1. Estimate tangent vectors by forward/backward difference at endpoints and central difference inside.
2. Compute `psi = atan2(t_y, t_x)` wherever XY motion is nonzero.
3. Fill undefined headings with nearest valid values.
4. Unwrap headings so angle discontinuities do not create artificial jumps.
5. Differentiate heading with respect to arc length.

The result is useful when heading is constrained to follow travel direction. If `dpsi/ds` is large, the drone would need high yaw rate at high speed.

## Speed Profile: `src/opt_engine/speed_profile.py`

This module turns geometry into a feasible speed curve.

### Curvature Speed Cap

Lateral acceleration satisfies:

```text
a_lat = v^2 * |kappa|
```

Solving for speed:

```text
v_kappa = sqrt(a_lat_max / (|kappa| + epsilon))
```

Then:

```text
v_cap = min(v_max, v_kappa)
```

`speed_caps_from_curvature()` returns both `v_cap` and the raw curvature cap `v_kappa`.

### Yaw-Rate Cap

If heading must follow path tangent:

```text
yaw_rate = v * |dpsi/ds|
```

So:

```text
v_yaw = yaw_rate_max / (|dpsi/ds| + epsilon)
```

`apply_yaw_rate_speed_cap()` takes the current cap and returns `min(v_cap, v_yaw)`.

### Forward/Backward Acceleration Pass

`solve_speed_profile()` starts with `v = v_cap` and then enforces acceleration limits.

Forward pass:

```text
v[k+1] <= sqrt(v[k]^2 + 2 * a_fwd_max * ds[k])
```

Backward pass:

```text
v[k] <= sqrt(v[k+1]^2 + 2 * a_brake_max * ds[k])
```

This is a common time-optimal path-parameterization trick for a fixed path and pointwise speed caps.

Important detail: start and end speeds are **free**. The first point starts at its cap, and the final point remains whatever the forward pass allows before the backward pass. There is no forced stop at either end.

Traversal time is approximated by summing segment time:

```text
time = sum(ds / average(v[k], v[k+1]))
```

The average speed is clamped by `v_min` to avoid division by zero.

## Diagnostics: `src/opt_engine/diagnostics.py`

`summarize_bindings()` counts which constraints are active:

- `bind_vmax_count`: final speed equals cap and cap equals `v_max`.
- `bind_curvature_count`: final speed equals cap and cap equals curvature cap.
- `bind_yaw_count`: final speed equals cap and cap equals yaw cap.

It also reports max curvature, max speed, min speed, and sample count.

This tells you whether the path is limited by straight-line speed, tight curvature, yaw-rate, or acceleration propagation.

## Optimization Flow: `src/opt_engine/optimize.py`

### `evaluate_lambda()`

This is the core geometric optimizer pipeline:

1. Convert scenario waypoints to internal meters.
2. Sample Hermite spline at the requested `lambda_`.
3. Compute arc length and segment distances.
4. Estimate curvature.
5. Compute curvature and max-speed caps.
6. Optionally compute heading/yaw derivative and yaw-rate speed cap.
7. Run the forward/backward speed-profile pass.
8. Generate diagnostics.
9. Check optional hard curvature feasibility.
10. Return an `OptimizationResult`.

The hard curvature feasibility check uses `Constraints.effective_kappa_max_1pm()`. If any sampled curvature exceeds that max, diagnostics mark:

- `feasible = False`
- `kappa_violation_count`
- `kappa_violation_max_1pm`

The speed profile is still computed even for infeasible geometry; the diagnostics tell the caller whether the path violated a hard geometry limit.

### `optimize_lambda_grid()`

This evaluates `lambda_` values on a uniform grid.

If there is one step, it simply evaluates `lambda_min`.

If multiple candidates exist:

- Feasible candidates are compared by lowest total time.
- If all candidates are infeasible, it returns the least-bad candidate:
  - lowest max curvature first.
  - then lowest violation count.
- It adds `status = "infeasible_all_candidates"` when every candidate violates the hard curvature limit.

## Plotting: `src/opt_engine/plotting.py`

These helpers build Plotly figures for optimizer output.

### `make_3d_figure()`

Plots:

- red waypoint markers with index labels.
- the sampled path as a line.
- optional speed-colored path line.
- optional orange markers where curvature violates a hard limit.

The scene uses `aspectmode="data"` so geometry is not visually distorted.

### `make_profile_figure()`

Plots speed versus arc length:

- final `v(s)`.
- optional total cap.
- optional curvature cap.
- optional yaw-rate cap.

### `make_curvature_figure()`

Plots curvature versus arc length and optionally a horizontal `kappa_max` line.

## Reference Trajectory Viewer: `src/app.py` and `src/app/`

This is the currently wired Dash UI.

### Entrypoint: `src/app.py`

`main()` inserts `Path_Optimizer/src` into `sys.path`, imports `app.dash_app.run`, and starts the Dash server.

This path insertion is a lightweight alternative to packaging/installing the project.

### App Construction: `src/app/dash_app.py`

`create_app()`:

1. Computes the project root from the file location.
2. Calls `discover_course_files(root_dir)`.
3. Converts file paths into Dash dropdown options.
4. Creates `Dash(__name__)`.
5. Sets layout from `build_layout()`.
6. Registers callbacks.

`run()` starts the Dash dev server with `debug=True`.

### Layout: `src/app/layout.py`

The layout contains:

- `dcc.Interval(id="planning-session-poll")`: fires every second so the app notices updated artifacts.
- `dcc.Store(id="scene-signature")`: remembers a signature of selected course plus reference file mtimes.
- Course JSON dropdown.
- Summary `html.Pre`.
- 3D Plotly graph.

The page title is "Reference Trajectory Viewer". It is intentionally focused on visual inspection, not editing optimizer parameters.

### Data Loading: `src/app/data.py`

This module defines UI-specific dataclasses:

- `CoursePoint`: label plus XYZ in meters.
- `Course`: loaded course with path length and bounds properties.
- `PlannerTrajectory`: times plus position samples.
- `ReferenceTrajectorySnapshot`: loaded artifact snapshot.
- `PlanningSession`: unused by the current callbacks, but designed to load richer planning-session artifacts.

`discover_course_files()` mirrors the optimizer discovery logic, preferring `course_model` over `data/scenarios`.

`load_course()` supports `targets` and legacy `waypoints`. It scales centimeters to meters and separates origin markers.

`load_reference_trajectory()` parses `artifacts/reference_trajectory.json`.

`load_planning_session()` parses a richer `planning_session.json` format but is not currently used by `callbacks.py`.

### Callbacks: `src/app/callbacks.py`

`register_callbacks()` defines one callback:

Inputs:

- selected course path.
- interval tick count.
- existing scene signature.

Outputs:

- summary text.
- 3D figure.
- updated scene signature.

The callback:

1. Validates the selected course path.
2. Loads the course.
3. Attempts to load `artifacts/reference_trajectory.json`.
4. Ignores the reference trajectory if its `course_path` does not match the selected course.
5. Builds a signature from course path, course mtime, and reference mtime.
6. Returns `no_update` if nothing changed.
7. Otherwise builds the summary and 3D figure.

The 3D figure plots:

- gates as blue markers.
- origin as green diamond.
- reference trajectory as red dashed line.
- drone start position as a red marker.

## Headless Reference Generation: `src/main.py`

`src/main.py` is a script that generates `artifacts/reference_trajectory.json`.

It does the following:

1. Loads quadrotor parameters from `src/quadrotor/params.yaml`.
2. Loads simulation config from `src/config/settings.yaml`.
3. Loads `course_model/targets-SimBlank-20260216_194648.json`.
4. Builds a `Quadrotor`.
5. Builds an `engine.PlanningEngine`.
6. Creates a 13-state initial state from config.
7. Extracts gate positions from course targets, converting cm to m.
8. Generates a heuristic reference trajectory.
9. Writes the artifact used by the viewer.

Important implementation note: the path `src/config/settings.yaml` does not match the current repository layout shown by `rg --files`, where settings live under `src/sim/config/settings.yaml`. If you run this script from `Path_Optimizer`, that path may need correction.

## Heuristic Planning Engine: `src/engine/planner.py`

`PlanningEngine.generate_initial_guess()` creates a simple trajectory dictionary with:

- `t`: time samples.
- `x`: state samples, shape `(N, 13)`.
- `u`: motor commands, shape `(N - 1, 4)`.
- `lam`, `mu`, `nu`: arrays intended for complementary-progress formulations.

It is not a full optimizer. It creates a high-speed point-mass style initial guess:

1. Convert waypoints to an `(M, 3)` array.
2. Estimate a conservative acceleration:

   ```text
   a_max = max(3.5 * kf / m - g, 8.0)
   ```

3. Estimate total straight-line path distance.
4. Estimate `v_max` from distance and acceleration, capped at `25 m/s`.
5. Estimate total time.
6. Build evenly spaced time samples.
7. Linearly interpolate position along the waypoint polyline.
8. Set velocity along the current segment:
   - ramp up for the first 15 percent.
   - hold high speed in the middle.
   - ramp down for the last 15 percent.
9. Set quaternion to identity and body rates to zero.
10. Set all motors to hover command:

    ```text
    u_hover = sqrt(m * g / (4 * kf))
    ```

11. Put small `mu` activations near expected gate crossing times.

The returned shape resembles an optimization warm start, even though the trajectory itself is just heuristic interpolation.

## Quadrotor Model: `src/quadrotor/model.py`

The `Quadrotor` class implements a 13-state nonlinear model:

```text
x = [position(3), velocity_inertial(3), quaternion_wxyz(4), body_rates(3)]
u = [motor_1, motor_2, motor_3, motor_4]
```

### Parameters

Loaded from YAML:

- mass `m`.
- diagonal inertia `Ixx`, `Iyy`, `Izz`.
- thrust coefficient `kf`.
- reaction torque coefficient `km`.
- rotor inertia `Jr`.
- motor arm lengths `l`.
- vertical offsets `h`.
- translational drag `cd`.
- rotational drag `cr`.
- spin directions `d`.
- gravity `g`.

The constructor computes motor positions `rho` in an X-configuration using `sqrt(2)/2`.

### State Derivative

`state_derivative(x, u)` returns `x_dot`.

It computes:

1. body-frame forces.
2. body-frame moments.
3. body-to-inertial rotation matrix.
4. quaternion kinematic matrix `Xi`.
5. translational acceleration.
6. quaternion derivative.
7. angular acceleration.

### Force Model

`_get_motor_thrusts_omegas(u)`:

```text
thrust_i = kf * u_i^2
omega_i = d_i * sqrt(kf) * u_i
```

`_compute_forces()`:

- thrust is body negative Z: `[0, 0, -sum(thrusts)]`.
- velocity is transformed to body frame.
- drag is quadratic and opposite body velocity: `-cd * v_b * |v_b|`.

Then inertial acceleration is:

```text
R @ force_body / m + [0, 0, g]
```

This sign convention means gravity is positive Z in this model, while thrust points along body negative Z.

### Moment Model

`_compute_moments()` includes:

- thrust moments from `rho_i x force_i`.
- yaw reaction torque from rotor spin directions.
- gyroscopic moment from rotor inertia.
- quadratic rotational drag.
- rigid-body coupling `omega x I omega` in the final angular acceleration equation.

## Rate Controller: `src/controller/rate/controller.py`

`RateController` is a body-rate PID controller.

State input:

- current rates from `state[10:13]`.

Command input:

- desired body rates `[p_des, q_des, r_des]`.

The controller:

1. Computes rate error.
2. Integrates error with clamping.
3. Computes derivative error.
4. Forms PID command.
5. Clips PID output by `max_rate_accel`.
6. Mixes roll/pitch/yaw commands into four motor commands around hover.
7. Clips motors to `[motor_min, motor_max]`.

The mixer is:

```text
m1 = hover - roll + pitch - yaw
m2 = hover - roll - pitch + yaw
m3 = hover + roll - pitch - yaw
m4 = hover + roll + pitch + yaw
```

`load_params()` merges `params.yaml` and `gains.yaml`.

## Simulation Utilities: `src/sim/utils/utils.py`

This module contains small reusable helpers:

- `project_root()`: resolves `Path_Optimizer`.
- `src_root()`: resolves `Path_Optimizer/src`.
- `load_yaml()`: safe YAML loading.
- `initial_state()`: builds a 13-state vector from config.
- `normalize_quaternion()`: rejects zero norm and normalizes.
- `quaternion_to_euler()`: returns roll, pitch, yaw.
- `euler_to_quaternion()`: inverse conversion.
- `rk4_step()`: Runge-Kutta integration and quaternion renormalization.
- `body_axes_from_quaternion()`: uses the model rotation matrix to get body axes.

## Simulator: `src/sim/simulator/simulator.py`

This is the main dynamics simulation stack.

### `SimulationResult`

A dataclass containing the entire simulated history:

- time.
- state.
- motor commands.
- desired rates/euler.
- acceleration.
- thrust.
- speed.
- target position.
- optional reference path, planned path, planned states/controls.
- optional gates and wireframes.
- phase labels.
- planner info.

It also exposes convenient properties:

- `position`
- `velocity`
- `quaternion`
- `rates`
- `euler`

### `HoverController`

This is an outer-loop position/attitude controller backed by `RateController`.

The update flow:

1. Read current position, velocity, and Euler attitude.
2. Use configured hover target position and yaw.
3. Compute desired acceleration:

   ```text
   kp_pos * position_error + kd_pos * velocity_error + feedforward_accel
   ```

4. Convert desired acceleration plus yaw into desired Euler angles.
5. Convert attitude error into desired body rates.
6. Use the rate controller for attitude stabilization.
7. Add collective motor adjustment for vertical acceleration.

`_desired_euler()` constructs a desired rotation matrix by aligning body Z with the needed force axis and body X/Y with target yaw.

`_collective_for_vertical_accel()` estimates the total thrust needed after accounting for current tilt:

```text
total_thrust = m * (g - desired_accel_z) / tilt_factor
```

### `TrackController`

`TrackController` inherits from `HoverController` but follows an MPCC plan after an initial hover period.

It adds:

- planned state interpolation over time.
- target velocity interpolation.
- feedforward acceleration from the planned velocity gradient.
- yaw from target velocity.
- gate crossing detection.
- post-gate braking and hover capture.
- deadbanded tracking errors to avoid overcorrecting along/lateral/vertical axes.

The major phases are:

- `"hover"` before track start.
- `"track"` while following the MPCC plan.
- `"post_gate_brake"` immediately after crossing while still moving fast.
- `"post_gate_hover"` once slow enough to capture a hover position.

Gate crossing uses the gate plane normal and aperture bounds:

- progress is dot product against gate normal.
- local gate coordinates check whether the drone is inside the square opening.
- crossing occurs when progress moves from negative to nonnegative while inside aperture.

### `Simulator`

`Simulator.run()`:

1. Allocates arrays for the full simulation history.
2. Initializes state.
3. For each timestep:
   - asks controller for motor commands and telemetry.
   - evaluates derivative for acceleration logging.
   - stores telemetry.
   - advances state with RK4.
4. Computes one final telemetry sample at the last state.
5. Packages everything into `SimulationResult`.

For track simulations, duration is:

```text
hover_duration + planned_duration + settle_duration
```

For hover simulations, duration is the config `duration`.

### Factory Functions

- `make_hover_simulator()`: loads config, quad params, rate params/gains, builds hover controller.
- `run_hover_simulation()`: builds and runs hover sim.
- `demo_track_gates()`: hard-coded single gate at `[12, 5, 0]`.
- `build_demo_track_plan()`: runs `MPCCPlanner`, extracts warm-start trajectory, computes time vector and acceleration.
- `make_track_simulator()`: builds track controller around the demo MPCC plan.
- `run_track_simulation()`: builds and runs track sim.
- `run_configured_simulation()`: if scenario name is `single_gate_min_time`, runs track; otherwise hover.

## Simulation Visualization: `src/sim/visualization/visualization.py`

This module builds a Dash app around a completed simulation.

`create_app()`:

1. Runs the configured simulation immediately.
2. Builds a `Quadrotor`.
3. Computes tracking error and headline metrics.
4. Creates a Dash layout with:
   - metric cards.
   - 3D animated scene.
   - kinematics plot.
   - attitude plot.
   - controls plot.

`build_scene_figure()` creates a 3D Plotly animation:

- optional MPCC reference path.
- optional MPCC planned path.
- gate wireframes.
- actual flight path.
- active target path.
- animated drone body traces.

Drone rendering uses seven traces:

- body plane mesh.
- two body arms.
- nose axis.
- rotor markers.
- thrust axis.
- center marker.

The rest of the file builds time-series plots for position/velocity/error, attitude/rates, and motor/thrust histories.

## MPCC Reference Path: `src/drone_mpcc_planner/reference_path.py`

This file defines gate geometry and simple reference generation.

Constants:

- `GATE_OUTER_M = 2.7`
- `GATE_INNER_M = 1.5`
- `GATE_DEPTH_M = 0.26`

Helpers:

- `R(q)`: quaternion to rotation matrix.
- `square(p, rot, x, size)`: one square frame of a gate at a local x offset.
- `gate_wireframe(gate)`: outer/inner front/back frames plus connecting edges.
- `yaw_quat(deg)`: quaternion for yaw-only rotation.

`generate_reference(start_pos, gates, num_points=50, pass_through_m=4.0)`:

1. Reads gate positions and gate forward directions.
2. Adds a finish point beyond the last gate.
3. Creates a piecewise-linear path from start to gates to finish.
4. Normalizes cumulative distance into `theta` from 0 to 1.
5. Interpolates positions at evenly spaced `theta`.
6. Computes tangent directions.
7. Builds gate wireframes.
8. Returns a dictionary with sampled positions, tangents, length, gates, gate theta values, wireframes, and dimensions.

## MPCC Dynamics: `src/drone_mpcc_planner/quadrotor_dynamics.py`

This is a CasADi version of the quadrotor dynamics.

It hard-codes a smaller vehicle parameter set rather than loading `src/quadrotor/params.yaml`.

It defines:

- constants for mass, inertia, thrust/torque coefficients, drag, gravity, motor positions.
- `rotation_matrix(q)`
- `quaternion_to_euler(q)`
- `thrust_allocation(u)`
- `compute_total_thrust(u)`
- `f(x, u)`: CasADi symbolic 13-state derivative.

The structure matches `Quadrotor.state_derivative()`:

- position derivative is velocity.
- acceleration is rotated body force over mass plus gravity.
- quaternion derivative is `0.5 * Xi(q) * omega`.
- angular acceleration uses moments, drag, gyro effects, and inertia.

Because this file uses CasADi expressions, it can be embedded in an optimization problem.

## MPCC Problem: `src/drone_mpcc_planner/mpcc_problem.py`

`setup_mpcc_problem(dynamics_func, N=60, dt=0.08)` creates a CasADi `Opti` problem.

Decision variables:

- `X`: states, shape `(13, N + 1)`.
- `U`: controls, shape `(4, N)`.
- `h`: timestep, scalar.

Parameters:

- `x0`: initial state.
- `pref`: reference positions.
- `tref`: reference tangents.
- `gate_pos`: gate position.
- `gate_normal`: gate forward normal.

Constraints:

- initial state fixed.
- motor commands bounded between 0 and 1.
- optimized timestep `h` bounded between 0.02 and 0.18.
- RK4 dynamics with quaternion renormalization.
- motor rate limit.
- quaternion norm stays near 1.
- at a selected gate index, position must be within `GATE_R` of gate.
- at the gate, velocity must have positive projection through the gate normal.

Objective terms:

- strong contouring error penalty: distance perpendicular to reference tangent.
- smaller lag error penalty: distance along tangent.
- control effort penalty.
- body-rate penalty.
- motor smoothing and body-rate smoothing.
- strong gate position penalty.
- final point penalty.
- time/gate-step penalty through `240 * gate_k * h`.
- reward for forward gate speed.

The solver is IPOPT with quiet printing, max 250 iterations, and tolerance `1e-4`.

## MPCC Solver: `src/drone_mpcc_planner/mpcc_solver.py`

`solve_mpcc(opti_pack, x0, reference, warm_start=None)` fills parameters, initializes variables, solves, and returns arrays.

Steps:

1. Pick `N + 1` reference samples from the generated reference path.
2. Set initial state, reference positions, reference tangents, gate position, and gate normal.
3. Build an initial `X0`:
   - tile current state.
   - set positions to reference positions.
   - set velocities from finite difference of reference positions.
   - preserve initial quaternion.
4. Use previous warm start if available.
5. Solve with IPOPT.
6. If solve fails, use `opti.debug.value()` to recover the best available debug values.
7. Print solve time, iteration count, status, cost, optimized timestep, and horizon.
8. Return `X`, `U`, a placeholder `theta=1.0`, and info dictionary.

The failure path is useful for interactive tuning because it still gives a trajectory-like output for inspection.

## MPCC Planner: `src/drone_mpcc_planner/planner.py`

`MPCCPlanner` wraps problem construction and repeated solves.

Constructor:

- creates the MPCC problem once.
- initializes `warm_start` and `reference`.
- stores whether to visualize results.

`plan(current_state, gates)` currently requires exactly one gate.

Flow:

1. Generate reference from current position through the single gate.
2. Solve MPCC using previous warm start if available.
3. Compute gate miss distance.
4. Compute gate normal and progress along that normal.
5. Determine the first step where the plan crosses the gate plane.
6. Save `X` and `U` for next warm start.
7. Extract desired quaternion from the second state.
8. Compute first-step thrust command from total thrust.
9. Add rich diagnostics:
   - gate miss distance.
   - crossing step/time.
   - gate speed.
   - gate forward speed.
   - motor rate limit.
10. Optionally visualize with Matplotlib.
11. Return desired quaternion, thrust command, and info.

The simulator uses the planner's warm-start `X` and `U` as the planned trajectory.

## MPCC Visualizer and Demo

`src/drone_mpcc_planner/visualizer.py` creates a Matplotlib diagnostic view:

- 3D trajectory colored by collective thrust.
- gate wireframes.
- velocity arrows.
- body axes along the trajectory.
- attitude, speed, body rates, acceleration, and motor thrust plots.

`src/drone_mpcc_planner/demo.py` is a runnable single-gate example:

1. Start at origin with identity quaternion.
2. Define one gate at `[12, 0, 0]`.
3. Run `MPCCPlanner`.
4. Print desired quaternion, thrust command, gate dimensions, and miss distance.
5. Visualize the result.

## Prototype CPC Notebook Planner: `notebooks/planner_copy.py`

This file is an experimental planner copied into the notebook folder. It is more ambitious than `src/engine/planner.py`.

It includes:

- parameter precomputation.
- heuristic initial guess generation.
- NumPy quadrotor dynamics.
- RK4 integration.
- solver flatten/unflatten helpers.
- `optimize_cpc()`, a CasADi complementary-progress-constraint optimization.

The CPC idea uses variables:

- `Lam`: remaining progress markers per gate.
- `Mu`: progress activation.
- `Nu`: slack tied to gate distance.

The rough intent is:

- `Lam` starts at 1 and ends at 0.
- `Lam[k+1] = Lam[k] - Mu[k]`.
- `Mu` activates when the drone is close to a gate.
- `Nu <= d_tol^2` controls allowed gate distance.
- sequencing constraints try to force gates to be handled in order.

The objective minimizes final time `T` subject to full quadrotor dynamics and motor bounds.

This is prototype code: it has notebook-oriented prints, high IPOPT verbosity, and is not wired into the main app or simulator.

`notebooks/optimizer_testing.ipynb` imports this `PlanningEngine`, loads params/course data, generates an initial guess, runs `optimize_cpc()`, and plots trajectory comparisons.

## Tests

The tests focus on the geometric optimizer, not the MPCC or simulator.

### `tests/test_course_model_loader.py`

Covers:

- parsing target files.
- excluding origin target.
- preserving file order except origin removal.
- converting centimeters to meters.
- checking target axes are near orthonormal.
- loading legacy waypoint format.

Current repository note: these loader tests still expect an older fixture with 16 course targets beginning at `Course_13`, and they also expect `data/scenarios/simple_demo.json` to exist. The checked-in course file currently has only `Course_01` and `Course_02` plus an origin, and the legacy demo JSON is absent.

### `tests/test_scenario_discovery.py`

Covers scenario discovery priority:

- prefer `course_model`.
- fall back to `data/scenarios`.
- return empty list if neither has JSON files.

### `tests/test_spline.py`

Verifies Hermite sampling exactly interpolates every input waypoint at `waypoint_indices`.

### `tests/test_geometry.py`

Verifies:

- straight-line curvature is near zero.
- heading derivative around a unit circle is near one radian per meter.

### `tests/test_speed_profile.py`

Verifies the solved speed profile never exceeds pointwise caps.

### `tests/test_optional_constraints.py`

Verifies:

- tilt limit derives lateral acceleration as `g * tan(theta)`.
- impossible hard turn limits mark a result infeasible.
- yaw-rate cap becomes the active speed cap when heading derivative is high.

## End-to-End Flows

### Geometric Optimizer Flow

```text
JSON course
  -> load_scenario()
  -> scenario_waypoints_to_internal_m()
  -> sample_hermite_spline(lambda_)
  -> arc_length()
  -> curvature_from_polyline()
  -> speed_caps_from_curvature()
  -> optional heading_and_dpsi_ds_xy()
  -> optional apply_yaw_rate_speed_cap()
  -> solve_speed_profile()
  -> summarize_bindings()
  -> OptimizationResult
```

If using grid search:

```text
for lambda in linspace(lambda_min, lambda_max, lambda_steps):
    evaluate_lambda(lambda)
choose fastest feasible result
```

### Reference Viewer Flow

```text
python src/app.py
  -> create Dash app
  -> discover course files
  -> build dropdown/layout
  -> every second:
       load selected course
       load matching artifacts/reference_trajectory.json if present
       build summary
       build 3D figure
```

### Simulator / MPCC Flow

```text
run_configured_simulation()
  -> load sim config
  -> if scenario.name == single_gate_min_time:
       build MPCC demo track plan
       build TrackController
     else:
       build HoverController
  -> Simulator.run()
       controller.update()
       Quadrotor.state_derivative()
       rk4_step()
  -> SimulationResult
  -> visualization app plots result
```

### MPCC Planning Flow

```text
current state + one gate
  -> generate_reference()
  -> setup_mpcc_problem() already built in constructor
  -> solve_mpcc()
  -> warm_start saved
  -> planner info and planned trajectory available
```

## Important Current Mismatches and Gotchas

- `README.md` describes a richer optimizer UI with controls for path optimization. The current `src/app/` code is a reference trajectory viewer.
- `src/main.py` references `src/config/settings.yaml`, but the repository has `src/sim/config/settings.yaml`.
- `src/main.py` uses `engine.PlanningEngine`, not the `opt_engine` lambda optimizer or the MPCC planner.
- `src/drone_mpcc_planner/quadrotor_dynamics.py` hard-codes vehicle parameters that differ from `src/quadrotor/params.yaml`.
- The MPCC planner currently supports exactly one gate.
- The geometric optimizer is kinematic/geometric; it does not simulate full quadrotor attitude or motor dynamics.
- The simulator is dynamic but currently uses a demo gate plan, not the full Unreal course.
- The notebook CPC planner is experimental and not wired into production code.

## How To Read Or Modify This Code Safely

If you want to change the geometric optimizer:

1. Start in `types.py` to understand inputs/outputs.
2. Follow `optimize.evaluate_lambda()`.
3. Change one stage at a time: spline, curvature, speed caps, or diagnostics.
4. Run the existing tests.

If you want to change course loading:

1. Update `opt_engine/scenario_io.py`.
2. Update `app/data.py` if the viewer also needs the new fields.
3. Add tests under `tests/test_course_model_loader.py`.

If you want to change physical drone behavior:

1. Update `quadrotor/model.py` for NumPy simulation.
2. Update `drone_mpcc_planner/quadrotor_dynamics.py` too if the MPCC should match.
3. Keep sign conventions consistent: thrust is body negative Z and gravity is positive inertial Z.

If you want multi-gate MPCC:

1. Extend `generate_reference()` and gate parameters.
2. Replace the single `gate_pos` / `gate_normal` constraint with per-gate constraints.
3. Decide whether each gate gets a fixed step, a progress variable, or a complementarity formulation.
4. Update `MPCCPlanner.plan()` so it no longer rejects `len(gates) != 1`.

## Glossary

- **Arc length `s`**: distance along the sampled path.
- **Curvature `kappa`**: how sharply the path bends, in `1/m`.
- **Hermite spline**: cubic curve controlled by endpoint positions and endpoint tangents.
- **`lambda_`**: scalar multiplier on waypoint tangents; the path smoothness dial.
- **Speed cap**: pointwise maximum speed imposed by max speed, curvature, or yaw-rate.
- **Forward/backward pass**: algorithm that propagates acceleration and braking limits along the path.
- **MPCC**: model predictive contouring control; optimization that tracks a path while respecting dynamics.
- **Contouring error**: error perpendicular to the path tangent.
- **Lag error**: error along the path tangent.
- **CPC**: complementary progress constraint, an experimental way to encode gate visitation/order.
- **RK4**: fourth-order Runge-Kutta numerical integration.
