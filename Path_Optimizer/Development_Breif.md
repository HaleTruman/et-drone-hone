# Optimization Engine — Remaining Development Brief

This document tracks **only the remaining work** for the Optimization Engine MVP, now that the initial scaffold (curve + speed-profile + λ search + Dash UI) exists.

## Locked decisions (do not change)

- Waypoints are **exactly interpolated** (the path must pass through every waypoint, in order).
- Start/end speed are **free** (no enforced \(v(0)=0\) or \(v(L)=0\)).
- Input waypoints are **Unreal coordinates in cm** by default; internal computation uses **meters** after conversion.
- Full **XYZ** is used (not a 2D-only model).

## Remaining MVP items (priority order)

### 1) Unreal coordinate parity (beyond cm→m)

Current state: `coords_unreal.py` only handles unit scaling (cm→m); axes/handedness are treated as identity.

Remaining:
- Confirm the exact Unreal→Plotly/internal axis mapping desired (if any), and implement it in one place.
- Add a small test that validates the mapping and scaling.
- Document the chosen convention in `README.md` and scenario schema comments.

### 2) “Constraint binding” detection + visualization

Goal: the UI should clearly show **which constraint is binding and where** along the path.

Remaining:
- Add per-sample flags for:
  - `binding_vmax`
  - `binding_curvature` (lateral accel cap)
  - `binding_accel` (forward pass active)
  - `binding_brake` (backward pass active)
- Add a visualization:
  - either a discrete “active constraint vs s” plot, or
  - color-coded path segments / markers.
- Update summary text to report **where** (approx indices / s positions) the constraints bind most strongly.

### 3) Diagnostics plots (human-friendly)

Remaining:
- Add plot(s) for:
  - implied lateral acceleration: \(a_{lat}(s) = v(s)^2 \kappa(s)\)
  - implied longitudinal acceleration (from \(v^2\) differences over \(\Delta s\))
- (Optional but recommended) plot “speed caps”:
  - `v_kappa(s)` and `v_cap(s)` overlaid with `v(s)` for explainability.

### 4) Optional constraints (described in brief, not implemented)

These should be implemented as toggles (defaults off) so the base MVP remains simple.

Spec (Name: Description : math term : function in calculation):
- `kappa_max / r_min`: Hard geometric turning limit (path infeasible if too “tight”) : `κ(s) ≤ κ_max` (equiv. `r(s)=1/κ(s) ≥ r_min`) : feasibility check (reject/penalize λ) + highlight violations along the sampled path.
- `theta_max → a_lat_max` coupling: Use max tilt angle to derive lateral acceleration capability : `a_lat_max = g * tan(θ_max)` : sets/overrides `a_lat_max`, which then drives curvature speed cap `v_kappa(s) = √(a_lat_max/(|κ(s)|+ε))`.
- `heading_constrained / yaw_rate_max`: Enforce yaw follows direction of travel and yaw-rate is limited : `ψ(s)=atan2(t_y,t_x)`, `ψ̇(s)≈(dψ/ds)*v(s)`, require `|ψ̇|≤ψ̇_max` ⇒ `v(s)≤ψ̇_max/|dψ/ds|` : adds an additional pointwise speed cap (min with existing caps) and/or a feasibility flag where violated.

Implementation requirements:
- Provide UI toggles and show derived quantities (e.g., derived `a_lat_max` when `theta_max` coupling is enabled).
- Use explicit units: `κ` in `1/m`, `r_min` in `m`, `θ_max` in `deg` (convert to rad), `ψ̇_max` in `rad/s`.
- Default all optional constraints to **off**, preserving baseline behavior.

## End-to-end integration plan (careful + accurate)

This section describes how to introduce the optional constraints into the codebase without breaking existing behavior, and how to expose them in the UI with clear diagnostics.

### Step A — Data model and toggles

1) Extend `Constraints` (and/or add a new `OptionalConstraints` struct) with:
- `kappa_max_1pm: float | None` and/or `r_min_m: float | None`
- `use_theta_max: bool`, `theta_max_deg: float`, `g_mps2: float = 9.81`
- `heading_constrained: bool`, `yaw_rate_max_rps: float`
- (Optional) `curvature_mode: "3d" | "xy"` (default to `"3d"` unless we later decide yaw should drive `"xy"` curvature)

2) Add validation helpers:
- enforce positive numeric ranges, sensible theta range (e.g. `0 < θ_max < 89°`)
- forbid simultaneous contradictory settings (e.g. both `kappa_max_1pm` and `r_min_m` if we choose one canonical input)

### Step B — Geometry signals needed for the new constraints

Add (and unit-test) geometry helpers to compute:

1) **3D curvature** `κ(s)` (already present) and optionally **XY-only curvature** if `curvature_mode="xy"` is introduced.

2) **Heading** and **heading derivative** along arc length:
- compute tangent in XY plane: `t_xy = (t_x, t_y)`
- compute heading `ψ(s)=atan2(t_y, t_x)` where `||t_xy|| > eps`
- unwrap `ψ` (to avoid `π ↔ -π` discontinuity spikes)
- compute `dψ/ds` via central differences on unwrapped `ψ` vs `s`

Numerical edge cases to handle explicitly:
- near-vertical segments where `||t_xy||` is ~0 (heading undefined): carry forward last valid `ψ` or mark as undefined and treat yaw cap as `∞` there.
- ensure `s` is strictly increasing (guard tiny `Δs` with eps).

### Step C — Integrate into the speed cap pipeline

1) **θ coupling**:
- if `use_theta_max`, compute `a_lat_max = g*tan(θ_max)` and use it for curvature speed cap.
- show the derived `a_lat_max` in diagnostics/UI so it’s obvious what value is in effect.

2) **yaw-rate speed cap** (only when `heading_constrained=True`):
- compute `v_yaw(s) = ψ̇_max / max(|dψ/ds|, eps)`
- update the pointwise cap: `v_cap(s) = min(v_max, v_kappa(s), v_yaw(s))`

3) **hard curvature feasibility**:
- if `kappa_max_1pm` (or derived from `r_min_m`) is enabled:
  - compute violations `|κ(s)| > κ_max`
  - treat the curve as **infeasible** for optimization selection (time = `∞` / reject λ)
  - still visualize the curve and highlight violations so the user understands why it was rejected

### Step D — Optimization selection semantics

To keep behavior predictable:
- In `optimize_lambda_grid`, filter infeasible candidates first (geometry/yaw feasibility if we choose yaw as feasibility rather than cap).
- If all candidates are infeasible, return the “least bad” result with a clear `status="infeasible"` and include violation details in diagnostics.

### Step E — Visualizations and UI additions

Add UI controls (toggles + sliders) with info tooltips:
- `Enable kappa_max / r_min` + a numeric field (prefer `r_min` in meters for humans, derive `kappa_max=1/r_min`)
- `Enable theta_max coupling` + `theta_max_deg` slider; show derived `a_lat_max`
- `Heading constrained` toggle + `yaw_rate_max` slider

Add plots/overlays:
- Overlay caps on speed plot: `v(s)`, `v_cap(s)`, `v_kappa(s)`, and `v_yaw(s)` when enabled.
- Add a “constraint active” plot or legend that indicates which constraint is binding at each `s`.
- Highlight `kappa_max` violations directly on the 3D path (e.g., red markers) and/or on a κ plot threshold line.

### Step F — Artifacts and tests

Artifacts:
- Export `ψ(s)`, `dψ/ds`, `v_yaw(s)`, `a_lat(s)` and violation masks when relevant.

Tests:
- θ coupling: verify `a_lat_max` equals `g*tan(θ)` within tolerance.
- yaw unwrapping: construct a path whose heading crosses `±π` and ensure `dψ/ds` is stable.
- kappa feasibility: create a tight turn and verify it is flagged infeasible when `κ_max` is small.

### 5) Artifact exports

Remaining:
- Extend `run_optimize.py` to optionally export:
  - CSV of samples: `x,y,z,s,kappa,v,v_cap,v_kappa`
  - Plotly HTML snapshot of the 3D plot (and/or diagnostics)
- Keep artifacts under `Optimization_Engine/artifacts/` with predictable filenames.

### 6) Numerical robustness pass

Remaining:
- Align curvature estimation with the arc-length/tangent finite-difference approach (or explicitly document the chosen estimator).
- Improve stability on:
  - near-collinear points,
  - very small \(\Delta s\),
  - large coordinate magnitudes (common with cm inputs).

### 7) Tests expansion

Remaining:
- Unit tests for:
  - Unreal conversion (cm→m + axis remap once defined)
  - binding detection logic
  - yaw-rate constraint logic (if implemented)
- Add a small regression test that ensures time decreases when increasing `a_lat_max` (all else equal) for a curved scenario.

## Open confirmations (needed before implementing some remaining items)

- Unreal axis/handedness: do we need a remap for display parity, or keep identity?
- Lateral acceleration model: should curvature be computed from full 3D curvature magnitude, or curvature in the XY plane only?
- Yaw constraint semantics: should yaw follow direction of travel (tangent), or be decoupled?
