# UE Content Automation — Dev Brief

This folder contains Unreal Editor Python scripts that **create/modify Unreal content** (assets + placed actors) via headless automation runs (use `UE_Automation/run_unreal_editor.py`).

## Current scripts (what exists today)
- `create_red_sphere_course.py`
  - Creates/updates a solid-color material and a generated sphere static mesh (no imports).
  - Spawns/updates labeled `StaticMeshActor`s in a chosen level to form a simple course layout.
- `replace_red_spheres_with_red_toruses.py`
  - Generates a torus static mesh (no imports) with a 1m inner opening (default) and assigns a solid red material.
  - Replaces existing labeled actors’ mesh + material and assigns random rotations (seeded).
- `optimized_path/`
  - Staging area for **external optimizer artifacts** intended to drive in-editor construction (see below).

## Integration point: optimizer → UE spline build
`Path_Optimizer/optimizer/devreadme.md` indicates the optimizer will output an “ideal path spline” JSON, and the intended landing path for UE consumption is:
- `UE_Drone_Env_1/UE_Project_Scripts/Content_Automation/optimized_path/dev-spline-example.json`

### Expected optimizer deliverables (in JSON)
- Basic optimized path geometry (spline control points or a sampled centerline).
- Orientation along the path (stable/performance-focused; exact representation TBD).
- Optional: calculated drone inputs (global → prop-local). UE may ignore these initially.

### Goal
Add a UE automation script (new) that:
1) Reads an externally generated “optimized path spline” JSON from `optimized_path/` (or a passed-in absolute path).
2) Builds/updates a spline representation in a chosen UE level (e.g., a `SplineComponent` on an actor).
3) Saves the updated level (and any created assets if required).

## Requirements (high-level)
- **Headless, repeatable runs:** all scripts must be runnable via `UE_Automation/run_unreal_editor.py --script ... -- <args>`.
- **No external asset imports:** geometry/material generation should remain procedural where applicable.
- **Deterministic outputs:** support `--seed` where randomness is used; keep ordering stable.
- **Explicit target level:** scripts should take `--level /Game/...` and not assume a single map forever.
- **Clear file I/O policy:** if consuming JSON from outside UE, accept absolute paths and fail with actionable errors.

## Work required (next steps)
1) Define the **optimized spline JSON schema** to be consumed by UE (minimum: control points, tangents, units, frame, closed/open).
2) Implement `build_spline_from_json.py` (name TBD) in this folder:
   - parse args: `--level`, `--input-json`, optional `--actor-name` / `--component-name`
   - load level
   - create/find a spline actor/component
   - set spline points + tangents from JSON
   - save level
3) Add a small example JSON to `optimized_path/` that matches the agreed schema (replace the empty placeholder).
4) Add minimal logging and validation so failures are diagnosable from `/tmp/ue_automation.log`.

## Open questions / concerns
- **Schema:** cubic Hermite vs sampled polyline vs Bezier/B-spline; which fields are mandatory (tangents, up/roll, speed profile)?
- **Units/frame:** confirm JSON uses Unreal world `cm` in `frame="unreal"` (preferred to avoid ambiguity).
- **Naming/versioning:** the optimizer notes “a filename similar to `targets-...json`” but the current placeholder path is a fixed `dev-spline-example.json`; confirm the long-term naming convention and whether UE should accept either.
- **Spline ownership:** should the spline live as:
  - a placed actor in the level only (simplest), or
  - a reusable asset (e.g., Blueprint/actor asset) plus an instance in the level?
- **Ordering:** if the optimizer emits a sequence, we assume order is authoritative; any “course order” inference should be explicit and opt-in.
