# Course_Content Generation Conventions

* TODO: Update file naming Convention for asset_assembly.py, and run_versioned_course_generation.py to course_asset_assembly.py, and run_course_content_generation.py

This folder contains Unreal Editor Python scripts for generating reusable `Course_Content` assets and assembling maps that use those assets.

## Folder layout

- `asset_assembly.py`
  - Shared asset assembly helper.
  - Owns low-level create/load/save logic for:
    - materials
    - meshes
    - blueprints
  - This is the single source of truth for reusable asset build operations.

- `Materials/`
  - Asset-specific material generation scripts.
  - Example: `gen_m_coursetorus_red.py`.
  - Should delegate heavy logic to `asset_assembly.py`.

- `Meshes/`
  - Asset-specific mesh generation scripts.
  - Example: `gen_sm_coursetorus_1mopening.py`.
  - Should delegate heavy logic to `asset_assembly.py`.

- `Course_Blueprints/`
  - Asset-specific blueprint generation scripts.
  - Example: `gen_bp_courselight_main.py`.
  - Should delegate heavy logic to `asset_assembly.py`.

- `Maps/`
  - Map assembly scripts that orchestrate asset generators and then place actors.
  - Example: `gen_l_coursetorus.py`.
  - Includes managed actor placement (toruses + one course light actor) and level save.
  - Should not duplicate material/mesh/blueprint asset assembly internals.

## Delegation model

1. Asset generation scripts (`Materials`, `Meshes`, `Course_Blueprints`) call into `asset_assembly.py`.
2. Map scripts call asset generation scripts, then perform map-only operations:
   - load/create map
   - ensure one managed light actor instance exists
   - clear managed actors
   - spawn/place actors
   - save level

This keeps asset creation modular and prevents drift between scripts.

## Script version/date policy

All scripts in this folder must define:

- `SCRIPT_NAME`
- `SCRIPT_VERSION` (semantic version string, e.g. `1.0.0`)
- `SCRIPT_DATE` (`YYYY-MM-DD`)

Rules:

1. Update `SCRIPT_DATE` whenever a script changes.
2. Bump `SCRIPT_VERSION` for behavior changes:
   - patch: bugfix/no interface change (`1.0.1`)
   - minor: backward-compatible feature (`1.1.0`)
   - major: breaking change (`2.0.0`)
3. Asset generation scripts are highest priority for strict version tracking.
4. Log script metadata at runtime so execution logs record provenance.

## Naming conventions

- Generators use `gen_<asset_name>.py`.
- Keep Unreal naming prefixes aligned to asset type:
  - `M_` material
  - `SM_` static mesh
  - `BP_` blueprint
  - `L_` level

## Running

Use `UE_Tooling/run_unreal_build_gen.py` and pass a specific script via `--script`, or use the versioned helper.

Examples:

- `python3 UE_Tooling/run_unreal_build_gen.py --script UE_Tooling/UE_Build/Content_Generation/Course_Content/Maps/gen_l_coursetorus.py -- --asset-dir /Game/Course_Content --level /Game/Course_Content/Maps/L_CourseTorus`
- `python3 UE_Tooling/UE_Build/Content_Generation/Course_Content/run_versioned_course_generation.py --course-name L_CourseTorus --version-id v001`

## Versioned runs and artifacts

For reproducible runs, generate each course instance into a unique subfolder under `Course_Content`:

- UE asset root pattern: `/Game/Course_Content/<course_name>_<version_id>_<timestamp>/`
- Map pattern: `/Game/Course_Content/<course_name>_<version_id>_<timestamp>/Maps/<course_name>`

This keeps map + generated assets grouped together for one run.

The recommended entrypoint is:

- `UE_Tooling/UE_Build/Content_Generation/Course_Content/run_versioned_course_generation.py`

This helper:

1. Builds run-scoped asset/map paths from course name + version + timestamp.
2. Runs the map generation script headlessly through `UE_Tooling/run_unreal_build_gen.py`.
3. Writes a run artifact JSON to `UE_Tooling/Artifacts/runs/` named:
   - `<course_name>_<version_id>_<timestamp>.json`

The artifact JSON captures:

- run identity (course, version, timestamp, run_name)
- map and asset target paths
- script provenance (script names, versions, dates, file mtimes)
- headless command details, log path, and return code
- post-run existence checks for generated `.uasset`/`.umap` files
