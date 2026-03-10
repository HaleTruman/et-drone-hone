# Drone_Content Generation Conventions

This folder is the source for generating reusable drone kit assets in `/Game/Drone_Content/*` (mesh, materials, BP modules, interfaces, and defaults).  
Unlike course maps, these assets are foundational dependencies for runtime IO/control/sampling and should stay stable across runs.

## Recommended assembly strategy

Do **not** use `gen_bp_dronepawn.py` as the full assembly package entrypoint.

Best-practice is:

1. Keep each `gen_*` script focused on one asset.
2. Use the shared helper `drone_asset_assembly.py` to centralize create/load/save/idempotent logic.
3. Use the orchestrator runner `run_drone_content_generation.py` to call generators in dependency order and emit a version manifest.

Why this is better:

- avoids logic duplication across many `gen_*` scripts,
- keeps composition separate from asset-specific logic,
- allows deterministic rebuilds and easier migration when UE schema/contracts evolve,
- keeps runtime references predictable for IO + controller + sample-manager systems.

## Versioning decision (recommended)

Use **stable canonical asset paths** as default output:

- `/Game/Drone_Content/Blueprints/...`
- `/Game/Drone_Content/Interfaces/...`
- `/Game/Drone_Content/Data/...`
- `/Game/Drone_Content/Materials/...`
- `/Game/Drone_Content/Meshes/...`

Do versioning through manifests (not per-run path churn by default):

- Write build metadata to `UE_Tooling/Artifacts/drone_content/<kit_name>_<version>_<timestamp>.json`
- Include: script name/version/date, command, return code, output asset paths, created/reused status, config/schema references.

Optional release mode (future): snapshot copy to a versioned UE subfolder only when intentionally freezing a release.

## Desired dependency order

1. Interfaces (`BPI_*`)
2. Data defaults (`DA_*`)
3. Meshes + materials (`SM_*`, `M_*`, `MI_*`)
4. Module blueprints (`BP_DroneMovement_6DOF`, `BP_DroneSensors`, `BP_DroneTelemetrySampler`)
5. Container pawn (`BP_DronePawn`) that wires modules + interfaces

This order minimizes broken references and keeps regeneration idempotent.

## Relationship to runtime IO/WebSocket

Drone_Content generation must stay compatible with:

- `UE_Tooling/Config/RunConfig.py` (`SET_CONFIG` structure),
- `UE_Tooling/WebSocket/protocol.py` (message envelope + command routing),
- io assets (`BP_SetDataConfig`, `BP_DroneController`, `BP_DroneSpawner`, `BP_SampleManager`).

Practical rule: generate stable assets first, then let runtime config/control/sampling apply behavior **in memory** per run.

## Naming conventions

- Generators: `gen_<asset_name>.py`
- Shared helper: `drone_asset_assembly.py`
- Orchestrator: `run_drone_content_generation.py`
- Unreal prefixes: `BP_`, `BPI_`, `DA_`, `M_`, `MI_`, `SM_`

## Current status

Current `gen_*` scripts, `drone_asset_assembly.py`, and `run_drone_content_generation.py` are placeholder scaffolds with documented roles and TODO runtime generation logic.
