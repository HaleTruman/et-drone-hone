# Project Milestones

## Unreal plugins (HTTP over localhost)

- **Asset Reader (Editor-only resolver)**  
  Runs inside Unreal Editor and binds `127.0.0.1`, resolving all cerialized UnrealEngine file paths into stable JSON: level name, actor label, class, GUID/stable ID, and canonical package/object path.  
  Maintains an O(1) cache/index via editor events (level load/unload, actor add/remove/rename/save); bonus endpoints: focus actor, list changed external actors since last save, and dump the full mapping to a JSON file for offline use.

- **Control Router (stable gameplay controller)**  
  Exposes one stable router target on `127.0.0.1` (doesn’t change on respawn/level reload) with endpoints like `POST /control/move|look|goto|stop|possess|run_scenario` that translate into real Unreal gameplay calls.  
  Defines timing semantics (persistent desired-state or expiring impulses), plus `GET /health` heartbeat and optional `GET /state` (pawn, location, velocity, scenario state) for closed-loop control.

## Repo layout
- Unreal plugins live under `UE_Project_Drone_Env_0/Plugins/`.
- External clients live at repo root (VS Code extension + Python controller) and call the plugins over localhost HTTP.
