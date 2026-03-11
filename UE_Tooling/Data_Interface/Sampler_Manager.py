"""
Sampler_Manager orchestrates data collection for a run.

Primary function:
- Own the run lifecycle for sampling (`run_id`, `capture_id`, output folder layout).
- Coordinate when capture requests are sent and how resulting observations are persisted.
- Keep persistence deterministic so offline training can replay exactly what was captured.

Pipeline relationships:
- Calls/uses `Sampler.py` for payload decode + normalization.
- Uses `WebSocket/protocol.py` (+ optional `WebSocket/schemas.py`) for message shape consistency.
- Should align captured metadata with `Config/RunConfig.py` outputs (`config_id`, `config_hash`, `schema_version`).
- Operates alongside `Drone_Controller/Drone_Controller.py` and `Drone_Controller/Drone_Spawner.py`
  so setup/control/capture share the same run context.

UE/runtime relationships:
- Receives OBS/STATUS-style messages emitted from UE-side runtime systems (for example SampleManager logic).
- Can issue capture-trigger messages (for example CAPTURE_NOW) over the shared bridge.

Current status:
- Documentation scaffold only. Runtime implementation is intentionally pending.
"""
