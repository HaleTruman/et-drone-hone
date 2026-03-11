"""
Drone_Spawner performs episode/setup-time drone spawn control.

Primary function:
- Send spawn/reset/setup commands so the expected drone set exists before control/capture starts.
- Keep spawn behavior explicit and repeatable across runs/episodes.

Pipeline relationships:
- Uses `WebSocket/protocol.py` message conventions (`run_id`, `drone_id`, `type`, `payload`).
- Expected to route through `WebSocket/ws_bridge.py` transport.
- Works with `Drone_Controller.py` (control after spawn) and `Sampler_Manager.py` (capture after spawn).

UE/runtime relationships:
- Targets UE-side spawner/controller runtime logic (for example BP_DroneSpawner / BP_DroneController wiring).
- Should support deterministic reset semantics so data runs are reproducible.

Current status:
- Documentation scaffold only. Spawn command implementation is intentionally pending.
"""
