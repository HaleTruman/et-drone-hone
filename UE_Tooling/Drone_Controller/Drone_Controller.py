"""
Drone_Controller sends real-time control commands to UE runtime drones.

Primary function:
- Emit command messages (for example pitch/roll/yaw/throttle or rate targets) for a specific `drone_id`.
- Optionally emit higher-level run controls (pause/reset) if/when protocol supports them.

Pipeline relationships:
- Must use `WebSocket/protocol.py` as the single message contract source.
- Routes through `WebSocket/ws_bridge.py` so transport behavior is centralized.
- Works with `Drone_Spawner.py` (ensure drone availability first) and `Sampler_Manager.py`
  (capture/metadata aligned to command timeline).

UE/runtime relationships:
- Command ingress is expected to land in UE controller logic and be applied to drone pawns
  via command-receiver interfaces.

Current status:
- Documentation scaffold only. Control loop implementation is intentionally pending.
"""
