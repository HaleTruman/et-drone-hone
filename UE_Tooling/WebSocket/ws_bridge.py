"""
ws_bridge hosts the centralized WebSocket runtime pipe for UE + tooling.

Primary function:
- Manage connection lifecycle for UE runtime clients.
- Send startup/runtime config payloads (`SET_CONFIG`) and route command traffic (`CMD`, `CAPTURE_NOW`).
- Receive and fan out incoming runtime messages (`OBS`, `STATUS`, `ACK`, `ERROR`) for controller/sampler use.

Pipeline relationships:
- Uses `protocol.py` for message envelope semantics.
- May use `schemas.py` to validate inbound/outbound payloads.
- Coordinates with `Config/RunConfig.py` output at startup.
- Serves `Drone_Controller/*` and `Data_Interface/*` as a shared transport endpoint.

UE/runtime relationships:
- UE acts as WebSocket client; this bridge acts as authoritative server process in tooling.

Current status:
- Documentation scaffold only. Bridge implementation details are intentionally pending.
"""
