"""
protocol defines the shared application message contract used across tooling and UE runtime.

Primary function:
- Define message envelope conventions (type, run_id, drone_id, timestamp/seq, schema_version, payload).
- Provide encode/decode helpers and type constants for `SET_CONFIG`, `CMD`, `CAPTURE_NOW`,
  `OBS`, `STATUS`, `ACK`, and `ERROR`.
- Keep command/capture/config/data flows interoperable across scripts.

Pipeline relationships:
- Used by `ws_bridge.py`, controller/spawner tooling, and sampling/data tooling.
- Intended to align with UE-side message parsing/serialization expectations.
- Should be paired with `schemas.py` for validation when strict checks are enabled.

UE/runtime relationships:
- Prevents drift between UE runtime consumers/producers and Python-side tooling scripts.

Current status:
- Documentation scaffold only. Protocol helper implementation is intentionally pending.
"""
