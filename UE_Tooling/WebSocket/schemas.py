"""
schemas provides optional strict validation for protocol payloads.

Primary function:
- Validate incoming/outgoing messages against expected field sets and types.
- Enforce schema version checks so incompatible producers/consumers fail loudly.
- Reduce silent data-quality failures during rapid iteration.

Pipeline relationships:
- Works with `protocol.py` envelope definitions.
- Used by `ws_bridge.py` and optionally by controller/sampler components.
- Complements `Config/RunConfig.py` traceability fields (`schema_version`, `config_hash`).

UE/runtime relationships:
- Supports stable cross-version communication expectations between UE runtime and tooling.

Current status:
- Documentation scaffold only. Validation implementation is intentionally pending.
"""
