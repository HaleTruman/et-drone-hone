"""
Sampler defines observation decoding and normalization behavior.

Primary function:
- Parse incoming observation/capture payloads from UE.
- Normalize payloads into one internal record shape independent of transport details.
- Return structured records to `Sampler_Manager.py` for persistence.

Pipeline relationships:
- Consumes message envelope helpers from `WebSocket/protocol.py`.
- May use `WebSocket/schemas.py` to validate payload version/shape before normalization.
- Produces records expected by dataset/run writers managed by `Sampler_Manager.py`.

UE/runtime relationships:
- Handles pose/viewpoint/telemetry payload sections and optional image references/bytes.
- Should preserve `run_id`, `capture_id`, `drone_id`, and timestamps for alignment with controller events.

Current status:
- Documentation scaffold only. Sampling logic implementation is intentionally pending.
"""
