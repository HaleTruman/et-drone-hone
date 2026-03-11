# WebSocket Runtime Bridge (Tooling Side)

This folder defines the shared transport/protocol layer between Unreal runtime and Python tooling.

## Intended startup order

1. Start `ws_bridge.py` (server process).
2. Launch UE runtime (client process) so it connects to the bridge.
3. Build `SET_CONFIG` from `UE_Tooling/Config/RunConfig.py`.
4. Send `SET_CONFIG` and wait for `ACK` before control/capture traffic.
5. Send `CMD` / `CAPTURE_NOW` as needed.
6. Receive `OBS` / `STATUS` and persist through data-interface tooling.

## Message model

Use `protocol.py` as the single source for message envelope/type semantics. Typical message types:

- `SET_CONFIG`
- `CMD`
- `CAPTURE_NOW`
- `OBS`
- `STATUS`
- `ACK`
- `ERROR`

Recommended common fields:

- `type`
- `run_id`
- `drone_id`
- `schema_version`
- `seq` or `timestamp`
- `payload`

## Validation

`schemas.py` is the optional strict validator for message shapes/versioning. Use it when you need fast
failure on malformed messages instead of best-effort parsing.

## Notes

- YAML remains tooling-side only; UE runtime should receive JSON payloads only.
- Keep transport logic centralized in this folder so controller/sampler scripts do not diverge.
- Current scripts are still scaffold-level; update this document with concrete command examples as
  runtime implementations are filled in.
