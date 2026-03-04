# UE_Tooling/AGENTS.md

This folder is the **outside-the-engine control plane** for the Unreal simulation. Its job is to (1) generate Unreal content through headless Editor scripts when needed, (2) drive simulation runs (spawn, configure, command, capture), and (3) persist datasets and run manifests **outside** the Unreal project. This is intentionally where the “IP” lives: YAML configs, run orchestration, and dataset organization.

## Ground rules (do not violate)
Unreal does **not** parse YAML at runtime. YAML is parsed in Python tooling only, then translated into a single `SET_CONFIG` JSON payload that Unreal can consume. Unreal receives config via WebSocket and applies it in memory.

WebSockets are treated as the “pipe.” Message meaning is defined by the application protocol in `WebSocket/protocol.py`. Don’t hardcode ad-hoc message shapes in multiple scripts.

## Folder map

### run_unreal.py
Launches Unreal (Editor or headless, depending on flags), points it at the correct `.uproject` and map, and ensures the sim boots into a consistent starting state. This is the “entrypoint runner” for most automated runs.

### Content_Generation/
Python scripts that run *through Unreal Editor* to create or modify assets and/or place actors in maps. These are editor-time operations (asset creation, placements), not runtime simulation control.
- Examples: replacing spheres with toruses, creating marker actors, stamping course geometry.

### Data_Interface/
Responsible for collecting data from Unreal and writing an orderly dataset to disk (outside UE). This includes:
- receiving observation/capture messages 
- writing structured metadata observations.jsonl,
- managing `run_id`, `capture_id`, and manifests.

If the simulation streams observations over WebSocket, this folder is where those messages get decoded. 

### Drone_Controller/
Responsible for sending control commands into Unreal. In the current design, this means emitting `CMD` messages with `drone_id` and control values (pitch/roll/yaw/throttle or rates) over WebSocket.

### Config/
Source-of-truth YAML configs live here. These are human-editable, versionable run recipes.
- `Data_Interface_Config/` defines what to capture/log (pose, viewpoint(s), telemetry fields, capture cadence).
- `Drone_Controller_Config/` defines movement tuning and limits.
- `RunConfig.py` is the canonical builder that merges YAML configs into one `SET_CONFIG` JSON payload.

Important: `RunConfig.py` should be the single place where YAML → JSON decisions are made. Everything else imports it.

### WebSocket/
This folder exists to keep the transport/protocol logic clean and reusable across the controller and sampler scripts.

- `ws_bridge.py`  
  A single-process WebSocket server (recommended) that Unreal connects to. It accepts UE connections, sends `SET_CONFIG`, relays `CMD`, and receives `OBS/STATUS` messages.  
  It does not “create two ports” unless you intentionally implement two endpoints (e.g., control and data channels). In v0 you can run one port and multiplex message types.

- `protocol.py`  
  The application protocol layer: message types, required fields, and encode/decode helpers. WebSockets already provide message boundaries; this file defines the JSON envelope that makes multi-drone routing and versioning sane.

- `schemas.py`
  Validation for message shape and schema_version. 

- `README.md`  
  Local run instructions: ports, URLs, “start server then launch UE,” example `SET_CONFIG` / `CMD` / `CAPTURE_NOW` messages.

## WebSocket model (the key concept)
Python is the WebSocket **server**. Unreal is the WebSocket **client** (via the DroneWebSocket plugin component). This keeps tooling authoritative and keeps Unreal clean.

Messages must include:
- `type` (SET_CONFIG, CMD, CAPTURE_NOW, OBS, ACK, ERROR)
- `run_id` (string)
- `drone_id` (int or string; `all` allowed when appropriate)
- `seq` or `timestamp` (for ordering/traceability)
- `schema_version` (strongly recommended)
- `payload` (the actual data)

## Expected run flow (v0)
1) Start `WebSocket/ws_bridge.py` (server).
2) Launch Unreal via `run_unreal.py`.
3) Build `SET_CONFIG` JSON using `Config/RunConfig.py` (merging YAML inputs).
4) Send `SET_CONFIG` once at startup; wait for `ACK`.
5) Send `CMD` messages to drive drone(s).
6) Request captures (or receive streaming observations).
7) Persist metadata/images and write a run manifest that includes config hash and schema version.

## Common blockers & fixes
If Unreal “connects but config doesn’t apply,” assume schema mismatch or parsing failure. Validate that:
- `SET_CONFIG` matches `ST_RunConfig` expectations (field names and types).
- you received an `ACK` after `SET_CONFIG`.

If Unreal never connects:
- port already in use, or wrong URL, or plugin not enabled/compiled.
- confirm UE plugin compiled and the WS client component is attached to the right Blueprint in-map.
- ensure UE side conflicts are captured as client issuse, and tooling side issues are captured as server issues to scope trouble shooting. 

If commands lag:
- if you later stream heavy data (images), split control/data into two connections. Do not create per-drone sockets.

## Naming/traceability rules
Every run must have:
- `run_id`
- `config_id` + `config_hash` (hash of merged YAML or the final JSON payload)
- `schema_version`
Every capture must have:
- `capture_id`
- `run_id`
- `drone_id`
- `config_id`