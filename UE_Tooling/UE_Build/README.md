# UE_Build

Build-time Unreal tooling lives here.

- `Content_Generation/`: Unreal Editor Python scripts that generate `.uasset`/`.umap` content.
- `WebSocket/`: source-of-truth scaffolding and sync/deploy helpers for the `DroneWebSocket` UE plugin.

Entry points:
- `UE_Tooling/run_unreal_build.py` is the canonical top-level build orchestrator.
- `UE_Tooling/UE_Build/run_unreal_build_gen.py` is the shared low-level Unreal executor used to run one UE Python script headlessly.

Build manifest outputs:
- Canonical top-level manifest: `UE_Tooling/Artifacts/build/<run_name>.json`
- Compatibility mirror: `UE_Tooling/Artifacts/build_runs/<run_name>/run_manifest.json`

Current orchestrated build order:
1. WebSocket
2. Course
3. Drone
4. IO (through `run_unreal_build_gen.py` with explicit authoritative `--level` handoff)
