# UE_Build

Build-time Unreal tooling lives here.

- `Content_Generation/`: Unreal Editor Python scripts that generate `.uasset`/`.umap` content.
- `WebSocket/`: source-of-truth scaffolding and sync/deploy helpers for the `DroneWebSocket` UE plugin.

`run_unreal_build_gen.py` lives at `UE_Tooling/run_unreal_build_gen.py` and is used to execute build/generation scripts headlessly.
