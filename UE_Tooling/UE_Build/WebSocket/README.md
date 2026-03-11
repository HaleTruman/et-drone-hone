# UE_Build/WebSocket

This folder is the **tooling-side source-of-truth** for the Unreal `DroneWebSocket` plugin.

## Layout

- `Plugin_Source/DroneWebSocket/`: hand-edited plugin source intended to mirror UE plugin structure.
- `Prebuilt/DroneWebSocket/`: optional packaged/prebuilt plugin payload for quick deployment.
- `bootstrap_plugin_scaffold.py`: initialize tooling plugin source from UE project plugin (or create scaffold).
- `sync_plugin_source.py`: sync source between tooling and `UE_Drone_Env/Plugins/DroneWebSocket`.
- `deploy_prebuilt_plugin.py`: copy a prebuilt plugin payload into the UE project plugin path.

## Recommended flow

1. Bootstrap tooling source once with `bootstrap_plugin_scaffold.py`.
2. Hand-edit plugin source under `Plugin_Source/DroneWebSocket`.
3. Push updates into the UE project using `sync_plugin_source.py --direction tooling_to_project`.
4. Optionally deploy packaged/prebuilt payloads with `deploy_prebuilt_plugin.py`.
