# UE_Build/WebSocket

This folder owns the tooling-side build path for the Unreal `DroneWebSocket` plugin.

## Scope

- `Plugin_Source/DroneWebSocket/` is the only hand-edited plugin source-of-truth.
- `bootstrap_plugin.py` installs or rebuilds the named plugin in `UE_Drone_Env/Plugins/`.
- `plugin_validate.py` checks whether the installed project plugin is actually usable.
- `run_websocket_build.py` runs bootstrap + validation as one logged build pass.
- This folder does **not** generate `.uasset` content. Unreal content generation stays in `UE_Tooling/UE_Build/Content_Generation/`.

## Files

- `Plugin_Source/DroneWebSocket/` — authored plugin descriptor, config, and C++ source.
- `assemble_websocket_assets.py` — shared path resolution, staging, project-build, validation, and artifact helpers.
- `bootstrap_plugin.py` — canonical project-plugin bootstrap/build entrypoint.
- `plugin_validate.py` — canonical post-bootstrap validation entrypoint.
- `run_websocket_build.py` — versioned runner that writes logs and a run manifest.
- `AGENTS.md` — Unreal build tips for working in this folder.

## Canonical flow

1. Edit plugin source under `Plugin_Source/DroneWebSocket/`.
2. Run `run_websocket_build.py` as the default entrypoint.
3. Review the emitted manifest and per-step logs under `UE_Tooling/Artifacts/websocket_build/`.

`run_websocket_build.py` is the primary workflow for this folder because it runs bootstrap + validation together and records versioned artifacts.

`bootstrap_plugin.py` and `plugin_validate.py` remain the underlying step scripts and can still be run directly for focused debugging.

## Script behavior

### `bootstrap_plugin.py`

- Targets one named plugin, default `DroneWebSocket`.
- If the project plugin already exists and `--overwrite` is **not** set, it logs that the plugin is already available and exits cleanly.
- If the project plugin is missing, or `--overwrite` is set, it stages the authoritative source into `UE_Drone_Env/Plugins/<PluginName>` and invokes the Unreal project editor build.
- It fails fast on missing inputs, staging errors, or build failures.
- It does **not** perform startup-readiness validation.

### `plugin_validate.py`

- Validates both `Plugin_Source/<PluginName>` and `UE_Drone_Env/Plugins/<PluginName>`.
- Confirms required files exist, source and project trees are in sync, the plugin descriptor is structurally valid, expected binaries exist, and Unreal can start headless with the plugin enabled.
- Owns the startup-ready decision for this folder.

### `run_websocket_build.py`

- Runs `bootstrap_plugin.py` then `plugin_validate.py` in order.
- Writes artifacts under `UE_Tooling/Artifacts/websocket_build/<run_name>/`.
- Writes step logs under `UE_Tooling/Artifacts/websocket_build/logs/<run_name>/`.
- Stops on the first failure unless `--continue-on-failure` is explicitly set.

## Typical commands

Build only:

`python3 UE_Tooling/UE_Build/WebSocket/bootstrap_plugin.py --overwrite`

Validate only:

`python3 UE_Tooling/UE_Build/WebSocket/plugin_validate.py`

Run the primary logged flow:

`python3 UE_Tooling/UE_Build/WebSocket/run_websocket_build.py --overwrite-project-plugin --version-id v001`

## Output and troubleshooting

- Build/validation runner manifests: `UE_Tooling/Artifacts/websocket_build/`
- Per-step logs: `UE_Tooling/Artifacts/websocket_build/logs/`
- Headless startup probe logs: `UE_Drone_Env/Saved/WebSocketValidationUserDir/`

If `bootstrap_plugin.py` succeeds but `plugin_validate.py` fails, treat the plugin as not ready. Validation is the final authority, not bootstrap.
