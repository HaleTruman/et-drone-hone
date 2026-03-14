# UE_Tooling/UE_Build/WebSocket/AGENTS.md

This file is for Unreal build workflow guidance only. Keep repo-specific product behavior out of this file.

## Useful Unreal build guidance

- Derive the Unreal engine install from the project `.uproject` `EngineAssociation` before hardcoding engine paths.
- Build project plugins against the project editor target, not the module name. On macOS the expected pattern is `Build.sh <ProjectName>Editor Mac Development <Project.uproject> -WaitMutex -NoHotReloadFromIDE`.
- Treat `Binaries/`, `Intermediate/`, `Saved/`, and `__pycache__/` as generated output. Do not use them as source-of-truth when comparing plugin trees.
- For overwrite-style rebuilds, delete the installed project plugin tree before rebuilding. Do not merge a fresh source tree onto stale generated outputs.
- Validate plugin builds with a headless Unreal startup probe after the build step. A successful compile is not enough if the editor cannot load the plugin cleanly.
- Prefer writing Unreal stdout/stderr to log files during validation runs. Headless editor startup can emit enough output to make in-memory capture noisy and brittle.
- If a `.uplugin` descriptor omits `EngineVersion`, use the project’s `EngineAssociation` as the local compatibility reference during validation, then confirm the actual startup result separately.
- When a build fails, preserve the exact build command, exit code, and output paths. Unreal build failures are much faster to debug with the original command intact.
