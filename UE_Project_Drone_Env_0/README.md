# Unreal Drone First Project

High-level scaffolding + architecture overview for this Unreal Engine project.

## Project type
- Unreal Engine project (EngineAssociation: **5.7**) defined by `FirstProject.uproject`.
- Currently **Blueprint-only** (no `Source/` directory / C++ modules in this repo).

## Key entry points
- Project descriptor: `FirstProject.uproject`
- Default startup + game map: `/Game/DemoTemplate/_Core/Lvl_IntroRoom`
- Default GameMode: `/Game/FirstPerson/Blueprints/BP_FirstPersonGameMode`
- Input system: **Enhanced Input** (default input classes point to `EnhancedPlayerInput` / `EnhancedInputComponent`)

## Repository scaffolding
- `Config/`: Project settings (`DefaultEngine.ini`, `DefaultInput.ini`, etc.).
- `Content/`: All game assets (mostly binary `.uasset` / `.umap`).
  - `Content/FirstPerson/`: First-person template Blueprints and related assets.
  - `Content/DemoTemplate/`: Demo/template content including the default intro level.
  - `Content/Input/`: Enhanced Input assets (e.g. `IMC_*`, `IA_*`).
  - `Content/__ExternalActors__/` + `Content/__ExternalObjects__/`: Externalized assets (one-file-per-actor style; typically editor-generated).
- `DerivedDataCache/`: Cached derived assets (generated; safe to delete and regenerate).
- `Intermediate/`: Build/intermediate artifacts (generated).
- `Saved/`: Editor/runtime outputs (logs, autosaves, derived config; generated).

## Plugins enabled (via `FirstProject.uproject`)
- `ModelingToolsEditorMode` (Editor)
- `Landmass`
- `InEditorDocumentation` (Editor)
- `AIAssistant`

## Notes
- Most gameplay logic lives in **Blueprint assets**, not text files—use Unreal Editor to inspect/diff.
- If you later add C++ code, Unreal will generate a `Source/` directory and build targets/modules.
