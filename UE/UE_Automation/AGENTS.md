# Unreal Automation (Host-side)

This folder contains host-side scripts that launch `UnrealEditor` in unattended/headless mode and run Unreal Editor Python automation against a `.uproject`.

## What this is for
- Repeatable “one command” runs locally or in CI (Continuous Integration) pipelines (automated jobs that run builds/tests/automation on a schedule or on every commit).
- Capturing logs deterministically (no interactive UI).
- A “formal run” entrypoint for Unreal automation (avoid manually opening the editor when you just want scripted changes).

## `run_unreal_editor_python.py`
Launches Unreal Editor and executes a UE Python script via `-ExecutePythonScript=...`.

### Requirements
- Unreal Editor installed (UE 5.7 for this project).
- A valid `.uproject` path.
- Project must have the needed editor plugins enabled (for example: `PythonScriptPlugin`, `GeometryScripting`, `EditorScriptingUtilities`).

### Usage
Run the default automation script for this repo:
```bash
python3 UE_Automation/run_unreal_editor_python.py
```
Defaults to `UE_Drone_Env_1/Scripts/create_red_sphere_course.py`.

Pass arguments through to the UE Python script (everything after `--`):
```bash
python3 UE_Automation/run_unreal_editor_python.py -- --diameter-cm 100 --color 1,0,0
```

Run a specific UE script (recommended for clarity in automation jobs):
```bash
python3 UE_Automation/run_unreal_editor_python.py \
  --script "$(pwd)/UE_Drone_Env_1/Scripts/replace_red_spheres_with_red_toruses.py"
```

Deterministic randomness (if the UE script supports `--seed`):
```bash
python3 UE_Automation/run_unreal_editor_python.py \
  --script "$(pwd)/UE_Drone_Env_1/Scripts/replace_red_spheres_with_red_toruses.py" \
  -- --seed 123
```

Override the Unreal Editor binary location:
```bash
UNREAL_EDITOR="/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor" \
  python3 UE_Automation/run_unreal_editor_python.py
```

### Output
- Combined stdout/stderr log path defaults to `/tmp/ue_automation.log` (override with `--log`).
- A temporary `-userdir=...` is created under `UE_Drone_Env_1/Saved/AutomationUserDir/` by default to avoid polluting your normal editor user settings.
- Unreal’s standard log is also written under the userdir (for example: `UE_Drone_Env_1/Saved/AutomationUserDir/<timestamp>/Saved/Logs/UE_Drone_Env_1.log`).

## Notes / Gotchas
- UE 5.7 Python API does not expose `unreal.StaticMeshFactoryNew`; create new mesh assets via Geometry Scripting utilities (e.g. `GeometryScript_NewAssetUtils.create_new_static_mesh_asset_from_mesh`).
- This runner uses `-ScriptErrorsAreFatal` so any UE Python exception will fail the run.
- macOS quoting: do **not** manually add quotes inside the `-ExecutePythonScript=...` argument; pass it as `-ExecutePythonScript=/abs/path/script.py <args...>` and let Unreal’s macOS launcher quote the value, otherwise Unreal may parse an empty script and never run it.
- If a run appears to hang and `/tmp/ue_automation.log` only shows macOS service/XPC errors, the editor may be blocked by a sandboxed/restricted execution environment; run the automation from a normal user session and clean up any stuck editor process before retrying (e.g. `pgrep -fl UnrealEditor` then `kill <pid>`).
- Component API mismatch: in UE 5.7 Python, some component convenience methods may be missing (e.g. `StaticMeshComponent.get_component_location()`), so prefer actor transform APIs or fetch component transforms via properties.
- Fatal-on-error cascades: when `-ScriptErrorsAreFatal` is set, a Python exception may be followed by a crash; diagnose using the *first* Python traceback in `/tmp/ue_automation.log`.
