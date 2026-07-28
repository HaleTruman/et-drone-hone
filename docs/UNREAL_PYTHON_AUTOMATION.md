# Unreal Python Automation

This project uses Unreal Editor Python scripts to randomize racing gate placement, move the capture camera, and optionally export render target frames for training data.

## Starting MCP Servers for running AI
In the CMD console in UE run the following:

```
ModelContextProtocol.StartServer 8000
ModelContextProtocol.GenerateClientConfig Codex [or other model]
```

## Current Script

Main dataset collection entrypoint:

```text
src/dataset_generation/generate.py
```

Compatibility wrapper:

```text
src/dataset_generation/dataset_collection/collect_gate_dataset.py
```

It currently:

- Finds gate actors in the outliner folder `Gates/`.
- Places all gates along a generated race-track path.
- Orients each gate toward the next gate.
- Moves the capture camera through multiple randomized positions per gate.
- Adds camera aim randomness while keeping the gate center inside the camera frame.
- Syncs the scene capture to the camera.
- Saves frames from `/Game/NewTextureRenderTarget640x360`.
- Writes corner projections, bboxes, and relative camera-frame pose metadata.

Frame saving is controlled near the top of the script:

```python
SAVE_FRAMES = False
```

Set it to `True` when you want image files written to:

```text
datasets/dataset_###/runs/run_###/frames/
```

Metadata is written to:

```text
datasets/dataset_###/runs/run_###/metadata.json
datasets/dataset_###/runs/run_###/metadata.jsonl
```

## Running A Python Script Inside Unreal

Run the script from the Python terminal in UE and run:
```python
exec(open(r"C:\Users\brend\Projects\UE_5\TestProject\ML\src\dataset_generation\generate.py").read())
```

Otherwise, you can do:

1. Open the project in Unreal Editor.
2. Open the target level, or let the script load `TARGET_LEVEL_PATH` from `src/dataset_generation/config.py` (`/Game/Hangar1` by default).
3. In Unreal, open:

```text
Tools > Execute Python Script
```

4. Select the script file from the project root, for example:

```text
C:\Users\brend\Projects\UE_5\TestProject\ML\src\dataset_generation\generate.py
```

5. Watch the Output Log for messages, warnings, or saved frame paths.

## Required Plugins

Make sure these plugins are enabled:

- Python Editor Script Plugin
- Editor Scripting Utilities

After enabling plugins, restart Unreal Editor before running scripts.
