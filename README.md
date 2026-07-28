# Gate Dataset Automation

This project automates Unreal Engine dataset generation for gate detection and pose estimation, then provides model training, evaluation, inference, and a small local viewer for inspecting captured frames and predictions.

## Layout

- `src/dataset_generation/`: Unreal Editor automation for randomizing gate tracks, waiting for the rendered track to settle, and collecting image frames plus metadata.
- `src/dataset_generation/hangar_extraction/`: scripts and obstacle data for extracting hangar columns/pylons used during track placement.
- `src/models/`: training, evaluation, inference, tracking, and validation code for the gate detector.
- `src/app.py` and `src/ui/`: local browser viewer for datasets, evaluation artifacts, and validation runs.
- `datasets/`: generated datasets. Each generation execution creates `dataset_###/runs/run_###/...`.
- `artifacts/`: trained model checkpoints and evaluation outputs.
- `validation/`: annotated reference-frame validation runs.
- `docs/`: workflow notes and deeper implementation documentation.

## Generation

Run the Unreal entrypoint from inside Unreal Editor:

```python
exec(open(r"C:\Users\brend\Projects\UE_5\TestProject\ML\src\dataset_generation\generate.py").read())
```

Generation settings live in `src/dataset_generation/config.py`. The generation runtime randomizes gates first, waits for the track to finish rendering or settle, then starts dataset collection.

To use an LLM with Unreal, install the ModelContextProtocol plugins in the Unreal project, then run these commands in the Unreal command console:

```text
ModelContextProtocol.StartServer 8000
ModelContextProtocol.GenerateClientConfig Codex
```

Replace `Codex` with another LLM/client name when generating config for a different tool.

## Models

Model defaults live in `src/models/config.py`. The main training/evaluation pipeline is:

```powershell
python -m src.models.run_pipeline
```

Generated run names use nested identifiers such as `dataset_001/run_001`.

## Viewer

Start the local viewer with:

```powershell
python src/app.py
```

Then open the printed local URL to inspect datasets, evaluation overlays, and validation outputs.
