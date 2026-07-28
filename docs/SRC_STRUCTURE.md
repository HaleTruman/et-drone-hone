# Dataset Automation Source

This folder separates the two main automation concerns for the gate-detection dataset.

## Folders

```text
src/
  app.py
  ui/
  dataset_generation/
    config.py
    common.py
    unreal_editor.py
    obstacles.py
    generate.py
    gate_randomization/
    dataset_collection/
    hangar_extraction/
  models/
    config.py
```

`dataset_generation/gate_randomization/` is for code that creates or mutates the training scene:

- random track generation
- gate placement
- gate orientation

`dataset_generation/dataset_collection/` is for code that loops through generated scenes and records data:

- camera pose generation
- scene capture
- frame export
- corner projection
- bbox calculation
- JSON/JSONL metadata writing

Shared files:

- `dataset_generation/config.py`: project paths, bounds, frame size, and generation constants
- `dataset_generation/common.py`: small pure helpers and Unreal object label/path helpers
- `dataset_generation/unreal_editor.py`: editor setup, level loading, actor lookup, and viewport invalidation
- `models/config.py`: model training, evaluation, inference, and validation defaults

Dataset collection files:

- `pipeline.py`: high-level orchestration and camera sequence tick loop
- `camera_poses.py`: random camera pose generation and camera FOV helpers
- `capture.py`: render target loading, scene capture syncing, and PNG export
- `projection.py`: camera-frame coordinates, relative orientation, 2D projection, and bboxes
- `visibility.py`: line traces and corner visibility flags
- `metadata.py`: gate corner discovery and JSON/JSONL metadata records
- `outputs.py`: run folder and metadata path management

`dataset_generation/generate.py` is the Unreal entrypoint and runs gate randomization before dataset collection.
