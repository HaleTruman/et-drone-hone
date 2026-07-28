# Multi-Gate Detector and Pose Solver

This package trains a one-class Keypoint R-CNN from the Unreal captures. For
each image it returns every detected gate with:

- a confidence score and bounding box;
- four outer corners ordered geometrically as image TL, TR, BL, and BR;
- camera-relative right/up/forward position from a calibrated planar solver;
- yaw modulo 180 degrees, plus pitch and roll.

The corner labels deliberately do not encode the actor's front or rear side.
Yaw is represented in `[-90, 90)`, so two poses 180 degrees apart are
equivalent. Position and orientation are solved independently for each
detection using the known 270 cm gate dimensions and camera calibration stored
in the checkpoint.

## Environment

Run the ML pipeline with normal Python, not Unreal Editor's embedded Python:

```powershell
python -m pip install -r src/models/requirements.txt
```

## Train

The command discovers every `run_*` directory. It trains on all but the
highest-numbered run and evaluates against that newest held-out run:

```powershell
python -m src.models.run_pipeline
```

Useful explicit overrides:

```powershell
python -m src.models.run_pipeline `
  --data-root datasets `
  --train-runs run_001 run_002 run_003 run_004 `
  --test-runs run_005 `
  --epochs 30 `
  --batch-size 2 `
  --output-dir artifacts/gate_pose
```

The first pretrained run downloads torchvision's COCO Keypoint R-CNN weights.
If the download is unavailable, training now fails clearly instead of silently
falling back to random initialization. Use
`--allow-random-init-on-pretrained-failure` only when you intentionally want to
train from scratch; `--no-pretrained` skips the attempt. Training writes a
format-v2 `best.pt`, `last.pt`, `history.jsonl`, and `split_manifest.json`.
`best.pt` is selected by validation detection F1, with validation loss as the
tie-breaker. Legacy single-gate checkpoints are intentionally incompatible and
must be retrained.

Current defaults are tuned to reduce duplicate gate detections:

- `--score-threshold 0.7`
- `--detections-per-image 12`
- `--duplicate-iou-threshold 0.25`
- `--suppress-contained-duplicates`

The postprocess keeps the highest-scoring gate first, then removes lower-score
boxes that overlap it or look like contained fragments of the same gate.

## Evaluate

```powershell
python -m src.models.testing.evaluate `
  --checkpoint artifacts/gate_pose/best.pt `
  --data-root datasets `
  --runs run_005
```

Evaluation writes detection precision/recall/F1, matched keypoint and pose
metrics, machine-readable `predictions.jsonl`, and annotated held-out frames
under `evaluation/overlays`. Use `--overlay-count N` to save an evenly sampled
subset, or `--overlay-count 0` to disable overlays.

Evaluation also writes `threshold_sweep` in `metrics.json` and prints the best
score threshold by detection F1. This helps choose the operating point without
rerunning inference.

## Infer

Run the detector -> per-gate pose-solver pipeline on one or more images:

```powershell
python -m src.models.inference frame.png another_frame.png `
  --checkpoint artifacts/gate_pose/best.pt `
  --output artifacts/predictions.jsonl `
  --overlay-dir artifacts/inference_overlays
```

Each JSONL record contains a `detections` list rather than one fixed gate.
Ordered input images also receive persistent `track_id` values by default; use
`--no-tracking` for unrelated still images.

## Test

```powershell
python -m pytest src/models/testing -q
```
