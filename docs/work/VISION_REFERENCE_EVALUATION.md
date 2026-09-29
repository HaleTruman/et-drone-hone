# Vision Reference Evaluation

`LegacyVision.review_pipeline.evaluate_reference_dataset` compares the live
Flight `deterministic_v3` and `deterministic_v3_2` backends against Unreal
reference truth. It resets both backends at every run boundary and writes
replaceable artifacts under `Viewer/logs/evaluation/runs`.

## Run

```powershell
python -m LegacyVision.review_pipeline.evaluate_reference_dataset dataset_001 `
  --run run_001 --run run_020 --render-overlays
```

Omit `--run` to process every run. Use `--start`, `--limit`, or `--max-runs`
for bounded iteration. Set `VISION_EVALUATION_DATASET_ROOT` when datasets do
not live at the default path in the root `AGENTS.md`.

Each evaluation contains:

- `metadata.json`: source selection, intrinsics, coordinate contract, and
  enabled artifacts.
- `frames.jsonl`: frame truth, both backend results, matches, errors, and
  per-instance metrics.
- `summary.json`: recall buckets, false positives, mask IoU, corner and pose
  error, tracking stability, publication latency, and runtime percentiles.
- `overlays/`: optional truth/prediction and prediction-mask review images.

## Coordinates

The evaluator uses one explicit transform:

```text
Unreal world (X, Y, Z-up) -> evaluation local NED (X, Y, -Z)
dataset camera (right, up, forward) -> OpenCV (right, -up, forward)
```

It derives the `VehicleState` body attitude that makes the live camera
extrinsic reconstruct the Unreal camera pose. Every frame verifies normalized
gate positions and pinhole corner projections before pose errors are accepted.

## Experimental Runtime Behavior

The `deterministic_v3_2` experiment adds a strict, fixed-cost red/orange HSV
mask to the existing calibrated LUT. This is intended to recover gate pixels
that the LUT rejects under the Unreal material and lighting.

To keep the recovered detections flight-ready:

- candidates are area-ranked and capped at six per frame;
- temporal history is capped at five views;
- current-frame planar PnP takes the fast path when its quality is at least
  `0.20`;
- the multi-view solver remains available for absent or weak metric geometry;
- apertures above a `3.0` aspect ratio require four consecutive detector
  frames before publication, preserving persistent high-angle gates while
  suppressing transient slender structures;
- non-finite pose solutions are rejected instead of failing the frame.

These bounds change only `deterministic_v3_2`. `deterministic_v3` remains the
comparison baseline.

## Validation Boundary

Synthetic acceptance requires higher detection/publication recall and mask
IoU than `deterministic_v3`, with reported mean, median, p90, p95, and maximum
latency. Occlusion, high-angle, zero-gate, tracking, corner, and pose results
must remain visible separately from aggregate metrics.

No real-flight image runs are present in this checkout. The warm-color
supplement can respond to unrelated red/orange objects, and synthetic pose
error remains higher under gate-on-gate occlusion. Before this experiment is
treated as flight-approved, replay available real logs and confirm that false
positives, pose error, identity stability, and latency do not regress.
