# Pose Estimation

This folder contains the deterministic 3D square pose-fit stage used by the
0721Vision pipeline.

## Inputs

```text
assets/bbox_contours/<run_key>/contours_manifest.json
assets/bbox_clipping/<run_key>/clipping_manifest.json
```

The stage reads bbox-local contour geometry and clipping validity from the
maskbits-derived pipeline.

## Output

```text
assets/pose_estimation/<run_key>/3d_pose_fit.json
```

The output contains per-frame, per-bbox 3D pose fit records for instance
tracking and the UI volume view.

## Runtime Boundaries

`pose_fit.py` supports the existing batch manifest builder and exposes a
frame-level `build_pose_frame(...)` function for `pipeline.py`.

## Inspect The Contract

```bash
python src/pose_estimation/pose_fit.py --describe
```
