# Instance Tracking

This folder contains the deterministic frame-to-frame instance mapping stage
used by the 0721Vision pipeline.

## Inputs

```text
assets/mask_bboxes_maskbits/<run_key>/bbox_manifest.json
assets/pose_estimation/<run_key>/3d_pose_fit.json
```

The stage associates bbox observations across frames, using pose fit records
when available.

## Output

```text
assets/instance_tracking/<run_key>/instance_mapping.json
```

The output contains stable instance IDs, per-frame observations, current pose
references, and compact association diagnostics for the UI JSON feed.

## Runtime Boundaries

`instance_mapping.py` supports the existing batch manifest builder and exposes
`InstanceTrackingRuntime` for ordered frame-by-frame pipeline execution.

## Inspect The Contract

```bash
python src/instance_tracking/instance_mapping.py --describe
```
