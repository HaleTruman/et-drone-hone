## Pipeline stages, in order

- `mask.py` — raw camera frame to base mask.
- `mask_fill.py` — closes small gaps.
- `mask_bridge.py` — bridges the outer hull to remove interior noise.
- `void_center.py` — finds real enclosed voids, validates them, computes centers.
- `mask_clipping.py` — flags detections touching the frame edge.
- `instance_tracking.py` — persistent identity for instances and voids across frames: IOU +
  distance + area matching, camera-motion compensation, earned reliability. Uses
  `final_2d_estimate.py` internally for per-frame confidence scoring; not a standalone stage.
- `camera_odometry.py` — telemetry-based rotation compensation used by tracking; not a
  standalone stage on its own.
- `final_2d_void_estimate.py` — judges tracked voids into a bounded set of published slots
  (`final_void_estimates`), with gap-filling and ghost-pool reconnection across brief losses.
- `final_3d_pose_estimate.py` — fuses tracked voids with telemetry into persistent 3D camera
  rays, published as `vision_observation` (`VisionObservation`/`VisionGateObservation`,
  camera-optical frame) — see `old_dontuse/v0_3d_pose_plan.md` for the design.
- `pipeline.py` — runs the stages above per frame and writes review artifacts.

## Naming conventions

- Instances: `instance_track_NNN`. Voids: `void_track_NNN`. IDs are minted once and never
  reused or renamed.
- `reliable` is earned (a streak) and persists through a track's grace window — it is not the
  same thing as "matched this exact frame" (`track_match`, which is `None` on frames with no
  match).
- `mask_clipping` means "touching the frame edge" — it gates tracking confidence, it does not
  exclude a detection from being tracked at all.
- In `vision_observation`, `gate_id` is the void's `track_id`, not `final_void_id` (which is a
  reused slot number, not a stable identity).
- "Passed" (a landmark moved behind the camera's current heading) isn't a published field — a
  passed `track_id` simply stops appearing in `vision_observation.gates`, with no explicit
  marker in the output.

## Run it

```bash
.venv/bin/python mask_review_pipeline/src/pipeline.py <run_dir>
```

`<run_dir>` needs `vision_frames/`; a `telemetry.json` alongside it (same shape as
`runs/run-20260724T165437Z/telemetry.json`) is used by `final_3d_pose_estimate.py` when
present and skipped gracefully when not.

## Test it

```bash
.venv/bin/python mask_review_pipeline/tests/test_pipeline.py
.venv/bin/python mask_review_pipeline/tests/test_final_3d_pose_estimate.py
```

No `pytest` in this environment — both test files are plain scripts.
