# Vision Reference Dataset Guidance

This directory documents work that supports offline evaluation and iteration of
the flight vision stack. Treat reference datasets as evaluation assets first:
they should make changes to `deterministic_v3_2` measurable before any flight
code is changed.

## Current Objective

Use Unreal-generated reference datasets to improve gate detection, tracking,
occlusion handling, and gate pose estimation in
`Flight/src/sensing/vision/models/deterministic_v3_2`. Treat
`deterministic_v3` as the baseline backend for comparison. The intended workflow
is:

1. Your datasets live at `C:\Users\brend\Projects\UE_5\TestProject\ml\datasets\evaluation`
2. Normalize camera, gate, and mask truth into flight-compatible coordinates.
3. Run baseline `deterministic_v3` and experimental `deterministic_v3_2` on the
   same frames.
4. Score predictions against truth.
5. Inspect overlays and failure cases.
6. Iterate inside `Flight/src/sensing/vision/models/deterministic_v3_2`.
7. Guard every improvement against real flight logs before touching shared
   flight code.

The primary engineering target is to make the `deterministic_v3_2` backend
better than the current `deterministic_v3` backend at gate detection and gate
pose estimation. When supported by truth data, also improve orientation
estimation. Give special attention to partially occluded gates, overlapping
gates, and high-angle views where only part of the aperture or frame is visible.
Speed matters. The backend should be suitable for flight, with observations
ideally produced at 20-30 Hz and at least around 10 Hz in extreme cases. Accuracy
improvements that make the model too slow for flight are not complete until they
are optimized or made conditional.

LegacyVision may be used as reference material for algorithms, diagnostics, or
review tooling, but it is not the only permitted source of improvements. New
approaches are acceptable when they are measurable against the reference dataset
and flight logs.

## Dataset Shape

A useful reference dataset should be large, representative, and contain
sequential clips, not only unrelated random frames. The vision stack uses
temporal tracking and multi-frame pose fusion, so flight-like sequences are
required to evaluate the behavior that matters in flight.

Prefer coverage split roughly as:

- Clean flight-like frames: normal approaches, side angles, and varied
  distances.
- Occlusion frames: obstacles crossing gates, gate-behind-gate, and partially
  hidden apertures.
- Overlapping or multi-gate frames: nearby gates, visually merged gates, and
  ambiguous apertures.
- Edge cases: edge-of-frame, close range, far range, steep yaw/pitch/roll, and
  low visible pixel count.
- Zero-gate or no-visible-gate frames for false-positive measurement.

## Required Truth

The highest-value fields are:

- `schema_version`, `dataset_id`, and `run_id`.
- Frame image path and stable frame id.
- Camera intrinsics: width, height, focal lengths, principal point, distortion if
  applicable, and FOV.
- Camera pose for every frame.
- Per-gate stable id or label.
- Per-gate world pose and camera-relative pose.
- Per-gate outer and inner projected 2D corners.
- Per-gate 3D corners, or enough pose and dimensions to derive them.
- Per-gate model dimensions.
- Per-gate visibility flags, visible pixel counts, `visibility_fraction`,
  `occlusion_fraction`, `occlusion_type`, `occluder_label`, and
  `geometric_visibility`.
- Per-gate masks with deterministic pixel values mapped by a legend.

When the course has known fixed gates, prefer listing all known gates on every
frame, including gates that are invisible, behind the camera, or outside the
frame. Use an empty `gates` list only for frames or scenes where no known gate
actor should be evaluated.

Obstacle masks or depth masks are not mandatory, but they are high value for
occlusion debugging. Gate-visible masks are enough to score visible-gate IoU and
partial-gate recall; obstacle/depth masks make it easier to tell whether a miss
is caused by true occlusion, segmentation failure, or geometry failure.

## Mask Convention

The current sample dataset uses single-channel cleaned masks with stencil values
mapped to gate labels, plus raw RGBA masks with equivalent grayscale values.
This is a good format. Keep a `mask_stencil_legend.json` next to each run and
avoid changing stencil values across frames within a dataset.

Evaluation code should treat cleaned single-channel mask values as authoritative
gate IDs, not colors. Do not infer gate identity from RGB appearance in the
source frame. Raw RGBA masks are useful for inspection, but cleaned `L` masks
should drive scoring when both are present.

Diagnostic fields such as `fallback_surface_sample_visible` should not override
the primary visibility contract. Prefer `visible_in_frame`, mask pixel count,
`visibility_fraction`, and `geometric_visibility` for scoring.

## Coordinate Handling

Unreal and flight code may use different axis conventions. Before scoring pose
accuracy, implement and verify one explicit transform from the dataset camera
and world frames into the flight conventions used by `VisionObservation`.

The sample dataset describes camera-relative axes as:

- `right`
- `up`
- `forward`

Do not assume this is identical to OpenCV camera coordinates or flight local NED.
Verify the transform with projected corners and a few visual overlays before
using pose-error metrics for tuning.

## Metrics To Produce

At minimum, score:

- Detection recall by gate and by visibility bucket.
- False positives per frame.
- Mask IoU by gate id.
- Projected corner error in pixels.
- Camera-frame position error.
- World/local position error after coordinate normalization.
- Depth error.
- Track identity stability across each sequence.
- Gate-map promotion latency when the map is part of the evaluation.
- Occlusion-specific recall and pose error.
- Runtime latency per frame, throughput in Hz, and worst-case latency for hard
  frames.

Keep synthetic metrics separate from real-flight metrics. Unreal data is a
controlled benchmark for geometry, occlusion, and tracking; it is not by itself
proof of real-flight robustness.

Accept a vision change only when it improves `deterministic_v3_2` on the
reference dataset without a meaningful regression on available real flight logs
or runtime performance.

## Performance Expectations

Treat runtime performance as a flight-readiness requirement. Evaluation reports
should include average, median, p90/p95, and max per-frame latency for both
`deterministic_v3` and `deterministic_v3_2`. If a change is slower, explain
whether the accuracy gain justifies the cost and whether the slow path can be
limited to hard cases such as occlusion, overlap, or high-angle views.

Prefer improvements that are bounded, deterministic, and incremental. Avoid
unbounded searches, heavyweight per-frame global optimization, or review-only
diagnostics in the live backend unless they are explicitly gated off for flight.

## Code Change Boundaries

Prefer experimental changes in:

- `Flight/src/sensing/vision/models/deterministic_v3_2`
- `LegacyVision/review_pipeline`
- evaluation-only UI or logging code

Feel free to adjust flight code anywhere under `Flight/` when it can improve how
the drone flies, as long as the change is documented and validated. This includes
vision service routing, mapping, planning, control, schemas, logging, and runtime
configuration when those changes are justified by evaluation results or simulator
testing. The user expects this work to happen on an isolated git branch, so broad
experimentation is acceptable. Still prefer `deterministic_v3_2` for vision
experiments unless a broader flight change is useful.

The end goal is racing performance, not only conservative gate detection. Once
vision improvements are measurable, it is acceptable to try higher-tempo flight
behavior, planner changes, path-following changes, and control gain adjustments.
Treat those as flight-performance experiments: validate in simulation or replay,
log the change, and compare whether the drone gets through gates faster and more
reliably.

If a change crosses out of `deterministic_v3_2`, document:

- what flight behavior is intended to change,
- why the reference dataset or flight logs justify it,
- what validation was run,
- what real-flight risk remains.

Do not silently alter shared runtime behavior. Make the reason visible in code,
logs, docs, or the final work summary.

## Simulation And Logs

The simulator may be running during future evaluation work. It can be used for
validation either through a direct UDP path or through the MAVLink code already
present under `Flight/`. Prefer reusing the existing flight interfaces when they
fit the test, because that exercises the same contracts used by runtime code.

Flight logs are available under:

```text
Logs/
```

Use `Logs/flight` for real or simulated flight-run replay and
`Logs/evaluation/runs` for generated comparison/evaluation artifacts. Dataset
improvements should be checked against both synthetic reference data and
available flight logs when practical.

## Review Artifacts

Evaluation runs should write structured output under:

```text
Logs/evaluation/runs
```

Useful artifacts include per-frame JSON, aggregate metrics, source-frame
overlays, mask/prediction/truth overlays, and side-by-side backend comparison
records. The existing viewer under `UI` and LegacyVision evaluation viewer are
appropriate places to add read-only review surfaces.
