# Pose Estimation And Instance Tracking Scaffold

This document records the minimal scaffold added for the final two systems in
the new 0721Vision pipeline: pose estimation and instance tracking.

The scaffold is intentionally not the final implementation. It exists to mark
the UI space, name the future artifacts, and provide source-code anchors for the
next implementation pass.

## Current Scaffold

The new system owns the full-width UI area above the legacy review workspace.
The old UI remains below it and keeps its current controls, canvases, and build
buttons.

Current new-system row sequence:

```text
Source Frame
-> Mask Frame
-> BBox Frame
-> Clipping Edges
-> Contours
-> Pose Estimation
-> Instance Tracking
```

The first five modules are active or reserved for already-started systems. The
last two are high-level placeholders:

```text
pipelinePoseEstimationModule
  title: Pose Estimation
  output placeholder: 3d_pose_fit_json

pipelineInstanceTrackingModule
  title: Instance Tracking
  output placeholder: instance_mapping_json
```

The large output scaffold under the per-frame modules reserves final review
space:

```text
poseInstanceVolumeScaffoldPane
  title: 3D Pose + Instance Volume
  purpose: visual volume for fitted pose and tracked instance placement

instanceTrackingJsonScaffoldPane
  title: Instance Tracking JSON
  purpose: compact frame-level mapping output for review
```

No new UI placeholder is wired to runtime data yet.

## Source Anchors

Two inert source folders were added under `src/`:

```text
src/pose_estimation/
  __init__.py
  README.md
  pose_fit.py

src/instance_tracking/
  __init__.py
  README.md
  instance_mapping.py
```

Both scripts only support `--describe`. They print JSON contracts and exit.
They do not import legacy tools, solve geometry, associate tracks, write files,
or call APIs.

Pose scaffold contract:

```bash
python src/pose_estimation/pose_fit.py --describe
```

Future pose input boundary:

```text
assets/bbox_contours/<run_key>/contours_manifest.json
assets/bbox_clipping/<run_key>/clipping_manifest.json
```

Future pose output:

```text
assets/pose_estimation/<run_key>/3d_pose_fit.json
```

Instance scaffold contract:

```bash
python src/instance_tracking/instance_mapping.py --describe
```

Future instance input boundary:

```text
assets/mask_bboxes_maskbits/<run_key>/bbox_manifest.json
assets/pose_estimation/<run_key>/3d_pose_fit.json
```

Future instance output:

```text
assets/instance_tracking/<run_key>/instance_mapping.json
```

## Implementation Plan For The Next Pass

The next pass should keep the same minimalist boundary and implement only the
new deterministic path. It should not reuse the old review-build roots as final
dependencies.

Recommended order:

1. Implement `src/pose_estimation/pose_fit.py`.
2. Read bbox-local contour geometry from `assets/bbox_contours`.
3. Read clipping validity from `assets/bbox_clipping`.
4. Fit the known square target into calibrated camera space.
5. Write compact per-frame pose records to `3d_pose_fit.json`.
6. Implement `src/instance_tracking/instance_mapping.py`.
7. Read maskbits bboxes and pose fit records in frame order.
8. Assign stable instance IDs across frames.
9. Write compact per-frame mappings to `instance_mapping.json`.
10. Wire the new UI placeholders to those two artifacts.

The pose implementation should preserve the useful math from the current
legacy pose builder, including quad refinement, edge support scoring, clipping
validity, and calibrated square pose fitting. The final source boundary should
come from the maskbits-derived artifacts, not from legacy color-rule manifests.

The tracking implementation should preserve the useful association signals from
the current legacy instance builder, including bbox center, IoU, size, layer
mix, 3D distance, and orientation delta. UI-only settings such as overlay
visibility and readout visibility should not be part of the final artifact
signature.

## Non-Goals For This Scaffold

This scaffold did not:

```text
implement OpenCV pose estimation
implement frame-to-frame association
generate pose or tracking artifacts
add server endpoints
remove old UI controls
change legacy build scripts
change color layer behavior
```

## Review Checks Used

The scaffold was verified with:

```bash
node --check app.js
git diff --check -- index.html styles.css src/pose_estimation src/instance_tracking
python src/pose_estimation/pose_fit.py --describe
python src/instance_tracking/instance_mapping.py --describe
```

The legacy review IDs remain present, including:

```text
squarePoseCanvas
instance3dCanvas
instanceReadout
squarePoseBuildButton
instanceBuildButton
classList
```
