# WORK IN PROGRESS: Camera-Pose 3-D World Replay

> **Status:** Future-work architecture and discovery record. This document does
> not authorize runtime, schema, or UI implementation. Ambiguous coordinate,
> synchronization, camera-extrinsic, and persistence decisions must be raised to
> the user and confirmed before implementation.

## Objective

Extend camera-relative `CameraPoseEstimate` review with an optional world-space
view that lets the reviewer orbit, pan, and zoom around the reconstructed scene.
The view should show:

- The selected historic image/frame and its camera pose.
- The drone and camera at the aligned odometry pose in LOCAL_NED.
- Each accepted PnP gate at its estimated world position and orientation.
- A visible camera-to-gate range segment so the PnP depth can be inspected.
- The odometry trail up to the selected frame.
- Logged test and planned flight paths when present.

“3-D reversal” is treated here as a view reversal, not an inversion of PnP: the
existing view remains locked to the camera and projects 3-D results onto the
image, while the new view places the camera and gate in a shared world and lets
the reviewer move an independent inspection camera around them.

## Recommended product boundary

Keep two synchronized modes:

1. **Camera projection:** the existing fixed camera view remains the direct
   check of `GeometryFrameResult.camera_calibration`, `gate_model`, and accepted
   `camera_pose_estimates` against the source image.
2. **World replay:** a free-orbit LOCAL_NED view shows the aligned vehicle,
   camera, PnP gate, depth ray, odometry history, and logged paths.

The same run and zero-based frame selection must drive both modes. Switching
modes must not change the selected frame. Camera projection remains the best
place to diagnose quadrilateral/PnP errors; world replay is the best place to
diagnose depth, coordinate transforms, odometry alignment, and temporal drift.

This should be implemented as read-only review infrastructure. It must not be
imported by the flight runtime, mutate historic run data, change the production
PnP result, or synthesize missing runtime fields.

## Existing implementation that should be reused

`Flight/src/ui` already contains most of the generic world viewer:

- `static/index.html` has a `3D Map` tab and its layer controls.
- `static/app.js::initMap3d()` creates a Three.js scene, perspective camera,
  `OrbitControls`, grid, lighting, and LOCAL_NED axes.
- `renderMap3d()` composes drone, trails, test path, planned path, gate map, and
  observation gates.
- `addDrone()` draws a BODY_FRD vehicle marker and forward/right/down axes.
- `addTrail()` and `addPathLayer()` draw telemetry and path polylines.
- `addGateFrame()` draws oriented outer and inner gate frames.
- `nedToThree()` maps LOCAL_NED `[north, east, down]` to Three.js
  `[x, y, z] = [north, -down, east]`.
- `data.py::frame_payload()` already produces a frame-aligned scene payload.
- `data.py::scene_payload()` already identifies the world as LOCAL_NED and the
  vehicle body as FRD.
- `data.py::nearest_cycle_for_frame()` already supports frame-ID, cycle, and
  timestamp matching.

The deterministic V3 review UI already contains the complementary evidence
viewer:

- `pnp_scene.html` is the current fixed camera-projection page.
- `pnp_scene_adapter.js` only renders accepted, schema-compatible
  `CameraPoseEstimate` records.
- `pnp_scene_app.js` consumes the serialized camera matrix, distortion, gate
  object points, rotation vector, and translation vector.
- The current runtime aggregate is stored under
  `deterministic_v3/ui/review_runs/<run-id>/standard-gate-pnp-runtime.json`.

The recommended implementation is therefore an integration and validation
effort. Do not create a second free-orbit renderer or duplicate coordinate math
inside the deterministic evidence viewer. Extract or reuse the current
`Flight/src/ui` scene primitives, then provide a small adapter for
`GeometryFrameResult`.

## Current evidence snapshot

The latest reviewed pair is:

- Historic run: `Flight/logs/run-20260801T031401Z`
- Runtime PnP aggregate:
  `Flight/src/sensing/vision/models/deterministic_v3/ui/review_runs/run-20260801T031401Z/standard-gate-pnp-runtime.json`

At the time of this review:

- The historic run contains 444 `frames.jsonl` frame records and 1,217 cycle
  records.
- The PnP aggregate contains 444 runtime-frame records and reports zero frame
  identity errors.
- It contains 514 `CameraPoseEstimate` results: 507 accepted and 7 rejected.
- The run contains position and attitude estimates that can be associated with
  all 444 frames.
- Association source is `vision_frame_id` for 336 frames and timestamp fallback
  for 108 frames.
- Across the selected associations, the mean reported alignment error is about
  314.36 ms, p95 is about 2,245.46 ms, and the maximum is about 3,016.27 ms.
- The run contains no usable simulator-truth position series, so the displayed
  trail is estimated odometry and must not be labeled ground truth.
- Logged initialization records contain a 20-degree camera tilt and a zero
  BODY_FRD camera translation for this run. These values must be read from run
  metadata rather than copied into the UI as universal constants.
- The test-path record already contains 200 `points_local_ned_m` values. Its
  source is `straight_line`; it is not a sinusoidal path.

The synchronization error tail is material. A visually plausible gate can be
placed incorrectly in world space if a valid PnP result is composed with a
vehicle state from a different moment. World rendering must therefore expose
alignment provenance and enforce an approved maximum error.

## Authoritative input records

### Image/frame identity

Use the run's `frames.jsonl` record as the source image identity:

- `frame_id`
- `sim_time_ns`
- `path`
- optional `cycle`

Join the PnP aggregate by **both** `GeometryFrameResult.frame_id` and
`GeometryFrameResult.sim_time_ns`. A run name or array index is insufficient.
If either identity differs, mark the result unavailable for world placement.

### Camera-relative PnP

Consume the upstream JSON field names without UI aliases:

- `GeometryFrameResult.camera_calibration`
- `GeometryFrameResult.gate_model`
- `GeometryFrameResult.camera_pose_estimates`
- `CameraPoseEstimate.rotation_vector_model_to_camera`
- `CameraPoseEstimate.position_camera_m`
- `CameraPoseEstimate.accepted`
- `CameraPoseEstimate.reprojection_rmse_px`
- `CameraPoseEstimate.position_confidence`
- `CameraPoseEstimate.orientation_confidence`
- `CameraPoseEstimate.rejection_reason`

Only `accepted == true` poses with finite vectors and matching calibration/model
IDs may be placed. Rejected records remain inspectable as JSON but are not
rendered as gates.

Use `gate_model.object_points_m` to build the gate plane. Do not replace the
serialized model with a UI gate-size slider or a legacy 2.7 m outer boundary.

### Vehicle state and odometry

For each selected frame, obtain the aligned cycle's canonical vehicle state:

- `position_local_ned_m`
- `attitude_quaternion` in `[w, x, y, z]` order
- `sim_time_ns`
- frame/cycle association source
- alignment error in milliseconds

The odometry trail is the ordered sequence of these LOCAL_NED positions. The
default trail must end at the selected frame/cycle. A complete-run trajectory
may be shown only as a separate, faint “full run” context layer; it must not be
confused with elapsed history.

### Camera mount

Read per-run mount values from
`run.metadata.initialization_constants` when present:

- `VIO_CAMERA_TILT_DEG`
- `VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M`

The latest run records 20 degrees and `[0, 0, 0]`, respectively. The naming is
VIO-specific, so these values must be explicitly approved as the perception
camera extrinsics before the resulting world gate is described as exact.
Missing extrinsics must disable gate world placement rather than silently use
20 degrees or a zero offset.

### Flight paths

Prefer serialized path points:

- `test_path.points_local_ned_m`
- `planned_path.points_local_ned_m`
- or `origin_local_ned_m + points_relative_ned_m` when absolute points are not
  serialized.

Do not regenerate a logged path from sine/cosine settings when exact logged
LOCAL_NED points are available. `PathManager.build_test_path()` does define a
sinusoidal path generator, but the latest run's test path is a serialized
straight line. If “CI script” or “sine of the flightpath” refers to a different
file or desired generated path, that source must be identified before work
begins. No repository CI workflow currently defines this projection; CI setup
is still listed as backlog work.

## Coordinate contract

The relevant frames are:

| Frame | Stored axes |
| --- | --- |
| OpenCV PnP camera (`CV`) | +x right, +y down, +z forward |
| Camera FRD (`C`) | +x forward, +y right, +z down |
| Vehicle body (`B`) | +x forward, +y right, +z down |
| Run local world (`N`) | +x north, +y east, +z down |
| Three.js display | +x north, +y up, +z east |

Use a proper right-handed intermediate Camera FRD frame. For a PnP translation
`p_g_cv = [x_right, y_down, z_forward]`:

```text
p_g_c = [z_forward, x_right, y_down]
```

This is a proper axis permutation and avoids trying to encode the Flight
camera-optical y-axis reflection as a quaternion.

Define:

- `p_b_n`: vehicle body origin in LOCAL_NED.
- `R_b_n`: BODY_FRD-to-LOCAL_NED rotation from the vehicle quaternion.
- `t_c_b`: camera origin expressed in BODY_FRD.
- `R_c_b`: Camera-FRD-to-BODY_FRD mount rotation.
- `C_cv_c`: OpenCV-camera-to-Camera-FRD axis permutation.
- `t_g_cv`: PnP `position_camera_m`.
- `R_g_cv`: Rodrigues rotation from gate model to OpenCV camera.

The camera and gate centers are:

```text
p_c_n = p_b_n + R_b_n * t_c_b

p_g_n = p_b_n
      + R_b_n * (t_c_b + R_c_b * C_cv_c * t_g_cv)
```

The candidate gate orientation is:

```text
R_g_n = R_b_n * R_c_b * C_cv_c * R_g_cv
```

Finally, display every LOCAL_NED point with:

```text
[x_three, y_three, z_three] = [north, -down, east]
```

All matrix direction names should be encoded in implementation identifiers.
Generic names such as `rotation` or `cameraTransform` are too easy to invert.
The backend should perform and test this transform; the browser should receive
auditable world-space results rather than maintain a second independent copy of
flight geometry math.

### Orientation approval boundary

Position and orientation do not have the same validation burden. The existing
legacy adapter explicitly flips OpenCV y for position and notes that Rodrigues
orientation handedness still requires validation. The project backlog also
requires fixtures for camera optical, BODY_FRD, LOCAL_NED, camera tilt, and gate
through-axis conventions.

Before gate orientation is presented as authoritative, fixtures must confirm:

- The model's upper-left/upper-right/lower-right/lower-left order.
- The sign of the gate-model normal.
- The direction represented by the Rodrigues transform.
- Quaternion ordering and BODY_FRD-to-LOCAL_NED direction.
- Positive camera tilt behavior.
- The planar IPPE alternative-pose selection.
- The world result reprojects to the same image quadrilateral.

Until those fixtures pass, the UI may render the gate center and a neutral
plane with an “orientation provisional” label. It must not hide low
`orientation_confidence` or smooth an ambiguous planar orientation into an
apparently stable result.

## Synchronization policy

Use this priority order:

1. Exact `frame_id` association, with timestamp error still recorded.
2. Exact logged frame/cycle association when present.
3. Nearest adjusted timestamp using logged TIMESYNC evidence.
4. Unavailable if no association satisfies the approved tolerance.

The adapter must return:

```json
{
  "match_source": "vision_frame_id | frame_cycle | timestamp | none",
  "frame_sim_time_ns": 0,
  "vehicle_state_sim_time_ns": 0,
  "alignment_error_ms": 0.0,
  "within_tolerance": false
}
```

The maximum acceptable alignment error is deliberately not selected in this
document. It must be chosen with the user after inspecting motion rate and
expected spatial error. The UI should color the synchronization state and omit
the world gate when the threshold fails. It may still show the camera-relative
PnP result in the fixed projection mode.

Interpolation between bracketing vehicle states can be a later improvement:
linear interpolation for position and quaternion SLERP for attitude. The first
implementation should use an auditable selected sample and report its error
rather than introduce unreviewed interpolation behavior.

## Proposed UI behavior

Add a mode control beside the current CameraPoseEstimate projection title:

```text
[ Camera projection ] [ World 3-D ]
```

World 3-D should provide only laboratory controls:

- Orbit: primary drag.
- Pan: secondary drag.
- Zoom: wheel.
- Fit selected evidence.
- Toggle odometry history.
- Toggle full-run trajectory context.
- Toggle test path and planned path.
- Toggle drone body axes and camera frustum.
- Toggle accepted gate plane and camera-to-gate depth segment.
- Toggle prior gate estimates only after an association identity exists.

The selected-frame readout should include frame ID, frame timestamp, vehicle
state timestamp, match source, alignment error, PnP depth, reprojection RMSE,
position confidence, and orientation confidence.

Recommended visual semantics:

- Estimated odometry history: solid neutral line through the selected time.
- Full future/whole-run context: thin, low-opacity line with an explicit label.
- Simulator truth: separate color only when actually serialized.
- Drone: BODY_FRD marker at aligned pose.
- Camera: small frustum at the camera extrinsic pose.
- Gate: runtime model dimensions, one color per `component_id` within a frame.
- Depth: line from camera center to `position_camera_m` transformed into world.
- Rejected/unsynchronized result: no world gate; retain a visible status reason.

Do not accumulate frame-local component IDs across time as if they were stable
gate IDs. The first version should show only the selected frame's PnP gates.
Historical gate persistence requires a separate association contract.

## Proposed read-only adapter

Add a UI-only adapter under `Flight/src/ui`, for example:

```text
Flight/src/ui/
  spatial_replay.py                 # joins run, vehicle state, and PnP result
  static/
    spatial_replay.js               # world-view controller
    spatial_scene.js                # Three.js scene primitives
    coordinate_display.js           # LOCAL_NED-to-Three display mapping only
```

The backend adapter should own:

- Direct and nested run discovery.
- Exact image/PnP identity joins.
- Vehicle-state association and synchronization diagnostics.
- Camera extrinsic lookup from run metadata.
- OpenCV-camera-to-LOCAL_NED transformation after fixture approval.
- Construction of a small, immutable world-review payload.

The frontend should own only:

- View state and mode switching.
- Three.js scene lifecycle.
- OrbitControls.
- Layer visibility.
- Rendering already-derived LOCAL_NED points and orientations.

Derived UI values must be kept under an explicit `ui_projection` object so
they cannot be mistaken for upstream runtime JSON. Original runtime objects and
field names should remain available unchanged for inspection.

An illustrative per-frame response is:

```json
{
  "frame": {
    "frame_id": 0,
    "sim_time_ns": 0,
    "image_url": "..."
  },
  "sync": {
    "match_source": "vision_frame_id",
    "vehicle_state_sim_time_ns": 0,
    "alignment_error_ms": 0.0,
    "within_tolerance": true
  },
  "vehicle_state": {
    "position_local_ned_m": [0, 0, 0],
    "attitude_quaternion": [1, 0, 0, 0]
  },
  "runtime_result": {
    "camera_calibration": {},
    "gate_model": {},
    "camera_pose_estimates": []
  },
  "ui_projection": {
    "camera_position_local_ned_m": [0, 0, 0],
    "accepted_gate_poses_local_ned": []
  }
}
```

Run-level trajectory and path data should be fetched once and cached in memory
after run selection. Per-frame responses should remain small. The browser must
not repeatedly fetch or scan the full 444-frame PnP aggregate while the slider
moves.

## Run discovery prerequisite

`Flight/src/ui/data.py::discover_runs()` currently scans:

- `Flight/logs/runs/run-*`
- `Flight/data/live_runs/run-*`

The reviewed historic runs are currently located directly under
`Flight/logs/run-*`. The world replay cannot discover them through the current
function. Phase 1 must add explicit direct-run discovery or accept a configured
run root. It must preserve read-only path validation and avoid silently scanning
unrelated worktrees.

## Implementation phases

### Phase 0: approve the geometry boundary

1. Confirm the perception-camera mount extrinsic source.
2. Confirm the OpenCV-to-Camera-FRD-to-BODY_FRD transform with fixtures.
3. Confirm gate model normal/through-axis convention.
4. Choose a frame-to-vehicle-state alignment tolerance.
5. Confirm whether estimated odometry is the desired default when truth is
   absent.

No world gate should be labeled valid until these items are resolved.

### Phase 1: make historic runs addressable

1. Extend UI run discovery to explicit `Flight/logs/run-*` inputs.
2. Load frame manifests, cycles/telemetry, paths, and run initialization data.
3. Load the matching deterministic V3 PnP aggregate by explicit run ID.
4. Require exact `frame_id` and `sim_time_ns` joins.
5. Produce sync diagnostics and expose unavailable reasons.

### Phase 2: replay odometry without PnP gates

1. Reuse the current Three.js LOCAL_NED grid, axes, OrbitControls, and drone.
2. Draw the odometry prefix through the selected frame.
3. Draw the current drone pose and optional camera frustum.
4. Draw serialized test/planned paths.
5. Preserve frame selection between camera and world views.

This phase isolates historic-run and odometry correctness from PnP geometry.

### Phase 3: place accepted PnP gates

1. Transform accepted `position_camera_m` values into LOCAL_NED in the backend.
2. Draw the exact serialized `gate_model.object_points_m` plane.
3. Draw the camera-to-gate depth segment and numeric range.
4. Add provisional orientation only after coordinate fixtures pass.
5. Suppress placement when identity, schema, extrinsic, or sync validation
   fails.

### Phase 4: diagnostic comparisons

1. Optionally compare selected-frame PnP gates with logged gate-map records.
2. Add simulator truth only for runs that serialize it.
3. Add odometry interpolation after nearest-sample behavior is established.
4. Add historical gate tracks only after a stable gate-association ID exists.

### Phase 5: performance hardening

1. Cache the parsed run manifest, trajectory, paths, and PnP frame index after
   run selection.
2. Return only the selected frame's image and spatial payload while scrubbing.
3. Reuse Three.js objects and update transforms instead of rebuilding the full
   scene on every frame.
4. Vendor or pin Three.js locally if offline/reproducible review is required;
   the current viewer loads Three.js 0.165.0 from a CDN.

## Validation and tests

The following tests are required before accepting the feature:

### Identity and schema

- Every rendered result matches run ID, `frame_id`, and `sim_time_ns`.
- Accepted and rejected PnP counts match the aggregate.
- Rejected or malformed poses never become world gates.
- The serialized camera calibration and gate model IDs are preserved.

### Coordinate fixtures

- Level body, zero camera translation, zero tilt, centered gate forward.
- Positive OpenCV x moves gate body-right and local direction appropriately.
- Positive OpenCV y moves gate body-down before local rotation.
- Positive depth moves gate camera-forward.
- Positive camera tilt moves the forward ray in the expected local direction.
- Nonzero BODY_FRD camera translation moves the camera and gate once, not
  twice.
- Quaternion `[w, x, y, z]` direction matches BODY_FRD-to-LOCAL_NED.
- All rotation matrices are orthonormal with determinant approximately +1.

### Round-trip projection

For a selected accepted pose:

1. Construct the world gate from the world transform.
2. Transform its corners back into the selected camera.
3. Project them with the serialized camera matrix and distortion.
4. Confirm they reproduce the runtime quadrilateral within the reported PnP
   reprojection error, allowing only a small numerical tolerance.

This is the strongest end-to-end check against axis swaps and transform
inversions.

### Synchronization and replay

- Exact frame association wins over fallback association.
- Alignment error and source are always visible.
- Results beyond the approved tolerance are not placed.
- The default trail ends at the selected frame and never includes future
  samples.
- Frame zero, last frame, and tab switching preserve the same selection.
- Missing paths, mount metadata, truth, or PnP data produce explicit empty
  states rather than defaults.

## Acceptance criteria

The first useful implementation is complete when a reviewer can:

1. Select a historic run and frame once.
2. Switch between fixed camera projection and free-orbit world replay without
   losing the frame.
3. See the aligned drone pose and only the elapsed odometry trail.
4. See each accepted PnP gate at an auditable LOCAL_NED pose and inspect its
   camera-relative depth.
5. See logged test/planned paths directly from their LOCAL_NED points.
6. See exact identity, synchronization, extrinsic, confidence, and rejection
   provenance.
7. Distinguish estimated odometry, optional simulator truth, and planned path.
8. Receive no plausible-looking 3-D gate when required evidence is missing or
   outside tolerance.

## Decisions requiring user confirmation

Before implementation, confirm:

1. Whether `VIO_CAMERA_TILT_DEG` and
   `VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M` are authoritative for the
   deterministic perception camera.
2. The maximum permitted frame-to-vehicle-state alignment error.
3. Whether the first version should render orientation provisionally or center
   and depth only until all orientation fixtures pass.
4. Whether “CI script” meant `Flight/src/ui/static/app.js`, a future automated
   CI workflow, or another script.
5. Whether “sine of the flightpath” refers to
   `PathManager.build_test_path()` or simply the serialized flight-path line.
6. Whether future gate history should remain frame-local or be fused through a
   separately approved gate-association identity.

The recommended defaults are: use serialized LOCAL_NED path points, show
estimated odometry with an explicit label, show only the selected frame's gates,
and withhold authoritative world orientation until the coordinate fixtures are
approved.
