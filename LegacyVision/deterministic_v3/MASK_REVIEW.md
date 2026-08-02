# Static mask review

The review command renders the active `void_detection` pipeline as independent
component layers for raw frames stored in `detection_v2/runs/*/vision_frames`.
The viewer shows all layers in a synchronized grid and labels each render with
its responsible Python function. Generated files are isolated under
`deterministic_v3/review_runs` and are replaced each time the command runs.

Use a Python interpreter with the Flight OpenCV and NumPy dependencies. On the
current development machine, `/usr/local/bin/python3.13` provides them.

Render a short frame slice while iterating:

```bash
cd /Users/trumanhale/0720-0802_et_drone_hone/ft-vision-detectionv2
PYTHONPATH=Flight/src /usr/local/bin/python3.13 \
  -m sensing.vision.models.deterministic_v3.src.void_detection \
  --run run-20260731T031521Z --start 0 --limit 200
```

Pass `--lut PATH` to review a different color mapping, repeat `--run` for
multiple runs, or use `--all` to render every run containing `vision_frames`.
Every CLI generation also appends the newest valid run under
`Flight/logs/runs`, unless a run with that ID is already selected. This keeps
new repository flight logs present in subsequent viewer generations.
The `Distance transform inside contours` layer uses a display gamma of `0.35`
by default so shallow distances remain bright and visible. Adjust it without
changing detector geometry:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 \
  -m sensing.vision.models.deterministic_v3.src.void_detection \
  --run run-20260731T031521Z --distance-gamma 0.2
```

Lower gamma is brighter; higher gamma emphasizes only the deepest regions.
The following `Local distance transform` layer instead divides each distance
by a fixed `9 px` radius and clips it there. This prevents a thick or distant
mask area from changing the evidence scale elsewhere. Change the radius with
`--local-distance-radius PIXELS`.
The next `Contour to local distance maxima` layer finds 3x3 local maxima in
the original Euclidean distance field. Within each connected mask component,
it maps progress from the contour line to the nearest maximum. Components are
processed independently so maxima from another mask area cannot affect it.
The `Inverse mask pixel density` layer directly after the final contour mask
uses a fixed 9 px circular neighborhood. Sparse and thin mask regions appear
bright, solid interiors appear dark, and pixels outside the mask remain black.
The immediately following `Local inverse mask pixel density` layer applies the
same calculation with a fixed 5 px circular neighborhood for a tighter view.
It applies a fixed gamma of 3.0 to emphasize the sparsest local mask regions.
The independent `Legacy inverse mask density` layer reproduces the older
`mask_neighbor_density` gamma-weighted inverse-support field: a radius-5 square window with an
edge-clipped denominator, `mask * (1 - density) ^ 1.5`, and its original heat
colors.
The viewer's `Legacy sweep` slider switches among nine precomputed variants:
radii 3, 5, and 7 px crossed with gamma 1.0, 1.5, and 3.0. Radius 5 px and
gamma 1.5 is selected initially.
Immediately after `Final contour input`, the `Calibrated final inverse density`
layer applies the documented mean-relative inverse-density and ridge-centroid
formula directly to the native-resolution contour mask. It does not crop,
tile, resize, or upscale the mask; radius 12 and ridge radius 10 are interpreted
as native input pixels.
The following five comparison layers scale radii independently for each
connected component. The square root of component pixel area is mapped from
20 to 90 equivalent pixels, then proportionally selects a radius between 2 and
each profile maximum: 20, 16, 11, 7, or 2. Ridge maxima preserve the calibrated
10:12 ratio as 17, 13, 9, 6, and 2. Components are evaluated independently.
Run `--help` for the complete CLI.

Serve the common models directory so the static paths resolve:

```bash
cd Flight/src/sensing/vision/models
python3 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/deterministic_v3/ui/frontend/>. Rerun the renderer
and refresh the page to inspect the latest detector output.

## DensityEvidence review tab

The independent full-resolution implementation is in
`src/legacy_inverse_density.py`. It renders five preprocessing stages followed
by the calibrated density and ridge-weighting fields described in
`INVERSE_DENSITY_COMPUTATION.md`:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 \
  -m sensing.vision.models.deterministic_v3.src.legacy_inverse_density \
  --run run-20260731T031521Z
```

The newest valid `Flight/logs/runs` entry is appended automatically. Open
<http://127.0.0.1:8765/deterministic_v3/ui/frontend/index.html> or select the
`DensityEvidence` tab in the viewer.

## TopologyDecision.topology_label review tab

Open <http://127.0.0.1:8765/topology.html> with the current UI backend to
filter the exact serialized values `standard`, `multi_void`, `c_shape`,
`unknown`, and `clipped`, or select `all`. Each component uses its serialized
`bbox_xywh` to aspect-fit the complete logged source crop into a fixed 250x250
view with empty letterbox space and a label-specific outer border. The tab does
not infer labels from route or rejection
fields; older dumps without `TopologyDecision.topology_label` report that the
review run must be regenerated.

Run and frame selection persists across tabs in the current browser session
when the destination tab supports the same run. Tabs without that run restore
their own last valid run and exact frame filename rather than reusing an index
against unrelated source data.

## StandardGateResult review tab

Open <http://127.0.0.1:8765/standard_gate.html> to review every serialized
`StandardGateResult` for the selected run, including accepted and rejected
component attempts. For faster run-wide inspection, each card renders exactly
two visual panels: the component-isolated `FrameObservation.closed_mask` and
the serialized `StandardGateResult.fitted_corners_uv` over its exact
`ComponentObservation.bbox_xywh` source crop. The source JPEG is requested only
when four fitted corners exist; the UI never recomputes geometry.

The tab uses explicit Previous/Next controls to render 200 records at a time.
It does not use infinite-scroll or per-card lazy loading; only the active batch
exists in the DOM and its visual evidence is loaded immediately.

The compact card retains `fit_confidence`, `high_confidence_threshold`,
`accepted`, and `rejection_reason`. Selected-profile `DensityEvidence` remains
losslessly serialized in review JSON and available in the primary
`DensityEvidence` tab, but the StandardGateResult tab does not request or render
those density layers.

## PnP evidence and final pose projection tab

Open <http://127.0.0.1:8765/pnp_scene.html> to review the run-level production
`GeometryFrameResult` JSON against its exact logged source frame. The backend
matches `frame_id` and `sim_time_ns`; the frontend consumes
`camera_calibration`, `gate_model`, `pnp_relative_pose_estimates`, and
`camera_pose_estimates` without deriving a pose from another field. It builds
the view projection from the serialized `camera_matrix`, builds the plane from
`object_points_m`, and applies each serialized
`rotation_vector_model_to_camera` and `position_camera_m` in the OpenCV optical
camera frame. Cyan shows the selected raw `PnPCandidateEstimate`; amber
optionally shows the secondary candidate; magenta shows the authoritative
post-regression `CameraPoseEstimate`. Yellow and green history layers retain
only selected raw candidates and final poses respectively. Rejected records
and their exact `rejection_reason` remain in the readout.

Three.js is pinned to `0.165.0` only on this page. The current runtime
calibration has zero `distortion_coefficients`. A future non-zero calibration
fails closed in the adapter until the viewer has a distortion-aware projection,
so the UI does not present an approximate overlay as exact runtime geometry.

For production computation without review layers, heat maps, or manifest work,
use `src/legacy_inverse_density_production.py`. It computes only the final
native-resolution field and omits the redundant second close:
Frames containing 500 or more foreground connected components are gated after
labeling and return an empty field without running the remaining operations.
The production path uses OpenCV box sums for the radius-12 density field and a
single three-channel box sum for the ridge mass, X moment, and Y moment. This
preserves the calibrated formula while changing floating-point accumulation
order relative to the review implementation.
After closing, retained components use fixed buckets based on maximum bounding-
box dimension. Compact components at 35 px or below use density/ridge radii
2/2; small components from 36 through 59 px use 4/3; medium components from 60
through 89 px use 7/6. Components at 90 px or above retain the global 20/16
field. The first three buckets use isolated square crops and component-local
mean normalization.

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 \
  -m sensing.vision.models.deterministic_v3.src.legacy_inverse_density_production \
  Flight/logs/runs/run-20260801T031401Z
```

Pass `--output PATH` only when float64 `.npy` fields should be written. Without
it, the command computes every field but performs no output encoding or writes.

To review straight-sided centerlines fitted between matching outer and inner
contours, pass `--centerline-review PATH` instead. This mode evaluates the full
continuous inverse-density field at 15 bounded offsets per side, retains a
small midpoint regularizer, and writes individual PNG crops, paged contact
sheets, and a JSON manifest:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 \
  -m sensing.vision.models.deterministic_v3.src.legacy_inverse_density_production \
  Flight/logs/runs/run-20260801T031401Z \
  --centerline-review production_samples/density-guided-centerlines
```

The magenta polygon is the unweighted outer/inner midpoint; yellow is the
density-guided result. Components without a valid inner contour are counted in
the manifest but cannot produce this centerline.

For the directly comparable density-only baseline, use
`--density-percentile-review PATH`. It keeps the same ring-eligible population
but renders only a one-pixel yellow quadrilateral fitted to the highest 10%
(P90) of positive field values; no inner, outer, or midpoint lines are drawn.
Output is separated into compact 2/2, small 4/3, medium 7/6, and large 20/16
density/ridge-radius folders so calibrated settings are never interleaved.

## Minimal P90 production checkpoint

`src/inverse_density_quadrilateral_production.py` is the clean production-only
checkpoint. Its `detect_density_quadrilaterals(image, lut)` API performs mask
preparation, isolated square component computation with one calibrated radius
profile, the optimized original inverse-density formula, and a convex P90
quadrilateral fit. It performs no rendering or file output.

The separate `ui/backend/validate_inverse_density_checkpoint.py` tool benchmarks a
run and creates the fixed five-gate report used to validate this checkpoint.
