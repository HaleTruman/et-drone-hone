# WORK IN PROGRESS: Deterministic V3 Geometry Technical Process

> **Status:** Working draft for technical consolidation and user review.
>
> This document is not a replacement for the authoritative `../AGENTS.md`.
> Where this document, historical notes, or implementation details disagree with
> `../AGENTS.md`, the authoritative contract wins. Any ambiguity, conflicting
> evidence, or unspecified calibration must be raised to the user for
> confirmation before it is treated as an approved production decision.

## 1. Purpose and evidence boundary

This document consolidates the process and calibration evidence currently
recorded by Markdown files under `src/` and its descendants. It describes the
shared processing path, topology routing, and the three supported instance
procedures:

1. Standard single-aperture gate.
2. C-shape gate with one unobserved side.
3. Multi-gate component containing exactly two apertures.

The reviewed Markdown sources are:

- `../AGENTS.md`: authoritative geometry mission and contract.
- `schema_draft.md`: non-authoritative implementation and process notes.
- `geometry/agents.md`: deprecation pointer back to the authoritative file; it
  contains no active technical requirements.

This is an evidence distillation, not approval of every historical setting.
Settings explicitly documented by the current notes are recorded as proposed
defaults. Missing or contradictory settings are collected in Section 9.

## 2. Required end-to-end contract

```text
Frame JPEG
  -> decode
  -> LUT mask
  -> initial connected-component gate
  -> size filtering and morphology
  -> connected-component isolation
  -> topology analysis and stability check
  -> topology route
  -> topology-aware density-profile selection
  -> calibrated density evidence
  -> specialized fitter
  -> common quadrilateral normalization and validation
  -> camera-relative PnP
  -> optional camera-to-NED transformation
  -> VisionObservation
```

The ingress `frame_id` and `sim_time_ns` remain authoritative throughout the
pipeline. Geometry must not create replacement frame identities. Vehicle state
is used only for the final camera-to-NED conversion. Camera-relative geometry
may be calculated without vehicle state, but an absolute local-NED position must
not be fabricated.

All rejection paths must produce a machine-readable reason. Rejected instances
are not published as gates.

## 3. Shared processing and configuration

### 3.1 Configuration policy

Shared thresholds and calibration values should be explicit configuration,
versioned with the preprocessing or calibration version, and recorded in trace
output. They must not be hidden in UI or experimental review code.

The following values are the documented starting defaults:

| Setting | Documented default | Scope |
|---|---:|---|
| Initial component-count gate | `0` (disabled); positive values reject at `>= value` | Frame preprocessing |
| Minimum connected-component area | `100 px` | Size filtering |
| Morphological close | `5 x 5` square kernel | Shared mask preparation |
| Connectivity | Not stated | Requires confirmation |
| Significant aperture area | `>= 12 px` contour area | Topology notes |
| Inverse-density relative cap | `2.0` | Density normalization |
| Inverse gamma | `2.10` | Inverse-density field |
| Ridge gamma | `3.00` | Ridge multiplier |
| Ignore frame-edge-clipped for density | `true` | Density eligibility |
| Primary high-density evidence | `P90` | All documented fitters |
| Broader support evidence | `P70` | Multi-gate gap support |
| Additional cached evidence | `P80` | Shared bank; consumer unresolved |

The component-count gate and minimum-area filter are required to be
configurable in this repository. `maximum_input_components = 0` explicitly
disables the frame-level count gate for the current baseline; it does not mean
reject every frame. A positive value restores rejection at `>=` that value.
The actual values used must be traceable for each frame.

`ignore_frame_edge_clipped_for_density = true` is an independent component-level
density policy. It retains the component, bounding box, contours, topology
decision, and rejection evidence, but does not construct any `DensityEvidence`
for that component. Other components in the same frame remain eligible.
`DensityBankConfiguration` records this toggle and the complete ordered profile
definitions in the runtime schema, including frames that produce zero density
records.

### 3.2 Shared mask preparation

1. Decode the JPEG once.
2. Apply the calibrated LUT to create the base binary mask.
3. Run the initial connected-component calculation.
4. When the component-count gate is positive, reject the frame when it is
   reached; bypass this gate when it is zero.
5. Remove components below the configured minimum pixel area.
6. Apply the configured morphological close.
7. Run final connected-component labeling and statistics.
8. Isolate each connected parent component in component-local coordinates.
9. Preserve its frame-space bounding box and origin.

The shared observation should retain the base, size-filtered, and closed masks
so topology changes introduced by filtering or morphology remain auditable.

### 3.3 Shared component evidence

Compute shared component evidence once and make it available by reference:

- Component identity, label, bounding box, origin, area, and dimensions.
- Frame-edge contact.
- Raw-mask and closed-mask views.
- Raw and closed contour hierarchy.
- Raw and closed significant-hole counts, contours, and areas.
- Raw and simplified outer contours.
- Fill ratio and solidity.
- Distance transform.
- Density-bank handle and selected calibration version.

Every post-close connected component has its own frame-space `bbox_xywh`,
`image_origin_uv`, square `analysis_shape`, and `component_id`. Separate nearby
components therefore retain independent density evidence and profiles. Their
frame-space bounding boxes and origins allow a later association procedure to
group geometrically related components without blending their density fields.
That association procedure is not implemented yet and must remain downstream
of shared component construction.

Frame-edge-clipped components must be rejected under the authoritative contract.
They must not be forced through a complete-gate fitter.

### 3.4 Density-bank computation

The density bank is shared computation. It should compute named, cached variants
once per component and allow the topology-specific profile selector to choose
among them without recomputing the field.

The documented inverse-density procedure is:

1. Compute foreground density in a square neighborhood with the selected
   density radius.
2. Normalize foreground density relative to the component foreground mean and
   the relative cap of `2.0`.
3. Apply inverse gamma `2.10`.
4. Compute the local inverse-density weighted centroid using the selected ridge
   radius.
5. Convert distance from that local centroid into a ridge response.
6. Apply ridge gamma `3.00`.
7. Multiply inverse density by the ridge response to obtain the final field.
8. Cache positive values, coordinates, weights, and P70/P80/P90 thresholds and
   masks.

The current ten-layer bank uses eligible, non-frame-clipped post-close
connected-component foreground area (`ComponentObservation.area_px`, the
retained red-pixel count) as its provisional recommendation factor:

| Profile | Maximum component area | Density radius | Ridge radius |
|---|---:|---:|---:|
| `scale_01` | `146 px` | `2 px` | `2 px` |
| `scale_02` | `190 px` | `3 px` | `2 px` |
| `scale_03` | `242 px` | `4 px` | `3 px` |
| `scale_04` | `260 px` | `5 px` | `4 px` |
| `scale_05` | `372 px` | `6 px` | `5 px` |
| `scale_06` | `827 px` | `7 px` | `6 px` |
| `scale_07` | `1,370 px` | `9 px` | `7 px` |
| `scale_08` | `3,113 px` | `12 px` | `10 px` |
| `scale_09` | `4,671 px` | `16 px` | `13 px` |
| `scale_10` | Unbounded | `20 px` | `16 px` |

All variants remain available to every eligible component. The ordinary
recommendation chooses the first profile whose maximum area contains the
component. C-shape and multi-gate selectors may choose a different named
variant, but they must record it. The radii and gamma values are unchanged from
the prior ten-layer review bank. This calibration changes only the provisional
area-to-profile recommendation; it does not establish geometric accuracy for
each radius or supersede topology-specific profile selection. The lookup is
implemented by `DensityBank.recommended_standard_profile()`; the topology-first
pipeline scaffold does not yet connect it to a live standard-gate fitter.

### 3.5 Aggregate non-clipped red-pixel calibration — 2026-08-01

This first implementation uses every raw historic run currently available in
`Flight/logs`: `run-20260731T093159Z`, `run-20260731T093616Z`, and
`run-20260731T093732Z`. The analysis processed all `1,715` frames with the
current LUT, `100 px` minimum-area filter, `5 x 5` close, and 8-connectivity.
No frame was component-count gated. Red-pixel count is the exact post-close
`ComponentObservation.area_px`; components touching any frame edge are
retained as evidence but excluded from this recommendation calibration.

| Run | Frames | Non-clipped | Clipped | Non-clipped mean | P10 | Median | P90 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run-20260731T093159Z` | `332` | `735` | `121` | `2,409.507` | `241` | `824` | `5,408` | `17,745` |
| `run-20260731T093616Z` | `692` | `1,537` | `393` | `1,616.012` | `137` | `317` | `4,485` | `16,676` |
| `run-20260731T093732Z` | `691` | `1,580` | `466` | `2,233.204` | `141` | `299` | `4,646` | `72,860` |

The aggregate eligible population contains `3,852` non-clipped components;
the excluded clipped population contains `980`. Aggregate non-clipped mean
area is `2,020.577 px`. Because area is a discrete integer, finite boundaries
use `numpy.percentile(..., method="higher")`, making every inclusive upper
bound an observed component area:

| Percentile | Area boundary |
|---|---:|
| P10 | `146 px` |
| P20 | `190 px` |
| P30 | `242 px` |
| P40 | `260 px` |
| P50 | `372 px` |
| P60 | `827 px` |
| P70 | `1,370 px` |
| P80 | `3,113 px` |
| P90 | `4,671 px` |

Aggregate minimum and maximum are `100 px` and `72,860 px`. The maximum belongs
to a continuous large-component sequence in `run-20260731T093732Z`; empirical
deciles keep that upper tail in the unbounded final profile without allowing it
to distort the first nine boundaries.

Applying the superseded single-run clipped boundaries to this eligible
population would assign `2,035`, `778`, `1,012`, `6`, `4`, `6`, `5`, `5`, `1`,
and `0` components to scales 01 through 10. That severe imbalance is why the
clipped calibration is no longer used for ordinary non-clipped recommendations.

| Profile | Component count | Median maximum bbox dimension |
|---|---:|---:|
| `scale_01` | `394` | `17 px` |
| `scale_02` | `380` | `19 px` |
| `scale_03` | `391` | `20 px` |
| `scale_04` | `381` | `19 px` |
| `scale_05` | `382` | `26 px` |
| `scale_06` | `388` | `41 px` |
| `scale_07` | `381` | `45 px` |
| `scale_08` | `385` | `63 px` |
| `scale_09` | `385` | `81 px` |
| `scale_10` | `385` | `128 px` |

The near-even counts confirm that all ten recommendation levels are exercised
by the actual non-clipped population. The non-monotonic median bounding-box
dimension between `scale_03` and `scale_04` also confirms that foreground area
is only a recommendation index, not a complete substitute for topology and
shape evidence.

## 4. Topology analysis and fitter dispatch

Topology routing is a shared production responsibility and must happen before
specialized fitting. A fitter may reject its assigned component, but it must not
classify the component or silently fall through to another fitter.

### 4.1 Required routing order

1. Reject a component that touches the image boundary.
2. Compare significant aperture counts in the raw and closed masks.
3. Reject unstable topology when those counts disagree.
4. Route stable topology:

| Stable evidence | Route |
|---|---|
| One significant enclosed aperture | Standard single-gate fitter |
| Zero apertures plus independent C-shape evidence | C-shape fitter |
| Exactly two apertures in one connected parent | Multi-gate fitter |
| Zero apertures with weak or contradictory C evidence | Reject unknown |
| More than two apertures | Reject unsupported |
| Any other uncertain topology | Reject ambiguous |

Separate connected parents are separate component instances. Two separate
single-aperture components are not one multi-gate instance.

### 4.2 Independent C-shape routing requirement

Zero apertures alone does not establish a C-shape. The topology stage must have
independent evidence for an open, three-sided component before dispatching the
C-shape fitter. The specialized fitter cannot declare its own input to be a
C-shape merely because it can identify a least-supported side.

#### First conservative assessor baseline — 2026-08-01 19:42 PDT

`src/topology.py` now implements a first explainable routing scaffold for clear
cases. It deliberately leaves contradictory and mixed evidence unknown rather
than asking a specialized fitter to classify its own input. This is a
development baseline for review, not a ground-truth topology model.

For each post-close component, the assessor:

1. Counts raw and post-close enclosed contours whose area is at least `12 px²`.
2. Reads `ComponentObservation.touches_frame` first. A clipped component is
   labeled `clipped`, rejected with `frame_edge_clipped`, and receives no
   density computation.
3. For a non-clipped component, obtains only cached `scale_01` evidence from the
   frame-owned density bank.
4. Uses the already shared foreground distance transform to record maximum
   closed-mask thickness.
5. Fills the convex hull of the closed outer contour and subtracts the closed
   component mask. Connected residual regions touching the convex-hull boundary
   are exterior-connected concavities. A local L2 distance transform measures
   each concavity's depth; the deepest region, then its area, is retained.
6. Records the `scale_01` P90 pixel count and P90 connected-component count as a
   simple density-coherence check.

A stable zero-hole component is labeled `c_shape` only when all of the following
first-pass conditions hold:

| Evidence | Initial threshold |
|---|---:|
| Bounding-box width and height | each `>= 20 px` |
| Fill ratio | `>= 0.32` |
| Solidity | `<= 0.80` |
| Foreground distance-transform maximum | `>= 2 px` |
| Dominant exterior-void area / convex-hull area | `>= 0.15` |
| Dominant exterior-void depth / maximum bbox side | `>= 0.08` |
| `scale_01` P90 pixels | `>= 8` |
| `scale_01` P90 components | `1` through `3` |

Stable one-hole evidence is labeled `standard`; stable two-hole evidence is
labeled `multi_void`. If stable two-hole evidence also passes the independent
C-shape conditions, it is held as `unknown` with
`mixed_multi_void_c_shape_deferred`. That explicit hold supports the planned
future sequence of multi-void ownership followed by a C-shape sub-procedure
without contaminating either current route. Raw/closed hole-count disagreement
is `unknown` with `topology_unstable`; stable zero-hole evidence below the C
threshold is `unknown` with `c_shape_evidence_insufficient`; more than two
stable holes is `unknown` with `unsupported_aperture_count`.

The schema-owned `TopologyDecision` retains the authoritative `frame_id` and
`sim_time_ns`, frame-local `component_id`, label, route, acceptance, clipped
state, raw and closed significant-hole counts, both distance-transform
measurements, exterior-void ratios, `scale_01` P90 evidence, and rejection
reason. The compound frame/component key keeps the evidence independently
auditable without a review render or temporal tracking system.

The assessor was run on all three currently available historic runs: `1,715`
frames and `4,832` post-close components. The first-baseline output was:

| Label | Components |
|---|---:|
| `standard` | `1,429` |
| `multi_void` | `108` |
| `c_shape` | `41` |
| `unknown` | `2,274` |
| `clipped` | `980` |

The unknown set contains `1,491` topology-unstable components, `776` stable
zero-hole components with insufficient C evidence, `6` unsupported aperture
counts, and exactly `1` deliberately deferred mixed case. That mixed evidence
is `run-20260731T093732Z`, frame `269727`, component `1`, matching the known
multi-void/C combination. Clipped components are counted only so exclusion is
auditable; they never enter the shape or density assessment.

#### Closed-mask hierarchy candidate iteration — 2026-08-01 21:21 PDT

The active topology implementation now supersedes the density-assisted
first-pass assessor above, while retaining that earlier record for historical
comparison. This iteration is pending visual acceptance. Classification starts
only from the final post-close mask; raw hole counts remain diagnostic and a
raw/closed count disagreement no longer rejects an otherwise usable closed
component. Topology does not request or inspect any `DensityEvidence`.

Preprocessing computes one frame-global `cv2.RETR_TREE` catalog from the closed
mask. Every `ContourNodeEvidence` records its contour ID, owning component ID,
parent and direct-child IDs, nesting depth, foreground/hole parity, area, and
full-frame UV points. Each `ComponentObservation` references its closed
foreground parent IDs and direct parent/child pairs. This retains nested
depth-2/depth-3 gate pairs without copying component masks.

The candidate routing order is:

1. Frame-edge-clipped component: reject as `clipped`.
2. Any foreground parent with more than two significant direct children:
   reject as `unknown`.
3. Any foreground parent with exactly two significant direct children: accept
   as `multi_void` and route to `multi_gate`.
4. One direct child plus a bounded exterior opening of at least `75 px`: accept
   as the mixed multi/C case and route to `multi_gate`.
5. One direct child whose aperture-center evidence passes: accept as
   `standard` and route to `standard_gate`.
6. Zero direct children plus any nonzero exterior background run bracketed by
   parent foreground: accept as `c_shape`.
7. A zero-child solid, a failed aperture-center check, or another unsupported
   structure: reject as `unknown` with an explicit reason.

A significant direct child retains the existing `12 px²` contour-area floor.
For standard confirmation, the direct-child contour is filled, its L2 distance
transform is computed with an explicit one-pixel background border, and the
center is the distance-weighted centroid of pixels at least 90% of the peak.
Contour support is projected onto 36 axes. The first calibration requires a
distance peak of at least `3 px`, inscribed-diameter/minimum-bbox-side ratio of
at least `0.50`, center/contour-centroid offset no greater than `0.20` when
normalized by square-root aperture area, 25th-percentile opposing support ratio
of at least `0.55`, and median opposing support ratio of at least `0.70`.

The exterior opening measurement fills only the closed outer parent contour,
so enclosed child apertures cannot create an opening. It records the longest
horizontal or vertical zero run bracketed by foreground pixels, its full-frame
line segment, axis, concavity area, and concavity distance-transform depth.
Any positive bounded span supplies the initial zero-child C evidence; the
configurable `75 px` threshold applies only to the one-child mixed case.

On `run-20260801T031401Z`, the 1,339 component decisions are:

| Label | Components |
|---|---:|
| `standard` | `514` |
| `multi_void` | `91` |
| `c_shape` | `105` |
| `unknown` | `343` |
| `clipped` | `286` |

The unknown set contains `301` zero-child solids, `31` unbalanced one-child
apertures, and `11` parents with more than two significant direct children.
The mixed rule identifies frame/component `106738/1` at `86 px` and
`106739/1` at `92 px`. Four nested standard candidates at hierarchy depth 2
remain independently associated with their connected components in frames
`106697` through `106700`.

### 4.3 Rejection behavior

For the present system, anything that cannot be assigned confidently to
standard, C-shape, or exactly-two-aperture multi-gate processing is discarded
from publication. Diagnostic trace data may be retained, but no gate output is
published for that instance.

Suggested reason categories include:

- `frame_component_gate`
- `component_too_small`
- `frame_edge_clipped`
- `topology_unstable`
- `c_shape_evidence_insufficient`
- `mixed_multi_void_c_shape_deferred`
- `unsupported_aperture_count`
- `specialized_fit_failed`
- `quadrilateral_validation_failed`
- `pnp_failed`
- `vehicle_state_unavailable_for_ned`

The topology-stage names above are now the implemented first-baseline values;
later fitter, validation, PnP, and publication reason vocabularies remain to be
finalized with those stages.

## 5. Standard single-aperture process

The standard route is the baseline for one stable significant enclosed
aperture:

1. Select the standard post-close-area density profile.
2. Consume the shared P90 evidence.
3. Fit a normalized four-corner quadrilateral using the standard P90 fitter.
4. Apply common quadrilateral ordering and validation.
5. Run shared camera-relative PnP.
6. Publish only when the required geometry and output conditions pass.

### 5.1 First standard-fit and PnP implementation — 2026-08-01

`standard_gate.py` now consumes only an accepted `standard_gate` topology
decision and the cached `DensityEvidence` chosen by
`DensityBank.recommended_standard_profile()`. It uses the stored P90 mask,
constructs a four-sided convex scaffold, assigns P90 pixels to their nearest
side, fits four lines, intersects adjacent lines, translates component-local
coordinates into frame UV coordinates, and emits a finite, convex, in-frame
`QuadrilateralEstimate`. The selected `DensityProfile` is retained in full and
is not confused with the separate `scale_01` evidence used by topology.

`gate_centerline_pnp.py` consumes that normalized quadrilateral through the
shared schema. It uses the documented `640 x 360` pinhole calibration,
`fx=fy=320`, principal point `(320, 180)`, zero distortion, and the ordered
`2.10 m` gate-face centerline square. OpenCV IPPE-square candidates with
positive camera depth are ordered by reprojection RMSE. The result records both
candidate count and secondary error so planar ambiguity remains visible; the
Rodrigues rotation maps gate-model coordinates into camera-optical coordinates.
Position confidence reflects reprojection fit, while orientation confidence
also incorporates the error separation between the two planar pose candidates.

`StandardGatePipeline` retains its compatibility name while implementing both
standard and C-shape routes through this frontier. It preserves
`(frame_id, sim_time_ns, component_id)` in every topology, quadrilateral, and
pose result and exposes the exact calibration, gate model, selected density
profile, fitted points, solver evidence, and rejection reason through
`GeometryFrameResult`. `GateGeometryPipeline` is its route-neutral alias.
Multi-gate decisions remain outside `processed_routes` until that fitter
implements the same quadrilateral contract.

The C-shape route emits its detailed `CShapeResult`, normalizes
`completed_quadrilateral_uv` through the same corner ordering, convexity, area,
and image-bound checks as a standard gate, and publishes a routed
`QuadrilateralEstimate`. The shared `gate_centerline_pnp.solve_gate_pose()` then
publishes its `CameraPoseEstimate`; no C-shape-specific PnP solver is used.

The first run across all three historic sources processed `1,715` frames and
`1,429` standard decisions. It produced `1,427` accepted quadrilaterals and
`1,425` accepted camera poses. Two quadrilaterals were rejected as
`standard_quadrilateral_fit_failed`; their corresponding pose records were
rejected as `quadrilateral_not_accepted`. Two further poses were rejected as
`pnp_reprojection_error`. All `7,690` serialized topology, quadrilateral, and
pose records retained exact frame identity. This establishes executable and
auditable behavior, not visual approval of every quadrilateral or pose.

## 6. C-shape process and settings

The C-shape procedure consumes a stable zero-aperture component that topology
has already classified using independent C-shape evidence.

### 6.1 Density and visible-side discovery

`configurations.py` preserves the approved fitter's existing maximum-dimension
radius bands by selecting equivalent named density-bank profiles:

| Maximum component dimension | Density profile | Density / ridge radius |
|---:|---|---:|
| `35 px` | `scale_01` | `2 / 2 px` |
| `59 px` | `scale_03` | `4 / 3 px` |
| `89 px` | `scale_06` | `7 / 6 px` |
| Unbounded | `scale_10` | `20 / 16 px` |

This mapping reproduces the prior C-shape field selection; it does not itself
approve the geometric accuracy of every band.

1. Select a C-shape-appropriate variant from the shared density bank.
2. Extract P90 evidence points from the selected final field.
3. Fit an initial convex four-sided P90 scaffold.
4. Assign every P90 point to its nearest scaffold side.
5. Count support for all four sides.
6. Treat the least-supported side as the missing-side hypothesis.
7. Require at least two evidence points on each of the other three sides.
8. Fit the three visible lines independently with `cv2.fitLine()`.

The least-supported-side rule is a fitting operation, not sufficient topology
classification evidence.

### 6.2 Finite geometry and contour evidence

1. Identify the spine opposite the missing side and the two adjacent arms.
2. Intersect both arms with the spine.
3. Sample the shared distance transform along the spine.
4. Use the median positive distance as the component half-width estimate.
5. Measure the visible extent of both arms.
6. Simplify the external contour with a `3 px` epsilon.
7. Densify the simplified contour to approximately pixel-spaced samples.
8. Select contour evidence adjacent to each visible line.
9. Divide that evidence into the two rails around the line.
10. Fit both rails and compute their parallel consensus direction.
11. Preserve the P90-derived line-center offset.

### 6.3 Bounded angular correction

| Setting | Documented value |
|---|---:|
| Contour angular influence | `0.50` |
| Maximum angular correction | `15 degrees` |

For each visible line, apply half of the axial angular difference between its
P90 direction and contour consensus, capped to plus or minus 15 degrees. Rebuild
all three lines and recompute their intersections and extents.

### 6.4 Missing-side construction

1. Cast both adjusted arms outward from their spine intersections.
2. Intersect each ray with the simplified outer contour.
3. Use the intersections as the final observed arm exits.
4. Measure both observed arm lengths and the visible spine length.
5. Set `longest_green` to the maximum of those three lengths.
6. Set the target length to `0.75 * longest_green`.
7. Calculate each arm independently:

```text
extension_i = max(0, 0.75 * longest_green - observed_arm_length_i)
```

8. Extend each arm collinearly by its individual shortfall.
9. Join the extended endpoints to construct the missing side.
10. Return the completed normalized quadrilateral with convexity and area.

`CShapeResult.refined_mask` rasterizes the three refined visible segments and
the inferred endpoint-to-endpoint missing side with a fixed `2 px` binary-mask
stroke. The fourth segment is the line historically rendered as magenta; its
endpoints remain explicit in `completed_quadrilateral_uv`. The fitted geometry
continues to use the distance-transform width independently of this display and
mask-output stroke.

The completed quadrilateral then enters common validation and PnP. A non-convex,
self-intersecting, too-small, non-finite, or out-of-frame completion produces a
rejected `QuadrilateralEstimate` and therefore a rejected pose; it is never
published as an accepted gate.

### 6.5 Official quadrilateral and PnP checkpoint — 2026-08-01

The first full replay through the common contract produced:

| Run | C-shape routes | Accepted quadrilaterals | Accepted camera poses |
|---|---:|---:|---:|
| `run-20260731T093159Z` | `57` | `12` | `11` |
| `run-20260801T031401Z` | `105` | `45` | `40` |

Every routed instance publishes one quadrilateral and one pose record, including
explicit rejection records when fitting, common validation, or PnP fails. Both
runs passed exact schema round-trip validation across all `776` frames.

## 7. Multi-gate process and settings

The multi-gate procedure consumes one connected parent with exactly two stable,
significant enclosed apertures.

### 7.1 Aperture and density evidence

1. Select a multi-gate-appropriate variant from the shared density bank.
2. Retrieve exactly two hole contours with documented contour area of at least
   `12 px`.
3. Sort the apertures largest-first.
4. Build an initial convex quadrilateral for each aperture.
5. Use a convex hull and four-sided approximation where reliable.
6. Use a minimum-area rectangle fallback for small or concave apertures.
7. Use P90 as the primary side evidence.
8. Use P70 as broader support for short gaps.
9. A `3 x 3` constrained close is documented as optional computed evidence; its
   production role is not confirmed.

### 7.2 Larger-first fitting and evidence ownership

1. Fit the larger aperture first.
2. Search outward from its aperture sides for parallel density support.
3. Measure density strength, P90 coverage, continuity, and foreground coverage.
4. Record compatibility with already fitted parallel sides.
5. Intersect selected adjacent lines to form a quadrilateral.
6. Make the larger fit's evidence available as compatibility context.
7. Remove P90 evidence owned by the accepted larger candidate.
8. Fit the smaller gate using residual evidence.
9. Calculate combined union coverage.

### 7.3 Bounded candidate optimization

Generate bounded scale, rotation, and translation variations. Each candidate
must:

- Contain its aperture seed.
- Remain within the component-local image.
- Provide sufficient weighted P90 side coverage.
- Provide sufficient minimum and mean per-side coverage.
- Bridge only short gaps, documented as up to `3 px`, and only with P70 support.

The permitted scale, rotation, and translation ranges and the coverage
thresholds are not specified in the reviewed Markdown evidence.

### 7.4 Outer-contour refinement

1. Extract contiguous parent outer-contour runs near each proposed gate side.
2. Fit robust local contour lines using `cv2.fitLine(..., DIST_HUBER)`.
3. Require adequate angular agreement, contour span, and sample count.
4. Compare no contour influence with contour influence `0.50`.
5. Cap angular modification at `15 degrees`.
6. Keep each density-line midpoint fixed while adjusting its direction.
7. Intersect adjacent adjusted lines.
8. Require convexity, image bounds, and containment of the original aperture.
9. Do not allow P90 coverage to regress beyond the approved slack.
10. Score P90 coverage with a smaller contour-alignment contribution.
11. Refine the larger gate first and the smaller gate against residual evidence.

The coverage slack, contour score weight, minimum span, minimum point count, and
angular-agreement threshold are not specified in the reviewed Markdown evidence.

## 8. Common fitter output and publication

All three fitters must return the same ordered four-corner contract. Shared code
then performs:

- Finite-value and four-corner checks.
- Consistent corner ordering.
- Convexity and non-self-intersection checks.
- Area and image-bound checks.
- Any route-independent evidence validation.
- Camera-relative PnP.
- Camera-to-NED conversion only with validated transforms and vehicle state.
- Construction of the repository's canonical `VisionObservation`.

Trace data should include at minimum:

- Frame and component identity.
- Preprocessing and density calibration versions.
- Raw and closed topology evidence.
- Selected topology route and density profile.
- Fitter name and significant settings.
- Validation and PnP status.
- Machine-readable rejection reason when rejected.

The implemented schema types at this frontier include `CShapeResult`,
`QuadrilateralEstimate`, `CameraPoseEstimate`, and `GeometryFrameResult`. Their
image corner order is
`upper_left`, `upper_right`, `lower_right`, `lower_left`. Because the physical
gate model is square, that deterministic image-space naming does not by itself
resolve the gate's fourfold rotational symmetry; downstream orientation use
must retain the recorded IPPE ambiguity evidence.

Production code must not render, annotate, or save review images. `tests/` and
`ui/` may consume structured production results, but production must not depend
on either support area.

## 9. Ambiguities requiring confirmation

The following items must be confirmed by the user or resolved through explicit
calibration evidence before being treated as approved production behavior:

1. **Ten-layer density calibration:** aggregate non-clipped post-close-area
   deciles define the provisional ten-layer recommendation bank. The standard
   route now executes those recommendations, but visual mask quality and pose
   stability have not yet approved every resulting fit.
2. **Topology-specific density selection:** standard-gate selection uses the
   aggregate area recommendation. C-shape selection now reproduces the prior
   dimension/radius bands through named profiles, but their geometric
   calibration remains unapproved; multi-gate selection remains unresolved.
3. **Connectivity:** connected-component connectivity is not stated in the
   Markdown evidence.
4. **Aperture significance:** `12 px` contour area is documented, but no
   relative-to-component-area threshold or scale-dependent rule is specified.
5. **C-shape dispatch:** the independent evidence and thresholds needed to
   classify a zero-hole component as a C-shape are not finalized.
6. **P80 evidence:** the shared schema requires it, but no current specialized
   consumer or acceptance rule is documented.
7. **Multi-gate optional close:** the exact role of the `3 x 3` constrained close
   is unresolved.
8. **Multi-gate search bounds:** scale, rotation, and translation search ranges
   are unspecified.
9. **Multi-gate scoring:** P90 coverage thresholds, allowed regression slack,
    contour-alignment weight, and contour support thresholds are unspecified.
10. **Standard-gate calibration:** detailed P90 fitting and acceptance settings
    are not supplied by the reviewed Markdown notes.
11. **Clipped components:** `schema_draft.md` mentions a possible partial
    observation process, while authoritative `../AGENTS.md` requires clipped
    components to be rejected. This draft follows the authoritative rejection
    rule unless the user explicitly changes that contract.

Until resolved, these items must remain explicit configuration gaps or rejection
conditions. They must not be filled with silent assumptions.
