# Pixel Density Calibration




This document is the authoritative calibration record for the inverse pixel
density field used by the deterministic-v3 gate-mask processing path.

> **Change control:** Do not change, remove, reinterpret, or extend any value in
> this document without explicit approval from the user. If an update appears
> necessary, ask the user for permission before modifying this document.

## Approved Density Baseline

These gamma values apply to all currently reviewed component sizes and
topologies:

- Inverse gamma: `2.10`
- Ridge gamma: `3.00`
- Relative density cap: `2.00` (unchanged)
- Density neighborhood shape: square
- Processing resolution: native mask resolution

The active code-owned settings are centralized in
`src/configurations.py` under its first section, **Density profiles**. Every
profile explicitly records its area boundary, density radius, ridge radius,
relative cap, inverse gamma, and ridge gamma so calibration changes remain
reviewable in one file.

## Approved Radius Observations

The following selections were approved from the fixed-gamma radius review:

| Reviewed component | Density radius | Ridge radius | Status |
| --- | ---: | ---: | --- |
| Compact, `35 px` | `1` | `3` | Approved |
| Overlap with two holes, `51 px` | `5` | `4` | Approved |
| Small, `59 px` | `3` | `4` | Approved |
| Medium, `82 px` | `7` | `8` | Approved |
| Large boundary, `91 px` | `24` | `12` | Approved |
| Large rotated, `185 px` | `20` | `16` | Approved; retain existing radii |

Pixel sizes above refer to the component's maximum bounding-box dimension.

## First Implementation Non-Clipped Recommendation Calibration — 2026-08-01

The first ten-profile recommendation calibration uses all `3,852` eligible
non-frame-clipped post-close components from the `1,715` frames currently
available across `run-20260731T093159Z`, `run-20260731T093616Z`, and
`run-20260731T093732Z`. Its reference factor is foreground area in pixels,
`ComponentObservation.area_px`, rather than maximum bounding-box dimension.
Discrete inclusive boundaries are the aggregate empirical P10 through P90
values computed with `numpy.percentile(..., method="higher")`.

Calibration version:
`inverse-density-v3:runs-20260731T093159Z+093616Z+093732Z:nonclipped-area-deciles-higher:cap2.0:inverse2.10:ridge3.00`

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

This is a first implementation of the ordinary recommendation buckets. All ten
profiles remain cached for every density-eligible component, and a specialized
topology path may select another named profile. The area analysis calibrates
the bucket boundaries only; the retained radius ladder still requires visual
and geometric validation by topology. The lookup exists in
`DensityBank.recommended_standard_profile()` but is not yet connected to a live
standard-gate fitter by the topology-first pipeline scaffold.

## Implementation Constraint

The approved radius observations do not by themselves define selection rules
and demonstrate that either area or maximum component size alone may be
insufficient:

- The `51 px` overlapping component and `59 px` regular component require
  different density radii.
- The `91 px` boundary component and `185 px` rotated component require
  different large-component profiles.
- Topology, including overlapping or multi-hole masks, may need to participate
  in profile selection.

Do not infer additional radius rules from the new area boundaries or apply one
ordinary recommendation as a topology-specific guarantee. Additional radius
rules must be reviewed and explicitly approved before being added here or
promoted to topology-specific production selection.

## Specialized Gate Cases Requiring Separate Processing

The standard single-void density and P90 quadrilateral path is not sufficient
for every visible gate condition. The following cases must be detected and
processed according to their own geometry and available evidence:

- Clipped gates whose structure extends beyond the frame boundary.
- Occluded gates whose structure is hidden by another object.
- Partially overlapping gates whose masks or density evidence merge.
- Gates whose orientation prevents the void from being visible.

These cases must not be forced through the standard single-visible-void fit as
if they were complete, isolated gates. Each case will likely require its own
specialized detection and processing script, followed by an explicit routing
step that selects the appropriate path. The specialized rules and scripts are
not yet defined and require separate review and explicit user approval before
implementation or production promotion.

### Topology Routing Design Record — 2026-08-01 13:50:52 PDT

The current single-void, approved C-shape, and approved exact-two-aperture
geometry procedures remain independent and are not connected to an automatic
router. The recommended topology-first dispatch boundary, current evidence,
explainable diagnostic contract, and validation requirements are recorded in:

`docs/ai-gp/20260801_135052_GATE_MASK_TOPOLOGY_ROUTING_DESIGN.md`

The design gives enclosed-aperture topology priority over specialized fit
success, checks topology both before and after morphology, and retains explicit
`CLIPPED`, `AMBIGUOUS`, and `UNSUPPORTED` outcomes. In particular, a successful
C-shape fit is not itself classification evidence: the broad discovery produced
successful fits for visually rejected non-gate components. This entry records a
recommended design only; routing thresholds and production integration still
require explicit review and approval.

### Solve for the C Shape — Controlled Development Record

Status: the finite-extent construction documented below was approved on
2026-08-01 as the working development and visual-review baseline. This approval
does not authorize automatic topology routing, PnP integration, confidence
thresholds, tracking consumption, or flight use. Those decisions remain
separately controlled.

A gate that appears C-shaped because one side is occluded may expose only two
reliable corners. Those two image-to-object point correspondences are not
sufficient for an unconstrained camera-relative PnP solve: they provide four
scalar image constraints for a six-degree-of-freedom pose. The missing corners
must not be synthesized and then treated as independent measurements. Doing so
can produce a low reprojection error and unjustifiably high confidence for a
pose that was determined largely by an assumption.

The preferred discovery direction is to retain the visible side evidence
instead of reducing the C shape to two points:

1. Fit a line to each visible centerline side of the gate and associate it with
   the known top, right, bottom, or left side.
2. Initialize a gate-pose hypothesis from a recent accepted pose, or from
   explicitly approved attitude and gate-orientation constraints.
3. Project the known 2.10 m gate-border centerline square into the image.
4. Optimize pose against distances between projected sides and the observed
   inverse-density pixels or fitted lines.
5. Score only sides that were observed; absence of evidence on the occluded
   side must not be treated as negative evidence.
6. Publish which sides and corners supported the result, along with pose
   uncertainty and any prior used.

A typical C shape can retain three visible side segments even when only two
corner intersections are reliable. Three labelled, non-collinear side lines
can provide substantially more pose information than the two corners alone,
although planar and symmetry ambiguities must still be scored and exposed.

The expected routing behavior, subject to approval, is:

| Available evidence | Candidate behavior | Pose authority |
| --- | --- | --- |
| Four ordered centerline corners | Use the existing IPPE prototype | Independent camera-relative candidate |
| Three labelled visible sides | Fit a partial model-to-contour pose | Candidate with measured uncertainty |
| Two identified points plus a recent pose or approved attitude constraints | Perform a constrained tracking update | Prior-dependent candidate |
| Two points without a prior or external constraints | Publish bearing/apparent-scale evidence only | No full pose |

The current 50-line `src/gate_centerline_pnp.py` prototype requires four
ordered corners and must not be called with two points. Its confidence measures
only reprojection fit; it does not establish that the corner roles or 2.10 m
centerline interpretation are correct.

The following decisions require explicit approval before the C-shape baseline
is extended into a pose solver or connected to production:

- Confirm that the visible C-shaped ridge represents the 2.10 m gate-border
  centerline rather than the 2.70 m outer or 1.50 m inner boundary.
- Define the topology rule that classifies a component as a C-shaped occluded
  gate and routes it away from the complete-quadrilateral solver.
- Define how visible sides are labelled and what minimum line length,
  continuity, and inverse-density support make a side usable.
- Choose the partial-pose optimizer and its initialization strategy.
- Approve which external constraints may be used: recent tracked pose, IMU
  gravity, camera attitude, known gate verticality, gate heading, or gate map.
- Define the maximum age and uncertainty of a prior pose used for a partial
  update.
- Define one-sided visible-edge scoring, rejection thresholds, ambiguity
  margins, and uncertainty reporting.
- Decide whether a partial result remains diagnostic-only or may be consumed by
  tracking or flight logic.
- Define the partial-observation data contract so an underconstrained result is
  not mislabeled as a complete six-degree-of-freedom pose.

#### Initial isolated three-line experiment

The user approved a narrow diagnostic first implementation with no production
routing and no new density calibration settings:

`src/c_shape_three_line_pose.py`

The experiment consumes the existing final component-local density field. It
uses the existing P90 quadrilateral only as an initial geometric scaffold,
assigns P90 evidence to its four sides, drops the least-supported side, and
refits the remaining three sides with `cv2.fitLine()`. The scaffold's IPPE pose
candidates initialize a six-parameter least-squares fit whose only residuals
are the distances from each projected 2.10 m model-side endpoint to its
associated observed image line.

This experiment deliberately returns both positive-depth solutions without
ranking them. It does not publish a confidence score, synthesize a measured
fourth corner, use a temporal prior, or alter the standard four-corner
`GateCenterlineQuad` contract.

The first isolated evaluation target is frame `00106766`, component label `3`,
from `run-20260801T031401Z`. Its `52 x 56 px` bounding box does not touch the
image boundary. The review output is stored in:

`production_samples/c-shape-three-line-pose-v1/`

The first result demonstrates the central ambiguity: both solutions fit the
three infinite observed lines to numerical precision, but their projected
missing sides differ substantially. Line residual alone therefore cannot
select the authoritative quadrilateral. The next refinement must address that
ambiguity explicitly; no selection rule is approved yet.

#### Finite-extent C-shape construction — Approved Working Baseline

The user approved this isolated construction as the working baseline. It pauses
PnP and estimates the missing image-space line from the finite component mask:

`src/c_shape_finite_extent.py`

It reuses the three fitted density lines, identifies the visible spine opposite
the missing side, and intersects the two outward arms with that spine. It then:

1. samples `cv2.distanceTransform()` between the two closed intersections and
   uses the median as the visible gate strip's half-width;
2. follows each arm outward through the complete binary component mask;
3. projects all mask pixels within one measured half-width of that arm and
   takes the maximum forward extent as its outer mask exit;
4. extends outward from each full-mask exit by the measured half-width into
   zero-mask space to estimate the missing-line endpoint; and
5. joins the two independently estimated endpoints with the magenta missing
   centerline.

This construction performs no PnP, pose scoring, temporal seeding, or
confidence calculation. Its baseline contract is:

- consume the existing component-local final density field and full binary
  component mask without changing their calibrated settings;
- select three supported density lines independently of `GateCenterlineQuad`;
- use the line opposite the missing side as the visible spine;
- measure mask half-width with `cv2.distanceTransform()`;
- end each green observed arm at its full-mask outer exit;
- extend outward from each exit by exactly the measured half-width into
  zero-mask space; and
- join the two exterior endpoints with the magenta line.

The approved reference evidence is the isolated, non-frame-clipped component
`00106766:3` from `run-20260801T031401Z`:

| Evidence | Recorded value |
| --- | --- |
| Component bounding box | `52 x 56 px` |
| Density threshold | `0.6284037960` |
| Missing/spine side indices | `1 / 3` |
| Visible-line P90 support counts | `26, 21, 46` |
| Distance-transform half-width | `4.996887 px` |
| Estimated full strip width | `9.993774 px` |
| Independent arm extents | `24.249112 px`, `24.542386 px` |
| First full-mask exit | `(296.052812, 152.102164)` |
| Second full-mask exit | `(275.400860, 180.679516)` |
| First exterior endpoint | `(300.130276, 154.990618)` |
| Second exterior endpoint | `(280.182286, 182.131016)` |

The reproducible implementation evidence is:

- estimator: `src/c_shape_finite_extent.py`;
- line extraction: `src/c_shape_three_line_pose.py::fit_c_shape_lines`;
- synthetic validation: `tests/test_c_shape_finite_extent.py`;
- renderer: `ui/legacy/render_c_shape_finite_extent_review.py`;
- rendered review and structured result:

`production_samples/c-shape-finite-extent-v1/`

The current baseline assumes that a full-mask ray exit is the appropriate
starting boundary for the outward half-width extension. If that exit was
created by an occluder cutting through an arm, it remains a visibility boundary
rather than a verified physical edge. That limitation does not invalidate the
baseline as a repeatable development reference, but it must be evaluated on
additional isolated examples before PnP connection or production routing is
approved.

#### Outer-contour-guided density lines — Experimental

The next independent experiment preserves the approved finite-extent baseline
and tests whether the original mask contour improves the angles of the three
visible density lines:

`src/c_shape_contour_guided.py`

For each provisional P90 side, the experiment uses its finite baseline segment
to isolate the two mask-contour rails. It excludes one measured half-width at
the segment ends, fits both rails, constructs paired midpoint samples, and fits
a contour center direction through those midpoints. The final guided line uses
that contour-derived direction while its perpendicular offset is the median of
the side's assigned P90 points. Thus the outer contour controls angle and the
P90 density continues to control center placement; no weighted blend parameter
is introduced.

The same approved reference component, `00106766:3`, produced:

| Side index | Contour-guided angle change | Guided-line P90 RMSE |
| ---: | ---: | ---: |
| `0` | `4.50 degrees` | `2.23 px` |
| `2` | `20.04 degrees` | `2.91 px` |
| `3` | `0.74 degrees` | `1.93 px` |

The original outer contour is drawn in cyan in both the baseline and guided
completion panels. Green denotes visible fitted segments, yellow the outward
half-width extensions, and magenta the estimated missing line. The rendered
comparison and full structured evidence are stored in:

`production_samples/c-shape-contour-guided-v1/`

The `20.04-degree` adjustment on side `2` is intentionally exposed rather than
automatically accepted. This experiment is not approved to replace the working
baseline until its rail isolation and angle changes have been reviewed on this
and additional isolated components.

##### Fixed 3 px outer-contour simplification trial

A follow-up trial applies `cv2.approxPolyDP()` with a fixed `3.0 px` epsilon to
the original outer contour before rail selection and fitting. The simplified
segments are sampled back to approximately one-pixel spacing for the existing
rail computation. This isolates contour simplification from any future
P90-versus-contour weighting change.

On reference component `00106766:3`, simplification reduced the stored contour
from `171` raw points to `10` vertices and produced:

| Side index | Unsimplified angle change | Simplified angle change | Simplified P90 RMSE |
| ---: | ---: | ---: | ---: |
| `0` | `4.50 degrees` | `4.46 degrees` | `2.24 px` |
| `2` | `20.04 degrees` | `22.48 degrees` | `3.00 px` |
| `3` | `0.74 degrees` | `0.13 degrees` | `1.93 px` |

The result is mixed: the spine becomes more stable relative to P90, while the
difficult side `2` receives a larger contour-driven rotation. The trial is
stored separately and does not replace either the approved baseline or the
unsimplified contour-guided result:

`production_samples/c-shape-contour-guided-simplify-3px-v1/`

No P90-versus-contour weighting was added in this trial.

##### Full parallel-angle alignment trial

`src/c_shape_parallel_aligned.py` replaces the overall contour-midpoint angle
fit with an explicit parallel-direction constraint. For each visible side, it
takes the two rail directions produced from the simplified outer contour and
computes their equal-weight axial consensus. The density line is made exactly
parallel to that consensus; its perpendicular position remains the median of
the assigned P90 pixels.

This is a `100%` contour-angle trial with no blend weight. P90 influences line
position but not angle. On reference component `00106766:3` it produced:

| Side | Paired-rail angle gap | P90-to-parallel angle change | P90 RMSE |
| ---: | ---: | ---: | ---: |
| `0` | `13.55 degrees` | `4.40 degrees` | `2.24 px` |
| `2` | `18.56 degrees` | `21.87 degrees` | `2.97 px` |
| `3` | `0.57 degrees` | `0.13 degrees` | `1.93 px` |

The large arm rail gaps show that isolating the correct parallel perimeter
segments remains more uncertain than the spine. The result is retained as a
separate diagnostic and does not replace the approved baseline:

`production_samples/c-shape-parallel-aligned-v1/`

Future angular weighting, if approved, can interpolate between the original
P90 direction and this axial contour consensus without changing the current
P90 offset rule.

##### Bounded contour-angle influence trial

`src/c_shape_bounded_angle.py` implements that interpolation as a separate
experimental stage. It gives the simplified-contour parallel consensus a
bounded influence on the shortest axial angle from the original P90 line and
limits the total correction with a configurable angular cap. The line's
perpendicular position is still the median projection of its assigned P90
pixels. Consequently, contour evidence can make a modest angular correction
but cannot move a line wholesale to the contour fit.

For axial line angles modulo `180 degrees`, the current operation is:

`applied_delta = clip(0.50 * shortest_axial_delta, -15 degrees, +15 degrees)`

The initial `15%` review on reference component `00106766:3` produced:

| Side | Raw P90-to-contour delta | Weighted delta | Applied delta | Capped | P90 RMSE |
| ---: | ---: | ---: | ---: | :---: | ---: |
| `0` | `-4.40 degrees` | `-0.66 degrees` | `-0.66 degrees` | No | `2.13 px` |
| `2` | `+21.87 degrees` | `+3.28 degrees` | `+3.00 degrees` | Yes | `1.73 px` |
| `3` | `-0.13 degrees` | `-0.02 degrees` | `-0.02 degrees` | No | `1.93 px` |

The bounded completion stays visually close to the approved P90 baseline,
while side `2` no longer inherits the full `21.87-degree` contour rotation.
The six-panel review preserves the source, simplified contour, line overlay,
approved baseline, full-parallel result, and bounded result side by side:

`production_samples/c-shape-bounded-contour-angle-15pct-3deg-v1/`

The next iteration raises only the influence to `30%`; the cap, P90 offset,
input evidence, and reference component are unchanged:

| Side | Raw P90-to-contour delta | Weighted delta | Applied delta | Capped | P90 RMSE |
| ---: | ---: | ---: | ---: | :---: | ---: |
| `0` | `-4.40 degrees` | `-1.32 degrees` | `-1.32 degrees` | No | `2.17 px` |
| `2` | `+21.87 degrees` | `+6.56 degrees` | `+3.00 degrees` | Yes | `1.73 px` |
| `3` | `-0.13 degrees` | `-0.04 degrees` | `-0.04 degrees` | No | `1.93 px` |

The current `30%` review and structured result are stored separately at:

`production_samples/c-shape-bounded-contour-angle-30pct-3deg-v1/`

A follow-up raises only the angular cap from `3 degrees` to `15 degrees`. This
makes the cap inactive on the reference component, so every side receives its
complete `30%` weighted correction:

| Side | Raw P90-to-contour delta | Weighted delta | Applied delta | Capped | P90 RMSE |
| ---: | ---: | ---: | ---: | :---: | ---: |
| `0` | `-4.40 degrees` | `-1.32 degrees` | `-1.32 degrees` | No | `2.17 px` |
| `2` | `+21.87 degrees` | `+6.56 degrees` | `+6.56 degrees` | No | `1.78 px` |
| `3` | `-0.13 degrees` | `-0.04 degrees` | `-0.04 degrees` | No | `1.93 px` |

The green observed arm, full-mask exit, yellow extension, and missing-line
endpoint are all recomputed from each adjusted line. The new review is stored
at:

`production_samples/c-shape-bounded-contour-angle-30pct-15deg-v1/`

The current narrow iteration raises only the influence from `30%` to `50%`;
the `15-degree` cap and all line/extent construction remain unchanged. On the
same reference component it produces:

| Side | Raw P90-to-contour delta | Weighted delta | Applied delta | Capped | P90 RMSE |
| ---: | ---: | ---: | ---: | :---: | ---: |
| `0` | `-4.40 degrees` | `-2.20 degrees` | `-2.20 degrees` | No | `2.21 px` |
| `2` | `+21.87 degrees` | `+10.93 degrees` | `+10.93 degrees` | No | `2.01 px` |
| `3` | `-0.13 degrees` | `-0.07 degrees` | `-0.07 degrees` | No | `1.93 px` |

Its review and structured result are stored at:

`production_samples/c-shape-bounded-contour-angle-50pct-15deg-v1/`

##### Decoupled observed-path and predicted-extension review

`src/c_shape_decoupled_extension.py` isolates the next endpoint experiment.
It first completes the current `50%` / `15-degree` line fitting. Each adjusted
green arm is then cast forward as a ray and intersected directly with the
simplified outer contour. These subpixel intersections replace the earlier
maximum projection of mask pixels inside a half-width corridor. The resulting
green paths and white contour-exit markers are frozen across the extension
sweep. Only the yellow predicted continuation is evaluated, always remaining
collinear with its corresponding green path.

No automatic scale selection or new confidence score is introduced. The first
review compares symmetric yellow lengths of `0.5`, `1.0`, `2.0`, and `3.0`
times the measured `4.393799 px` half-width. All four reference candidates are
convex, with image-space areas of `1026.04`, `1109.54`, `1275.22`, and
`1439.11 px2`, respectively. The `1.0` candidate retains the prior one-
half-width yellow-length rule but starts at the direct contour intersection.

On reference component `00106766:3`, the former corridor-derived exits and the
new contour intersections are:

| Arm | Former mask-corridor exit | Direct simplified-contour exit |
| ---: | --- | --- |
| `0` | `(296.716145, 150.968275)` | `(296.586981, 150.884025)` |
| `2` | `(274.802654, 182.479016)` | `(273.315963, 181.694460)` |

The implementation already accepts independent scales for the two arms, but
this first visual uses equal scales to isolate longitudinal extension from
asymmetric-shape selection. Nothing in this experiment is promoted as the
selected quadrilateral. Its renderer, synthetic validation, and review output
are:

- `ui/legacy/render_c_shape_decoupled_extension_review.py`
- `tests/test_c_shape_decoupled_extension.py`
- `production_samples/c-shape-decoupled-contour-exit-sweep-50pct-15deg-v2/`

The contour-intersection and decoupled-extension behavior is accepted as a
development checkpoint for broader visual review. This checkpoint status does
not select an extension scale and does not integrate the result into production.

##### Non-edge-clipped C-shape checkpoint sweep

`ui/backend/render_c_shape_checkpoint_sweep.py` scans every available recorded
run and applies the checkpoint without changing its parameters. Discovery
requires zero enclosed holes, a minimum `20 px` width and height, fill ratio at
least `0.32`, solidity no greater than `0.80`, no contact with any of the four
image edges, and a successful contour-exit checkpoint fit.

The scan covered all eight runs and recorded:

| Measurement | Count |
| --- | ---: |
| Frame files | `4,928` |
| Prepared-mask components | `11,076` |
| Edge-clipped components excluded | `2,806` |
| Open-shape topology candidates | `229` |
| Successful checkpoint fits | `213` |
| Fit failures | `16` |
| Visually confirmed non-gate components excluded | `14` |
| Temporally deduplicated gate candidates | `84` |
| Diverse review examples | `24` |

The 24-example subset spans all eight runs, maximum component dimensions from
`21 px` through `205 px`, and all four inferred missing-side orientations
(`6`, `8`, `6`, and `4` examples for sides `0` through `3`). Selection first
removes nearby same-run repetitions, then greedily maximizes diversity across
size, aspect ratio, solidity, fill ratio, contour-angle disagreement, missing
side, and run coverage. The known reference `00106766:3` is retained.

Each row preserves the same six-panel review: source, fixed green paths with
white direct-contour exits, and symmetric yellow extensions at `0.5x`, `1x`,
`2x`, and `3x` half-width. The combined sheet, individual full rows, complete
candidate catalog, and structured selection manifest are stored at:

`production_samples/c-shape-checkpoint-nonclipped-sweep-v2/`

The manifest preserves both aggregate discovery counts and exact run, frame,
component, bounding-box, topology, and fit metadata for every selected example.

##### First longest-green target trial

The next narrow change replaces fixed yellow half-width multiples with an
observed-length rule in `src/c_shape_decoupled_extension.py`. It measures the
two protruding green arms from their closed intersections to their direct
contour exits, also measures the green spine between the closed intersections,
and sets:

`target_length = max(first_green_arm, second_green_arm, green_spine)`

Each protruding arm receives only the collinear yellow length required for its
combined green-plus-yellow length to equal that target:

`yellow_i = max(0, target_length - green_arm_i)`

No angle, line offset, contour exit, or density calculation changes. There is
no fixed yellow multiplier or yellow-length cap in this trial. If a protruding
arm is already the longest of the three green segments, its yellow extension
is zero.

For reference component `00106766:3`, the two green arms measure `24.719188 px`
and `23.074593 px`, while the green spine measures `40.658803 px`. The spine is
therefore the target, producing yellow extensions of `15.939615 px` and
`17.584210 px`, equivalent to `3.6278x` and `4.0021x` its measured half-width.

The same 24-example, non-edge-clipped subset was regenerated with columns for
the prior `1x`, `2x`, and `3x` comparisons followed by the new adaptive result.
All 24 adaptive quadrilaterals are convex. Across the 48 protruding arms, `15`
are already longest and receive effectively zero yellow extension. The
observed scale range is `0.0x` through `10.5388x`; the high end occurs where a
green arm is extremely short and is intentionally exposed for review rather
than silently capped.

The combined review, individual examples, and per-example green lengths,
target, yellow lengths, scales, convexity, and area are stored at:

`production_samples/c-shape-checkpoint-longest-green-extension-v3/`

##### Rejected longest-green baseline plus arm-shortfall trial

The clarified rule treats the longest of the three green segments as the
baseline *yellow* extension rather than the final completed-arm target. For
each protruding green arm it also adds that arm's shortfall from the longest
green length:

`shortfall_i = longest_green - green_arm_i`

`yellow_i = longest_green + shortfall_i`

Equivalently, `yellow_i = 2 * longest_green - green_arm_i`, so every completed
green-plus-yellow arm has length exactly `2 * longest_green`. All green fits,
direct contour exits, angular influence, and angular cap remain unchanged.

On reference component `00106766:3`, the longest green length is
`40.658803 px`. Green arms of `24.719188 px` and `23.074593 px` therefore
receive yellow extensions of `56.598418 px` and `58.243013 px`; both completed
arms measure `81.317606 px`.

The same 24-example review exposes the rule without a yellow-length cap. The
yellow half-width multiples range from `5.3190x` through `22.6016x`. Twenty-two
results remain convex; two become non-convex where one or both observed arms
are extremely short. No fallback or rejection was added. The comparison
sheet, individual examples, and full numeric manifest are stored at:

`production_samples/c-shape-checkpoint-longest-green-plus-shortfall-v4/`

This interpretation was rejected because it added the longest green length to
the arm shortfall instead of using the shortfall itself as the tailored yellow
extension. It is retained only as traceable comparison evidence.

##### Tailored green-length shortfall rule

The corrected rule keeps the longest of all three green segments as the final
completed-arm target. Each of the two protruding green arms receives its own
extension equal only to its individual shortfall:

`longest_green = max(first_green_arm, second_green_arm, green_spine)`

`extension_i = longest_green - green_arm_i`

Thus `green_arm_i + extension_i = longest_green` independently for both
protruding arms. The final review panel draws the extension attached to the
shorter protruding green arm in red and the extension attached to the longer
protruding green arm in yellow. Color does not change the geometry.

On reference component `00106766:3`, the target remains `40.658803 px`. The
green arms measure `24.719188 px` and `23.074593 px`, so their tailored
extensions measure `15.939615 px` and `17.584210 px`; both completed arms equal
the target to floating-point precision.

All 24 broad-review results are convex under this rule. The maximum absolute
completed-length error is below `1e-14 px`. The review and structured evidence
are stored at:

`production_samples/c-shape-checkpoint-tailored-shortfall-v5/`

##### 75% longest-green tailored shortfall

The next minimal calibration multiplies the longest-green target by `0.75`
before computing the two tailored arm extensions:

`scaled_target = 0.75 * longest_green`

`extension_i = max(0, scaled_target - green_arm_i)`

The zero clamp prevents an arm already longer than the scaled target from
receiving a backward extension. Red continues to identify the extension on the
shorter protruding green arm and yellow the extension on the longer protruding
green arm.

For reference component `00106766:3`, the longest green segment remains
`40.658803 px`, giving a scaled target of `30.494102 px`. Green arms of
`24.719188 px` and `23.074593 px` receive extensions of `5.774915 px` and
`7.419510 px`; their completed lengths both equal the scaled target.

All 24 broad-review quadrilaterals remain convex. Across the 48 protruding
arms, `26` already meet or exceed the scaled target and receive zero extension.
The review and structured evidence are stored at:

`production_samples/c-shape-checkpoint-75pct-tailored-shortfall-v6/`

#### Approved C-Shape Geometry Baseline — 2026-08-01

The user accepted the `75%` tailored-shortfall result as the new net C-shape
geometry baseline. It supersedes the earlier finite-extent construction as the
starting point for subsequent C-shape geometry work, while preserving all
earlier artifacts as historical review evidence. This approval does not alter
the separately approved multi-gate overlap baseline.

The accepted C-shape contract is:

1. consume the existing component-local inverse-density field and P90 evidence;
2. fit the three visible green lines independently;
3. simplify the original outer contour by `3 px`;
4. apply `50%` of the shortest axial difference toward the paired-rail contour
   consensus, capped at `15 degrees`, while retaining P90 line offsets;
5. intersect each adjusted protruding line directly with the simplified outer
   contour to obtain its observed exit;
6. measure the two green arms and green spine;
7. set `scaled_target = 0.75 * max(all three green lengths)`; and
8. apply `max(0, scaled_target - green_arm_i)` independently and collinearly to
   each protruding arm.

The accepted 24-example, eight-run, non-edge-clipped review is:

`production_samples/c-shape-checkpoint-75pct-tailored-shortfall-v6/`

All 24 reviewed quadrilaterals are convex. A pinned geometry-local copy now
lives at:

`src/c_shape/c_shape_tailored_shortfall_baseline.py`

The pinned module explicitly fixes the accepted `0.50`, `15-degree`, and
`0.75` settings and exposes `fit_c_shape_geometry_baseline()`. The working
source module remains in place for historical reproducibility and continued
experimentation. Future accepted C-shape geometry refinements should branch
from the geometry-local baseline rather than mutating this pinned file.

The corresponding renderer is
`ui/legacy/render_c_shape_bounded_angle_review.py`, and the correction-cap and
three-line behavior are covered by `tests/test_c_shape_bounded_angle.py`.
This remains experimental and does not replace the approved finite-extent
baseline until it is reviewed on additional isolated components.

### Initial Two-Aperture Overlap Solver

The user approved an independent first-iteration solver for known components
containing exactly two visible gate apertures:

`src/overlapping_gate_aperture_solver.py`

This script is exploratory and is not integrated into the standard production
detector. It accepts repeatable `--scenario FRAME_ID:LABEL` arguments and uses
the following intentionally limited process:

1. Detect exactly two significant apertures and order them by aperture area.
2. Fit the larger aperture first without removing or claiming its pixels.
3. Seed four side orientations from each aperture.
4. Search outward from every aperture side using bounded parallel candidate
   lines and score whole-line density coverage and continuity.
5. Allow the smaller fit to reuse larger-gate evidence only through a parallel
   line compatibility bonus.
6. Intersect the four selected side lines to form one quadrilateral per
   aperture.

Small apertures and substantially concave, overlap-damaged apertures currently
fall back to a minimum-area rectangle before the parallel density search. The
outward search is capped at the active ridge radius plus two pixels to prevent a
side from jumping to a farther gate or unrelated density structure.

The initial known scenarios are:

- Frame `00106710`, component label `3`, maximum dimension `82 px`.
- Frame `00107025`, component label `1`, maximum dimension `65 px`.
- Frame `00107026`, component label `1`, maximum dimension `61 px`.

Current notes and issues:

- The solver requires exactly two visible apertures; it does not handle a fully
  hidden void.
- It is not yet connected to an automatic topology router.
- The aperture rectangle fallback improves overlap-damaged contours but can
  discard genuine perspective information.
- The first `82 px` scenario remains the most difficult because the overlap
  distorts both the smaller aperture and the larger gate junction.
- Shared evidence is recognized only through local parallel-line compatibility;
  no global joint optimization is performed.
- The search thresholds and concavity cutoff are first-iteration experimental
  values and are not approved production calibration.
- It currently uses the existing size-bucket radii. The newly approved
  per-example radius observations remain unpromoted until routing boundaries
  are explicitly approved.

The initial rendered review is stored at:

`production_samples/two-aperture-overlap-solver-v1/`

#### Traceable Post-Density Experiment History

The following two attempts were reviewed and rejected. Their outputs remain in
the review folder as evidence, but both operations have been removed from the
active overlap-solver code.

1. **P80 directional close (`10x1`, then `1x10`).** This joined horizontal and
   vertical density evidence too aggressively. It added `99`, `50`, and `49`
   pixels to the three known scenarios and overfilled the shared junction in
   the first scenario. It is not active. Review artifacts:
   `three-known-two-gate-scenarios-morph-p80-close10.png` and
   `three-known-two-gate-scenarios-fit-from-morph-p80-close10.png`.
2. **Iterative P90-through-P70 geodesic reconstruction (`3x3`).** Repeated
   dilation of the P90 seeds, constrained to P70, reached the entire connected
   P70 support in only `3` to `4` iterations for all three scenarios. It did not
   preserve meaningful selectivity between strong and merely allowed density
   evidence. It is not active. Review artifacts:
   `three-known-two-gate-scenarios-reconstruct-p90-through-p70.png` and
   `three-known-two-gate-scenarios-fit-reconstruction-p90-through-p70.png`.

The next reviewed experiment performed one ordinary `3x3` close on the
P90 seed and then intersects the result with the pre-existing P70 support. This
is a single one-pixel-radius pass: it is neither a directional close nor an
iterative dilation, and it cannot add pixels outside the P70 evidence mask. It
remains as comparison evidence and is not an input to the current checkpoint
fit. Its artifacts are:
`three-known-two-gate-scenarios-constrained-close-p90-in-p70-k3.png` and
`three-known-two-gate-scenarios-fit-constrained-close-p90-in-p70-k3.png`.

#### Coverage-Optimized P90/P70 Checkpoint — 2026-08-01

Status: promising and sufficient to preserve as the current overlap-solver
checkpoint, but still exploratory and not integrated into production. A
broader sample sweep is required before promotion.

The current checkpoint remains in the independent script:

`src/overlapping_gate_aperture_solver.py`

It uses the approved inverse-density gamma baseline and the existing radius
routing, then applies this bounded and explainable process:

1. Extract P90 pixels as the primary, value-weighted geometric evidence and
   retain P70 as allowed gap evidence only.
2. Detect exactly two apertures and generate one initial P90 quadrilateral for
   each, ordered by aperture area.
3. Solve the larger-aperture candidate first against all P90 pixels.
4. Solve the smaller-aperture candidate against P90 pixels not already covered
   by the larger candidate. No small-to-large scale, area, containment, or
   overlap ratio is imposed.
5. Generate candidates only through coherent scale, rotation, and translation
   of the initial quadrilateral. Independent corner optimization is not used.
6. Rank candidates lexicographically by weighted P90 perimeter coverage,
   minimum side coverage, mean side coverage, fewer P70 additions, and smaller
   geometric change.
7. Permit P70 additions only across an internal gap of at most `3 px`, bounded
   by P90 evidence at both ends, inside a `+/-1 px` side corridor. Added bridge
   evidence is one pixel wide and may not leave the P70 mask.

A P90 point is covered when it lies within `1.25 px` of either selected
quadrilateral perimeter. Coverage refers to proximity to the perimeter, not
containment inside the quadrilateral area; an oversized polygon therefore does
not receive credit merely for surrounding evidence.

The scale search is staged rather than always evaluating the full range:

- Initial scales: `0.95`, `1.00`, `1.05`, `1.10`, and `1.15`.
- Evaluate `1.20` only if `1.15` is the winning initial scale.
- Evaluate `1.25` only if `1.20` becomes the winner.
- Retain the best candidate seen and stop when the newest boundary scale does
  not win.

The other bounded candidate values are rotations `-4`, `0`, and `+4` degrees
and translations `-2`, `0`, and `+2 px` independently in each image axis. The
initial stage therefore evaluates `135` coherent candidates per gate; each
scale extension adds only `27` candidates. Candidates must contain their own
aperture seed and remain within the component crop.

The three checkpoint results are:

| Frame and label | Large transform | Small transform | Staged extension | Weighted P90 union coverage | P70 additions | Minimum small-side support |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `00106710:3` | `s=1.00`, `rot=+4`, `move=(0,0)` | `s=1.05`, `rot=0`, `move=(0,0)` | None | `79.4%` (`189/237`) | `26 px` | `28%` |
| `00107025:1` | `s=1.00`, `rot=0`, `move=(0,0)` | `s=1.15`, `rot=-4`, `move=(-2,0)` | `1.20` evaluated and rejected | `91.9%` (`198/216`) | `16 px` | `30%` |
| `00107026:1` | `s=1.00`, `rot=0`, `move=(0,0)` | `s=1.10`, `rot=0`, `move=(0,2)` | None | `95.9%` (`205/214`) | `6 px` | `13%` |

No bridge pixel in these cases fell outside P70. The `00107025:1` result also
shows that the earlier `1.15` selection was not forced by a search boundary:
the staged search evaluated `1.20`, which scored lower, and correctly stopped
without evaluating `1.25`.

Known limitations at this checkpoint:

- Validation currently covers only the three hand-selected two-aperture
  components above.
- Minimum support on the smaller quadrilateral remains low, particularly in
  `00107026:1`; high union coverage alone does not establish that every side is
  reliable.
- The `1.25` scale stage has not yet been exercised by a winning `1.20`
  candidate.
- Larger-first residual assignment is intentionally simple and may allocate
  ambiguous shared evidence imperfectly.
- Candidate geometry is still seeded only from the apertures and P90 field.
  Original-mask outer corners do not yet influence orientation, scale, corner
  placement, candidate ranking, or rejection.
- Topology routing and production integration remain unimplemented.

The authoritative visual checkpoint is:

`production_samples/two-aperture-overlap-solver-v1/three-known-two-gate-scenarios-coverage-staged-scale-p90-p70.png`

Its structured measurements are stored in the same folder's `manifest.json`.
The immediately preceding fixed-maximum review remains preserved as
`three-known-two-gate-scenarios-coverage-optimized-p90-p70.png`.

The planned outer-contour stage was subsequently exercised as the separately
reviewable checkpoint below. The P90/P70 behavior above remains preserved as
its unchanged baseline.

#### Multi-Gate Overlap Checkpoint: Outer-Only Post-Pass — 2026-08-01

Status: locked exploratory checkpoint. It is not integrated into production
and its constants or selection rules must not be changed without explicit user
approval.

The implementation is:

`src/overlapping_gate_contour_side_refinement.py`

It calls the existing `src/overlapping_gate_aperture_solver.py` first and then
performs one bounded outer-contour-only post-pass. The post-pass does not use an
inner aperture contour to position or orient a quadrilateral. Its active
behavior is:

1. Preserve the existing P90/P70 larger-first and residual-evidence fits as the
   baseline candidates.
2. Extract the raw outer component contour with `CHAIN_APPROX_NONE`.
3. For every baseline side, find a contiguous outer-contour run within the
   local side corridor. Accept a run only when it has at least four points,
   spans at least `20%` of the side, and agrees with the baseline direction to
   within `25 degrees`.
4. Compare exactly two post-pass candidates: unchanged influence `0.00` and
   outer-contour angular influence `0.50`.
5. Keep each baseline side midpoint fixed. Rotate only its direction halfway
   toward the fitted raw outer-contour direction, capped at `15 degrees`. An
   unmatched side remains unchanged.
6. Re-intersect adjacent side lines and reject a candidate that is non-convex,
   leaves the component crop, or no longer contains its associated aperture.
7. Reject an adjusted candidate that loses more than one percentage point of
   weighted P90 coverage. Rank remaining candidates primarily by P90 coverage,
   with outer-contour alignment carrying a `0.01` secondary weight.
8. Refine the larger gate first and compute the smaller fit against the
   resulting residual P90 evidence.

The post-pass currently makes these measurable changes:

| Frame and label | Baseline union | Outer-only union | Large accepted | Small accepted |
| --- | ---: | ---: | :---: | :---: |
| `00106710:3` | `79.41%` | `80.21%` | Yes (`0.50`) | No: aperture containment |
| `00107025:1` | `91.87%` | `94.59%` | Yes (`0.50`) | Yes (`0.50`) |
| `00107026:1` | `95.90%` | `96.36%` | Yes (`0.50`) | No: P90 loss exceeded `1 pp` |

The following data and operations are present for review or diagnostics but do
not influence the accepted post-pass quadrilaterals:

- Inner aperture contours are drawn as context only. They do not gate, rotate,
  translate, scale, rank, or reject a post-pass candidate.
- Simplified outer and inner contours returned by `component_boundaries()` are
  currently computed and discarded. Only the raw outer contour is fitted.
- The fitted outer run's direction affects rotation. Its absolute fitted point
  and endpoints are retained only for review drawing; they do not move a side.
- Side midpoints are never translated, and the post-pass performs no scale or
  independent corner search.
- The final recomputation of P70 bridge pixels, union coverage, colored panels,
  image resizing, PNG output, and JSON manifest is review instrumentation. It
  does not feed back into post-pass selection.
- The review entry point calls `solve_two_apertures()` to obtain the density
  field but discards that call's quadrilateral fits; the base coverage solver
  then constructs aperture-seeded fits again from P90 evidence. The first fit
  is redundant for this path and accounted for about `39 ms` across the three
  profiled examples. A batch runner can compute `density_field()` directly
  without changing checkpoint math.

The base P90/P70 solver still evaluates P70 side gaps and side support for every
scale/rotation/translation candidate. That operation is active in the base
candidate ranking, even though it did not change a selected transform in the
three known examples. A diagnostic timing run on the same process measured:

| Base solve | Current behavior | Diagnostic without per-candidate side tie-break |
| --- | ---: | ---: |
| `00106710:3` | `468.77 ms` | `23.04 ms` |
| `00107025:1` | `494.56 ms` | `25.53 ms` |
| `00107026:1` | `473.59 ms` | `24.83 ms` |

All three diagnostic runs selected the same large and small transforms and the
same weighted union coverage as the current behavior. This is evidence for a
future two-stage optimization—rank geometry first, then calculate P70 side
support only for tied finalists—but it is not sufficient to remove the
tie-break from this locked checkpoint.

Under `cProfile`, the three complete `solve()` calls consumed `2.206 s`; the
unchanged base solver consumed `2.184 s` of that total. The six outer-contour
`refine_gate()` calls consumed about `5 ms` total. The contour post-pass is
therefore a small incremental compute cost; the existing per-candidate P70
side sampling dominates the current overlap path.

The authoritative outer-only visual and structured manifest are:

`production_samples/two-aperture-contour-side-refinement-v3-outer-only/`

The prior inner-and-outer contour and lower-influence reviews remain historical
comparison artifacts only. They are not part of this checkpoint.

##### Minimal broader-validation path

Broader coverage should reuse the locked solver without adding new fit logic:

1. Add a separate batch-review entry point that discovers components with
   exactly two visible aperture contours. Do not put discovery, rendering, or
   run traversal into the solver.
2. Run the unchanged baseline and outer-only post-pass on every discovered
   component and record acceptance/rejection reason, component area, crop size,
   aperture areas, guided-side counts, P90 coverage before and after, selected
   influence, and corner movement.
3. Produce one compact four-panel card per component: source, P90/P70 baseline,
   outer-only result, and a coverage/difference panel. Preserve a JSON manifest
   as the machine-readable result.
4. Sort the contact sheets by component size and separate accepted-small,
   rejected-small, clipped, and ambiguous/more-than-two-aperture cases. The
   latter categories must be reported but not forced through this solver.
5. Review representative successes and failures before deciding whether the
   outer-only post-pass should be promoted, removed, or routed only to a subset
   of two-aperture components.
6. Only after output equivalence is demonstrated across that broader catalog,
   evaluate deferring P70 side-support computation to tied finalists.

#### Approved Multi-Gate Geometry Standard — 2026-08-01

The user explicitly approved the locked P90/P70 larger-first overlap solver and
its bounded outer-contour-only post-pass as the standard baseline for continued
multi-gate geometry work. Do not replace this baseline with the experimental
multi-layer morphology path unless a future change is explicitly reviewed and
approved.

The broader validation processed `89` exact-two, non-clipped components without
solver errors. Weighted P90 union coverage improved in `53`, remained unchanged
in `35`, and regressed by `0.26 percentage points` in one component. The
outer-only post-pass adjusted both gates in `24` cases, only the larger gate in
`30`, only the smaller gate in `12`, and neither gate in `23`.

The separate multi-layer morphology side track tested a one-pixel-radius
(`3x3`) P90 close constrained to P70, a one-pixel-radius P40-support open, and
normalized layer influences of `1.00`, `0.70`, and `0.40`. Across its `12`
reviewed examples, weighted P90 coverage was unchanged in `2`, lower in `10`,
and changed on average from `73.48%` to `67.55%`. That experiment remains
review evidence only and is not part of the approved standard.

Pinned copies of the approved geometry baseline now live in:

- `src/multi_gate/overlapping_gate_aperture_solver.py`
- `src/multi_gate/overlapping_gate_contour_side_refinement.py`

The first file preserves the P90/P70 aperture-seeded, larger-first residual
solver. The second preserves the `0.50` raw outer-contour angular influence,
`15-degree` correction cap, aperture-containment validation, and one-percentage-
point P90 safety rule. It imports the geometry-local aperture solver so future
geometry work can proceed within this package.

The original source modules remain in place and unchanged for historical
review reproducibility. Future multi-gate geometry refinements should branch
from `src/multi_gate/`; review renderers, batch traversal, morphology side tracks,
and generated images remain outside the approved runtime baseline.

## Current Review Artifact

The selections above came from:

`production_samples/radius-sweep-fixed-gamma-2p10-3p00/`
