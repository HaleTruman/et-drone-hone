# Overlapping-Gate Identification Iteration Guide

Status: isolated discovery baseline, 2026-08-02.

This document records the current overlapping-gate identification experiment
so it can be reviewed, reproduced, and refined without silently changing the
production topology pipeline. The implementation lives entirely in this
`profile_calibration/local_thickness` folder. Production code does not import
it.

## Purpose and boundary

The experiment asks whether variation in component-local wall thickness,
combined with simple mask topology, can propose connected components that may
contain overlapping gates and expose useful regional density evidence.

It currently produces:

- an `experimental_overlap_candidate` proposal;
- auditable thickness-junction and aperture evidence;
- optional thinner and thicker population masks;
- an adaptive composite of the existing ten production density fields; and
- regional P70, P80, and P90 evidence masks.

It does **not** currently produce:

- two proven gate identities;
- a production topology route;
- quadrilaterals for the two thickness populations;
- temporal instance tracking; or
- permission for clipped observations to enter production inference.

The thinner and thicker masks are thickness classes. They must not be renamed
`gate_a` and `gate_b` until independent geometry establishes ownership.

## Authoritative inputs

The experiment consumes existing runtime evidence without changing it:

```text
Historic JPEG
  -> production preprocessing
  -> FrameObservation.closed_mask
  -> FrameObservation.component_labels
  -> ComponentObservation
       component_id
       bbox_xywh
       image_origin_uv
       area_px
       touches_frame
       topology.closed_hole_count
```

`FrameObservation.closed_mask` is the mask used for all thickness and density
work. Every calculation is component-local and is copied back into full-frame
UV only where `component_labels == component_id`. Two disconnected components
therefore cannot influence one another's thickness statistics.

## Implemented evidence waterfall

### 1. Component-local distance transform

The component mask is padded with an explicit zero border and processed using
the precise OpenCV L2 distance transform:

```text
D(x) = distance from foreground pixel x to observed background
```

`D(x)` is an inscribed radius only at medial locations. It is not treated as a
complete wall-thickness map by itself.

### 2. Covering local thickness

Maximal disk centers are retained when a neighboring disk does not completely
contain them. The largest retained disk covering each foreground pixel is then
rasterized:

```text
R_cover(x) = largest retained foreground-disk radius covering x
T_cover(x) = 2 * R_cover(x)
```

This spreads a wall-scale estimate over the complete mask rather than leaving
the evidence only on the medial ridge.

### 3. Regional stabilization

`T_cover` is smoothed using a component-constrained normalized Gaussian with a
fixed initial sigma of `1 px`. Both numerator and support mask are blurred, then
divided, so background and neighboring connected components cannot leak into
the result:

```text
T_regional = blur(T_cover * component_mask) / blur(component_mask)
```

Raw and stabilized maps remain separate review evidence.

### 4. Component-relative thickness

For a complete component, the initial reference is:

```text
T50 = median(T_regional inside the component)
relative_thickness(x) = T_regional(x) / T50
```

For clipped components, pixels close enough to a touched image side that their
supporting disk would extend beyond the recorded frame are excluded from the
candidate statistics. They remain visible in the review maps.

### 5. Excess-thickness proposals

A pixel enters the initial excess mask when:

```text
relative_thickness(x) >= 1.20
```

Each connected excess island must have at least:

```text
max(3, ceil(0.15 * T50^2)) pixels
```

This scales the minimum proposal area with the component's observed wall
thickness rather than using one global pixel count.

### 6. Directional junction support

For each retained excess island:

1. Compute an excess-weighted center.
2. Convert island area to an equivalent radius.
3. Sample the original component mask in a 36-bin annulus.
4. Bridge one empty angular bin to reduce one-pixel sampling breaks.
5. Retain branch runs at least two angular bins wide.

The initial annulus is:

```text
inner_radius = equivalent_island_radius + 0.50 * T50
outer_radius = equivalent_island_radius + 1.00 * T50
```

Three or more stable arms provide
`thickness_excess_with_multi_arm_junction`. An opposing pair is recorded as
supporting evidence but is not mandatory.

### 7. Multiple-aperture support

A component with at least two already-observed closed apertures may provide
`thickness_excess_with_multiple_apertures` when it also has a retained,
observable excess-thickness island.

The aperture count is shared mask topology evidence. It is not a result from a
multi-gate fitter and does not independently prove overlap.

### 8. Clipped-observation policy

Clipping is handled as an observability question instead of automatically
discarding all review evidence.

A clipped proposal can be retained only when its complete sampling annulus lies
inside the recorded image. Its state is:

```text
assessment = candidate_clipped_review_only
touches_frame = true
routing_eligible = false
```

If the required support intersects unknown space beyond the frame edge, the
assessment remains `indeterminate`. Production behavior continues to reject
clipped components.

### 9. Candidate decision

The current deterministic decision table is:

| Observed evidence | Experimental result |
|---|---|
| Three or more boundary-safe annulus arms | overlap candidate |
| At least two closed apertures plus a boundary-safe thickness island | overlap candidate |
| Clipped candidate with complete observed support | clipped review candidate; never routing eligible |
| Clipped proposal with incomplete support | indeterminate |
| Two thickness populations without junction/aperture support | not an overlap candidate |
| No scale-supported thickness excess | not a candidate |
| Separate connected components | separate observations, not one overlap candidate |

## Retained evidence contract

The review result keeps the following evidence instead of reducing the outcome
to one boolean.

`ComponentThicknessEvidence` retains:

- component identity, bounding box, clipping state, and closed-hole count;
- foreground count, P50/P90/P95/maximum thickness, display cap, and peak ratio;
- excess-pixel/island counts and the scale-dependent minimum island area;
- assessment, terminal reason, evidence scope, candidate state, and
  `routing_eligible`;
- `thickness_split_ready` and the complete population record; and
- all junction records.

Each `ThicknessJunctionEvidence` retains:

- frame-UV and component-local center;
- peak ratio, island area, and area normalized by `T50^2`;
- inner/outer annulus radii;
- occupied angular bins, branch count, angles, and branch widths;
- opposing-pair support;
- whether the complete support was observed; and
- the junction candidate result.

`ThicknessPopulationEvidence` retains:

- availability and selected boundary;
- thinner/thicker counts and fractions;
- both medians and their ratio;
- explained variance; and
- a terminal reason when no split is accepted.

The whole-frame result separately retains distance, covering radius/thickness,
regional/relative/normalized thickness, maximal centers, candidate/core maps,
both population masks, physical bands, adaptive profile IDs and seams, the
composite field, regional P70/P80/P90, and all component records in metrics.

## Two-population segmentation attempt

The corpus-derived, tie-preserving thickness boundaries are:

```text
6.25, 8, 10, 12, 14, 16, 18, 21, 25 px
```

Every boundary that leaves sufficient evidence on both sides is evaluated. The
current minimum support is:

```text
max(24 pixels, 10% of the usable component pixels) per side
```

For each possible split, the implementation measures within-class variance,
explained variance, population balance, and the two medians. The boundary with
the largest `explained_variance * balance` is retained only when:

```text
thicker_median / thinner_median >= 1.25
explained_variance >= 0.35
```

The masks are materialized only when the component is also an experimental
overlap candidate:

```text
thinner_population_mask = T_regional < selected_boundary
thicker_population_mask = T_regional >= selected_boundary
```

The UI renders the selected population in black over the exact white closed
mask. Off-mask pixels are dark gray so the black selection remains visible.

### Known limitation

Covering disks grow at ordinary corners and junctions. A single rectangular
gate can therefore produce several disconnected high-thickness islands. The
two masks currently summarize different local wall scales; they do not yet
assign complete gate ownership.

## Adaptive composite density evidence

The physical thickness bands are descriptive evidence. The production density
profiles are calibrated by component area, so the adaptive selector translates
local thickness back into that native evidence domain:

```text
A_equivalent(x)
  = component.area_px * (T_regional(x) / T50)^2
```

`A_equivalent` is resolved through the existing production
`maximum_component_area_px` limits. The selected profile is initially clamped
to the component's ordinary area profile plus or minus two profile positions.

Each required profile's `DensityEvidence.final_field` is lazily computed and
cached using the complete component mask. The composite is an exact spatial
selection:

```text
adaptive_field(x) = final_field[selected_profile(x)](x)
```

The source fields are not recomputed from the thinner or thicker population
masks. Keeping the complete component as input preserves neighboring support at
an occlusion or profile boundary.

P70, P80, and P90 are then recomputed independently inside each connected
same-component, same-profile region. This prevents a thick region from setting
the percentile threshold for an unrelated thin region.

There is no cross-profile blending yet. White seams in the profile map make
hard selection boundaries explicit.

Profile selection depends only on the closed mask, component area, and
distance-transform-derived thickness. It never uses the resulting density
field or percentile masks to select their own profile; this avoids a circular
calibration rule.

## Current review evidence

### Non-clipped overlap example

`run-20260801T031401Z / frame 106710 / component 3`:

```text
selected split             14.0 px
thinner population         1,229 px
thicker population         1,139 px
thinner median             12.16 px
thicker median             15.23 px
median ratio               1.252
explained variance         0.608
```

The split is measurable, but both populations fragment around corners. This is
the accepted baseline example for distinguishing thickness segmentation from
gate ownership.

### Clipped review example

`run-20260801T031401Z / frame 106888 / component 2` provides a boundary-safe
three-arm proposal and two thickness populations, but visual review shows
non-gate clutter. It is retained as a deliberate false-positive example and
remains `routing_eligible=false`.

A deterministic clipped sample found:

| Sample | Two populations | Boundary-safe candidates | Candidate plus split |
|---|---:|---:|---:|
| 60 clipped components stratified by area | 17 | 0 | 0 |
| 24 one-edge-clipped components with at least two holes | 14 | 4 | 3 |

These counts are discovery measurements, not accuracy estimates. They show
that bimodal thickness is common and cannot independently define overlap.

## Known failure modes

1. Ordinary corners create high-thickness disks.
2. Morphological closing can fabricate a bridge or aperture.
3. Text, reflections, and unrelated foreground can create incidental holes.
4. A clipped boundary can bias thickness unless unobservable support is
   excluded.
5. Two wall scales can exist in one non-overlapping object.
6. One overlapping pair can have nearly equal wall thickness and fail the
   population split.
7. Same-profile regional thresholding can produce small evidence islands.
8. Hard profile seams can interrupt an otherwise continuous density ridge.
9. The current process has no temporal consistency or instance history.
10. Density evidence can strengthen geometry, but it does not assign gate
    ownership by itself.

## Recommended refinement sequence

Change one stage at a time and keep the two pinned examples visible.

### Phase 1: establish labeled review evidence

For every reviewed component, retain:

- run ID, frame ID, simulation time, and component ID;
- clipped status;
- human label: overlap, standard, C-shape, clutter, or uncertain;
- whether the thickness proposal is useful;
- whether the two populations correspond to different visible gate walls; and
- the exact configuration/version used.

Do not tune thresholds from only known-positive overlap examples.

### Phase 2: measure spatial coherence

Add evidence without changing the candidate route first:

- island count for each population;
- largest-island fraction for each population;
- thin/thick shared-interface length;
- distance between population skeletons;
- profile-seam length; and
- fraction of each population supported by stable medial ridges.

This should distinguish a continuous thick wall from four unrelated corner
blobs.

### Phase 3: estimate wall ownership

A practical next experiment is:

1. Extract stable medial-ridge segments.
2. Exclude junction centers and extreme corner radii from the scale fit.
3. Cluster stable ridge radii into at most two wall-scale modes.
4. Propagate the two modes along component-constrained paths.
5. Keep intersection pixels in an explicit shared/ambiguous region.
6. Compare the propagated regions with aperture contours and straight-line
   support.

This is more defensible than assigning every high-thickness corner directly to
the thicker gate.

### Phase 4: introduce a scored proposal only after labels exist

A possible explainable structure is:

```text
population_score
  = observability
  * thickness_contrast
  * population_balance
  * spatial_coherence

overlap_score
  = observability
  * topology_support
  * (0.5 + 0.5 * population_score)
```

Keep all terms in the review evidence. Do not add an opaque aggregate score
until the individual terms have been compared against labeled failures.

### Phase 5: evaluate geometry usefulness

Compare, on the same component:

1. the ordinary single area-selected density profile;
2. the adaptive composite field;
3. regional P70, P80, and P90 masks; and
4. any wall-ownership segmentation.

The useful outcome is improved recovery of independent straight sides or
apertures, not merely a more colorful field.

### Phase 6: production promotion requirements

Promotion requires a separate approved change. At minimum:

- move accepted controls into the code-owned configuration schema;
- add schema-owned evidence and machine-readable rejection reasons;
- place shared thickness computation outside specialized fitters;
- keep topology routing ahead of gate-specific fitting;
- retain production rejection of clipped components;
- define how an adaptive multi-profile field is represented without pretending
  it is one ordinary `DensityEvidence` profile;
- establish labeled regression cases and runtime budgets; and
- ensure production never imports this review folder.

## Review procedure

Launch from the repository root:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.local_thickness.server
```

Recommended initial frames:

- `run-20260801T031401Z / 106710`: non-clipped overlap proposal and split.
- `run-20260801T031401Z / 106888`: clipped, review-only false-positive example.
- `run-20260801T031401Z / 106765`: several components at different scales.

For each change, review:

1. the exact closed mask and component labels;
2. raw and regional thickness;
3. empirical thickness bands;
4. thinner and thicker black-on-white masks;
5. candidate reason, clipping state, and supporting junctions;
6. adaptive profile seams and final field; and
7. regional P70/P80/P90 continuity.

Then run:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m unittest discover \
  -s Flight/src/sensing/vision/models/deterministic_v3/src/profile_calibration \
  -t Flight/src -v
```

## Implementation map

- `analysis.py`: thickness, proposals, population split, adaptive composite,
  regional percentile evidence, and rendered panels.
- `server.py`: recorded-run adapter and read-only review API.
- `ui/`: static controls and visualization surface.
- `tests/test_analysis.py`: focused deterministic invariants.
- `OVERLAP_CANDIDATE_EXPLORATION.md`: prior corpus sweep and observations.
- `assets/`: pinned visual examples; never imported by production.
