# Whole-frame Local Thickness Discovery

This is a one-off review instrument inside the isolated `profile_calibration`
folder. It reads recorded JPEGs, invokes production preprocessing and
`DensityBank`, and never writes configuration or participates in the runtime
pipeline.

It deliberately separates these concepts:

1. `distance_to_background_px` is the padded, precise L2 distance transform of
   each post-close component. Away from a medial center it is not wall
   thickness.
2. `covering_local_thickness_px` is twice the radius of the largest retained
   foreground disk that contains each pixel. This spreads a meaningful local
   thickness estimate across the complete wall.
3. `FrameObservation.component_labels` shows the exact connected-component
   segmentation in the original frame UV. The thickness computation was
   component-local before this panel was added; this makes that boundary
   visible.
4. `regional_local_thickness_px` applies a component-constrained normalized
   Gaussian (`sigma=1 px`) before classification, so background and neighboring
   connected components cannot leak into a region.
5. `component_relative_thickness` divides every component's regional thickness
   by that component's median and uses one fixed `0..2x` display scale. This
   prevents a thick near component from visually suppressing a thin far
   component without changing the absolute pixel evidence.
6. `thickness_band_index` applies the tie-preserving physical boundaries
   `6.25, 8, 10, 12, 14, 16, 18, 21, 25 px`. This is descriptive thickness
   evidence and is not renamed as the area-calibrated production profile deck.
7. `experimental adaptive profile_id map` converts each regional thickness to
   `component.area_px * (local_thickness / component_P50)^2`, resolves that
   equivalent area through the current production limits, and clamps selection
   to the component's ordinary area profile plus or minus two profiles.
8. `adaptive composite of DensityEvidence.final_field` lazily computes each
   selected production field at most once per component and splices it using
   the experimental map. P70/P80/P90 are then recomputed independently inside
   each connected same-profile region.
9. `thinner_population_mask` and `thicker_population_mask` are a deterministic
   two-class attempt using the physical band boundaries. Both sides require at
   least 10%/24 pixels, `1.25x` median separation, and 35% explained variance.
   They are rendered as black selections over the exact white closed mask.

The profile map is not the production recommendation policy. Production still
selects a profile provisionally from whole-component area. The adaptive mosaic
can contain hard seams, and classical covering-disk thickness becomes large at
corners and overlap junctions by definition. Those behaviors are evidence to
review, not accepted pipeline behavior.

Nonzero maximal-center smoothing is an optional discovery regularizer. The
unmodified geometric reference is sigma `0.0`.

One `include clipped components` toggle controls the adaptive-density path.
It defaults on so nothing is filtered: clipped components receive profile
assignments and adaptive fields. Turning it off makes them gray and excludes
them from the adaptive field, matching production density eligibility.

The separate `show experimental overlap_candidate flags` toggle only changes
the review overlay. The server computes the evidence regardless of display
state. For each component, the current first-pass rule uses median
covering thickness as its local reference and proposes excess islands at
`1.20x` that reference. Islands smaller than
`max(3, ceil(0.15 * median_thickness_px^2))` are rejected. A retained island
becomes a candidate when a scale-relative annulus observes at least three
stable foreground arms. A component with two or more already-observed closed
apertures also becomes a candidate when it has a retained local-thickness
island; this supporting aperture count is generic mask structure, not a
gate-solver result. A clipped proposal is retained only when its complete
supporting annulus lies inside the recorded frame. It remains
`routing_eligible=false` and is magenta; incomplete clipped evidence remains
`indeterminate`. Orange outlines are complete candidates and white
outlines/markers are the supporting thickness regions.

This is deliberately a candidate generator, not an overlap classifier. A
multi-arm non-gate junction can be flagged, a close operation can fabricate a
bridge, and a true overlap with no measurable thickness excess can be missed.
The two thickness populations are likewise not gate identities: ordinary
rectangular corners can form a disconnected high-thickness population.
Exact per-component metrics and terminal reasons remain in
`ThicknessAnalysis.component_evidence` for offline review.

The corpus results, known-positive behavior, false-positive observations, and
timing are recorded in
[`OVERLAP_CANDIDATE_EXPLORATION.md`](OVERLAP_CANDIDATE_EXPLORATION.md).
The reproducible stage contract, decision table, known limitations, and
recommended refinement sequence are recorded in
[`OVERLAP_IDENTIFICATION_ITERATION_GUIDE.md`](OVERLAP_IDENTIFICATION_ITERATION_GUIDE.md).

## Launch

From the repository root:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.local_thickness.server
```

Open <http://127.0.0.1:8785/>. The default evidence frame is
`run-20260801T031401Z / 106765`, which contains four non-clipped gate components
with substantially different wall thicknesses.
