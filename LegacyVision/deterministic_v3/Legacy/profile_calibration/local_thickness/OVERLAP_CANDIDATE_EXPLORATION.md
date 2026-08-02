# Component-local Thickness Overlap Exploration

Status: isolated review experiment, 2026-08-02. This evidence is not imported
by the deterministic-v3 production pipeline and does not define an accepted
topology label.

## Question

Can covering local thickness be normalized independently inside every
post-close connected component, displayed in one frame UV, and used to propose
components that may contain overlapping gates?

## Implemented evidence

- Absolute `covering_local_thickness_px` remains physical pixel evidence.
- `component_relative_thickness` is `thickness_px / component P50`, rendered on
  one fixed `0..2x` scale across the frame.
- `component_local_normalized_thickness` is separately retained as
  `clip(thickness_px / component P99, 0, 1)`.
- `FrameObservation.component_labels` is rendered as a deterministic full-frame
  segmentation so component ownership is visible.
- Clipped components retain review maps. A proposal is review-eligible only
  when its complete supporting annulus is inside the recorded frame; it remains
  `routing_eligible=false`. Incomplete clipped support is `indeterminate`.

The candidate rule starts with islands whose thickness is at least `1.20x` the
owning component's median. Islands smaller than
`max(3, ceil(0.15 * median_thickness_px^2))` are rejected. A retained island is
accepted through either of two explicit reasons:

1. `thickness_excess_with_multi_arm_junction`: a scale-relative 36-bin annulus
   observes at least three stable foreground arms. An opposing arm pair is
   recorded as supporting evidence but is not mandatory because the reviewed
   `106710:3` junction has three roughly 120-degree arms.
2. `thickness_excess_with_multiple_apertures`: the component has at least two
   already-observed closed holes and a scale-supported thickness island. This
   is generic mask structure, not a gate-solver output.

The second path is deliberately visible in the reason string. It is necessary
for current recall, but proves that local thickness alone is not sufficient for
all reviewed overlaps.

## Local calibration and clipped-review extension

The 2026-08-02 extension adds the measured physical thickness boundaries
`6.25, 8, 10, 12, 14, 16, 18, 21, 25 px`, a deterministic two-population
proposal, and an area-equivalent adaptive density mosaic. Physical bands remain
separate from the area-calibrated profile IDs. Local profile selection uses:

```text
equivalent_area(x)
  = component.area_px * (regional_thickness(x) / component_P50)^2
```

The selected profile is clamped to the component's normal area profile plus or
minus two positions. P70/P80/P90 are recomputed within each connected
same-profile region after the ten cached full-component fields are composed.

A deterministic clipped sample found:

| Sample | Two thickness populations | Boundary-safe overlap candidates | Candidate + split |
|---|---:|---:|---:|
| 60 clipped components stratified by area | 17 | 0 | 0 |
| 24 one-edge-clipped components with at least two holes | 14 | 4 | 3 |

This confirms that a two-thickness split is common evidence, not an overlap
decision. The black-on-white masks are named thinner/thicker populations and
must not be treated as gate instance ownership without contour/aperture support.

## Recorded-run sweep

The rule was evaluated on the newest three available runs: 1,827 frames and
5,315 post-close components. Of those components, 4,170 were non-clipped and
1,145 were clipped.

| Run | Frames | Candidate components | Frames with candidates |
|---|---:|---:|---:|
| `run-20260731T093616Z` | 692 | 210 | 203 |
| `run-20260731T093732Z` | 691 | 190 | 174 |
| `run-20260801T031401Z` | 444 | 102 | 102 |
| **Total** | **1,827** | **502** | **479** |

The 502 candidates are 12.04% of non-clipped components. Reasons split into
255 multi-arm candidates and 247 multiple-aperture candidates.

Known review cases:

| Component | Result | Evidence path |
|---|---|---|
| `106710:3` | candidate | 3 arms at approximately 120, 235, and 355 degrees |
| `107025:1` | candidate | two closed holes plus supported thickness excess; annulus saw 1 arm |
| `107026:1` | candidate | two closed holes plus supported thickness excess; annulus saw 1 arm |

As a rough specificity proxy, 57 of 2,293 one-hole components were flagged
(2.49%), no zero-hole component was flagged, 406 of 467 two-hole components
were flagged, and 39 of 51 components with three or more holes were flagged.
These are not ground-truth accuracy figures.

## Findings and limits

- Component-local computation and normalization are behaving correctly:
  ownership, finite-value, mask-containment, P50-relative, and P99-normalized
  invariants passed across the corpus.
- A strong multi-arm thickness junction is useful proposal evidence and rejects
  ordinary one- and two-arm endpoints/corners in focused synthetic tests.
- Thickness geometry alone did not recover `107025:1` or `107026:1`; both need
  the contextual multiple-aperture path in this first implementation.
- Incidental holes remain a known false-positive source. In frames `106765`
  and `106932`, contaminating foreground divides a visually standard gate and
  activates the multiple-aperture path.
- A persistent standard-looking sequence also produced three-arm annulus
  proposals. Therefore the flag must remain `experimental_overlap_candidate`,
  not `overlap_detected` or a routing decision.
- Classical covering disks create round plateaus at ordinary corners as well
  as true junctions. The component-relative display improves comparison but
  does not make every red region independent evidence.

## Timing observation

For the complete review `analyze_frame` path, mean time was 190.53 ms/frame
(5.25 Hz), median 43.59 ms, P95 1,094.47 ms, and maximum 4,154.78 ms.
Preprocessing added 4.77 ms/frame on average. Large clipped components dominate
the tail because their covering-radius maps are still computed for display
before adaptive density is skipped. These are discovery-UI timings, not the
cost of a proposed production overlap flag.
