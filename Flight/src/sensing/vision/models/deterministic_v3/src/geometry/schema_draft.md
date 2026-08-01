# user notes none-authorative


The major overlap is everything through component extraction, topology, contours, density fields, and percentile evidence. The implementations, diverge when they interpret that evidence: multi-gate fits two complete apertures, while C-shape identifies three visible sides and constructs the missing fourth side.

  ## 1. Current multi-gate operations

  The accepted implementation is split between Flight/src/sensing/vision/
  models/deterministic_v3/src/geometry/
  overlapping_gate_aperture_solver.py:325 and Flight/src/sensing/vision/
  models/deterministic_v3/src/geometry/
  overlapping_gate_contour_side_refinement.py:258.

  ### Frame and mask preparation

  1. Convert the image to a LUT mask.
  2. Run connected components.
  3. Gate frames containing at least 500 components.
  4. Remove components below 100 pixels.
  5. Apply the 5×5 close.
  6. Isolate one connected parent component.
  7. Create a square component-local mask and record its frame origin.
  8. Select a density/ridge profile from maximum component dimension.

  ### Density evidence

  9. Compute square-neighborhood pixel density.
  10. Normalize density relative to its component foreground mean.
  11. Apply inverse gamma 2.10.
  12. Compute the local weighted centroid/ridge multiplier.
  13. Apply ridge gamma 3.00.
  14. Produce the final inverse-density field.
  15. Extract the P90 evidence mask.
  16. Extract the P70 support/limit mask.
  17. Compute the optional 3×3 constrained close, although the accepted
     broader path primarily consumes the P90 seed and P70 limit.

  ### Multi-aperture topology

  18. Extract contour hierarchy using RETR_CCOMP.
  19. Select holes with at least 12 pixels of contour area.
  20. Require exactly two holes in the same parent component.
  21. Sort the apertures largest-first.
  22. Build an initial convex quadrilateral for each aperture:

  - Convex hull and four-sided approximation for reliable apertures.
  - Minimum-area rectangle fallback for small or concave apertures.

  ### Initial two-gate fitting

  23. Fit the larger aperture first.
  24. Search outward from each aperture side for parallel density support.
  25. Measure density strength, P90 coverage, continuity, and foreground
     coverage.

  26. Record compatibility with previously fitted parallel sides.
  27. Intersect adjacent selected lines to form a quadrilateral.
  28. Fit the smaller aperture with the larger gate’s side evidence
     available as compatibility context.

  ### Bounded coverage optimization

  29. Generate small scale, rotation, and translation variations.
  30. Require the candidate to contain its aperture seed.
  31. Reject candidates outside the component-local image.
  32. Measure weighted P90 side coverage.
  33. Measure minimum and mean per-side coverage.
  34. Identify short gaps along each proposed side.
  35. Allow gaps up to three pixels to bridge when supported by P70.
  36. Select the larger-gate candidate first.
  37. Remove P90 evidence covered by the larger candidate.
  38. Fit the smaller gate against the residual evidence.
  39. Calculate union coverage for the two gates.

  ### Outer-contour post-refinement

  40. Extract the parent’s outer contour.
  41. Search for contiguous outer-contour runs near each proposed gate
     side.

  42. Fit robust local contour lines using cv2.fitLine(..., DIST_HUBER).
  43. Require minimum angular agreement, contour span, and point count.
  44. Compare zero contour influence with 0.50 influence.
  45. Cap angular modification at 15 degrees.
  46. Keep each density-line midpoint fixed while modifying its direction.
  47. Intersect the adjusted lines.
  48. Require convexity, image bounds, and containment of the original
     aperture.

  49. Prevent P90 coverage from regressing by more than the approved
     slack.

  50. Select the candidate using P90 coverage plus a small contour-
     alignment contribution.

  51. Again refine the larger gate first and the smaller gate against
     residual evidence.

  ———

  ## 2. Current C-shape operations

  The accepted result is exposed by Flight/src/sensing/vision/models/
  deterministic_v3/src/geometry/
  c_shape_tailored_shortfall_baseline.py:110.

  ### Component and density input

  1. Consume one component-local mask.
  2. Consume one selected inverse-density field.
  3. Retain the component-to-frame coordinate origin.
  4. Extract positive density values.
  5. Calculate the P90 threshold.
  6. Extract P90 evidence-point coordinates.

  ### Three-visible-side discovery

  7. Fit an initial convex P90 quadrilateral scaffold.
  8. Measure every P90 point’s distance to the four scaffold sides.
  9. Assign each point to its nearest side.
  10. Count support for all four sides.
  11. Treat the least-supported side as missing.
  12. Require at least two points on each of the remaining three sides.
  13. Fit the three visible lines independently with cv2.fitLine().

  ### Finite component geometry

  14. Identify the visible spine opposite the missing side.
  15. Identify the two protruding arms adjacent to the missing side.
  16. Intersect each arm with the spine.
  17. Compute the component distance transform.
  18. Sample the distance transform along the spine.
  19. Use the median positive distance as the mask half-width.
  20. Measure the visible extent of both arms.

  ### Outer-contour evidence

  21. Extract the component’s external contour.
  22. Simplify it by three pixels.
  23. Densify the simplified contour back into approximately pixel-spaced
     samples.

  24. Find contour points adjacent to each visible P90 line.
  25. Separate those points into the two outer rails around that line.
  26. Fit a line to each contour rail.
  27. Estimate the parallel consensus direction of the paired rails.
  28. Preserve the P90-derived center offset.

  ### Bounded angular correction

  29. Compare each P90 direction with its contour direction.
  30. Apply 0.50 of the angular difference.
  31. Cap the correction at 15 degrees.
  32. Rebuild all three visible lines with the corrected directions.
  33. Recompute their spine/arm intersections and extents.

  ### Missing-side construction

  34. Cast each adjusted arm outward from its spine intersection.
  35. Intersect the ray with the simplified outer contour.
  36. Use those intersections as the final observed green exits.
  37. Measure the two green arm lengths.
  38. Measure the green spine length.
  39. Find the longest of those three green lengths.
  40. Set the target length to 0.75 × longest.
  41. For each arm, calculate:

  extension = max(0, target − observed_arm_length)

  42. Extend each arm collinearly by its individual shortfall.
  43. Join the two exterior endpoints to construct the missing side.
  44. Assemble the completed quadrilateral.
  45. Record convexity and area.

  ———

  ## 3. Operations that overlap

  These are strong candidates for one shared component feature cache.

  ### Exact shared computations

  - Frame and component identity.
  - Bounding box and component-to-frame origin.
  - Frame-edge contact.
  - Raw component mask.
  - Size-filtered component mask.
  - Closed component mask.
  - Connected-component labels and statistics.
  - Contour hierarchy.
  - Outer contour.
  - Hole contours and hole areas.
  - Simplified outer contour.
  - Component area, dimensions, fill ratio, and solidity.
  - Density profile definitions.
  - Component-local density fields at multiple radii.
  - Positive density coordinates and values.
  - P70, P80, and P90 thresholds.
  - P70/P80/P90 binary evidence masks.
  - Weighted evidence-point arrays.
  - Distance transform.
  - Line sampling.
  - Point-to-segment distances.
  - Line fitting.
  - Line intersection.
  - Quadrilateral ordering.
  - Convexity, area, bounds, and self-intersection validation.

  ### Shared concepts with different application

   Operation             C-shape                   Multi-gate
  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━
   Topology              Expects zero stable       Requires exactly two
                         apertures                 apertures
  ────────────────────  ────────────────────────  ────────────────────────
   P90 evidence          Divided among three       Divided between two
                         visible sides             complete gates
  ────────────────────  ────────────────────────  ────────────────────────
   Contour guidance      Paired rails influence    Local outer contour
                         three visible             influences up to eight
                         directions                sides
  ────────────────────  ────────────────────────  ────────────────────────
   Density profile       Needs a profile           Needs a profile
                         effective for open/       effective for merged
                         occluded structure        two-aperture evidence
  ────────────────────  ────────────────────────  ────────────────────────
   Line fitting          Fits three visible        Fits four sides per
                         lines directly            aperture
  ────────────────────  ────────────────────────  ────────────────────────
   Evidence ownership    Three side assignments    Larger gate first,
                         in one object             smaller gate gets
                                                   residual evidence
  ────────────────────  ────────────────────────  ────────────────────────
   Missing geometry      Constructs one            Does not intentionally
                         unobserved side           synthesize a missing
                                                   side
  ────────────────────  ────────────────────────  ────────────────────────
   Completion            Individual arm            Bounded scale/
                         shortfall extensions      rotation/translation
                                                   search
  ────────────────────  ────────────────────────  ────────────────────────
   Validation            Convex completed          Two convex quads
                         quadrilateral             containing their
                                                   aperture seeds

  ### Operations that should remain specialized

  C-shape only:

  - Selecting the least-supported side as missing.
  - Spine and protruding-arm identification.
  - Distance-transform half-width sampling along the spine.
  - Ray-to-outer-contour exit calculation.
  - The 0.75 × longest green length extension rule.
  - Constructing an unobserved fourth side.

  Multi-gate only:

  - Requiring and ordering exactly two apertures.
  - Larger-gate-first evidence ownership.
  - Parallel compatibility between gates.
  - P90 seed with P70 gap support.
  - Short side-gap bridging.
  - Bounded scale/rotation/translation candidate search.
  - Residual P90 evidence for the smaller gate.
  - Weighted union coverage across two gates.

  #
```text
RAW IMAGE FRAME
│
▼
1. SHARED FRAME PREPROCESSING
│  ├── Apply LUT → base mask
│  ├── Run initial connected components
│  ├── Apply 500-component frame gate
│  ├── Remove components below 100 px
│  ├── Apply 5×5 close
│  └── Run final connected components
│
▼
2. CREATE FrameObservation
│  ├── frame_id
│  ├── image_shape
│  ├── preprocessing_version
│  ├── base_mask
│  ├── size_filtered_mask
│  ├── closed_mask
│  └── component_labels
│
▼
3. CREATE ONE ComponentObservation PER COMPONENT
│  ├── component_id
│  ├── bounding box and frame origin
│  ├── area and dimensions
│  ├── frame-edge contact
│  ├── raw-mask view
│  ├── closed-mask view
│  ├── raw and closed contour hierarchy
│  ├── raw and closed aperture counts
│  ├── outer contour
│  ├── simplified outer contour
│  ├── fill ratio and solidity
│  ├── distance transform
│  └── density-bank handle
│
▼
4. TOPOLOGY STABILITY CHECK
│
├── Raw and closed aperture counts disagree
│      └── AMBIGUOUS / diagnostic review
│
├── Component touches frame boundary
│      └── CLIPPED / partial-observation process
│
└── Stable topology
       │
       ▼
5. TOPOLOGY ROUTER
       │
       ├── One enclosed aperture
       │      └── STANDARD SINGLE-GATE
       │
       ├── Zero enclosed apertures
       │      ├── Strong three-sided/opening evidence
       │      │      └── C-SHAPE
       │      └── Weak or contradictory evidence
       │             └── AMBIGUOUS
       │
       ├── Exactly two enclosed apertures
       │      └── MULTI-GATE
       │
       └── More than two apertures
              └── UNSUPPORTED / diagnostic review
```

After routing, density selection branches independently:

```text
                     COMPONENT TOPOLOGY
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   STANDARD GATE          C-SHAPE           MULTI-GATE
          │                  │                  │
          ▼                  ▼                  ▼
 Standard density      C-shape density     Multi-gate density
 profile selector      profile selector    profile selector
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                       SHARED DENSITY BANK
                ┌────────┬────────┬────────┬────────┐
                │  r2    │  r7    │  r11   │ r16/r20│
                └────────┴────────┴────────┴────────┘
``