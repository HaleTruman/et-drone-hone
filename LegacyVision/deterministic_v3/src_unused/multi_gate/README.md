# Multi-gate production-local scaffold

This folder owns the specialized overlap proposal and two-aperture fitting
work required after shared preprocessing. It does not own JPEG decoding,
connected components, the density-bank math, PnP, NED conversion, or review
artifacts.

## Current flow

```text
FrameObservation + all TopologyDecision records + frame DensityBank
  -> identification.identify_multi_gate_candidates
       assess every component
       retain local-thickness, junction, aperture, population, and density data
       never replace the authoritative topology route
  -> process.process_multi_gate
       require accepted multi_gate / multi_void topology
       require exactly two topology-owned aperture contours
       consume one cached named density profile
       fit larger aperture first and smaller aperture on residual P90 support
  -> process.quadrilaterals_from_multi_gate
       publish gate_index 0 and 1
  -> shared camera PnP
```

`experimental_overlap_candidate`, `routing_eligible`, and `fitter_ready` are
separate states. A non-clipped thickness candidate is only fitter-ready when
the existing topology decision also selected `multi_gate` and supplied exactly
two significant child apertures. A fitter rejection never falls through to a
standard or C-shape fit.

The live configuration records clipped components as `indeterminate` without
running their expensive covering-disk analysis because they cannot route.
`REVIEW_MULTI_GATE_IDENTIFICATION_CONFIGURATION` enables the complete
boundary-safe clipped methodology when offline review evidence is required.

The identifier is a production-safe port of the deterministic waterfall in
`profile_calibration/local_thickness/OVERLAP_IDENTIFICATION_ITERATION_GUIDE.md`.
Production code does not import that calibration/review package. Thickness
populations remain named `thinner` and `thicker`; they are not gate ownership.
The density-overlap record is diagnostic and does not currently change the
candidate decision.

## Historical checkpoints

`overlapping_gate_aperture_solver.py` and
`overlapping_gate_contour_side_refinement.py` remain untouched historical
checkpoints. `aperture_fitting.py` is a narrow transition adapter around the
approved coverage kernel. It supplies the topology-owned aperture contours so
the fitter does not rediscover topology internally.

## Explicit first-scaffold limits

- The topology route `one_child_plus_large_opening` cannot yet be fit by the
  two-aperture kernel and returns
  `multi_gate_fitter_requires_two_aperture_seeds`.
- The adaptive multi-profile density mosaic remains discovery evidence. The
  fitter consumes one named, cached density profile under the shared schema.
- Outer-contour side refinement has not yet been extracted from its historical
  review dependency into a production-only helper.
- No union-coverage acceptance threshold is imposed yet; the complete metric
  is retained for labeled calibration.
- Gate identity is frame-local `(component_id, gate_index)` until temporal
  association assigns a persistent identity downstream.
