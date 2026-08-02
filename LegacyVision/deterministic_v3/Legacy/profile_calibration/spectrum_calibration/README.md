# Ten-column density spectrum calibration

This is a temporary, isolated review instrument. It is not imported by the
production geometry pipeline or by either existing calibration UI.

The asset builder replays the latest three valid historic runs through the
production preprocessing API with one local review configuration:

- `minimum_component_area_px = 250` before the existing close operation;
- `ComponentObservation.touches_frame = true` records are excluded after the
  post-close connected-component pass;
- every retained post-close component is stored exactly once;
- retained components are sorted by `ComponentObservation.area_px` descending
  and assigned to ten balanced rank columns. The persisted rank assignment is
  authoritative because equal-area records can cross a column boundary.

The browser shows the largest rank column on the left and the smallest on the
right. Each component contains the exact saved source analysis crop, the
persisted closed mask, the resolved candidate input mask, and candidate
`DensityEvidence.final_field`, `p70_mask`, `p80_mask`, and `p90_mask`.
Candidate evidence is recomputed by the server via the production
`DensityBank`; JavaScript performs no image processing.

The left and right endpoint controls independently set
`density_radius_px`, `ridge_radius_px`, `relative_cap`, `inverse_gamma`,
`ridge_gamma`, `open_kernel_radius_px`, and `close_kernel_radius_px` for the
largest and smallest columns. The server linearly resolves columns two through
nine by ordinal position. Integer radii use half-up rounding. The normalization
cap retains the original calibrator's `1.00` through `5.00` range and `0.05`
step.

The two morphology controls are experimental candidate-input operations. A
radius of `0` disables that operation; radius `1` means a `3x3` rectangular
kernel, radius `2` means `5x5`, and so on. The server zero-pads the persisted
closed component mask, applies open and then close, crops back to the analysis
square, and passes that exact candidate mask to the production `DensityBank`.
Consequently `final_field`, thresholds, and `p70_mask`/`p80_mask`/`p90_mask`
are all recomputed from the same support. The original closed mask and the
candidate input mask remain separate panels. The initial morphology endpoints
are radius `1` for the largest column and `0` for the smallest column; the
normalization endpoints retain the current `2.00` baseline.

These are experimental candidate settings and do not modify
`configurations.py`.

## Build and launch

From the repository root:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.spectrum_calibration.build_assets
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.spectrum_calibration.server \
  --port 8786
```

Then open `http://127.0.0.1:8786/`.

The collection endpoint returns one synchronized rank band with up to 100
records per column. The browser appends bands until every manifest record is
available and only requests candidate images for cards near the viewport.
