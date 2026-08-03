# Clipped-gate historic review scaffold

This directory is an isolated, offline-only first implementation for components
that the historic deterministic-v3 review records already classify as
`clipped`. It does not register a topology route, modify live density
eligibility, call PnP, or participate in live inference. Its quadrilaterals are
unvalidated visual-review trials, not pipeline results.

The historic `TopologyDecision.topology_label == "clipped"` value is trusted as
authoritative input. This tool does not retest or reclassify clipping.

## Files

- `process.py` isolates one component from the serialized frame masks and emits
  its clipping pattern, touched sides, raw/post-close boundary contacts, and
  compact mask row runs.
- `density_fit.py` reconstructs named density variants for clipped masks,
  performs an isolated four-line P90 quadrilateral trial, and emits explicit
  provenance and review-only scoring.
- `review_json.py` decodes historic review JSON without importing the runtime
  pipeline or UI backend, invokes both processing stages, and writes one review
  document.
- `review/index.html` is a self-contained static viewer for the generated review
  document.

## Run the historic JSON loader

From the repository root:

```sh
/usr/local/bin/python3.13 \
  Flight/src/sensing/vision/models/deterministic_v3/src/clipped_gate/review_json.py \
  Flight/src/sensing/vision/models/deterministic_v3/ui/review_runs/run-20260731T093159Z \
  --output /tmp/clipped-gate-review.json \
  --limit 100 \
  --pretty
```

Inputs may be individual historic frame JSON files, a `frames/` directory, or a
review-run directory containing `frames/`. `--limit` limits the number of
clipped component predictions, not the number of input frames scanned.

The loader consumes these existing records:

- one `FrameObservation`, including `size_filtered_mask`, `closed_mask`, and
  `component_labels`;
- its nested `ComponentObservation` values;
- root `TopologyDecision` values whose topology label is exactly `clipped`.
- the recorded `DensityBankConfiguration` and any component-owned
  `DensityEvidence` records that are present.

Historic review dumps normally contain no `DensityEvidence` for clipped
components because the live density bank's recorded eligibility policy excludes
them. In that case, `density_fit.py` evaluates every profile from the serialized
configuration. Its calculation follows the shared density-bank equations while
using an observed-frame-domain denominator so pixels beyond the camera frame are
treated as unknown rather than background. Each result says whether its field
was serialized or reconstructed and records the boundary rule used.

Malformed source documents are listed under `errors` and skipped. Pass
`--strict` to stop at the first malformed input.

## Review the output

Open `review/index.html` in a browser and choose the generated
`clipped-gate-review.json` file. The page operates entirely in the browser and
does not require a server.

When `review/clipped-gate-review.json` exists and the directory is served over
HTTP, the page loads that result set automatically. The generated file is
ignored by Git.

The canvas displays:

- the post-close clipped component mask in dark red;
- original size-filtered mask pixels in bright red;
- pixels added by the close operation in cyan;
- the component bounding box in yellow;
- authoritative clipped frame sides and their contact runs in blue/green.
- the selected reconstructed density field as a turbo heat map;
- selected P90 evidence in white;
- the first-pass fitted quadrilateral as a dashed orange outline.

The `density`, `P90`, and `quad` controls can be toggled independently. The
right-hand details panel includes all ten profile trials without embedding the
large PNG payload in the displayed text.

## Current prediction boundary

`clipped-mask-status-v1` predicts only the observed clipping pattern:

- `single_edge`;
- `adjacent_corner`;
- `opposite_edges`;
- `multiple_edges`;
- `edge_unspecified`.

Every result remains `routing_eligible=false`, `pose_eligible=false`,
`fitter_ready=false`, and `promotion_status=not_evaluated`. The density stage
fits four robust lines to the selected profile's P90 hull, allows line
intersections to fall outside the image, and reports a ranking score only for
review. It does not apply the official in-frame quadrilateral contract and its
candidate is not a gate pose or public vision observation.
