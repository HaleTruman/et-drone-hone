# Shared Schema Review Runs

The replay adapter writes replaceable offline output under
`Logs/review/runs/`.
That directory is ignored by Git and is never read by production inference.

From the project root, replay a flight run with:

```sh
python LegacyVision/run_review_pipeline.py Logs/flight/runs/<run-id> --force
```

An explicit run path, `--path`, `--output-root`, `--limit`, or `--force`
may be supplied. The adapter reads
`frames.jsonl`, preserves its `frame_id` and `sim_time_ns`, and supplies the
recorded JPEG bytes through the same shared production functions used by live
ingress.

Each output run contains `manifest.json` plus one review-format-v8 JSON
document per frame. The current review frontier includes the complete
`FrameObservation`, nested component/contour/topology fields, every
`TopologyDecision`, the exact `DensityBankConfiguration`, and all ten
`DensityEvidence` profiles for every density-eligible component. It records
the exact `StandardGateConfiguration` and one accepted or rejected
`StandardGateResult` for every component. Those root records are lossless
copies of the canonical tuple retained in
`GeometryFrameResult.standard_gate_results`; tests require field-for-field
equality rather than a UI reconstruction. Each standard result preserves its
selected `DensityProfile`, P90 fitter evidence, fitted corners, confidence,
threshold, acceptance, and rejection reason. The matching selected-profile
`final_field`, `p70_mask`, `p80_mask`, and `p90_mask` remain in their original
`DensityEvidence` record.

The replay also records the exact `CShapeConfiguration` and one `CShapeResult`
for every accepted `c_shape` route. Each result retains the selected profile,
three refined lines and construction geometry when the approved fitter
succeeds, a completed `refined_mask` including the inferred
endpoint-to-endpoint side, or an explicit fitter rejection reason.
Frame-edge-clipped components remain represented by their component,
topology, and rejected standard-gate records but produce no density evidence
by default. Standard, C-shape, and multi-gate quadrilaterals and
camera-relative PnP records are retained through the production
`GeometryFrameResult`; specialized evidence that is not a root schema record
remains nested there.

A run-level production frontier is written to
`Logs/review/runs/<run-id>/gate-geometry-pnp-runtime.json`. Its `frames` entries
use run-level format 2 and carry exact JSON-compatible `GeometryFrameResult`
values under
`runtime_result`, including `camera_calibration`, `gate_model`,
`quadrilateral_estimates`, `pnp_relative_pose_estimates`, and
`camera_pose_estimates`. Every raw `PnPRelativePoseEstimate` retains all finite
positive-depth ranked `candidates` and its exact `selected_candidate_rank`.
The authoritative final `camera_pose_estimates` tuple is populated only when
the recorded vision frame joins exactly to recorded telemetry by `inner_cycle`;
otherwise it remains empty rather than assuming stationary camera motion. The UI backend matches
each entry to a logged image only when both `frame_id` and `sim_time_ns` agree,
then serves that original frame entry without reshaping it. This aggregate does
not substitute for per-frame `FrameObservation` or `DensityEvidence` records;
tabs requiring mask arrays or `ComponentObservation.bbox_xywh` remain limited
to the per-frame replay contract.

JSON has no native non-finite number representation. If a rejected runtime
record legitimately contains `inf`, `-inf`, or `nan`, the aggregate retains it
with the same explicit `{"__float__":"..."}` envelope used by the lossless
runtime-record serializer. It is never converted to `null` or omitted.

The read-only endpoint `/api/runs/<run-id>/standard-gate-results` joins each
canonical run-level result to its per-frame evidence only by exact
`frame_id`, `sim_time_ns`, `component_id`, and selected `profile_id`. Its layer
endpoints isolate the selected component from `FrameObservation.closed_mask`
using `component_labels` and expose only the already serialized selected-profile
density arrays. Missing or inconsistent provenance fails closed; the backend
does not refit, substitute a profile, or fabricate P70/P80 evidence.

The compact StandardGateResult frontend currently requests only the isolated
closed-mask layer and, when fitted corners exist, the matching logged source
frame needed for the quadrilateral overlay. Density arrays remain available to
the primary DensityEvidence viewer and are not requested by that compact tab.

The read-only UI endpoint
`/api/runs/<run-id>/pnp-world-replay` joins that unchanged runtime aggregate to
logged `vision_frames.jsonl` and `telemetry.jsonl` records by exact
`frame_id`/`sim_time_ns` plus `inner_cycle`. It preserves every source runtime
result and places all derived camera-path and gate values under
`ui_projection`. Selected raw PnP candidates and final poses receive LOCAL_NED
projections; secondary candidates receive an optional current-frame projection
but are deliberately excluded from history. The current review configuration accepts at most 100 ms
alignment error, requires logged camera-mount values to match production
`src/void_ned.py`, and marks rendered gate orientation as provisional. Failure
of any gate suppresses world placement with a serialized reason while retaining
the logged vehicle trajectory. The adapter does not rewrite review artifacts or
alter replay/runtime classes.

Both `standard_gate` and `c_shape` use the same `QuadrilateralEstimate` corner
order, shared validation, camera calibration, 2.10 m gate model, and
`gate_centerline_pnp.solve_gate_pose()` implementation. Route provenance remains
explicit on every quadrilateral and pose record.

`TopologyDecision` records are produced through the current closed-mask
contour-based `assess_frame` API; its `density_bank` argument is compatibility
only. They serialize `topology_label` exactly as `standard`,
`multi_void`, `c_shape`, `unknown`, or `clipped`. The UI must not infer these
labels from `route`, `accepted`, or `rejection_reason`.

Schema records use `python-schema-runtime-record-v2`. Every record retains its
exact `schema.py` dataclass name and declared field names. Tuple, list, mapping,
bytes, NumPy scalar, and NumPy array types are tagged rather than flattened.
Arrays use `numpy-contiguous-zlib-base64-v2`, retaining dtype, shape, strides,
memory order, writeability, and complete bytes.

Generation fails explicitly if it encounters an unsupported or non-contiguous
runtime value; it never silently drops or coerces evidence. Prove a generated
run against freshly computed objects with:

```sh
PYTHONPATH=../../../.. /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.ui.backend.validate_schema_review_dump
```

## Runtime diagnostics

Run the modular shared-stage benchmark with:

```sh
python -m LegacyVision.review_pipeline.diagnose_shared_pipeline_runtime
```

The console and `runtime-diagnostics.json` report JPEG read, decode,
preprocessing, topology, each density profile, combined density-bank work, and
total shared compute in milliseconds per frame and effective hertz. The report
also evaluates P95 latency against `shared_pipeline_runtime_limits.json`.
It records eligible and ignored component counts so excluded clipped work is
auditable. Serialization and LUT loading are explicitly excluded from
production timing.
