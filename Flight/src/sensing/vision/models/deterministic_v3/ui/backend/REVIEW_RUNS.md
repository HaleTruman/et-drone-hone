# Shared Schema Review Runs

The replay adapter writes replaceable offline output under `ui/review_runs/`.
That directory is ignored by Git and is never read by production inference.

From `deterministic_v3`, replay the current historic source with:

```sh
PYTHONPATH=../../../.. /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.ui.backend.replay_historic_run
```

The default source is `Flight/logs/run-20260731T093159Z`. An explicit run path,
`--output-root`, `--limit`, or `--force` may be supplied. The adapter reads
`frames.jsonl`, preserves its `frame_id` and `sim_time_ns`, and supplies the
recorded JPEG bytes through the same shared production functions used by live
ingress.

Each output run contains `manifest.json` plus one JSON document per frame. The
current review frontier includes the complete `FrameObservation`, nested
component/contour/topology fields, every `TopologyDecision`, the exact
`DensityBankConfiguration`, and all ten `DensityEvidence` profiles for every
density-eligible component. Frame-edge-clipped components remain represented
by their component and topology records but produce no density evidence by
default. Specialized fitters, PnP, and publication are not fabricated while
their production implementations are still absent.

A separate run-level production frontier may be placed directly under the run
directory, for example
`review_runs/<run-id>/standard-gate-pnp-runtime.json`. Its `frames` entries
carry exact JSON-compatible `GeometryFrameResult` values under
`runtime_result`, including `camera_calibration`, `gate_model`,
`quadrilateral_estimates`, and `camera_pose_estimates`. The UI backend matches
each entry to a logged image only when both `frame_id` and `sim_time_ns` agree,
then serves that original frame entry without reshaping it. This aggregate does
not substitute for per-frame `FrameObservation` or `DensityEvidence` records;
tabs requiring mask arrays or `ComponentObservation.bbox_xywh` remain limited
to the per-frame replay contract.

`TopologyDecision` records are produced through the finalized density-aware
`assess_frame` API and serialize `topology_label` exactly as `standard`,
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
PYTHONPATH=../../../.. /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.ui.backend.diagnose_shared_pipeline_runtime
```

The console and `runtime-diagnostics.json` report JPEG read, decode,
preprocessing, topology, each density profile, combined density-bank work, and
total shared compute in milliseconds per frame and effective hertz. The report
also evaluates P95 latency against `shared_pipeline_runtime_limits.json`.
It records eligible and ignored component counts so excluded clipped work is
auditable. Serialization and LUT loading are explicitly excluded from
production timing.
