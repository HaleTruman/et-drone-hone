# WORK IN PROGRESS: Geometry Evaluation UI Architecture

> **Authority and ambiguity:** The package-level `../AGENTS.md` and
> `../REPOSITORY_STANDARDS.md` govern this folder. This document narrows those
> rules for UI and calibration work; it does not replace them. The architecture
> and controls below are deliberately incomplete. Any ambiguous requirement,
> undocumented setting, or proposed expansion of runtime behavior must be
> raised to the user for confirmation before implementation.

## Mission

This folder owns a minimal laboratory interface for inspecting deterministic V3
geometry. It is an evaluation instrument, not a product UI. Clarity of evidence,
fast iteration, reproducibility, and negligible production-runtime impact take
priority over visual polish.

The UI must support two related uses:

1. Offline review and calibration from recorded run frames.
2. Sampled live debugging using the same production `src` computations.

Offline review is the first implementation milestone. Live recalculation and
calibration controls follow only after the production contracts and performance
boundary are stable.

## Folder ownership

```text
ui/
  AGENTS.md
  backend/
    Offline traversal, production API adapters, calibration requests,
    debug-stream transport, manifest generation, and review rendering.
  frontend/
    Static HTML, CSS, JavaScript, local manifests, and browser rendering.
```

- `frontend/` must not implement OpenCV or geometry math.
- `backend/` may import stable production APIs from `src`.
- Production `src` must never import `ui`, depend on a browser, write review
  images, or read UI state.
- Tests may import both sides to verify the adapter boundary.
- Generated images, manifests, and caches are not runtime inputs.

The files initially moved into `backend/` are historical offline calibration
scripts. Their placement does not make their duplicated computations or
hard-coded paths approved architecture. They should be migrated incrementally
to consume structured production results.

## Evaluation layout

Keep the current frame viewer's simple frame-by-frame and layer-by-layer model.
The first review tab should present:

1. Full-frame shared stages: source, LUT/base mask, size filtering, close,
   components, topology routes, density evidence, normalized geometry, and
   accepted/rejected output.
2. C-shape stages for components routed to C-shape processing.
3. Multi-gate stages for components routed to exactly-two-aperture processing.

Standard single-aperture results may be shown with the shared/standard output;
the two specialized sections are C-shape and multi-gate.

A second tab should be reserved for calibration:

- Shared calibration controls.
- C-shape-specific calibration controls.
- Multi-gate-specific calibration controls.

Each displayed layer must include a compact label, owning module or operation,
component identity where applicable, active settings, and rejection reason when
applicable.

## Initial control scope

Controls must begin intentionally narrow. The first candidates are:

- Initial maximum component count.
- Minimum component pixel area.
- Density radius.
- Ridge radius.
- Inverse-density gamma.
- Ridge gamma.

Do not expose every constant. Specialized controls should be added only when a
specific calibration question requires them and the user confirms their scope.
Every control must map to an explicit, serializable configuration value. UI
controls must not mutate module globals or create a separate hidden calibration
path.

## Offline-first implementation

The first iteration should preserve the existing static viewer behavior:

1. Accept explicit run/frame inputs and output locations.
2. Invoke production computation through stable `src` entry points.
3. Adapt structured results into independently rendered review layers.
4. Write a versioned manifest containing inputs, settings, stage ownership,
   frame identity, component identity, routes, and rejection reasons.
5. Let the static frontend select runs and frames and display each stage.

Runtime review dumps are ingested exclusively from `ui/review_runs/`, using the
layout `review_runs/<run-id>/frames/<frame-stem>.json`. Do not scan the older
package-level `review_runs/`, `legacy_review_runs/`, `production_samples/`, or
another implicit location for runtime JSON. Logged source images remain
read-only and may be resolved from an explicitly supplied `Flight/logs` root.

The existing viewer navigation, layer grid, source labels, density sweep, and
3x magnifier are acceptable first-iteration features. Generated artifacts must
remain replaceable and must not be imported by production code.

## Later live-debug boundary

Live debugging must use the same production APIs and configuration schema as
offline review. It must not introduce alternate geometry math.

The intended boundary is a bounded external debug adapter:

```text
production result / trace
  -> optional sampled debug adapter
  -> bounded latest-frame queue
  -> UI backend
  -> browser
```

- The production thread must never block on the UI.
- Rendering and image encoding remain outside `src`.
- Backpressure drops stale debug frames instead of slowing inference.
- Debugging is disabled by default.
- Large arrays are copied or encoded only for enabled, sampled stages.
- Live parameter changes create an explicit candidate configuration and trigger
  recomputation through the same production API used offline.
- Slider requests must be debounced or superseded so obsolete calculations do
  not accumulate.

## Calibration comparison

Calibration must preserve both baseline and candidate results. A slider change
must not silently replace the approved configuration. The UI should display:

- Baseline settings and output.
- Candidate settings and output.
- The source run, frame, and component.
- Compute time for the recalculated stages.
- Route or rejection changes caused by the candidate.

Approval and persistence of a candidate configuration are separate future
operations and require explicit user direction.

## Testing expectations

UI/backend tests should verify:

- Production `src` has no dependency on `ui`.
- Offline and live adapters consume the same structured result contract.
- Manifest and configuration schemas are versioned and deterministic.
- Source run frames are never modified.
- Baseline and candidate settings remain distinguishable.
- Unknown, clipped, unstable, and fitter-rejected components are visible with
  machine-readable reasons.
- Rendering is not included in production compute-time measurements.
- A slow or disconnected frontend cannot block the runtime pipeline.

## Phased implementation

1. **Current structural step:** place static viewer assets in `frontend/`, place
   offline review scripts in `backend/`, and preserve existing offline review.
2. **Offline adapter:** replace duplicated script computations with structured
   `src` results and one versioned review manifest.
3. **Evaluation layout:** add shared, C-shape, and multi-gate sections plus
   topology and rejection evidence.
4. **Scoped calibration:** add the limited pixel, radius, and gamma controls and
   explicit baseline-versus-candidate recomputation.
5. **Live debugging:** add bounded sampled transport only after offline behavior
   and runtime measurements are validated.

Do not advance an ambiguous phase boundary or add unapproved controls without
raising the concern to the user.

## UI terminology standard

The JSON output schema is the sole authority for evidence labels and process
terminology in this UI. Every displayed type, field, profile, route, status,
and rejection value must use the exact upstream schema name or serialized
value. Do not introduce UI-only aliases, including historical names, for
upstream evidence. If a desired term does not exist in the JSON schema, refine
and approve that upstream schema first; only then may the UI adopt the term.
Every visualization, grouping, overlay, and provenance cue must likewise be
rooted in evidence explicitly present in that JSON. Styling may clarify
serialized evidence but must not fabricate, infer, or imply upstream output.

## JSON inspector responsiveness

Selecting a run must begin loading its initial review JSON before the complete
frame catalog is available. The frontend may retain a small, bounded cache of
parsed and preformatted review documents and must share an in-flight request
between preloading and the JSON inspector. It must not preload a complete run,
persist review JSON outside the current browser session, or create a second
interpretation of the serialized records.

Changing frames should prime the selected frame using the same cache. Cache
failures must remain isolated from frame review and be retried when the user
opens the inspector. Reloading the page is the explicit cache reset for this
first implementation.

## Viewer entry-point terminology

The schema-driven evaluation viewer uses this baseline layout:

- `index.html` and `app.js`: primary preprocessing and `DensityEvidence`
  review.
- `source.html` and `source_app.js`: source-only frame review.
- `topology.html` and `topology_app.js`: run-level exact
  `TopologyDecision.topology_label` component review.
- `pnp_scene.html`, `pnp_scene_app.js`, and `pnp_scene_adapter.js`: fixed
  optical-camera `CameraPoseEstimate` projection from the exact run-level
  `GeometryFrameResult` JSON fields.

Any entry-point change must update both navigation links, static-server
default-entry behavior, cache-buster references, `MASK_REVIEW.md`, and
`tests/test_ui_architecture.py` atomically. Acceptance requires that `/` and
`/index.html` open the primary review, `/source.html` opens the source-only
view, `/topology.html` opens topology-label review, and active UI or
documentation references use these baseline names. `/pnp_scene.html` opens
the camera-aligned `CameraPoseEstimate` projection.
Production `src`, review JSON, and replay contracts remain unchanged.

## Current UI features and script status

This inventory is authoritative for the current UI as of 2026-08-01. A script
is **active** only when it is part of the browser dependency graph, serves the
current viewer, or is the documented producer/validator/diagnostic for the
current `ui/review_runs/` JSON contract. Merely residing in `ui/backend/` does
not make a historical renderer active.

The active UI currently provides:

- `index.html`: primary zero-based frame review for source, preprocessing,
  `ComponentObservation.touches_frame`, and dynamically discovered
  `DensityEvidence` profiles.
- `source.html`: source-only frame review.
- `topology.html`: run-level `TopologyDecision.topology_label` filtering with
  complete `bbox_xywh` source crops aspect-fitted into 250x250 views without
  stretching or clipping; unused canvas area remains empty.
- `pnp_scene.html`: fixed OpenCV optical-camera projection over the selected
  source frame using serialized `CameraCalibration`, `PlanarGateModel`, and
  accepted `CameraPoseEstimate` values plus metric camera-space guides.
- Four-column evidence layout with three non-evidence placeholders completing
  the preprocessing batch.
- JSON-rooted profile provenance borders, exact schema labels, and a 3x hover
  magnifier.
- Per-layer `i` inspection of the complete frame JSON, with run-selection
  preloading, shared in-flight requests, preformatted text, and a bounded
  four-frame browser-session cache.
- Session-scoped run and exact frame-filename restoration across compatible
  tabs; tabs without the selected run restore their own last valid run/frame.
- Read-only source images from `Flight/logs` and runtime review JSON exclusively
  from `ui/review_runs/`.

The active UI does not yet provide live calibration controls, a persistent
cross-frame component identity, bounding-box interaction, specialized C-shape
or multi-gate result panels, or an optimized large-document JSON renderer.

### Authoritative active scripts

Frontend implementation:

- `frontend/app.js`: primary preprocessing and `DensityEvidence` viewer.
- `frontend/source_app.js`: source-only viewer.
- `frontend/topology_app.js`: exact topology-label filtering, category color,
  and `bbox_xywh` source-crop rendering; it does not use the JSON inspector.
- `frontend/pnp_scene_app.js`: isolated Three.js camera scene, source-frame
  background, exact runtime camera projection, metric guides, and gate planes
  built from serialized model points and accepted poses.
- `frontend/pnp_scene_adapter.js`: validates the authoritative
  `GeometryFrameResult` fields, preserves all `CameraPoseEstimate` records, and
  fails closed when an accepted pose or camera distortion cannot be rendered
  exactly.
- `frontend/schema_json_cache.js`: bounded parsed/preformatted review cache and
  run/frame priming.
- `frontend/schema_json_inspector.js`: per-layer JSON modal and schema-field
  targeting.
- `frontend/schema_provenance.js`: stable JSON-value provenance styling.
- `frontend/magnifier.js`: automatically enabled 3x image magnifier.
- `frontend/frame_navigation_state.js`: shared session-scoped run/frame
  selection keyed by exact `source.relative_path` filename.

Backend implementation and current offline contract tools:

- `backend/serve_review_ui.py`: authoritative static server, run/frame catalog,
  source delivery, per-frame schema-layer rendering, and exact frame-addressed
  delivery from run-level `GeometryFrameResult` JSON.
- `backend/replay_historic_run.py`: authoritative current producer for
  `ui/review_runs/<run-id>/`.
- `backend/schema_json.py`: lossless current schema serialization and recovery.
- `backend/validate_schema_review_dump.py`: exact recomputation and round-trip
  validator for current review JSON.
- `backend/diagnose_shared_pipeline_runtime.py`: current shared-stage timing and
  runtime diagnostic producer.
- `backend/test_serve_review_ui.py`: active server/catalog/layer contract tests;
  this is verification code, not a launched UI process.

Active non-script dependencies are `frontend/index.html`,
`frontend/source.html`, `frontend/topology.html`, `frontend/pnp_scene.html`,
`frontend/styles.css`, `frontend/pnp_scene.css`, pinned Three.js `0.165.0`,
`backend/shared_pipeline_runtime_limits.json`, and `backend/REVIEW_RUNS.md`.

### Legacy archive and inactive scripts

The definitively inactive scripts below were moved on 2026-08-01 into the
Git-ignored local `legacy/` archive documented by `legacy/README.md`. They are
not imported, invoked, or served by the current UI and do not produce its
current runtime-record review contract.

Earlier density and parameter-sweep renderers:

- `legacy/render_earlier_field_only_five_gate_review.py`
- `legacy/render_earlier_field_p90_quad_five_gate_review.py`
- `legacy/render_density_gamma_sweep_fixed_radii.py`
- `legacy/render_density_gamma_sweep_overlaps.py`
- `legacy/render_radius_sweep_fixed_gamma.py`

Earlier C-shape development renderers:

- `legacy/render_c_shape_three_line_review.py`
- `legacy/render_c_shape_finite_extent_review.py`
- `legacy/render_c_shape_contour_guided_review.py`
- `legacy/render_c_shape_parallel_aligned_review.py`
- `legacy/render_c_shape_bounded_angle_review.py`
- `legacy/render_c_shape_decoupled_extension_review.py`

Earlier overlap and morphology development renderers:

- `legacy/render_overlap_checkpoint_stage_audit.py`
- `legacy/explore_overlap_multilayer_morphology.py`

Inactive scripts requiring an explicit retention decision:

- `backend/render_current_production_review.py`: its name implies current
  authority, but it is absent from the current server/replay dependency graph.
- `backend/render_all_current_production_overlaps.py`: likewise named current,
  but it only belongs to the earlier standalone overlap-review chain.
- `backend/validate_inverse_density_checkpoint.py`: still described in
  `MASK_REVIEW.md`, but validates the earlier standalone detector rather than
  the current schema-record viewer.
- `backend/render_c_shape_checkpoint_sweep.py`: still referenced by historical
  calibration notes and may remain useful for future specialized review, but
  is not connected to the current UI. Its two helper imports now reside in the
  ignored local archive, so it is not a portable supported entry point.
- `backend/render_overlap_checkpoint_broad_review.py`: may remain useful for a
  future multi-gate panel, but is not connected to the current UI.

Before one of these inactive scripts is reused, its inputs, output location,
schema terminology, and production-module dependencies must be revalidated.
It must not be represented as an active UI feature merely because it remains
executable.

## Potential future work

A JSON-rooted overlay may expose `ComponentObservation.bbox_xywh` on hover or click.
Cross-frame following requires an upstream persistent identity; `component_id` is frame-local.
