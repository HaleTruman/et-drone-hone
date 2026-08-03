# Legacy UI Instrumentation Archive

Archive date: **2026-08-01**

## Justification

These scripts were moved out of `ui/backend/` because the authoritative UI
inventory classifies them as definitively inactive. They are not imported,
invoked, or served by the current schema-driven viewer and do not produce the
current `ui/review_runs/` runtime-record contract. Most preserve earlier
density sweeps, C-shape experiments, or overlap-review computations that
duplicate or predate the current production APIs.

Keeping them beside the active server, replay, schema, validation, and timing
tools made the supported instrumentation boundary ambiguous. The archive is
ignored by Git so these local historical utilities cannot be mistaken for
maintained repository entry points. This README is retained as the durable
record of what moved and why.

## Archived scripts

Density and parameter sweeps:

- `render_earlier_field_only_five_gate_review.py`
- `render_earlier_field_p90_quad_five_gate_review.py`
- `render_density_gamma_sweep_fixed_radii.py`
- `render_density_gamma_sweep_overlaps.py`
- `render_radius_sweep_fixed_gamma.py`

C-shape development renderers:

- `render_c_shape_three_line_review.py`
- `render_c_shape_finite_extent_review.py`
- `render_c_shape_contour_guided_review.py`
- `render_c_shape_parallel_aligned_review.py`
- `render_c_shape_bounded_angle_review.py`
- `render_c_shape_decoupled_extension_review.py`

Overlap and morphology experiments:

- `render_overlap_checkpoint_stage_audit.py`
- `explore_overlap_multilayer_morphology.py`

## Use restriction

Archived scripts are local historical evidence, not production or active UI
dependencies. Before reuse, restore the required files deliberately, validate
their imports and source assumptions, update their JSON terminology, and obtain
approval before representing any result as current behavior.

`backend/render_c_shape_checkpoint_sweep.py` remains pending a separate
retention decision and currently imports two helpers from this local archive.
It is therefore not portable or supported as an active backend entry point.
