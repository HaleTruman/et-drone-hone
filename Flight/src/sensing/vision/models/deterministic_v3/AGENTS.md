> **Branch isolation notice:** This work has moved to the `ft-vision-gate-geometry-a` branch. All work for this directory must remain contained within the current `ft-vision-gate-geometry-a` working branch and its associated worktree; do not make or mirror these changes in another branch or worktree.

# Geometry Pipeline Mission and Contract

**important** this document must remain unchanged without prior authorization from the user 

## Scope

This directory owns the geometry work required to turn each standard vision frame into camera-relative gate instance poses. The final output must include both position and orientation when the available image evidence supports them.

This is an internal processing pipeline. Do not change the public frame-ingestion schema or `Flight/src/sensing/vision/service.py` merely to accommodate geometry work in this directory.

## Essential pipeline

```text
Frame
  -> decode JPEG
  -> LUT mask
  -> component-count gate,
  -> isolate connected components
  -> topology analysis
  -> topology-aware density-profile selection
  -> calibrated density fields
  -> fitter dispatch
       one aperture  -> standard P90 fitter
       zero aperture -> C-shape fitter
       two apertures -> overlap fitter
       other/uncertain -> reject 
  -> common quadrilateral normalization and validation
  -> camera-relative PnP
  -> camera-to-NED transform
  -> VisionObservation
```

Topology routing must occur before specialized fitting. A fitter may reject its assigned component, but it must not classify its own input or silently fall through to a different fitter.

# Approximate repo architecture 

  deterministic_v3/src/
  ├── pipeline.py
  ├── schema.py
  ├── preprocessing.py
  ├── density_bank.py
  ├── topology.py
  ├── standard_gate.py
  │
  ├── c_shape/
  │   ├── __init__.py
  │   ├── process.py
  │   ├── profile_selection.py
  │   ├── line_fitting.py
  │   ├── geometry.py
  │   └── validation.py
  │
  └── multi_gate/
      ├── __init__.py
      ├── process.py
      ├── profile_selection.py
      ├── aperture_fitting.py
      ├── geometry.py
      └── validation.py

 - schema.py: Data contracts only—no OpenCV processing.
  - pipeline.py: Orchestration, routing, cache lifecycle, and dispatch.
  - preprocessing.py: Shared masks, components, contours, and topology
    inputs.

  - density_bank.py: Computes and caches named density variants.
  - topology.py: Decides standard, C-shape, multi-gate, clipped, or
    ambiguous. Frame-edge-clipped or topology-unstable components must be rejected and not be forced into a valid observation path

  - standard_gate.py: Existing ordinary one-aperture process.
  - c_shape/: Everything unique to missing-side construction.
  - multi_gate/: Everything unique to two-aperture evidence ownership and
    fitting.

## Standard ingress

Geometry must remain compatible with the existing vision-service entry point:

```python
def process_frame(
    self,
    *,
    frame_id: int,
    sim_time_ns: int,
    jpeg_bytes: bytes,
    vehicle_state: VehicleState | None = None,
) -> VisionObservation:
    ...
```

The JPEG is the image input:
The ingress `frame_id` is the authoritative frame identity for the complete geometry pipeline. Every intermediate component, topology decision, density result, fit, rejection, trace, and final `VisionObservation` must remain associated with it; do not generate a replacement geometry-frame identifier.
`frame_id` and `sim_time_ns` must be preserved unchanged in the resulting observation. 
`vehicle_state` is applied *only* to the final conversion camera-to-NED 

## geometry schema

The geometry system uses one explicit internal schema to carry evidence through the pipeline. It must not change the standard ingress or the required published output.

At a high level, the schema must represent:

- Frame identity: `frame_id` and `sim_time_ns` ingress.
- Frame context: decoded-image dimensions, camera calibration reference, and optional vehicle state.
- Shared mask evidence: LUT mask, preprocessing status, connected-component identity, bounding box, and raw contour hierarchy.

FrameObservation
  ├── frame_id
  ├── image_shape
  ├── preprocessing_version
  ├── base_mask
  ├── size_filtered_mask
  ├── closed_mask
  ├── component_labels 
  └── components: ComponentObservation[]
  

 ComponentObservation
  ├── component_id
  ├── bbox_xywh
  ├── image_origin_uv
  ├── touches_frame
  ├── label
  ├── area / fill / solidity
  ├── topology
  │   ├── raw_hole_count
  │   ├── closed_hole_count
  │   ├── hole_contours
  │   └── hole_areas
  ├── contours
  │   ├── outer_raw
  │   ├── outer_simplified
  │   └── hierarchy
  ├── distance_transform
  └── density_bank

 DensityBank
  └── variants: DensityEvidence[]

  DensityEvidence
  ├── profile_id
  ├── calibration_version
  ├── density_radius_px
  ├── ridge_radius_px
  ├── final_field
  ├── positive_points
  ├── positive_weights
  ├── p70_threshold / mask
  ├── p80_threshold / mask
  └── p90_threshold / mask

  The specialized outputs:

  CShapeResult
  ├── selected_density_profile
  ├── three visible lines
  ├── missing side
  ├── contour exits
  ├── extension lengths
  └── completed quadrilateral

  MultiGateResult
  ├── selected_density_profile
  ├── two aperture observations
  ├── larger gate fit
  ├── smaller gate fit
  ├── residual ownership
  ├── contour refinements
  └── union coverage


Shared evidence is referenced by specialized fitters. 


## Shared versus specialized computation

Perform shared work once per frame or once per component and reuse it. Shared stages include JPEG decoding, LUT masking, connected-component labeling and statistics, size filtering, morphology, contour hierarchy, topology evidence, density calibration, quadrilateral normalization, validation, PnP, and output construction.

Only topology-dependent fitting belongs in specialized paths:

- One significant enclosed aperture: standard P90 quadrilateral fitter.
- No significant enclosed aperture and independently supported C-shape evidence: C-shape fitter.
- Exactly two significant apertures in one connected parent: overlap fitter.
- Separate connected parents are separate instances, not a single overlap instance.
- More than two apertures, unstable topology, or insufficient evidence: reject with a reason.

All successful fitters must return the same normalized four-corner contract so validation and PnP are implemented once.



## Required output

Publish each accepted instance through the existing observation schema:

```python
@dataclass(frozen=True)
class VisionGateObservation:
    gate_id: str
    position_camera_m: tuple[float, float, float]
    position_confidence: float
    orientation_camera: tuple[float, float, float] | None = None
    orientation_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionObservation:
    frame_id: int
    sim_time_ns: int
    gates: list[VisionGateObservation]
    source: str = "vision"
    trace: dict[str, Any] = field(default_factory=dict)
```

Use the repository's actual schema definitions as the source of truth if their container annotations differ. Do not create a parallel public observation schema in this geometry package.

Geometry is first solved in the current OpenCV camera frame and is then converted at publication into an absolute run-local NED position. `VisionGateObservation.position_local_ned` is the published gate position. The trace may retain `position_camera_cv_m`, `position_relative_ned_m`, and `camera_position_ned_m` so the transformation remains auditable:

```text
position_local_ned
  = camera_position_ned_m
  + position_relative_ned_m
```

When `vehicle_state` is unavailable, camera-relative geometry may still be computed internally, but an absolute local-NED gate observation must not be fabricated. Adding it requires an explicitly validated camera-to-NED orientation transform and must use the canonical schema field.

The per-instance trace should make failures and accepted computations auditable without requiring review-image generation. At minimum, retain the selected topology route, density profile, fitter name, validation result, PnP status, and rejection reason where applicable.

## Performance requirements

This is an ingest-compute-publish path and must remain suitable for rapid per-frame execution.



## Development rules

- Preserve the ingress and publication contracts.
- Keep shared calculations independent from topology-specific geometry.
- Add new topology procedures through dispatch, not by duplicating the complete pipeline.
- Every rejection must have a machine-readable reason.
- Experimental review code must not become a hidden dependency of production inference.
- Camera-relative PnP is the required terminal geometry result. Camera-to-NED conversion depends on valid vehicle state and camera extrinsics.

## Testing and UI support

The `testing/` and `ui/` folders are support areas for this geometry package. `testing/` is intended for speed measurements, standard functional tests, regression tests, and normal code-quality validation. `ui/` is intended to generate visualizations that aid visual analysis of geometry procedures and their intermediate data.

Production geometry procedures must not generate, annotate, render, save, or otherwise process review images internally. They must expose structured computational results and trace evidence. Testing and UI code may consume those results to perform measurements, assertions, and visualization without becoming a dependency of the production geometry pipeline.
