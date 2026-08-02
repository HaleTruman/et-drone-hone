# Temporary Profile Calibration Laboratory

This folder is a one-off, offline calibration tool. It is intentionally absent
from the deterministic-v3 production pipeline, the supported review UI, and all
runtime imports.

Its recorded-run reader, asset database, HTTP server, tests, and frontend all
live in this folder. No existing review backend is imported.

It preserves two boundaries:

- recorded frames and production configuration are read-only inputs;
- candidate slider values exist only in the browser request and never modify
  `src/configurations.py` or another runtime file.

The asset database stores every post-close `ComponentObservation` bounding box
from the selected runs, including frame-edge-clipped components. The default UI
view hides clipped components because the active
`DensityBankConfiguration.ignore_frame_edge_clipped` policy excludes them from
production density computation.

## Build the isolated asset database

From the repository root:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.build_assets
```

By default, the builder selects the newest three valid `Flight/logs/run-*`
directories. Repeat `--run PATH` exactly three times to pin another corpus.

## Launch the standalone UI

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.server
```

Open <http://127.0.0.1:8784/>.

The dropdown uses the production area recommendation ranges for
`scale_01` through `scale_10`. Candidate recomputation calls the production
`DensityBank` on each saved square closed-component mask. Adjustable values are
the schema-owned `density_radius_px`, `ridge_radius_px`, `inverse_gamma`, and
`ridge_gamma`; `relative_cap`, area boundaries, and calibration version remain
visible and read-only.

Both gamma controls accept values from `0.01` through `100.00` in the browser
and server-side candidate validation.

Each selected profile is one continuous scrollable collection. Component
metadata is appended in bounded batches, and density comparisons are computed
only as cards approach the viewport.

## Separate local-thickness discovery UI

The one-off [`local_thickness`](local_thickness/README.md) subfolder explores
component-constrained wall thickness and experimental per-pixel selection among
the ten current density profiles. It is a separate server on port `8785` and is
not referenced by the profile browser or production pipeline.
