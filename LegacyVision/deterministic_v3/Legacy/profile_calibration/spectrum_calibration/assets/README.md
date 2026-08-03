# Spectrum Calibration Assets

This directory is the atomic output target for the one-off
`spectrum_calibration.build_assets` command. Generated database and manifest
files are intentionally not production inputs and must not be imported by the
deterministic-v3 pipeline.

The builder uses the latest three valid `Flight/logs/run-*` directories unless
exactly three `--run` arguments are supplied. It clones the production
preprocessing configuration with an explicit `a250` pre-close component-area
threshold, performs the normal 5×5 close, and discards every resulting
`ComponentObservation` that touches the image boundary.

For the currently pinned three-run corpus, the expected retained count is
2,441 components. Components are sorted by:

```text
area_px DESC, run_id ASC, frame_id ASC, component_id ASC
```

The resulting zero-based ranks are partitioned into ten contiguous balanced
bins. `bin_index=0` contains the largest components and `bin_index=9` the
smallest. Persisting both `rank_index` and `bin_index` makes equal-area boundary
ties reproducible.

Each component stores two lossless PNG blobs with identical analysis-square
dimensions:

- `source_png`: the BGR source frame aligned to `image_origin_uv`, padded black
  wherever that analysis square extends beyond the frame;
- `closed_mask_png`: the exact binary square mask returned by
  `component_mask(..., stage="closed")`.

Generated outputs are:

- `spectrum_assets.sqlite3`
- `manifest.json`

The manifest records source runs, the complete preprocessing and density
configuration snapshots, sort and bin contracts, counts, bin statistics, and
the database byte size and SHA-256. The builder runs SQLite integrity,
foreign-key, rank/bin, per-bin-statistic, PNG-shape, binary-mask, byte-size, and
SHA-256 validation before individually replacing the database and manifest
paths atomically.

From the repository root, build with:

```bash
PYTHONPATH=Flight/src /usr/local/bin/python3.13 -m \
  sensing.vision.models.deterministic_v3.src.profile_calibration.\
spectrum_calibration.build_assets
```

Do not commit generated database snapshots unless they are explicitly accepted
as calibration evidence.
