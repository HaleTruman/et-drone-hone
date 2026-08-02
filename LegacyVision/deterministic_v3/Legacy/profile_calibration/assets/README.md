# Generated profile-calibration assets

`build_assets.py` writes two replaceable files here:

- `profile_assets.sqlite3`: lossless source bounding-box crops, exact square
  closed-component masks, component provenance, and profile snapshots.
- `manifest.json`: selected runs, frame/component counts, profile ranges, and
  the configuration/preprocessing versions used to build the database.

The SQLite database is an offline review asset, never a production input.

