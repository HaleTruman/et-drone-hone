# Vision Runtime

Vision is a clean deterministic vision runtime. It processes JPEG frames
through RGB LUT masking, connected-component bboxes, clipping diagnostics,
contours, pose fitting, rolling instance tracking, and a Flight-compatible
observation bridge.

## Run

```bash
python vision/main.py run --source-dir path/to/jpegs --output-root /tmp/vision-runs --debug --max-frames 10
python vision/main.py run --single-frame path/to/frame.jpg --output-root /tmp/vision-runs
python vision/main.py watch --source-dir path/to/jpegs --output-root /tmp/vision-runs --debug
python vision/main.py review --host 127.0.0.1 --port 8788
python vision/main.py review --run-root /tmp/vision-runs/run-... --host 127.0.0.1 --port 8788
```

Production runs write only final frame JSON plus `latest.json`, `status.json`,
and `run_manifest.json`. Debug runs additionally write one per-stage JSON file
per frame and the color-mask `.bin` for review.

Review mode can open an existing run with `--run-root`, or start without one
and let the UI run debug output from a selected folder containing
`vision_frames/`. UI-launched debug runs are written under `vision_run/`.
