# Live Simulator Smoke Flight

The first live integration intentionally bypasses the MPCC planner. It connects
to the organizer simulator, waits in `IDLE`, arms after the configured delay,
runs preflight checks, enters `RACING`, and sends a local-NED forward velocity
target while saving FPV JPEG frames.

From `Path_Optimizer`:

```powershell
$env:PYTHONPATH = "src"
python -m autonomy.modes.live_forward_flight --idle-s 2 --racing-s 8 --forward-speed-mps 2
```

Each run is written under `data/live_runs/run-<timestamp>/`:

- `run.json` contains config, system mode transitions, starting odometry,
  starting attitude, telemetry samples, race status, collisions, commands, and
  vision counters.
- `frames/*.jpg` contains the raw FPV JPEG stream.
- `frames/frames.jsonl` maps saved filenames to simulator timestamps.

The MAVLink endpoint defaults to `udpin:127.0.0.1:14550`. The vision listener
defaults to UDP port `5600`.
