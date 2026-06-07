# Live Simulator Hover Smoke Test

The first live integration intentionally bypasses the MPCC planner. It connects
to the organizer simulator, remains in `IDLE` until heartbeat and odometry are
available, arms, and sends attitude targets for a hover 1 meter above the
starting local-NED position while saving FPV JPEG frames.

From `Path_Optimizer`:

```powershell
$env:PYTHONPATH = "src"
python src/test.py --hover-s 8 --hover-altitude-m 1
```

Each new run is written under `logs/runs/run-<timestamp>/`:

- `run.json` contains config, system mode transitions, starting odometry,
  starting attitude, telemetry samples, race status, collisions, commands, and
  vision counters.
- `frames/*.jpg` contains the raw FPV JPEG stream.
- `frames/frames.jsonl` maps saved filenames to simulator timestamps.

The MAVLink endpoint defaults to `udpin:127.0.0.1:14550`. The vision listener
defaults to UDP port `5600`.
