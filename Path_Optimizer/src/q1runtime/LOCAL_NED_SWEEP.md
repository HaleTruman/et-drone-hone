# Local-NED Command Sweep

This module is an independent probe for the simulator's `SET_POSITION_TARGET_LOCAL_NED` command surface. It bypasses the vision stack, target tracker, and q1 autonomy loop so one live session can answer which local-NED command payloads produce which telemetry response.

## Runner

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m q1runtime.local_ned_sweep \
  --command-hz 30 \
  --sample-hz 30 \
  --pulse-s 0.25 \
  --rest-s 0.75 \
  --speeds 0.05,0.1,0.25 \
  --axes x,y \
  --output logs/q1runtime/local-ned-sweep.json
```

Default behavior is intentionally conservative:

- Streams only velocity-only local-NED cases on X and Y.
- Uses short pulses and a stop/rest period after every case.
- Resets between cases by default.
- Arms at start, rearms after reset, and disarms on exit.
- Aborts on excessive displacement, Z bounds, speed limits, or unexpected reset count changes.

Preview the exact planned cases without opening MAVLink:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m q1runtime.local_ned_sweep \
  --speeds 0.05,0.1,0.25 \
  --axes x,y \
  --plan-only
```

## Opt-In Surfaces

The default sweep only probes horizontal velocity because that is the command mode used by the current q1runtime path. Additional surfaces require explicit flags:

- `--include-z --axes x,y,z`: add positive and negative Z velocity/position cases.
- `--include-position`: send position-bearing setpoints with zero velocity. The emitted `position_local_ned_m` is computed from the case's start telemetry plus the configured local offset.
- `--include-position-velocity`: send both `position_local_ned_m` and `velocity_local_ned_mps` in the same payload.
- `--include-yaw`: include yaw-bearing setpoints with zero velocity.

Example broader probe:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m q1runtime.local_ned_sweep \
  --command-hz 30 \
  --sample-hz 30 \
  --pulse-s 0.25 \
  --rest-s 0.75 \
  --speeds 0.05,0.1 \
  --position-offsets 0.5 \
  --axes x,y \
  --include-position \
  --include-position-velocity \
  --output logs/q1runtime/local-ned-sweep-position.json
```

## Output

The JSON log contains:

- The exact sweep config.
- Every planned case.
- Every completed case result.
- The exact command payload emitted for the case.
- Start/end telemetry snapshots.
- Sampled telemetry during the pulse.
- Displacement, dominant response axis/sign, max observed speed, mean local velocity, cross-axis ratio, reset changes, and actuator-output presence.
- Safety stop reason/details if a case tripped a bound.

Use this log to build the next transform or control-path decision from observed command-response data instead of assuming the simulator's local-NED interpretation.
