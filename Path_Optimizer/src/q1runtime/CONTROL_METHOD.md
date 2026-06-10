# q1runtime Control Method

`q1runtime` is the isolated highest-level control runtime for Q1. It does not use the older `src/test.py` forward-attitude harness. Its command path is:

```text
Vision UDP frame
  -> CNN/regressor
  -> regressor passthrough controller payload, top_k=1
  -> camera optical target [right, up, forward]
  -> local NED target [north, east, down]
  -> fixed-rate velocity-only SET_POSITION_TARGET_LOCAL_NED
```

## Required Simulator Streams

- MAVLink endpoint: `udpin:127.0.0.1:14550` by default.
- Vision UDP endpoint: `0.0.0.0:5600` by default.
- Required telemetry before command emission: `ODOMETRY` with `position_local_ned_m` and `attitude`.
- Required command readiness: simulator heartbeat, telemetry receive loop, and armed state unless `--no-arm` is used for diagnostics.

## Vision Target Selection

The runtime constructs the vision service with:

```python
VisionPerceptionConfig(
    run_landmarker=True,
    passthrough_regressor_targets=True,
    top_k=1,
)
```

This intentionally skips persistent landmarker state and exposes the nearest current-frame regressor target in the controller payload shape. `q1runtime` still enforces the top-1 rule locally by sorting accepted gates by camera distance and selecting the nearest gate above `min_position_confidence`.

The selected gate position is camera optical:

```text
position_xyz = [right_m, up_m, forward_m]
```

## Coordinate Conversion

`TargetMapper` uses the existing `GatePoseEstimator` transform:

```text
camera optical [right, up, forward]
  -> body FRD [forward, right, down]
  -> local NED using telemetry attitude quaternion
  -> add telemetry position_local_ned_m
```

The default camera tilt is `20 deg`, matching the existing perception tests. With identity attitude and a camera target `[0, 0, 10]`, the default mapping points roughly forward and upward in local NED:

```text
[9.397, 0.0, -3.420]
```

## Command Emission

Vision frames update target state. Commands are emitted by a separate fixed-rate command streamer, defaulting to a safe validation rate of `30 Hz`. Higher rates such as `250 Hz` are explicit opt-in because live runs showed that high-rate commands have strong authority and can run away if the command frame is wrong.

The default command payload passed to `MavlinkBridge.send_position_target()` is velocity-only:

```python
{
    "velocity_local_ned_mps": [vn_mps, ve_mps, vd_mps],
    "yaw_rad": None,
    "target_position_local_ned_m": [north_m, east_m, down_m],
}
```

The bridge converts this to MAVLink `SET_POSITION_TARGET_LOCAL_NED` in `MAV_FRAME_LOCAL_NED` with position, acceleration, yaw, and yaw-rate ignored. This matches the simulator package's demonstrated `PyAIPilotExample` position-control path, which is actually a local-NED velocity setpoint.

`target_position_local_ned_m` is diagnostic metadata for logs only. It is intentionally not named `position_local_ned_m`, because that would make the bridge activate the MAVLink position fields.

Velocity is a capped approach vector:

```text
delta = target_position_local_ned_m - current_position_local_ned_m
speed = min(max_approach_speed_mps, max(0, norm(delta) - arrival_radius_m) * approach_gain_hz)
velocity = normalize(delta) * speed
velocity.z = clamp(velocity.z, -max_vertical_speed_mps, max_vertical_speed_mps)
```

Inside `arrival_radius_m`, velocity is `[0, 0, 0]`. The default `vertical_mode` is `hold`, so vertical velocity is zero unless `--vertical-mode target` and a nonzero `--max-vertical-speed` are explicitly selected. The optional diagnostic mode `--command-mode position_velocity` restores the earlier position-plus-velocity payload, but it is not the default because the live run showed poor response with position fields active.

## Safety And Calibration

The live runtime has safety enabled by default. It stops commands and disarms on large horizontal excursions, unsafe NED z, excessive telemetry velocity, stale telemetry, and simulator reset detection.

Before vision-follow runs, use the command-authority probe:

```bash
PYTHONPATH=src python3 -m q1runtime.authority_probe --command-hz 30 --speed 0.25
```

The probe sends isolated velocity pulses and writes a report with a recommended command transform. Load that transform with:

```bash
PYTHONPATH=src python3 -m q1runtime --command-transform logs/q1runtime/authority-probe.json
```

## Target Tracking

`TargetTracker` stores the last valid local-NED target and keeps it active across short perception gaps. Default `target_hold_s` is `3.0`.

Frame behavior:

- valid top-1 gate: update tracker with a new local-NED target,
- no confident gate: leave tracker unchanged,
- missing telemetry: do not update tracker.

Command behavior:

- active target: stream velocity toward the last tracked target,
- expired or never-seen target: stream zero velocity when `stream_stop_when_target_lost` is enabled.

## Suppression Rules

No target update is produced for a frame when:

- no top-1 gate is present above confidence threshold,
- telemetry is missing,
- telemetry lacks local NED position,
- telemetry lacks attitude.

Suppressed frames are logged with the reason. Suppressed frames no longer imply command silence; the fixed-rate command ticker continues streaming from the tracker.

## Live Run

From `aigpq1/Path_Optimizer`:

```bash
PYTHONPATH=src python3 -m q1runtime
```

Useful diagnostics:

```bash
PYTHONPATH=src python3 -m q1runtime --dry-run --no-arm
PYTHONPATH=src python3 -m q1runtime --run-s 30 --command-hz 30 --max-speed 0.5 --min-confidence 0.6
PYTHONPATH=src python3 -m q1runtime --command-mode position_velocity --dry-run --no-arm
```

Run logs are written under `logs/q1runtime` by default. Raw JPEG persistence is disabled by default; enable it with `--save-raw-frames --frame-output-dir <dir>`.
