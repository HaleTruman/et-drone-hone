# q1runtime Live Run Evaluation

## Evidence Reviewed

- `logs/q1runtime/run-20260610T023220Z`: sandbox startup failed before flight because UDP bind to `0.0.0.0:5600` was not permitted.
- `logs/q1runtime/run-20260610T023233Z`: live q1runtime run with `run_landmarker=True`, `passthrough_regressor_targets=True`, `top_k=1`.
- `logs/runs/run-20260610T020548Z`: previous harness run with saved camera frames and MPCC-side vision logs.

## Findings

The passthrough boolean was correct. The live q1runtime metadata shows:

```text
source = cnn_regressor_passthrough
run_landmarker = true
passthrough_regressor_targets = true
top_k = 1
landmarker_loaded = false
```

The top-1 target data was plausible. The first live q1runtime commands had high-confidence camera targets roughly 23-44 m forward, and the older saved frames show visible gates centered or near the lower field of view.

The poor flight path came from the control architecture:

- q1runtime processed 5,287 frames but emitted only 262 commands.
- Commands appeared in only three bursts: `0-27`, `54-64`, and `2986-3208`.
- 5,025 frames were suppressed for `no_top1_gate_above_confidence`.
- During suppressed spans, the runtime sent no setpoints, so the simulator was not continuously driven.
- The first implementation sent position-plus-velocity setpoints. The simulator package's own demonstrated high-level path uses velocity-only `SET_POSITION_TARGET_LOCAL_NED`, with position and yaw ignored.

## Correction

The runtime now separates perception from command streaming:

```text
Vision frame -> target update -> TargetTracker
TargetTracker -> fixed-rate CommandEmitter -> velocity-only SET_POSITION_TARGET_LOCAL_NED
```

The default MAVLink payload now omits `position_local_ned_m`, so `MavlinkBridge` masks position fields exactly as the simulator example does. The local-NED target is still retained in metadata as `target_position_local_ned_m`.

The command streamer runs independently from vision processing at `command_hz`, defaulting to `30 Hz` for safe live validation. It follows the last tracked target for `target_hold_s`, then streams zero velocity if the target expires.

The next validation step is `q1runtime.authority_probe`, which isolates velocity-command authority without vision. Vision following should use safe defaults until a command transform is proven by that probe.
