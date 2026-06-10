# AIGP Q1 Demo: Known-Good Vision Flight

This is a minimal, self-contained extraction of the exact Q1 runtime path that successfully flew in the AI-GP simulator.

Validated source run:

`aigpq1/Flight/logs/q1runtime/run-20260610T083907Z`

Observed result from that run:

- Active gate progressed `0 -> 1 -> 2`.
- Control method was `body_rate_guidance`.
- Commands streamed at 30 Hz.
- Vision produced target updates continuously.
- No frame suppressions or safety stops were recorded.
- Final motion followed the expected negative local-NED X route direction.

## Minimal Contents

- `scripts/run_known_good.py`: the only demo runner.
- `config/known_good_body_rate_guidance.json`: exact validated flight profile.
- `src/q1runtime/`: runtime, target selection, body-rate guidance, safety, logging hooks, and required support modules.
- `src/sensing/telemetry/`: MAVLink UDP bridge used for heartbeat, telemetry, reset, arm/disarm, and `SET_ATTITUDE_TARGET`.
- `src/sensing/vision/`: UDP JPEG receiver plus the exact CNN -> regressor -> passthrough path.
- `src/sensing/perception/`: gate observation and local-NED mapping helpers.
- `src/core/logging/`: JSON run logger.
- `requirements.txt`: Python dependencies.

The demo does not import from `aigpq1/Flight/src`. The runner prepends this folder's local `src` directory to `sys.path`.

## Install

```bash
cd "/Users/trumanhale/aigp sim v1 download/aigpq1_demo"
python3 -m pip install -r requirements.txt
```

## Run

Start the simulator first, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_known_good.py
```

The profile defaults to the validated 30 second run. The runner accepts only practical overrides:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_known_good.py --run-s 30
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_known_good.py --device cpu
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_known_good.py --no-save-raw-frames
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_known_good.py --no-reset-on-start
```

## Exact Control Path

```text
sim UDP JPEG frames on port 5600
  -> CNN lightmask/depth checkpoint
  -> gate position regressor checkpoint
  -> regressor passthrough candidates
  -> local-NED target mapping with 20 degree camera tilt
  -> forward-progress candidate selection using local-position route velocity
  -> per-frame target tracker
  -> clipped local target delta
  -> body-rate roll/pitch guidance plus thrust
  -> MAVLink SET_ATTITUDE_TARGET to udpin:127.0.0.1:14550
```

Important validated settings:

- `passthrough_regressor_targets = true`
- `vision_candidate_top_k = 8`
- `min_position_confidence = 0.1`
- `target_selection_mode = forward_progress`
- `target_selection_velocity_source = local_position`
- `command_hz = 30`
- `vision_target_max_horizontal_m = 1.0`
- `vision_target_max_up_m = 0.4`
- `vision_target_max_down_m = 2.0`
- `body_rate_position_kp_deg_per_m = 32.0`
- `body_rate_max_guidance_tilt_deg = 32.0`

## Key Finding

The winning command surface was not `SET_POSITION_TARGET_LOCAL_NED`. The validated path uses MAVLink `SET_ATTITUDE_TARGET` with body rates and thrust. The payload uses type mask `128`, current attitude quaternion, body rates in radians per second, and normalized thrust.

The demo intentionally contains no track-gate center correction path. During live testing that strategy could command backwards motion when cached track-gate coordinates disagreed with the current vision/local frame. The successful flight used only current-frame regressor candidates plus route-state forward filtering.
