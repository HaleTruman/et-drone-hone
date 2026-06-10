# Q1 Runtime Control Surface Notes

These notes capture live simulator findings from clean, instrumented command probes. They are intentionally empirical: use them as the current truth table for command tooling until superseded by later runs.

## Reliable Procedure

Use a clean simulator reset before command characterization. Runs without reset are dominated by existing inertia, gravity, attitude, and velocity, which can obscure the command being tested.

The most useful logs so far are:

- `logs/q1runtime/motor-command-probe-equal-trim-search.json`
- `logs/q1runtime/motor-command-probe-patterns-clean-reset.json`
- `logs/q1runtime/local-ned-sweep-z-up-guarded.json`
- `logs/q1runtime/attitude-thrust-probe-level-fresh.json`
- `logs/q1runtime/attitude-thrust-bodyrate-only-upper-reset-grid.json`
- `logs/q1runtime/attitude-thrust-bodyrate-pitch-plus.json`
- `logs/q1runtime/attitude-thrust-bodyrate-pitch-minus.json`
- `logs/q1runtime/body-rate-feedback-level-hold-024-fastpitch-cleanend.json`
- `logs/q1runtime/body-rate-feedback-alt-hold-up050-k014-max034.json`
- `logs/q1runtime/body-rate-feedback-prelevel100-alt-up050.json`
- `logs/q1runtime/body-rate-feedback-prelevel0-alt-pitch-plus3.json`
- `logs/q1runtime/body-rate-feedback-prelevel0-alt-pitch-minus3.json`
- `logs/q1runtime/body-rate-feedback-prelevel0-alt-roll-plus3.json`
- `logs/q1runtime/body-rate-feedback-prelevel0-alt-roll-minus3.json`
- `logs/q1runtime/body-rate-position-guidance-xplus1-zup040.json`
- `logs/q1runtime/body-rate-position-guidance-xplus1-zup040-kd4.json`

## SET_POSITION_TARGET_LOCAL_NED

Velocity-only, position-only, and position+velocity local-NED setpoints were accepted by the transport but did not produce practical motion in the tested startup/ground state.

Important observed cases:

- XY velocity-only at `0.05`, `0.10`, `0.25 m/s`: essentially zero displacement and zero reported velocity.
- Position-bearing XY offsets of `0.5 m`: essentially zero displacement.
- Upward NED `z=-0.5 m` target with `vz=-0.25 m/s`: essentially zero displacement.

Current interpretation: local-NED setpoints should not be the first control layer for vision following until a lower-level attitude/motor controller is stable.

## SET_ATTITUDE_TARGET

This surface has strong authority, but open-loop quaternion targets are dangerous because the engine computes motor mixing from attitude error:

`attitude_error = desired_quaternion * conjugate(current_quaternion)`

Observed behavior:

- `quaternion=[1,0,0,0]` with thrust `0.03-0.20`: mostly idle/no actuator response from clean start.
- `quaternion=[0,0,0,1]` with tiny thrust values: very large motion. This indicates large attitude error, not clean collective thrust.
- Measured-attitude open-loop pulses can become unstable if the vehicle is already rotating/moving.

Current interpretation: `SET_ATTITUDE_TARGET` has two very different operating regimes:

1. Attitude-hold mode, `attitude_type_mask=7`, uses the quaternion and can create high-gain attitude-error motor spikes. Do not use this as the first live control path.
2. Body-rate-only mode, `attitude_type_mask=128`, ignores the quaternion attitude field and uses `body_rates_rps` plus `thrust`. This is currently the cleanest mid-level command surface.

### Body-Rate-Only Findings

Command shape:

```json
{
  "quaternion": "<current telemetry quaternion; ignored by mask 128>",
  "thrust": 0.22,
  "attitude_type_mask": 128,
  "body_rates_rps": [0.0, 0.25, 0.0]
}
```

Clean thrust threshold with `body_rates_rps=[0,0,0]`, `pulse_s=0.35`, reset between cases:

| Thrust | Result |
| --- | --- |
| `0.02-0.14` | Inert; actuator output stays at `0.05` floor |
| `0.18` | Very small movement; still actuator floor |
| `0.22` | Threshold crossing; actuator max about `0.23`, controlled movement |
| `0.26` | Stronger but controlled; actuator max about `0.26` |
| `0.30` | Controlled but larger; max speed about `0.98 m/s` |

Body-rate sign table at thrust `0.22`, `pulse_s=0.35`:

| Commanded body rate | Main Euler response | Notes |
| --- | --- | --- |
| `[+0.25, 0, 0]` | Roll decreases by about `8.1 deg` | Lateral motion one direction |
| `[-0.25, 0, 0]` | Roll increases by about `9.5 deg` | Lateral motion opposite direction |
| `[0, +0.25, 0]` | Pitch increases by about `6.8 deg` | Corrects spawn pitch from `-17.8 deg` toward level |
| `[0, -0.25, 0]` | Pitch decreases by about `11.9 deg` | Drives nose-down/forward acceleration |
| `[0, 0, +0.25]` | Yaw decreases by about `8.1 deg` | Mild roll/pitch coupling |
| `[0, 0, -0.25]` | Yaw increases by about `8.2 deg` | Mild roll/pitch coupling |

Closed-loop body-rate feedback result:

- Module: `python3 -m q1runtime.body_rate_feedback_probe`
- Config: `thrust=0.24`, `target_roll_deg=0`, `target_pitch_deg=0`, `pitch_kp=2.0`, `max_pitch_rate_rps=0.5`
- Result: pitch corrected from about `-17.8 deg` to about `0.0 deg`, max speed stayed below `0.30 m/s`, and controlled-window displacement was about `[-0.174, -0.103, +0.016] m`.

The same module now has an opt-in altitude-hold thrust loop:

```bash
python3 -m q1runtime.body_rate_feedback_probe \
  --altitude-hold \
  --thrust 0.24 \
  --vertical-kp 0.08 \
  --vertical-kd 0.12 \
  --min-thrust 0.18 \
  --max-thrust 0.30
```

This does not change the command surface. It still emits body-rate-only `SET_ATTITUDE_TARGET` with `attitude_type_mask=128`; only the `thrust` field is adjusted. Because local NED uses positive `z` as down, the loop increases thrust when `z_error_ned_m` is positive or `vz_ned_mps` is positive.

Live altitude results:

| Config | Result |
| --- | --- |
| Upward target `-0.30 m`, conservative thrust around `0.266` | Mostly arrested gravity; did not climb meaningfully |
| Upward target `-0.50 m`, `vertical_kp=0.14`, `max_thrust=0.34` | Climbed about `0.31 m`; max speed about `0.43 m/s`; horizontal drift about `0.86 m` |
| Prelevel `1.0 s` at thrust `0.20`, then same upward target | Climbed about `0.34 m`; max speed about `0.29 m/s`; horizontal drift reduced to about `0.48 m` |

The prelevel phase is important. The vehicle spawns around `-17.8 deg` pitch. Correcting that pitch while already producing climb thrust creates horizontal acceleration. Correcting attitude first at low thrust produces a cleaner climb.

Telemetry velocity note: ODOMETRY velocity and `LOCAL_POSITION_NED` velocity disagreed in X/Y sign during the clean prelevel/climb run. With yaw near `180 deg`, this indicates ODOMETRY velocity is not safe to treat as local NED for horizontal control. `q1runtime.body_rate_feedback_probe` now defaults to `--velocity-source local_position` and records ODOMETRY velocity in `velocity_debug` for comparison.

Clean attitude-to-motion steering results with prelevel `1.0 s`, altitude-hold target `-0.40 m`, and main attitude target held for `2.5 s`:

| Main target attitude | Result |
| --- | --- |
| `pitch=+3 deg`, `roll=0 deg` | Positive local X motion; displacement about `[+1.71, +0.04, -0.23] m`; max speed about `1.29 m/s` |
| `pitch=-3 deg`, `roll=0 deg` | Negative local X motion; displacement about `[-0.97, -0.01, -0.23] m`; max speed about `0.97 m/s` |
| `roll=+3 deg`, `pitch=0 deg` | Positive local Y motion; displacement about `[+0.38, +1.26, -0.23] m`; max speed about `1.11 m/s` |
| `roll=-3 deg`, `pitch=0 deg` | Negative local Y motion; displacement about `[+0.37, -1.20, -0.23] m`; max speed about `1.08 m/s` |

Current guidance mapping from local target error to attitude target:

- Positive local X request should become positive `target_pitch_deg`.
- Negative local X request should become negative `target_pitch_deg`.
- Positive local Y request should become positive `target_roll_deg`.
- Negative local Y request should become negative `target_roll_deg`.
- Horizontal velocity damping must use `LOCAL_POSITION_NED` velocity.

### Local Position Guidance Layer

Module: `src/q1runtime/body_rate_position_guidance.py`

Probe integration: `python3 -m q1runtime.body_rate_feedback_probe --position-guidance`

The guidance layer converts a local-NED target into attitude targets, then the body-rate loop tracks those attitudes:

```text
target_position_local_ned_m - current_position_local_ned_m
  -> target_pitch_deg from local X error and local X velocity damping
  -> target_roll_deg from local Y error and local Y velocity damping
  -> body-rate-only SET_ATTITUDE_TARGET
```

Best first result:

- Commanded offset after prelevel: `target_position_offset_local_ned_m=[1.0, 0.0, -0.40]`
- Gains: `position_kp_deg_per_m=2.0`, `velocity_kd_deg_per_mps=4.0`, `max_guidance_tilt_deg=3.0`
- Result: final displacement about `[+1.04, -0.01, -0.24] m`, max speed about `0.37 m/s`, no abort.

This is currently the strongest path for future top-1 vision target following: map the vision target to a local-NED target, feed that target into the position guidance layer, and let the body-rate/altitude layers emit the actual command stream.

Current best control foundation: body-rate-only `SET_ATTITUDE_TARGET`, not local-NED position targets and not attitude-quaternion hold mode.

## SET_ACTUATOR_CONTROL_TARGET

Direct motor commands are accepted and are the clearest truth layer.

Clean equal-motor trim search:

| Equal motor command | Result |
| --- | --- |
| `0.18` | Near idle; tiny displacement; max speed about `0.012 m/s` |
| `0.22` | Mild response; max speed about `0.17 m/s` |
| `0.26` | Noticeable response; max speed about `0.47 m/s` |
| `0.30` | Large response; about `2.63 m` Z displacement in a short pulse |

This suggests the live useful collective range is much closer to `0.22-0.26` than the offline hover estimate around `0.495`.

Clean motor pattern map at value `0.24`:

| Pattern | Command | Main observed response |
| --- | --- | --- |
| `front` | `[0.24, 0.24, 0, 0]` | Small negative pitch-rate delta |
| `rear` | `[0, 0, 0.24, 0.24]` | Positive pitch-rate response |
| `left` | `[0.24, 0, 0.24, 0]` | Negative pitch / coupled roll response |
| `right` | `[0, 0.24, 0, 0.24]` | Positive pitch / coupled roll response |
| `diag_a` | `[0.24, 0, 0, 0.24]` | Coupled response with yaw contribution |
| `diag_b` | `[0, 0.24, 0.24, 0]` | Coupled response with weaker yaw contribution |

Current interpretation: direct motors are excellent for identifying motor order, thrust threshold, and torque signs. They are useful as a diagnostic layer, but body-rate-only `SET_ATTITUDE_TARGET` now appears to be the safer foundation for live closed-loop control.

## Near-Term Direction

1. Keep the prelevel phase enabled before climb or vision-following tests.
2. Use `LOCAL_POSITION_NED` velocity, not ODOMETRY velocity, for local-NED velocity damping and future target-position control.
3. Use `BodyRatePositionGuidance` as the local target-to-attitude layer.
4. Reconnect the top-1 vision target output as the source of `target_position_local_ned_m`, with conservative clipping and fresh telemetry on every frame.
