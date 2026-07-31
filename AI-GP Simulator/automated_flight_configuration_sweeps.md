# Automated Flight Configuration Sweeps

Document status: execution runbook for already configured Flight systems.

This procedure assumes one simulator environment is already open and ready.
It does not describe CrossOver, Wine, or simulator launch. See
[simulation_environment.md](simulation_environment.md) for environment
startup, port bindings, direct reset, protocol details, and known failures.
The live validation behind this procedure is for V2 (`VQ_2`); V1 (`VQ_1`)
has not been validated.

## 1. Non-negotiable rules

1. Run one simulator and one Flight process only.
2. Give every trial an explicit duration. Never automate `RUN_S=None`.
3. Use a new Flight process for every trial; there is no complete in-process
   controller reset.
4. Stop Flight and confirm `14550`/`5600` are free before using a one-shot
   MAVLink client.
5. After every trial, verify disarm and reset the simulator physics/run
   state before starting the next configuration.
6. Do not leave Flight connected between trials. It records every cycle and
   continues saving frames, which can exhaust workspace storage.
7. Do not leave an armed or active physics run unattended. Continued physics
   and collision traffic create unnecessary compute and hardware load.
8. Keep configurations, manifests, recordings, and results inside this
   repository. Do not read or write external experiment trees.
9. Stop the sweep on a port conflict, failed reset/disarm, missing readiness
   event, unfinalized recording, or insufficient disk space.

## 2. Configuration authority

The sweep runner executes prepared configurations; it must not rewrite them.
Use these locations to identify the configuration under test:

| Area | Repository location |
| --- | --- |
| Runtime constants and orchestration | [`Flight/src/main.py`](../Flight/src/main.py) |
| Flight controllers | [`Flight/src/core/control/`](../Flight/src/core/control/) |
| Paths and planning | [`Flight/src/autonomy/`](../Flight/src/autonomy/) |
| Modes and failsafes | [`Flight/src/core/modes/`](../Flight/src/core/modes/) |
| Telemetry, odometry, and vision | [`Flight/src/sensing/`](../Flight/src/sensing/) |
| deterministic_v3 vision | [`Flight/src/sensing/vision/models/deterministic_v3/`](../Flight/src/sensing/vision/models/deterministic_v3/) |
| Gate mapping | [`Flight/src/mapping/gates/`](../Flight/src/mapping/gates/) |
| Recording behavior | [`Flight/src/core/logging/`](../Flight/src/core/logging/) |
| Run output | [`Flight/logs/runs/`](../Flight/logs/runs/) |

Before a sweep, create a repository-local manifest containing the sweep ID,
ordered trial IDs, configuration identifier, source revision/diff identity,
requested control duration, repeat number, and abort criteria. Hold every
non-swept setting constant.

## 3. Trial state machine

```text
PREFLIGHT -> START -> READY -> TIMED CONTROL -> FINALIZE
          -> DISARM -> RESET SIM -> VERIFY -> NEXT TRIAL
```

Never enter `NEXT TRIAL` until the preceding Flight process has exited,
recording is finalized, disarm is confirmed, and simulator reset is proven.

### Preflight

From the repository root:

```bash
lsof -nP \
  -iUDP:14550 -iUDP:14560 \
  -iUDP:5600 -iUDP:5601

df -h .
du -sh Flight/logs/runs
```

Required state:

- Simulator owns `14560` and `5601`.
- Nothing owns Flight ports `14550` and `5600`.
- The simulator PID is unchanged.
- Free space exceeds the sweep's estimated output plus its safety margin.
- The exact prepared configuration and trial duration match the manifest.

### Start a bounded trial

Run Flight in the foreground or as a monitored child process:

```bash
cd Flight/src

TRIAL_SECONDS=8.0 /usr/local/bin/python3.13 -c \
  'import os, main; main.RUN_S=float(os.environ["TRIAL_SECONDS"]); raise SystemExit(main.main())'
```

`TRIAL_SECONDS` is measured from Flight's `flight_began` boundary, after
connection, simulator reset, fresh telemetry/vision, IMU calibration,
vehicle-state initialization, and arming. Do not time a trial by sleeping
from process launch.

### Gate readiness and timing

Require these ordered events before accepting a trial:

```text
mavlink_connected
vision_started
simulator_reset_sent
simulator_settle_complete
mavlink_receiving
vision_receiving
stationary_imu_calibrated
vehicle_state_initialized
armed
flight_began
```

Record these durations with a monotonic host clock:

| Metric | Definition |
| --- | --- |
| Reset recovery | Last fresh-input event minus `simulator_reset_sent` |
| Controller readiness | `flight_began` minus process start |
| Controlled duration | `flight_finished` minus `flight_began` |
| Finalization | Process exit minus `flight_finished` |

Use simulator boot/race time for simulator outcomes and monotonic time for
process timing. Do not subtract vision epoch timestamps from IMU boot
timestamps. Apply identical readiness deadlines, duration, ordering, and
repeat count to every configuration.

### Finalize and close Flight

The normal bounded process must reach `flight_finished`, `shutdown`, save its
record, and exit. Require exit code `0` and `summary.json` with
`finalized=true`. Reject a trial containing `flight_exception` or missing
required events.

If Flight exceeds its duration plus shutdown allowance, send one `SIGINT` to
the exact Flight PID and wait for finalization:

```bash
kill -INT "$FLIGHT_PID"
```

Do not start another trial while Flight is serializing or still owns a port.
If finalization stalls, stop the sweep and investigate; repeated forced
termination can lose the consolidated record.

Confirm teardown:

```bash
lsof -nP -iUDP:14550 -iUDP:5600
```

Both ports must be free. Recheck output size and disk space before continuing.

### Disarm and reset the simulator

Flight shutdown does not reliably disarm V2. After Flight exits:

1. Use a one-shot repository `MavlinkClient` to connect, subscribe, send
   disarm, wait for `armed=false`, then shut the client down.
2. With `14550`/`5600` free again, execute **Restart the current run through
   the direct MAVLink connection** in
   [simulation_environment.md](simulation_environment.md).
3. Require simulator boot time to drop and fresh telemetry to resume.
4. Record disarm/reset results in the sweep manifest.

Do not infer reset from elapsed time alone. Do not reset while Flight remains
connected; competing clients can conflict and the old controller may keep
sending commands.

## 4. Trial acceptance

A trial is usable only when all conditions pass:

- Configuration identity and requested duration were recorded.
- Readiness events arrived in order within their deadlines.
- Post-reset telemetry and vision were fresh.
- `flight_began` and `flight_finished` bound the requested duration.
- The process exited successfully and recording finalized.
- No stale Flight process or port owner remained.
- Disarm was confirmed and simulator boot time proved the reset.
- Disk use remained below the sweep stop threshold.

Record collision, race/pass, failsafe, timing, and estimator metrics when
present. For V2, race/pass and collision packets do not contain absolute
position; any same-cycle position is an estimator correlation, not simulator
truth. Treat null `sim_truth` as unavailable truth, not as a zero-valued
measurement.

## 5. End the sweep

After the final trial:

1. Finalize and stop Flight.
2. Confirm `14550`/`5600` are free.
3. Explicitly disarm.
4. Reset the simulator run/physics state and verify the reset.
5. Confirm no recording continues and record final disk usage.
6. Mark the sweep manifest complete with accepted/rejected trial counts and
   reasons.

Leave the simulator GUI/runtime open unless the operator explicitly directs
otherwise. Environment shutdown or menu navigation is outside this runbook;
use [simulation_environment.md](simulation_environment.md).
