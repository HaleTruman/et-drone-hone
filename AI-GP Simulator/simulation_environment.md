# AI-GP simulator, CrossOver, Flight binding, and vision-calibration discovery

Date: 2026-07-30  
Workspace: `/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness`

Document status: **authoritative operational reference** for the simulator,
CrossOver launch, session control, Flight UDP binding, sensing configuration,
and vision-calibration workflow in this workspace.

## 1. Outcome

The repository contains all three pieces needed for a macOS simulation run:

1. A complete 64-bit Windows Unreal simulator build under
   `AI-GP Simulator/AIGP_3391/`.
2. A working CrossOver 26.2 installation and an existing, up-to-date,
   private `win10_64` bottle named `aigp-flightsim`.
3. A Flight runtime that listens for MAVLink on `127.0.0.1:14550`, listens
   for chunked JPEG vision on `0.0.0.0:5600`, and currently selects the
   `deterministic_v3` vision backend.

The intended local UDP topology is:

```text
CrossOver/Wine simulator                         Native macOS Flight

DCGame-Win64-Shipping.exe
  simulator MAVLink socket :14560  ----------->  127.0.0.1:14550
                                               MavlinkClient ("udpin")
                                  <-----------  MAVLink TIMESYNC/control

  simulator vision socket  :5601   ----------->  0.0.0.0:5600
                                               VisionStreamReceiver
                                               -> deterministic_v3
                                               -> VisionObservation
                                               -> GateMap/path/control/logs
```

The `14560` and `5601` values were confirmed as the live simulator-side
source/bound ports. Flight must own the complementary receive ports `14550`
and `5600`. Do not change Flight to bind `14560` or `5601`.

The Mac-side launch and port-binding path was validated after the initial
static discovery. CrossOver launched the existing `aigp-flightsim` bottle,
the `AI-GP` window and `DCGame-Win64-Shipping.exe` runtime remained alive,
D3DMetal was selected as the graphics backend, and the expected simulator
ports appeared. Flight was then connected and disconnected in controlled
runs, and both UI-driven and direct-MAVLink session restart procedures were
confirmed. The reproducible procedures are in sections 4 and 5.

## 2. What was verified on this Mac

### Host and CrossOver

| Item | Verified value |
| --- | --- |
| Host architecture | Apple Silicon, `arm64` |
| macOS | 15.6.1, build 24G90 |
| CrossOver command | `/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine` |
| CrossOver version | 26.2.0.39821 |
| CrossOver build | `cxoffice-26.2.0rc2`, timestamp `20260604T163056Z` |
| Bottle | `aigp-flightsim` |
| Bottle status | `uptodate` |
| Bottle mode | private |
| Bottle template | `win10_64` |
| Bottle architecture | `win64` |
| Existing simulator/Wine process | none at inspection time |
| Existing use of the four UDP ports | none at inspection time |

The bottle description is `AI-GP FlightSim validation bottle`. No explicit
graphics override is active in its bottle configuration, so the first test
should use CrossOver's current default graphics path.

The CrossOver wrapper itself advertises the options needed here:
`--bottle`, `--workdir`, `--cx-app`, `--wait-children`, `--verbose`, and
`--cx-log`.

### Simulator package

| Item | Finding |
| --- | --- |
| Package root | `AI-GP Simulator/AIGP_3391` |
| Package size | approximately 4.6 GiB |
| Bootstrap executable | `FlightSim.exe`, PE32+ x86-64 Windows GUI executable |
| Runtime executable | `FlightSim/Binaries/Win64/DCGame-Win64-Shipping.exe`, PE32+ x86-64 |
| Main content | `FlightSim/Content/Paks/FlightSim-WindowsNoEditor.pak`, approximately 4.3 GiB |
| Unreal support | Packaged Unreal Windows build with D3D11, D3D12, Vulkan, FMOD, PhysX, and other bundled DLLs |
| Build evidence | Manifest timestamps from 2026-07-23; package directory identifies revision 3391 |

`FlightSim.exe` is a small Unreal bootstrap. It uses Windows
`CreateProcessW` and hands off to `DCGame-Win64-Shipping.exe`. Launch the
root bootstrap with the package root as its working directory; do not make
the shipping binary the normal entry point.

### Native Python/Flight prerequisites

The repository-required interpreter, `/usr/local/bin/python3.13`, imported
the current integration successfully:

```text
numpy=2.4.4
cv2=4.13.0
scipy=1.17.1
pymavlink=2.4.49
```

`VisionPerceptionService` successfully lazy-loaded `DeterministicVision`
and resolved the in-repository LUT:

```text
Flight/src/sensing/vision/models/deterministic_v3/assets/color_lut_v1.npz
```

A native bind-and-release probe succeeded on both Flight receive endpoints:

```text
127.0.0.1:14550  OK
0.0.0.0:5600     OK
```

The relevant deterministic_v3 unit/configuration tests passed without
writing a pytest cache:

```text
49 passed in 0.94s
```

The tested files were `test_deterministic_vision.py`, `test_pipeline.py`,
`test_estimator_config.py`, and `test_final_3d_pose_estimate.py`.

## 3. Source-of-truth order

Several documents describe older states. For execution decisions, use this
precedence:

1. This `AI-GP Simulator/simulation_environment.md` document for the
   verified operational runbook, CrossOver/session procedures, port
   ownership, and known integration findings.
2. Current code in `Flight/src/main.py`, `Flight/src/sensing/`, and
   `Flight/src/sensing/vision/models/deterministic_v3/`.
3. The newly supplied official simulator example in
   `AI-GP Simulator/PyAIPilotExample/`.
4. Captured results in `Flight/docs/ai-gp-mavlink-udp.md` and
   deterministic_v3's `sweep/` records.
5. `Flight/README.md`, deterministic_v3's short `README.md`, and the
   simulator's general `README.md`.
6. Copied examples under `Flight/examples/`.

The current code remains the authority for what an executable actually does.
If it conflicts with this runbook, stop, verify the live behavior, and update
this document rather than silently relying on an older note. The startup
`AGENTS.md` is deliberately absent from this precedence because this document
replaces it.

### `AGENTS.md` cull audit

Every relevant startup-note item has a disposition in this document:

| Former startup-note item | Authoritative disposition here |
| --- | --- |
| CrossOver Wine entry point | Complete, validated command in section 4 |
| Monitored/foreground launch shape | `--wait-children`, process checks, and scoped shutdown in sections 4–5 |
| Primary runtime PID | Defined as the current macOS PID of `DCGame-Win64-Shipping.exe`; resolve per launch |
| Unreal runtime name | Confirmed as `DCGame-Win64-Shipping.exe` |
| Simulator ports | Live-confirmed `14560` and `5601`, with Flight complements `14550` and `5600` |
| `lsof` verification | Exact four-port command and expected Wine/Unreal ownership in sections 4–5 |
| Output locations | Consolidated in section 4; no simulator-owned repository output was confirmed |
| Raw background Wine warning | Retained; foreground child monitoring is the current supported procedure |
| Automatic stale-process cleanup | Rejected as unsupported: no such launcher exists; manual scoped preflight is required |
| Fixed-port/multi-instance warning | Retained: no simulator-side override is confirmed, so run one local instance |

No launch, session-control, port-binding, output, or failure-isolation
procedure depends on `AGENTS.md` after this audit. That file can be removed
without losing operational information.

Important documentation drift:

- `Flight/README.md` says the live loop constructs the CNN/regressor backend
  with `run_landmarker=False`. Current `Flight/src/main.py` instead selects
  `VisionPerceptionConfig(backend="deterministic_v3")`.
- `Flight/README.md` describes an older single-loop/control arrangement.
  Current code runs a 100 Hz inner loop and a 30 Hz outer/vision loop.
- deterministic_v3's short `README.md` still shows obsolete
  `mask_review_pipeline/...` commands and describes the published position
  as camera-optical. Current estimator code publishes absolute local-NED
  landmarks and labels them with
  `trace.position_semantics = "absolute_landmark"`.
- `sweep/RECOMMENDATION.md` reports a historical downstream double-add.
  The current `GateMap.update()` simply copies `gate.position_local_ned`
  into `GateRecord.position_local_ned_m`; that particular issue is no longer
  present in current code.

## 4. macOS CrossOver launch contract

### Canonical command

Run this from a terminal. All paths are quoted because both the workspace
and application paths contain spaces.

```bash
WORKSPACE="/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness"
SIM_ROOT="$WORKSPACE/AI-GP Simulator/AIGP_3391"
CROSSOVER_WINE="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine"

"$CROSSOVER_WINE" \
  --bottle aigp-flightsim \
  --workdir "$SIM_ROOT" \
  --wait-children \
  --verbose \
  "$SIM_ROOT/FlightSim.exe"
```

Why each part matters:

- `--bottle aigp-flightsim` prevents an accidental launch in CrossOver's
  default bottle.
- `--workdir "$SIM_ROOT"` preserves Unreal's expected relative lookup of
  `FlightSim/`, `Engine/`, content, and bundled DLLs.
- `"$SIM_ROOT/FlightSim.exe"` is the first non-option argument and launches
  the repository-native executable through Wine. Do not use `--cx-app` for
  this path: CrossOver interprets `--cx-app` as an application installed
  inside the bottle's Windows filesystem and rejects this native workspace
  path as not installed.
- `--wait-children` keeps the terminal attached after the bootstrap spawns
  `DCGame-Win64-Shipping.exe`; this avoids interpreting a bootstrap exit as
  a simulator failure.
- Foreground execution makes the first validation run observable and gives
  one place to collect wrapper errors.

For a diagnostic run, add an ephemeral CrossOver log:

```bash
  --cx-log /tmp/aigp-flightsim-crossover.log
```

Do not begin with a raw background command or `--no-wait`. The bootstrap can
return while its child is still starting, which makes process ownership and
failure detection ambiguous.

### First-launch graphics fallback

Start with no renderer override. If the runtime process exists but Unreal
fails before presenting a usable window, repeat the same command with
`-dx11` after the executable:

```bash
"$CROSSOVER_WINE" \
  --bottle aigp-flightsim \
  --workdir "$SIM_ROOT" \
  --wait-children \
  --verbose \
  --cx-log /tmp/aigp-flightsim-crossover.log \
  "$SIM_ROOT/FlightSim.exe" \
  -dx11
```

`-dx11` is a compatibility experiment, not a repository-established
requirement. The binary contains both D3D11 and D3D12 paths. Record which
path succeeds and keep the same path for all comparative vision runs so
rendering behavior is not another sweep variable.

Do not run `UE4PrereqSetup_x64.exe` preemptively. The package carries its
runtime DLLs and the existing bottle is already marked up to date. Use the
prerequisite installer only if the CrossOver log names a missing Microsoft
runtime component.

### Process and port monitor

In a second terminal:

```bash
ps -ax -o pid=,ppid=,command= \
  | rg -i 'CrossOver|wineserver|wine-preloader|DCGame|FlightSim'

lsof -nP \
  -iUDP:14550 -iUDP:14560 \
  -iUDP:5600 -iUDP:5601
```

Successful engine launch is established by:

1. A live `DCGame-Win64-Shipping.exe` process under the
   `aigp-flightsim` CrossOver runtime.
2. Simulator-side UDP ownership on `*:14560` and `*:5601` once the relevant
   qualifier/session is active.
3. A responsive Unreal window that reaches authentication and the target
   simulation session.

Record the macOS PID of `DCGame-Win64-Shipping.exe` as the **primary runtime
PID** for that launch. It is the PID used for scoped process checks and
process-targeted `AI-GP` menu automation. Resolve it again after every
launch; never reuse the validation-session PID:

```bash
ps -ax -o pid=,ppid=,etime=,state=,command= \
  | rg 'DCGame-Win64-Shipping\.exe FlightSim'
```

On the validated CrossOver/macOS combination, `lsof` represented each shared
simulator socket under both `wineserver` and the Unreal runtime. Expected
simulator-side lines are equivalent to:

```text
wineserver  ... UDP *:14560
wineserver  ... UDP *:5601
DCGame-Wi   ... UDP *:14560
DCGame-Wi   ... UDP *:5601
```

The network sockets may not become active at the initial menu. If the game
window is healthy but the ports are absent, authenticate and enter the
VQ_2/qualifier session before declaring a launch failure.

Simulator-side ports `14560` and `5601` appear fixed in revision 3391; no
supported override was found or validated. Do not assume multiple simulator
instances can share one macOS network namespace. Run one local simulator
instance at a time unless a simulator-side port override and full isolation
procedure are explicitly confirmed.

Stop the foreground test through the simulator UI first. If the child
survives the UI exit, use CrossOver's bottle UI to stop all applications in
`aigp-flightsim`. Do not kill every system Wine process indiscriminately;
other bottles may be in use.

The current foreground command does **not** automatically stop stale bottle
processes before launch. The older statement that a monitored launcher did
so was aspirational; no such repository-local launcher was found. Perform
the Phase A process/port checks, resolve exact stale processes to the
`aigp-flightsim` bottle, and stop them through the simulator or CrossOver UI
before relaunching. Never treat a blanket Wine kill as the runbook.

### Output and artifact locations

| Output | Location | Persistence/notes |
| --- | --- | --- |
| Optional CrossOver diagnostic log | `/tmp/aigp-flightsim-crossover.log` | Ephemeral; exists only when `--cx-log` is supplied |
| Flight run root | `Flight/logs/runs/run-<UTC timestamp>/` | Created by importing/running `Flight/src/main.py` |
| Flight consolidated run record | `Flight/logs/runs/run-<UTC timestamp>/run.json` | Saved during graceful Flight shutdown |
| Captured vision frames | `Flight/logs/runs/run-<UTC timestamp>/vision_frames/` | Written after the simulator reset/startup sequence enables saving |
| deterministic_v3 calibration artifacts | `Flight/src/sensing/vision/models/deterministic_v3/sweep/` | Repository-local caches/results governed by the calibration procedure |

No simulator-owned repository output directory was defined or confirmed for
revision 3391. Do not invent one or redirect simulator data outside this
workspace. CrossOver bottle internals are runtime state, not calibration
artifacts or a substitute for the Flight run directory.

## 5. End-to-end launch, bind, and monitor procedure

### Phase A: preflight

From the workspace root:

```bash
test -f "AI-GP Simulator/AIGP_3391/FlightSim.exe"
test -f "AI-GP Simulator/AIGP_3391/FlightSim/Binaries/Win64/DCGame-Win64-Shipping.exe"
test -f "AI-GP Simulator/AIGP_3391/FlightSim/Content/Paks/FlightSim-WindowsNoEditor.pak"

"/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine" --version

"/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/cxbottle" \
  --bottle aigp-flightsim \
  --status

lsof -nP \
  -iUDP:14550 -iUDP:14560 \
  -iUDP:5600 -iUDP:5601
```

Pass conditions:

- All three simulator files exist.
- CrossOver reports 26.2.0 or the deliberately selected replacement.
- The bottle reports `Status=uptodate`.
- No stale process owns any of the four ports.

### Phase B: launch the simulator

Use the canonical command in section 4. Enter the expected simulator
account/session. In the monitor terminal, wait for:

```text
DCGame-Win64-Shipping.exe
UDP *:14560
UDP *:5601
```

Do not start Flight until the simulator window is stable and the intended
session is selected. `Flight/src/main.py` waits for MAVLink before it starts
its vision listener, so simulator-first is the reliable order for the
current entry point.

### Phase B.1: control only the current simulator run

These procedures operate on the current run without relaunching CrossOver,
the bottle, `FlightSim.exe`, or `DCGame-Win64-Shipping.exe`. They were
validated on simulator `AIGP 1.0.3391` on July 30, 2026.

#### Return from the active SIM state through its parent menu

The verified UI sequence is:

```text
Escape -> Down Arrow once -> Enter
```

Target the simulator process exactly. Do not post a global keyboard/HID
event: if the simulator is not frontmost, a global event can reach the
terminal or another application.

First resolve the current macOS host PID. The PID changes on every simulator
launch, so the value `34122` from the validation session must not be
hardcoded:

```bash
ps -ax -o pid=,command= \
  | rg 'DCGame-Win64-Shipping\.exe FlightSim'
```

Set the single PID returned for the intended `AI-GP` window, then send the
three keys through that exact Accessibility process:

```bash
SIM_HOST_PID=34122  # example only; replace with the PID from ps

osascript - "$SIM_HOST_PID" <<'APPLESCRIPT'
on run argv
    set simulatorPid to (item 1 of argv) as integer

    tell application "System Events"
        set simulatorProcess to first application process whose unix id is simulatorPid
        set frontmost of simulatorProcess to true
        delay 0.2
        tell simulatorProcess to key code 53 -- Escape
        delay 0.25
        tell simulatorProcess to key code 125 -- Down Arrow
        delay 0.25
        tell simulatorProcess to key code 36 -- Enter/Return
    end tell
end run
APPLESCRIPT
```

This command requires macOS Accessibility permission for the terminal or
automation host executing `osascript`. Before sending input, confirm that
the resolved process is `DCGame-Win64-Shipping.exe` and owns the `AI-GP`
window. Do not infer UI success solely from continued UDP traffic: the
simulator can continue publishing MAVLink/race status while an in-game menu
is open. Operator confirmation of the visible state remains the acceptance
signal for this UI transition.

#### Exit to the environments menu

The operator-confirmed sequence for leaving the current run menu and
returning to the environments menu is:

```text
Escape -> Down Arrow -> Down Arrow -> Down Arrow -> Enter
```

> **Caution:** This is an intentional, state-changing menu action. Do not
> execute it as a probe, readiness check, automated cleanup step, or inferred
> recovery action. Use it only when the operator explicitly directs the
> current run to be exited to the environments menu. Confirm the target PID
> before sending input and wait for operator visual confirmation afterward.

This sequence is distinct from restarting the current run and from the
single-Down-Arrow parent-menu action above. Resolve the current primary
runtime PID again; never reuse the example PID from a prior launch. Then
target only that `DCGame-Win64-Shipping.exe` Accessibility process:

```bash
SIM_HOST_PID=34122  # example only; replace with the PID from ps

osascript - "$SIM_HOST_PID" <<'APPLESCRIPT'
on run argv
    set simulatorPid to (item 1 of argv) as integer

    tell application "System Events"
        set simulatorProcess to first application process whose unix id is simulatorPid
        set frontmost of simulatorProcess to true
        delay 0.2
        tell simulatorProcess to key code 53 -- Escape
        delay 0.25
        tell simulatorProcess to key code 125 -- Down Arrow 1
        delay 0.2
        tell simulatorProcess to key code 125 -- Down Arrow 2
        delay 0.2
        tell simulatorProcess to key code 125 -- Down Arrow 3
        delay 0.25
        tell simulatorProcess to key code 36 -- Enter/Return
    end tell
end run
APPLESCRIPT
```

Do not send additional input after Enter. The acceptance signal is explicit
operator confirmation that the environments menu is visible; continued or
stopped UDP traffic alone is not sufficient to determine this UI state.

#### Restart the current run through the direct MAVLink connection

Use this when the simulator GUI/runtime must remain open but its current run
must restart. Flight must not be running: the one-shot client temporarily
owns `127.0.0.1:14550`, and a live Flight process would conflict with it and
could continue sending control.

Preflight:

```bash
lsof -nP -iUDP:14550 -iUDP:5600
```

Both ports must have no native Flight owner. Then run the repository client:

```bash
cd "/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness/Flight/src"

/usr/local/bin/python3.13 - <<'PY'
import time

from sensing.telemetry import MavlinkClient

client = MavlinkClient(
    endpoint="udpin:127.0.0.1:14550",
    sim_runtime="VQ_2",
)

try:
    client.connect(heartbeat_timeout_s=10.0)
    client.subscribe_telemetry()
    time.sleep(0.75)

    before_boot_us = (
        None if client.latest_imu is None else client.latest_imu.time_boot_us
    )
    before_race = client.race_status

    client.send_sim_reset_command()  # repository MAVLink command 31000
    print("sim_reset_command_sent=31000", flush=True)
    time.sleep(2.0)

    after_boot_us = (
        None if client.latest_imu is None else client.latest_imu.time_boot_us
    )
    after_race = client.race_status

    print(f"before_boot_us={before_boot_us}", flush=True)
    print(f"after_boot_us={after_boot_us}", flush=True)
    print(f"before_race={before_race}", flush=True)
    print(f"after_race={after_race}", flush=True)
    print(f"armed_after_reset={client.armed}", flush=True)

    if before_boot_us is None or after_boot_us is None:
        raise RuntimeError("No HIGHRES_IMU sample available to verify reset")
    if after_boot_us >= before_boot_us:
        raise RuntimeError("Simulator boot time did not reset")
finally:
    client.shutdown()
PY
```

The direct reset is confirmed when simulator boot time drops from its
pre-command value to a new low value and fresh telemetry continues. In the
validated run it dropped from about `26.37 s` to `1.50 s`. The command does
not restart the Unreal process or GUI. Simulator revision 3391 may report
itself armed again after reset; record that state explicitly rather than
assuming reset implies disarm.

After the one-shot process exits, confirm that the Flight-side ports are
unowned and the original Unreal PID remains:

```bash
lsof -nP \
  -iUDP:14550 -iUDP:14560 \
  -iUDP:5600 -iUDP:5601

ps -p "$SIM_HOST_PID" -o pid=,lstart=,etime=,state=,command=
```

Expected result: no owner on `14550` or `5600`, while the existing simulator
continues to own `14560` and `5601`.

### Phase C: start Flight

The current native command is:

```bash
cd "/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness/Flight"
PYTHONPATH=src /usr/local/bin/python3.13 src/main.py
```

This is not a passive connection test. Current `main.py`:

1. Creates a new run directory as soon as the module is imported.
2. Binds MAVLink through `udpin:127.0.0.1:14550`.
3. Waits up to 120 seconds for a heartbeat.
4. Starts 10 Hz TIMESYNC requests and the MAVLink receive thread.
5. Binds the vision socket to `0.0.0.0:5600`.
6. Sends simulator reset command `31000`.
7. Starts saving frames after the reset delay.
8. Collects stationary IMU samples.
9. Initializes vehicle state.
10. Arms the simulated vehicle.
11. Runs a vision/gate-map initialization window.
12. Enters a 100 Hz control loop with `ALLOW_FLIGHT=True`.

Only run that command when reset, arm, and closed-loop commands are intended.
Neither `AI-GP Simulator/PyAIPilotExample/main.py` nor the copies under
`Flight/examples/` are passive alternatives: they also arm and continuously
send control.

#### Coordinated environment/controller restart

There is no atomic in-process controller reset. The validated alignment
mechanism is process replacement plus Flight's ordered startup:

1. Stop the old Flight process, explicitly disarm through a one-shot
   `MavlinkClient`, and confirm `14550`/`5600` are free.
2. Start a new Flight process. This reconstructs the estimator, perception,
   map, planner, mode manager, and controllers.
3. Flight binds MAVLink and vision, sends simulator reset command `31000`,
   and waits the configured `1.5 s` settle delay.
4. Flight clears pre-reset telemetry and frame buffers, resets VIO, then
   requires a fresh `HIGHRES_IMU` sample and vision frame.
5. Only after fresh inputs arrive does Flight calibrate the IMU, initialize
   vehicle state, arm, and begin control.

This aligns the controller to post-reset inputs, but it is not a reset
acknowledgement protocol: completion is inferred from the fixed delay,
restarted simulator boot time, and advancing telemetry/vision.

A 2026-07-30 validation used three consecutive new Flight processes, each
with a three-second bounded control interval. Times below are seconds from
process startup:

| Cycle | Reset sent | Telemetry fresh | Vision fresh | State initialized/armed | Control began | Shutdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.080 | 1.607 | 1.608 | 3.111 | 4.637 | 7.740 |
| 2 | 0.048 | 1.578 | 1.603 | 3.112 | 4.645 | 7.702 |
| 3 | 0.028 | 1.557 | 1.603 | 3.113 | 4.631 | 7.699 |
| Mean | 0.052 | 1.581 | 1.605 | 3.112 | 4.637 | 7.714 |

All three cycles completed without startup or shutdown exceptions. Simulator
boot time was `1.01–1.03 s` when fresh telemetry was accepted, providing
direct evidence that each environment reset landed. State initialization
varied by about `1 ms`; control start varied by about `14 ms`.

Two limitations prevent treating this as a fully reliable reset contract:

- Normal Flight shutdown reported the simulator still armed after two of
  three cycles. Always perform and verify an explicit disarm between runs.
- Vision UDP reacquisition succeeded, but deterministic_v3 repeatedly raised
  `VisionObservation` has no attribute `to_controller_payload`; transport
  alignment does not yet prove usable vision-to-controller observations.

### Phase D: confirm the complete bind

With the simulator and Flight both active:

```bash
lsof -nP \
  -iUDP:14550 -iUDP:14560 \
  -iUDP:5600 -iUDP:5601
```

Expected ownership:

| Port | Expected owner | Role |
| ---: | --- | --- |
| 14550 | native Python/Flight | MAVLink receive endpoint |
| 14560 | Wine/`DCGame-Win64-Shipping.exe` | simulator MAVLink source/reply endpoint |
| 5600 | native Python/Flight | FPV JPEG receive endpoint |
| 5601 | Wine/`DCGame-Win64-Shipping.exe` | simulator vision source endpoint |

Only one native Flight process may own `14550` and `5600`. The live
`VisionStreamReceiver` does not set `SO_REUSEADDR`; a second receiver should
fail fast with `Address already in use`.

### Phase E: monitor data health

The Flight console should reach messages equivalent to:

```text
MAVLink client connected...
MAVLink heartbeat started...
MAVLink subscribed to telemetry...
Vision receiver started...
```

The active run is announced as:

```text
Starting run at .../Flight/logs/runs/run-<timestamp>...
```

In another terminal, set the announced path:

```bash
RUN_DIR="/absolute/path/printed/by/Flight"

tail -f "$RUN_DIR/lists/events.jsonl"
tail -f "$RUN_DIR/lists/telemetry.jsonl"
tail -f "$RUN_DIR/lists/vision_observations.jsonl"
tail -f "$RUN_DIR/frames.jsonl"
```

Inspect the rolling status without relying on the console:

```bash
python3 -m json.tool "$RUN_DIR/status.json"
```

End-to-end acceptance requires all of the following:

- Heartbeat received and `connected=true`.
- Repeating `HIGHRES_IMU` samples with increasing `time_boot_us`.
- TIMESYNC replies after Flight starts sending requests.
- Completed frames with increasing `frame_id` and `sim_time_ns`.
- JPEG files named
  `frame-<eight-digit-frame-id>-<sim_time_ns>.jpg`.
- `invalid_packet_count` is not increasing continuously.
- The perception snapshot names `deterministic_v3`.
- Each processed frame produces a valid `VisionObservation`, including an
  empty `gates` list when nothing is detected.
- Shutdown finalizes `run.json`, `telemetry.json`, `gate_map.json`,
  `summary.json`, and JSONL sidecars.

## 6. Flight transport and sensing contract

### MAVLink

The active endpoint is hardcoded in `Flight/src/main.py`:

```python
MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
SIM_RUNTIME = "VQ_2"
```

`pymavlink`'s `udpin` connection listens on `14550` and learns the source
address of incoming datagrams. Flight's TIMESYNC, arm/reset, and control
messages are sent back through that connection to the simulator's source
endpoint, expected to be `14560`.

Current VQ_2 message handling:

| Message | Use |
| --- | --- |
| `HEARTBEAT` | connection, armed state, system status |
| `TIMESYNC` | cached response to Flight's 10 Hz requests |
| `HIGHRES_IMU` | authoritative raw acceleration and gyro input |
| `ACTUATOR_OUTPUT_STATUS` | cached actuator state |
| `ENCAPSULATED_DATA`, type 1 | race status |
| `COLLISION` | collision event cache, if emitted |

The current VQ_2 runtime intentionally does not use MAVLink `ODOMETRY`,
`ATTITUDE`, or `LOCAL_POSITION_NED`. Those messages are only accepted as
simulation truth when `sim_runtime == "VQ_1"`. For VQ_2, local position,
velocity, and attitude are generated by the native
`VehicleStateEstimator` from IMU integration, with optional VIO currently
disabled.

#### V2 training-simulator findings

These 2026-07-30 findings apply only to the V2 (`VQ_2`) training simulator;
equivalent tests have **not** been conducted in the V1 (`VQ_1`) environment.

| Finding | V2 evidence |
| --- | --- |
| Recording teardown | Flight released `14550`/`5600`, the final record stopped growing, and the simulator remained open. |
| Gate pass | An active test advanced `active_gate_index` from `0` to `1` at race time `3.866 s`. Race status contains no position; the same-cycle `[12.973, -0.086, -0.770] m` value was an IMU-derived estimate. |
| Collision | No collision was recorded during the active tests. A passive live inventory did receive `COLLISION` packets (`id=1001`, threat `1`), proving stream availability; packets contain separation deltas, not absolute position. |
| Truth data | Every tested `sim_truth` value was null, and no `ODOMETRY`, `ATTITUDE`, or `LOCAL_POSITION_NED` message was observed. Vehicle and gate positions were estimator/perception outputs, not simulator truth. |

The July 3 passive capture in `Flight/docs/ai-gp-mavlink-udp.md` observed
approximately:

- `HIGHRES_IMU`: 116.65 Hz.
- `ACTUATOR_OUTPUT_STATUS`: 95.15 Hz.
- `HEARTBEAT`: 9.95 Hz.
- race-status `ENCAPSULATED_DATA`: 3.95 Hz.

These are useful health baselines, not hard protocol guarantees.

### Vision UDP protocol

The live receiver binds:

```python
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
```

Every datagram begins with the 24-byte little-endian header:

```text
<IHHIIQ

uint32 frame_id
uint16 chunk_id
uint16 total_chunks
uint32 jpeg_size
uint32 payload_size
uint64 sim_time_ns
```

The rest of the datagram is a JPEG chunk. The current receiver:

- Rejects datagrams shorter than the header.
- Confirms `payload_size`.
- Confirms positive chunk count and JPEG size.
- Confirms `chunk_id < total_chunks`.
- Reassembles chunks in numerical order.
- Confirms reconstructed JPEG length.
- Holds at most 120 complete frames and 30 partial frames.
- Saves completed frames only after `begin_saving_frames()`.

In the live outer loop, `get_latest_frame()` keeps the newest complete frame
and clears older queued frames. This is a low-latency policy, not a
lossless-capture policy.

### Observation flow

The current source path is:

```text
simulator UDP chunks
  -> VisionStreamReceiver.process_packet()
  -> VisionFrame(frame_id, sim_time_ns, jpeg_bytes, saved_path)
  -> VisionPerceptionService.process_vision_frame()
  -> DeterministicVision.process_frame()
  -> mask
  -> fill
  -> bridge
  -> void center and clipping
  -> instance tracking
  -> final 2D void estimate
  -> final 3D pose estimator
  -> VisionObservation
  -> GateMap.update()
  -> PathManager
  -> controller
  -> MAVLink SET_ATTITUDE_TARGET
```

The deterministic_v3 backend is stateful. One `DeterministicVision`
instance owns one `InstanceTracker`, one `FinalVoidEstimator`, and one
`Final3DPoseEstimator` for the life of the service. Frame order and
continuity affect identities, ghost/reconnection behavior, confidence, and
3D landmark history.

The output contract is:

```text
VisionObservation
  frame_id
  sim_time_ns
  gates[]
  source
  trace

VisionGateObservation
  gate_id
  position_local_ned
  position_confidence
  orientation_local_ned_quat (optional)
  orientation_confidence (optional)
  trace
```

Current deterministic_v3 positions are absolute local-NED landmark
coordinates, not camera-relative coordinates. The estimator also records
the camera-relative NED vector in trace for diagnostics.

## 7. Vision capture and calibration procedure

### Live capture

Use `Flight/src/main.py` only for an intentional closed-loop run. Its run
directory becomes the self-contained calibration input:

```text
Flight/logs/runs/run-<timestamp>/
  run.json
  telemetry.json
  gate_map.json
  frames.jsonl
  vision_frames/
    frame-<frame_id>-<sim_time_ns>.jpg
  lists/
    telemetry.jsonl
    vision_observations.jsonl
    ...
```

For a calibration run, record:

- CrossOver version and bottle.
- simulator revision/package (`AIGP_3391`).
- render path, including whether `-dx11` was used.
- VQ runtime (`VQ_2`).
- Flight commit/worktree identity.
- deterministic_v3 LUT path.
- estimator configuration, including a diff from defaults.
- run start/end, frame count, telemetry count, and any packet drops.

Do not compare sweeps collected with different renderer, resolution,
camera-FOV, camera-tilt, or frame-rate settings as if they were only
estimator changes.

### Offline full-pipeline replay

Run from the deterministic_v3 directory so the default relative LUT resolves
inside this repository:

```bash
cd "/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness/Flight/src/sensing/vision/models/deterministic_v3"

PYTHONPATH=src /usr/local/bin/python3.13 \
  src/pipeline.py \
  "/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness/Flight/logs/runs/run-<timestamp>" \
  --review-root review/default-config
```

For a small smoke replay, add `--limit 120`.

The batch path expects:

- `vision_frames/` under the run.
- `telemetry.json` alongside it for camera pose.
- filenames in the exact
  `frame-<frame-id>-<sim_time_ns>.jpg` form.

The default pipeline and the live `DeterministicVision` share the same
upstream `advance_analysis()` sequence. Batch mode can use recorded-frame
lookahead; live mode cannot.

### Recommended-config replay

The repository has a separate driver that preserves baseline artifacts:

```bash
PYTHONPATH=src /usr/local/bin/python3.13 \
  sweep/run_recommended_pipeline.py \
  "/Users/trumanhale/0720-0802_et_drone_hone/ft-vision-sim-harness/Flight/logs/runs/run-<timestamp>" \
  --review-root review/recommended-config
```

It applies:

```python
EstimatorConfig(
    merge_radius_m=9.0,
    beam_radius_m=0.55,
    robust_loss_scale_m=0.10,
)
```

Treat this as a measured hypothesis, not a settled production default. Its
six-run result was modest, one holdout run regressed, and the corrected
multiple-comparison evidence is not conclusive.

### Sweep sequence

The intended order is:

```bash
python sweep/build_and_verify.py
python sweep/build_ground_truth.py
python sweep/run_sweep.py baseline arm1 arm2 arm3 arm4 arm5
python sweep/analyze.py
python sweep/validate_final.py
```

The equivalence gate is blocking: a default-config replay must reproduce the
full-pipeline baseline frame for frame before any configuration comparison
is meaningful.

The current sweep cannot be regenerated under this repository's confinement
rule without a later code change. `build_and_verify.py` and
`ground_truth.py` hardcode run/review roots under:

```text
/Users/trumanhale/ft-vision-0723-experimentation/...
```

That is outside the permitted workspace. Existing caches/results are
available in-repository, but new sweeps should first make these inputs
explicit CLI arguments and point them into `Flight/logs/runs/` and an
in-repository review root. This discovery did not make that change because
only this report was authorized.

## 8. Readiness gaps and calibration risks

These findings should be resolved or consciously accepted before declaring
an end-to-end calibration authoritative.

### P0: body-rate command compatibility

The official example shipped beside revision 3391 says simulator revision
3390 introduced extension bit `16` in `SET_ATTITUDE_TARGET.type_mask`:

```python
ATTITUDE_TARGET_TYPEMASK_DCL_BODY_RATES_RADS = 16
```

Setting it opts into documented physical rad/s interpretation. Current
`MavlinkClient.send_attitude_target()` sends body rates with only
`ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` and omits bit `16`.

The current controller emits `body_rates_rps`, so this is a closed-loop
behavior risk. It does not prevent telemetry or vision capture, but it can
change the trajectory used to generate calibration data. Resolve and test
this before comparing closed-loop sensing sweeps.

### P0: live estimator configuration is not plumbed

The offline estimator has a comprehensive frozen `EstimatorConfig`, and the
sweep's recommended combination is available in
`run_recommended_pipeline.py`. The live chain does not expose it:

- `DeterministicVisionV3Config` contains only `lut_path`.
- `VisionPerceptionService` constructs `DeterministicVision(lut_path=...)`.
- `DeterministicVision` constructs `Final3DPoseEstimator()` with defaults.

Therefore the live runtime currently uses default estimator values, not the
recommended sweep values. A live comparison cannot claim to test a swept
configuration until the configuration reaches this constructor and is
logged with the run.

### P0: telemetry and frame capture do not begin together

`main.py` begins saving frames after reset, then spends time collecting
stationary IMU and running vision initialization. Telemetry samples are
written by `logger.log_cycle()` only after entry into the main 100 Hz loop.

The existing sweep found that prior captures had about 3.1 seconds, or
85-87 frames, before the first valid camera pose. Current startup ordering
still has the same structural problem and adds explicit IMU and gate-map
initialization windows. Early saved frames can therefore have no matching
telemetry, causing deterministic_v3 to emit
`missing_or_stale_camera_pose`.

Widening the 0.25-second telemetry tolerance is not a valid fix when no
earlier pose exists. Frame and telemetry logging need a common capture
window.

### P1: frame/state synchronization is approximate

`VisionPerceptionService._vehicle_state_for_frame()` does not look up or
interpolate a state at the frame timestamp. If timestamps differ, it copies
the supplied current state and merely replaces its `sim_time_ns` with the
frame timestamp. This bypasses the estimator's stale-time guard without
changing the pose values.

There is a `DataSynchronizer` in `sensing/telemetry/sync.py`, but current
`main.py` does not use it. Fast motion, queued vision work, and dropped
frames can therefore produce a pose/frame mismatch that appears perfectly
time-aligned to deterministic_v3.

### P1: live frame dropping affects a stateful pipeline

The 30 Hz outer loop calls `get_latest_frame()`, which clears all older
complete frames. Only one perception future can be pending. When processing
is slower than the frame stream, intermediate frames are discarded before
the stateful tracker/estimator sees them.

The measured deterministic_v3 p95 was about 65 ms/frame at defaults and
about 54 ms/frame for the recommendation, both over a 33 ms budget. Frame
drops are therefore expected under load, and they alter track identity,
missing counters, confidence decay, landmark history, and final metrics.
Monitor actual processed-frame continuity rather than only receiver FPS.

### P1: simulator track ground truth is discarded

The official simulator example implements track-data type `2`, including
handshake/chunk reassembly and per-gate NED position, quaternion, width, and
height. Current `MavlinkClient` handles only encapsulated type `1` race
status. `core.schema.TrackGate` exists, but the current client never
populates it.

This ground truth may be nulled in some qualifier modes, as the official
example comments warn, and the July 3 capture did not observe it. The client
should nevertheless decode and record it when present. A single successful
capture would greatly improve calibration truth and determine whether VQ_2
still publishes it.

### P1: VQ_2 state is dead-reckoned

With `SIM_RUNTIME="VQ_2"` and `ENABLE_VIO=False`, the vehicle pose used for
vision rays is IMU-integrated rather than simulator truth. Accelerometer-only
initialization cannot observe yaw, and integration drift grows over a run.
Existing ground-truth analysis shows range-dependent along-track
disagreement consistent with either state drift or camera-model error.

### P1: hardcoded camera model

deterministic_v3 uses:

```text
fx=320, fy=320, cx=320, cy=180
camera tilt=20 degrees
body-to-camera translation=zero
```

The intrinsics and tilt are duplicated between `camera_odometry.py` and
`final_3d_pose_estimate.py`. They are not part of `EstimatorConfig`, and the
3D estimator assumes the camera origin equals the vehicle origin. Any
simulator render-resolution, FOV, camera-angle, or lever-arm mismatch will
look like estimator error.

### P2: examples are active and divergent

The simulator-side `PyAIPilotExample` is useful protocol evidence but is not
a safe passive probe: it arms, then loops control at 250 Hz. The copies in
`Flight/examples/pypilot` and `Flight/examples/PyAIPilotExample-v2` differ
from the newly supplied official example. In particular, the Flight copies
omit the rad/s extension bit.

Use current `Flight/src/sensing/` as the implementation source and the
new simulator example as the protocol-delta reference. Do not copy one
example wholesale over the other.

### P2: runtime configuration is hardcoded

Endpoints, VQ runtime, rates, reset/arm behavior, flight enablement, backend,
and output behavior are module constants in `Flight/src/main.py`; there is
no CLI or environment configuration surface. Some declared constants
(`RESET_READY_TIMEOUT_S`, reset stability values, and `TARGET_HOLD_S`) are
not used by the current startup path. `CONTROL_METHOD` is logged as
`carrot_motor_test`, while the loop actually sends attitude/body-rate
targets.

This makes run metadata less trustworthy than the code unless the effective
values are recorded explicitly.

### P2: dependency record drift

deterministic_v3's `requirements.md` names future versions
`numpy==2.5.1` and `opencv-python==5.0.0.93`, while the tested working host
has NumPy 2.4.4 and OpenCV 4.13.0. The current tests pass, but a reproducible
calibration environment needs one installable, repository-local dependency
contract.

## 9. Failure isolation

| Symptom | First checks | Interpretation/action |
| --- | --- | --- |
| `wine` command returns and no runtime remains | Use `--wait-children`; confirm `--workdir`; inspect `ps` and `/tmp/aigp-flightsim-crossover.log` | Bootstrap/child monitoring or early Unreal failure |
| `DCGame-Win64-Shipping.exe` exists but no window | CrossOver log; retry one controlled `-dx11` run | Graphics initialization issue |
| Window works but `14560`/`5601` absent | Enter authenticated VQ_2 session; re-run `lsof` | Stream may not start at menus |
| Flight reports address in use | `lsof` on `14550` and `5600`; stop stale native clients | Only one Flight receiver is allowed |
| Flight waits forever for heartbeat | Confirm simulator owns `14560`; confirm it sends to host `14550`; confirm endpoint remains `udpin:127.0.0.1:14550` | MAVLink path missing |
| Heartbeat works, vision timeout | Confirm simulator `5601`, Flight `5600`, and session camera stream; inspect `invalid_packet_count` | Vision stream/config/protocol problem |
| Frames save but no observations | Confirm backend snapshot and LUT; inspect vision future errors | Perception initialization/runtime problem |
| Observations are always empty with `missing_or_stale_camera_pose` | Compare first frame and telemetry timestamps | Capture-start or state-alignment problem |
| High dropped-frame/id churn | Compare input frame IDs with processed observations; measure perception p95 | deterministic_v3 cannot sustain 30 Hz |
| Closed-loop trajectory differs drastically from expected | Check MAVLink body-rate bit `16` before tuning controls | Simulator rate-unit compatibility |
| Sweep scripts access an external experiment tree | Stop; do not relax the repository rule | Parameterize run/review roots into this workspace in a separate authorized change |

## 10. Acceptance checklist

### CrossOver/engine

- [ ] CrossOver wrapper and bottle status pass.
- [ ] Root bootstrap is launched with the package root as working directory.
- [ ] The current primary runtime PID is resolved and recorded.
- [ ] `DCGame-Win64-Shipping.exe` remains alive.
- [ ] No second local simulator instance competes for fixed ports.
- [ ] Simulator UI reaches the intended authenticated VQ_2 session.
- [ ] Simulator owns UDP `14560` and `5601`.
- [ ] Renderer choice is recorded and held constant.
- [ ] CrossOver diagnostics and Flight run-output locations are recorded.

### Flight binding

- [ ] No stale owner exists on `14550` or `5600`.
- [ ] Flight owns `127.0.0.1:14550`.
- [ ] Flight owns `0.0.0.0:5600`.
- [ ] Heartbeat, HIGHRES_IMU, actuator status, race status, and TIMESYNC are observed.
- [ ] Frame IDs and timestamps advance.
- [ ] Packet errors/dropped partial frames are monitored.

### Sensing and calibration

- [ ] Snapshot reports `backend=deterministic_v3`.
- [ ] LUT path is the in-repository deterministic_v3 asset.
- [ ] Frame and vehicle-state timestamps are evaluated for real alignment.
- [ ] Effective `EstimatorConfig` is recorded; do not infer it from a recommendation document.
- [ ] Processed-frame gaps are measured.
- [ ] Absolute local-NED semantics are preserved through GateMap/planning.
- [ ] Frame and telemetry capture start together before scoring earliness.
- [ ] Default replay passes the equivalence gate before a sweep.
- [ ] Ground-truth uncertainty and holdout separation are kept intact.
- [ ] All run, review, cache, and result inputs remain inside this workspace.

## 11. Recommended next implementation sequence

This report is discovery only; none of these changes were made.

1. Add a repository-local monitored launcher around the canonical CrossOver
   command, with foreground child monitoring, exact bottle selection, port
   checks, an ephemeral diagnostic log, and scoped shutdown behavior.
2. Add a passive native preflight command that binds `14550`/`5600`, reports
   message/frame rates, and never resets, arms, or sends control.
3. Reconcile the official rad/s type-mask bit with
   `MavlinkClient.send_attitude_target()` and add a protocol test.
4. Expose `EstimatorConfig` through `DeterministicVisionV3Config`, inject it
   into `Final3DPoseEstimator`, and serialize the effective configuration
   into every live run.
5. Start telemetry logging at the same boundary as frame saving.
6. Use a timestamped vehicle-state history/interpolator rather than
   relabeling the current state with the frame time.
7. Decode and log track-info type `2` when the simulator emits it.
8. Parameterize sweep run/review roots and keep all new calibration inputs
   under `Flight/`.
9. Add processed-frame continuity and end-to-end latency metrics before
   comparing more estimator settings.

At that point the launch, bind, capture, replay, and sweep chain will be
reproducible entirely within this workspace and suitable for authoritative
vision calibration.
