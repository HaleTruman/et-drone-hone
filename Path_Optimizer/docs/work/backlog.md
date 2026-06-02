# Path Optimizer Production Readiness Backlog

Treat the domain packages under `src/` as the future live product, `src/autonomy/planning/mpcc/` as the planning core, and preserve `src/simulator/` as first-class test infrastructure.

## P0: Establish The Production Shape

- [ ] Define the canonical live runtime in `src/main.py`.
- [ ] Define one canonical offline entrypoint for deterministic simulation and replay.
- [ ] Keep `src/drone.py` as a fast telemetry/control-loop rig, but give it the same interfaces as the live stack.
- [x] Consolidate deterministic telemetry and the richer hover/track simulator under `src/simulator/`.
- [ ] Define explicit schemas for state, telemetry, gate observations, reference paths, planner results, controller outputs, and logs.
- [ ] Centralize LOCAL_NED, Unreal coordinates, units, quaternion order, and gate through-axis conventions.
- [ ] Decide which simulator interface is authoritative: Unreal WebSocket, MAVLink UDP, or an adapter supporting both.

## P0: Make MPCC Release-Capable

- [ ] Extend `src/autonomy/planning/mpcc/planner.py` from exactly one gate to the next 1-3 visible gates.
- [ ] Replace placeholder `theta=1.0` with real optimized progress.
- [ ] Shift warm-start trajectories forward between planning cycles instead of simply reusing the previous solve.
- [ ] Replace piecewise-linear references with smooth multi-gate splines.
- [ ] Enforce gate aperture, depth, sequential crossing, and through-axis alignment constraints.
- [ ] Add obstacle and free-space constraints when the simulator interface exposes them.
- [ ] Add solver timeout, infeasibility handling, degraded fallback, and last-known-good trajectory behavior.
- [ ] Set and measure a real-time solve budget for the intended planning frequency.
- [ ] Remove parameter drift: MPCC currently uses a hardcoded `0.75 kg` model while `src/core/quadrotor/params.yaml` uses `1.2 kg`.
- [ ] Add numerical parity tests between CasADi dynamics and the NumPy `Quadrotor` model.

## P0: Implement The Live Loop

- [ ] Implement the production control loop in `src/main.py`.
- [ ] Replace the transport-neutral `MavlinkBridge` with the selected MAVLink client.
- [ ] Implement connect, reconnect, heartbeat, subscription, command masks, system/component IDs, and clean shutdown.
- [ ] Verify the external telemetry contract. Standard `HIGHRES_IMU` does not normally carry velocity; the current simulator adds a custom extension.
- [ ] Implement UDP vision frame parsing, chunk reassembly, bounded buffering, packet-loss handling, and JPEG decoding.
- [ ] Implement TIMESYNC alignment, telemetry interpolation, stale-data rejection, and monotonic timestamp validation.
- [ ] Add bounded queues and backpressure so delayed perception cannot destabilize control timing.

## P0: Perception, Estimation, And Safety

- [ ] Define the CNN output schema and implement gate pose estimation via PnP or validated relative pose.
- [ ] Validate the camera tilt sign, camera-to-body transform, and NED conversion against simulator imagery.
- [ ] Upgrade weighted position correction to a complementary filter or EKF with uncertainty tracking.
- [ ] Add landmark association, confidence decay, occlusion handling, and optional seeded course maps.
- [ ] Implement differential-flatness control; keep the existing hover controller as a tested fallback.
- [ ] Add SE(3) control only after the baseline controller is reliable.
- [ ] Expand the system mode manager with arming checks, stale telemetry, heartbeat loss, vision loss, low-speed timeout, collision, geofence, saturation, solver failure, maximum run time, and emergency shutdown.

## P1: Strengthen Simulation And Replay

- [ ] Preserve deterministic seeded simulation and add recorded-log replay.
- [ ] Run the same planner, controller, estimator, and safety interfaces in offline, Unreal, and live modes.
- [ ] Add wind, latency, jitter, packet loss, sensor noise, dropped frames, bad detections, actuator saturation, and restart scenarios.
- [ ] Add multi-gate racing scenarios, not only hover and single-gate traversal.
- [ ] Capture planner references, controls, gate estimates, safety flags, timing, and optional downsampled frames in logs.
- [ ] Consolidate the saved-run viewer in `src/app/` and the simulator visualization into a clear operator workflow.

## P1: Testing And Release Infrastructure

- [ ] Repair the local Python environment; the checked-in `.venv` points to an inaccessible Windows Store Python executable.
- [ ] Fix stale tests: course-loader tests expect 16 gates, while the checked-in course contains 2 gates plus origin; the legacy fixture is also missing.
- [ ] Add unit tests for every live-stack module and every safety transition.
- [ ] Add MPCC regression tests for multi-gate paths, infeasibility, timeout, and warm starts.
- [ ] Add full replay integration tests from telemetry and detections through emitted MAVLink commands.
- [ ] Add simulator-in-the-loop and Unreal-in-the-loop smoke tests.
- [ ] Add staged real-world testing: props-off bench, restrained hover, low-speed gate pass, multi-gate run, and fault injection.
- [ ] Add `pyproject.toml`, pinned runtime/dev dependencies, supported Python version, linting, type checks, and CI.

## P2: Cleanup And Documentation

- [ ] Update `README.md`; it describes an older tree and commands that no longer exist.
- [x] Retire the legacy `src/main.py` implementation and preserve it as the canonical production entry-point placeholder.
- [ ] Decide whether `opt_engine` remains a useful spline/pre-planning library or moves to an archived prototype area.
- [ ] Remove or archive `notebooks/planner_copy.py` and stale notebook experiments.
- [ ] Decide whether tracked instance-pose review images are fixtures, documentation examples, or disposable artifacts.
- [ ] Expand `.gitignore` for `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, run logs, generated artifacts, and test logs.
- [ ] Document canonical commands, config ownership, interface contracts, simulator setup, live setup, log schema, and the release checklist.
