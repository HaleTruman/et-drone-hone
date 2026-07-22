# Flight Improvement Backlog

This backlog focuses on improvements beyond the known temporary test observation for path planning and the unfinished full vision/planning integration in the main flight loop.

## P0: Safety And Runtime Control

- [ ] Route every outbound command through a centralized safety supervisor.
- [ ] Wire `SystemModeManager` or a replacement state machine into the live loop.
- [ ] Gate command emission on heartbeat freshness, IMU freshness, estimator health, arming state, collision events, run timeout, and command saturation.
- [ ] Add explicit stale-data rejection for MAVLink telemetry, IMU samples, and vision frames.
- [ ] Add a deliberate safe-stop shutdown path that sends zero or neutral commands, then disarms when connected.
- [ ] Log full exception tracebacks during runtime failures instead of only printing the exception message.
- [ ] Replace `ALLOW_FLIGHT = True` with an explicit config or CLI opt-in for live command emission.

## P0: State Estimation

- [ ] Stop treating pure IMU dead reckoning as authoritative position truth beyond short propagation windows.
- [ ] Add a fused estimator boundary that supports IMU propagation plus external corrections.
- [ ] Define supported correction sources, such as simulator pose, gate observations, optical flow, VIO, or another measured state input.
- [ ] Add estimator health outputs: initialized, stale, divergent, high drift, max `dt` exceeded, and correction residual too large.
- [ ] Add monotonic timestamp validation and maximum integration `dt` limits in `VehicleStateEstimator`.
- [ ] Validate gravity sign, accelerometer convention, gyro sign correction, and quaternion integration against known motion fixtures.

## P0: Runtime Configuration

- [ ] Move live constants from `src/main.py` into a typed runtime config.
- [ ] Include MAVLink endpoint, vision host/port, loop rates, reset behavior, arming behavior, thrust limits, controller gains, lookahead, and safety thresholds in config.
- [ ] Log the exact resolved config at run start.
- [ ] Add separate profiles for offline replay, simulator bench, props-off bench, and live flight.
- [ ] Reject unsafe config combinations unless explicitly acknowledged by the operator.

## P0: Main Loop Structure

- [ ] Split `src/main.py` into composable runtime pieces.
- [ ] Extract startup/reset/calibration into a startup sequence module.
- [ ] Extract inner and outer loop scheduling into a flight-loop module.
- [ ] Extract command publishing into a MAVLink command publisher with validation.
- [ ] Keep logging calls close to runtime events, but avoid making the logger the only source of runtime state.
- [ ] Make the same planner, controller, estimator, and safety interfaces usable from replay tests.

## P0: MAVLink Command And Telemetry Contracts

- [ ] Add contract tests for `SET_ATTITUDE_TARGET` type masks.
- [ ] Verify quaternion order, body-rate axes, thrust range, target system/component IDs, and simulator expectations.
- [ ] Add tests for heartbeat parsing, armed-state parsing, `HIGHRES_IMU` parsing, collision parsing, and race-status parsing.
- [ ] Use `TIMESYNC` data to align simulator and local timing or explicitly document why it is informational only.
- [ ] Add reconnect or controlled-fault behavior for MAVLink receive-loop failures.
- [ ] Track and log message rates, dropped updates, and latest message age.

## P1: Gate Mapping

- [ ] Replace ID-based record overwrites with tracked gate estimates.
- [ ] Add nearest-neighbor or geometry-based gate association for repeated observations.
- [ ] Maintain observation count, last-observed time, confidence accumulation, confidence decay, and frozen high-confidence records.
- [ ] Reject implausible gate jumps using distance, orientation, and residual thresholds.
- [ ] Track duplicate detections and reordered detections explicitly.
- [ ] Validate camera optical, body FRD, local NED, camera tilt, and gate through-axis conventions with fixtures.

## P1: Planning And Path Following Robustness

- [ ] Add path-manager behavior for empty paths, one-gate paths, repeated gates, crossed gates, and end-of-path hold.
- [ ] Add bounds on lookahead, commanded acceleration, tilt, body rates, thrust, and cross-track correction.
- [ ] Add fallback behavior when path generation fails or returns too few valid points.
- [ ] Log planner input gates, selected path, carrot point, cross-track error, and controller command limits every cycle.
- [ ] Add regression tests for path projection and carrot-point selection.

## P1: Control

- [ ] Add command envelope validation before sending MAVLink targets.
- [ ] Add controller unit tests for attitude error direction, body-rate damping, saturation, and thrust clamping.
- [ ] Validate carrot-controller thrust sign and tilt compensation against simple hover and forward-flight cases.
- [ ] Add configurable controller gains and thrust limits through runtime config.
- [ ] Keep hover or neutral command behavior as a tested fallback mode.

## P1: Replay And Test Infrastructure

- [ ] Add a `Flight/tests` directory with focused unit tests for live-stack modules.
- [ ] Add recorded-log replay that runs estimator, planner, controller, safety, and command publishing without a simulator.
- [ ] Add replay fixtures for latency, jitter, packet loss, repeated IMU timestamps, stale telemetry, bad detections, collisions, and actuator saturation.
- [ ] Add simulator-in-the-loop smoke tests for startup, reset, arming, telemetry receipt, command emission, and shutdown.
- [ ] Add CI commands for tests, linting, formatting, and type checks.

## P2: Packaging And Repository Hygiene

- [ ] Add `pyproject.toml` with supported Python version, package metadata, lint configuration, and test configuration.
- [ ] Split dependencies into runtime, vision, dashboard, training, and dev/test groups.
- [ ] Reduce `requirements.txt` to the minimum needed for the selected environment or generate it from managed dependency groups.
- [ ] Keep model checkpoints, generated run logs, videos, cache files, and disposable review artifacts out of normal source churn.
- [ ] Document canonical commands for app viewer, live simulator run, replay, tests, and validation utilities.

