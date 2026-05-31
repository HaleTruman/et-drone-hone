# Racing Stack Backlog

This backlog records changes intentionally deferred while wireframing the live autonomous racing stack.
Existing implementations were not modified.

## Existing Components To Adapt

### MPCC planner

- Reuse `src/drone_mpcc_planner/planner.py` instead of creating a second planner.
- Adapt its interface to expose `optimize()`, `warm_start()`, `set_reference_path()`, and `get_reference_trajectory()`.
- Extend the current single-gate restriction to support the next 1-3 visible gates.
- Replace the placeholder returned `theta=1.0` with an optimized progress variable.
- Decide whether the live stack should call the current planner directly or use a thin adapter.

### Reference path generation

- Reconcile `src/drone_mpcc_planner/reference_path.py` with the Hermite spline engine in `src/opt_engine/`.
- Replace the current piecewise-linear MPCC reference with a smooth multi-gate reference.
- Enforce gate through-axis alignment using target orientation where applicable.

### Simulation harness

- Decide whether `src/racing_stack/sim_harness.py` should remain a small adapter around
  `src/quadrotor/model.py` or whether the existing `src/sim/simulator/simulator.py` should expose
  the `step()`, `reset()`, and `run_trajectory()` harness interface directly.
- Add parity checks between the NumPy dynamics and CasADi dynamics.

### Controllers

- Decide whether the existing hover/track controller in `src/sim/simulator/simulator.py` remains
  an offline test controller or becomes a fallback live controller.
- Implement `DifferentialFlatnessController`.
- Implement the optional `SE3GeometricController` after the differential-flatness path works.

## Live Transport And Timing

- Select the MAVLink client library and implement `MavlinkBridge`.
- Confirm the UDP endpoint, system ID, component ID, and command masks.
- Implement a heartbeat worker at 2 Hz or faster.
- Confirm whether velocity arrives in BODY_NED or LOCAL_NED and normalize it in one place.
- Implement the port 5600 UDP vision packet header parser and frame reassembly policy.
- Choose the JPEG decoder and image container passed into perception.
- Implement TIMESYNC handling and define the telemetry interpolation policy for frame alignment.
- Add stale-frame, stale-telemetry, packet-loss, and reconnect behavior.

## Perception And State Estimation

- Define the CNN output schema: gate ID, confidence, bounding box, keypoints, relative translation,
  and relative orientation.
- Implement PnP or direct least-squares gate pose estimation from known gate dimensions.
- Validate the camera-to-body 20-degree tilt sign against simulator imagery.
- Confirm LOCAL_NED axis conventions end to end.
- Extend `StateEstimator` from weighted position correction to an EKF or complementary filter.
- Define landmark association and confidence decay for the persistent `GateMap`.
- Decide whether approximate course locations seed the gate map at startup.

## Path And Race Management

- Implement `PathManager.generate_spline()` using the selected existing spline implementation.
- Define gate ordering for start, intermediate gates, and finish.
- Advance the active reference path after sequential crossings.
- Add occlusion handling for gates that temporarily disappear from perception.
- Confirm whether the 1.5 m aperture and 0.26 m depth tolerance are sufficient for scoring parity.

## Commands And Safety

- Map `CommandMapper` payloads to MAVLink `SET_POSITION_TARGET_LOCAL_NED`.
- Map `CommandMapper` payloads to MAVLink `SET_ATTITUDE_TARGET`.
- Confirm normalized-thrust scaling and saturation against the simulator.
- Expand `FlightStateMachine` with heartbeat loss, collision, low-speed timeout, stale-data, arming,
  and maximum-run checks.
- Define shutdown behavior for faults and completed runs.

## Logging And Replay

- Expand `Logger` to capture timestamps, raw and fused gate detections, planner references,
  optimized controls, sent commands, frame IDs, decode errors, state transitions, and safety flags.
- Add CSV plus JSON metadata export or choose a lightweight database.
- Add an offline replay entrypoint for perception, planner, and controller tuning.
- Decide whether downsampled JPEG frames should be retained.

## Existing Repository Issues

- Fix `src/main.py`: it references `src/config/settings.yaml`, but the checked-in config is
  `src/sim/config/settings.yaml`.
- Decide which entrypoint is canonical: saved-reference viewer, MPCC demo, simulation dashboard,
  or live `AIGPStack`.
- Update `README.md`; it describes files and optimizer UI behavior that do not match the current tree.
- Repair or recreate `.venv`; its Python executable currently resolves to an inaccessible Windows
  Python installation.
- Add automated tests for `src/racing_stack/`, MPCC behavior, controller behavior, and live-loop
  integration.

## Wireframe Completion Criteria

- Replace each intentional `NotImplementedError` in `src/racing_stack/` with tested behavior.
- Add an integration test that replays synchronized telemetry and vision detections through the full
  loop and verifies sequential gate crossings plus emitted MAVLink payloads.
- Add a simulator-in-the-loop test that exercises the same planner and controller interfaces used by
  the live stack.
