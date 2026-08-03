# Stack Brief

Note: this is a historical planning brief. The current runtime does not consume MAVLink `ODOMETRY`, `LOCAL_POSITION_NED`, or `ATTITUDE` for vehicle pose. `MavlinkClient` keeps MAVLink telemetry raw, and `VehicleStateEstimator` in `src/sensing/odometry/state.py` owns the flight-facing `VehicleState` estimate.

The full conceptual pipeline is:
> Vision > Telemetry > Perception > Planning > Control > Pilot Commands > Stabilized Controller

This document will focus on the software stack downstream of Vision. 

## 1. Information Sources (Raw Inputs)
- Telemetry (MAVLink, ~100 Hz capable)
  - HEARTBEAT (connection health, system status flags)
  - HIGHRES_IMU-derived vehicle state via `VehicleStateEstimator`
  - HIGHRES_IMU (raw accels/gyros, and per spec §4.5 also linear velocities)
  - TIMESYNC (for precise sim-time alignment)
   
  Current runtime estimates attitude/orientation, body angular rates, linear velocities, and position locally from `HIGHRES_IMU` instead of reading MAVLink pose packets.\

- Vision Stream (UDP port 5600, 30 Hz, 640×360 JPEG)

   Chunked packets with frame_id, chunk_id, total_chunks, jpeg_size, sim_time_ns. You must reassemble frames, decode JPEGs, and timestamp them against sim time.

- Camera model (spec §3.8): Pinhole, no distortion, intrinsics given (fx = fy = 320, cx = 320, cy = 180, vertical FoV 90°), body-to-camera transform is identity except 20° upward tilt. All coordinates are NED → you will need a small rotation when projecting CNN outputs.



## 2. Perception Layer (CNN + State Estimation)
CNN output -> per visible gate: relative 3D position (in camera frame) + orientation (at minimum yaw; ideally full pose or at least gate normal).

#### Immediate post-processing:

1. Convert pixel detections → 3D ray in camera frame using intrinsics.
2. Use known gate geometry (outer 2.7 m, inner 1.5 m square, 0.26 m depth) to solve for distance/orientation (PnP or direct least-squares).
3. Rotate from camera → body → LOCAL_NED using current drone attitude + 20° tilt correction.

#### Drone state update (fusion):

- Attitude / angular rates: estimated locally from `HIGHRES_IMU` through `VehicleStateEstimator`.
- Linear velocity: estimated locally by integrating IMU-derived acceleration.
- Position: integrate velocity in LOCAL_NED (simple Euler or RK4). IMU acceleration can be used as a secondary check or for short-term prediction, but vision gate detections provide the absolute correction (landmark-based localization). Because the environment is deterministic, a lightweight EKF or even a simple complementary filter on gate-derived position updates will keep drift negligible.

#### Store a dictionary/list of gates in global LOCAL_NED coordinates:

- Gate ID (start, 1…N, finish), center position, orientation (quaternion or normal + up vector), confidence.
- On every new CNN detection: transform relative pose → global using current estimated drone pose → fuse (weighted average or Kalman update) into the map.
- Benefits:
  - Global MPCC planning becomes trivial once you have 2–3 gates.
  - Predict/occlusion handling for gates you haven’t seen yet.
  - Robustness: multiple observations refine the map even if CNN misses occasionally.
  - Hot-starting the planner is easy because you always have the latest map.


Because the course geometry is identical for every run (§3.5), you can optionally seed the map with approximate gate locations if you want, but vision-only incremental mapping works fine and is more general.


## 3. Planning Layer (MPCC)
MPCC is an excellent choice for drone racing — it directly optimizes progress along a reference path while penalizing contouring error and respecting dynamics/collision constraints.

#### Stateful elements to store:

- Current gate map (global LOCAL_NED).
- Previous MPCC solution (state trajectory + control sequence) → hot-start / warm-start the optimizer on every iteration. This is critical for real-time performance at 30–100 Hz.
- Current reference path (sequence of gates + spline or waypoints between them).

#### Stateless:

- Raw vision frames, raw MAVLink packets (process and discard after use).

#### System modes / drone condition:

- `IDLE` > `ARMED` > `RACING` > `FINISHED` / `FAULT` (collision timeout, max 8 min run, loss of comms, etc.).
- Monitor HEARTBEAT + system status flags + your own safety checks (e.g., velocity too low for too long = crash).

## 4. Control / Path-Following Layer
Two strong options you mentioned; both compatible with the nonlinear 13-state quadrotor model you already derived.

- Differential flatness (simpler, very fast):
  - Flat outputs: position (x,y,z) + yaw.
  - Compute desired thrust magnitude + body rates from 2nd/3rd derivatives of reference trajectory.
  - Feed into SET_POSITION_TARGET_LOCAL_NED (or SET_ATTITUDE_TARGET if you want more authority).
  - Works extremely well for racing because the sim’s stabilized controller already handles low-level attitude loops.

- Geometric tracking on SE(3) (more aggressive, matches your nonlinear model):
  - Error metric directly on position (R³) and attitude (SO(3) via quaternion).
  - Uses the exact same 13-state dynamics (quaternion kinematics, Euler rotational eqns, thrust allocation, gyroscopic terms, drag).
  - Excellent for high-speed gate slicing and recovery from large attitude errors.
  - Output can still be mapped to `SET_POSITION_TARGET_LOCAL_NED` (position + velocity + yaw) or `SET_ATTITUDE_TARGET` + thrust.


Recommendation: Start with differential flatness (quicker to implement, robust), then add SE(3) geometric control for the final performance push. Both can run at >100 Hz on a decent GPU/CPU.
Command rate: Keep <100 Hz, send `SET_POSITION_TARGET_LOCAL_NED` (or `SET_ATTITUDE_TARGET`) with the desired local-NED position/velocity/yaw/thrust. The sim applies it at physics rate.

## 5. Gate Crossing Detection
Define a gate as “crossed” when the drone’s position (in LOCAL_NED) satisfies all:

- Lies within the gate’s plane ± depth tolerance (260 mm).
- Inside the inner square (1.5 m × 1.5 m) projected on that plane.
- Sequential order (start → gate 1 → … → finish).

You can also use CNN detection confidence + velocity direction through the gate as a secondary signal. Once crossed, mark it in the map and advance the reference path.

## 6. What to Store: Stateful vs. Stateless
#### Stateful (keep across control loop iterations):

- Drone state estimate (13-state vector or subset).
- Global gate map.
- Last MPCC solution (for warm-start).
- Current system mode + run timer.
- Recent telemetry/vision timestamps (for sync checks).

#### Stateless (process-and-discard):

- Raw JPEG frames.
- Individual MAVLink packets.
- Temporary perception outputs (only the fused gate updates matter).

## 7. Logging (Critical for Debugging & Iteration)
Log everything at least at 30 Hz (or on change):

- Sim time (ns), drone pose (position + quaternion), velocity, body rates.
- Raw & fused gate detections (relative + global).
- MPCC reference trajectory + solved controls.
- Sent MAVLink commands.
- Vision frame IDs and any decoding errors.
- System mode, run duration, safety flags.
- (Optional) downsampled vision images for post-run visualization.

Store as structured logs (CSV + JSON metadata) or a lightweight database. Because runs are deterministic, you can replay logs offline to tune perception/planner/control independently.

## 8. Additional Practical Considerations

- Timing & synchronization: Use `TIMESYNC` and sim_time_ns from vision packets. Your control loop should run at ~30–60 Hz (vision-limited) and predict forward to physics rate using the nonlinear model if needed.
- Heartbeat: Must send at ≥2 Hz or the sim will drop you.
- Compliance: No human intervention during a scored run (§7) — your stack must be fully autonomous once launched.
- SITL bridge: The low-latency UDP bridge is exactly what you’re using; keep everything in one process or use shared memory if you split perception/planning.
- Testing: Because the course is identical and deterministic, you can iterate extremely fast. Record a “perfect” run with manual control first to get ground-truth gate positions if you want a seeded map.
- Edge cases: Vision occlusion, aggressive attitudes (camera tilt matters), high-speed drag, and MAVLink simulator-in-the-loop testing.

This pipeline keeps the system modular, leverages the exact interfaces in the spec, and directly uses the 13-state nonlinear model you already have for both planning and control validation. It also gives you a clean separation: perception builds the world map, MPCC plans the race line, and the geometric/diff-flat controller executes it with minimal latency.



# Classes and Methods

This is a table of all of the planned classes and methods downstream of Vision. 
| File/Module | Primary Class(es) | Key Methods | Description |
|---|---|---|---|
| `mavlink_bridge.py` | `MavlinkBridge` | `connect()`, `start_heartbeat()`, `subscribe_telemetry()`, `send_position_target()`, `send_attitude_target()`, `get_latest_telemetry()` | MAVLink UDP client. Handles connection, HEARTBEAT, raw telemetry subscriptions (`HIGHRES_IMU`, `TIMESYNC`), and command sending. Vehicle pose is estimated locally, not read from MAVLink pose packets. |
| `vision_stream.py` | `VisionStreamReceiver` | `start_listener()`, `reassemble_frame()`, `decode_jpeg()`, `get_next_frame()` | UDP listener (port 5600) that reassembles chunked packets (§4.6 header format), decodes 640×360 JPEGs, and timestamps with `sim_time_ns`. |
| `sync.py` | `DataSynchronizer` | `align_frame_with_telemetry()`, `get_synchronized_data()` | Aligns vision frames with MAVLink telemetry using sim timestamps. Outputs clean (`frame`, `telemetry`) tuples at ~30 Hz. |
| `gate_pose.py` | `GatePoseEstimator` | `estimate_gate_pose(body_frame)`, `camera_to_body_transform()`, `body_to_local_ned()` | Converts CNN output (ID, bbox/keypoints, relative pose) into 3D gate pose in body frame then LOCAL_NED using camera intrinsics + 20° tilt (§3.8) and known gate geometry (§3.7). |
| `sensing/gates/gate_map.py` | `GateMap` | `add_or_update_gate()`, `get_gate(id)`, `get_next_gates(n)`, `get_reference_path()`, `fuse_gate()` | Persistent global map of all gates in LOCAL_NED coordinates. Handles fusion, sequencing, and reference-path generation. |
| `state_estimator.py` | `StateEstimator` | `integrate_velocity()`, `vision_correction()`, `get_13_state()`, `reset()` | Maintains the full 13-state nonlinear quadrotor model (pos, vel, quat, rates). Velocity integration + vision-based landmark updates for drift-free pose. |
| `path_manager.py` | `PathManager` | `update_from_gate_map()`, `generate_spline()`, `get_waypoints()` | Converts GateMap into smooth spline/waypoint sequence for the MPCC planner. Handles start → intermediate → finish sequencing. |
| `mpcc_planner.py` | `MPCCPlanner` | `optimize()`, `warm_start()`, `set_reference_path()`, `get_reference_trajectory()` | Wrapper around your MPCC optimizer. Accepts current drone state + gate map, outputs reference trajectory with warm-start support. |
| `diff_flat_controller.py` | `DifferentialFlatnessController` | `compute_commands(reference_traj)`, `flat_to_thrust_and_rates()` | Recommended first controller. Uses differential flatness to generate desired thrust, body rates, and yaw from reference trajectory. |
| `se3_controller.py` | `SE3GeometricController` | `compute_control()`, `position_error()`, `attitude_error()` | (Optional upgrade) Full geometric tracking on SE(3) using the exact 13-state nonlinear model you derived. |
| `command_mapper.py` | `CommandMapper` | `to_position_target()`, `to_attitude_target()`, `scale_thrust()` | Converts controller output into correct MAVLink SET_POSITION_TARGET_LOCAL_NED or SET_ATTITUDE_TARGET messages. |
| `system_mode.py` | `SystemModeManager` | `update_mode()`, `check_gate_crossing()`, `is_racing()`, `handle_fault()` | System modes: IDLE → ARMED → RACING → FINISHED / FAULT. Includes 8-minute timer, sequential gate-crossing logic, and safety checks (§8.3, §7). |
| `logging.py` | `Logger` | `log_telemetry()`, `log_gate_map()`, `log_mpcc_solution()`, `save_run()` | Structured logging (CSV + JSON) of sim time, 13-state, gates, reference traj, commands, vision frames, etc. Enables offline replay. |
| `replay.py` | run replay helpers | `load_run()`, `iter_cycles()`, `replay_commands()` | Replays recorded MAVLink telemetry and command logs for planner + controller regression tests. |
| `main.py` | production runtime | control loop, shutdown | Main entry point. Ties everything together at 30–60 Hz: vision → perception → state_est → planner → controller → mavlink. |

These files and classes must be independently testable and expose clean Python interfaces, effectively plug and play.
