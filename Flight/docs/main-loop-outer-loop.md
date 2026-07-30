# `main.py` Lines 426-523: Outer Loop Design

This section is the slower "outer loop" that runs inside the main flight loop. The main loop runs at `INNER_LOOP_HZ` and is responsible for high-rate state estimation and command output. The outer loop runs at `OUTER_LOOP_HZ` and handles lower-rate work: consuming completed vision results, updating the gate map, replanning the path, generating the carrot-control target, and queueing the next camera frame for perception.

## Where It Fits

Immediately before this block, the code ingests telemetry, optionally feeds IMU samples into VIO, and updates `vehicle_state` through `vehicle_state_estimator.update(...)`. That happens every inner-loop tick. The outer loop begins only when:

```python
outer_loop_ran = inner_loop_started_s >= next_outer_cycle_s
```

If that condition is true, the code advances `next_outer_cycle_s` by `outer_period_s` and runs the lower-rate perception/planning work.

The design separates fast control from slower perception. IMU/state updates and actuator commands need predictable high-rate timing. Vision inference, map updates, and path planning can tolerate lower frequency and variable latency, so they are isolated in the outer loop.

## Step-by-Step Flow

### 1. Schedule The Next Outer Tick

At line 427:

```python
next_outer_cycle_s += outer_period_s
```

The scheduler advances by the intended period, not by "now plus period." That preserves a stable cadence over time. If one outer iteration runs late, the next scheduled time still follows the original clock rather than drifting later every cycle.

### 2. Finish Any Pending Vision Job

Lines 429-475 check whether `vision_pending` exists and whether its future is done. `vision_pending` stores:

```python
(
    vision_future,
    frame_log,
    frame_outer_cycle,
    frame_vehicle_state,
)
```

The future is produced by `ThreadPoolExecutor.submit(...)` later in the same block. When it completes, the code pulls the `observation` from the future and uses it to update the gate map and planned path.

The important design choice is that the frame is processed asynchronously. CNN or perception work can take longer than a control tick, so the flight loop does not block waiting for vision. It only consumes the result once it is ready.

The other important choice is that the code stores `frame_vehicle_state` alongside the frame. A vision observation describes what the camera saw at the time that frame was captured, not necessarily the drone's current state when inference finishes. Using the captured state keeps the observation, gate update, and planning origin tied to the same moment in time.

When a completed observation is available, the code:

1. Adds gate count and serialized observation data to `frame_log`.
2. Logs the processed vision frame.
3. Updates `gate_map` using the observation and the vehicle state from the frame time.
4. Marks gates as passed if the frame-time position is close enough.
5. Logs the updated gate map.
6. Filters observed gates into planning candidates.
7. Replans through `path_manager.plan_for_mode(...)`.
8. Logs the planned path.

The `try`/`except` around this work prevents a perception or mapping failure from crashing the whole flight loop. Failed frames are logged with `status="failed"`, then the system clears `vision_pending` so it can accept another frame later.

### 3. Pull A New Frame Only When Vision Is Idle

Line 477 intentionally gates frame intake:

```python
latest_frame = None if vision_pending is not None else vision_rx.get_next_frame()
```

If a vision job is still running, the loop does not queue another one. Since the executor has `max_workers=1`, this avoids building an unbounded backlog of stale frames. For flight control, a recent frame is usually more valuable than processing every frame late.

This creates a "latest available frame, one in flight" model:

- no blocking on perception,
- no piling up old work,
- at most one expensive vision inference running,
- deterministic ownership of the pending frame's log metadata and state snapshot.

### 4. Compute The Carrot Target

Lines 479-487 compute a path-following target:

```python
carrot = path_manager.carrot_point(
    vehicle_state.position_local_ned_m,
    carrot_controller.lookahead_m,
)
carrot_target = carrot_controller.compute_control(
    vehicle_state=vehicle_state,
    carrot=carrot,
)
```

The "carrot" is a lookahead point on the active path. Instead of trying to control directly to the entire path, the controller aims toward a point some distance ahead of the drone. This is a common path-following pattern because it smooths steering and makes the controller behave locally: follow the next reachable target, then advance that target as the vehicle moves.

This target is later consumed by the high-rate command section after the outer-loop block. The code also has a fallback later that recomputes `carrot_target` if the outer loop did not run on that inner tick.

### 5. Process A Newly Received Frame

If `latest_frame` exists, lines 489-520 prepare it for VIO and perception.

First, the receiver records which inner cycle saw the frame:

```python
vision_rx.record_frame_cycle(latest_frame.frame_id, inner_cycle)
```

Then, if VIO is enabled, the VIO provider processes the frame. Any resulting measurement is wrapped in a `VioCorrection` object and stored in `pending_vio_correction`. It is not applied immediately in this block. Instead, the next inner-loop state-estimator update consumes it:

```python
vio_measurement_for_update = pending_vio_correction
vehicle_state = vehicle_state_estimator.update(
    imu_data_t=imu_data_t,
    vio_measurement=vio_measurement_for_update,
)
pending_vio_correction = None
```

That handoff keeps all state-estimator mutations in one place: the inner-loop estimator update. The outer loop can produce corrections, but the estimator owns when corrections are fused.

Next, the code builds `frame_log` with identifying metadata:

- frame ID,
- inner and outer cycle numbers,
- simulation timestamp,
- saved image path,
- JPEG byte size.

Finally, line 515 submits the frame to the vision executor:

```python
vision_pending = (
    vision_executor.submit(vision_perception.process_vision_frame, latest_frame),
    frame_log,
    outer_cycle,
    vehicle_state,
)
```

This starts perception work in the background and records the context needed to interpret its result later.

### 6. Increment The Outer Cycle Counter

At line 523:

```python
outer_cycle += 1
```

The counter increments once per outer-loop execution. It is used mostly for logging and correlating events across telemetry, frames, perception results, map updates, and path plans.

## Design Idea

The main design idea is a two-rate flight architecture:

- the inner loop keeps state estimation and command output responsive and predictable;
- the outer loop handles slower autonomy updates;
- perception runs asynchronously so expensive inference does not block control;
- frame metadata and frame-time vehicle state travel with the async job so delayed vision results are interpreted consistently;
- VIO corrections are queued back into the estimator instead of directly mutating state from the outer loop;
- the path manager maintains the active path, while the carrot controller turns that path into a local target for the attitude controller.

This gives the system a practical real-time structure. The drone can keep flying from the latest known path and state even while the next image is still being analyzed. When the vision result arrives, the map and plan are refreshed, and the controller naturally starts tracking the updated path through the next carrot target.

## Things To Watch

The code logs `planned_path` with `cycle=inner_cycle` and `outer_cycle=outer_cycle` while processing a completed frame whose stored outer cycle is `frame_outer_cycle`. That is probably intentional if the log should show both "when this was consumed" and "which frame produced it," but it is worth being consistent about whether each log field means capture time, processing-completion time, or current loop time.

The async model also means a vision result can be based on an older vehicle pose. Storing `frame_vehicle_state` handles that correctly for mapping, but planning from that old position may lag the current vehicle position if perception is slow. The current design limits this risk by allowing only one pending frame and by keeping control independent of waiting for vision.
