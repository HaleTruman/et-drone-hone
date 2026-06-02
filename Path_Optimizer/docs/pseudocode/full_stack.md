# AIGPStack High-Level Pseudocode (Plain English Description)

# Overall purpose of AIGPStack
# This is the main orchestrator class that runs the entire downstream pipeline
# at a fixed rate (default 30 Hz). It glues together communication, perception,
# mapping, state estimation, planning, control, and logging while switching
# seamlessly between real simulator mode and offline simulator harness mode.

initialize AIGPStack:
    load configuration (drone params, gains, mode: real or offline)
    create CommunicationInterface (MAVLink+Vision or OfflineHarnessBridge)
    create GateMap (empty persistent map in LOCAL_NED)
    create StateEstimator (starts at position [0,0,0] from arm time)
    create GatePoseEstimator (camera intrinsics + 20° tilt)
    create DataSynchronizer
    create PathManager
    create MPCCPlanner (with warm-start capability)
    create DifferentialFlatnessController (or SE3GeometricController)
    create CommandMapper
    create SystemModeManager (starts in IDLE)
    create Logger
    reset simulation harness if in offline mode

main control loop (run at target_hz = 30-60):
    while not shutdown requested:
        current_sim_time = get latest synchronized timestamp
        
        # 1. Ingest fresh data
        latest_telemetry = communication.get_latest_state()
        vision_frame = communication.get_next_vision_frame()
        
        # 2. Perception: turn raw CNN output into usable observations
        if vision_frame is valid:
            cnn_detections = process_cnn_on_frame(vision_frame)  # assumed upstream
            for each detection in cnn_detections:
                gate_obs = GatePoseEstimator.estimate_pose(detection, latest_telemetry)
                GateMap.add_or_update(gate_obs, latest_telemetry)
        
        # 3. State estimation (VIO and SLAM)
        StateEstimator.predict_from_telemetry(latest_telemetry)  # integrate velocity + quaternion kinematics
        StateEstimator.correct_with_gate_observations(GateMap)   # landmark innovations + multi-frame fusion
        
        current_drone_state = StateEstimator.get_13_state()
        
        # 4. System mode transitions & safety checks
        SystemModeManager.update_mode(current_drone_state, GateMap)
        if SystemModeManager.is_racing():
            check_gate_crossing_events(GateMap, current_drone_state)
            enforce_8_minute_timeout()
        
        # 5. Planning
        if in racing mode:
            PathManager.update_from_gate_map(GateMap)
            reference_path = PathManager.get_reference_path()
            mpcc_solution = MPCCPlanner.optimize(current_drone_state, reference_path, warm_start=previous_solution)
            previous_solution = mpcc_solution  # save for next warm-start
        
        # 6. Control
        if in racing mode:
            control_commands = DifferentialFlatnessController.compute_commands(mpcc_solution, current_drone_state)
            mavlink_cmd = CommandMapper.to_position_target_local_ned(control_commands)
            
            # 7. Send command to simulator
            communication.send_command(mavlink_cmd)
        
        # 8. Logging
        Logger.log_state(current_sim_time, current_drone_state, GateMap, mpcc_solution, control_commands)
        
        # 9. Sleep to maintain exact loop rate
        wait_until_next_cycle(target_hz)

on shutdown:
    send disarm command if needed
    save complete log file
    cleanly close communication channels

# End of high-level pseudocode
# This loop ensures real-time operation, drift-resistant mapping, and full autonomy per the technical specification.
