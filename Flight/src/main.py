import time
import traceback
import numpy as np 

from core.initialization import initialize
from core.logging import generate_mp4
from core.schema import MavlinkHighresImu, StateRecord, VioCorrection
from core.utils import time_since


def main() -> int:
    (
        settings,
        logger,
        vehicle_state_estimator,
        vio_provider,
        mavlink_client,
        vision_rx,
        vision_perception,
        vision_executor,
        obs_recorder,
        system_mode_manager,
        gate_map,
        path_manager,
        attitude_controller,
        carrot_controller,
        geometric_path_follower,
        hover_controller,
    ) = initialize()

    print(f"Starting run at {settings.run_dir}...")

    inner_period_s = 1.0 / settings.inner_loop_hz
    outer_period_s = 1.0 / settings.outer_loop_hz

    # holders
    imu_data_t = None
    telemetry = None
    latest_frame = None
    observation = None
    planned_path = None
    carrot_target = None
    geometric_target = None
    vision_pending = None
    pending_vio_correction: VioCorrection | None = None

    inner_cycle = 0
    outer_cycle = 0

    started_s = time.perf_counter()
    next_inner_cycle_s = started_s
    next_outer_cycle_s = started_s


# =================================================================== STARTUP PROCESS ===================================================================
    try:
    
        mavlink_client.connect(heartbeat_timeout_s=settings.heartbeat_timeout_s)
        mavlink_client.start_heartbeat()
        mavlink_client.subscribe_telemetry()
        logger.log_event("mavlink_connected", time_since_startup_s = time_since(started_s), bridge=mavlink_client.snapshot())

        vision_rx.start_listener()
        logger.log_event("vision_started", time_since_startup_s = time_since(started_s), receiver=vision_rx.snapshot())

        # RESET
        mavlink_client.send_sim_reset_command()
        time_sim_reset_s = time.perf_counter()
        logger.log_event("simulator_reset_sent", time_since_startup_s = time_since(started_s))

        if settings.post_reset_delay_s > 0.0:
            time.sleep(settings.post_reset_delay_s)
            logger.log_event("simulator_settle_complete", time_since_startup_s = time_since(started_s), elapsed_s=time.perf_counter() - time_sim_reset_s, settle_delay_s=settings.post_reset_delay_s)

        # clear pre-reset samples, then wait for fresh post-reset telemetry and vision
        mavlink_client.clear_cached_telemetry()
        vision_rx.clear_buffer()
        vision_rx.begin_saving_frames()
        vio_provider.reset()

        telemetry = mavlink_client.wait_until_receiving(timeout_s=settings.startup_data_timeout_s)
        logger.log_event("mavlink_receiving", sim_time_ns=telemetry.sim_time_ns)

        frame = vision_rx.wait_until_receiving(timeout_s=settings.startup_data_timeout_s)
        logger.log_event("vision_receiving", sim_time_ns=frame.sim_time_ns)

        # start screen recording
        if settings.record_screen:
            logger.log_event("obs_recording_started", time_since_startup_s = time_since(started_s), sim_time_ns=telemetry.sim_time_ns) if obs_recorder.start_recording() else logger.log_event("obs_recording_failed", sim_time_ns=telemetry.sim_time_ns)


        # IMU calibration
        imu_calibration_samples: list[MavlinkHighresImu] = []
        last_calibration_imu_time_boot_us: int | None = None
        imu_calibration_deadline_s = time.perf_counter() + settings.imu_init_timeout_s
        
        while time.perf_counter() < imu_calibration_deadline_s: # rate: settings.inner_loop_hz
            imu_data_t = mavlink_client.latest_imu

            # collect imu samples for estimating sensor drift/bias
            if (imu_data_t is not None and imu_data_t.time_boot_us != last_calibration_imu_time_boot_us):
                imu_calibration_samples.append(imu_data_t)
                last_calibration_imu_time_boot_us = imu_data_t.time_boot_us

            # timing
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())

            if sleep_s > 0.0:
                time.sleep(sleep_s)
  
        if imu_calibration_samples:
            gyro_bias = np.asarray([sample.gyro_body_frd_rps for sample in imu_calibration_samples], dtype=float).mean(axis=0)
            accel_bias = np.asarray([sample.acceleration_body_frd_mps2 for sample in imu_calibration_samples], dtype=float).mean(axis=0)
            logger.log_event(
                "stationary_imu_calibrated",
                sample_count=len(imu_calibration_samples),
                time_since_startup_s = time_since(started_s),
                duration_s=(imu_calibration_samples[-1].time_boot_us - imu_calibration_samples[0].time_boot_us) / 1_000_000,
                gyro_bias_body_frd_rps=gyro_bias,
                acceleration_rest_body_frd_mps2=accel_bias
            )

        else:
            logger.log_event(
                "stationary_imu_calibration_failed",
                reason="no_imu_samples",
                system_mode=system_mode_manager.system_mode.value,
            )
            raise ValueError("IMU calibration failed due to no or eratic samples.")


        # Initialize state
        vehicle_state = vehicle_state_estimator.initialize_from_stationary_imu_samples(imu_calibration_samples)
        logger.log_event(
            "vehicle_state_initialized",
            time_since_startup_s = time_since(started_s),
            vehicle_state = vehicle_state_estimator.state,
        )

        if vehicle_state_estimator.initialized is False:
            system_mode_manager.handle_fault("no_stationary_imu_samples_or_other_failure")
            logger.log_event(
                "vehicle_state_initialization_failed",
                time_since_startup_s = time_since(started_s),
                reason=system_mode_manager.fault_reason,
                system_mode=system_mode_manager.system_mode.value,
            )
            raise Exception("Initialization failed.")

        
        # Drone enter ARM mode
        mavlink_client.arm_and_wait(timeout_s=settings.arm_timeout_s)
        system_mode_manager.update_mode("arm")
        logger.log_event("armed", time_since_startup_s = time_since(started_s), bridge=mavlink_client.snapshot(), system_mode=system_mode_manager.system_mode.value)


        # Vision calibration and initalization
        gate_map_init_deadline_s = time.perf_counter() + settings.gate_map_init_timeout_s
                
        while time.perf_counter() < gate_map_init_deadline_s: # rate: settings.outer_loop_hz
            latest_frame = vision_rx.get_next_frame()

            # init gate map and path plan
            if latest_frame is not None:
                observation = vision_perception.process_vision_frame(
                    latest_frame,
                    vehicle_state=vehicle_state_estimator.state,
                )
                
                logger.log_vision_observation(
                    observation,
                    frame_id=latest_frame.frame_id,
                    sim_time_ns=latest_frame.sim_time_ns,
                    gate_count=len(observation.gates),
                    perception=vision_perception.snapshot(),
                )

                gate_map.update(
                    observation,
                    observer_position_local_ned_m=vehicle_state_estimator.state.position_local_ned_m,
                )
                gate_map.update_crossed_gates(vehicle_state_estimator.state.position_local_ned_m)
                logger.log_gate_map(
                    gate_map.gates,
                    time_since_startup_s=time_since(started_s),
                    cycle=inner_cycle,
                    outer_cycle=outer_cycle,
                    frame_id=latest_frame.frame_id,
                    sim_time_ns=latest_frame.sim_time_ns,
                    gate_count=len(gate_map.gates),
                    source=observation.source,
                )

                planned_path = path_manager.plan(
                    gates=gate_map.uncrossed_gates(),
                    vehicle_state=vehicle_state_estimator.state,
                )

            else:
                logger.log_event("vision_calibration_skipped", reason="no_latest_frame")

            # timing
            next_outer_cycle_s += outer_period_s
            sleep_s = max(0.0, next_outer_cycle_s - time.perf_counter())

            if sleep_s > 0.0:
                time.sleep(sleep_s)



        if planned_path is None:
            if observation is not None:
                gate_map.update(
                    observation,
                    observer_position_local_ned_m=vehicle_state.position_local_ned_m,
                )
                gate_map.update_crossed_gates(vehicle_state.position_local_ned_m)
                logger.log_gate_map(
                    gate_map.gates,
                    time_since_startup_s=time_since(started_s),
                    cycle=inner_cycle,
                    outer_cycle=outer_cycle,
                    frame_id=observation.frame_id,
                    sim_time_ns=observation.sim_time_ns,
                    gate_count=len(gate_map.gates),
                    source=observation.source,
                )

            planned_path = path_manager.plan(
                gates=gate_map.uncrossed_gates(),
                vehicle_state=vehicle_state,
            )

        if path_manager.test_path is not None:
            logger.log_test_path(
                path_manager.test_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
                cycle=inner_cycle,
                planner="straight_line_test_path",
            )
        if planned_path is not None:
            logger.log_planned_path(
                planned_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
                cycle=inner_cycle,
                planner=path_manager.planning_mode,
            )



        ## MAIN LOOP
        control_started_s = time.perf_counter()
        next_inner_cycle_s = control_started_s
        next_outer_cycle_s = control_started_s

        # set system mode to RACING
        system_mode_manager.update_mode("start")
        if not system_mode_manager.is_racing():
            system_mode_manager.handle_fault("system_not_racing_before_flight")
            logger.log_event(
                "flight_start_mode_check_failed",
                time_since_startup_s = time_since(started_s),
                flight_began_s=control_started_s,
                reason=system_mode_manager.fault_reason,
                system_mode=system_mode_manager.system_mode.value,
            )
            raise Exception("System mode failed to enter RACING.")

        logger.log_event(
            "flight_began",
            flight_began_s=control_started_s,
            system_mode=system_mode_manager.system_mode.value,
        )
        


# =================================================================== BEGIN MAIN LOOP ===================================================================

        while settings.run_s is None or time.perf_counter() - control_started_s < settings.run_s:

            # ======================== INNER LOOP START ========================
            inner_loop_started_s = time.perf_counter()
            next_inner_scheduled_s = next_inner_cycle_s

            # ingest telemetry and update state
            telemetry = mavlink_client.get_telemetry()
            imu_data_t = mavlink_client.latest_imu

            if settings.enable_vio and imu_data_t is not None:
                vio_provider.add_imu_sample(imu_data_t)

            vio_measurement_for_update = pending_vio_correction
            vehicle_state = vehicle_state_estimator.update(
                imu_data_t=imu_data_t,
                vio_measurement=vio_measurement_for_update,
            )
            pending_vio_correction = None

            outer_loop_ran = inner_loop_started_s >= next_outer_cycle_s

            
            # ======================== OUTER LOOP START ========================
            if outer_loop_ran:
                next_outer_cycle_s += outer_period_s

                # if the vision job has completed
                if vision_pending is not None and vision_pending[0].done():
                    vision_future, frame_log, frame_outer_cycle, frame_vehicle_state = vision_pending
                    vision_pending = None

                    # do things with the observation 
                    try:
                        observation = vision_future.result()
                        state_frame_delta_ns = (
                            frame_log["vehicle_state_elapsed_ns"] - frame_log["frame_elapsed_ns"]
                            if frame_log.get("vehicle_state_elapsed_ns") is not None
                            and frame_log.get("frame_elapsed_ns") is not None
                            else None
                        )
                        frame_log["gate_count"] = len(observation.gates)
                        frame_log["observation"] = observation.to_controller_payload(output_dir="memory")
                        logger.log_vision_observation(
                            observation,
                            frame_id=frame_log["frame_id"],
                            inner_cycle=frame_log["inner_cycle"],
                            outer_cycle=frame_outer_cycle,
                            sim_time_ns=frame_log["sim_time_ns"],
                            state_frame_delta_ns=state_frame_delta_ns,
                            gate_count=len(observation.gates),
                            perception=vision_perception.snapshot(),
                        )
                        logger.log_vision_frame(frame_log, cycle=frame_outer_cycle, status="processed")

                        # update gatemap
                        gate_map.update(
                            observation,
                            observer_position_local_ned_m=frame_vehicle_state.position_local_ned_m,
                        )
                        gate_map.update_crossed_gates(frame_vehicle_state.position_local_ned_m)
                        logger.log_gate_map(
                            gate_map.gates,
                            time_since_startup_s=time_since(started_s),
                            cycle=inner_cycle,
                            outer_cycle=frame_outer_cycle,
                            frame_id=frame_log["frame_id"],
                            sim_time_ns=frame_log["sim_time_ns"],
                            gate_count=len(gate_map.gates),
                            source=observation.source,
                        )

                        planned_path = path_manager.plan(
                            gates=gate_map.uncrossed_gates(),
                            vehicle_state=frame_vehicle_state,
                        )
                        logger.log_planned_path(
                            planned_path.to_log_dict(origin_local_ned_m=frame_vehicle_state.position_local_ned_m),
                            time_since_startup_s = time_since(started_s),
                            cycle=inner_cycle,
                            outer_cycle=outer_cycle,
                            frame_id=frame_log["frame_id"],
                            planner=path_manager.planning_mode,
                        )

                    except Exception as error:  # noqa: BLE001
                        logger.log_vision_frame(frame_log, cycle=frame_outer_cycle, status="failed", error=str(error))
                        print(f"vision frame={frame_log['frame_id']} failed: {error}", flush=True)


                # Pull latest frame
                latest_frame = vision_rx.get_latest_frame()

                if latest_frame is not None:
                    vision_rx.record_frame_cycle(latest_frame.frame_id, inner_cycle)

                # do VIO
                if settings.enable_vio and latest_frame is not None:
                    vio_measurement = vio_provider.process_frame(latest_frame)
                    pending_vio_correction = (
                        None
                        if vio_measurement is None
                        else VioCorrection(
                            measurement=vio_measurement,
                            frame_id=latest_frame.frame_id,
                            frame_sim_time_ns=latest_frame.sim_time_ns,
                            queued_inner_cycle=inner_cycle,
                            queued_outer_cycle=outer_cycle,
                            source="opencv_monocular_vio",
                        )
                    )

                # if there is no job queued and we have a frame
                if vision_pending is None and latest_frame is not None and vehicle_state is not None:
                    frame_log = {
                        "frame_id": latest_frame.frame_id,
                        "inner_cycle": inner_cycle,
                        "outer_cycle": outer_cycle,
                        "sim_time_ns": latest_frame.sim_time_ns,
                        "saved_path": latest_frame.saved_path,
                        "jpeg_size": len(latest_frame.jpeg_bytes),
                        "vehicle_state_elapsed_ns": vehicle_state.elapsed_time_ns,
                        "frame_elapsed_ns": latest_frame.elapsed_time_ns
                    }

                    # queue vision job
                    vision_pending = (
                        vision_executor.submit(vision_perception.process_vision_frame, latest_frame, vehicle_state=vehicle_state),
                        frame_log,
                        outer_cycle,
                        vehicle_state,
                    )

                crossed_gates = gate_map.update_crossed_gates(vehicle_state.position_local_ned_m)
                if crossed_gates:
                    logger.log_gate_map(
                        gate_map.gates,
                        time_since_startup_s=time_since(started_s),
                        cycle=inner_cycle,
                        outer_cycle=outer_cycle,
                        sim_time_ns=vehicle_state.sim_time_ns,
                        gate_count=len(gate_map.gates),
                        crossed_gate_ids=[gate.gate_id for gate in crossed_gates],
                    )
                    planned_path = path_manager.plan(
                        gates=gate_map.uncrossed_gates(),
                        vehicle_state=vehicle_state,
                    )
                    logger.log_planned_path(
                        planned_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
                        time_since_startup_s=time_since(started_s),
                        cycle=inner_cycle,
                        outer_cycle=outer_cycle,
                        planner=path_manager.planning_mode,
                    )

                # CHECK FAILSAFE
                if settings.allow_flight and system_mode_manager.is_racing():
                    path_projection = path_manager.project(vehicle_state.position_local_ned_m)
                    path_error_m = float(path_projection.cross_track_error_m)
                    if path_error_m > settings.failsafe_distance_m:
                        system_mode_manager.update_mode("finish")
                        carrot_target = None
                        logger.log_event(
                            "path_failsafe_finished",
                            reason="path_deviation_exceeded",
                            cross_track_error_m=path_error_m,
                            failsafe_distance_m=settings.failsafe_distance_m,
                            projection=path_projection.to_log_dict(),
                            system_mode=system_mode_manager.system_mode.value,
                            inner_cycle=inner_cycle,
                            outer_cycle=outer_cycle,
                            sim_time_ns=vehicle_state.sim_time_ns,
                        )

                # compute attitude target for path-following controller
                if settings.allow_flight and system_mode_manager.is_racing():
                    if settings.control_method == "geometric_path_follower":
                        geometric_target = geometric_path_follower.compute_control(vehicle_state)
                    else:
                        carrot = path_manager.carrot_point(
                            vehicle_state.position_local_ned_m,
                            carrot_controller.lookahead_m,
                            carrot_controller.speed_lookahead_m,
                        )
                        carrot_target = carrot_controller.compute_control(
                            vehicle_state=vehicle_state,
                            carrot=carrot,
                        )
           


                outer_cycle += 1
            # ======================== OUTER LOOP END ========================


            control_target = {}

            if imu_data_t is None:
                command_result = {
                    "emitted": False,
                    "sim_time_ns": telemetry.sim_time_ns if telemetry else None,
                    "reason": "missing_highres_imu"
                }

            else:
                command_result = None

                if settings.allow_flight and system_mode_manager.is_racing():
                    if settings.control_method == "geometric_path_follower":
                        if geometric_target is None:
                            geometric_target = geometric_path_follower.compute_control(vehicle_state)
                        path_following_target = geometric_target
                    else:
                        if carrot_target is None:
                            carrot = path_manager.carrot_point(
                                vehicle_state.position_local_ned_m,
                                carrot_controller.lookahead_m,
                                carrot_controller.speed_lookahead_m,
                            )
                            carrot_target = carrot_controller.compute_control(
                                vehicle_state=vehicle_state,
                                carrot=carrot,
                            )
                        path_following_target = carrot_target

                    if path_following_target:
                        control_target = attitude_controller.compute_control(
                            vehicle_state,
                            desired_attitude_quaternion=path_following_target["quaternion"],
                            thrust=path_following_target["thrust"]
                        )
                    
                    mavlink_client.send_attitude_target(control_target)

                    command_result = {
                        "emitted": True,
                        "sim_time_ns": telemetry.sim_time_ns,
                        "reason": path_following_target.get("source", "path_following") if path_following_target else "path_following",
                        "attitude_target": control_target,
                        "inner_loop_cycle": inner_cycle,
                        "outer_loop_cycle": outer_cycle,
                    }
                elif settings.allow_flight and system_mode_manager.is_finished():
                    hover_target = hover_controller.compute_control(vehicle_state)
                    control_target = attitude_controller.compute_control(
                        vehicle_state,
                        desired_attitude_quaternion=hover_target["quaternion"],
                        thrust=hover_target["thrust"],
                    )

                    mavlink_client.send_attitude_target(control_target)

                    command_result = {
                        "emitted": True,
                        "sim_time_ns": telemetry.sim_time_ns,
                        "reason": "finished_hover",
                        "attitude_target": control_target,
                        "inner_loop_cycle": inner_cycle,
                        "outer_loop_cycle": outer_cycle,
                    }
                else:
                    command_result = {
                            "emitted": False,
                            "sim_time_ns": telemetry.sim_time_ns if telemetry else None,
                            "reason": "flight_disabled" if not settings.allow_flight else "system_mode_not_racing",
                            "attitude_target": control_target,
                            "inner_loop_cycle": inner_cycle,
                            "outer_loop_cycle": outer_cycle,
                        }


            # timing
            if isinstance(command_result, dict):
                command_result.setdefault("system_mode", system_mode_manager.system_mode.value)
                command_result.setdefault(
                    "modes",
                    {
                        "system": system_mode_manager.system_mode.value,
                    },
                )

            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())
            loop_elapsed_ms = (time.perf_counter() - inner_loop_started_s) * 1000.0

            if inner_cycle % int(settings.inner_loop_hz*5) == 0:
                print(f"inner_cycle={inner_cycle} - outer_cycle={outer_cycle} - loop_ms={loop_elapsed_ms:.2f}\n", flush=True)

            state_record = (None if telemetry is None else StateRecord.from_telemetry(
                    telemetry,
                    vehicle_state=vehicle_state,
                    vio_correction=vio_measurement_for_update,
                    vio_status=vehicle_state_estimator.last_vio_status,
                    vio_residual=vehicle_state_estimator.last_vio_residual,
                    kalman_status=vehicle_state_estimator.last_kalman_status,
                )
            )

            logger.log_cycle(
                inner_cycle=inner_cycle,
                outer_cycle=outer_cycle,
                time_since_startup_s = time_since(started_s),
                sim_time_ns=state_record.sim_time_ns if state_record else None,
                wall_elapsed_ms=(inner_loop_started_s - started_s) * 1000.0,
                loop_elapsed_ms=loop_elapsed_ms,
                deadline_lateness_ms=max(0.0, inner_loop_started_s - next_inner_scheduled_s) * 1000.0,
                sleep_ms=sleep_s * 1000.0,
                telemetry=state_record,
                system_mode=system_mode_manager.system_mode.value,
                modes={
                    "system": system_mode_manager.system_mode.value,
                },
                vision_frame_id=latest_frame.frame_id if latest_frame else None,
                bridge=mavlink_client.snapshot(),
                command=command_result,
                inner_loop={
                    "inner_cycle": inner_cycle,
                    "hz": settings.inner_loop_hz,
                },
                outer_loop={
                    "outer_cycle": outer_cycle,
                    "hz": settings.outer_loop_hz,
                    "ran": outer_loop_ran,
                },
                carrot=carrot_controller.last_payload,
                geometric_path_follower=geometric_path_follower.last_payload,
                vision={
                    **vision_rx.snapshot(),
                    "perception": vision_perception.snapshot()
                },
                vio=vio_provider.snapshot(),
            )
     
            inner_cycle += 1

            # sleep until next inner cycle begins
            if sleep_s > 0.0:
                time.sleep(sleep_s)

            # ======================== INNER LOOP END ======================== 

# =================================================================== END MAIN LOOP ===================================================================

        system_mode_manager.update_mode("finish")
        logger.log_event("flight_finished", system_mode=system_mode_manager.system_mode.value)

    except KeyboardInterrupt:
        system_mode_manager.handle_fault("keyboard_interrupt")
        logger.log_event("interrupted", system_mode=system_mode_manager.system_mode.value)

    except Exception as error:
        if system_mode_manager.system_mode.value != "FAULT":
            system_mode_manager.handle_fault(str(error))
        logger.log_exception("flight_exception", error, system_mode=system_mode_manager.system_mode.value)
        print("Error occured: ", error)
        traceback.format_exc()

# =================================================================== SHUTDOWN ===================================================================
    finally:
        if settings.record_screen:
            obs_recorder.stop_recording()

        vision_executor.shutdown(wait=False, cancel_futures=True)
        vision_rx.shutdown()
        if getattr(mavlink_client, "connected", False):
            try:
                shutdown_telemetry = telemetry
                if shutdown_telemetry is None:
                    raw_shutdown_telemetry = mavlink_client.get_telemetry()
                    shutdown_telemetry = vehicle_state_estimator.update_telemetry(raw_shutdown_telemetry)
                
            except Exception as error:  # noqa: BLE001
                logger.log_event(
                    "shutdown_stop_failed",
                    error=str(error),
                    system_mode=system_mode_manager.system_mode.value,
                )
        mavlink_client.shutdown()
        logger.log_event(
            "shutdown",
            system_mode=system_mode_manager.system_mode.value,
            bridge=mavlink_client.snapshot(),
            vision=vision_rx.snapshot(),
            perception=vision_perception.snapshot(),
        )
        logger.save_run(settings.log_path)
        print(f"Log saved to {settings.log_path}", flush=True)

        if settings.create_video:
            generate_mp4(settings.run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
