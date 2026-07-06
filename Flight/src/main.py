"""Minimal live MAVLink and vision stream entry point."""

from pathlib import Path
import time

from autonomy.planning import HotStartPlanner
from core.control.body_rate_guidance import BodyRateGuidanceController
from core.logging import Logger
from sensing.perception import GateMap, GatePoseEstimator, GateTargetTracker, select_guidance_gate
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.odometry import VehicleStateEstimator
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
INNER_LOOP_HZ = 100.0
OUTER_LOOP_HZ = 30.0
HEARTBEAT_TIMEOUT_S = 120.0
STARTUP_DATA_TIMEOUT_S = 5.0
RUN_S: float | None = None
RESET_ON_START = True
RESET_WAIT_S = 3.0
RESET_READY_TIMEOUT_S = 20.0
RESET_STABLE_S = 0.5
RESET_STABLE_MAX_SPEED_MPS = 0.03
POST_RESET_DELAY_S = 1.5
ARM_ON_START = True
ARM_TIMEOUT_S = 5.0
PRELEVEL_S = 1.0
PRELEVEL_THRUST = 0.50
TARGET_HOLD_S = 0.75
MIN_GATE_CONFIDENCE = 0.10
CONTROL_METHOD = "level_attitude_hover"
GATE_ASSOCIATION_DISTANCE_M = 6.0
GATE_MIN_OBSERVATIONS = 2
HOVER_THRUST = 0.50
LEVEL_QUATERNION = (1.0, 0.0, 0.0, 0.0)
TAKEOFF = False

def main() -> int:

    inner_period_s = 1.0 / INNER_LOOP_HZ
    outer_period_s = 1.0 / OUTER_LOOP_HZ

    run_dir = Logger.timestamped_dir(Path(__file__).resolve().parents[1] / "logs" / "runs")
    log_path = run_dir / "run.json"
    logger = Logger(
        {
            "scenario": "live_stream_minimal",
            "mavlink_endpoint": MAVLINK_ENDPOINT,
            "vision_host": VISION_HOST,
            "vision_port": VISION_PORT,
            "loop_hz": INNER_LOOP_HZ,
            "inner_loop_hz": INNER_LOOP_HZ,
            "outer_loop_hz": OUTER_LOOP_HZ,
            "control_method": CONTROL_METHOD,
        }
    )

    print(f"Starting run at {run_dir}...")
    print(f">> Inner loop rate {INNER_LOOP_HZ}")
    print(f">> Outer loop rate {OUTER_LOOP_HZ}")

    vehicle_state_estimator = VehicleStateEstimator()
    mavlink_client = MavlinkClient(endpoint=MAVLINK_ENDPOINT)

    vision_rx = VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=run_dir / "vision_frames")
    vision_perception = VisionPerceptionService(VisionPerceptionConfig(run_landmarker=False))

    gate_pose_estimator = GatePoseEstimator()
    gate_map = GateMap(
        association_distance_m=GATE_ASSOCIATION_DISTANCE_M,
        min_observations=GATE_MIN_OBSERVATIONS,
    )
    gate_target_tracker = GateTargetTracker(hold_s=TARGET_HOLD_S)

    body_rate_guidance = BodyRateGuidanceController()
    hot_start_planner = HotStartPlanner()

    inner_cycle = 0
    outer_cycle = 0

    started_s = time.perf_counter()
    next_inner_cycle_s = started_s
    
    imu_data_t = None
    telemetry = None
    latest_frame = None

    try:

        ## STARTUP PROCESS
        vision_rx.start_listener()
        logger.log_event("vision_started", receiver=vision_rx.snapshot(), perception=vision_perception.snapshot())

        mavlink_client.connect(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
        mavlink_client.start_heartbeat()
        mavlink_client.subscribe_telemetry()
        logger.log_event("mavlink_connected", bridge=mavlink_client.snapshot())

        telemetry = mavlink_client.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event("mavlink_receiving", sim_time_ns=telemetry.sim_time_ns)

        frame = vision_rx.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event(
            "vision_receiving",
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            receiver=vision_rx.snapshot(),
        )

        ## RESET SIM AFTER START
        if RESET_ON_START:
            mavlink_client.send_sim_reset_command()
            logger.log_event("simulator_reset_sent")

            # clear states, buffers, and maps
            vehicle_state_estimator.reset()
            vision_rx.clear_buffer()
            gate_target_tracker.clear()
            gate_map.clear()
            
            reset_countdown_deadline_s = time.perf_counter() + RESET_WAIT_S

            # before the countdown deadline has been reached
            while time.perf_counter() < reset_countdown_deadline_s:
                loop_started_s = time.perf_counter()
                imu_data_t = mavlink_client.latest_imu

                if not vehicle_state_estimator.initialized and imu_data_t:
                    vehicle_state_estimator.initialize_from_imu(imu_data_t=imu_data_t)

                vehicle_state = vehicle_state_estimator.update(imu_data_t=imu_data_t)
                latest_frame = vision_rx.get_next_frame()

                try:
                    # init gate map
                    observation = vision_perception.process_vision_frame(frame=latest_frame)
                    gates = observation.gates

                    # init hot start path plan
                    if vehicle_state is not None and gates:
                        gate_pose_estimator.update_gate_map_from_observation(
                            observation,
                            vehicle_state=vehicle_state,
                            gate_map=gate_map,
                        )
                        hot_start_path = hot_start_planner.plan_from_gate_map(gate_map)

                        logger.log_planned_path(
                            hot_start_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
                            cycle=inner_cycle,
                            planner="hot_start_reset",
                        )

                except Exception as error:
                    print(error)

                # timing
                next_inner_cycle_s += inner_period_s
                sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())

                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            if POST_RESET_DELAY_S > 0.0:
                time.sleep(POST_RESET_DELAY_S)

        ## DO NOT RESET ON START
        else:

            telemetry = vehicle_state_estimator.wait_for_update(mavlink_client.get_telemetry, timeout_s=HEARTBEAT_TIMEOUT_S)

        if ARM_ON_START:
            mavlink_client.arm_and_wait(timeout_s=ARM_TIMEOUT_S)
            logger.log_event("armed", bridge=mavlink_client.snapshot())

        # body_rate_guidance.run_prelevel(
        #     telemetry_client=mavlink_client,
        #     vehicle_state_estimator=vehicle_state_estimator,
        #     send_attitude_target=mavlink_client.send_attitude_target,
        #     duration_s=PRELEVEL_S,
        #     thrust=PRELEVEL_THRUST,
        #     hz=INNER_LOOP_HZ,
        #     log_event=logger.log_event,
        # )


        ## MAIN LOOP
        control_started_s = time.perf_counter()
        next_inner_cycle_s = control_started_s
        next_outer_cycle_s = control_started_s

        while RUN_S is None or time.perf_counter() - control_started_s < RUN_S:
            # inner loop timing
            loop_started_s = time.perf_counter()
            scheduled_s = next_inner_cycle_s

            # inner loop: ingest telemetry and update state
            telemetry = mavlink_client.get_telemetry()
            imu_data_t = mavlink_client.latest_imu
            vehicle_state = vehicle_state_estimator.update(imu_data_t=imu_data_t)

            if inner_cycle % int(INNER_LOOP_HZ) == 0:
                print(
                    (f"Vehicle state position - {vehicle_state.position_local_ned_m if telemetry else 'No Telemetry yet'} -- " f"Vehicle state acceleration (local NED) - {vehicle_state_estimator.state.acceleration_local_ned_mps2}"),
                    flush=True,
                )
                print(
                    (f"Vehicle attitude quat local NED - {vehicle_state_estimator.attitude_euler_local_ned(unit="deg")}"),
                    flush=True
                )

            if telemetry is not None:
                logger.log_telemetry(telemetry, cycle=inner_cycle)

            outer_loop_ran = loop_started_s >= next_outer_cycle_s
            if outer_loop_ran:
                next_outer_cycle_s += outer_period_s
                latest_frame = vision_rx.get_next_frame()
                outer_cycle += 1

            # outer loop: update vision, map, planner, and target selection
            if outer_loop_ran and latest_frame is not None:
                vision_rx.record_frame_cycle(latest_frame.frame_id, inner_cycle)
                frame_log = {
                    "frame_id": latest_frame.frame_id,
                    "cycle": inner_cycle,
                    "outer_cycle": outer_cycle,
                    "sim_time_ns": latest_frame.sim_time_ns,
                    "saved_path": latest_frame.saved_path,
                    "jpeg_size": len(latest_frame.jpeg_bytes),
                }

                # analyze frame with CNN, update gate maps, and generate hot start path
                try:
                    pass
                    # observation = vision_perception.process_vision_frame(latest_frame)
                    # frame_log["gate_count"] = len(observation.gates)
                    # frame_log["observation"] = observation.to_controller_payload(output_dir="memory")

                    # if latest_imu is not None and vehicle_state is not None and vehicle_state.position_local_ned_m is not None:
                    #     mapped_gates = gate_pose_estimator.update_gate_map_from_observation(
                    #         observation,
                    #         vehicle_state=vehicle_state,
                    #         gate_map=gate_map,
                    #         allow_new_gates=not gate_map.has_authoritative_gates(),
                    #     )
                    #     hot_start_path = hot_start_planner.plan_from_gate_map(gate_map)
                    #     hot_start_log = hot_start_path.to_log_dict(
                    #         origin_local_ned_m=vehicle_state.position_local_ned_m,
                    #     )
                    #     frame_log["hot_start_path"] = hot_start_log
                    #     logger.log_planned_path(
                    #         hot_start_log,
                    #         cycle=inner_cycle,
                    #         frame_id=latest_frame.frame_id,
                    #         sim_time_ns=telemetry.sim_time_ns,
                    #         planner="hot_start",
                    #     )
                    #     selected_gate = select_guidance_gate(
                    #         mapped_gates,
                    #         telemetry=telemetry,
                    #         min_confidence=MIN_GATE_CONFIDENCE,
                    #     )

                    #     if selected_gate is not None:
                    #         gate_target_tracker.update(selected_gate, now_s=loop_started_s, frame_id=latest_frame.frame_id)
                    #         frame_log["selected_guidance_gate"] = {
                    #             "id": selected_gate.gate_id,
                    #             "position_local_ned_m": [float(value) for value in selected_gate.position_local_ned_m],
                    #             "position_relative_ned_m": None
                    #             if selected_gate.position_relative_ned_m is None
                    #             else [float(value) for value in selected_gate.position_relative_ned_m],
                    #             "confidence": float(selected_gate.confidence),
                    #             "sequence": selected_gate.sequence,
                    #         }
                    #     frame_log["mapped_gates"] = [
                    #         {
                    #             "id": gate.gate_id,
                    #             "position_local_ned_m": [float(value) for value in gate.position_local_ned_m],
                    #             "position_relative_ned_m": None
                    #             if gate.position_relative_ned_m is None
                    #             else [float(value) for value in gate.position_relative_ned_m],
                    #             "quaternion": [float(value) for value in gate.quaternion],
                    #             "confidence": float(gate.confidence),
                    #             "sequence": gate.sequence,
                    #             "observation_count": gate.observation_count,
                    #             "last_observed_cycle": gate.last_observed_cycle,
                    #         }
                    #         for gate in mapped_gates
                    #     ]

                    # else:
                    #     frame_log["mapping_status"] = "skipped_no_telemetry"
                    
                    # logger.log_vision_frame(frame_log, cycle=outer_cycle, status="processed")
                except Exception as error:  # noqa: BLE001
                    pass
                    # logger.log_vision_frame(frame_log, cycle=inner_cycle, status="failed", error=str(error))
                    # print(f"vision frame={latest_frame.frame_id} failed: {error}", flush=True)

            if telemetry is None:
                command_result = {"emitted": False, "reason": "missing_highres_imu"}
            else:
                command_result = None
                if TAKEOFF:
                    payload = body_rate_guidance.build_attitude_command(
                        quaternion=LEVEL_QUATERNION,
                        thrust=HOVER_THRUST,
                        source="main_level_attitude_hover",
                        phase="hover",
                        metadata={
                            "vehicle_state_sim_time_ns": int(vehicle_state.sim_time_ns),
                            "target_pitch_rad": 0.0,
                        },
                    )
                    # mavlink_client.send_attitude_target(payload)
                    test_motor_cmd = 0.274
                    mavlink_client.send_motor_target([test_motor_cmd] * 4)

                    command_result = {
                        "emitted": True,
                        "reason": "streamed_level_attitude_hover",
                        "command": payload,
                        "target": None,
                        "target_age_s": None,
                        "hover": {
                            "thrust": HOVER_THRUST,
                            "quaternion": LEVEL_QUATERNION,
                        },
                    }

            # timing
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())
            loop_elapsed_ms = (time.perf_counter() - loop_started_s) * 1000.0

            logger.log_cycle(
                cycle=inner_cycle,
                sim_time_ns=telemetry.sim_time_ns if telemetry else None,
                wall_elapsed_ms=(loop_started_s - started_s) * 1000.0,
                loop_elapsed_ms=loop_elapsed_ms,
                deadline_lateness_ms=max(0.0, loop_started_s - scheduled_s) * 1000.0,
                sleep_ms=sleep_s * 1000.0,
                telemetry=telemetry,
                vision_frame_id=latest_frame.frame_id if latest_frame else None,
                bridge=mavlink_client.snapshot(),
                command=command_result,
                inner_loop={
                    "cycle": inner_cycle,
                    "hz": INNER_LOOP_HZ,
                },
                outer_loop={
                    "cycle": outer_cycle,
                    "hz": OUTER_LOOP_HZ,
                    "ran": outer_loop_ran,
                },
                gate_target_tracker=gate_target_tracker.snapshot(now_s=loop_started_s),
                hover={
                    "thrust": HOVER_THRUST,
                    "quaternion": LEVEL_QUATERNION,
                },
                body_rate_guidance=body_rate_guidance.snapshot(),
                vision={
                    **vision_rx.snapshot(),
                    "perception": vision_perception.snapshot(),
                    "gate_count": len(gate_map.get_next_gates(10_000)),
                },
            )
            logger.log_gate_map(
                gate_map.get_next_gates(10_000),
                cycle=inner_cycle,
                sim_time_ns=telemetry.sim_time_ns if telemetry else None,
            )

            if inner_cycle % int(INNER_LOOP_HZ) == 0:
                print(
                    f"inner_cycle={inner_cycle} outer_cycle={outer_cycle} vehicle_state.position_local_ned={vehicle_state_estimator.state.position_local_ned_m if telemetry else 'no vehicle state'} "
                    f"IMU.Acceleration_body_frd={mavlink_client.latest_imu.acceleration_body_frd_mps2 if mavlink_client.latest_imu else 'no HIGHRES_IMU'} "
                    f"vision_frame={latest_frame.frame_id if latest_frame else 'none'} "
                    f"loop_ms={loop_elapsed_ms:.2f}",
                    flush=True,
                )

            inner_cycle += 1

            # sleep until next cycle begins
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.log_event("interrupted")

    # SHUTDOWN
    finally:

        vision_rx.shutdown()
        if getattr(mavlink_client, "connected", False):
            try:
                shutdown_telemetry = telemetry
                if shutdown_telemetry is None:
                    raw_shutdown_telemetry = mavlink_client.get_telemetry()
                    shutdown_telemetry = vehicle_state_estimator.update_telemetry(raw_shutdown_telemetry)
                mavlink_client.send_attitude_target(
                    body_rate_guidance.build_stop_command(
                        vehicle_state_estimator.state if shutdown_telemetry is not None else None,
                        source="main_shutdown_stop",
                    )
                )
            except Exception as error:  # noqa: BLE001
                logger.log_event("shutdown_stop_failed", error=str(error))
        mavlink_client.shutdown()
        logger.log_event(
            "shutdown",
            bridge=mavlink_client.snapshot(),
            vision=vision_rx.snapshot(),
            perception=vision_perception.snapshot(),
        )
        logger.save_run(log_path)
        print(f"Log saved to {log_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
