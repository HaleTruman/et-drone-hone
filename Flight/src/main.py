"""Minimal live MAVLink and vision stream entry point."""

from pathlib import Path
import time

from core.control.hover.controller import HoverController
from core.coordinates import quaternion_from_roll_pitch_yaw_deg
from autonomy.planning import HotStartPlanner, PathManager
from core.control.attitude import AttitudeController
from core.control.carrot import CarrotController
from core.logging import Logger, generate_mp4
from core.logging.obs import OBSRecorder
from core.schemas import MavlinkHighresImu, MavlinkTelemetry
from sensing.gates import GateMap
from sensing.perception import GatePoseEstimator
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
RESET_WAIT_S = 3.0
RESET_READY_TIMEOUT_S = 20.0
RESET_STABLE_S = 0.5
RESET_STABLE_MAX_SPEED_MPS = 0.03
POST_RESET_DELAY_S = 1.5
ARM_TIMEOUT_S = 5.0
TARGET_HOLD_S = 0.75
MIN_GATE_CONFIDENCE = 0.10
CONTROL_METHOD = "carrot_motor_test"
GATE_ASSOCIATION_DISTANCE_M = 6.0
GATE_MIN_OBSERVATIONS = 2
CARROT_LOOKAHEAD_M = 3.0
ALLOW_FLIGHT = True
CREATE_VIDEO = False
RECORD_SCREEN = True

# Test
TARGET_QUATERNION = quaternion_from_roll_pitch_yaw_deg(0.0, -0.8, 0)
TARGET_THRUST = 0.265

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

    # clients and managers
    vehicle_state_estimator = VehicleStateEstimator()
    mavlink_client = MavlinkClient(endpoint=MAVLINK_ENDPOINT)
    vision_rx = VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=run_dir / "vision_frames")
    vision_perception = VisionPerceptionService(VisionPerceptionConfig(run_landmarker=False))
    gate_pose_estimator = GatePoseEstimator()
    gate_map = GateMap()
    hot_start_planner = HotStartPlanner() # TODO: create new hot start path schema that contains a control state rather than just a raw path. i.e we need position (local NED) and also a control state (quat/thrust)
    path_manager = PathManager()

    # path_manager.build_test_path(
    #     length_m=70,
    #     width_m=20,
    #     height_m=2.0,
    #     point_count=150
    # )
    path_manager.build_straight_line(
        length_m=100,
        up_down_angle_deg=4.5,
        left_right_angle_deg=17,
        point_count=200
    )
    obs_recorder = OBSRecorder(run_dir)

    # controllers
    attitude_controller = AttitudeController(
        roll_gain=1.0,
        pitch_gain=1.0,
        yaw_gain=0.5,
        damping=0.15,
        max_body_rate_rps=1.0
        )
    
    carrot_controller = CarrotController(
        speed_mps=1,
        position_gain=0.75,
        velocity_gain=1.0,
        initial_thrust=0.265
        )
    
    hover_controller = HoverController()

    # holders
    imu_data_t = None
    telemetry = None
    latest_frame = None
    observation = None


    ## STARTUP PROCESS
    mavlink_client.connect(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
    mavlink_client.start_heartbeat()
    mavlink_client.subscribe_telemetry()
    logger.log_event("mavlink_connected", bridge=mavlink_client.snapshot())

    telemetry = mavlink_client.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
    logger.log_event("mavlink_receiving", sim_time_ns=telemetry.sim_time_ns)

    vision_rx.start_listener()
    logger.log_event("vision_started", receiver=vision_rx.snapshot(), perception=vision_perception.snapshot())

    frame = vision_rx.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
    logger.log_event("vision_receiving", sim_time_ns=frame.sim_time_ns)

    inner_cycle = 0
    outer_cycle = 0

    started_s = time.perf_counter()
    next_inner_cycle_s = started_s
    
    # Main try/except/finally
    try:
        ## RESET SIM AFTER START
        mavlink_client.send_sim_reset_command()
        logger.log_event("simulator_reset_sent", sim_time_ns=telemetry.sim_time_ns)

        # start screen recording
        if RECORD_SCREEN:
            logger.log_event("obs_recording_started", sim_time_ns=telemetry.sim_time_ns) if obs_recorder.start_recording() else logger.log_event("obs_recording_failed", sim_time_ns=telemetry.sim_time_ns)

        # clear states, buffers, and maps
        vision_rx.clear_buffer()
        gate_map.clear()
        
        reset_countdown_deadline_s = time.perf_counter() + RESET_WAIT_S
        stationary_imu_samples: list[MavlinkHighresImu] = []
        last_calibration_imu_time_boot_us: int | None = None

        # before the countdown deadline has been reached
        while time.perf_counter() < reset_countdown_deadline_s: # TODO: change to montoring the start lights rather than a countdown
            imu_data_t = mavlink_client.latest_imu
            latest_frame = vision_rx.get_next_frame()

            # collect imu samples for estimating sensor drift/bias
            if (imu_data_t is not None and imu_data_t.time_boot_us != last_calibration_imu_time_boot_us):
                stationary_imu_samples.append(imu_data_t)
                last_calibration_imu_time_boot_us = imu_data_t.time_boot_us

            # initialize vehicle state
            if not vehicle_state_estimator.initialized and imu_data_t:
                vehicle_state = vehicle_state_estimator.initialize_from_imu(imu_data_t)
        
            # init gate map
            if latest_frame is not None:
                observation = vision_perception.process_vision_frame(frame=latest_frame)
                logger.log_vision_observation(
                    observation,
                    frame_id=latest_frame.frame_id,
                    sim_time_ns=latest_frame.sim_time_ns,
                    gate_count=len(observation.gates),
                    source=observation.source,
                    phase="reset_countdown",
                )

                # try:
                #     # init hot start path plan
                #     gate_pose_estimator.update_gate_map_from_observation(
                #         observation,
                #         vehicle_state=vehicle_state,
                #         gate_map=gate_map,
                #     )
                #     hot_start_path = hot_start_planner.plan_from_gate_map(gate_map)

                #     logger.log_planned_path(
                #         hot_start_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
                #         cycle=inner_cycle,
                #         planner="hot_start_reset",
                #     )

                # except Exception as error:
                #     print(">> Initializing hot-start failed due to: ", error)

            # timing
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())

            if sleep_s > 0.0:
                time.sleep(sleep_s)

        # init state with bias
        if stationary_imu_samples:
            vehicle_state = vehicle_state_estimator.initialize_from_stationary_imu_samples(stationary_imu_samples)
            logger.log_event(
                "stationary_imu_calibrated",
                sample_count=len(stationary_imu_samples),
                duration_s=(
                    stationary_imu_samples[-1].time_boot_us - stationary_imu_samples[0].time_boot_us
                )
                / 1_000_000,
                gyro_bias_body_frd_rps=vehicle_state_estimator.gyro_bias_body_frd_rps,
                acceleration_rest_body_frd_mps2=vehicle_state_estimator.acceleration_rest_body_frd_mps2,
                attitude_quaternion=vehicle_state.attitude_quaternion,
            )
        else:
            logger.log_event("stationary_imu_calibration_skipped", reason="no_imu_samples")

        # DELAY BUFFER POST RESET
        if POST_RESET_DELAY_S > 0.0:
            time.sleep(POST_RESET_DELAY_S)

        # arm drone
        mavlink_client.arm_and_wait(timeout_s=ARM_TIMEOUT_S)
        logger.log_event("armed", bridge=mavlink_client.snapshot())
        
        # TODO: work on path_manager and allow it to update the remainder of the path from the current position of the drone (i.e reset origin and "initial" velocity and have the drone continue from there)
        print("LATEST OBSERVATION: ", observation)
        
        ## MAIN LOOP
        control_started_s = time.perf_counter()
        next_inner_cycle_s = control_started_s
        next_outer_cycle_s = control_started_s

        logger.log_event("flight_began", flight_began_s=control_started_s)
        carrot_target = None

        while RUN_S is None or time.perf_counter() - control_started_s < RUN_S:
            # inner loop timing
            loop_started_s = time.perf_counter()
            scheduled_s = next_inner_cycle_s

            # inner loop: ingest telemetry and update state
            raw_telemetry = mavlink_client.get_telemetry()
            imu_data_t = mavlink_client.latest_imu
            vehicle_state = vehicle_state_estimator.update(imu_data_t=imu_data_t)
            telemetry = raw_telemetry
            if raw_telemetry is not None and vehicle_state is not None:
                telemetry = MavlinkTelemetry(
                    sim_time_ns=vehicle_state.sim_time_ns,
                    vehicle_state=vehicle_state,
                    imu=raw_telemetry.imu,
                    system_status=raw_telemetry.system_status,
                    reset_count=raw_telemetry.reset_count,
                    raw={**raw_telemetry.raw, "vehicle_state_source": "vehicle_state_estimator_highres_imu"},
                )

            if telemetry is not None:
                logger.log_telemetry(telemetry, inner_cycle=inner_cycle)

            outer_loop_ran = loop_started_s >= next_outer_cycle_s

            # outer loop: update vision, map, planner, and target selection
            if outer_loop_ran:
                next_outer_cycle_s += outer_period_s
                latest_frame = vision_rx.get_next_frame()
                outer_cycle += 1

                # compute attitude target for path-following test
                carrot = path_manager.carrot_point(
                    vehicle_state.position_local_ned_m,
                    CARROT_LOOKAHEAD_M,
                )
                carrot_target = carrot_controller.compute_control(
                    vehicle_state=vehicle_state,
                    carrot=carrot,
                    lookahead_m=CARROT_LOOKAHEAD_M,
                )
                # hover_target = hover_controller.compute_control(vehicle_state=vehicle_state)
           
                if latest_frame is not None:
                    vision_rx.record_frame_cycle(latest_frame.frame_id, inner_cycle)
                    frame_log = {
                        "frame_id": latest_frame.frame_id,
                        "inner_cycle": inner_cycle,
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

            if imu_data_t is None:
                command_result = {
                    "emitted": False,
                    "sim_time_ns": telemetry.sim_time_ns,
                    "reason": "missing_highres_imu"
                }

            else:
                command_result = None
                control_target = {}

                # Send flight commands for one second
                if ALLOW_FLIGHT:
                    # carrot target
                    if carrot_target is None:
                        carrot = path_manager.carrot_point(
                            vehicle_state.position_local_ned_m,
                            CARROT_LOOKAHEAD_M,
                        )
                        carrot_target = carrot_controller.compute_control(
                            vehicle_state=vehicle_state,
                            carrot=carrot,
                            lookahead_m=CARROT_LOOKAHEAD_M,
                        )

                    if carrot_target:
                        control_target = attitude_controller.compute_control(
                            vehicle_state,
                            desired_attitude_quaternion=carrot_target["quaternion"],
                            thrust=carrot_target["thrust"]
                        )
                    
                    ## TESTING ##
                    # control_target = attitude_controller.compute_control(
                    #     vehicle_state,
                    #     desired_attitude_quaternion=TARGET_QUATERNION,
                    #     thrust=TARGET_THRUST
                    # )

                    mavlink_client.send_attitude_target(control_target)

                    command_result = {
                        "emitted": True,
                        "sim_time_ns": telemetry.sim_time_ns,
                        "reason": "carrot_path_following",
                        "attitude_target": control_target,
                        "inner_loop_cycle": inner_cycle,
                        "outer_loop_cycle": outer_cycle,
                    }
                else:
                    # mavlink_client.send_motor_target(motor_commands=(0.0, 0.0, 0.0, 0.0))
                    command_result = {
                            "emitted": False,
                            "sim_time_ns": telemetry.sim_time_ns,
                            "reason": "carrot_path_following",
                            "attitude_target": control_target,
                            "inner_loop_cycle": inner_cycle,
                            "outer_loop_cycle": outer_cycle,
                        }


            # timing
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())
            loop_elapsed_ms = (time.perf_counter() - loop_started_s) * 1000.0

            if inner_cycle % int(INNER_LOOP_HZ/2) == 0:
                print(
                    f"inner_cycle={inner_cycle} - outer_cycle={outer_cycle} - loop_ms={loop_elapsed_ms:.2f}\n"
                    # f"State position (local NED) - {tuple(round(x, 2) for x in vehicle_state.position_local_ned_m) if telemetry else 'No Telemetry yet'}  |  " 
                    f"State attitude euler (local NED) - {tuple(round(x, 2) for x in vehicle_state_estimator.attitude_euler_frd_deg)}  |  ",
                    f"Attitude control command (body FRD rps) - {tuple(round(x, 4) for x in control_target["body_rates_rps"]) if "body_rates_rps" in control_target else "NO COMMAND YET"}  |  ",
                    f"Body angle error (body FRD euler) - {tuple(round(x, 4) for x in control_target["body_angle_error"]) if "body_angle_error" in control_target else "NO COMMAND YET"}  |  ",
                    f"State acceleration (local NED) - {tuple(round(x,4) for x in vehicle_state_estimator.state.acceleration_local_ned_mps2)}\n",
                    # f"IMU accel (body FRD) - {tuple(round(x, 4) for x in imu_data_t.acceleration_body_frd_mps2)}  |  ",
                    # f"IMU body rates (body FRD) - {tuple(round(x, 4) for x in imu_data_t.gyro_body_frd_rps)}\n",
                    flush=True,
                )

            logger.log_cycle(
                inner_cycle=inner_cycle,
                outer_cycle=outer_cycle,
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
                    "inner_cycle": inner_cycle,
                    "hz": INNER_LOOP_HZ,
                },
                outer_loop={
                    "outer_cycle": outer_cycle,
                    "hz": OUTER_LOOP_HZ,
                    "ran": outer_loop_ran,
                },
                carrot=carrot_controller.last_payload,
                # hover={
                #     "target": hover_controller.last_payload,
                # },
                vision={
                    **vision_rx.snapshot(),
                    "perception": vision_perception.snapshot(),
                    # "gate_count": len(gate_map.get_next_gates(10_000)),
                },
            )
            logger.log_gate_map(
                gate_map.get_next_gates(10_000),
                inner_cycle=inner_cycle,
                sim_time_ns=telemetry.sim_time_ns if telemetry else None,
            )

            inner_cycle += 1

            # sleep until next inner cycle begins
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.log_event("interrupted")

    except Exception as error:
        print("Error occured: ", error)

    # SHUTDOWN
    finally:
        if RECORD_SCREEN:
            obs_recorder.stop_recording()

        vision_rx.shutdown()
        if getattr(mavlink_client, "connected", False):
            try:
                shutdown_telemetry = telemetry
                if shutdown_telemetry is None:
                    raw_shutdown_telemetry = mavlink_client.get_telemetry()
                    shutdown_telemetry = vehicle_state_estimator.update_telemetry(raw_shutdown_telemetry)
                
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

        if CREATE_VIDEO:
            generate_mp4(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
