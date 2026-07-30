from pathlib import Path
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np 

from core.control.hover.controller import HoverController
from autonomy.pathing import PathManager
from core.control.attitude import AttitudeController
from core.control.carrot import CarrotController
from core.logging import Logger, generate_mp4
from core.logging.obs import OBSRecorder
from core.schema import MavlinkHighresImu, StateRecord, VioCorrection
from core.modes.system_mode import SystemModeManager
from core.utils import time_since
from mapping.gates import GateMap
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.odometry import OpenCvMonocularVioProvider, VehicleStateEstimator, VioCorrectionConfig, VioFrontendConfig
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

# Simulator and network endpoints.
MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
SIM_RUNTIME = "VQ_2"
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600

# Control loop timing.
INNER_LOOP_HZ = 100.0
OUTER_LOOP_HZ = 30.0
RUN_S: float | None = None

# Startup, reset, and arming timeouts.
HEARTBEAT_TIMEOUT_S = 120.0
STARTUP_DATA_TIMEOUT_S = 5.0
IMU_INIT_TIMEOUT_S = 1.5
GATE_MAP_INIT_TIMEOUT_S = 1.5
RESET_READY_TIMEOUT_S = 20.0
RESET_STABLE_S = 0.5
RESET_STABLE_MAX_SPEED_MPS = 0.03
POST_RESET_DELAY_S = 1.5
ARM_TIMEOUT_S = 5.0
TARGET_HOLD_S = 0.75

# Vision filtering and path planning.
PLANNING_MODE = "test_path" # test_path, center_targets, gate_map
PLANNING_GATE_COUNT = 2
EXCLUSION_DISTANCE = 2.0
GATE_MAX_PLANNING_DISTANCE_M = 40.0
GATE_PASSED_DISTANCE_M = 2.0
GATE_CENTER_TOLERANCE_M = 0.05
SPLINE_CORNER_TIGHTNESS = 2

# Control and output behavior.
CONTROL_METHOD = "carrot_motor_test"
CARROT_LOOKAHEAD_M = 1.5
SPEED_LOOKAHEAD_M = 8
FAILSAFE_DISTANCE = 10
ALLOW_FLIGHT = True
CREATE_VIDEO = False
RECORD_SCREEN = False

# Visual odometry configuration.
ENABLE_VIO = False
VIO_CAMERA_HORIZONTAL_FOV_DEG = 90.0
VIO_CAMERA_TILT_DEG = 20.0
VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M = (0.0, 0.0, 0.0)

# Logging
RUN_DIR = Logger.timestamped_dir(Path(__file__).resolve().parents[1] / "logs" / "runs")
LOG_PATH = RUN_DIR / "run.json"
logger = Logger(
    {
        "scenario": "live_stream_minimal",
        "sim_runtime": SIM_RUNTIME,
        "mavlink_endpoint": MAVLINK_ENDPOINT,
        "vision_host": VISION_HOST,
        "vision_port": VISION_PORT,
        "loop_hz": INNER_LOOP_HZ,
        "inner_loop_hz": INNER_LOOP_HZ,
        "outer_loop_hz": OUTER_LOOP_HZ,
        "control_method": CONTROL_METHOD,
        "planning_mode": PLANNING_MODE,
    }
)

print(f"Starting run at {RUN_DIR}...")


def main() -> int:
    inner_period_s = 1.0 / INNER_LOOP_HZ
    outer_period_s = 1.0 / OUTER_LOOP_HZ

    # clients and managers
    vehicle_state_estimator = VehicleStateEstimator(
        vio_config=VioCorrectionConfig(
            position_alpha=0.02,
            velocity_alpha=0.05,
            attitude_alpha=0.03,
        )
    )

    vio_provider = OpenCvMonocularVioProvider(
        frontend_config=VioFrontendConfig(horizontal_fov_deg=VIO_CAMERA_HORIZONTAL_FOV_DEG),
        body_to_camera_translation_body_frd_m=VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M,
        camera_tilt_deg=VIO_CAMERA_TILT_DEG,
    )

    mavlink_client = MavlinkClient(endpoint=MAVLINK_ENDPOINT, sim_runtime=SIM_RUNTIME)
    vision_rx = VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=RUN_DIR / "vision_frames")
    vision_perception = VisionPerceptionService(VisionPerceptionConfig(backend="deterministic_v3"))
    vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
    obs_recorder = OBSRecorder(RUN_DIR)
    system_mode_manager = SystemModeManager()
    gate_map = GateMap()

    path_manager = PathManager(
        max_gates=PLANNING_GATE_COUNT,
        exclusion_distance_m=EXCLUSION_DISTANCE,
        max_gate_distance_m=GATE_MAX_PLANNING_DISTANCE_M,
        passed_gate_distance_m=GATE_PASSED_DISTANCE_M,
        gate_center_tolerance_m=GATE_CENTER_TOLERANCE_M,
        spline_corner_tightness=SPLINE_CORNER_TIGHTNESS,
        planning_mode=PLANNING_MODE,
    )


    path_manager.build_straight_line(
        length_m=100.0,
        point_count=200,
        up_down_angle_deg=2.5,
        left_right_angle_deg=0.0
    )

    attitude_controller = AttitudeController(
        roll_gain=1.2,
        pitch_gain=1.2,
        yaw_gain=0.5,
        damping=0.15,
        max_body_rate_rps=3.0
    )
    
    carrot_controller = CarrotController(
        max_speed_mps=10,
        lookahead_m=CARROT_LOOKAHEAD_M,
        speed_lookahead_m=SPEED_LOOKAHEAD_M,
        position_gain=5.5,
        velocity_gain=0.75,
        initial_thrust=0.265
    )
    
    hover_controller = HoverController(
        lateral_velocity_gain=2.5,
        vertical_velocity_gain=0.18,
        vertical_acceleration_gain=0.035,
    )

    # holders
    imu_data_t = None
    telemetry = None
    latest_frame = None
    observation = None
    planned_path = None
    carrot_target = None
    vision_pending = None
    pending_vio_correction: VioCorrection | None = None

    inner_cycle = 0
    outer_cycle = 0

    started_s = time.perf_counter()
    next_inner_cycle_s = started_s
    next_outer_cycle_s = started_s


# =================================================================== STARTUP PROCESS ===================================================================
    try:
    
        mavlink_client.connect(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
        mavlink_client.start_heartbeat()
        mavlink_client.subscribe_telemetry()
        logger.log_event("mavlink_connected", time_since_startup_s = time_since(started_s), bridge=mavlink_client.snapshot())

        vision_rx.start_listener()
        logger.log_event("vision_started", time_since_startup_s = time_since(started_s), receiver=vision_rx.snapshot())

        # RESET
        mavlink_client.send_sim_reset_command()
        time_sim_reset_s = time.perf_counter()
        logger.log_event("simulator_reset_sent", time_since_startup_s = time_since(started_s))

        if POST_RESET_DELAY_S > 0.0:
            time.sleep(POST_RESET_DELAY_S)
            logger.log_event("simulator_settle_complete", time_since_startup_s = time_since(started_s), elapsed_s=time.perf_counter() - time_sim_reset_s, settle_delay_s=POST_RESET_DELAY_S)

        # clear pre-reset samples, then wait for fresh post-reset telemetry and vision
        mavlink_client.clear_cached_telemetry()
        vision_rx.clear_buffer()
        vision_rx.begin_saving_frames()
        vio_provider.reset()

        telemetry = mavlink_client.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event("mavlink_receiving", sim_time_ns=telemetry.sim_time_ns)

        frame = vision_rx.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event("vision_receiving", sim_time_ns=frame.sim_time_ns)

        # start screen recording
        if RECORD_SCREEN:
            logger.log_event("obs_recording_started", time_since_startup_s = time_since(started_s), sim_time_ns=telemetry.sim_time_ns) if obs_recorder.start_recording() else logger.log_event("obs_recording_failed", sim_time_ns=telemetry.sim_time_ns)


        # IMU calibration
        imu_calibration_samples: list[MavlinkHighresImu] = []
        last_calibration_imu_time_boot_us: int | None = None
        imu_calibration_deadline_s = time.perf_counter() + IMU_INIT_TIMEOUT_S
        
        while time.perf_counter() < imu_calibration_deadline_s: # rate: INNER_LOOP_HZ
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
        mavlink_client.arm_and_wait(timeout_s=ARM_TIMEOUT_S)
        system_mode_manager.update_mode("arm")
        logger.log_event("armed", time_since_startup_s = time_since(started_s), bridge=mavlink_client.snapshot(), system_mode=system_mode_manager.system_mode.value)


        # Vision calibration and initalization
        gate_map_init_deadline_s = time.perf_counter() + GATE_MAP_INIT_TIMEOUT_S
                
        while time.perf_counter() < gate_map_init_deadline_s: # rate: OUTER_LOOP_HZ
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

                gate_map.update(observation)

                planned_path = path_manager.plan(
                    gates=gate_map.gates,
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
                gate_map.update(observation)

            planned_path = path_manager.plan(
                gates=gate_map.gates,
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

        while RUN_S is None or time.perf_counter() - control_started_s < RUN_S:

            # ======================== INNER LOOP START ========================
            inner_loop_started_s = time.perf_counter()
            next_inner_scheduled_s = next_inner_cycle_s

            # ingest telemetry and update state
            telemetry = mavlink_client.get_telemetry()
            imu_data_t = mavlink_client.latest_imu

            if ENABLE_VIO and imu_data_t is not None:
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
                        frame_log["gate_count"] = len(observation.gates)
                        frame_log["observation"] = observation.to_controller_payload(output_dir="memory")
                        logger.log_vision_observation(
                            observation,
                            frame_id=frame_log["frame_id"],
                            inner_cycle=frame_log["inner_cycle"],
                            outer_cycle=frame_outer_cycle,
                            sim_time_ns=frame_log["sim_time_ns"],
                            gate_count=len(observation.gates),
                            perception=vision_perception.snapshot(),
                        )
                        logger.log_vision_frame(frame_log, cycle=frame_outer_cycle, status="processed")

                        # update gatemap
                        gate_map.update(observation)

                        planned_path = path_manager.plan(
                            gates=gate_map.gates,
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
                if ENABLE_VIO and latest_frame is not None:
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
                if vision_pending is None and latest_frame is not None:
                    frame_log = {
                        "frame_id": latest_frame.frame_id,
                        "inner_cycle": inner_cycle,
                        "outer_cycle": outer_cycle,
                        "sim_time_ns": latest_frame.sim_time_ns,
                        "saved_path": latest_frame.saved_path,
                        "jpeg_size": len(latest_frame.jpeg_bytes),
                    }

                    # queue vision job
                    vision_pending = (
                        vision_executor.submit(vision_perception.process_vision_frame, latest_frame, vehicle_state=vehicle_state),
                        frame_log,
                        outer_cycle,
                        vehicle_state,
                    )

                # CHECK FAILSAFE
                if ALLOW_FLIGHT and system_mode_manager.is_racing():
                    path_projection = path_manager.project(vehicle_state.position_local_ned_m)
                    path_error_m = float(path_projection["cross_track_error_m"])
                    if path_error_m > FAILSAFE_DISTANCE:
                        system_mode_manager.update_mode("finish")
                        carrot_target = None
                        logger.log_event(
                            "path_failsafe_finished",
                            reason="path_deviation_exceeded",
                            cross_track_error_m=path_error_m,
                            failsafe_distance_m=FAILSAFE_DISTANCE,
                            projection=path_projection,
                            system_mode=system_mode_manager.system_mode.value,
                            inner_cycle=inner_cycle,
                            outer_cycle=outer_cycle,
                            sim_time_ns=vehicle_state.sim_time_ns,
                        )

                # compute attitude target for path-following controller
                if ALLOW_FLIGHT and system_mode_manager.is_racing():
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

                if ALLOW_FLIGHT and system_mode_manager.is_racing():
                    # carrot target
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

                    if carrot_target:
                        control_target = attitude_controller.compute_control(
                            vehicle_state,
                            desired_attitude_quaternion=carrot_target["quaternion"],
                            thrust=carrot_target["thrust"]
                        )
                    
                    mavlink_client.send_attitude_target(control_target)

                    command_result = {
                        "emitted": True,
                        "sim_time_ns": telemetry.sim_time_ns,
                        "reason": "carrot_path_following",
                        "attitude_target": control_target,
                        "inner_loop_cycle": inner_cycle,
                        "outer_loop_cycle": outer_cycle,
                    }
                elif ALLOW_FLIGHT and system_mode_manager.is_finished():
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
                            "reason": "flight_disabled" if not ALLOW_FLIGHT else "system_mode_not_racing",
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

            if inner_cycle % int(INNER_LOOP_HZ*5) == 0:
                print(f"inner_cycle={inner_cycle} - outer_cycle={outer_cycle} - loop_ms={loop_elapsed_ms:.2f}\n", flush=True)

            state_record = (None if telemetry is None else StateRecord.from_telemetry(
                    telemetry,
                    vehicle_state=vehicle_state,
                    vio_correction=vio_measurement_for_update,
                    vio_status=vehicle_state_estimator.last_vio_status,
                    vio_residual=vehicle_state_estimator.last_vio_residual,
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
                    "hz": INNER_LOOP_HZ,
                },
                outer_loop={
                    "outer_cycle": outer_cycle,
                    "hz": OUTER_LOOP_HZ,
                    "ran": outer_loop_ran,
                },
                carrot=carrot_controller.last_payload,
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
        if RECORD_SCREEN:
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
        logger.save_run(LOG_PATH)
        print(f"Log saved to {LOG_PATH}", flush=True)

        if CREATE_VIDEO:
            generate_mp4(RUN_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
