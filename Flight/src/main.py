from pathlib import Path
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from core.control.hover.controller import HoverController
from core.coordinates import quaternion_from_roll_pitch_yaw_deg
from autonomy.planning import PathManager
from core.control.attitude import AttitudeController
from core.control.carrot import CarrotController
from core.logging import Logger, generate_mp4
from core.logging.obs import OBSRecorder
from core.schemas import MavlinkHighresImu, MavlinkTelemetry
from core.modes.system_mode import SystemModeManager
from core.utils import time_since
from mapping.gates import GateMap
from mapping.perception import VisionGateObservation, VisionObservation
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.odometry import OpenCvMonocularVioProvider, VehicleStateEstimator, VioCorrectionConfig, VioFrontendConfig
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
SIM_RUNTIME = "VQ_1"
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
INNER_LOOP_HZ = 100.0
OUTER_LOOP_HZ = 30.0
HEARTBEAT_TIMEOUT_S = 120.0
STARTUP_DATA_TIMEOUT_S = 5.0
RUN_S: float | None = None
RESET_WAIT_S = 1.5
GATE_MAP_RESET_WAIT_S = 1.5
RESET_READY_TIMEOUT_S = 20.0
RESET_STABLE_S = 0.5
RESET_STABLE_MAX_SPEED_MPS = 0.03
POST_RESET_DELAY_S = 1.5
ARM_TIMEOUT_S = 5.0
TARGET_HOLD_S = 0.75
MIN_GATE_CONFIDENCE = 0.10
CONTROL_METHOD = "carrot_motor_test"
PLANNING_MODE = "observed_next_two"
GATE_ASSOCIATION_DISTANCE_M = 6.0
GATE_MIN_OBSERVATIONS = 2
CARROT_LOOKAHEAD_M = 1.5
PLANNING_GATE_COUNT = 2
EXCLUSION_DISTANCE = 2.0
GATE_MAX_PLANNING_DISTANCE_M = 25.0
GATE_PASSED_DISTANCE_M = 2.0
GATE_CENTER_TOLERANCE_M = 0.05
SPLINE_CORNER_TIGHTNESS = 0.75
ALLOW_FLIGHT = True
CREATE_VIDEO = False
RECORD_SCREEN = False
ACTIVATE_PLANNED_PATH = True
ENABLE_VIO = False
VIO_CAMERA_HORIZONTAL_FOV_DEG = 90.0
VIO_CAMERA_TILT_DEG = 20.0
VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M = (0.0, 0.0, 0.0)

# Test
TARGET_QUATERNION = quaternion_from_roll_pitch_yaw_deg(0.0, -0.8, 0)
TARGET_THRUST = 0.265


def _observed_gate_records_for_planning(
    observation: VisionObservation,
    *,
    gate_map: GateMap,
    vehicle_state,
) -> list:
    records = []
    for sequence, observed_gate in enumerate(observation.gates):
        if float(observed_gate.position_confidence) < MIN_GATE_CONFIDENCE:
            continue
        if float(observed_gate.position_camera_m[2]) <= 0.0:
            continue
        records.append(
            gate_map.gate_record_from_observation(
                observed_gate,
                vehicle_state=vehicle_state,
                sequence=sequence,
                observed_cycle=observation.frame_id,
            )
        )
    return records


def main() -> int:
    inner_period_s = 1.0 / INNER_LOOP_HZ
    outer_period_s = 1.0 / OUTER_LOOP_HZ

    run_dir = Logger.timestamped_dir(Path(__file__).resolve().parents[1] / "logs" / "runs")
    log_path = run_dir / "run.json"
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
    print(f"Starting run at {run_dir}...")

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
    vision_rx = VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=run_dir / "vision_frames")
    vision_perception = VisionPerceptionService(VisionPerceptionConfig(backend="deterministic_0721", run_landmarker=False))
    vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
    obs_recorder = OBSRecorder(run_dir)
    gate_map = GateMap(
        confirmation_observations=2,
        merge_radius_m=2,
        min_candidate_orientation_confidence=0.8,
        min_candidate_position_confidence=0.9,
        min_promotion_orientation_confidence=0.8,
        min_promotion_position_confidence=0.9
    )
    path_manager = PathManager(
        max_gates=PLANNING_GATE_COUNT,
        exclusion_distance_m=EXCLUSION_DISTANCE,
        max_gate_distance_m=GATE_MAX_PLANNING_DISTANCE_M,
        passed_gate_distance_m=GATE_PASSED_DISTANCE_M,
        gate_center_tolerance_m=GATE_CENTER_TOLERANCE_M,
        spline_corner_tightness=SPLINE_CORNER_TIGHTNESS,
        planning_mode=PLANNING_MODE,
        gate_association_distance_m=GATE_ASSOCIATION_DISTANCE_M,
        gate_min_observations=GATE_MIN_OBSERVATIONS,
    )
    system_mode_manager = SystemModeManager()
    logger.log_event("system_mode_initialized", system_mode=system_mode_manager.system_mode.value)

    test_path = path_manager.build_straight_line(
        length_m=100.0,
        point_count=200,
        up_down_angle_deg=2.5,
        left_right_angle_deg=0.0
    )

    # test_path = path_manager.build_test_path(
    #     length_m=120,
    #     width_m=10,
    #     height_m=0,
    #     point_count=200
    # )

    planned_path = None

    # controllers
    attitude_controller = AttitudeController(
        roll_gain=1.2,
        pitch_gain=1.2,
        yaw_gain=0.5,
        damping=0.15,
        max_body_rate_rps=2.0
        )
    
    carrot_controller = CarrotController(
        speed_mps=3,
        lookahead_m=CARROT_LOOKAHEAD_M,
        position_gain=5.5,
        velocity_gain=1.5,
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
    logger.log_event(
        "vision_started",
        receiver=vision_rx.snapshot(),
        perception=vision_perception.snapshot(),
        vio=vio_provider.snapshot(),
    )

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
        time_sim_reset_s = time.perf_counter()
        logger.log_event(
            "simulator_reset_sent",
            pre_reset_sim_time_ns=telemetry.sim_time_ns if telemetry else None,
        )

        if POST_RESET_DELAY_S > 0.0:
            time.sleep(POST_RESET_DELAY_S)
            logger.log_event(
                "simulator_settle_complete",
                elapsed_s=time.perf_counter() - time_sim_reset_s,
                settle_delay_s=POST_RESET_DELAY_S,
            )

        # clear pre-reset samples, then wait for fresh post-reset telemetry and vision
        mavlink_client.clear_cached_telemetry()
        vision_rx.clear_buffer()
        vio_provider.reset()
        telemetry = mavlink_client.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event("post_reset_mavlink_receiving", sim_time_ns=telemetry.sim_time_ns)

        frame = vision_rx.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event("post_reset_vision_receiving", sim_time_ns=frame.sim_time_ns)

        # start screen recording
        if RECORD_SCREEN:
            logger.log_event("obs_recording_started", sim_time_ns=telemetry.sim_time_ns) if obs_recorder.start_recording() else logger.log_event("obs_recording_failed", sim_time_ns=telemetry.sim_time_ns)

        stationary_imu_samples: list[MavlinkHighresImu] = []
        last_calibration_imu_time_boot_us: int | None = None

        reset_countdown_deadline_s = time.perf_counter() + RESET_WAIT_S
        
        while time.perf_counter() < reset_countdown_deadline_s: # TODO: change to montoring the start lights rather than a countdown
            imu_data_t = mavlink_client.latest_imu
            latest_frame = vision_rx.get_next_frame()

            # collect imu samples for estimating sensor drift/bias
            if (imu_data_t is not None and imu_data_t.time_boot_us != last_calibration_imu_time_boot_us):
                stationary_imu_samples.append(imu_data_t)
                last_calibration_imu_time_boot_us = imu_data_t.time_boot_us

            # init gate map and path plan
            if latest_frame is not None:
                observation = vision_perception.process_vision_frame(latest_frame)
        

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

        if vehicle_state_estimator.initialized is False:
            system_mode_manager.handle_fault("no_stationary_imu_samples_or_other_failure")
            logger.log_event(
                "initialization_failed",
                reason=system_mode_manager.fault_reason,
                system_mode=system_mode_manager.system_mode.value,
            )
            raise Exception("Initialization failed.")

        # if latest_frame is not None:
        #     observation = vision_perception.process_vision_frame(latest_frame)
        #     logger.log_event(
        #         "deterministic_vision_test_output",
        #         frame_id=latest_frame.frame_id,
        #         sim_time_ns=latest_frame.sim_time_ns,
        #         gate_count=len(observation.gates),
        #         observation=observation.to_controller_payload(output_dir="memory"),
        #         perception=vision_perception.snapshot(),
        #     )
        # else:
        #     logger.log_event("deterministic_vision_test_skipped", reason="no_latest_frame")

        # gate_map.update_from_observation(observation=observation, vehicle_state=vehicle_state_estimator.state)
        # gate_map.mark_passed_near_position(
        #     vehicle_state.position_local_ned_m,
        #     distance_m=GATE_PASSED_DISTANCE_M,
        # )
        # logger.log_gate_map(
        #     gate_map.gates,
        #     cycle=inner_cycle,
        #     inner_cycle=inner_cycle,
        #     outer_cycle=outer_cycle,
        #     frame_id=latest_frame.frame_id if latest_frame else None,
        #     frame_sim_time_ns=latest_frame.sim_time_ns if latest_frame else None,
        #     source="deterministic_vision_test_output",
        # )

        gate_map_reset_countdown_deadline_s = time.perf_counter() + GATE_MAP_RESET_WAIT_S
                
        while time.perf_counter() < gate_map_reset_countdown_deadline_s: # TODO: change to montoring the start lights rather than a countdown
            imu_data_t = mavlink_client.latest_imu
            latest_frame = vision_rx.get_next_frame()

            # init gate map and path plan
            if latest_frame is not None:
                observation = vision_perception.process_vision_frame(latest_frame)
                logger.log_event(
                    "deterministic_vision_test_output",
                    frame_id=latest_frame.frame_id,
                    sim_time_ns=latest_frame.sim_time_ns,
                    gate_count=len(observation.gates),
                    observation=observation.to_controller_payload(output_dir="memory"),
                    perception=vision_perception.snapshot(),
                )

                gate_map.update_from_observation(observation=observation, vehicle_state=vehicle_state_estimator.state)
                gate_map.mark_passed_near_position(
                    vehicle_state.position_local_ned_m,
                    distance_m=GATE_PASSED_DISTANCE_M,
                )
                logger.log_gate_map(
                    gate_map.gates,
                    cycle=inner_cycle,
                    inner_cycle=inner_cycle,
                    outer_cycle=outer_cycle,
                    frame_id=latest_frame.frame_id if latest_frame else None,
                    frame_sim_time_ns=latest_frame.sim_time_ns if latest_frame else None,
                    source="deterministic_vision_test_output",
                )
                if path_manager.planning_mode == "observed_next_two":
                    observed_planning_gates = _observed_gate_records_for_planning(
                        observation,
                        gate_map=gate_map,
                        vehicle_state=vehicle_state,
                    )
                    planned_path = path_manager.plan_from_observed_gates(
                        observed_planning_gates,
                        position_local_ned_m=vehicle_state.position_local_ned_m,
                        activate=ACTIVATE_PLANNED_PATH,
                    )
            else:
                logger.log_event("deterministic_vision_test_skipped", reason="no_latest_frame")

            # timing
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())

            if sleep_s > 0.0:
                time.sleep(sleep_s)

        if path_manager.planning_mode == "observed_next_two":
            if planned_path is None and observation is not None:
                observed_planning_gates = _observed_gate_records_for_planning(
                    observation,
                    gate_map=gate_map,
                    vehicle_state=vehicle_state,
                )
                planned_path = path_manager.plan_from_observed_gates(
                    observed_planning_gates,
                    position_local_ned_m=vehicle_state.position_local_ned_m,
                    activate=ACTIVATE_PLANNED_PATH,
                )
        else:
            planned_path = path_manager.plan_from_gate_map(
                gate_map.gates,
                position_local_ned_m=vehicle_state.position_local_ned_m,
                activate=ACTIVATE_PLANNED_PATH,
            )

        logger.log_test_path(
            test_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
            cycle=inner_cycle,
            planner="straight_line_test_path",
        )
        if planned_path is not None:
            logger.log_planned_path(
                planned_path.to_log_dict(origin_local_ned_m=vehicle_state.position_local_ned_m),
                cycle=inner_cycle,
                planner=path_manager.planning_mode,
            )

        # arm drone
        mavlink_client.arm_and_wait(timeout_s=ARM_TIMEOUT_S)
        system_mode_manager.update_mode("arm")
        logger.log_event("armed", bridge=mavlink_client.snapshot(), system_mode=system_mode_manager.system_mode.value)

        if not system_mode_manager.is_armed():
            system_mode_manager.handle_fault("system_not_armed_after_arm_command")
            logger.log_event(
                "arm_mode_check_failed",
                bridge=mavlink_client.snapshot(),
                reason=system_mode_manager.fault_reason,
                system_mode=system_mode_manager.system_mode.value,
            )
            raise Exception("System mode failed to enter ARMED.")
        
        ## MAIN LOOP
        control_started_s = time.perf_counter()
        next_inner_cycle_s = control_started_s
        next_outer_cycle_s = control_started_s

        system_mode_manager.update_mode("start")
        if not system_mode_manager.is_racing():
            system_mode_manager.handle_fault("system_not_racing_before_flight")
            logger.log_event(
                "flight_start_mode_check_failed",
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
        
        carrot_target = None
        vision_pending = None
        vio_measurement_pending = None

        time_began_flight = time.perf_counter()

        while RUN_S is None or time.perf_counter() - control_started_s < RUN_S:
            # inner loop timing
            loop_started_s = time.perf_counter()
            scheduled_s = next_inner_cycle_s

            # inner loop: ingest telemetry and update state
            raw_telemetry = mavlink_client.get_telemetry()
            imu_data_t = mavlink_client.latest_imu
            if ENABLE_VIO and imu_data_t is not None:
                vio_provider.add_imu_sample(imu_data_t)
            vio_measurement_for_update = vio_measurement_pending
            vehicle_state = vehicle_state_estimator.update(
                imu_data_t=imu_data_t,
                vio_measurement=vio_measurement_for_update,
            )
            vio_measurement_pending = None
            telemetry = raw_telemetry
            if raw_telemetry is not None and vehicle_state is not None:
                telemetry = MavlinkTelemetry(
                    sim_time_ns=vehicle_state.sim_time_ns,
                    vehicle_state=vehicle_state,
                    imu=raw_telemetry.imu,
                    system_status=raw_telemetry.system_status,
                    reset_count=raw_telemetry.reset_count,
                    sim_truth=raw_telemetry.sim_truth,
                    raw={
                        **raw_telemetry.raw,
                        "vehicle_state_source": "vehicle_state_estimator_highres_imu_vio"
                        if vio_measurement_for_update is not None
                        and vehicle_state_estimator.last_vio_status == "accepted"
                        else "vehicle_state_estimator_highres_imu",
                        "vio_status": vehicle_state_estimator.last_vio_status,
                        "vio_residual": vehicle_state_estimator.last_vio_residual,
                    },
                )

            if telemetry is not None:
                logger.log_telemetry(telemetry, inner_cycle=inner_cycle)

            outer_loop_ran = loop_started_s >= next_outer_cycle_s

            # outer loop: update vision, map, planner, and target selection
            if outer_loop_ran:
                next_outer_cycle_s += outer_period_s
                outer_cycle += 1

                if vision_pending is not None and vision_pending[0].done():
                    vision_future, frame_log, frame_outer_cycle, frame_vehicle_state = vision_pending
                    try:
                        observation = vision_future.result()
                        frame_log["gate_count"] = len(observation.gates)
                        frame_log["observation"] = observation.to_controller_payload(output_dir="memory")
                        logger.log_vision_frame(frame_log, cycle=frame_outer_cycle, status="processed")
                        gate_map.update_from_observation(
                            observation=observation,
                            vehicle_state=frame_vehicle_state,
                        )
                        gate_map.mark_passed_near_position(
                            frame_vehicle_state.position_local_ned_m,
                            distance_m=GATE_PASSED_DISTANCE_M,
                        )
                        logger.log_gate_map(
                            gate_map.gates,
                            cycle=frame_log["inner_cycle"],
                            inner_cycle=frame_log["inner_cycle"],
                            outer_cycle=frame_outer_cycle,
                            frame_id=frame_log["frame_id"],
                            frame_sim_time_ns=frame_log["sim_time_ns"],
                            source="vision_frame_processed",
                        )
                        if path_manager.planning_mode == "observed_next_two":
                            observed_planning_gates = _observed_gate_records_for_planning(
                                observation,
                                gate_map=gate_map,
                                vehicle_state=frame_vehicle_state,
                            )
                            planned_path = path_manager.plan_from_observed_gates(
                                observed_planning_gates,
                                position_local_ned_m=frame_vehicle_state.position_local_ned_m,
                                activate=ACTIVATE_PLANNED_PATH,
                            )
                        else:
                            planned_path = path_manager.plan_from_gate_map(
                                gate_map.gates,
                                position_local_ned_m=frame_vehicle_state.position_local_ned_m,
                                activate=ACTIVATE_PLANNED_PATH,
                            )
                        logger.log_planned_path(
                            planned_path.to_log_dict(origin_local_ned_m=frame_vehicle_state.position_local_ned_m),
                            cycle=inner_cycle,
                            outer_cycle=outer_cycle,
                            frame_id=frame_log["frame_id"],
                            planner=path_manager.planning_mode,
                        )
                    except Exception as error:  # noqa: BLE001
                        logger.log_vision_frame(frame_log, cycle=frame_outer_cycle, status="failed", error=str(error))
                        print(f"vision frame={frame_log['frame_id']} failed: {error}", flush=True)
                    vision_pending = None

                latest_frame = None if vision_pending is not None else vision_rx.get_next_frame()

                # compute attitude target for path-following test
                carrot = path_manager.carrot_point(
                    vehicle_state.position_local_ned_m,
                    carrot_controller.lookahead_m,
                )
                carrot_target = carrot_controller.compute_control(
                    vehicle_state=vehicle_state,
                    carrot=carrot,
                )
           
                if latest_frame is not None:
                    vision_rx.record_frame_cycle(latest_frame.frame_id, inner_cycle)
                    if ENABLE_VIO:
                        vio_measurement_pending = vio_provider.process_frame(latest_frame)
                    frame_log = {
                        "frame_id": latest_frame.frame_id,
                        "inner_cycle": inner_cycle,
                        "outer_cycle": outer_cycle,
                        "sim_time_ns": latest_frame.sim_time_ns,
                        "saved_path": latest_frame.saved_path,
                        "jpeg_size": len(latest_frame.jpeg_bytes),
                    }

                    # analyze frame with CNN, update gate maps, and generate path
                    vision_pending = (
                        vision_executor.submit(vision_perception.process_vision_frame, latest_frame),
                        frame_log,
                        outer_cycle,
                        vehicle_state,
                    )

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
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())
            loop_elapsed_ms = (time.perf_counter() - loop_started_s) * 1000.0

            if inner_cycle % int(INNER_LOOP_HZ/2) == 0:
                print(
                    f"inner_cycle={inner_cycle} - outer_cycle={outer_cycle} - loop_ms={loop_elapsed_ms:.2f}\n"
                    f"Position (local_ned_m) - {tuple(round(x, 2) for x in vehicle_state.position_local_ned_m) if telemetry else 'No Telemetry yet'}  |  " 
                    # f"State attitude euler (local NED) - {tuple(round(x, 2) for x in vehicle_state_estimator.attitude_euler_frd_deg)}  |  ",
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
                # hover={
                #     "target": hover_controller.last_payload,
                # },
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

        system_mode_manager.update_mode("finish")
        logger.log_event("flight_finished", system_mode=system_mode_manager.system_mode.value)

    except KeyboardInterrupt:
        system_mode_manager.handle_fault("keyboard_interrupt")
        logger.log_event("interrupted")

    except Exception as error:
        if system_mode_manager.system_mode.value != "FAULT":
            system_mode_manager.handle_fault(str(error))
        logger.log_exception("flight_exception", error, system_mode=system_mode_manager.system_mode.value)
        print("Error occured: ", error)
        traceback.format_exc()

    # SHUTDOWN
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
                logger.log_event("shutdown_stop_failed", error=str(error))
        mavlink_client.shutdown()
        logger.log_event(
            "shutdown",
            system_mode=system_mode_manager.system_mode.value,
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
