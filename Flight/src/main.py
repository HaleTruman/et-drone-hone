"""Minimal live MAVLink and vision stream entry point."""

import math
from pathlib import Path
import time

from autonomy.planning import HotStartPlanner, PathManager
from core.control.attitude import AttitudeMotorController
from core.control.carrot import CarrotChaserConfig, CarrotChaserController
from core.logging import Logger
from core.schemas import VehicleState
from sensing.gates import GateMap
from sensing.perception import GatePoseEstimator, GateTargetTracker, select_guidance_gate
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
CONTROL_METHOD = "carrot_motor_test"
GATE_ASSOCIATION_DISTANCE_M = 6.0
GATE_MIN_OBSERVATIONS = 2
HOVER_THRUST = 0.50
MOTOR_HOVER_COMMAND = 0.274
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
    gate_map = GateMap()

    hot_start_planner = HotStartPlanner()

    # test path
    path_manager = PathManager()

    # controllers
    carrot_controller = CarrotChaserController(
        path_manager,
        config=CarrotChaserConfig(
            desired_speed_mps=0.6,
            lookahead_time_s=1.5,
            min_lookahead_m=2.0,
            path_tolerance_m=0.60,
            lateral_position_gain=0.25,
            velocity_gain=0.35,
            max_along_track_acceleration_mps2=0.35,
            max_lateral_acceleration_mps2=0.45,
            max_vertical_acceleration_mps2=0.30,
            mass_kg=0.5094496144145988,
            thrust_coefficient_n=16.65,
            min_thrust=0.24,
            max_thrust=0.32,
        ),
    )
    attitude_motor_controller = AttitudeMotorController()

    inner_cycle = 0
    outer_cycle = 0

    started_s = time.perf_counter()
    next_inner_cycle_s = started_s
    
    imu_data_t = None
    telemetry = None
    latest_frame = None
    latest_carrot_attitude_target = None

    try:
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

        ## RESET SIM AFTER START
        if RESET_ON_START:
            mavlink_client.send_sim_reset_command()
            logger.log_event("simulator_reset_sent", sim_time_ns=telemetry.sim_time_ns)

            # clear states, buffers, and maps
            vehicle_state = vehicle_state_estimator.reset(sim_time_ns=telemetry.sim_time_ns)
            vision_rx.clear_buffer()
            gate_map.clear()
            
            reset_countdown_deadline_s = time.perf_counter() + RESET_WAIT_S

            # before the countdown deadline has been reached
            # TODO: to be changed to montoring the start lights rather than a countdown
            while time.perf_counter() < reset_countdown_deadline_s:
                imu_data_t = mavlink_client.latest_imu
                latest_frame = vision_rx.get_next_frame()

                # initialize vehicle state
                if not vehicle_state_estimator.initialized and imu_data_t:
                    vehicle_state = vehicle_state_estimator.initialize_from_imu(imu_data_t)
            
                # init gate map
                if latest_frame is not None:
                    observation = vision_perception.process_vision_frame(frame=latest_frame)
                    try:

                        # init hot start path plan
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
                        print(">> Initializing hot-start failed due to: ", error)

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
        
        # TODO: work on path_manager and allow it to update the remainder of the path from the current position of the drone (i.e reset origin and "initial" velocity and have the drone continue from there)
        test_path = build_controller_test_path((0.0, 0.0, 0.0))
        path_manager.set_waypoints(test_path)
        logger.log_planned_path(
            {
                "type": "carrot_controller_test_path",
                "waypoints_local_ned_m": [[float(axis) for axis in waypoint] for waypoint in test_path],
                "pitch_up_angle_deg": 2.5,
            },
            cycle=inner_cycle,
            planner="main_test_path",
        )

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

            if telemetry is not None:
                logger.log_telemetry(telemetry, cycle=inner_cycle)

            outer_loop_ran = loop_started_s >= next_outer_cycle_s

            # outer loop: update vision, map, planner, and target selection
            if outer_loop_ran:
                next_outer_cycle_s += outer_period_s
                latest_frame = vision_rx.get_next_frame()
                outer_cycle += 1
                if CONTROL_METHOD == "carrot_motor_test" and vehicle_state is not None:
                    latest_carrot_attitude_target = carrot_controller.compute_control(vehicle_state)

                if latest_frame is not None:
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
                    if CONTROL_METHOD == "carrot_motor_test":
                        if latest_carrot_attitude_target is None:
                            command_result = {"emitted": False, "reason": "missing_outer_loop_carrot_target"}
                        else:
                            motor_commands = attitude_motor_controller.compute_motor_commands(
                                vehicle_state,
                                latest_carrot_attitude_target,
                            )
                            mavlink_client.send_motor_target(motor_commands)

                            command_result = {
                                "emitted": True,
                                "reason": "carrot_motor_test_inner_attitude",
                                "attitude_target": latest_carrot_attitude_target,
                                "motor_commands": [float(value) for value in motor_commands],
                                "path_waypoints_local_ned_m": [
                                    [float(axis) for axis in waypoint]
                                    for waypoint in path_manager.get_waypoints()
                                ],
                                "outer_loop_cycle": outer_cycle,
                            }

                
                # test_motor_cmd = MOTOR_HOVER_COMMAND
                # mavlink_client.send_motor_target([test_motor_cmd] * 4)


            # timing
            next_inner_cycle_s += inner_period_s
            sleep_s = max(0.0, next_inner_cycle_s - time.perf_counter())
            loop_elapsed_ms = (time.perf_counter() - loop_started_s) * 1000.0

            if inner_cycle % int(INNER_LOOP_HZ) == 0:
                print(
                    f"inner_cycle={inner_cycle} - outer_cycle={outer_cycle} - loop_ms={loop_elapsed_ms:.2f}\n"
                    f"State position (local NED) - {tuple(round(x, 2) for x in vehicle_state.position_local_ned_m) if telemetry else 'No Telemetry yet'}  |  " 
                    f"State attitude euler (local NED) - {tuple(round(x, 2) for x in vehicle_state_estimator.attitude_euler_local_ned(unit="deg"))}  |  ",
                    f"State acceleration (local NED) - {tuple(round(x,4) for x in vehicle_state_estimator.state.acceleration_local_ned_mps2)}  |  " 
                    f"IMU accel (body FRD) - {tuple(round(x, 4) for x in imu_data_t.acceleration_body_frd_mps2)}  |  ",
                    f"IMU body rates (body FRD) - {tuple(round(x, 4) for x in imu_data_t.gyro_body_frd_rps)}\n",
                    flush=True,
                )

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
                hover={
                    "thrust": HOVER_THRUST,
                    "quaternion": LEVEL_QUATERNION,
                },
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


def build_controller_test_path(
    start_position_local_ned_m: tuple[float, float, float],
    *,
    length_m: float = 30.0,
    point_count: int = 7,
    pitch_up_angle_deg: float = 2.5,
) -> list[tuple[float, float, float]]:
    start_x, start_y, start_z = (float(value) for value in start_position_local_ned_m)
    up_slope = math.tan(math.radians(float(pitch_up_angle_deg)))
    waypoints: list[tuple[float, float, float]] = []
    for index in range(point_count):
        fraction = index / max(point_count - 1, 1)
        forward_m = float(length_m) * fraction
        curve_y_m = 2.0 * math.sin(fraction * math.pi / 2.0)
        up_m = up_slope * forward_m
        waypoints.append((start_x + forward_m, start_y + curve_y_m, start_z - up_m))
    return waypoints


def run_motor_prelevel(
    *,
    mavlink_client: MavlinkClient,
    vehicle_state_estimator: VehicleStateEstimator,
    attitude_motor_controller: AttitudeMotorController,
    duration_s: float,
    hz: float,
) -> None:
    attitude_motor_controller.reset()
    period_s = 1.0 / float(hz)
    deadline_s = time.perf_counter() + float(duration_s)
    next_cycle_s = time.perf_counter()
    target = {"quaternion": LEVEL_QUATERNION, "thrust": MOTOR_HOVER_COMMAND}
    while time.perf_counter() < deadline_s:
        vehicle_state = vehicle_state_estimator.update(mavlink_client.latest_imu)
        if vehicle_state is not None:
            mavlink_client.send_motor_target(
                attitude_motor_controller.compute_motor_commands(vehicle_state, target)
            )
        next_cycle_s += period_s
        sleep_s = max(0.0, next_cycle_s - time.perf_counter())
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
