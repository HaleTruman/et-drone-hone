"""Minimal live MAVLink and vision stream entry point."""

from pathlib import Path
import time

from autonomy.planning import HotStartPlanner
from core.control.body_rate_guidance import BodyRateGuidanceController
from core.logging import Logger
from core.schemas import MavlinkTelemetry, OdometryState
from sensing.perception import GateMap, GatePoseEstimator, GateTargetTracker, TrackGateReceiver, select_guidance_gate
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.odometry import VehicleState
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
LOOP_HZ = 30.0
HEARTBEAT_TIMEOUT_S = 120.0
STARTUP_DATA_TIMEOUT_S = 5.0
TRACK_GATE_WAIT_S = 1.0
RUN_S: float | None = None
RESET_ON_START = True
RESET_WAIT_S = 3.0
RESET_READY_TIMEOUT_S = 20.0
RESET_STABLE_S = 0.5
RESET_STABLE_MAX_SPEED_MPS = 0.03
POST_RESET_DELAY_S = 0.5
ARM_ON_START = True
ARM_TIMEOUT_S = 5.0
PRELEVEL_S = 1.0
PRELEVEL_THRUST = 0.20
TARGET_HOLD_S = 0.75
MIN_GATE_CONFIDENCE = 0.10
CONTROL_METHOD = "body_rate_guidance"
GATE_ASSOCIATION_DISTANCE_M = 6.0
GATE_MIN_OBSERVATIONS = 2


def _with_vehicle_state(telemetry: MavlinkTelemetry | None, odometry: OdometryState | None) -> MavlinkTelemetry | None:
    if telemetry is None or odometry is None:
        return None
    return MavlinkTelemetry(
        sim_time_ns=odometry.sim_time_ns,
        odometry=odometry,
        imu=telemetry.imu,
        system_status=telemetry.system_status,
        reset_count=telemetry.reset_count,
        raw={**telemetry.raw, "odometry_source": "vehicle_state_highres_imu"},
    )


def main() -> int:

    period_s = 1.0 / LOOP_HZ
    run_dir = Logger.timestamped_dir(Path(__file__).resolve().parents[1] / "logs" / "runs")
    log_path = run_dir / "run.json"
    logger = Logger(
        {
            "scenario": "live_stream_minimal",
            "mavlink_endpoint": MAVLINK_ENDPOINT,
            "vision_host": VISION_HOST,
            "vision_port": VISION_PORT,
            "loop_hz": LOOP_HZ,
            "control_method": CONTROL_METHOD,
        }
    )

    print(f"Initializing run at {run_dir}...")
    print(f">> Loop rate {LOOP_HZ}")

    vehicle_state = VehicleState()
    track_gate_receiver = TrackGateReceiver()
    mavlink_client = MavlinkClient(endpoint=MAVLINK_ENDPOINT, track_gate_receiver=track_gate_receiver)

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

    cycle = 0
    started_s = time.perf_counter()
    next_cycle_s = started_s
    telemetry = None

    try:

        ## STARTUP PROCESS
        vision_rx.start_listener()
        logger.log_event("vision_started", receiver=vision_rx.snapshot(), perception=vision_perception.snapshot())

        mavlink_client.connect(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
        mavlink_client.start_heartbeat()
        mavlink_client.subscribe_telemetry()
        logger.log_event("mavlink_connected", bridge=mavlink_client.snapshot())

        startup_telemetry = mavlink_client.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event("mavlink_receiving", sim_time_ns=startup_telemetry.sim_time_ns)

        startup_frame = vision_rx.wait_until_receiving(timeout_s=STARTUP_DATA_TIMEOUT_S)
        logger.log_event(
            "vision_receiving",
            frame_id=startup_frame.frame_id,
            sim_time_ns=startup_frame.sim_time_ns,
            receiver=vision_rx.snapshot(),
        )

        ## RESET SIM AFTER START
        if RESET_ON_START:
            mavlink_client.send_sim_reset_command()
            logger.log_event("simulator_reset_sent")

            # clear states, buffers, and maps
            vehicle_state.reset()
            vision_rx.clear_buffer()
            gate_target_tracker.clear()
            gate_map.clear()

            reset_countdown_deadline_s = time.perf_counter() + RESET_WAIT_S

            # before the countdown deadline has been reached
            while time.perf_counter() < reset_countdown_deadline_s:
                loop_started_s = time.perf_counter()
                scheduled_s = next_cycle_s

                telemetry = mavlink_client.get_telemetry()
                frame = vision_rx.get_next_frame()

                odometry = vehicle_state.update(None if telemetry is None else telemetry.imu)
                telemetry = _with_vehicle_state(telemetry, odometry) or telemetry

                # init gate map

                try:
                    observation = vision_perception.process_vision_frame(frame)
            
                    if telemetry is not None and telemetry.odometry is not None and telemetry.odometry.position_local_ned_m is not None:
                        mapped_gates = gate_pose_estimator.update_gate_map_from_observation(
                            observation,
                            telemetry=telemetry,
                            gate_map=gate_map,
                            allow_new_gates=not gate_map.has_authoritative_gates(),
                        )
                        
                        selected_gate = select_guidance_gate(
                            mapped_gates,
                            telemetry=telemetry,
                            min_confidence=MIN_GATE_CONFIDENCE,
                        )
                        if selected_gate is None:
                            selected_gate = gate_map.active_track_gate_record(
                                mavlink_client.race_status,
                                track_gate_receiver.track_gates,
                            )
                        if selected_gate is not None:
                            gate_target_tracker.update(selected_gate, now_s=loop_started_s, frame_id=frame.frame_id)
                            
                    else:
                        frame_log["mapping_status"] = "skipped_no_telemetry"
                    logger.log_vision_frame(frame_log, cycle=cycle, status="processed")
                except Exception as error:  # noqa: BLE001
                    print(f"vision frame={frame.frame_id} failed: {error}", flush=True)

                # init hot start path plan

                origin = vehicle_state.position_local_ned_m

                if gate_map.seed_from_track_gates(
                    track_gate_receiver.track_gates,
                    target_tracker=gate_target_tracker,
                    origin_local_ned_m=origin,
                ):
                    logger.log_event(
                        "track_gate_map_seeded",
                        reason="reset_countdown",
                        gate_count=len(track_gate_receiver.track_gates),
                        gate_ids=[gate.gate_id for gate in track_gate_receiver.track_gates],
                    )
                if telemetry is not None and telemetry.odometry is not None:
                    hot_start_path = hot_start_planner.plan_from_gate_map(gate_map)
                    logger.log_planned_path(
                        hot_start_path.to_log_dict(origin_local_ned_m=telemetry.odometry.position_local_ned_m),
                        cycle=cycle,
                        sim_time_ns=telemetry.sim_time_ns,
                        planner="hot_start_reset_countdown",
                    )
                
                next_cycle_s += period_s
                sleep_s = max(0.0, next_cycle_s - time.perf_counter())
                loop_elapsed_ms = (time.perf_counter() - loop_started_s) * 1000.0

                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            # Wait until drone has not been moving for a period of time (POTENTIALLY DEPRECIATED)
            # telemetry = vehicle_state.wait_until_stable(
            #     mavlink_client.get_telemetry,
            #     timeout_s=RESET_READY_TIMEOUT_S,
            #     stable_s=RESET_STABLE_S,
            #     stable_max_speed_mps=RESET_STABLE_MAX_SPEED_MPS,
            # )
            # speed_mps = sum(value * value for value in telemetry.odometry.velocity_local_ned_mps) ** 0.5
            # logger.log_event("simulator_reset_ready", speed_mps=speed_mps)

            if POST_RESET_DELAY_S > 0.0:
                time.sleep(POST_RESET_DELAY_S)

            track_gate_receiver.wait_for_track_gates(timeout_s=TRACK_GATE_WAIT_S)
            if gate_map.seed_from_track_gates(
                track_gate_receiver.track_gates,
                target_tracker=gate_target_tracker,
                origin_local_ned_m=telemetry.odometry.position_local_ned_m,
            ):
                logger.log_event(
                    "track_gate_map_seeded",
                    reason="post_reset",
                    gate_count=len(track_gate_receiver.track_gates),
                    gate_ids=[gate.gate_id for gate in track_gate_receiver.track_gates],
                )

        ## DO NOT RESET ON START
        else:
            telemetry = vehicle_state.wait_for_update(mavlink_client.get_telemetry, timeout_s=HEARTBEAT_TIMEOUT_S)
            track_gate_receiver.wait_for_track_gates(timeout_s=TRACK_GATE_WAIT_S)
            if gate_map.seed_from_track_gates(
                track_gate_receiver.track_gates,
                target_tracker=gate_target_tracker,
                origin_local_ned_m=telemetry.odometry.position_local_ned_m,
            ):
                logger.log_event(
                    "track_gate_map_seeded",
                    reason="startup",
                    gate_count=len(track_gate_receiver.track_gates),
                    gate_ids=[gate.gate_id for gate in track_gate_receiver.track_gates],
                )
        if ARM_ON_START:
            mavlink_client.arm_and_wait(timeout_s=ARM_TIMEOUT_S)
            logger.log_event("armed", bridge=mavlink_client.snapshot())

        body_rate_guidance.run_prelevel(
            telemetry_client=mavlink_client,
            vehicle_state=vehicle_state,
            send_attitude_target=mavlink_client.send_attitude_target,
            duration_s=PRELEVEL_S,
            thrust=PRELEVEL_THRUST,
            hz=LOOP_HZ,
            log_event=logger.log_event,
        )


        ## MAIN LOOP
        while RUN_S is None or time.perf_counter() - started_s < RUN_S:
            loop_started_s = time.perf_counter()
            scheduled_s = next_cycle_s
            telemetry = mavlink_client.get_telemetry()
            odometry = vehicle_state.update(None if telemetry is None else telemetry.imu)
            telemetry = _with_vehicle_state(telemetry, odometry)
            frame = vision_rx.get_next_frame()
            origin = None if telemetry is None or telemetry.odometry is None else telemetry.odometry.position_local_ned_m
            if gate_map.seed_from_track_gates(
                track_gate_receiver.track_gates,
                target_tracker=gate_target_tracker,
                origin_local_ned_m=origin,
            ):
                logger.log_event(
                    "track_gate_map_seeded",
                    reason="runtime_update",
                    gate_count=len(track_gate_receiver.track_gates),
                    gate_ids=[gate.gate_id for gate in track_gate_receiver.track_gates],
                )

            if cycle % int(LOOP_HZ) == 0:
                print(
                    f"Vehicle state position - {vehicle_state.state.position_local_ned_m if telemetry else "No Telemetry yet"}",
                    flush=True,
                )

            if telemetry is not None:
                logger.log_telemetry(telemetry, cycle=cycle)
            if frame is not None:
                vision_rx.record_frame_cycle(frame.frame_id, cycle)
                frame_log = {
                    "frame_id": frame.frame_id,
                    "cycle": cycle,
                    "sim_time_ns": frame.sim_time_ns,
                    "saved_path": frame.saved_path,
                    "jpeg_size": len(frame.jpeg_bytes),
                }
                try:
                    observation = vision_perception.process_vision_frame(frame)
                    frame_log["gate_count"] = len(observation.gates)
                    frame_log["observation"] = observation.to_controller_payload(output_dir="memory")
                    if telemetry is not None and telemetry.odometry is not None and telemetry.odometry.position_local_ned_m is not None:
                        mapped_gates = gate_pose_estimator.update_gate_map_from_observation(
                            observation,
                            telemetry=telemetry,
                            gate_map=gate_map,
                            allow_new_gates=not gate_map.has_authoritative_gates(),
                        )
                        hot_start_path = hot_start_planner.plan_from_gate_map(gate_map)
                        hot_start_log = hot_start_path.to_log_dict(
                            origin_local_ned_m=telemetry.odometry.position_local_ned_m,
                        )
                        frame_log["hot_start_path"] = hot_start_log
                        logger.log_planned_path(
                            hot_start_log,
                            cycle=cycle,
                            frame_id=frame.frame_id,
                            sim_time_ns=telemetry.sim_time_ns,
                            planner="hot_start",
                        )
                        selected_gate = select_guidance_gate(
                            mapped_gates,
                            telemetry=telemetry,
                            min_confidence=MIN_GATE_CONFIDENCE,
                        )
                        if selected_gate is None:
                            selected_gate = gate_map.active_track_gate_record(
                                mavlink_client.race_status,
                                track_gate_receiver.track_gates,
                            )
                        if selected_gate is not None:
                            gate_target_tracker.update(selected_gate, now_s=loop_started_s, frame_id=frame.frame_id)
                            frame_log["selected_guidance_gate"] = {
                                "id": selected_gate.gate_id,
                                "position_local_ned_m": [float(value) for value in selected_gate.position_local_ned_m],
                                "position_relative_ned_m": None
                                if selected_gate.position_relative_ned_m is None
                                else [float(value) for value in selected_gate.position_relative_ned_m],
                                "confidence": float(selected_gate.confidence),
                                "sequence": selected_gate.sequence,
                            }
                        frame_log["mapped_gates"] = [
                            {
                                "id": gate.gate_id,
                                "position_local_ned_m": [float(value) for value in gate.position_local_ned_m],
                                "position_relative_ned_m": None
                                if gate.position_relative_ned_m is None
                                else [float(value) for value in gate.position_relative_ned_m],
                                "quaternion": [float(value) for value in gate.quaternion],
                                "confidence": float(gate.confidence),
                                "sequence": gate.sequence,
                                "observation_count": gate.observation_count,
                                "last_observed_cycle": gate.last_observed_cycle,
                            }
                            for gate in mapped_gates
                        ]
                    else:
                        frame_log["mapping_status"] = "skipped_no_telemetry"
                    logger.log_vision_frame(frame_log, cycle=cycle, status="processed")
                except Exception as error:  # noqa: BLE001
                    logger.log_vision_frame(frame_log, cycle=cycle, status="failed", error=str(error))
                    print(f"vision frame={frame.frame_id} failed: {error}", flush=True)

            if telemetry is None or telemetry.odometry is None or telemetry.odometry.position_local_ned_m is None:
                command_result = {"emitted": False, "reason": "missing_telemetry_for_body_rate_command"}
            else:
                target = gate_target_tracker.latest(now_s=loop_started_s)
                payload = body_rate_guidance.build_guidance_command(
                    telemetry=telemetry,
                    target=target,
                    source="main_body_rate_guidance" if target is not None else "main_body_rate_hold_no_target",
                )
                mavlink_client.send_attitude_target(payload)
                command_result = {
                    "emitted": True,
                    "reason": "streamed_body_rate_guidance" if target is not None else "streamed_body_rate_hold_no_target",
                    "command": payload,
                    "target": None if target is None else target.__dict__,
                    "target_age_s": gate_target_tracker.age_s(now_s=loop_started_s),
                }

            next_cycle_s += period_s
            sleep_s = max(0.0, next_cycle_s - time.perf_counter())
            loop_elapsed_ms = (time.perf_counter() - loop_started_s) * 1000.0
            logger.log_cycle(
                cycle=cycle,
                sim_time_ns=telemetry.sim_time_ns if telemetry else None,
                wall_elapsed_ms=(loop_started_s - started_s) * 1000.0,
                loop_elapsed_ms=loop_elapsed_ms,
                deadline_lateness_ms=max(0.0, loop_started_s - scheduled_s) * 1000.0,
                sleep_ms=sleep_s * 1000.0,
                telemetry=telemetry,
                vision_frame_id=frame.frame_id if frame else None,
                bridge=mavlink_client.snapshot(),
                command=command_result,
                gate_target_tracker=gate_target_tracker.snapshot(now_s=loop_started_s),
                body_rate_guidance=body_rate_guidance.snapshot(),
                vision={
                    **vision_rx.snapshot(),
                    "perception": vision_perception.snapshot(),
                    "gate_count": len(gate_map.get_next_gates(10_000)),
                },
            )
            logger.log_gate_map(
                gate_map.get_next_gates(10_000),
                cycle=cycle,
                sim_time_ns=telemetry.sim_time_ns if telemetry else None,
            )
            if cycle % int(LOOP_HZ) == 0:
                print(
                    f"cycle={cycle} vehicle_state.position_local_ned={vehicle_state.state.position_local_ned_m if telemetry else 'no vehicle state'} "
                    f"IMU.Acceleration_body_frd={mavlink_client.latest_imu.acceleration_body_frd_mps2 if mavlink_client.latest_imu else 'no HIGHRES_IMU'} "
                    f"vision_frame={frame.frame_id if frame else 'none'} "
                    f"loop_ms={loop_elapsed_ms:.2f}",
                    flush=True,
                )

            cycle += 1
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.log_event("interrupted")
    finally:
        vision_rx.shutdown()
        if getattr(mavlink_client, "connected", False):
            try:
                shutdown_telemetry = telemetry
                if shutdown_telemetry is None:
                    raw_shutdown_telemetry = mavlink_client.get_telemetry()
                    shutdown_telemetry = _with_vehicle_state(
                        raw_shutdown_telemetry,
                        vehicle_state.update(None if raw_shutdown_telemetry is None else raw_shutdown_telemetry.imu),
                    )
                mavlink_client.send_attitude_target(
                    body_rate_guidance.build_stop_command(
                        shutdown_telemetry,
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
