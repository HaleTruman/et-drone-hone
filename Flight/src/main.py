"""Minimal live MAVLink and vision stream entry point."""

from pathlib import Path
import time

from autonomy.planning import HotStartPlanner
from core.control.body_rate_guidance import BodyRateGuidanceController
from core.logging import Logger
from sensing.perception import GateMap, GatePoseEstimator, GateTargetTracker, select_guidance_gate
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.odometry import VehicleState


MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
LOOP_HZ = 30.0
HEARTBEAT_TIMEOUT_S = 120.0
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


def main() -> int:
    from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

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

    vehicle_state = VehicleState()
    vehicle_state_initialized = False
    mavlink_client = MavlinkClient(MAVLINK_ENDPOINT)
    vision = VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=run_dir / "vision_frames")
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
    track_gate_signature: tuple | None = None

    def seed_track_gate_map_if_available(reason: str) -> bool:
        nonlocal track_gate_signature
        if not mavlink_client.track_gates:
            return False
        latest_track_signature = tuple(
            (
                gate.gate_id,
                tuple(round(float(value), 4) for value in gate.position_local_ned_m),
                tuple(round(float(value), 4) for value in gate.quaternion),
            )
            for gate in mavlink_client.track_gates
        )
        if latest_track_signature == track_gate_signature:
            return False
        gate_map.clear()
        mavlink_client.populate_gate_map(gate_map)
        gate_target_tracker.clear()
        track_gate_signature = latest_track_signature
        logger.log_event(
            "track_gate_map_seeded",
            reason=reason,
            gate_count=len(mavlink_client.track_gates),
            gate_ids=[gate.gate_id for gate in mavlink_client.track_gates],
        )
        return True

    def active_track_gate_record():
        race_status = mavlink_client.race_status
        if race_status is None or not gate_map.has_authoritative_gates():
            return None
        active_gate_index = int(race_status.active_gate_index)
        for track_gate in mavlink_client.track_gates:
            if int(track_gate.gate_id) == active_gate_index:
                gate = gate_map.get_gate(str(track_gate.gate_id))
                return gate if gate is not None and gate.position_relative_ned_m is not None else None
        if 0 <= active_gate_index < len(mavlink_client.track_gates):
            gate = gate_map.get_gate(str(mavlink_client.track_gates[active_gate_index].gate_id))
            return gate if gate is not None and gate.position_relative_ned_m is not None else None
        return None

    try:
        vision.start_listener()
        logger.log_event("vision_started", receiver=vision.snapshot(), perception=vision_perception.snapshot())

        mavlink_client.connect(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
        mavlink_client.start_heartbeat()
        mavlink_client.subscribe_telemetry()
        logger.log_event("mavlink_connected", bridge=mavlink_client.snapshot())
        if RESET_ON_START:
            mavlink_client.reset_simulator_and_wait_ready(
                reset_wait_s=RESET_WAIT_S,
                ready_timeout_s=RESET_READY_TIMEOUT_S,
                stable_s=RESET_STABLE_S,
                stable_max_speed_mps=RESET_STABLE_MAX_SPEED_MPS,
                post_reset_delay_s=POST_RESET_DELAY_S,
                log_event=logger.log_event,
            )
            gate_target_tracker.clear()
            gate_map.clear()
            track_gate_signature = None
            mavlink_client.wait_for_track_gates(timeout_s=TRACK_GATE_WAIT_S)
            seed_track_gate_map_if_available("post_reset")
        else:
            mavlink_client.wait_for_local_ned_telemetry(timeout_s=HEARTBEAT_TIMEOUT_S)
            mavlink_client.wait_for_track_gates(timeout_s=TRACK_GATE_WAIT_S)
            seed_track_gate_map_if_available("startup")
        if ARM_ON_START:
            mavlink_client.arm_and_wait(timeout_s=ARM_TIMEOUT_S)
            logger.log_event("armed", bridge=mavlink_client.snapshot())
        mavlink_client.run_prelevel(
            guidance_controller=body_rate_guidance,
            duration_s=PRELEVEL_S,
            thrust=PRELEVEL_THRUST,
            hz=LOOP_HZ,
            log_event=logger.log_event,
        )

        while RUN_S is None or time.perf_counter() - started_s < RUN_S:
            loop_started_s = time.perf_counter()
            scheduled_s = next_cycle_s
            telemetry = mavlink_client.get_latest_telemetry()
            frame = vision.get_next_frame()
            seed_track_gate_map_if_available("runtime_update")

            if (
                not vehicle_state_initialized
                and telemetry is not None
                and mavlink_client.latest_odometry is not None
                and mavlink_client.latest_odometry.position_local_ned_m is not None
            ):
                vehicle_state.reset()
                vehicle_state.ingest_mavlink_odometry(telemetry.odometry)
                vehicle_state_initialized = True
                logger.log_event("vehicle_state_initialized", odometry=vehicle_state.odometry)

            if vehicle_state_initialized and mavlink_client.latest_imu is not None:
                vehicle_state.update_from_imu(latest_imu=mavlink_client.latest_imu)

            if cycle % int(LOOP_HZ) == 0:
                print(
                    f"Vehicle state position - {mavlink_client.latest_odometry.position_local_ned_m if mavlink_client.latest_odometry else "No Telemetry yet"}",
                    flush=True,
                )

            if telemetry is not None:
                logger.log_telemetry(telemetry, cycle=cycle)
            if frame is not None:
                vision.record_frame_cycle(frame.frame_id, cycle)
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
                            selected_gate = active_track_gate_record()
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

            command_result = mavlink_client.stream_gate_body_rate_command(
                gate_target_tracker,
                telemetry,
                body_rate_guidance,
                now_s=loop_started_s,
            )

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
                    **vision.snapshot(),
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
                    f"cycle={cycle} Odometry.position_local_ned={mavlink_client.latest_odometry.position_local_ned_m if mavlink_client.latest_odometry else 'no ODOMETRY'} "
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
        vision.shutdown()
        if getattr(mavlink_client, "connected", False):
            try:
                mavlink_client.send_body_rate_stop(
                    body_rate_guidance,
                    telemetry=mavlink_client.get_latest_telemetry(),
                    source="main_shutdown_stop",
                )
            except Exception as error:  # noqa: BLE001
                logger.log_event("shutdown_stop_failed", error=str(error))
        mavlink_client.shutdown()
        logger.log_event(
            "shutdown",
            bridge=mavlink_client.snapshot(),
            vision=vision.snapshot(),
            perception=vision_perception.snapshot(),
        )
        logger.save_run(log_path)
        print(f"Log saved to {log_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
