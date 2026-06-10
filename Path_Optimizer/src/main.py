"""Minimal live MAVLink and vision stream entry point."""

from pathlib import Path
import time

from core.control.hover import HoverController
from core.logging import Logger
from sensing.perception import GateMap, GatePoseEstimator
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService
from sensing.odometry import VehicleState


MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
LOOP_HZ = 30.0
HEARTBEAT_TIMEOUT_S = 120.0
RUN_S: float | None = None


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
        }
    )

    vehicle_state = VehicleState()
    vehicle_state_initialized = False
    hover_controller = HoverController(dt_s=period_s)
    mavlink_client = MavlinkClient(MAVLINK_ENDPOINT)
    vision = VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=run_dir / "vision_frames")
    vision_perception = VisionPerceptionService(VisionPerceptionConfig(run_landmarker=False))
    gate_pose_estimator = GatePoseEstimator()
    gate_map = GateMap()

    cycle = 0
    started_s = time.perf_counter()
    next_cycle_s = started_s
    hover_target_logged = False

    try:
        vision.start_listener()
        logger.log_event("vision_started", receiver=vision.snapshot(), perception=vision_perception.snapshot())

        mavlink_client.connect(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
        mavlink_client.start_heartbeat()
        mavlink_client.subscribe_telemetry()
        logger.log_event("mavlink_connected", bridge=mavlink_client.snapshot())

        ## MAIN LOOP
        while RUN_S is None or time.perf_counter() - started_s < RUN_S:
            loop_started_s = time.perf_counter()
            scheduled_s = next_cycle_s
            telemetry = mavlink_client.get_latest_telemetry()
            frame = vision.get_next_frame()

            if (
                not vehicle_state_initialized
                and telemetry is not None
                and mavlink_client.latest_odometry is not None
                and mavlink_client.latest_odometry.position_local_ned_m is not None
            ):
                vehicle_state.reset(telemetry.odometry)
                vehicle_state_initialized = True
                logger.log_event("vehicle_state_initialized", odometry=vehicle_state.odometry)

            if vehicle_state_initialized and mavlink_client.latest_imu is not None:
                vehicle_state.update_from_imu(latest_imu=mavlink_client.latest_imu)
            
            if cycle % int(LOOP_HZ) == 0:
                print(
                    f"Vehicle state position - {vehicle_state.position_local_ned_m}\n"
                    f"IMU body accelerations - {mavlink_client.latest_imu.acceleration_body_frd_mps2 if mavlink_client.latest_imu else 'NO TELEMETRY YET'}\n"
                    f"ODO position - {mavlink_client.latest_odometry.position_local_ned_m if mavlink_client.latest_odometry else 'NO ODO YET'}\n",
                    flush=True
                )

            if telemetry is not None:
                logger.log_telemetry(telemetry, cycle=cycle)
            if frame is not None:
                frame_log = {
                    "frame_id": frame.frame_id,
                    "sim_time_ns": frame.sim_time_ns,
                    "saved_path": frame.saved_path,
                    "jpeg_size": len(frame.jpeg_bytes),
                }
                try:
                    observation = vision_perception.process_vision_frame(frame)
                    frame_log["gate_count"] = len(observation.gates)
                    frame_log["observation"] = observation.to_controller_payload(output_dir="memory")
                    if telemetry is not None and telemetry.position_local_ned_m is not None:
                        mapped_gates = gate_pose_estimator.update_gate_map_from_observation(
                            observation,
                            telemetry=telemetry,
                            gate_map=gate_map,
                        )
                        frame_log["mapped_gates"] = [
                            {
                                "id": gate.gate_id,
                                "position_local_ned_m": [float(value) for value in gate.position_local_ned_m],
                                "quaternion": [float(value) for value in gate.quaternion],
                                "confidence": float(gate.confidence),
                                "sequence": gate.sequence,
                            }
                            for gate in mapped_gates
                        ]
                    else:
                        frame_log["mapping_status"] = "skipped_no_telemetry"
                    logger.log_vision_frame(frame_log, cycle=cycle, status="processed")
                except Exception as error:  # noqa: BLE001
                    logger.log_vision_frame(frame_log, cycle=cycle, status="failed", error=str(error))
                    print(f"vision frame={frame.frame_id} failed: {error}", flush=True)

            ## ADD CONTROL CODE

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
                vision={
                    **vision.snapshot(),
                    "perception": vision_perception.snapshot(),
                    "gate_count": len(gate_map.get_next_gates(10_000)),
                },
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
        mavlink_client.shutdown()
        logger.log_event(
            "shutdown",
            bridge=mavlink_client.snapshot(),
            vision=vision.snapshot(),
            perception=vision_perception.snapshot(),
            gate_map=gate_map.get_next_gates(10_000),
        )
        logger.save_run(log_path)
        print(f"Log saved to {log_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
