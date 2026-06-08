import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from autonomy.planning.mpcc import MPCCPlanner
from core.control.command_mapper import CommandMapper
from core.control.hover import HoverPIDController
from core.logging import Logger
from sensing.perception import GateMap, GatePoseEstimator, VisionGateObservation, VisionObservation
from sensing.telemetry.mavlink_bridge import MavlinkBridge, TelemetrySample
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService
from sensing.vision.vision_stream import VisionStreamReceiver


ENDPOINT = "udpin:127.0.0.1:14550"
CONTROL_HZ = 30.0
RUN_S = 60.0
STARTUP_TIMEOUT_S = 120.0
ARM_TIMEOUT_S = 20.0
VISION_HOST = "0.0.0.0"
VISION_PORT = 5600
PLANNER_GATE_LIMIT = 5
FORWARD_TILT_RAD = 0.12
FORWARD_THRUST_SCALE = 0.79


def telemetry_to_state(telemetry: TelemetrySample) -> np.ndarray:
    state = np.zeros(13, dtype=float)
    state[0:3] = np.asarray(telemetry.position_local_ned_m or (0.0, 0.0, 0.0), dtype=float)
    state[3:6] = np.asarray(telemetry.velocity_local_ned_mps, dtype=float)
    quaternion = np.asarray(telemetry.attitude, dtype=float)
    state[6:10] = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    state[10:13] = np.asarray(telemetry.body_rates_rps, dtype=float)
    return state


def body_to_local_ned_rotation(quaternion: tuple[float, float, float, float] | list[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def quaternion_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=float)


def forward_velocity_reference(telemetry: TelemetrySample) -> np.ndarray:
    forward = body_to_local_ned_rotation(telemetry.attitude)[:, 0]
    forward_level = np.asarray([forward[0], forward[1], 0.0], dtype=float)
    norm = np.linalg.norm(forward_level)
    if norm <= 1e-9:
        return np.zeros(3, dtype=float)
    return forward_level / norm


def level_forward_attitude_target(hover_thrust: float) -> dict[str, Any]:
    """Command gentle forward motion with the known-stable attitude target path."""

    return {
        "quaternion": quaternion_from_roll_pitch(0.0, FORWARD_TILT_RAD).tolist(),
        "thrust": float(np.clip(hover_thrust * FORWARD_THRUST_SCALE, 0.0, 1.0)),
        "mode": "level_forward_attitude_bias",
        "forward_tilt_rad": FORWARD_TILT_RAD,
        "forward_thrust_scale": FORWARD_THRUST_SCALE,
    }


def gate_records_to_mpcc_gates(records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": record.gate_id,
            "pos": [float(value) for value in record.position_local_ned_m],
            "quat": [float(value) for value in record.quaternion],
            "confidence": float(record.confidence),
            "sequence": record.sequence,
        }
        for record in records
    ]


def current_frame_gate_observation(observation: VisionObservation) -> VisionObservation:
    sorted_gates = sorted(
        observation.gates,
        key=lambda gate: float(np.linalg.norm(np.asarray(gate.position_camera_m, dtype=float))),
    )[:PLANNER_GATE_LIMIT]
    gates = tuple(
        VisionGateObservation(
            gate_id=f"gate-current-{index:03d}",
            position_camera_m=gate.position_camera_m,
            position_confidence=gate.position_confidence,
            orientation_camera=gate.orientation_camera,
            orientation_confidence=gate.orientation_confidence,
        )
        for index, gate in enumerate(sorted_gates)
    )
    return VisionObservation(
        frame_id=observation.frame_id,
        sim_time_ns=observation.sim_time_ns,
        gates=gates,
        source=f"{observation.source}_current_frame",
    )


class PlannerWorker:
    def __init__(self, logger: Logger, log_lock: threading.Lock) -> None:
        self.logger = logger
        self.log_lock = log_lock
        self.planner = MPCCPlanner(visualize_result=False)
        self.requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mpcc-planner", daemon=True)
        self.lock = threading.Lock()
        self.latest_planned_path: dict[str, Any] | None = None
        self.latest_error: str | None = None
        self.plan_count = 0

    def start(self) -> None:
        self.thread.start()

    def submit(self, request: dict[str, Any]) -> None:
        if self.requests.full():
            try:
                self.requests.get_nowait()
            except queue.Empty:
                pass
        self.requests.put_nowait(request)
        print(
            f"vision->planning frame={request['frame_id']} gates={len(request['gates'])} queued",
            flush=True,
        )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "plan_count": self.plan_count,
                "latest_error": self.latest_error,
                "latest_planned_path": self.latest_planned_path,
                "queued_requests": self.requests.qsize(),
            }

    def shutdown(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            frame_id = request["frame_id"]
            gates = request["gates"]
            print(f"MPCC planning frame={frame_id} gates={len(gates)} started", flush=True)
            started = time.monotonic()
            try:
                _q_des, _thrust, info = self.planner.plan(request["state"], gates)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                planned_path = {
                    "schema_version": "planned_path_local_ned_v1",
                    "frame_id": int(frame_id),
                    "sim_time_ns": int(request["sim_time_ns"]),
                    "elapsed_ms": elapsed_ms,
                    "status": info.get("status"),
                    "points_local_ned_m": info.get("planned_path_local_ned_m", []),
                    "reference_path_local_ned_m": info.get("reference_path_local_ned_m", []),
                    "gates": gates,
                    "gate_misses_m": info.get("gate_misses_m", []),
                }
                with self.lock:
                    self.plan_count += 1
                    self.latest_error = None
                    self.latest_planned_path = planned_path
                with self.log_lock:
                    self.logger.log_planned_path(planned_path, frame_id=frame_id, sim_time_ns=request["sim_time_ns"])
                    self.logger.log_mpcc_solution(info)
                print(
                    f"MPCC planning frame={frame_id} status={planned_path['status']} "
                    f"points={len(planned_path['points_local_ned_m'])} elapsed_ms={elapsed_ms:.1f}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                with self.lock:
                    self.latest_error = str(error)
                with self.log_lock:
                    self.logger.log_event("mpcc_planning_failed", frame_id=frame_id, error=str(error), gates=gates)
                print(f"MPCC planning frame={frame_id} failed: {error}", flush=True)


class VisionWorker:
    def __init__(
        self,
        *,
        receiver: VisionStreamReceiver,
        service: VisionPerceptionService,
        bridge: MavlinkBridge,
        gate_map: GateMap,
        planner: PlannerWorker,
        logger: Logger,
        log_lock: threading.Lock,
    ) -> None:
        self.receiver = receiver
        self.service = service
        self.bridge = bridge
        self.gate_map = gate_map
        self.pose_estimator = GatePoseEstimator()
        self.planner = planner
        self.logger = logger
        self.log_lock = log_lock
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="vision-planning-ingress", daemon=True)
        self.gate_map_lock = threading.Lock()
        self.frames_processed = 0
        self.latest_error: str | None = None
        self.latest_frame_gates: list[dict[str, Any]] = []

    def start(self) -> None:
        self.receiver.start_listener()
        self.thread.start()

    def snapshot(self) -> dict[str, Any]:
        with self.gate_map_lock:
            gates = list(self.latest_frame_gates)
        return {
            "frames_processed": self.frames_processed,
            "latest_error": self.latest_error,
            "receiver": self.receiver.snapshot(),
            "gates": gates,
        }

    def shutdown(self) -> None:
        self.stop_event.set()
        self.receiver.shutdown()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            frame = self.receiver.get_next_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is None or telemetry.position_local_ned_m is None:
                with self.log_lock:
                    self.logger.log_vision_frame(
                        {"frame_id": frame.frame_id, "sim_time_ns": frame.sim_time_ns, "saved_path": frame.saved_path},
                        status="dropped_no_telemetry",
                    )
                continue
            try:
                observation = self.service.process_frame(
                    frame_id=frame.frame_id,
                    sim_time_ns=frame.sim_time_ns,
                    jpeg_bytes=frame.jpeg_bytes,
                )
                observation_for_mapping = current_frame_gate_observation(observation)
                with self.gate_map_lock:
                    mapped = self.pose_estimator.update_gate_map_from_observation(
                        observation_for_mapping,
                        telemetry=telemetry,
                        gate_map=self.gate_map,
                    )
                    gates = gate_records_to_mpcc_gates(mapped)
                    self.latest_frame_gates = gates
                self.frames_processed += 1
                self.latest_error = None
                frame_log = {
                    "schema_version": "vision_frame_to_planning_v1",
                    "frame_id": frame.frame_id,
                    "sim_time_ns": frame.sim_time_ns,
                    "saved_path": frame.saved_path,
                    "gate_count": len(observation.gates),
                    "observation": observation.to_controller_payload(output_dir="memory"),
                    "mapping_observation": observation_for_mapping.to_controller_payload(output_dir="memory"),
                    "mapped_gates": gate_records_to_mpcc_gates(mapped),
                }
                with self.log_lock:
                    self.logger.log_vision_frame(frame_log, status="processed")
                print(
                    f"vision frame={frame.frame_id} gates={len(observation.gates)} "
                    f"mapped={len(mapped)} passed_to_planning={len(gates)}",
                    flush=True,
                )
                if gates:
                    self.planner.submit(
                        {
                            "frame_id": frame.frame_id,
                            "sim_time_ns": frame.sim_time_ns,
                            "state": telemetry_to_state(telemetry),
                            "gates": gates,
                        }
                    )
            except Exception as error:  # noqa: BLE001
                self.latest_error = str(error)
                with self.log_lock:
                    self.logger.log_vision_frame(
                        {"frame_id": frame.frame_id, "sim_time_ns": frame.sim_time_ns, "saved_path": frame.saved_path},
                        status="failed",
                        error=str(error),
                    )
                print(f"vision frame={frame.frame_id} failed: {error}", flush=True)


def main():
    bridge = MavlinkBridge(ENDPOINT)
    mapper = CommandMapper()
    hover = HoverPIDController(dt_s=1.0 / CONTROL_HZ)
    run_dir = Logger.timestamped_dir(Path(__file__).resolve().parents[1] / "logs" / "runs")
    log_path = run_dir / "run.json"
    logger = Logger(
        {
            "scenario": "hover_vision_mpcc_test",
            "endpoint": ENDPOINT,
            "control_hz": CONTROL_HZ,
            "run_s": RUN_S,
            "forward_tilt_rad": FORWARD_TILT_RAD,
            "forward_thrust_scale": FORWARD_THRUST_SCALE,
            "vision": {
                "schema_version": "vision_frame_to_planning_v1",
                "host": VISION_HOST,
                "port": VISION_PORT,
                "frame_output_dir": str(run_dir / "frames"),
            },
            "planning": {
                "schema_version": "planned_path_local_ned_v1",
                "planner": "MPCCPlanner",
                "gate_limit": PLANNER_GATE_LIMIT,
                "threaded": True,
            },
        }
    )
    log_lock = threading.Lock()
    planner_worker = PlannerWorker(logger, log_lock)
    vision_worker = VisionWorker(
        receiver=VisionStreamReceiver(host=VISION_HOST, port=VISION_PORT, output_dir=run_dir / "frames"),
        service=VisionPerceptionService(VisionPerceptionConfig(run_landmarker=False, top_k=PLANNER_GATE_LIMIT)),
        bridge=bridge,
        gate_map=GateMap(),
        planner=planner_worker,
        logger=logger,
        log_lock=log_lock,
    )
    cycle = 0
    start = time.monotonic()
    next_print = start

    try:
        planner_worker.start()
        vision_worker.start()
        logger.log_event("vision_started", receiver=vision_worker.receiver.snapshot())

        bridge.connect(heartbeat_timeout_s=STARTUP_TIMEOUT_S)
        bridge.start_heartbeat()
        bridge.subscribe_telemetry()
        logger.log_event("connected", bridge=bridge.snapshot())

        while time.monotonic() - start < STARTUP_TIMEOUT_S:
            telemetry = bridge.get_latest_telemetry()
            if telemetry and telemetry.position_local_ned_m is not None:
                break
            time.sleep(1.0 / CONTROL_HZ)
        else:
            raise TimeoutError("No ODOMETRY position received")

        bridge.arm()
        deadline = time.monotonic() + ARM_TIMEOUT_S
        while not bridge.armed and time.monotonic() < deadline:
            time.sleep(0.02)
        if not bridge.armed:
            raise TimeoutError("Simulator did not confirm armed state")
        logger.log_event("hover_started", bridge=bridge.snapshot(), telemetry=telemetry)

        end = time.monotonic() + RUN_S
        while time.monotonic() < end:
            telemetry = bridge.get_latest_telemetry()
            if telemetry is None or telemetry.position_local_ned_m is None:
                raise RuntimeError("ODOMETRY position is required during hover")
            hover_quaternion, hover_thrust = hover.update(
                telemetry.acceleration_local_ned_mps2 or (0.0, 0.0, 0.0),
                telemetry.attitude,
            )
            target = level_forward_attitude_target(hover_thrust)
            forward_reference = forward_velocity_reference(telemetry)
            bridge.send_attitude_target(target)
            now = time.monotonic()
            vision_snapshot = vision_worker.snapshot()
            planning_snapshot = planner_worker.snapshot()
            with log_lock:
                logger.log_cycle(
                    cycle=cycle,
                    sim_time_ns=telemetry.sim_time_ns,
                    wall_elapsed_ms=(now - start) * 1000.0,
                    telemetry=telemetry,
                    bridge=bridge.snapshot(),
                    command={
                        "set_attitude_target": target,
                        "forward_reference_local_ned_unit": forward_reference.tolist(),
                        "hover_attitude_reference": mapper.to_attitude_target(hover_quaternion, hover_thrust),
                    },
                    vision=vision_snapshot,
                    planning=planning_snapshot,
                    planned_path=planning_snapshot.get("latest_planned_path"),
                )
            if now >= next_print:
                latest_path = planning_snapshot.get("latest_planned_path") or {}
                print(
                    f"cycle={cycle:04d} "
                    f"pos={np.asarray(telemetry.position_local_ned_m).round(3).tolist()} "
                    f"vel={np.asarray(telemetry.velocity_local_ned_mps).round(3).tolist()} "
                    f"vision_frames={vision_snapshot['frames_processed']} "
                    f"planning_count={planning_snapshot['plan_count']} "
                    f"planned_points={len(latest_path.get('points_local_ned_m', []))} "
                    f"forward_ref={np.asarray(forward_reference).round(3).tolist()} "
                    f"cmd_q={np.asarray(target['quaternion']).round(3).tolist()} thrust={target['thrust']:.3f} "
                    f"hover_q={np.asarray(hover_quaternion).round(3).tolist()} hover_thrust={hover_thrust:.3f}",
                    flush=True,
                )
                next_print = now + 0.1
            cycle += 1
            time.sleep(1.0 / CONTROL_HZ)
    finally:
        vision_worker.shutdown()
        planner_worker.shutdown()
        if bridge.connected and bridge.is_live:
            try:
                bridge.disarm()
            except Exception as error:
                logger.log_event("disarm_failed", error=str(error))
        bridge.shutdown()
        logger.log_event("shutdown", bridge=bridge.snapshot(), vision=vision_worker.snapshot(), planning=planner_worker.snapshot())
        logger.save_run(log_path)
        print(f"Hover/vision/MPCC log saved to {log_path}", flush=True)


if __name__ == "__main__":
    main()
