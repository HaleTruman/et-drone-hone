import time
from pathlib import Path

import numpy as np

from core.control.command_mapper import CommandMapper
from core.control.hover import HoverPIDController
from core.logging import Logger
from sensing.telemetry.mavlink_bridge import MavlinkBridge


ENDPOINT = "udpin:127.0.0.1:14550"
CONTROL_HZ = 30.0
RUN_S = 60.0
STARTUP_TIMEOUT_S = 120.0
ARM_TIMEOUT_S = 20.0


def main():
    bridge = MavlinkBridge(ENDPOINT)
    mapper = CommandMapper()
    hover = HoverPIDController(dt_s=1.0 / CONTROL_HZ)
    log_path = Logger.timestamped_dir(Path(__file__).resolve().parents[1] / "logs" / "runs") / "run.json"
    logger = Logger({"scenario": "hover_test", "endpoint": ENDPOINT, "control_hz": CONTROL_HZ, "run_s": RUN_S})
    cycle = 0
    start = time.monotonic()
    next_print = start

    try:
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
            quaternion, thrust = hover.update(telemetry.acceleration_local_ned_mps2 or (0.0, 0.0, 0.0), telemetry.attitude)
            target = mapper.to_attitude_target(quaternion, thrust)
            bridge.send_attitude_target(target)
            now = time.monotonic()
            logger.log_cycle(cycle=cycle, wall_elapsed_ms=(now - start) * 1000.0, telemetry=telemetry, bridge=bridge.snapshot(), command=target)
            if now >= next_print:
                print(
                    f"cycle={cycle:04d} "
                    f"pos={np.asarray(telemetry.position_local_ned_m).round(3).tolist()} "
                    f"vel={np.asarray(telemetry.velocity_local_ned_mps).round(3).tolist()} "
                    f"att={np.asarray(telemetry.attitude).round(3).tolist()} "
                    f"acc={np.asarray(telemetry.acceleration_local_ned_mps2 or (0.0, 0.0, 0.0)).round(3).tolist()} "
                    f"cmd_q={np.asarray(quaternion).round(3).tolist()} thrust={thrust:.3f}",
                    flush=True,
                )
                next_print = now + 0.1
            cycle += 1
            time.sleep(1.0 / CONTROL_HZ)
    finally:
        if bridge.connected and bridge.is_live:
            try:
                bridge.disarm()
            except Exception as error:
                logger.log_event("disarm_failed", error=str(error))
        bridge.shutdown()
        logger.log_event("shutdown", bridge=bridge.snapshot())
        logger.save_run(log_path)
        print(f"Hover log saved to {log_path}", flush=True)


if __name__ == "__main__":
    main()
