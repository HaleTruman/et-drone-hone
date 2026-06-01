"""Run the first live simulator smoke flight: idle, arm, then fly forward."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .command_mapper import CommandMapper
from .flight_state import FlightMode, FlightStateMachine
from .logger import Logger
from .mavlink_bridge import MavlinkBridge, TelemetrySample
from .vision_stream import VisionStreamReceiver


@dataclass(frozen=True)
class LiveForwardFlightConfig:
    endpoint: str = "udpin:127.0.0.1:14550"
    vision_host: str = "0.0.0.0"
    vision_port: int = 5600
    idle_s: float = 2.0
    racing_s: float = 8.0
    control_hz: float = 30.0
    forward_speed_mps: float = 2.0
    startup_timeout_s: float = 10.0
    arm_timeout_s: float = 5.0
    heartbeat_stale_s: float = 2.0
    disarm_on_exit: bool = True


class LiveForwardFlightRunner:
    """Exercise the real simulator link before introducing the MPCC planner."""

    def __init__(
        self,
        config: LiveForwardFlightConfig,
        *,
        data_dir: str | Path | None = None,
        bridge: MavlinkBridge | None = None,
        vision_stream: VisionStreamReceiver | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.run_dir = self._create_run_dir(data_dir)
        self.bridge = bridge or MavlinkBridge(config.endpoint)
        self.vision_stream = vision_stream or VisionStreamReceiver(
            config.vision_host,
            config.vision_port,
            output_dir=self.run_dir / "frames",
        )
        self.sleep = sleep
        self.flight_state = FlightStateMachine()
        self.command_mapper = CommandMapper()
        self.logger = Logger(metadata={"scenario": "live_idle_arm_forward_flight", "config": asdict(config)})
        self._cycle = 0

    def run(self) -> Path:
        run_log = self.run_dir / "run.json"
        try:
            self.bridge.connect(heartbeat_timeout_s=self.config.startup_timeout_s)
            self.bridge.start_heartbeat()
            self.bridge.subscribe_telemetry()
            self.vision_stream.start_listener()
            self._wait_for_telemetry()
            self._log_event("connected")

            self._hold_idle()
            self.bridge.arm()
            self._wait_for_armed()
            self.flight_state.update_state("arm")
            self._log_event("armed")

            self._perform_preflight_checks()
            self.flight_state.update_state("start")
            self._log_event("racing_started")
            self._fly_forward()
        except Exception as error:
            self.flight_state.handle_fault(str(error))
            self._log_event("fault", error=str(error))
            raise
        finally:
            if self.config.disarm_on_exit and self.bridge.connected and self.bridge.is_live:
                try:
                    self.bridge.disarm()
                except Exception as error:
                    self._log_event("disarm_failed", error=str(error))
            self.vision_stream.shutdown()
            self.bridge.shutdown()
            self._log_event("shutdown")
            self.logger.save_run(run_log)
        return run_log

    def _wait_for_telemetry(self) -> TelemetrySample:
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None:
                return telemetry
            self.sleep(0.02)
        raise TimeoutError("No ODOMETRY telemetry received before startup timeout")

    def _wait_for_armed(self) -> None:
        deadline = time.monotonic() + self.config.arm_timeout_s
        while time.monotonic() < deadline:
            if self.bridge.armed:
                return
            self.sleep(0.02)
        raise TimeoutError("Simulator did not confirm armed state with a heartbeat")

    def _hold_idle(self) -> None:
        deadline = time.monotonic() + self.config.idle_s
        while time.monotonic() < deadline:
            self._perform_link_checks()
            self._log_cycle(command=None)
            self.sleep(1.0 / self.config.control_hz)

    def _fly_forward(self) -> None:
        target = self.command_mapper.to_velocity_target(np.array([self.config.forward_speed_mps, 0.0, 0.0]))
        deadline = time.monotonic() + self.config.racing_s
        while time.monotonic() < deadline:
            self._perform_link_checks()
            self.bridge.send_position_target(target)
            self._log_cycle(command={"set_position_target_local_ned": target})
            self.sleep(1.0 / self.config.control_hz)

    def _perform_preflight_checks(self) -> None:
        self._perform_link_checks()
        telemetry = self.bridge.get_latest_telemetry()
        assert telemetry is not None
        if telemetry.position_local_ned_m is None:
            raise RuntimeError("ODOMETRY position is required for live racing")
        if not math.isclose(np.linalg.norm(telemetry.attitude), 1.0, abs_tol=0.05):
            raise RuntimeError("ODOMETRY attitude quaternion is not normalized")
        if telemetry.reset_count is None:
            raise RuntimeError("ODOMETRY reset counter is missing")

    def _perform_link_checks(self) -> None:
        if not self.bridge.connected:
            raise RuntimeError("MAVLink connection is not active")
        if self.bridge.last_heartbeat_monotonic_s is None:
            raise RuntimeError("No MAVLink heartbeat has been observed")
        if time.monotonic() - self.bridge.last_heartbeat_monotonic_s > self.config.heartbeat_stale_s:
            raise RuntimeError("MAVLink heartbeat is stale")
        if self.bridge.collisions:
            raise RuntimeError(f"Collision reported: {self.bridge.collisions[-1]}")

    def _log_cycle(self, command: dict[str, Any] | None) -> None:
        self.logger.log_cycle(
            cycle=self._cycle,
            flight_mode=self.flight_state.mode,
            telemetry=self.bridge.get_latest_telemetry(),
            race_status=self.bridge.race_status,
            collisions=self.bridge.collisions,
            vision=self.vision_stream.snapshot(),
            command=command,
        )
        self._cycle += 1

    def _log_event(self, event: str, **data: Any) -> None:
        self.logger.log_event(
            event,
            flight_mode=self.flight_state.mode,
            bridge=self.bridge.snapshot(),
            vision=self.vision_stream.snapshot(),
            **data,
        )

    @staticmethod
    def _create_run_dir(data_dir: str | Path | None) -> Path:
        root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data" / "live_runs"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = root / f"run-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:127.0.0.1:14550")
    parser.add_argument("--vision-host", default="0.0.0.0")
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--idle-s", type=float, default=2.0)
    parser.add_argument("--racing-s", type=float, default=8.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--forward-speed-mps", type=float, default=2.0)
    args = parser.parse_args()
    config = LiveForwardFlightConfig(
        endpoint=args.endpoint,
        vision_host=args.vision_host,
        vision_port=args.vision_port,
        idle_s=args.idle_s,
        racing_s=args.racing_s,
        control_hz=args.control_hz,
        forward_speed_mps=args.forward_speed_mps,
    )
    path = LiveForwardFlightRunner(config).run()
    print(f"Live forward-flight capture saved to {path}", flush=True)


if __name__ == "__main__":
    main()
