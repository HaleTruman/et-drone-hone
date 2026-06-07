"""Live simulator smoke test: idle until startup, then hover above start."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from core.control.command_mapper import CommandMapper
from core.control.hover import HoverPIDController
from core.logging import Logger
from core.modes.system_mode import SystemModeManager
from sensing.telemetry.mavlink_bridge import MavlinkBridge, TelemetrySample
from sensing.vision.vision_stream import VisionStreamReceiver


@dataclass(frozen=True)
class LiveHoverTestConfig:
    endpoint: str = "udpin:127.0.0.1:14550"
    vision_host: str = "0.0.0.0"
    vision_port: int = 5600
    control_hz: float = 30.0
    hover_s: float = 8.0
    hover_altitude_m: float = 1.0
    startup_timeout_s: float = 10.0
    arm_timeout_s: float = 5.0
    heartbeat_stale_s: float = 2.0
    disarm_on_exit: bool = True


class LiveHoverTestRunner:
    """Exercise the simulator link with a minimal idle-to-hover sequence."""

    def __init__(
        self,
        config: LiveHoverTestConfig,
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
        self.system_mode = SystemModeManager()
        self.command_mapper = CommandMapper()
        params = self._load_quadrotor_params()
        self.hover_controller = HoverPIDController(
            mass_kg=params["m"],
            gravity_mps2=params["g"],
            thrust_coefficient=params["kf"],
            dt_s=1.0 / config.control_hz,
        )
        self.logger = Logger(metadata={"scenario": "live_idle_then_hover_test", "config": asdict(config)})
        self._cycle = 0

    def run(self) -> Path:
        run_log = self.run_dir / "run.json"
        try:
            self.bridge.connect(heartbeat_timeout_s=self.config.startup_timeout_s)
            self.bridge.start_heartbeat()
            self.bridge.subscribe_telemetry()
            self.vision_stream.start_listener()
            starting_telemetry = self._wait_for_simulator_start()
            self._set_hover_target(starting_telemetry)
            self._log_event(
                "simulator_started",
                starting_position_local_ned_m=starting_telemetry.position_local_ned_m,
                target_position_local_ned_m=self.hover_controller.target_position_local_ned_m,
            )

            self.bridge.arm()
            self._wait_for_armed()
            self.system_mode.update_mode("arm")
            self._log_event("hover_started")
            self._hover()
        except Exception as error:
            self.system_mode.handle_fault(str(error))
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

    def _wait_for_simulator_start(self) -> TelemetrySample:
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            self._perform_link_checks(allow_missing_heartbeat=False)
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None:
                if telemetry.position_local_ned_m is None:
                    raise RuntimeError("ODOMETRY position is required before hover can start")
                return telemetry
            self._log_cycle(command=None)
            self.sleep(1.0 / self.config.control_hz)
        raise TimeoutError("No ODOMETRY telemetry received before startup timeout")

    def _wait_for_armed(self) -> None:
        deadline = time.monotonic() + self.config.arm_timeout_s
        while time.monotonic() < deadline:
            if self.bridge.armed:
                return
            self.sleep(0.02)
        raise TimeoutError("Simulator did not confirm armed state with a heartbeat")

    def _set_hover_target(self, telemetry: TelemetrySample) -> None:
        assert telemetry.position_local_ned_m is not None
        start = np.asarray(telemetry.position_local_ned_m, dtype=float)
        self.hover_controller.target_position_local_ned_m = start + np.array(
            [0.0, 0.0, -self.config.hover_altitude_m],
            dtype=float,
        )

    def _hover(self) -> None:
        deadline = time.monotonic() + self.config.hover_s
        while time.monotonic() < deadline:
            self._perform_link_checks()
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is None or telemetry.position_local_ned_m is None:
                raise RuntimeError("ODOMETRY position is required during hover")
            acceleration = telemetry.acceleration_local_ned_mps2 or (0.0, 0.0, 0.0)
            quaternion, thrust = self.hover_controller.update(
                acceleration,
                np.asarray(telemetry.position_local_ned_m, dtype=float),
                telemetry.velocity_local_ned_mps,
            )
            target = self.command_mapper.to_attitude_target(quaternion, thrust)
            self.bridge.send_attitude_target(target)
            self._log_cycle(command={"set_attitude_target": target})
            self.sleep(1.0 / self.config.control_hz)

    def _perform_link_checks(self, *, allow_missing_heartbeat: bool = False) -> None:
        if not self.bridge.connected:
            raise RuntimeError("MAVLink connection is not active")
        if self.bridge.last_heartbeat_monotonic_s is None:
            if allow_missing_heartbeat:
                return
            raise RuntimeError("No MAVLink heartbeat has been observed")
        if time.monotonic() - self.bridge.last_heartbeat_monotonic_s > self.config.heartbeat_stale_s:
            raise RuntimeError("MAVLink heartbeat is stale")
        if self.bridge.collisions:
            raise RuntimeError(f"Collision reported: {self.bridge.collisions[-1]}")

    def _log_cycle(self, command: dict[str, Any] | None) -> None:
        self.logger.log_cycle(
            cycle=self._cycle,
            system_mode=self.system_mode.system_mode,
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
            system_mode=self.system_mode.system_mode,
            bridge=self.bridge.snapshot(),
            vision=self.vision_stream.snapshot(),
            **data,
        )

    @staticmethod
    def _load_quadrotor_params() -> dict[str, Any]:
        params_path = Path(__file__).resolve().parent / "core" / "quadrotor" / "params.yaml"
        return yaml.safe_load(params_path.read_text(encoding="utf-8"))

    @staticmethod
    def _create_run_dir(data_dir: str | Path | None) -> Path:
        root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "logs" / "runs"
        run_dir = Logger.timestamped_dir(root)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:127.0.0.1:14550")
    parser.add_argument("--vision-host", default="0.0.0.0")
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--hover-s", type=float, default=8.0)
    parser.add_argument("--hover-altitude-m", type=float, default=1.0)
    args = parser.parse_args()
    config = LiveHoverTestConfig(
        endpoint=args.endpoint,
        vision_host=args.vision_host,
        vision_port=args.vision_port,
        control_hz=args.control_hz,
        hover_s=args.hover_s,
        hover_altitude_m=args.hover_altitude_m,
    )
    path = LiveHoverTestRunner(config).run()
    print(f"Live hover test capture saved to {path}", flush=True)


if __name__ == "__main__":
    main()
