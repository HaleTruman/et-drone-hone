from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.logging.logging import Logger
from sensing.telemetry import MavlinkBridge
from sensing.vision.vision_stream import VisionFrame, VisionStreamReceiver

from .command_emitter import CommandEmitter
from .command_streamer import CommandStreamer
from .command_transform import CommandFrameTransform
from .config import Q1RuntimeConfig
from .guidance import VelocityPlanner, VelocityPlannerConfig
from .safety import SafetyConfig, SafetyDecision, SafetySupervisor
from .target_filter import TargetFilter
from .target_mapper import LocalVisionTarget, TargetMapper
from .target_tracker import TargetTracker
from .vision_targeting import Q1VisionTargeter


@dataclass(frozen=True)
class FrameCommandResult:
    frame_id: int
    sim_time_ns: int
    emitted: bool
    reason: str
    observation_source: str | None = None
    gate_id: str | None = None
    target: LocalVisionTarget | None = None
    command: dict[str, Any] | None = None
    target_updated: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class Q1FrameProcessor:
    """Processes exactly one vision frame into at most one tracked target update."""

    def __init__(
        self,
        *,
        targeter: Q1VisionTargeter,
        mapper: TargetMapper,
        tracker: TargetTracker,
        target_filter: TargetFilter | None = None,
        command_on_missing_target: bool = False,
    ):
        self.targeter = targeter
        self.mapper = mapper
        self.tracker = tracker
        self.target_filter = target_filter
        self.command_on_missing_target = bool(command_on_missing_target)
        self.frames_processed = 0
        self.targets_updated = 0
        self.frames_suppressed = 0

    @classmethod
    def from_config(cls, config: Q1RuntimeConfig, tracker: TargetTracker) -> "Q1FrameProcessor":
        return cls(
            targeter=Q1VisionTargeter.from_config(config),
            mapper=TargetMapper(camera_tilt_deg=config.camera_tilt_deg),
            tracker=tracker,
            target_filter=TargetFilter(
                max_jump_m=config.target_filter_max_jump_m,
                smoothing_alpha=config.target_filter_smoothing_alpha,
            ),
            command_on_missing_target=config.command_on_missing_target,
        )

    def process_frame(self, frame: VisionFrame, telemetry: Any | None, *, now_s: float | None = None) -> FrameCommandResult:
        self.frames_processed += 1
        now = time.monotonic() if now_s is None else float(now_s)
        observation = self.targeter.process_frame(frame)
        gate = self.targeter.select_top_gate(observation)
        if gate is None:
            return self._suppressed(frame, "no_top1_gate_above_confidence", observation.source)
        if telemetry is None:
            return self._suppressed(frame, "missing_telemetry", observation.source, gate.gate_id)
        if getattr(telemetry, "position_local_ned_m", None) is None:
            return self._suppressed(frame, "missing_position_local_ned_m", observation.source, gate.gate_id)
        if getattr(telemetry, "attitude", None) is None:
            return self._suppressed(frame, "missing_attitude", observation.source, gate.gate_id)

        target = self.mapper.map_gate(
            gate,
            frame_id=observation.frame_id,
            sim_time_ns=observation.sim_time_ns,
            telemetry=telemetry,
        )
        if self.target_filter is not None:
            filter_result = self.target_filter.update(target)
            if not filter_result.accepted or filter_result.target is None:
                return self._suppressed(frame, filter_result.reason, observation.source, target.gate_id)
            target = filter_result.target
        self.tracker.update(target, now_s=now)
        self.targets_updated += 1
        return FrameCommandResult(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            emitted=False,
            reason="updated_top1_local_ned_target",
            observation_source=observation.source,
            gate_id=target.gate_id,
            target=target,
            target_updated=True,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "targets_updated": self.targets_updated,
            "frames_suppressed": self.frames_suppressed,
            "command_on_missing_target": self.command_on_missing_target,
            "targeter": self.targeter.snapshot(),
            "tracker": self.tracker.snapshot(),
            "target_filter": None if self.target_filter is None else self.target_filter.snapshot(),
        }

    def _suppressed(
        self,
        frame: VisionFrame,
        reason: str,
        observation_source: str | None = None,
        gate_id: str | None = None,
    ) -> FrameCommandResult:
        self.frames_suppressed += 1
        return FrameCommandResult(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            emitted=False,
            reason=reason,
            observation_source=observation_source,
            gate_id=gate_id,
        )


@dataclass(frozen=True)
class ControlTickResult:
    emitted: bool
    reason: str
    command: dict[str, Any] | None = None
    target: LocalVisionTarget | None = None
    target_age_s: float | None = None
    safety: SafetyDecision | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class Q1Runtime:
    """Live q1runtime shell around the pure frame processor."""

    def __init__(
        self,
        config: Q1RuntimeConfig | None = None,
        *,
        bridge: Any | None = None,
        receiver: Any | None = None,
        processor: Q1FrameProcessor | None = None,
        tracker: TargetTracker | None = None,
        emitter: CommandEmitter | None = None,
        planner: VelocityPlanner | None = None,
        command_transform: CommandFrameTransform | None = None,
        safety: SafetySupervisor | None = None,
        logger: Logger | None = None,
    ):
        self.config = config or Q1RuntimeConfig()
        self.bridge = bridge or MavlinkBridge(self.config.mavlink_endpoint)
        self.receiver = receiver or self._build_receiver()
        self.tracker = tracker or getattr(processor, "tracker", None) or TargetTracker(hold_s=self.config.target_hold_s)
        self.emitter = emitter or CommandEmitter(
            self.bridge,
            command_mode=self.config.command_mode,
            max_approach_speed_mps=self.config.max_approach_speed_mps,
            max_vertical_speed_mps=self.config.max_vertical_speed_mps,
            approach_gain_hz=self.config.approach_gain_hz,
            arrival_radius_m=self.config.arrival_radius_m,
            dry_run=self.config.dry_run,
        )
        self.planner = planner or VelocityPlanner(
            VelocityPlannerConfig(
                max_speed_mps=self.config.max_approach_speed_mps,
                max_vertical_speed_mps=self.config.max_vertical_speed_mps,
                approach_gain_hz=self.config.approach_gain_hz,
                arrival_radius_m=self.config.arrival_radius_m,
                vertical_mode=self.config.vertical_mode,
                max_accel_mps2=self.config.max_accel_mps2,
            )
        )
        self.command_transform = command_transform or self._build_command_transform()
        self.safety = safety or SafetySupervisor(
            SafetyConfig(
                enabled=self.config.safety_enabled,
                max_horizontal_radius_m=self.config.safety_max_horizontal_radius_m,
                min_z_ned_m=self.config.safety_min_z_ned_m,
                max_z_ned_m=self.config.safety_max_z_ned_m,
                max_velocity_mps=self.config.safety_max_velocity_mps,
                telemetry_timeout_s=self.config.safety_telemetry_timeout_s,
            )
        )
        self.processor = processor or Q1FrameProcessor.from_config(self.config, self.tracker)
        self.logger = logger
        self.cycle = 0
        self.command_cycle = 0
        self.commands_emitted = 0
        self.command_suppressions = 0
        self.safety_stop_count = 0
        self.command_streamer = CommandStreamer(
            lambda now_s: self.tick_command(now_s=now_s),
            hz=self.config.command_hz,
            on_result=self._log_command_result,
        )

    def process_next_frame(self, *, now_s: float | None = None) -> FrameCommandResult | None:
        frame = self.receiver.get_next_frame()
        if frame is None:
            return None
        telemetry = self.bridge.get_latest_telemetry()
        result = self.processor.process_frame(frame, telemetry, now_s=now_s)
        self._log_result(result, telemetry)
        self.cycle += 1
        return result

    def tick_command(self, *, now_s: float | None = None) -> ControlTickResult:
        now = time.monotonic() if now_s is None else float(now_s)
        telemetry = self.bridge.get_latest_telemetry()
        safety_decision = self.safety.evaluate(telemetry, now_s=now)
        if safety_decision.reason == "simulator_reset_detected":
            self.tracker.clear()
            self.planner.reset()
            if self.processor.target_filter is not None:
                self.processor.target_filter.reset()
        if safety_decision.should_stop:
            emitted = self.emitter.emit_stop()
            self.commands_emitted += 1
            self.command_cycle += 1
            self.safety_stop_count += 1
            if safety_decision.action == "stop_and_disarm":
                self._disarm_quietly()
            return ControlTickResult(
                emitted=True,
                reason=f"safety_{safety_decision.reason}",
                command=emitted.payload,
                safety=safety_decision,
            )
        tracked = self.tracker.latest(now_s=now)
        if telemetry is None:
            return self._command_suppressed("missing_telemetry_for_command")
        if tracked is None:
            if self.config.stream_stop_when_target_lost:
                emitted = self.emitter.emit_stop()
                self.commands_emitted += 1
                self.command_cycle += 1
                return ControlTickResult(emitted=True, reason="emitted_stop_no_active_target", command=emitted.payload)
            return self._command_suppressed("no_active_target")

        target_age = self.tracker.age_s(now_s=now)
        desired_velocity = self.planner.desired_velocity(tracked.target, telemetry, now_s=now)
        command_velocity = self.command_transform.apply(desired_velocity)
        emitted = self.emitter.emit_velocity(
            command_velocity,
            target=tracked.target,
            source="q1runtime_guided_velocity",
            extra={
                "desired_velocity_local_ned_mps": [float(value) for value in desired_velocity],
                "command_transform": self.command_transform.to_payload(),
                "vertical_mode": self.config.vertical_mode,
            },
        )
        self.commands_emitted += 1
        self.command_cycle += 1
        return ControlTickResult(
            emitted=True,
            reason="streamed_velocity_target_local_ned",
            command=emitted.payload,
            target=emitted.target,
            target_age_s=target_age,
            safety=safety_decision,
        )

    def run(self) -> None:
        run_dir = Logger.timestamped_dir(self.config.log_dir)
        self.logger = self.logger or Logger(self._metadata(run_dir))
        start = time.monotonic()
        next_status = start
        run_deadline = None if self.config.run_s is None else start + float(self.config.run_s)

        try:
            self.receiver.start_listener()
            self.logger.log_event("vision_started", receiver=self.receiver.snapshot())
            self.bridge.connect(heartbeat_timeout_s=self.config.heartbeat_timeout_s)
            self.bridge.start_heartbeat()
            self.bridge.subscribe_telemetry()
            self.logger.log_event("connected", bridge=self.bridge.snapshot())
            self._wait_for_telemetry()
            if self.config.arm_on_start:
                self._arm_and_wait()
            self.logger.log_event("q1runtime_started", processor=self.processor.snapshot())
            self.command_streamer.start()

            while run_deadline is None or time.monotonic() < run_deadline:
                now = time.monotonic()
                result = self.process_next_frame(now_s=now)
                if result is None:
                    time.sleep(self.config.idle_sleep_s)
                if now >= next_status:
                    self._print_status()
                    next_status = now + self.config.status_interval_s
        finally:
            self.command_streamer.stop()
            self._shutdown_controls()
            if self.logger is not None:
                self.logger.log_event(
                    "q1runtime_stopped",
                    processor=self.processor.snapshot(),
                    bridge=self.bridge.snapshot(),
                    safety=self.safety.snapshot(),
                    planner=self.planner.snapshot(),
                    command_transform=self.command_transform.to_payload(),
                )
                self.logger.save_run(run_dir / "run.json")
            if hasattr(self.receiver, "shutdown"):
                self.receiver.shutdown()
            if hasattr(self.bridge, "shutdown"):
                self.bridge.shutdown()

    def _build_receiver(self) -> VisionStreamReceiver:
        output_dir = self.config.frame_output_dir if self.config.save_raw_frames else None
        return VisionStreamReceiver(
            host=self.config.vision_host,
            port=self.config.vision_port,
            output_dir=output_dir,
            max_buffered_frames=self.config.vision_max_buffered_frames,
        )

    def _build_command_transform(self) -> CommandFrameTransform:
        transform = (
            CommandFrameTransform.load(self.config.command_transform_path)
            if self.config.command_transform_path is not None
            else CommandFrameTransform.identity()
        )
        manual = CommandFrameTransform.from_options(
            invert_x=self.config.command_invert_x,
            invert_y=self.config.command_invert_y,
            invert_z=self.config.command_invert_z,
            swap_xy=self.config.command_swap_xy,
            name="manual_cli",
        )
        if manual.to_payload()["matrix"] == CommandFrameTransform.identity().to_payload()["matrix"]:
            return transform
        return CommandFrameTransform.from_matrix(
            np.asarray(manual.matrix, dtype=float) @ np.asarray(transform.matrix, dtype=float),
            name=f"{manual.name}_after_{transform.name}",
        )

    def _wait_for_telemetry(self) -> None:
        deadline = time.monotonic() + self.config.wait_for_telemetry_timeout_s
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None and getattr(telemetry, "position_local_ned_m", None) is not None:
                return
            time.sleep(self.config.idle_sleep_s)
        raise TimeoutError("No local-NED telemetry received before q1runtime startup timeout.")

    def _arm_and_wait(self) -> None:
        self.bridge.arm()
        deadline = time.monotonic() + self.config.arm_timeout_s
        while time.monotonic() < deadline:
            if getattr(self.bridge, "armed", False):
                return
            time.sleep(0.02)
        raise TimeoutError("Simulator did not confirm armed state before q1runtime arm timeout.")

    def _log_result(self, result: FrameCommandResult, telemetry: Any | None) -> None:
        if self.logger is None:
            return
        self.logger.log_cycle(
            cycle=self.cycle,
            sim_time_ns=result.sim_time_ns,
            telemetry=telemetry,
            q1runtime=result.to_log_dict(),
            processor=self.processor.snapshot(),
            tracker=self.tracker.snapshot(),
        )

    def _log_command_result(self, result: ControlTickResult) -> None:
        if self.logger is None:
            return
        log_every = max(1, round(float(self.config.command_hz) / max(1e-6, float(self.config.command_log_hz))))
        if result.emitted and self.command_cycle % log_every != 0:
            return
        self.logger.log_event(
            "command_tick",
            command_cycle=self.command_cycle,
            q1runtime_command=result.to_log_dict(),
            tracker=self.tracker.snapshot(now_s=time.monotonic()),
            emitter=self.emitter.snapshot(),
            planner=self.planner.snapshot(),
            safety=self.safety.snapshot(),
        )

    def _command_suppressed(self, reason: str) -> ControlTickResult:
        self.command_suppressions += 1
        return ControlTickResult(emitted=False, reason=reason)

    def _metadata(self, run_dir: Path) -> dict[str, Any]:
        config = asdict(self.config)
        for key in ("log_dir", "frame_output_dir", "command_transform_path"):
            if config.get(key) is not None:
                config[key] = str(config[key])
        return {
            "runtime": "q1runtime",
            "control_method": "SET_POSITION_TARGET_LOCAL_NED",
            "vision_mode": "cnn_regressor_passthrough_top1",
            "command_stream": "fixed_hz_velocity_only",
            "run_dir": str(run_dir),
            "config": config,
            "command_streamer": self.command_streamer.snapshot(),
            "command_transform": self.command_transform.to_payload(),
            "planner": self.planner.snapshot(),
            "safety": self.safety.snapshot(),
        }

    def _shutdown_controls(self) -> None:
        if self.config.stop_on_exit_s > 0.0:
            end = time.monotonic() + float(self.config.stop_on_exit_s)
            interval = 1.0 / max(1e-6, float(self.config.command_hz))
            while time.monotonic() < end:
                self.emitter.emit_stop()
                time.sleep(interval)
        if self.config.disarm_on_exit:
            self._disarm_quietly()

    def _disarm_quietly(self) -> None:
        if self.config.dry_run or not hasattr(self.bridge, "disarm"):
            return
        try:
            self.bridge.disarm()
        except Exception as exc:
            if self.logger is not None:
                self.logger.log_event("disarm_failed", error=str(exc))

    def _print_status(self) -> None:
        telemetry = self.bridge.get_latest_telemetry()
        position = getattr(telemetry, "position_local_ned_m", None)
        print(
            "q1runtime "
            f"cycle={self.cycle} "
            f"pos={None if position is None else [round(float(value), 3) for value in position]} "
            f"frames={self.processor.frames_processed} "
            f"target_updates={self.processor.targets_updated} "
            f"commands={self.commands_emitted} "
            f"frame_suppressed={self.processor.frames_suppressed} "
            f"command_suppressed={self.command_suppressions} "
            f"safety_stops={self.safety_stop_count}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-1 vision passthrough local-NED control.")
    parser.add_argument("--endpoint", default=Q1RuntimeConfig.mavlink_endpoint)
    parser.add_argument("--vision-host", default=Q1RuntimeConfig.vision_host)
    parser.add_argument("--vision-port", type=int, default=Q1RuntimeConfig.vision_port)
    parser.add_argument("--run-s", type=float, default=Q1RuntimeConfig.run_s)
    parser.add_argument("--command-hz", type=float, default=Q1RuntimeConfig.command_hz)
    parser.add_argument("--command-log-hz", type=float, default=Q1RuntimeConfig.command_log_hz)
    parser.add_argument("--command-mode", choices=("velocity", "position_velocity"), default=Q1RuntimeConfig.command_mode)
    parser.add_argument("--device", default=Q1RuntimeConfig.vision_device)
    parser.add_argument("--min-confidence", type=float, default=Q1RuntimeConfig.min_position_confidence)
    parser.add_argument("--max-speed", type=float, default=Q1RuntimeConfig.max_approach_speed_mps)
    parser.add_argument("--max-vertical-speed", type=float, default=Q1RuntimeConfig.max_vertical_speed_mps)
    parser.add_argument("--vertical-mode", choices=("hold", "target"), default=Q1RuntimeConfig.vertical_mode)
    parser.add_argument("--max-accel", type=float, default=Q1RuntimeConfig.max_accel_mps2)
    parser.add_argument("--arrival-radius", type=float, default=Q1RuntimeConfig.arrival_radius_m)
    parser.add_argument("--target-hold-s", type=float, default=Q1RuntimeConfig.target_hold_s)
    parser.add_argument("--target-filter-max-jump", type=float, default=Q1RuntimeConfig.target_filter_max_jump_m)
    parser.add_argument("--target-filter-alpha", type=float, default=Q1RuntimeConfig.target_filter_smoothing_alpha)
    parser.add_argument("--command-transform", type=Path, default=None)
    parser.add_argument("--invert-command-x", action="store_true")
    parser.add_argument("--invert-command-y", action="store_true")
    parser.add_argument("--invert-command-z", action="store_true")
    parser.add_argument("--swap-command-xy", action="store_true")
    parser.add_argument("--disable-safety", action="store_true")
    parser.add_argument("--safety-radius-m", type=float, default=Q1RuntimeConfig.safety_max_horizontal_radius_m)
    parser.add_argument("--safety-min-z", type=float, default=Q1RuntimeConfig.safety_min_z_ned_m)
    parser.add_argument("--safety-max-z", type=float, default=Q1RuntimeConfig.safety_max_z_ned_m)
    parser.add_argument("--safety-max-velocity", type=float, default=Q1RuntimeConfig.safety_max_velocity_mps)
    parser.add_argument("--safety-telemetry-timeout-s", type=float, default=Q1RuntimeConfig.safety_telemetry_timeout_s)
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--stop-on-exit-s", type=float, default=Q1RuntimeConfig.stop_on_exit_s)
    parser.add_argument("--log-dir", type=Path, default=Q1RuntimeConfig.log_dir)
    parser.add_argument("--frame-output-dir", type=Path, default=None)
    parser.add_argument("--save-raw-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = Q1RuntimeConfig(
        mavlink_endpoint=args.endpoint,
        vision_host=args.vision_host,
        vision_port=args.vision_port,
        run_s=args.run_s,
        command_hz=args.command_hz,
        command_log_hz=args.command_log_hz,
        command_mode=args.command_mode,
        vision_device=args.device,
        min_position_confidence=args.min_confidence,
        max_approach_speed_mps=args.max_speed,
        max_vertical_speed_mps=args.max_vertical_speed,
        vertical_mode=args.vertical_mode,
        max_accel_mps2=args.max_accel,
        arrival_radius_m=args.arrival_radius,
        target_hold_s=args.target_hold_s,
        target_filter_max_jump_m=args.target_filter_max_jump,
        target_filter_smoothing_alpha=args.target_filter_alpha,
        command_transform_path=args.command_transform,
        command_invert_x=args.invert_command_x,
        command_invert_y=args.invert_command_y,
        command_invert_z=args.invert_command_z,
        command_swap_xy=args.swap_command_xy,
        safety_enabled=not args.disable_safety,
        safety_max_horizontal_radius_m=args.safety_radius_m,
        safety_min_z_ned_m=args.safety_min_z,
        safety_max_z_ned_m=args.safety_max_z,
        safety_max_velocity_mps=args.safety_max_velocity,
        safety_telemetry_timeout_s=args.safety_telemetry_timeout_s,
        disarm_on_exit=not args.no_disarm_on_exit,
        stop_on_exit_s=args.stop_on_exit_s,
        log_dir=args.log_dir,
        frame_output_dir=args.frame_output_dir,
        save_raw_frames=args.save_raw_frames,
        dry_run=args.dry_run,
        arm_on_start=not args.no_arm,
    )
    Q1Runtime(config).run()
