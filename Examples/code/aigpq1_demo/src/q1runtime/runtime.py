from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from Examples.aigpq1_demo.src.core.logging.logging import Logger
from Examples.aigpq1_demo.src.sensing.telemetry import MavlinkBridge
from Examples.aigpq1_demo.src.sensing.vision.vision_stream import VisionFrame, VisionStreamReceiver

from .body_rate_guidance import BodyRateGuidanceControlConfig, BodyRateGuidanceController
from .command_streamer import CommandStreamer
from .command_transform import CommandFrameTransform
from .config import Q1RuntimeConfig
from .route_state import RouteStateSnapshot, RouteStateTracker
from .safety import SafetyConfig, SafetyDecision, SafetySupervisor
from .target_filter import TargetFilter
from .target_mapper import LocalVisionTarget, TargetMapper
from .target_selector import TargetSelectionConfig, TargetSelectionResult, TargetSelector
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
    vision_candidate_count: int = 0
    confident_candidate_count: int = 0
    selected_candidate_count: int = 0
    selection_reason: str | None = None
    selection: TargetSelectionResult | None = None
    route_state: RouteStateSnapshot | None = None

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
        target_selector: TargetSelector | None = None,
        command_on_missing_target: bool = False,
    ):
        self.targeter = targeter
        self.mapper = mapper
        self.tracker = tracker
        self.target_filter = target_filter
        self.target_selector = target_selector or TargetSelector()
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
            target_selector=TargetSelector(
                TargetSelectionConfig(
                    mode=config.target_selection_mode,
                    min_camera_forward_m=config.target_selection_min_camera_forward_m,
                    min_forward_m=config.target_selection_min_forward_m,
                    velocity_min_mps=config.target_selection_velocity_min_mps,
                    allow_camera_fallback_when_routed=config.target_selection_allow_camera_fallback_when_routed,
                )
            ),
            command_on_missing_target=config.command_on_missing_target,
        )

    def process_frame(
        self,
        frame: VisionFrame,
        telemetry: Any | None,
        *,
        now_s: float | None = None,
        route_state: RouteStateSnapshot | None = None,
    ) -> FrameCommandResult:
        self.frames_processed += 1
        now = time.monotonic() if now_s is None else float(now_s)
        observation = self.targeter.process_frame(frame)
        candidate_gates = self.targeter.candidate_gates(observation)
        if not candidate_gates:
            return self._suppressed(
                frame,
                "no_top1_gate_above_confidence",
                observation.source,
                vision_candidate_count=len(observation.gates),
                route_state=route_state,
            )
        if telemetry is None:
            return self._suppressed(
                frame,
                "missing_telemetry",
                observation.source,
                candidate_gates[0].gate_id,
                vision_candidate_count=len(observation.gates),
                confident_candidate_count=len(candidate_gates),
                route_state=route_state,
            )
        if getattr(telemetry, "position_local_ned_m", None) is None:
            return self._suppressed(
                frame,
                "missing_position_local_ned_m",
                observation.source,
                candidate_gates[0].gate_id,
                vision_candidate_count=len(observation.gates),
                confident_candidate_count=len(candidate_gates),
                route_state=route_state,
            )
        if getattr(telemetry, "attitude", None) is None:
            return self._suppressed(
                frame,
                "missing_attitude",
                observation.source,
                candidate_gates[0].gate_id,
                vision_candidate_count=len(observation.gates),
                confident_candidate_count=len(candidate_gates),
                route_state=route_state,
            )

        targets = [
            self.mapper.map_gate(
                gate,
                frame_id=observation.frame_id,
                sim_time_ns=observation.sim_time_ns,
                telemetry=telemetry,
            )
            for gate in candidate_gates
        ]
        route_velocity = None if route_state is None else route_state.route_velocity_local_ned_mps
        selection = self.target_selector.select(targets, telemetry, route_velocity_local_ned_mps=route_velocity)
        if selection.target is None:
            return self._suppressed(
                frame,
                selection.reason,
                observation.source,
                vision_candidate_count=len(observation.gates),
                confident_candidate_count=len(candidate_gates),
                selected_candidate_count=selection.eligible_count,
                selection=selection,
                route_state=route_state,
        )
        target = selection.target
        if self.target_filter is not None:
            filter_result = self.target_filter.update(target)
            if not filter_result.accepted or filter_result.target is None:
                return self._suppressed(
                    frame,
                    filter_result.reason,
                    observation.source,
                    target.gate_id,
                    vision_candidate_count=len(observation.gates),
                    confident_candidate_count=len(candidate_gates),
                    selected_candidate_count=selection.eligible_count,
                    selection=selection,
                    route_state=route_state,
                )
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
            vision_candidate_count=len(observation.gates),
            confident_candidate_count=len(candidate_gates),
            selected_candidate_count=selection.eligible_count,
            selection_reason=selection.reason,
            selection=selection,
            route_state=route_state,
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
            "target_selector": self.target_selector.snapshot(),
        }

    def _suppressed(
        self,
        frame: VisionFrame,
        reason: str,
        observation_source: str | None = None,
        gate_id: str | None = None,
        vision_candidate_count: int = 0,
        confident_candidate_count: int = 0,
        selected_candidate_count: int = 0,
        selection: TargetSelectionResult | None = None,
        route_state: RouteStateSnapshot | None = None,
    ) -> FrameCommandResult:
        self.frames_suppressed += 1
        return FrameCommandResult(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            emitted=False,
            reason=reason,
            observation_source=observation_source,
            gate_id=gate_id,
            vision_candidate_count=vision_candidate_count,
            confident_candidate_count=confident_candidate_count,
            selected_candidate_count=selected_candidate_count,
            selection_reason=None if selection is None else selection.reason,
            selection=selection,
            route_state=route_state,
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
        command_transform: CommandFrameTransform | None = None,
        body_rate_controller: BodyRateGuidanceController | None = None,
        route_state_tracker: RouteStateTracker | None = None,
        safety: SafetySupervisor | None = None,
        logger: Logger | None = None,
    ):
        self.config = config or Q1RuntimeConfig()
        self.bridge = bridge or MavlinkBridge(self.config.mavlink_endpoint)
        self.receiver = receiver or self._build_receiver()
        self.tracker = tracker or getattr(processor, "tracker", None) or TargetTracker(hold_s=self.config.target_hold_s)
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
        self.route_state_tracker = route_state_tracker or RouteStateTracker(
            velocity_source=self.config.target_selection_velocity_source
        )
        self.body_rate_controller = body_rate_controller or BodyRateGuidanceController(
            self.bridge,
            BodyRateGuidanceControlConfig(
                velocity_source=self.config.body_rate_velocity_source,
                base_thrust=self.config.body_rate_base_thrust,
                vertical_kp=self.config.body_rate_vertical_kp,
                vertical_kd=self.config.body_rate_vertical_kd,
                min_thrust=self.config.body_rate_min_thrust,
                max_thrust=self.config.body_rate_max_thrust,
                roll_kp=self.config.body_rate_roll_kp,
                pitch_kp=self.config.body_rate_pitch_kp,
                max_roll_rate_rps=self.config.body_rate_max_roll_rate_rps,
                max_pitch_rate_rps=self.config.body_rate_max_pitch_rate_rps,
                yaw_rate_rps=self.config.body_rate_yaw_rate_rps,
                position_kp_deg_per_m=self.config.body_rate_position_kp_deg_per_m,
                velocity_kd_deg_per_mps=self.config.body_rate_velocity_kd_deg_per_mps,
                max_guidance_tilt_deg=self.config.body_rate_max_guidance_tilt_deg,
                position_deadband_m=self.config.body_rate_position_deadband_m,
                max_horizontal_target_m=self.config.vision_target_max_horizontal_m,
                max_up_target_m=self.config.vision_target_max_up_m,
                max_down_target_m=self.config.vision_target_max_down_m,
                dry_run=self.config.dry_run,
            ),
            command_transform=self.command_transform,
        )
        self.logger = logger
        self.cycle = 0
        self.command_cycle = 0
        self.commands_emitted = 0
        self.command_suppressions = 0
        self.safety_stop_count = 0
        self.stop_requested = False
        self.stop_reason: str | None = None
        self.last_frame_result: FrameCommandResult | None = None
        self.last_command_result: ControlTickResult | None = None
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
        route_state = self.route_state_tracker.update(self.bridge, telemetry)
        if route_state.gate_index_changed:
            self._handle_gate_index_changed(route_state)
        result = self.processor.process_frame(frame, telemetry, now_s=now_s, route_state=route_state)
        self._log_result(result, telemetry)
        self.cycle += 1
        return result

    def tick_command(self, *, now_s: float | None = None) -> ControlTickResult:
        now = time.monotonic() if now_s is None else float(now_s)
        telemetry = self.bridge.get_latest_telemetry()
        safety_decision = self.safety.evaluate(telemetry, now_s=now)
        if safety_decision.reason == "simulator_reset_detected":
            self.tracker.clear()
            if self.processor.target_filter is not None:
                self.processor.target_filter.reset()
        if safety_decision.should_stop:
            command = self.body_rate_controller.emit_stop(
                telemetry,
                source=f"q1runtime_safety_{safety_decision.reason}",
            )
            self.commands_emitted += 1
            self.command_cycle += 1
            self.safety_stop_count += 1
            if safety_decision.action == "stop_and_disarm":
                self._disarm_quietly()
            if self.config.stop_runtime_on_safety_stop:
                self.request_stop(f"safety_{safety_decision.reason}")
            return ControlTickResult(
                emitted=True,
                reason=f"safety_{safety_decision.reason}",
                command=command,
                safety=safety_decision,
            )
        return self._tick_body_rate_command(telemetry, now_s=now, safety_decision=safety_decision)

    def _tick_body_rate_command(
        self,
        telemetry: Any | None,
        *,
        now_s: float,
        safety_decision: SafetyDecision,
    ) -> ControlTickResult:
        if telemetry is None:
            return self._command_suppressed("missing_telemetry_for_body_rate_command")

        target_age = self.tracker.age_s(now_s=now_s)
        tracked = self.tracker.latest(now_s=now_s)
        if tracked is None and target_age is not None and target_age > float(self.config.body_rate_target_stale_abort_s):
            command = self.body_rate_controller.emit_stop(telemetry, source="q1runtime_body_rate_stale_target_abort")
            self.commands_emitted += 1
            self.command_cycle += 1
            self.safety_stop_count += 1
            self._disarm_quietly()
            if self.config.stop_runtime_on_safety_stop:
                self.request_stop("body_rate_stale_target_abort")
            return ControlTickResult(
                emitted=True,
                reason="body_rate_stale_target_abort",
                command=command,
                target_age_s=target_age,
                safety=safety_decision,
            )

        target = None if tracked is None else tracked.target
        command = self.body_rate_controller.emit_guidance(
            telemetry=telemetry,
            target=target,
            source="q1runtime_body_rate_guidance" if target is not None else "q1runtime_body_rate_hold_no_target",
        )
        self.commands_emitted += 1
        self.command_cycle += 1
        return ControlTickResult(
            emitted=True,
            reason="streamed_body_rate_guidance" if target is not None else "streamed_body_rate_hold_no_target",
            command=command,
            target=target,
            target_age_s=target_age,
            safety=safety_decision,
        )

    def run(self) -> None:
        run_dir = Logger.timestamped_dir(self.config.log_dir)
        self.logger = self.logger or Logger(self._metadata(run_dir))
        previous_signal_handlers = self._install_signal_handlers()

        try:
            self.receiver.start_listener()
            self.logger.log_event("vision_started", receiver=self.receiver.snapshot())
            self.bridge.connect(heartbeat_timeout_s=self.config.heartbeat_timeout_s)
            self.bridge.start_heartbeat()
            self.bridge.subscribe_telemetry()
            self.logger.log_event("connected", bridge=self.bridge.snapshot())
            self._wait_for_telemetry()
            if self.stop_requested:
                return
            if self.config.reset_on_start:
                self._reset_simulator()
            if self.stop_requested:
                return
            if self.config.arm_on_start:
                self._arm_and_wait()
            if self.stop_requested:
                return
            if self.config.control_method == "body_rate_guidance":
                self._run_body_rate_prelevel()
            if self.stop_requested:
                return
            self.logger.log_event("q1runtime_started", processor=self.processor.snapshot())

            active_start = time.monotonic()
            next_status = active_start
            next_checkpoint = active_start + max(0.1, float(self.config.checkpoint_interval_s))
            run_deadline = None if self.config.run_s is None else active_start + float(self.config.run_s)
            self.command_streamer.start()

            while not self.stop_requested and (run_deadline is None or time.monotonic() < run_deadline):
                now = time.monotonic()
                result = self.process_next_frame(now_s=now)
                if result is None:
                    time.sleep(self.config.idle_sleep_s)
                if now >= next_status:
                    self._print_status()
                    next_status = now + self.config.status_interval_s
                if now >= next_checkpoint:
                    self._save_checkpoint(run_dir, reason="periodic")
                    next_checkpoint = now + max(0.1, float(self.config.checkpoint_interval_s))
        finally:
            self.command_streamer.stop()
            self._shutdown_controls()
            if self.logger is not None:
                self.logger.log_event(
                    "q1runtime_stopped",
                    stop_requested=self.stop_requested,
                    stop_reason=self.stop_reason,
                    processor=self.processor.snapshot(),
                    bridge=self.bridge.snapshot(),
                    safety=self.safety.snapshot(),
                    command_transform=self.command_transform.to_payload(),
                    body_rate_controller=self.body_rate_controller.snapshot_state(),
                    route_state=self.route_state_tracker.snapshot(),
                )
                self._save_checkpoint(run_dir, reason="final")
            if hasattr(self.receiver, "shutdown"):
                self.receiver.shutdown()
            if hasattr(self.bridge, "shutdown"):
                self.bridge.shutdown()
            self._restore_signal_handlers(previous_signal_handlers)

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
        while not self.stop_requested and time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None and getattr(telemetry, "position_local_ned_m", None) is not None:
                return
            time.sleep(self.config.idle_sleep_s)
        if self.stop_requested:
            return
        raise TimeoutError("No local-NED telemetry received before q1runtime startup timeout.")

    def _reset_simulator(self) -> None:
        if self.config.dry_run:
            if self.logger is not None:
                self.logger.log_event("simulator_reset_skipped_dry_run")
            return
        previous = self.bridge.get_latest_telemetry()
        previous_reset_count = None if previous is None else getattr(previous, "reset_count", None)
        self.bridge.send_sim_reset_command()
        time.sleep(self.config.reset_wait_s)
        if self.stop_requested:
            return
        self._wait_for_telemetry()
        if self.stop_requested:
            return
        self._wait_for_reset_ready(previous_reset_count)

    def _wait_for_reset_ready(self, previous_reset_count: Any | None) -> None:
        deadline = time.monotonic() + max(0.01, float(self.config.reset_ready_timeout_s))
        stable_start: float | None = None
        while not self.stop_requested and time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is None or getattr(telemetry, "position_local_ned_m", None) is None:
                stable_start = None
                time.sleep(0.05)
                continue
            reset_count = getattr(telemetry, "reset_count", None)
            if previous_reset_count is not None and reset_count == previous_reset_count:
                stable_start = None
                time.sleep(0.05)
                continue
            velocity = getattr(telemetry, "velocity_local_ned_mps", (0.0, 0.0, 0.0))
            speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
            if speed <= float(self.config.reset_stable_max_speed_mps):
                if stable_start is None:
                    stable_start = time.monotonic()
                if time.monotonic() - stable_start >= float(self.config.reset_stable_s):
                    if self.logger is not None:
                        self.logger.log_event(
                            "simulator_reset_ready",
                            reset_count=reset_count,
                            speed_mps=speed,
                            stable_s=self.config.reset_stable_s,
                            post_reset_delay_s=self.config.post_reset_delay_s,
                        )
                    if self.config.post_reset_delay_s > 0.0:
                        time.sleep(self.config.post_reset_delay_s)
                    return
            else:
                stable_start = None
            time.sleep(0.05)
        if self.stop_requested:
            return
        raise TimeoutError("Simulator did not reach stable post-reset telemetry before timeout.")

    def _arm_and_wait(self) -> None:
        self.bridge.arm()
        deadline = time.monotonic() + self.config.arm_timeout_s
        while not self.stop_requested and time.monotonic() < deadline:
            if getattr(self.bridge, "armed", False):
                return
            time.sleep(0.02)
        if self.stop_requested:
            return
        raise TimeoutError("Simulator did not confirm armed state before q1runtime arm timeout.")

    def _run_body_rate_prelevel(self) -> None:
        if self.config.body_rate_prelevel_s <= 0.0:
            return
        deadline = time.monotonic() + float(self.config.body_rate_prelevel_s)
        interval = 1.0 / max(1e-6, float(self.config.command_hz))
        next_tick = time.monotonic()
        prelevel_cycle = 0
        if self.logger is not None:
            self.logger.log_event(
                "body_rate_prelevel_started",
                duration_s=self.config.body_rate_prelevel_s,
                thrust=self.config.body_rate_prelevel_thrust,
            )
        while not self.stop_requested and time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += interval
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is None:
                continue
            command = self.body_rate_controller.emit_prelevel(
                telemetry=telemetry,
                target_roll_deg=self.config.body_rate_prelevel_target_roll_deg,
                target_pitch_deg=self.config.body_rate_prelevel_target_pitch_deg,
                thrust=self.config.body_rate_prelevel_thrust,
            )
            if self.logger is not None:
                log_every = max(1, round(float(self.config.command_hz) / max(1e-6, float(self.config.command_log_hz))))
                if prelevel_cycle % log_every == 0:
                    self.logger.log_event("body_rate_prelevel_tick", prelevel_cycle=prelevel_cycle, command=command)
            prelevel_cycle += 1
        if self.logger is not None:
            self.logger.log_event("body_rate_prelevel_finished", cycles=prelevel_cycle)

    def _log_result(self, result: FrameCommandResult, telemetry: Any | None) -> None:
        self.last_frame_result = result
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
        self.last_command_result = result
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
            body_rate_controller=self.body_rate_controller.snapshot_state(),
            safety=self.safety.snapshot(),
            route_state=self.route_state_tracker.snapshot(),
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
            "control_method": self.config.control_method,
            "vision_mode": "cnn_regressor_passthrough_top1",
            "command_stream": "fixed_hz_body_rate_guidance",
            "run_dir": str(run_dir),
            "config": config,
            "command_streamer": self.command_streamer.snapshot(),
            "command_transform": self.command_transform.to_payload(),
            "safety": self.safety.snapshot(),
            "body_rate_controller": self.body_rate_controller.snapshot_state(),
            "route_state": self.route_state_tracker.snapshot(),
        }

    def _handle_gate_index_changed(self, route_state: RouteStateSnapshot) -> None:
        self.tracker.clear()
        if self.processor.target_filter is not None:
            self.processor.target_filter.reset()
        self.processor.target_selector.reset()
        if self.logger is not None:
            self.logger.log_event("race_gate_index_changed", route_state=route_state.to_log_dict())

    def request_stop(self, reason: str) -> None:
        self.stop_requested = True
        self.stop_reason = reason

    def _install_signal_handlers(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}

        def _handler(signum: int, _frame: Any) -> None:
            try:
                signal_name = signal.Signals(signum).name.lower()
            except ValueError:
                signal_name = str(signum)
            self.request_stop(f"signal_{signal_name}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, _handler)
            except (ValueError, OSError):
                continue
        return previous

    def _restore_signal_handlers(self, previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                continue

    def _save_checkpoint(self, run_dir: Path, *, reason: str) -> None:
        if self.logger is None:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.save_run(run_dir / "run.json")
        status_path = run_dir / "status.json"
        status_path.write_text(
            json.dumps(self._status_snapshot(reason=reason), indent=2, default=self.logger._json_default),
            encoding="utf-8",
        )

    def _status_snapshot(self, *, reason: str) -> dict[str, Any]:
        telemetry = self.bridge.get_latest_telemetry()
        position = getattr(telemetry, "position_local_ned_m", None)
        velocity = getattr(telemetry, "velocity_local_ned_mps", None)
        target_age = self.tracker.age_s(now_s=time.monotonic())
        return {
            "reason": reason,
            "stop_requested": self.stop_requested,
            "stop_reason": self.stop_reason,
            "cycle": self.cycle,
            "command_cycle": self.command_cycle,
            "frames_processed": self.processor.frames_processed,
            "target_updates": self.processor.targets_updated,
            "frames_suppressed": self.processor.frames_suppressed,
            "commands_emitted": self.commands_emitted,
            "command_suppressions": self.command_suppressions,
            "safety_stop_count": self.safety_stop_count,
            "position_local_ned_m": None if position is None else [float(value) for value in position],
            "velocity_local_ned_mps": None if velocity is None else [float(value) for value in velocity],
            "target_age_s": target_age,
            "last_frame_reason": None if self.last_frame_result is None else self.last_frame_result.reason,
            "last_command_reason": None if self.last_command_result is None else self.last_command_result.reason,
            "last_command": None if self.last_command_result is None else self.last_command_result.to_log_dict(),
            "tracker": self.tracker.snapshot(now_s=time.monotonic()),
            "route_state": self.route_state_tracker.snapshot(),
            "safety": self.safety.snapshot(),
        }

    def _shutdown_controls(self) -> None:
        if self.config.stop_on_exit_s > 0.0:
            end = time.monotonic() + float(self.config.stop_on_exit_s)
            interval = 1.0 / max(1e-6, float(self.config.command_hz))
            while time.monotonic() < end:
                try:
                    self.body_rate_controller.emit_stop(
                        self.bridge.get_latest_telemetry(),
                        source="q1runtime_shutdown_stop",
                    )
                except Exception as exc:
                    if self.logger is not None:
                        self.logger.log_event("shutdown_stop_failed", error=str(exc))
                    break
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
        velocity = getattr(telemetry, "velocity_local_ned_mps", None)
        speed = None if velocity is None else float(np.linalg.norm(np.asarray(velocity, dtype=float)))
        target_age = self.tracker.age_s(now_s=time.monotonic())
        route_state = self.route_state_tracker.snapshot()
        print(
            "q1runtime "
            f"cycle={self.cycle} "
            f"pos={None if position is None else [round(float(value), 3) for value in position]} "
            f"speed={None if speed is None else round(speed, 3)} "
            f"active_gate={route_state.get('active_gate_index')} "
            f"route_velocity_source={route_state.get('velocity_source')} "
            f"frames={self.processor.frames_processed} "
            f"target_updates={self.processor.targets_updated} "
            f"target_age={None if target_age is None else round(float(target_age), 3)} "
            f"commands={self.commands_emitted} "
            f"last_command={None if self.last_command_result is None else self.last_command_result.reason} "
            f"control_method={self.config.control_method} "
            f"frame_suppressed={self.processor.frames_suppressed} "
            f"command_suppressed={self.command_suppressions} "
            f"safety_stops={self.safety_stop_count}"
        )
