import math

import numpy as np

from q1runtime import (
    CommandEmitter,
    CommandStreamer,
    CommandFrameTransform,
    LocalVisionTarget,
    Q1FrameProcessor,
    Q1Runtime,
    Q1RuntimeConfig,
    Q1VisionTargeter,
    SafetyConfig,
    SafetySupervisor,
    TargetMapper,
    TargetFilter,
    TargetTracker,
    VelocityPlanner,
    VelocityPlannerConfig,
)
from q1runtime.authority_probe import AuthorityProbe, AxisPulse
from q1runtime.body_rate_feedback_probe import BodyRateFeedbackConfig, BodyRateFeedbackProbe
from q1runtime.body_rate_position_guidance import BodyRatePositionGuidance, BodyRatePositionGuidanceConfig
from q1runtime.local_ned_sweep import LocalNedCommandSweep, LocalNedSweepConfig, TelemetrySnapshot
from q1runtime.live_snapshot import analyze_motion, build_timeline, format_snapshot_text, summarize_vision
from sensing.perception import VisionGateObservation, VisionObservation
from sensing.telemetry import TelemetrySample
from sensing.vision.vision_stream import VisionFrame


def telemetry(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> TelemetrySample:
    return TelemetrySample(
        sim_time_ns=100,
        attitude=(1.0, 0.0, 0.0, 0.0),
        velocity_local_ned_mps=velocity,
        body_rates_rps=(0.0, 0.0, 0.0),
        position_local_ned_m=position,
        reset_count=1,
    )


def gate(gate_id: str, position_camera_m, confidence=0.9) -> VisionGateObservation:
    return VisionGateObservation(
        gate_id=gate_id,
        position_camera_m=tuple(float(value) for value in position_camera_m),
        position_confidence=confidence,
        orientation_camera=(0.0, 0.0, 1.0),
        orientation_confidence=confidence,
    )


class FakeService:
    def __init__(self, observations):
        self.observations = observations

    def process_frame(self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes) -> VisionObservation:
        return self.observations[int(frame_id)]

    def snapshot(self):
        return {"fake": True}


class FakeBridge:
    def __init__(self, sample: TelemetrySample):
        self.sample = sample
        self.commands = []
        self.armed = True
        self.disarm_count = 0

    def get_latest_telemetry(self):
        return self.sample

    def send_position_target(self, payload):
        self.commands.append(payload)

    def snapshot(self):
        return {"commands": len(self.commands)}

    def disarm(self):
        self.disarm_count += 1
        self.armed = False


class FakeReceiver:
    def __init__(self, frames):
        self.frames = list(frames)

    def get_next_frame(self):
        return self.frames.pop(0) if self.frames else None


def test_default_config_builds_regressor_passthrough_top1_targeter() -> None:
    targeter = Q1VisionTargeter.from_config(Q1RuntimeConfig())

    assert targeter.service.config.run_landmarker is True
    assert targeter.service.config.passthrough_regressor_targets is True
    assert targeter.service.config.top_k == 1
    assert targeter.min_position_confidence == 0.5


def test_targeter_selects_nearest_gate_above_confidence() -> None:
    observation = VisionObservation(
        frame_id=7,
        sim_time_ns=77,
        gates=(
            gate("far", (0.0, 0.0, 8.0)),
            gate("low", (0.0, 0.0, 1.0), confidence=0.2),
            gate("near", (0.0, 0.0, 2.0)),
        ),
        source="fake",
    )
    targeter = Q1VisionTargeter(service=FakeService({7: observation}), min_position_confidence=0.5)

    selected = targeter.select_top_gate(observation)

    assert selected is not None
    assert selected.gate_id == "near"


def test_target_mapper_converts_camera_target_to_local_ned() -> None:
    target = TargetMapper(camera_tilt_deg=0.0).map_gate(
        gate("gate-001", (0.0, 0.0, 10.0)),
        frame_id=3,
        sim_time_ns=99,
        telemetry=telemetry(position=(1.0, 2.0, -3.0)),
    )

    assert target.gate_id == "gate-001"
    np.testing.assert_allclose(target.position_local_ned_m, (11.0, 2.0, -3.0))
    assert math.isclose(target.yaw_rad, 0.0)


def test_command_emitter_defaults_to_velocity_only_payload() -> None:
    bridge = FakeBridge(telemetry())
    emitter = CommandEmitter(
        bridge,
        max_approach_speed_mps=2.0,
        max_vertical_speed_mps=2.0,
        approach_gain_hz=1.0,
        arrival_radius_m=0.0,
    )
    target = LocalVisionTarget(
        frame_id=5,
        sim_time_ns=500,
        gate_id="gate-005",
        position_camera_m=(0.0, 0.0, 1.0),
        position_local_ned_m=(3.0, 4.0, 0.0),
        yaw_rad=math.atan2(4.0, 3.0),
        position_confidence=0.8,
    )

    emitted = emitter.emit(target, telemetry())

    assert bridge.commands == [emitted.payload]
    assert "position_local_ned_m" not in emitted.payload
    assert emitted.payload["target_position_local_ned_m"] == [3.0, 4.0, 0.0]
    np.testing.assert_allclose(emitted.payload["velocity_local_ned_mps"], [1.2, 1.6, 0.0])
    assert emitted.payload["yaw_rad"] is None
    assert emitted.payload["source"] == "q1runtime_top1_regressor_passthrough"


def test_frame_processor_suppresses_frames_without_confident_top1_gate() -> None:
    observation = VisionObservation(
        frame_id=1,
        sim_time_ns=10,
        gates=(gate("low", (0.0, 0.0, 1.0), confidence=0.2),),
        source="fake",
    )
    bridge = FakeBridge(telemetry())
    tracker = TargetTracker()
    processor = Q1FrameProcessor(
        targeter=Q1VisionTargeter(service=FakeService({1: observation}), min_position_confidence=0.5),
        mapper=TargetMapper(camera_tilt_deg=0.0),
        tracker=tracker,
    )

    result = processor.process_frame(VisionFrame(1, 10, b"jpeg"), telemetry(), now_s=0.0)

    assert result.emitted is False
    assert result.target_updated is False
    assert result.reason == "no_top1_gate_above_confidence"
    assert bridge.commands == []
    assert tracker.latest(now_s=0.0) is None


def test_runtime_updates_target_from_frames_and_streams_velocity_commands() -> None:
    observations = {
        1: VisionObservation(1, 10, (gate("first", (0.0, 0.0, 10.0)),), source="fake"),
        2: VisionObservation(2, 20, (gate("second", (0.0, 0.0, 5.0)),), source="fake"),
    }
    bridge = FakeBridge(telemetry())
    tracker = TargetTracker(hold_s=3.0)
    processor = Q1FrameProcessor(
        targeter=Q1VisionTargeter(service=FakeService(observations), min_position_confidence=0.5),
        mapper=TargetMapper(camera_tilt_deg=0.0),
        tracker=tracker,
    )
    runtime = Q1Runtime(
        Q1RuntimeConfig(
            dry_run=False,
            command_mode="velocity",
            max_approach_speed_mps=2.0,
            max_vertical_speed_mps=2.0,
            max_accel_mps2=0.0,
        ),
        bridge=bridge,
        receiver=FakeReceiver([VisionFrame(1, 10, b"a"), VisionFrame(2, 20, b"b")]),
        processor=processor,
        tracker=tracker,
        emitter=CommandEmitter(
            bridge,
            command_mode="velocity",
            max_approach_speed_mps=2.0,
            max_vertical_speed_mps=2.0,
            approach_gain_hz=1.0,
            arrival_radius_m=0.0,
        ),
    )

    first = runtime.process_next_frame(now_s=0.0)
    assert first is not None
    assert first.emitted is False
    assert first.target_updated is True
    first_tick = runtime.tick_command(now_s=0.0)
    assert first_tick.emitted is True
    assert len(bridge.commands) == 1
    assert bridge.commands[0]["target_position_local_ned_m"] == [10.0, 0.0, 0.0]
    np.testing.assert_allclose(bridge.commands[0]["velocity_local_ned_mps"], [2.0, 0.0, 0.0])

    repeat_tick = runtime.tick_command(now_s=1.0)
    assert repeat_tick.emitted is True
    assert len(bridge.commands) == 2
    assert bridge.commands[1]["target_position_local_ned_m"] == [10.0, 0.0, 0.0]

    second = runtime.process_next_frame(now_s=1.1)
    assert second is not None
    assert second.target_updated is True
    second_tick = runtime.tick_command(now_s=1.1)
    assert second_tick.emitted is True
    assert len(bridge.commands) == 3
    assert bridge.commands[2]["target_position_local_ned_m"] == [5.0, 0.0, 0.0]
    assert runtime.process_next_frame() is None


def test_live_snapshot_interprets_high_energy_motion_with_relative_timeline() -> None:
    telemetry_samples = [
        {"t_plus_s": 0.0, "position_local_ned_m": [0.0, 0.0, 0.0], "velocity_local_ned_mps": [9.0, 0.0, 6.0]},
        {"t_plus_s": 1.0, "position_local_ned_m": [9.0, 0.0, 6.0], "velocity_local_ned_mps": [9.0, 0.0, 6.0]},
    ]
    attitude_samples = [
        {"t_plus_s": 0.0, "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0},
        {"t_plus_s": 1.0, "roll_rad": math.radians(35.0), "pitch_rad": 0.0, "yaw_rad": 0.0},
    ]
    actuator_samples = [
        {"t_plus_s": 0.0, "actuator_first4": [0.1, 0.2, 0.3, 0.4]},
        {"t_plus_s": 1.0, "actuator_first4": [0.2, 0.1, 0.4, 0.3]},
    ]

    motion = analyze_motion(telemetry_samples, attitude_samples, actuator_samples, [])
    timeline = build_timeline(telemetry_samples, attitude_samples, actuator_samples, step_s=1.0, duration_s=1.0)

    assert motion["state"] == "high_energy_motion"
    assert motion["displacement_local_ned_m"] == [9.0, 0.0, 6.0]
    assert motion["z_direction"] == "descending in NED (+z)"
    assert "Attitude excursions are large" in motion["description"]
    assert timeline[0]["t_plus_s"] == 0.0
    assert timeline[-1]["t_plus_s"] == 1.0


def test_live_snapshot_formats_stream_summary_without_raw_chunks() -> None:
    vision = summarize_vision(
        {
            7: {
                "first_seen_t_plus_s": 0.2,
                "total_chunks": 2,
                "jpeg_size": 100,
                "sim_time_ns": 77,
                "chunks": {0, 1},
                "packet_count": 5,
            }
        },
        invalid_count=0,
    )
    summary = {
        "snapshot_started_wall": "2026-06-10T00:00:00.000-07:00",
        "snapshot_ended_wall": "2026-06-10T00:00:01.000-07:00",
        "duration_s": 1.0,
        "receive_only": True,
        "packet_counts": {"telemetry": 1},
        "sources": {"telemetry": {"127.0.0.1:14560": 1}},
        "mavlink_message_counts": {"LOCAL_POSITION_NED": 1},
        "bind_errors": {},
        "vision": vision,
        "motion": {"state": "stationary_or_hover_like", "description": "Drone is nearly stationary."},
        "timeline": [
            {
                "t_plus_s": 0.0,
                "position_local_ned_m": [0.0, 0.0, 0.0],
                "displacement_from_start_m": [0.0, 0.0, 0.0],
                "velocity_local_ned_mps": [0.0, 0.0, 0.0],
                "speed_mps": 0.0,
            }
        ],
    }

    text = format_snapshot_text(summary)

    assert vision["complete_frames"] == 1
    assert vision["duplicate_packet_estimate"] == 3
    assert "Snapshot: 2026-06-10T00:00:00.000-07:00" in text
    assert "t+0.000s" in text


def test_runtime_streams_stop_command_after_target_hold_expires() -> None:
    observations = {
        1: VisionObservation(1, 10, (gate("first", (0.0, 0.0, 10.0)),), source="fake"),
    }
    bridge = FakeBridge(telemetry())
    tracker = TargetTracker(hold_s=0.5)
    processor = Q1FrameProcessor(
        targeter=Q1VisionTargeter(service=FakeService(observations), min_position_confidence=0.5),
        mapper=TargetMapper(camera_tilt_deg=0.0),
        tracker=tracker,
    )
    runtime = Q1Runtime(
        Q1RuntimeConfig(dry_run=False, target_hold_s=0.5, stream_stop_when_target_lost=True),
        bridge=bridge,
        receiver=FakeReceiver([VisionFrame(1, 10, b"a")]),
        processor=processor,
        tracker=tracker,
        emitter=CommandEmitter(bridge, command_mode="velocity"),
    )

    runtime.process_next_frame(now_s=0.0)
    active_tick = runtime.tick_command(now_s=0.1)
    stale_tick = runtime.tick_command(now_s=0.6)

    assert active_tick.reason == "streamed_velocity_target_local_ned"
    assert stale_tick.reason == "emitted_stop_no_active_target"
    assert bridge.commands[-1] == {
        "velocity_local_ned_mps": [0.0, 0.0, 0.0],
        "yaw_rad": None,
        "source": "q1runtime_stop_no_active_target",
    }


def test_command_streamer_steps_at_fixed_rate_without_vision_loop() -> None:
    calls = []
    results = []
    streamer = CommandStreamer(
        lambda now_s: calls.append(now_s) or {"now_s": now_s},
        hz=2.0,
        on_result=results.append,
    )

    assert streamer.step(10.0) == {"now_s": 10.0}
    assert streamer.step(10.25) is None
    assert streamer.step(10.50) == {"now_s": 10.5}

    assert calls == [10.0, 10.5]
    assert results == [{"now_s": 10.0}, {"now_s": 10.5}]


def test_command_transform_supports_sign_flip_and_axis_swap() -> None:
    transform = CommandFrameTransform.from_options(invert_x=True, swap_xy=True)

    np.testing.assert_allclose(transform.apply((2.0, 3.0, 4.0)), [-3.0, 2.0, 4.0])


def test_velocity_planner_holds_vertical_axis_by_default() -> None:
    planner = VelocityPlanner(
        VelocityPlannerConfig(
            max_speed_mps=1.0,
            max_vertical_speed_mps=1.0,
            vertical_mode="hold",
            max_accel_mps2=0.0,
            arrival_radius_m=0.0,
        )
    )
    target = LocalVisionTarget(
        frame_id=1,
        sim_time_ns=1,
        gate_id="gate",
        position_camera_m=(0.0, 0.0, 1.0),
        position_local_ned_m=(10.0, 0.0, -5.0),
        yaw_rad=0.0,
        position_confidence=1.0,
    )

    velocity = planner.desired_velocity(target, telemetry(), now_s=1.0)

    np.testing.assert_allclose(velocity, [1.0, 0.0, 0.0])


def test_target_filter_rejects_large_jumps_and_smooths_valid_targets() -> None:
    target_filter = TargetFilter(max_jump_m=5.0, smoothing_alpha=0.5)
    first = LocalVisionTarget(1, 1, "a", (0.0, 0.0, 1.0), (10.0, 0.0, 0.0), 0.0, 1.0)
    second = LocalVisionTarget(2, 2, "b", (0.0, 0.0, 1.0), (12.0, 0.0, 0.0), 0.0, 1.0)
    jump = LocalVisionTarget(3, 3, "c", (0.0, 0.0, 1.0), (30.0, 0.0, 0.0), 0.0, 1.0)

    assert target_filter.update(first).accepted is True
    smoothed = target_filter.update(second)
    rejected = target_filter.update(jump)

    assert smoothed.accepted is True
    assert smoothed.target is not None
    assert smoothed.target.position_local_ned_m == (11.0, 0.0, 0.0)
    assert rejected.accepted is False
    assert rejected.reason == "rejected_target_jump"


def test_safety_supervisor_trips_on_reset_and_bounds() -> None:
    safety = SafetySupervisor(SafetyConfig(max_horizontal_radius_m=2.0, min_z_ned_m=-2.0, max_z_ned_m=2.0))

    assert safety.evaluate(telemetry(position=(0.0, 0.0, 0.0)), now_s=0.0).action == "continue"
    assert safety.evaluate(telemetry(position=(3.0, 0.0, 0.0)), now_s=0.1).reason == "horizontal_radius_exceeded"

    safety = SafetySupervisor(SafetyConfig())
    first = telemetry()
    second = TelemetrySample(
        sim_time_ns=2,
        attitude=first.attitude,
        velocity_local_ned_mps=first.velocity_local_ned_mps,
        body_rates_rps=first.body_rates_rps,
        position_local_ned_m=first.position_local_ned_m,
        reset_count=1,
    )
    third = TelemetrySample(
        sim_time_ns=3,
        attitude=first.attitude,
        velocity_local_ned_mps=first.velocity_local_ned_mps,
        body_rates_rps=first.body_rates_rps,
        position_local_ned_m=first.position_local_ned_m,
        reset_count=2,
    )

    assert safety.evaluate(second, now_s=0.0).action == "continue"
    assert safety.evaluate(third, now_s=0.1).reason == "simulator_reset_detected"


def test_authority_probe_summarizes_axis_response_and_transform() -> None:
    probe = AuthorityProbe(FakeBridge(telemetry()))
    result = probe.summarize_pulse(
        AxisPulse("+x", (0.25, 0.0, 0.0)),
        start_position_local_ned_m=(0.0, 0.0, 0.0),
        end_position_local_ned_m=(-1.0, 0.0, 0.0),
    )

    transform = probe.recommended_transform([result])

    assert result.dominant_axis == "x"
    assert result.dominant_sign == -1
    np.testing.assert_allclose(transform.apply((1.0, 0.0, 0.0)), [-1.0, 0.0, 0.0])


def test_runtime_shutdown_controls_stop_and_disarm() -> None:
    bridge = FakeBridge(telemetry())
    runtime = Q1Runtime(
        Q1RuntimeConfig(stop_on_exit_s=0.001, disarm_on_exit=True),
        bridge=bridge,
        receiver=FakeReceiver([]),
    )

    runtime._shutdown_controls()

    assert bridge.commands[-1]["velocity_local_ned_mps"] == [0.0, 0.0, 0.0]
    assert bridge.disarm_count == 1


def test_runtime_clears_tracker_and_stops_on_reset_change() -> None:
    bridge = FakeBridge(telemetry())
    tracker = TargetTracker(hold_s=10.0)
    target = LocalVisionTarget(1, 1, "gate", (0.0, 0.0, 1.0), (5.0, 0.0, 0.0), 0.0, 1.0)
    tracker.update(target, now_s=0.0)
    runtime = Q1Runtime(
        Q1RuntimeConfig(max_accel_mps2=0.0),
        bridge=bridge,
        receiver=FakeReceiver([]),
        tracker=tracker,
    )

    assert runtime.tick_command(now_s=0.0).reason == "streamed_velocity_target_local_ned"
    bridge.sample = TelemetrySample(
        sim_time_ns=2,
        attitude=bridge.sample.attitude,
        velocity_local_ned_mps=bridge.sample.velocity_local_ned_mps,
        body_rates_rps=bridge.sample.body_rates_rps,
        position_local_ned_m=bridge.sample.position_local_ned_m,
        reset_count=2,
    )
    reset_tick = runtime.tick_command(now_s=0.1)

    assert reset_tick.reason == "safety_simulator_reset_detected"
    assert bridge.commands[-1]["velocity_local_ned_mps"] == [0.0, 0.0, 0.0]
    assert tracker.latest(now_s=0.1) is None


def test_local_ned_sweep_builds_conservative_velocity_cases_by_default() -> None:
    sweep = LocalNedCommandSweep(
        FakeBridge(telemetry()),
        LocalNedSweepConfig(speeds_mps=(0.1,), axes=("x", "y"), include_z=False),
    )

    cases = sweep.build_cases()

    assert [case.name for case in cases] == [
        "velocity_plusx_0.1mps",
        "velocity_minusx_0.1mps",
        "velocity_plusy_0.1mps",
        "velocity_minusy_0.1mps",
    ]
    assert all(case.mode == "velocity" for case in cases)
    assert all(case.position_offset_local_ned_m is None for case in cases)
    assert {case.axis for case in cases} == {"x", "y"}


def test_local_ned_sweep_requires_z_axis_opt_in() -> None:
    sweep = LocalNedCommandSweep(
        FakeBridge(telemetry()),
        LocalNedSweepConfig(speeds_mps=(0.1,), axes=("z",), include_z=False),
    )

    try:
        sweep.build_cases()
    except ValueError as exc:
        assert "include_z=True" in str(exc)
    else:
        raise AssertionError("Expected z-axis sweep to require include_z=True.")


def test_local_ned_sweep_builds_opt_in_position_payload_from_case_start() -> None:
    sweep = LocalNedCommandSweep(
        FakeBridge(telemetry(position=(1.0, 2.0, -3.0))),
        LocalNedSweepConfig(
            speeds_mps=(0.1,),
            position_offsets_m=(0.5,),
            axes=("x", "z"),
            include_z=True,
            include_position=True,
            include_position_velocity=True,
            include_yaw=True,
            yaw_angles_rad=(1.0,),
        ),
    )

    cases = sweep.build_cases()
    position_case = next(case for case in cases if case.name == "position_plusx_0.5m")
    mixed_case = next(case for case in cases if case.name == "position_velocity_plusz_0.5m_0.1mps")
    yaw_case = next(case for case in cases if case.mode == "yaw")

    position_payload = sweep.build_payload(position_case, sweep.snapshot())
    mixed_payload = sweep.build_payload(mixed_case, sweep.snapshot())
    yaw_payload = sweep.build_payload(yaw_case, sweep.snapshot())

    assert position_payload["position_local_ned_m"] == [1.5, 2.0, -3.0]
    assert position_payload["velocity_local_ned_mps"] == [0.0, 0.0, 0.0]
    assert mixed_payload["position_local_ned_m"] == [1.0, 2.0, -2.5]
    assert mixed_payload["velocity_local_ned_mps"] == [0.0, 0.0, 0.1]
    assert yaw_payload["yaw_rad"] == 1.0


def test_local_ned_sweep_summarizes_response_metrics() -> None:
    sweep = LocalNedCommandSweep(
        FakeBridge(telemetry()),
        LocalNedSweepConfig(speeds_mps=(0.25,), pulse_s=0.5),
    )
    case = sweep.build_cases()[0]
    start = TelemetrySnapshot(
        monotonic_s=10.0,
        sim_time_ns=100,
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        acceleration_local_ned_mps2=None,
        attitude=(1.0, 0.0, 0.0, 0.0),
        body_rates_rps=(0.0, 0.0, 0.0),
        reset_count=1,
        actuator_output=None,
    )
    end = TelemetrySnapshot(
        monotonic_s=10.5,
        sim_time_ns=200,
        position_local_ned_m=(0.5, 0.1, -0.05),
        velocity_local_ned_mps=(0.2, 0.0, 0.0),
        acceleration_local_ned_mps2=None,
        attitude=(1.0, 0.0, 0.0, 0.0),
        body_rates_rps=(0.0, 0.0, 0.0),
        reset_count=1,
        actuator_output=(0.1, 0.2, 0.3, 0.4),
    )

    result = sweep.summarize_case(
        case,
        command_payload=sweep.build_payload(case, start),
        start=start,
        end=end,
        samples=(start, end),
    )

    np.testing.assert_allclose(result.displacement_local_ned_m, [0.5, 0.1, -0.05])
    assert result.dominant_axis == "x"
    assert result.dominant_sign == 1
    assert result.commanded_axis == "x"
    assert result.reset_changed is False
    assert result.actuator_observed is True
    assert result.cross_axis_ratio == 0.2
    assert result.response_gain_m_per_mps_s is not None
    assert result.response_gain_m_per_mps_s > 4.0


def test_body_rate_feedback_uses_discovered_roll_pitch_signs() -> None:
    probe = BodyRateFeedbackProbe(
        FakeBridge(telemetry()),
        BodyRateFeedbackConfig(roll_kp=1.0, pitch_kp=1.0, max_roll_rate_rps=1.0, max_pitch_rate_rps=1.0),
    )

    payload = probe.build_payload(
        {
            "attitude": [1.0, 0.0, 0.0, 0.0],
            "euler_deg": [10.0, -10.0, 0.0],
        }
    )

    assert payload["attitude_type_mask"] == 128
    assert payload["body_rates_rps"][0] > 0.0
    assert payload["body_rates_rps"][1] > 0.0


def test_body_rate_feedback_altitude_hold_uses_ned_down_sign() -> None:
    probe = BodyRateFeedbackProbe(
        FakeBridge(telemetry()),
        BodyRateFeedbackConfig(
            thrust=0.24,
            altitude_hold=True,
            vertical_kp=0.1,
            vertical_kd=0.2,
            min_thrust=0.18,
            max_thrust=0.32,
        ),
    )
    probe._target_z_ned_m = 0.0

    falling_payload = probe.build_payload(
        {
            "attitude": [1.0, 0.0, 0.0, 0.0],
            "euler_deg": [0.0, 0.0, 0.0],
            "position_local_ned_m": [0.0, 0.0, 1.0],
            "velocity_local_ned_mps": [0.0, 0.0, 0.5],
        }
    )
    rising_payload = probe.build_payload(
        {
            "attitude": [1.0, 0.0, 0.0, 0.0],
            "euler_deg": [0.0, 0.0, 0.0],
            "position_local_ned_m": [0.0, 0.0, -1.0],
            "velocity_local_ned_mps": [0.0, 0.0, -0.5],
        }
    )

    assert falling_payload["thrust"] > 0.24
    assert rising_payload["thrust"] < 0.24
    assert falling_payload["thrust_control"]["mode"] == "altitude_hold"


def test_body_rate_feedback_prefers_local_position_velocity_when_available() -> None:
    bridge = FakeBridge(telemetry(velocity=(-1.0, -2.0, 3.0)))
    bridge.latest_local_position = {"velocity_local_ned_mps": (1.0, 2.0, -3.0)}
    probe = BodyRateFeedbackProbe(bridge, BodyRateFeedbackConfig(velocity_source="local_position"))

    sample = probe.snapshot()

    assert sample["velocity_source"] == "local_position_ned"
    assert sample["velocity_local_ned_mps"] == [1.0, 2.0, -3.0]
    assert sample["velocity_debug"]["odometry_velocity_mps"] == [-1.0, -2.0, 3.0]


def test_body_rate_position_guidance_maps_local_error_to_discovered_attitude_axes() -> None:
    guidance = BodyRatePositionGuidance(
        BodyRatePositionGuidanceConfig(position_kp_deg_per_m=2.0, velocity_kd_deg_per_mps=1.0, max_tilt_deg=4.0)
    )

    result = guidance.compute(
        current_position_local_ned_m=(0.0, 0.0, 0.0),
        current_velocity_local_ned_mps=(0.5, -0.5, 0.0),
        target_position_local_ned_m=(2.0, -2.0, -0.4),
    )

    assert result["target_pitch_deg"] > 0.0
    assert result["target_roll_deg"] < 0.0
    assert result["target_pitch_deg"] <= 4.0
    assert result["target_roll_deg"] >= -4.0
