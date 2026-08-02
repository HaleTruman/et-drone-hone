from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Q1RuntimeConfig:
    """Configuration for the known-good body-rate vision runtime."""

    mavlink_endpoint: str = "udpin:127.0.0.1:14550"
    heartbeat_timeout_s: float = 10.0
    arm_timeout_s: float = 5.0
    wait_for_telemetry_timeout_s: float = 10.0
    arm_on_start: bool = True

    vision_host: str = "0.0.0.0"
    vision_port: int = 5600
    vision_max_buffered_frames: int = 4
    save_raw_frames: bool = False

    run_s: float | None = 60.0
    idle_sleep_s: float = 0.002
    status_interval_s: float = 1.0
    checkpoint_interval_s: float = 5.0
    command_hz: float = 30.0
    command_log_hz: float = 5.0
    control_method: str = "body_rate_guidance"

    vision_device: str = "auto"
    run_landmarker: bool = True
    passthrough_regressor_targets: bool = True
    vision_candidate_top_k: int = 8
    min_position_confidence: float = 0.50
    gate_threshold: float = 0.50
    confidence_threshold: float = 0.50
    min_component_area: int = 3
    max_candidates: int = 32

    camera_tilt_deg: float = 20.0
    target_hold_s: float = 1.0
    target_filter_max_jump_m: float = 15.0
    target_filter_smoothing_alpha: float = 0.35
    target_selection_mode: str = "forward_progress"
    target_selection_min_camera_forward_m: float = 0.10
    target_selection_min_forward_m: float = 0.50
    target_selection_velocity_min_mps: float = 0.50
    target_selection_velocity_source: str = "local_position"
    target_selection_allow_camera_fallback_when_routed: bool = False
    command_on_missing_target: bool = False
    reset_on_start: bool = False
    reset_wait_s: float = 2.0
    reset_ready_timeout_s: float = 20.0
    reset_stable_s: float = 2.5
    reset_stable_max_speed_mps: float = 0.03
    post_reset_delay_s: float = 3.0
    command_transform_path: Path | None = None
    command_invert_x: bool = False
    command_invert_y: bool = False
    command_invert_z: bool = False
    command_swap_xy: bool = False
    safety_enabled: bool = True
    safety_max_horizontal_radius_m: float = 25.0
    safety_min_z_ned_m: float = -8.0
    safety_max_z_ned_m: float = 3.0
    safety_max_velocity_mps: float = 8.0
    safety_telemetry_timeout_s: float = 0.5
    stop_runtime_on_safety_stop: bool = True
    disarm_on_exit: bool = True
    stop_on_exit_s: float = 0.2
    dry_run: bool = False

    body_rate_velocity_source: str = "local_position"
    body_rate_prelevel_s: float = 1.0
    body_rate_prelevel_thrust: float = 0.20
    body_rate_prelevel_target_roll_deg: float = 0.0
    body_rate_prelevel_target_pitch_deg: float = 0.0
    body_rate_base_thrust: float = 0.24
    body_rate_vertical_kp: float = 0.14
    body_rate_vertical_kd: float = 0.12
    body_rate_min_thrust: float = 0.20
    body_rate_max_thrust: float = 0.34
    body_rate_roll_kp: float = 1.2
    body_rate_pitch_kp: float = 2.0
    body_rate_max_roll_rate_rps: float = 0.25
    body_rate_max_pitch_rate_rps: float = 0.5
    body_rate_yaw_rate_rps: float = 0.0
    body_rate_position_kp_deg_per_m: float = 2.0
    body_rate_velocity_kd_deg_per_mps: float = 4.0
    body_rate_max_guidance_tilt_deg: float = 2.0
    body_rate_position_deadband_m: float = 0.05
    vision_target_max_horizontal_m: float = 1.0
    vision_target_max_up_m: float = 0.40
    vision_target_max_down_m: float = 0.10
    body_rate_target_stale_abort_s: float = 2.0

    log_dir: Path = _project_root() / "logs" / "q1runtime"
    frame_output_dir: Path | None = None
