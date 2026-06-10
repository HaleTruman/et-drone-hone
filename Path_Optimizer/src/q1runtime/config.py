from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Q1RuntimeConfig:
    """Configuration for the top-1 vision-to-position-target runtime."""

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
    command_hz: float = 30.0
    command_log_hz: float = 5.0

    vision_device: str = "auto"
    run_landmarker: bool = True
    passthrough_regressor_targets: bool = True
    top_k: int = 1
    min_position_confidence: float = 0.50
    gate_threshold: float = 0.50
    confidence_threshold: float = 0.50
    min_component_area: int = 3
    max_candidates: int = 32

    camera_tilt_deg: float = 20.0
    command_mode: str = "velocity"
    max_approach_speed_mps: float = 0.5
    max_vertical_speed_mps: float = 0.0
    approach_gain_hz: float = 0.8
    arrival_radius_m: float = 0.35
    vertical_mode: str = "hold"
    max_accel_mps2: float = 1.0
    target_hold_s: float = 1.0
    target_filter_max_jump_m: float = 15.0
    target_filter_smoothing_alpha: float = 0.35
    stream_stop_when_target_lost: bool = True
    command_on_missing_target: bool = False
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
    disarm_on_exit: bool = True
    stop_on_exit_s: float = 0.2
    dry_run: bool = False

    log_dir: Path = _project_root() / "logs" / "q1runtime"
    frame_output_dir: Path | None = None

    def with_overrides(self, **values: object) -> "Q1RuntimeConfig":
        return replace(self, **values)
