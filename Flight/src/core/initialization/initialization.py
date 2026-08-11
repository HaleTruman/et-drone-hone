"""
The Emperor is our guiding light, a beacon of hope for humanity in a galaxy of darkness.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import time

import yaml

from core.control.autipilot import AutiPilot, AutiPilotGains
from core.logging import Logger
from core.logging.obs import OBSRecorder
from core.modes.system_mode import SystemModeManager
from mapping.gates import GateMap
from sensing.odometry import (
    GyroSpikeFilterConfig,
    KalmanFilterConfig,
    OpenCvMonocularVioProvider,
    VehicleStateEstimator,
    VioCorrectionConfig,
    VioFrontendConfig,
)
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

# DEFINE START TIME
DEFINED_START_TIME_NS: float = time.perf_counter_ns()
FLIGHT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = FLIGHT_ROOT / "config" / "flight.yaml"
CONFIG_PATH = Path(os.environ.get("FLIGHT_CONFIG_PATH", DEFAULT_CONFIG_PATH))


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Flight config must be a YAML mapping: {path}")
    return config


def _config_value(*keys: str) -> object:
    value: object = _CONFIG
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(keys)
            raise KeyError(f"Missing Flight config key: {dotted}")
        value = value[key]
    return value


def _config_path(*keys: str) -> Path:
    value = str(_config_value(*keys))
    path = Path(value)
    return path if path.is_absolute() else (FLIGHT_ROOT / path).resolve()


_CONFIG = _load_config(CONFIG_PATH)

# Simulator and network endpoints.
MAVLINK_ENDPOINT = str(_config_value("simulator", "mavlink_endpoint"))
SIM_RUNTIME = str(_config_value("simulator", "sim_runtime"))
VISION_HOST = str(_config_value("vision", "host"))
VISION_PORT = int(_config_value("vision", "port"))

# Control loop timing.
INNER_LOOP_HZ = float(_config_value("loops", "inner_loop_hz"))
OUTER_LOOP_HZ = float(_config_value("loops", "outer_loop_hz"))
RUN_S: float | None = (
    None
    if _config_value("loops", "run_s") is None
    else float(_config_value("loops", "run_s"))
)

# Startup, reset, and arming timeouts.
HEARTBEAT_TIMEOUT_S = float(_config_value("startup", "heartbeat_timeout_s"))
STARTUP_DATA_TIMEOUT_S = float(_config_value("startup", "startup_data_timeout_s"))
IMU_INIT_TIMEOUT_S = float(_config_value("startup", "imu_init_timeout_s"))
GATE_MAP_INIT_TIMEOUT_S = float(_config_value("startup", "gate_map_init_timeout_s"))
POST_RESET_DELAY_S = float(_config_value("startup", "post_reset_delay_s"))
ARM_TIMEOUT_S = float(_config_value("startup", "arm_timeout_s"))

# Gate mapping
GATE_MERGE_DISTANCE_M = float(_config_value("gate_map", "merge_distance_m"))
REQUIRED_MINIMUM_OBSERVATION_COUNT = int(_config_value("gate_map", "required_minimum_observation_count"))
GATE_LOCKOUT_COUNT = int(_config_value("gate_map", "lock_observation_count"))
GATE_MAX_OBSERVATION_DISTANCE_M = float(_config_value("gate_map", "max_observation_distance_m"))
GATE_TARGET_CANDIDATE_MIN_OBSERVATION_COUNT = int(_config_value("gate_map", "target_candidate_min_observation_count"))
GATE_TARGET_CANDIDATE_MIN_POSITION_CONFIDENCE = float(_config_value("gate_map", "target_candidate_min_position_confidence"))
GATE_TARGET_CANDIDATE_MAX_AVERAGE_RESIDUAL_M = float(_config_value("gate_map", "target_candidate_max_average_residual_m"))
GATE_TARGET_CANDIDATE_MAX_DISTANCE_M = float(_config_value("gate_map", "target_candidate_max_distance_m"))
GATE_PASSED_DISTANCE_M = float(_config_value("gate_map", "gate_passed_distance_m"))

# Control mode and safety envelope.
ALLOW_FLIGHT = bool(_config_value("flight", "allow_flight"))

# MAVLink attitude target scaling.
ATTITUDE_ERROR_QUATERNION_ROLL_SCALE = float(_config_value("mavlink_attitude_target", "error_quaternion_roll_scale"))
ATTITUDE_ERROR_QUATERNION_PITCH_SCALE = float(_config_value("mavlink_attitude_target", "error_quaternion_pitch_scale"))
ATTITUDE_ERROR_QUATERNION_YAW_SCALE = float(_config_value("mavlink_attitude_target", "error_quaternion_yaw_scale"))

# IMU gyro spike rejection.
GYRO_SPIKE_FILTER_ENABLED = bool(_config_value("gyro_spike_filter", "enabled"))
GYRO_SPIKE_MAX_RATE_RPS = float(_config_value("gyro_spike_filter", "max_rate_rps"))
GYRO_SPIKE_MAX_DELTA_RPS = float(_config_value("gyro_spike_filter", "max_delta_rps"))
GYRO_SPIKE_MAX_REJECTIONS = int(_config_value("gyro_spike_filter", "max_rejections"))

# Speed limits and turn-speed shaping.
AUTIPILOT_MAX_SPEED_MPS = float(_config_value("autipilot", "v_max_mps"))
AUTIPILOT_MIN_SPEED_MPS = float(_config_value("autipilot", "v_min_mps"))
AUTIPILOT_MAX_LATERAL_ACCELERATION_MPS2 = float(_config_value("autipilot", "a_lat_max_mps2"))
AUTIPILOT_MEDIUM_TURN_ANGLE_DEG = float(_config_value("autipilot", "medium_turn_angle_deg"))
AUTIPILOT_TIGHT_TURN_ANGLE_DEG = float(_config_value("autipilot", "tight_turn_angle_deg"))
AUTIPILOT_MEDIUM_TURN_LATERAL_ACCELERATION_MPS2 = float(_config_value("autipilot", "medium_turn_a_lat_mps2"))
AUTIPILOT_TIGHT_TURN_LATERAL_ACCELERATION_MPS2 = float(_config_value("autipilot", "tight_turn_a_lat_mps2"))

# Speed control near gates.
AUTIPILOT_SPEED_GAIN = float(_config_value("autipilot", "kp_speed"))
AUTIPILOT_NEAR_GATE_SPEED_GAIN_NEAR_M = float(_config_value("autipilot", "near_gate_speed_gain_near_m"))
AUTIPILOT_NEAR_GATE_SPEED_GAIN_FAR_M = float(_config_value("autipilot", "near_gate_speed_gain_far_m"))
AUTIPILOT_NEAR_GATE_SPEED_GAIN_MIN_SCALE = float(_config_value("autipilot", "near_gate_speed_gain_min_scale"))
AUTIPILOT_NEAR_GATE_SPEED_GAIN_RAMP_S = float(_config_value("autipilot", "near_gate_speed_gain_ramp_s"))

# Direction, lateral, and vertical gains.
AUTIPILOT_DIRECTION_GAIN = float(_config_value("autipilot", "kp_dir"))
AUTIPILOT_DIRECTION_DAMPING = float(_config_value("autipilot", "kd_dir"))
AUTIPILOT_LATERAL_POSITION_GAIN = float(_config_value("autipilot", "kp_lat"))
AUTIPILOT_LATERAL_DAMPING = float(_config_value("autipilot", "kd_lat"))
AUTIPILOT_VERTICAL_POSITION_GAIN = float(_config_value("autipilot", "kp_z"))
AUTIPILOT_VERTICAL_DAMPING = float(_config_value("autipilot", "kd_z"))

# Next-gate lookahead and fly-through behavior.
AUTIPILOT_LOOKAHEAD_NEAR_M = float(_config_value("autipilot", "lookahead_near_m"))
AUTIPILOT_LOOKAHEAD_FAR_M = float(_config_value("autipilot", "lookahead_far_m"))
AUTIPILOT_FLY_THROUGH_ENABLED = bool(_config_value("autipilot", "fly_through_enabled"))
AUTIPILOT_FLY_THROUGH_TURN_ANGLE_START_DEG = float(_config_value("autipilot", "fly_through_turn_angle_start_deg"))
AUTIPILOT_FLY_THROUGH_TURN_ANGLE_FULL_DEG = float(_config_value("autipilot", "fly_through_turn_angle_full_deg"))
AUTIPILOT_FLY_THROUGH_NEXT_GATE_DISTANCE_START_M = float(_config_value("autipilot", "fly_through_next_gate_distance_start_m"))
AUTIPILOT_FLY_THROUGH_NEXT_GATE_DISTANCE_FULL_M = float(_config_value("autipilot", "fly_through_next_gate_distance_full_m"))
AUTIPILOT_FLY_THROUGH_LOOKAHEAD_MIN_SCALE = float(_config_value("autipilot", "fly_through_lookahead_min_scale"))

# Gate crossing and post-cross recovery.
AUTIPILOT_GATE_CROSS_VELOCITY_DAMPING = float(_config_value("autipilot", "gate_cross_velocity_damping"))
AUTIPILOT_GATE_CROSS_VELOCITY_CANCEL_NEAR_M = float(_config_value("autipilot", "gate_cross_velocity_cancel_near_m"))
AUTIPILOT_GATE_CROSS_VELOCITY_CANCEL_FAR_M = float(_config_value("autipilot", "gate_cross_velocity_cancel_far_m"))
AUTIPILOT_APPROACH_GAIN_MIN_SCALE = float(_config_value("autipilot", "approach_gain_min_scale"))
AUTIPILOT_POST_CROSS_TURN_SCALE = float(_config_value("autipilot", "post_cross_turn_scale"))
AUTIPILOT_POST_CROSS_RAMP_DISTANCE_M = float(_config_value("autipilot", "post_cross_ramp_distance_m"))

# Launch, thrust, and command limiting.
AUTIPILOT_LAUNCH_SPEED_RAMP_S = float(_config_value("autipilot", "launch_speed_ramp_s"))
AUTIPILOT_MAX_SPECIFIC_THRUST_MPS2 = float(_config_value("autipilot", "max_specific_thrust_mps2"))
AUTIPILOT_MIN_NORMALIZED_THRUST = float(_config_value("autipilot", "min_normalized_thrust"))
AUTIPILOT_MAX_NORMALIZED_THRUST = float(_config_value("autipilot", "max_normalized_thrust"))
AUTIPILOT_MAX_COMMANDED_ACCELERATION_MPS2 = float(_config_value("autipilot", "max_commanded_acceleration_mps2"))
AUTIPILOT_GATE_SWITCH_ACCELERATION_RAMP_S = float(_config_value("autipilot", "gate_switch_acceleration_ramp_s"))
AUTIPILOT_MAX_COMMANDED_JERK_MPS3 = float(_config_value("autipilot", "max_commanded_jerk_mps3"))

# Output and recording.
CREATE_VIDEO = bool(_config_value("recording", "create_video"))
RECORD_SCREEN = bool(_config_value("recording", "record_screen"))

# Visual odometry configuration.
ENABLE_VIO = bool(_config_value("vio", "enabled"))
VIO_CAMERA_HORIZONTAL_FOV_DEG = float(_config_value("vio", "camera_horizontal_fov_deg"))
VIO_CAMERA_TILT_DEG = float(_config_value("vio", "camera_tilt_deg"))
VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M = tuple(
    float(value)
    for value in _config_value("vio", "body_to_camera_translation_body_frd_m")
)
VIO_POSITION_ALPHA = float(_config_value("vio", "position_alpha"))
VIO_VELOCITY_ALPHA = float(_config_value("vio", "velocity_alpha"))
VIO_ATTITUDE_ALPHA = float(_config_value("vio", "attitude_alpha"))

# State-estimator Kalman filter configuration.
ENABLE_KALMAN_FILTER = bool(_config_value("kalman", "enabled"))
KALMAN_INITIAL_POSITION_VARIANCE_M2 = float(_config_value("kalman", "initial_position_variance_m2"))
KALMAN_INITIAL_VELOCITY_VARIANCE_M2PS2 = float(_config_value("kalman", "initial_velocity_variance_m2ps2"))
KALMAN_ACCELERATION_PROCESS_NOISE_MPS2 = float(_config_value("kalman", "acceleration_process_noise_mps2"))
KALMAN_VIO_POSITION_MEASUREMENT_VARIANCE_M2 = float(_config_value("kalman", "vio_position_measurement_variance_m2"))
KALMAN_VIO_VELOCITY_MEASUREMENT_VARIANCE_M2PS2 = float(_config_value("kalman", "vio_velocity_measurement_variance_m2ps2"))
KALMAN_MIN_MEASUREMENT_CONFIDENCE = float(_config_value("kalman", "min_measurement_confidence"))

# Initialization defaults.
VISION_PERCEPTION_BACKEND = str(_config_value("vision", "perception_backend"))
VISION_EXECUTOR_MAX_WORKERS = int(_config_value("vision", "executor_max_workers"))
VISION_EXECUTOR_THREAD_PREFIX = str(_config_value("vision", "executor_thread_prefix"))

# Logging.
RUNS_ROOT = _config_path("logging", "runs_root")


def _initialization_constants() -> dict[str, object]:
    return {
        name: _constant_log_value(value)
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_")
    }


def _constant_log_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_constant_log_value(item) for item in value]
    if isinstance(value, list):
        return [_constant_log_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _constant_log_value(item)
            for key, item in value.items()
        }
    return value


@dataclass(frozen=True)
class RuntimeSettings:
    run_dir: Path
    log_path: Path
    inner_loop_hz: float
    outer_loop_hz: float
    run_s: float | None
    heartbeat_timeout_s: float
    startup_data_timeout_s: float
    imu_init_timeout_s: float
    gate_map_init_timeout_s: float
    post_reset_delay_s: float
    arm_timeout_s: float
    allow_flight: bool
    create_video: bool
    record_screen: bool
    enable_vio: bool


def initialize() -> tuple[
    RuntimeSettings,
    Logger,
    VehicleStateEstimator,
    OpenCvMonocularVioProvider,
    MavlinkClient,
    VisionStreamReceiver,
    VisionPerceptionService,
    ThreadPoolExecutor,
    OBSRecorder,
    SystemModeManager,
    GateMap,
    AutiPilot,
]:
    run_dir = Logger.timestamped_dir(RUNS_ROOT)
    log_path = run_dir / "run.json"
    settings = RuntimeSettings(
        run_dir=run_dir,
        log_path=log_path,
        inner_loop_hz=INNER_LOOP_HZ,
        outer_loop_hz=OUTER_LOOP_HZ,
        run_s=RUN_S,
        heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S,
        startup_data_timeout_s=STARTUP_DATA_TIMEOUT_S,
        imu_init_timeout_s=IMU_INIT_TIMEOUT_S,
        gate_map_init_timeout_s=GATE_MAP_INIT_TIMEOUT_S,
        post_reset_delay_s=POST_RESET_DELAY_S,
        arm_timeout_s=ARM_TIMEOUT_S,
        allow_flight=ALLOW_FLIGHT,
        create_video=CREATE_VIDEO,
        record_screen=RECORD_SCREEN,
        enable_vio=ENABLE_VIO,
    )
    logger = Logger(
        {
            "scenario": "live_stream_minimal",
            "sim_runtime": SIM_RUNTIME,
            "mavlink_endpoint": MAVLINK_ENDPOINT,
            "vision_host": VISION_HOST,
            "vision_port": VISION_PORT,
            "loop_hz": INNER_LOOP_HZ,
            "inner_loop_hz": INNER_LOOP_HZ,
            "outer_loop_hz": OUTER_LOOP_HZ,
            "initialization_constants": _initialization_constants(),
        }
    )
    logger.log_event("initialization_constants", constants=_initialization_constants())

    vehicle_state_estimator = VehicleStateEstimator(
        vio_config=VioCorrectionConfig(
            position_alpha=VIO_POSITION_ALPHA,
            velocity_alpha=VIO_VELOCITY_ALPHA,
            attitude_alpha=VIO_ATTITUDE_ALPHA,
        ),
        kalman_config=KalmanFilterConfig(
            enabled=ENABLE_KALMAN_FILTER,
            initial_position_variance_m2=KALMAN_INITIAL_POSITION_VARIANCE_M2,
            initial_velocity_variance_m2ps2=KALMAN_INITIAL_VELOCITY_VARIANCE_M2PS2,
            acceleration_process_noise_mps2=KALMAN_ACCELERATION_PROCESS_NOISE_MPS2,
            vio_position_measurement_variance_m2=KALMAN_VIO_POSITION_MEASUREMENT_VARIANCE_M2,
            vio_velocity_measurement_variance_m2ps2=KALMAN_VIO_VELOCITY_MEASUREMENT_VARIANCE_M2PS2,
            min_measurement_confidence=KALMAN_MIN_MEASUREMENT_CONFIDENCE,
        ),
        gyro_spike_filter_config=GyroSpikeFilterConfig(
            enabled=GYRO_SPIKE_FILTER_ENABLED,
            max_rate_rps=GYRO_SPIKE_MAX_RATE_RPS,
            max_delta_rps=GYRO_SPIKE_MAX_DELTA_RPS,
            max_rejections=GYRO_SPIKE_MAX_REJECTIONS,
        ),
    )

    vio_provider = OpenCvMonocularVioProvider(
        frontend_config=VioFrontendConfig(horizontal_fov_deg=VIO_CAMERA_HORIZONTAL_FOV_DEG),
        body_to_camera_translation_body_frd_m=VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M,
        camera_tilt_deg=VIO_CAMERA_TILT_DEG,
    )

    mavlink_client = MavlinkClient(endpoint=MAVLINK_ENDPOINT, sim_runtime=SIM_RUNTIME)
    vision_rx = VisionStreamReceiver(
        host=VISION_HOST,
        port=VISION_PORT,
        output_dir=run_dir / "vision_frames",
    )
    vision_perception = VisionPerceptionService(
        VisionPerceptionConfig(backend=VISION_PERCEPTION_BACKEND)
    )
    vision_executor = ThreadPoolExecutor(
        max_workers=VISION_EXECUTOR_MAX_WORKERS,
        thread_name_prefix=VISION_EXECUTOR_THREAD_PREFIX,
    )
    obs_recorder = OBSRecorder(run_dir)
    system_mode_manager = SystemModeManager()
    gate_map = GateMap(
        merge_distance_m=GATE_MERGE_DISTANCE_M,
        required_minimum_observation_count=REQUIRED_MINIMUM_OBSERVATION_COUNT,
        lock_observation_count=GATE_LOCKOUT_COUNT,
        gate_passed_distance_m=GATE_PASSED_DISTANCE_M,
        max_observation_distance_m=GATE_MAX_OBSERVATION_DISTANCE_M,
        candidate_target_min_observation_count=(
            GATE_TARGET_CANDIDATE_MIN_OBSERVATION_COUNT
        ),
        candidate_target_min_position_confidence=(
            GATE_TARGET_CANDIDATE_MIN_POSITION_CONFIDENCE
        ),
        candidate_target_max_average_residual_m=(
            GATE_TARGET_CANDIDATE_MAX_AVERAGE_RESIDUAL_M
        ),
        candidate_target_max_distance_m=GATE_TARGET_CANDIDATE_MAX_DISTANCE_M,
    )

    autipilot = AutiPilot(
        AutiPilotGains(
            # Speed limits and turn-speed shaping.
            v_max_mps=AUTIPILOT_MAX_SPEED_MPS,
            v_min_mps=AUTIPILOT_MIN_SPEED_MPS,
            a_lat_max_mps2=AUTIPILOT_MAX_LATERAL_ACCELERATION_MPS2,
            medium_turn_angle_deg=AUTIPILOT_MEDIUM_TURN_ANGLE_DEG,
            tight_turn_angle_deg=AUTIPILOT_TIGHT_TURN_ANGLE_DEG,
            medium_turn_a_lat_mps2=AUTIPILOT_MEDIUM_TURN_LATERAL_ACCELERATION_MPS2,
            tight_turn_a_lat_mps2=AUTIPILOT_TIGHT_TURN_LATERAL_ACCELERATION_MPS2,

            # Speed control near gates.
            kp_speed=AUTIPILOT_SPEED_GAIN,
            near_gate_speed_gain_near_m=AUTIPILOT_NEAR_GATE_SPEED_GAIN_NEAR_M,
            near_gate_speed_gain_far_m=AUTIPILOT_NEAR_GATE_SPEED_GAIN_FAR_M,
            near_gate_speed_gain_min_scale=AUTIPILOT_NEAR_GATE_SPEED_GAIN_MIN_SCALE,
            near_gate_speed_gain_ramp_s=AUTIPILOT_NEAR_GATE_SPEED_GAIN_RAMP_S,

            # Direction, lateral, and vertical gains.
            kp_dir=AUTIPILOT_DIRECTION_GAIN,
            kd_dir=AUTIPILOT_DIRECTION_DAMPING,
            kp_lat=AUTIPILOT_LATERAL_POSITION_GAIN,
            kd_lat=AUTIPILOT_LATERAL_DAMPING,
            kp_z=AUTIPILOT_VERTICAL_POSITION_GAIN,
            kd_z=AUTIPILOT_VERTICAL_DAMPING,

            # Next-gate lookahead and fly-through behavior.
            lookahead_near_m=AUTIPILOT_LOOKAHEAD_NEAR_M,
            lookahead_far_m=AUTIPILOT_LOOKAHEAD_FAR_M,
            fly_through_turn_angle_start_deg=(
                AUTIPILOT_FLY_THROUGH_TURN_ANGLE_START_DEG
            ),
            fly_through_turn_angle_full_deg=(
                AUTIPILOT_FLY_THROUGH_TURN_ANGLE_FULL_DEG
            ),
            fly_through_next_gate_distance_start_m=(
                AUTIPILOT_FLY_THROUGH_NEXT_GATE_DISTANCE_START_M
            ),
            fly_through_next_gate_distance_full_m=(
                AUTIPILOT_FLY_THROUGH_NEXT_GATE_DISTANCE_FULL_M
            ),
            fly_through_lookahead_min_scale=(
                AUTIPILOT_FLY_THROUGH_LOOKAHEAD_MIN_SCALE
                if AUTIPILOT_FLY_THROUGH_ENABLED
                else 1.0
            ),
            gate_cross_velocity_damping=AUTIPILOT_GATE_CROSS_VELOCITY_DAMPING,
            gate_cross_velocity_cancel_near_m=(
                AUTIPILOT_GATE_CROSS_VELOCITY_CANCEL_NEAR_M
            ),
            gate_cross_velocity_cancel_far_m=(
                AUTIPILOT_GATE_CROSS_VELOCITY_CANCEL_FAR_M
            ),

            # Gate crossing and post-cross recovery.
            approach_gain_min_scale=AUTIPILOT_APPROACH_GAIN_MIN_SCALE,
            post_cross_turn_scale=AUTIPILOT_POST_CROSS_TURN_SCALE,
            post_cross_ramp_distance_m=AUTIPILOT_POST_CROSS_RAMP_DISTANCE_M,

            # Launch, thrust, and command limiting.
            launch_speed_ramp_s=AUTIPILOT_LAUNCH_SPEED_RAMP_S,
            max_specific_thrust_mps2=AUTIPILOT_MAX_SPECIFIC_THRUST_MPS2,
            min_normalized_thrust=AUTIPILOT_MIN_NORMALIZED_THRUST,
            max_normalized_thrust=AUTIPILOT_MAX_NORMALIZED_THRUST,
            max_commanded_acceleration_mps2=AUTIPILOT_MAX_COMMANDED_ACCELERATION_MPS2,
            gate_switch_acceleration_ramp_s=AUTIPILOT_GATE_SWITCH_ACCELERATION_RAMP_S,
            max_commanded_jerk_mps3=AUTIPILOT_MAX_COMMANDED_JERK_MPS3,
        ),
        error_quaternion_scales=(
            ATTITUDE_ERROR_QUATERNION_ROLL_SCALE,
            ATTITUDE_ERROR_QUATERNION_PITCH_SCALE,
            ATTITUDE_ERROR_QUATERNION_YAW_SCALE,
        ),
    )

    return (
        settings,
        logger,
        vehicle_state_estimator,
        vio_provider,
        mavlink_client,
        vision_rx,
        vision_perception,
        vision_executor,
        obs_recorder,
        system_mode_manager,
        gate_map,
        autipilot,
    )
