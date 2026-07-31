from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from autonomy.pathing import PathManager
from core.control.attitude import AttitudeController
from core.control.carrot import CarrotController
from core.control.hover.controller import HoverController
from core.control.path_follower import GeometricPathFollower
from core.logging import Logger
from core.logging.obs import OBSRecorder
from core.modes.system_mode import SystemModeManager
from mapping.gates import GateMap
from sensing.odometry import (
    KalmanFilterConfig,
    OpenCvMonocularVioProvider,
    VehicleStateEstimator,
    VioCorrectionConfig,
    VioFrontendConfig,
)
from sensing.telemetry import MavlinkClient
from sensing.vision import VisionStreamReceiver
from sensing.vision.service import VisionPerceptionConfig, VisionPerceptionService

# Simulator and network endpoints.
MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"  # selects the MAVLink UDP endpoint used to talk to the simulator or vehicle bridge.
SIM_RUNTIME = "VQ_2"  # identifies the simulator/runtime profile passed into the MAVLink client.
VISION_HOST = "0.0.0.0"  # is the local interface where the vision receiver listens for incoming frames.
VISION_PORT = 5600  # is the UDP/TCP port used by the vision frame receiver.

# Control loop timing.
INNER_LOOP_HZ = 100.0  # is the fast control loop rate used for state updates and attitude commands.
OUTER_LOOP_HZ = 30.0  # is the slower loop rate used for vision, path planning, and path-following targets.
RUN_S: float | None = None  # optionally limits flight duration in seconds; None runs until interrupted or finished.

# Startup, reset, and arming timeouts.
HEARTBEAT_TIMEOUT_S = 120.0  # is how long startup waits for the MAVLink heartbeat before failing.
STARTUP_DATA_TIMEOUT_S = 5.0  # is how long startup waits for fresh telemetry and vision data.
IMU_INIT_TIMEOUT_S = 1.5  # is the stationary sample window used for initial IMU bias estimation.
GATE_MAP_INIT_TIMEOUT_S = 1.5  # is the startup window used to collect initial vision observations and build a path.
POST_RESET_DELAY_S = 0.6  # gives the simulator time to settle after a reset command.
ARM_TIMEOUT_S = 5.0  # is how long the system waits for the vehicle to arm successfully.

# Vision filtering and path planning.
PLANNING_MODE = "test_path"  # chooses the PathManager strategy: test_path, center_targets, or gate_map.
PLANNING_GATE_COUNT = 2  # limits how many upcoming gates are included in each path plan.
EXCLUSION_DISTANCE = 2.0  # ignores gates that are too close to the current vehicle position.
GATE_MAX_PLANNING_DISTANCE_M = 40.0  # ignores gates farther than this from the current vehicle position.
GATE_PASSED_DISTANCE_M = 2.0  # treats gates closer than this as already passed for planning purposes.
GATE_CENTER_TOLERANCE_M = 0.15  # is the allowed path distance from each selected gate center.
SPLINE_CORNER_TIGHTNESS = 0.75  # controls how tightly generated splines follow corner anchor points.
ADAPTIVE_SPLINE_TIGHTNESS = True  # enables automatic corner tightness changes based on segment geometry.
DISTANT_SPLINE_CORNER_TIGHTNESS = 0.10  # is the looser spline tightness used for distant or gentle turns.
MIN_SPLINE_CORNER_TIGHTNESS = 0.55  # is the lower bound for adaptive spline tightness near turns.
MAX_SPLINE_CORNER_TIGHTNESS = 0.95  # is the upper bound for adaptive spline tightness near sharp turns.
GENTLE_TURN_ANGLE_DEG = 20.0  # defines the turn angle below which corners are treated as gentle.
SHARP_TURN_ANGLE_DEG = 70.0  # defines the turn angle at which corners receive maximum adaptive tightness.
SHORT_SEGMENT_REFERENCE_M = 12.0  # marks the segment length where nearby turns become more tightly constrained.
LONG_SEGMENT_REFERENCE_M = 25.0  # marks the segment length where distance-based spline tightening fades out.

# Control mode and safety envelope.
CONTROL_METHOD = "geometric_path_follower"  # selects which path-following controller produces the attitude target.
FAILSAFE_DISTANCE = 10  # is the maximum allowed cross-track path error before ending racing flight.
ALLOW_FLIGHT = True  # enables sending flight commands when the system mode allows it.

# Path preview distances.
CARROT_LOOKAHEAD_M = 1.4  # is the lookahead distance used by the simpler carrot controller.
SPEED_LOOKAHEAD_M = 15  # is how far ahead curvature is checked for speed planning.
GEOMETRIC_LOOKAHEAD_M = 2.0  # is the lookahead distance used for geometric follower heading preview.

# Geometric path-following feedback.
GEOMETRIC_CROSS_TRACK_GAIN = 8.0  # scales position correction back toward the path.
GEOMETRIC_CROSS_TRACK_DAMPING = 4.0  # scales velocity damping perpendicular to the path.
GEOMETRIC_ACCELERATION_FILTER_ALPHA = 1.0  # smooths outer-loop acceleration commands; 1.0 disables smoothing.

# Speed planner.
GEOMETRIC_MAX_SPEED_MPS = 3  # is the maximum along-track speed requested by the geometric follower.
GEOMETRIC_MAX_LATERAL_ACCELERATION_MPS2 = 50.0  # limits speed in curves based on available lateral acceleration.
GEOMETRIC_CURVATURE_SPEED_DEADBAND = 3.5  # ignores small curvature when computing curve-limited speed.
GEOMETRIC_CURVATURE_SPEED_RAMP = 0.5  # controls how quickly commanded speed drops as curvature increases.

# Curvature feed-forward.
GEOMETRIC_CURVATURE_FEEDFORWARD_GAIN = 3.0  # scales proactive acceleration into upcoming turns.
GEOMETRIC_CURVATURE_FEEDFORWARD_MAX_ACCELERATION_MPS2 = 25.0  # caps proactive turn acceleration.

# Thrust and acceleration limits.
GEOMETRIC_HOVER_THRUST = 0.265  # is the normalized thrust command expected to hold hover.
GEOMETRIC_MAX_COMMANDED_ACCELERATION_MPS2 = 50  # caps the total desired acceleration magnitude.
GEOMETRIC_MAX_UPWARD_ACCELERATION_MPS2 = 15.0  # caps upward commanded acceleration in local-NED terms.
GEOMETRIC_MAX_DOWNWARD_ACCELERATION_MPS2 = 10.0  # caps downward commanded acceleration in local-NED terms.
GEOMETRIC_MAX_TILT_DEG = 60.0  # caps the tilt implied by the desired acceleration command.

# Output and recording.
CREATE_VIDEO = False  # enables post-run MP4 generation from logged visual outputs.
RECORD_SCREEN = False  # enables OBS screen recording during the run.

# Visual odometry configuration.
ENABLE_VIO = False  # enables monocular visual-inertial odometry corrections.
VIO_CAMERA_HORIZONTAL_FOV_DEG = 90.0  # is the camera horizontal field of view used by the VIO frontend.
VIO_CAMERA_TILT_DEG = 20.0  # is the camera pitch angle relative to the vehicle body.
VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M = (0.0, 0.0, 0.0)  # is the camera offset from the body origin in FRD coordinates.
VIO_POSITION_ALPHA = 0.02  # controls how strongly VIO position corrections affect the state estimate.
VIO_VELOCITY_ALPHA = 0.05  # controls how strongly VIO velocity corrections affect the state estimate.
VIO_ATTITUDE_ALPHA = 0.03  # controls how strongly VIO attitude corrections affect the state estimate.

# State-estimator Kalman filter configuration.
ENABLE_KALMAN_FILTER = False  # enables Kalman filtering for position and velocity estimation.
KALMAN_INITIAL_POSITION_VARIANCE_M2 = 0.0  # sets initial position uncertainty for the Kalman state.
KALMAN_INITIAL_VELOCITY_VARIANCE_M2PS2 = 0.0  # sets initial velocity uncertainty for the Kalman state.
KALMAN_ACCELERATION_PROCESS_NOISE_MPS2 = 1.0  # controls how much IMU acceleration uncertainty grows covariance.
KALMAN_VIO_POSITION_MEASUREMENT_VARIANCE_M2 = 0.25  # sets trusted variance for VIO position measurements.
KALMAN_VIO_VELOCITY_MEASUREMENT_VARIANCE_M2PS2 = 1.0  # sets trusted variance for VIO velocity measurements.
KALMAN_MIN_MEASUREMENT_CONFIDENCE = 0.05  # prevents low-confidence VIO updates from becoming infinitely noisy.

# Initialization defaults.
VISION_PERCEPTION_BACKEND = "deterministic_v3"  # selects the gate perception implementation used for vision frames.
VISION_EXECUTOR_MAX_WORKERS = 1  # controls the number of background workers for vision processing.
VISION_EXECUTOR_THREAD_PREFIX = "vision"  # names background vision worker threads for debugging.
PATH_SPACING_M = 0.25  # is the waypoint spacing used when sampling generated paths.
PATH_MAX_POINTS = 1000  # caps the number of sampled waypoints kept in a generated path.
TEST_PATH_LENGTH_M = 100.0  # is the length of the startup straight-line test path.
TEST_PATH_POINT_COUNT = 200  # is the number of samples used for the startup straight-line test path.
TEST_PATH_UP_DOWN_ANGLE_DEG = 2.5  # tilts the startup straight-line test path vertically.
TEST_PATH_LEFT_RIGHT_ANGLE_DEG = 0.0  # tilts the startup straight-line test path horizontally.
ATTITUDE_ROLL_GAIN = 1.5  # scales roll attitude error into commanded body rate.
ATTITUDE_PITCH_GAIN = 1.5  # scales pitch attitude error into commanded body rate.
ATTITUDE_YAW_GAIN = 0.6  # scales yaw attitude error into commanded body rate.
ATTITUDE_DAMPING = 0.15  # subtracts current body-rate feedback from attitude commands.
ATTITUDE_MAX_BODY_RATE_RPS = 10.0  # caps commanded body rates from the attitude controller.
CARROT_MAX_SPEED_MPS = 10  # is the maximum speed used by the simpler carrot controller.
CARROT_POSITION_GAIN = 5.5  # scales position error in the simpler carrot controller.
CARROT_VELOCITY_GAIN = 0.75  # scales velocity damping in the simpler carrot controller.
CARROT_INITIAL_THRUST = 0.265  # is the baseline normalized thrust for the simpler carrot controller.
HOVER_LATERAL_VELOCITY_GAIN = 2.5  # damps horizontal velocity when the finish hover controller is active.
HOVER_VERTICAL_VELOCITY_GAIN = 0.18  # damps vertical velocity when the finish hover controller is active.
HOVER_VERTICAL_ACCELERATION_GAIN = 0.035  # scales vertical acceleration feedback in the finish hover controller.

# Logging.
RUNS_ROOT = Path(__file__).resolve().parents[3] / "logs" / "runs"  # is the root directory where timestamped run logs are created.


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
    planning_mode: str
    control_method: str
    failsafe_distance_m: float
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
    PathManager,
    AttitudeController,
    CarrotController,
    GeometricPathFollower,
    HoverController,
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
        planning_mode=PLANNING_MODE,
        control_method=CONTROL_METHOD,
        failsafe_distance_m=FAILSAFE_DISTANCE,
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
            "control_method": CONTROL_METHOD,
            "planning_mode": PLANNING_MODE,
        }
    )

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
        )
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
    gate_map = GateMap()

    path_manager = PathManager(
        max_gates=PLANNING_GATE_COUNT,
        exclusion_distance_m=EXCLUSION_DISTANCE,
        max_gate_distance_m=GATE_MAX_PLANNING_DISTANCE_M,
        passed_gate_distance_m=GATE_PASSED_DISTANCE_M,
        gate_center_tolerance_m=GATE_CENTER_TOLERANCE_M,
        spline_corner_tightness=SPLINE_CORNER_TIGHTNESS,
        adaptive_spline_tightness=ADAPTIVE_SPLINE_TIGHTNESS,
        distant_spline_corner_tightness=DISTANT_SPLINE_CORNER_TIGHTNESS,
        min_spline_corner_tightness=MIN_SPLINE_CORNER_TIGHTNESS,
        max_spline_corner_tightness=MAX_SPLINE_CORNER_TIGHTNESS,
        gentle_turn_angle_deg=GENTLE_TURN_ANGLE_DEG,
        sharp_turn_angle_deg=SHARP_TURN_ANGLE_DEG,
        short_segment_reference_m=SHORT_SEGMENT_REFERENCE_M,
        long_segment_reference_m=LONG_SEGMENT_REFERENCE_M,
        planning_mode=PLANNING_MODE,
        spacing_m=PATH_SPACING_M,
        max_points=PATH_MAX_POINTS,
    )

    path_manager.build_straight_line(
        length_m=TEST_PATH_LENGTH_M,
        point_count=TEST_PATH_POINT_COUNT,
        up_down_angle_deg=TEST_PATH_UP_DOWN_ANGLE_DEG,
        left_right_angle_deg=TEST_PATH_LEFT_RIGHT_ANGLE_DEG,
    )

    attitude_controller = AttitudeController(
        roll_gain=ATTITUDE_ROLL_GAIN,
        pitch_gain=ATTITUDE_PITCH_GAIN,
        yaw_gain=ATTITUDE_YAW_GAIN,
        damping=ATTITUDE_DAMPING,
        max_body_rate_rps=ATTITUDE_MAX_BODY_RATE_RPS,
    )

    carrot_controller = CarrotController(
        max_speed_mps=CARROT_MAX_SPEED_MPS,
        lookahead_m=CARROT_LOOKAHEAD_M,
        speed_lookahead_m=SPEED_LOOKAHEAD_M,
        position_gain=CARROT_POSITION_GAIN,
        velocity_gain=CARROT_VELOCITY_GAIN,
        initial_thrust=CARROT_INITIAL_THRUST,
    )

    geometric_path_follower = GeometricPathFollower(
        path_manager,
        kp_cross=GEOMETRIC_CROSS_TRACK_GAIN,
        kd_cross=GEOMETRIC_CROSS_TRACK_DAMPING,
        acceleration_filter_alpha=GEOMETRIC_ACCELERATION_FILTER_ALPHA,
        lookahead_m=GEOMETRIC_LOOKAHEAD_M,
        speed_lookahead_m=SPEED_LOOKAHEAD_M,
        v_max=GEOMETRIC_MAX_SPEED_MPS,
        a_lat_max=GEOMETRIC_MAX_LATERAL_ACCELERATION_MPS2,
        curvature_speed_deadband=GEOMETRIC_CURVATURE_SPEED_DEADBAND,
        curvature_speed_ramp=GEOMETRIC_CURVATURE_SPEED_RAMP,
        curvature_feedforward_gain=GEOMETRIC_CURVATURE_FEEDFORWARD_GAIN,
        curvature_feedforward_max_acceleration_mps2=(
            GEOMETRIC_CURVATURE_FEEDFORWARD_MAX_ACCELERATION_MPS2
        ),
        hover_thrust=GEOMETRIC_HOVER_THRUST,
        max_commanded_acceleration_mps2=GEOMETRIC_MAX_COMMANDED_ACCELERATION_MPS2,
        max_upward_acceleration_mps2=GEOMETRIC_MAX_UPWARD_ACCELERATION_MPS2,
        max_downward_acceleration_mps2=GEOMETRIC_MAX_DOWNWARD_ACCELERATION_MPS2,
        max_tilt_deg=GEOMETRIC_MAX_TILT_DEG,
    )

    hover_controller = HoverController(
        lateral_velocity_gain=HOVER_LATERAL_VELOCITY_GAIN,
        vertical_velocity_gain=HOVER_VERTICAL_VELOCITY_GAIN,
        vertical_acceleration_gain=HOVER_VERTICAL_ACCELERATION_GAIN,
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
        path_manager,
        attitude_controller,
        carrot_controller,
        geometric_path_follower,
        hover_controller,
    )
