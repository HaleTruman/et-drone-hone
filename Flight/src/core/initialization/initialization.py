"""
The Emperor is our guiding light, a beacon of hope for humanity in a galaxy of darkness.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time

from autonomy.pathing import PathManager
from core.control.attitude import AttitudeController
from core.control.autipilot import AutiPilot, AutiPilotGains
from core.control.hover.controller import HoverController
from core.control.path_follower import GeometricPathFollower
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

# Simulator and network endpoints.
MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"  # selects the MAVLink UDP endpoint used to talk to the simulator or vehicle bridge.
SIM_RUNTIME = "VQ_2"  # identifies the simulator/runtime profile passed into the MAVLink client.
VISION_HOST = "0.0.0.0"  # is the local interface where the vision receiver listens for incoming frames.
VISION_PORT = 5600  # is the UDP/TCP port used by the vision frame receiver.

# Control loop timing.
INNER_LOOP_HZ = 120.0  # is the fast control loop rate used for state updates and attitude commands.
OUTER_LOOP_HZ = 30.0  # is the slower loop rate used for vision, path planning, and path-following targets.
RUN_S: float | None = None  # optionally limits flight duration in seconds; None runs until interrupted or finished.

# Startup, reset, and arming timeouts.
HEARTBEAT_TIMEOUT_S = 120.0  # is how long startup waits for the MAVLink heartbeat before failing.
STARTUP_DATA_TIMEOUT_S = 4.0  # is how long startup waits for fresh telemetry and vision data.
IMU_INIT_TIMEOUT_S = 1.5  # is the stationary sample window used for initial IMU bias estimation.
GATE_MAP_INIT_TIMEOUT_S = 1.5  # is the startup window used to collect initial vision observations and build a path.
POST_RESET_DELAY_S = 0.7  # gives the simulator time to settle after a reset command.
ARM_TIMEOUT_S = 5.0  # is how long the system waits for the vehicle to arm successfully.

# Gate mapping
GATE_MERGE_DISTANCE_M = 5.0  # merges repeated gate observations within this local-NED distance.
REQUIRED_MINIMUM_OBSERVATION_COUNT = 5  # requires this many merged observations before a gate is published.
GATE_LOCKOUT_COUNT = 500  # locks a gate pose after this many merged observations.
GATE_MAX_OBSERVATION_DISTANCE_M = 40.0  # ignores gate observations farther than this from the vehicle.
GATE_TARGET_CANDIDATE_MIN_OBSERVATION_COUNT = 2  # allows tentative target use before full gate-map publication.
GATE_TARGET_CANDIDATE_MIN_POSITION_CONFIDENCE = 0.35  # rejects weak candidate targets from noisy detections.
GATE_TARGET_CANDIDATE_MAX_AVERAGE_RESIDUAL_M = 2.5  # rejects candidate targets whose merged positions are unstable.
GATE_TARGET_CANDIDATE_MAX_DISTANCE_M = 45.0  # ignores tentative target candidates too far from the vehicle.

# Path planning.
PLANNING_MODE = "test"  # chooses the PathManager strategy: test or gate.
PATH_UPDATE_MODE = "persist_crossed"  # chooses how new plans replace the active path: original, projected, persist_crossed, or splice.
PATH_UPDATE_OBSERVATION_INTERVAL = 50  # updates the planned path after this many processed vision observations; 1 updates every observation.
GATE_PASSED_DISTANCE_M = 1.7  # treats gates closer than this as already passed for planning purposes.
PATH_CROSSED_GATE_PERSIST_DISTANCE_M = 15.0  # keeps a crossed gate as the path start anchor while the drone is near it.
PATH_CROSSED_GATE_CURVATURE_PRESERVE_M = 3.0  # preserves this much active-path curvature after a persisted crossed gate.
SPLINE_CORNER_TIGHTNESS = 0.03  # controls how tightly generated splines follow corner anchor points.
ADAPTIVE_SPLINE_TIGHTNESS = False  # enables automatic corner tightness changes based on segment geometry.
DISTANT_SPLINE_CORNER_TIGHTNESS = 0.10  # is the looser spline tightness used for distant or gentle turns.
MIN_SPLINE_CORNER_TIGHTNESS = 0.20  # is the lower bound for adaptive spline tightness near turns.
MAX_SPLINE_CORNER_TIGHTNESS = 0.95  # is the upper bound for adaptive spline tightness near sharp turns.
GENTLE_TURN_ANGLE_DEG = 20.0  # defines the turn angle below which corners are treated as gentle.
SHARP_TURN_ANGLE_DEG = 60.0  # defines the turn angle at which corners receive maximum adaptive tightness.
SHORT_SEGMENT_REFERENCE_M = 5.0  # marks the segment length where nearby turns become more tightly constrained.
LONG_SEGMENT_REFERENCE_M = 25.0  # marks the segment length where distance-based spline tightening fades out.
PATH_SPACING_M = 0.25  # is the waypoint spacing used when sampling generated paths.
PATH_TAIL_LENGTH_M = 10.0  # extends planned paths beyond the last gate along the terminal gate-to-gate tangent.
PATH_SPLICE_LOOKAHEAD_GAIN_S = 1.5  # multiplies vehicle speed to choose how far ahead on the active path a new plan is spliced.

# Control mode and safety envelope.
CONTROL_METHOD = "geometric_path_follower"  # selects which path-following controller produces the attitude target.
FAILSAFE_DISTANCE = 10  # is the maximum allowed cross-track path error before ending racing flight.
ALLOW_FLIGHT = True  # enables sending flight commands when the system mode allows it.

# Attitude inner-loop response.
ATTITUDE_ERROR_QUATERNION_ROLL_SCALE = 1.65  # Test-only sim command boost; increase to bank faster when sending quaternion error targets.
ATTITUDE_ERROR_QUATERNION_PITCH_SCALE = 1.65  # Test-only sim command boost for pitch error quaternion vector component.
ATTITUDE_ERROR_QUATERNION_YAW_SCALE = 1.65  # Test-only sim command boost for yaw error quaternion vector component.
ATTITUDE_ROLL_GAIN = 1.5  # Roll attitude P gain; increase for faster banking, decrease if roll oscillates.
ATTITUDE_PITCH_GAIN = 1.5  # Pitch attitude P gain; increase for faster pitch response, decrease if pitch oscillates.
ATTITUDE_YAW_GAIN = 1.5  # Yaw attitude P gain; increase for faster heading alignment, decrease if yaw hunts.
ATTITUDE_DAMPING = 0.2  # Body-rate damping; increase to reduce attitude overshoot, decrease if response feels sluggish.
ATTITUDE_RATE_FILTER_ALPHA = 0.8  # Body-rate filter alpha; higher follows gyro faster, lower smooths noisy damping.
ATTITUDE_MAX_BODY_RATE_RPS = 7.0  # Body-rate command cap; increase for faster attitude changes, decrease for gentler motion.

# IMU gyro spike rejection.
GYRO_SPIKE_FILTER_ENABLED = False  # Rejects isolated angular-rate spikes before attitude integration and control use.
GYRO_SPIKE_MAX_RATE_RPS = 20.0  # Rejects a single gyro sample above this body-rate norm unless it persists.
GYRO_SPIKE_MAX_DELTA_RPS = 10.0  # Rejects a single gyro sample whose rate jump from the previous accepted rate exceeds this norm.
GYRO_SPIKE_MAX_REJECTIONS = 1  # Accepts the sample after this many consecutive rejections so real sustained motion is not hidden.

# Path lookahead and preview.
CARROT_LOOKAHEAD_M = 4.0 # Carrot-point preview distance; increase to turn earlier/smoother, decrease to track nearby path more tightly.
SPEED_LOOKAHEAD_M = 30  # Curvature preview distance for speed planning; increase to slow earlier, decrease to react later.

# Cross-track position hold: horizontal component.
GEOMETRIC_CROSS_TRACK_GAIN = 3.5  # Horizontal cross-track P gain; increase to pull harder toward the path, decrease if it weaves.
GEOMETRIC_CROSS_TRACK_DAMPING =  2.0 # Horizontal cross-track D gain; increase to damp sideways drift, decrease if turns feel over-braked.

# Cross-track position hold: vertical component.
GEOMETRIC_VERTICAL_CROSS_TRACK_GAIN = 3.5  # Vertical cross-track P gain; increase to correct altitude error sooner, decrease if altitude oscillates.
GEOMETRIC_VERTICAL_CROSS_TRACK_DAMPING = 2.0  # Vertical cross-track D gain; increase to damp climb/descent rate, decrease if altitude lags.

# Cross-track gain scheduling from upcoming curvature.
GEOMETRIC_CROSS_GAIN_CURVATURE_DEADBAND = 0.25  # Curvature below this leaves cross-track gains unchanged; raise to ignore gentler turns.
GEOMETRIC_CROSS_GAIN_CURVATURE_RAMP = 0.5  # Curvature span to full scheduled gain; lower makes boosts arrive faster, higher makes them gradual.
GEOMETRIC_HORIZONTAL_CROSS_GAIN_CURVATURE_BOOST = 2.3  # Max fractional horizontal gain boost in curves; increase for tighter turns.
GEOMETRIC_VERTICAL_CROSS_GAIN_CURVATURE_BOOST = 1.3  # Max fractional vertical gain boost in curves; keep modest to avoid altitude coupling.

# Command smoothing.
GEOMETRIC_ACCELERATION_FILTER_ALPHA = 0.2  # Desired-acceleration filter alpha; 1 disables smoothing, lower softens command jumps.

# Along-track speed loop and curve speed planning.
GEOMETRIC_SPEED_GAIN = 0.65  # Along-track speed P gain; increase to reach target speed faster, decrease if it surges.
GEOMETRIC_SPEED_DAMPING = 0.6  # Along-track speed D gain on acceleration; increase to reduce speed overshoot, decrease for quicker response.
GEOMETRIC_MAX_SPEED_MPS = 10  # Straight-path target speed cap; increase for faster runs, decrease if tracking cannot keep up.
GEOMETRIC_MAX_LATERAL_ACCELERATION_MPS2 = 50.0  # Curve-speed lateral accel budget; increase to carry more speed through turns.
GEOMETRIC_CURVATURE_SPEED_DEADBAND = 0.25  # Curvature below this commands max speed; raise to ignore mild curves, lower to slow sooner.
GEOMETRIC_CURVATURE_SPEED_RAMP = 0.1  # Curvature softening ramp for speed reduction; reduce for a lower speed in tight corners.
GEOMETRIC_CROSS_TRACK_SPEED_DERATE_START_M = 0.5  # Cross-track error where speed derating starts; raise to ignore small tracking errors.
GEOMETRIC_CROSS_TRACK_SPEED_DERATE_FULL_M = 2.0  # Cross-track error where derating reaches full strength; lower to slow harder sooner.
GEOMETRIC_CROSS_TRACK_SPEED_DERATE_MIN_SCALE = 0.7  # Minimum speed scale at full derate; lower to slow more while far off path.
GEOMETRIC_LAUNCH_SPEED_RAMP_S = 1.7  # Seconds to ramp path-following speed from zero after takeoff; increase to soften launch.

# Curvature turn feed-forward.
GEOMETRIC_CURVATURE_FEEDFORWARD_GAIN = 0.7  # Turn feed-forward gain; increase to bank into turns earlier, decrease if it over-turns.
GEOMETRIC_CURVATURE_FEEDFORWARD_MAX_ACCELERATION_MPS2 = 10.0  # Feed-forward accel cap; increase for stronger turn anticipation.

# Geometric acceleration, tilt, and thrust limits.
GEOMETRIC_HOVER_THRUST = 0.2644  # Normalized hover thrust; tune to the thrust that holds level hover.
GEOMETRIC_MAX_COMMANDED_ACCELERATION_MPS2 = 20  # Total desired-accel cap; decrease to soften all path-follower commands.
GEOMETRIC_MAX_COMMANDED_JERK_MPS3 = 50.0  # Desired-accel slew cap; decrease to soften command-vector jumps.
GEOMETRIC_MAX_UPWARD_ACCELERATION_MPS2 = 15.0  # Upward accel cap in NED (-Z); increase for harder climbs, decrease to prevent pop-ups.
GEOMETRIC_MAX_DOWNWARD_ACCELERATION_MPS2 = 9.0  # Downward accel cap in NED (+Z); increase to descend faster, keep below gravity for margin.
GEOMETRIC_MAX_TILT_DEG = 90  # Desired tilt cap; increase for more aggressive banking/inversion, decrease for upright flight.
GEOMETRIC_TILT_THRUST_ALIGNMENT_MIN = 0.0 # Minimum thrust scale while actual tilt catches desired tilt; raise to preserve thrust, lower to suppress climb-before-bank.

# Gate-aware acceleration controller.

# Speed limits and turn-speed shaping.
AUTIPILOT_MAX_SPEED_MPS = 25  # Straight speed cap; raise for faster straights, lower if braking into turns is too abrupt.
AUTIPILOT_MIN_SPEED_MPS = 2.0  # Minimum curvature-limited speed; raise to avoid crawling, lower to permit very slow tight turns.
AUTIPILOT_MAX_LATERAL_ACCELERATION_MPS2 = 36.0  # Default turn-speed accel budget; raise to carry more speed, lower to slow earlier.
AUTIPILOT_MEDIUM_TURN_ANGLE_DEG = 40.0  # Angle where medium-turn speed shaping starts; lower to affect gentler turns sooner.
AUTIPILOT_TIGHT_TURN_ANGLE_DEG = 95.0  # Angle where tight-turn speed shaping is full; lower to treat more turns as tight.
AUTIPILOT_MEDIUM_TURN_LATERAL_ACCELERATION_MPS2 = 24.0  # Medium-turn speed budget; raise to fly through medium turns faster, lower to slow them more.
AUTIPILOT_TIGHT_TURN_LATERAL_ACCELERATION_MPS2 = 15.0  # Tight-turn speed budget; raise to carry speed through sharp turns, lower to stay tighter/slower.

# Speed control near gates.
AUTIPILOT_SPEED_GAIN = 0.7  # Along-aim speed P gain; raise for harder accel/braking, lower to reduce pitch-up/pitch-down snaps.
AUTIPILOT_NEAR_GATE_SPEED_GAIN_NEAR_M = 1.0  # Distance where speed gain reaches minimum; raise to soften speed control earlier.
AUTIPILOT_NEAR_GATE_SPEED_GAIN_FAR_M = 5.0  # Distance where speed gain starts reducing; raise to begin pitch/brake suppression farther out.
AUTIPILOT_NEAR_GATE_SPEED_GAIN_MIN_SCALE = 1.0  # Minimum speed-gain multiplier near gates; lower to reduce pitch from accel/braking more.
AUTIPILOT_NEAR_GATE_SPEED_GAIN_RAMP_S = 0.5  # Time to slew speed-gain scale up/down; raise for smoother changes, lower for faster response.

# Direction, lateral, and vertical gains.
AUTIPILOT_DIRECTION_GAIN = 1.5  # Velocity alignment P gain; raise to point velocity toward aim faster, lower if lateral commands are twitchy.
AUTIPILOT_DIRECTION_DAMPING = 0.6  # Measured lateral-accel damping; raise to resist oscillation, lower if response feels sluggish.
AUTIPILOT_LATERAL_POSITION_GAIN = 1.7  # Gate-line position correction; raise to pull harder to the aim line, lower to allow more drift/fly-through.
AUTIPILOT_LATERAL_DAMPING = 2.1  # Lateral velocity damping; raise to kill sideways drift sooner, lower if it over-corrects near gates.
AUTIPILOT_VERTICAL_POSITION_GAIN = 2.6  # Gate altitude P gain; raise to hit gate height harder, lower to reduce pop-up/drop-through behavior.
AUTIPILOT_VERTICAL_DAMPING = 1.8  # Vertical velocity damping; raise to suppress climb/descent rate, lower if altitude response is too lazy.

# Next-gate lookahead and fly-through behavior.
AUTIPILOT_LOOKAHEAD_NEAR_M = 0.6  # Distance where next-gate lookahead reaches full blend; raise to delay full turn-in closer to gate.
AUTIPILOT_LOOKAHEAD_FAR_M = 2.6  # Distance where next-gate lookahead begins; raise to start turning earlier, lower to stay aimed at current gate longer.
AUTIPILOT_FLY_THROUGH_ENABLED = True  # Temporarily disables fly-through lookahead scaling; set True to restore this group.
AUTIPILOT_FLY_THROUGH_TURN_ANGLE_START_DEG = 70.0  # Turn angle where long-exit fly-through starts; lower to enable it on milder turns.
AUTIPILOT_FLY_THROUGH_TURN_ANGLE_FULL_DEG = 120.0  # Turn angle for maximum fly-through; lower for stronger effect on less severe turns.
AUTIPILOT_FLY_THROUGH_NEXT_GATE_DISTANCE_START_M = 12.0  # Next-gate distance where fly-through starts; lower to enable it on shorter exits.
AUTIPILOT_FLY_THROUGH_NEXT_GATE_DISTANCE_FULL_M = 30.0  # Next-gate distance for maximum fly-through; lower to reach full effect sooner.
AUTIPILOT_FLY_THROUGH_LOOKAHEAD_MIN_SCALE = 0.25  # Minimum next-gate lookahead scale; lower to fly straighter through gates, raise for earlier turn-in.

# Gate crossing and post-cross recovery.
AUTIPILOT_GATE_CROSS_VELOCITY_DAMPING = 2.0  # Near-gate cross-velocity cancel gain; raise to stop lateral gate drift, lower if it spikes a_des.
AUTIPILOT_GATE_CROSS_VELOCITY_CANCEL_NEAR_M = 2.0  # Distance for full cross-velocity cancel; raise to apply full damping earlier.
AUTIPILOT_GATE_CROSS_VELOCITY_CANCEL_FAR_M = 8.0  # Distance where cross-velocity cancel starts; raise to start drift cancellation earlier.
AUTIPILOT_APPROACH_GAIN_MIN_SCALE = 0.8  # Minimum lateral gain near crossing; lower to relax through gates, raise to hold the aim line.
AUTIPILOT_POST_CROSS_TURN_SCALE = 0.12  # Post-cross turn gain boost; raise to turn harder after crossing, lower to soften gate-exit snaps.
AUTIPILOT_POST_CROSS_RAMP_DISTANCE_M = 8.0  # Distance to ramp post-cross gain; raise for slower recovery, lower for faster turn commitment.

# Launch, thrust, and command limiting.
AUTIPILOT_LAUNCH_SPEED_RAMP_S = 0.25  # Launch speed ramp time; raise to soften launch, lower to reach speed sooner.
AUTIPILOT_MAX_SPECIFIC_THRUST_MPS2 = 37.090204236006045  # Thrust scaling reference; adjust only when hover thrust calibration changes.
AUTIPILOT_MIN_NORMALIZED_THRUST = 0.05  # Lower thrust clamp; raise to prevent low-thrust drops, lower to allow stronger unloading.
AUTIPILOT_MAX_NORMALIZED_THRUST = 0.95  # Upper thrust clamp; raise for more authority, lower to cap climb/accel spikes.
AUTIPILOT_MAX_COMMANDED_ACCELERATION_MPS2 = 28.0  # Total a_des magnitude cap; lower to soften all commands, raise for more aggressive control.
AUTIPILOT_GATE_SWITCH_ACCELERATION_RAMP_S = 0.35  # Gate-switch a_des blend time; raise to soften target changes, lower for faster response.
AUTIPILOT_MAX_COMMANDED_JERK_MPS3 = 40.0  # Desired-accel slew cap; lower to smooth command jumps, raise for sharper response.

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
VISION_PERCEPTION_BACKEND = "deterministic_v3_2"  # selects the gate perception implementation used for vision frames.
VISION_EXECUTOR_MAX_WORKERS = 1  # controls the number of background workers for vision processing.
VISION_EXECUTOR_THREAD_PREFIX = "vision"  # names background vision worker threads for debugging.
PATH_MAX_POINTS = 1000  # caps the number of sampled waypoints kept in a generated path.
TEST_PATH_GATE_POINTS_LOCAL_NED_M = [
    (10.9, 0.0, -0.5),
    (26, 8.5, -2.8),
    (34.535, 11.5, -2.252),
    (43.9, 3.7, -1.2),
    (60.848, -13.544, -0.085),
]  # mock gate centers used by build_test_path() when PLANNING_MODE is test.

# Finish-hover velocity damping.
HOVER_LATERAL_VELOCITY_GAIN = 2.5  # Hover horizontal velocity damping; increase to stop XY drift faster, decrease if it rocks.
HOVER_VERTICAL_VELOCITY_GAIN = 0.18  # Hover vertical velocity damping; increase to stop climb/descent faster, decrease if it bounces.
HOVER_VERTICAL_ACCELERATION_GAIN = 0.035  # Hover vertical accel damping; increase to resist vertical acceleration, decrease if noisy.

# Logging.
RUNS_ROOT = Path(__file__).resolve().parents[4] / "Logs" / "flight" / "runs"  # is the root directory where timestamped run logs are created.


def _initialization_constants() -> dict[str, object]:
    return {
        name: _constant_log_value(value)
        for name, value in globals().items()
        if name.isupper()
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
    planning_mode: str
    path_update_observation_interval: int
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
    GeometricPathFollower,
    AutiPilot,
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
        path_update_observation_interval=max(1, int(PATH_UPDATE_OBSERVATION_INTERVAL)),
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

    path_manager = PathManager(
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
        path_update_mode=PATH_UPDATE_MODE,
        path_crossed_gate_persist_distance_m=PATH_CROSSED_GATE_PERSIST_DISTANCE_M,
        path_crossed_gate_curvature_preserve_m=PATH_CROSSED_GATE_CURVATURE_PRESERVE_M,
        spacing_m=PATH_SPACING_M,
        path_tail_length_m=PATH_TAIL_LENGTH_M,
        path_splice_lookahead_gain_s=PATH_SPLICE_LOOKAHEAD_GAIN_S,
        max_points=PATH_MAX_POINTS,
    )
    if PLANNING_MODE == "test":
        path_manager.build_test_path(
            gate_points_local_ned_m=TEST_PATH_GATE_POINTS_LOCAL_NED_M,
        )

    attitude_controller = AttitudeController(
        roll_gain=ATTITUDE_ROLL_GAIN,
        pitch_gain=ATTITUDE_PITCH_GAIN,
        yaw_gain=ATTITUDE_YAW_GAIN,
        damping=ATTITUDE_DAMPING,
        rate_filter_alpha=ATTITUDE_RATE_FILTER_ALPHA,
        max_body_rate_rps=ATTITUDE_MAX_BODY_RATE_RPS,
        error_quaternion_roll_scale=ATTITUDE_ERROR_QUATERNION_ROLL_SCALE,
        error_quaternion_pitch_scale=ATTITUDE_ERROR_QUATERNION_PITCH_SCALE,
        error_quaternion_yaw_scale=ATTITUDE_ERROR_QUATERNION_YAW_SCALE,
    )

    geometric_path_follower = GeometricPathFollower(
        kp_cross=GEOMETRIC_CROSS_TRACK_GAIN,
        kd_cross=GEOMETRIC_CROSS_TRACK_DAMPING,
        kp_cross_vertical=GEOMETRIC_VERTICAL_CROSS_TRACK_GAIN,
        kd_cross_vertical=GEOMETRIC_VERTICAL_CROSS_TRACK_DAMPING,
        cross_gain_curvature_deadband=GEOMETRIC_CROSS_GAIN_CURVATURE_DEADBAND,
        cross_gain_curvature_ramp=GEOMETRIC_CROSS_GAIN_CURVATURE_RAMP,
        horizontal_cross_gain_curvature_boost=(
            GEOMETRIC_HORIZONTAL_CROSS_GAIN_CURVATURE_BOOST
        ),
        vertical_cross_gain_curvature_boost=GEOMETRIC_VERTICAL_CROSS_GAIN_CURVATURE_BOOST,
        kp_speed=GEOMETRIC_SPEED_GAIN,
        kd_speed=GEOMETRIC_SPEED_DAMPING,
        acceleration_filter_alpha=GEOMETRIC_ACCELERATION_FILTER_ALPHA,
        lookahead_m=CARROT_LOOKAHEAD_M,
        speed_lookahead_m=SPEED_LOOKAHEAD_M,
        v_max=GEOMETRIC_MAX_SPEED_MPS,
        a_lat_max=GEOMETRIC_MAX_LATERAL_ACCELERATION_MPS2,
        curvature_speed_deadband=GEOMETRIC_CURVATURE_SPEED_DEADBAND,
        curvature_speed_ramp=GEOMETRIC_CURVATURE_SPEED_RAMP,
        cross_track_speed_derate_start_m=GEOMETRIC_CROSS_TRACK_SPEED_DERATE_START_M,
        cross_track_speed_derate_full_m=GEOMETRIC_CROSS_TRACK_SPEED_DERATE_FULL_M,
        cross_track_speed_derate_min_scale=GEOMETRIC_CROSS_TRACK_SPEED_DERATE_MIN_SCALE,
        launch_speed_ramp_s=GEOMETRIC_LAUNCH_SPEED_RAMP_S,
        curvature_feedforward_gain=GEOMETRIC_CURVATURE_FEEDFORWARD_GAIN,
        curvature_feedforward_max_acceleration_mps2=(
            GEOMETRIC_CURVATURE_FEEDFORWARD_MAX_ACCELERATION_MPS2
        ),
        hover_thrust=GEOMETRIC_HOVER_THRUST,
        max_commanded_acceleration_mps2=GEOMETRIC_MAX_COMMANDED_ACCELERATION_MPS2,
        max_commanded_jerk_mps3=GEOMETRIC_MAX_COMMANDED_JERK_MPS3,
        max_upward_acceleration_mps2=GEOMETRIC_MAX_UPWARD_ACCELERATION_MPS2,
        max_downward_acceleration_mps2=GEOMETRIC_MAX_DOWNWARD_ACCELERATION_MPS2,
        max_tilt_deg=GEOMETRIC_MAX_TILT_DEG,
        tilt_thrust_alignment_min=GEOMETRIC_TILT_THRUST_ALIGNMENT_MIN,
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
        )
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
        geometric_path_follower,
        autipilot,
        hover_controller,
    )
