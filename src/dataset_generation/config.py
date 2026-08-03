"""Configuration for Unreal gate dataset generation."""

# Level and actor lookup.
GATE_FOLDER = "Gates"
CAMERA_LABEL = "AI_Hangar_Wide_Long_Interior_Camera"
TARGET_LEVEL_PATH = "/Game/Hangar1"
LEVEL_PATH = TARGET_LEVEL_PATH
RENDER_TARGET_PATH = "/Game/NewTextureRenderTarget640x360"
MASK_RENDER_TARGET_PATH = "/Game/RT_GateStencilMask_640x360"

# Output folders and files.
DATASET_SCHEMA_VERSION = 1
DATASETS_DIRECTORY_NAME = "datasets/evaluation"
DATASET_DIRECTORY_PREFIX = "dataset_"
RUNS_DIRECTORY_NAME = "runs"
RUN_DIRECTORY_PREFIX = "run_"
FRAME_DIRECTORY_NAME = "frames"
MASK_DIRECTORY_NAME = "masks"
FRAME_FILE_EXTENSION = "png"
FRAME_METADATA_FILE_NAME = "metadata.json"
FRAME_METADATA_JSONL_FILE_NAME = "metadata.jsonl"

# Output toggles.
SAVE_FRAMES = True
SAVE_FRAME_METADATA = True
SAVE_MASKS = True
MAX_EXPECTED_MASK_STENCIL_VALUES = 32
SAVE_RAW_MASK_DEBUG = True
MASK_ID_TOLERANCE = 2
MASK_MIN_COMPONENT_PIXELS = 20

# Number of full randomize-and-collect batches to run from one script execution.
# Each batch creates its own run directory and is equivalent to executing the
# script again after the previous batch finishes.
RUN_COUNT = 100

# Background-only dataset collection.
# Set FORCE_ZERO_GATE_RUN=True to collect background-only frames even when gate
# actors exist in the level. Existing gate actors will be hidden for the run.
FORCE_ZERO_GATE_RUN = False

# Percentage of frames in each normal run batch that should be background-only
# zero-gate frames. Uses final-run percentage semantics:
# 20.0 means 20% zero-gate frames and 80% normal gate-targeted frames.
ZERO_GATE_FRAME_PERCENT = 10

# When True, the script can still collect frames if no gate actors are found.
# In that case it creates this many random camera poses per run batch and writes
# metadata with target_gate=None and gates=[].
ALLOW_ZERO_GATES = True
ZERO_GATE_CAMERA_POSITIONS = 100

# Render target / frame geometry.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
SCENE_CAPTURE_WARMUP_CAPTURES = 2

# Gate corner component names.
GATE_CORNER_NAMES = (
    "Corner_outer_TL",
    "Corner_outer_TR",
    "Corner_outer_BL",
    "Corner_outer_BR",
    "Corner_inner_TL",
    "Corner_inner_TR",
    "Corner_inner_BL",
    "Corner_inner_BR",
)
GATE_VISIBILITY_SAMPLE_STEPS = 4

BBOX_CORNER_NAMES = (
    "Bbox_TL_front",
    "Bbox_TR_front",
    "Bbox_BL_front",
    "Bbox_BR_front",
    "Bbox_TL_rear",
    "Bbox_TR_rear",
    "Bbox_BL_rear",
    "Bbox_BR_rear"
)

# Gate segmentation mask setup.
# Requires Project Settings > Rendering > Custom Depth-Stencil Pass:
# "Enabled with Stencil".
ENABLE_GATE_CUSTOM_DEPTH_STENCIL = True
GATE_CUSTOM_DEPTH_STENCIL_VALUE = 1
USE_PER_GATE_CUSTOM_DEPTH_STENCIL_VALUES = True
GATE_CUSTOM_DEPTH_STENCIL_BASE_VALUE = 240
GATE_CUSTOM_DEPTH_STENCIL_BANDS = (
    16,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
)

# Gate and camera placement volume, in Unreal centimeters.
X_RANGE_CM = (-5000.0, 5000.0)
Y_RANGE_CM = (-10000.0, 10000.0)
Z_RANGE_CM = (150.0, 2000.0)

# Gate orientation.
PITCH_RANGE_DEG = (-15.0, 15.0)  # Unreal pitch is rotation around Y.
ROLL_DEG = 0.0
YAW_RANGE_DEG = (0.0, 360.0)  # Unreal yaw is rotation around Z.
GATE_FORWARD_YAW_OFFSET_DEG = 0.0

# Gate spacing.
MIN_GATE_DISTANCE_CM = 600.0
GATE_CENTER_SPLINE_Z_OFFSET_CM = 60.0
TRACK_GATE_SPACING_WEIGHT_RANGE = (0.55, 1.45)

# Obstacle clearance.
OBSTACLE_MAP_JSONL = "hangar_extraction/hangar_columns_pylons.jsonl"
OBSTACLE_CLEARANCE_CM = 300.0
MAX_TRACK_LAYOUT_ATTEMPTS = 100

# Track generation.
TRACK_TOTAL_LENGTH_CM = 8000.0
TRACK_BEND_COUNT_RANGE = (2, 3, 4)
TRACK_BEND_TURN_RANGE_DEG = (45.0, 95.0)
TRACK_GATE_RANDOM_DISTANCE_MARGIN_CM = 300.0
TRACK_GATE_DISTANCE_RANGE_CM = (600.0, 3500.0)
TRACK_EDGE_MARGIN_CM = 300.0
TRACK_WAYPOINTS_PER_GATE = 1.0
TRACK_MIN_WAYPOINTS = 7
TRACK_MAX_WAYPOINTS = 10
TRACK_SPLINE_SAMPLES_PER_SEGMENT = 4
TRACK_STEP_RANGE_CM = (1200.0, 2200.0)
TRACK_TURN_RANGE_DEG = (-20.0, 20.0)
TRACK_Z_STEP_RANGE_CM = (-420.0, 420.0)
TRACK_VOLUME_ANCHOR_CHANCE = 0.0
TRACK_START_X_CM = 0
TRACK_START_Y_CM = -9300.0
TRACK_START_Z_CM = 150.0
TRACK_END_X_CM = 2700.0
TRACK_END_Y_CM = 8600.0
TRACK_END_Z_CM = 350.0
CREATE_TRACK_SPLINE_ACTOR = True
TRACK_SPLINE_ACTOR_LABEL = "BP_Track_Indicator_Test_Spline"
TRACK_SPLINE_ACTOR_FOLDER = "Track"
TRACK_SPLINE_CREATE_IF_MISSING = False
CREATE_TRACK_INDICATOR_SEGMENTS = False
TRACK_INDICATOR_TEMPLATE_LABEL = "TrackIndicatorSegment"
TRACK_INDICATOR_SEGMENT_PREFIX = "Generated_TrackIndicatorSegment_"
TRACK_INDICATOR_SEGMENTS_FOLDER = "Track/Segments"
TRACK_INDICATOR_SEGMENT_SPACING_CM = 100.0
TRACK_INDICATOR_SEGMENT_ROTATION_OFFSET_DEG = (0.0, 0.0, 0.0)
HIDE_TRACK_INDICATOR_TEMPLATE = True
VISIBILITY_TRACE_IGNORE_ACTOR_PREFIXES = (
    TRACK_SPLINE_ACTOR_LABEL,
    TRACK_INDICATOR_SEGMENT_PREFIX,
)
VISIBILITY_TRACE_IGNORE_COMPONENT_NAME_FRAGMENTS = (
    "TrackIndicator",
    "Track_Indicator",
    "SplineMesh",
)
VISIBILITY_TRACE_IGNORE_COMPONENT_TAGS = (
    "GeneratedTrackSplineMesh",
)
TRACK_RENDER_SETTLE_SECONDS = 1.0
TRACK_RENDER_READY_BOOL_PROPERTY = "TrackRenderReady"
TRACK_RENDER_STABLE_TICKS = 5
TRACK_RENDER_MAX_WAIT_SECONDS = 30.0
COMPILE_TRACK_BLUEPRINT_AFTER_SPLINE_UPDATE = True

# Camera placement and aim.
MIN_CAMERA_GATE_DISTANCE_CM = 180.0
MAX_CAMERA_GATE_DISTANCE_CM = 1100.0
MAX_CAMERA_POSE_ATTEMPTS = 250
CAMERA_POSITIONS_PER_GATE = 5
CAMERA_HOLD_SECONDS = 0.05
CAMERA_FRAME_MARGIN = 0.95

# Coverage-driven camera pose sampling.
# Candidate poses are generated in gate-relative space, projected into the
# image, then accepted against under-filled dataset buckets.
ENABLE_COVERAGE_CAMERA_SAMPLING = True
CAMERA_CANDIDATES_PER_POSE = 80
CAMERA_ACCEPT_SCORE_THRESHOLD = 0.15
CAMERA_GATE_DISTANCE_BINS_CM = (
    (180.0, 360.0),
    (360.0, 700.0),
    (700.0, 1100.0),
)
CAMERA_GATE_AZIMUTH_BINS_DEG = {
    "front": (-30.0, 30.0),
    "front_left": (30.0, 75.0),
    "left": (75.0, 120.0),
    "rear_left": (120.0, 170.0),
    "rear": (170.0, 190.0),
    "rear_right": (190.0, 240.0),
    "right": (240.0, 285.0),
    "front_right": (285.0, 330.0),
}
CAMERA_GATE_ELEVATION_BINS_DEG = {
    "below": (-28.0, -8.0),
    "level": (-8.0, 8.0),
    "above": (8.0, 28.0),
}
CAMERA_IMAGE_TARGET_BINS = {
    "center": (-0.25, 0.25, -0.25, 0.25),
    "left_edge": (-0.85, -0.45, -0.35, 0.35),
    "right_edge": (0.45, 0.85, -0.35, 0.35),
    "top_edge": (-0.35, 0.35, -0.85, -0.45),
    "bottom_edge": (-0.35, 0.35, 0.45, 0.85),
    "corner": (-0.9, 0.9, -0.9, 0.9),
}
CAMERA_BBOX_AREA_BINS = {
    "tiny": (0.0005, 0.01),
    "small": (0.01, 0.04),
    "medium": (0.04, 0.14),
    "large": (0.14, 0.45),
    "extreme": (0.45, 1.50),
}
CAMERA_TRUNCATION_TARGET_FRACTION = 0.25
CAMERA_HARD_NEGATIVE_TARGET_FRACTION = 0.08

# Spline-flight camera pose sampling.
# When a generated track spline is available, captures can be taken as one
# continuous time-based flight over the full spline.
ENABLE_SPLINE_FLIGHT_CAMERA_SAMPLING = True
ENABLE_CONTINUOUS_SPLINE_FLIGHT_CAPTURE = True
FLIGHT_SIM_DT_RANGE_SECONDS = (0.04, 0.09)
FLIGHT_SPEED_RANGE_MPS = (5.0, 20.0)
FLIGHT_MAX_ACCEL_MPS2 = 4.0
FLIGHT_LOOKAHEAD_TIME_SECONDS = 1.1
FLIGHT_LOOKAHEAD_DISTANCE_RANGE_CM = (450.0, 1700.0)
FLIGHT_SPLINE_OFFSET_DISTANCE_RANGE_CM = (0.0, 50.0)
FLIGHT_SPLINE_LATERAL_OFFSET_RANGE_CM = (-50.0, 50.0)
FLIGHT_SPLINE_VERTICAL_OFFSET_RANGE_CM = (0.0, 50.0)
FLIGHT_OFFSET_CHANGE_PER_SECOND_CM = 120.0
FLIGHT_ATTITUDE_YAW_FRAME_FRACTION = 0.22
FLIGHT_ATTITUDE_PITCH_FRAME_FRACTION = 0.18
FLIGHT_ATTITUDE_CHANGE_PER_SECOND_DEG = 18.0
FLIGHT_ROLL_RANGE_DEG = (-12.0, 12.0)
FLIGHT_CAMERA_APPROACH_DISTANCE_RANGE_CM = (220.0, 950.0)
FLIGHT_CAMERA_SPLINE_LATERAL_OFFSET_RANGE_CM = (-180.0, 180.0)
FLIGHT_CAMERA_SPLINE_VERTICAL_OFFSET_RANGE_CM = (-30.0, 220.0)
FLIGHT_CAMERA_YAW_FRAME_FRACTION = 0.24
FLIGHT_CAMERA_PITCH_FRAME_FRACTION = 0.20
FLIGHT_CAMERA_ROLL_RANGE_DEG = (-8.0, 8.0)
FLIGHT_CAMERA_MIN_GATE_POINTS_IN_FRAME = 1
