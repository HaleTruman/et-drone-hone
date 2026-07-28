"""Configuration for Unreal gate dataset generation."""

# Level and actor lookup.
GATE_FOLDER = "Gates"
CAMERA_LABEL = "AI_Hangar_Wide_Long_Interior_Camera"
TARGET_LEVEL_PATH = "/Game/Hangar1"
LEVEL_PATH = TARGET_LEVEL_PATH
RENDER_TARGET_PATH = "/Game/RenderTarget1920x1080"

# Output folders and files.
DATASETS_DIRECTORY_NAME = "datasets"
DATASET_DIRECTORY_PREFIX = "dataset_"
RUNS_DIRECTORY_NAME = "runs"
RUN_DIRECTORY_PREFIX = "run_"
FRAME_DIRECTORY_NAME = "frames"
FRAME_FILE_EXTENSION = "png"
FRAME_METADATA_FILE_NAME = "metadata.json"
FRAME_METADATA_JSONL_FILE_NAME = "metadata.jsonl"

# Output toggles.
SAVE_FRAMES = True
SAVE_FRAME_METADATA = True

# Number of full randomize-and-collect batches to run from one script execution.
# Each batch creates its own run directory and is equivalent to executing the
# script again after the previous batch finishes.
RUN_COUNT = 1

# Background-only dataset collection.
# Set FORCE_ZERO_GATE_RUN=True to collect background-only frames even when gate
# actors exist in the level. Existing gate actors will be hidden for the run.
FORCE_ZERO_GATE_RUN = False

# Percentage of frames in each normal run batch that should be background-only
# zero-gate frames. Uses final-run percentage semantics:
# 20.0 means 20% zero-gate frames and 80% normal gate-targeted frames.
ZERO_GATE_FRAME_PERCENT = 20

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

# Gate and camera placement volume, in Unreal centimeters.
X_RANGE_CM = (-3000.0, 3000.0)
Y_RANGE_CM = (-7500.0, 7500.0)
Z_RANGE_CM = (150.0, 1250.0)

# Gate orientation.
PITCH_RANGE_DEG = (-15.0, 15.0)  # Unreal pitch is rotation around Y.
ROLL_DEG = 0.0
YAW_RANGE_DEG = (0.0, 360.0)  # Unreal yaw is rotation around Z.
GATE_FORWARD_YAW_OFFSET_DEG = 0.0

# Gate spacing.
MIN_GATE_DISTANCE_CM = 1000.0
GATE_CENTER_SPLINE_Z_OFFSET_CM = 60.0

# Obstacle clearance.
OBSTACLE_MAP_JSONL = "hangar_extraction/hangar_columns_pylons.jsonl"
OBSTACLE_CLEARANCE_CM = 300.0
MAX_TRACK_LAYOUT_ATTEMPTS = 100

# Track generation.
TRACK_EDGE_MARGIN_CM = 300.0
TRACK_WAYPOINTS_PER_GATE = 1.0
TRACK_MIN_WAYPOINTS = 7
TRACK_MAX_WAYPOINTS = 10
TRACK_SPLINE_SAMPLES_PER_SEGMENT = 4
TRACK_STEP_RANGE_CM = (1200.0, 4200.0)
TRACK_TURN_RANGE_DEG = (-170.0, 170.0)
TRACK_Z_STEP_RANGE_CM = (-520.0, 520.0)
TRACK_VOLUME_ANCHOR_CHANCE = 0.45
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
    TRACK_INDICATOR_SEGMENT_PREFIX,
)
TRACK_RENDER_SETTLE_SECONDS = 1.0
TRACK_RENDER_READY_BOOL_PROPERTY = "TrackRenderReady"
TRACK_RENDER_STABLE_TICKS = 5
TRACK_RENDER_MAX_WAIT_SECONDS = 30.0
COMPILE_TRACK_BLUEPRINT_AFTER_SPLINE_UPDATE = True

# Camera placement and aim.
MIN_CAMERA_GATE_DISTANCE_CM = 0.0
MAX_CAMERA_GATE_DISTANCE_CM = 1100.0
MAX_CAMERA_POSE_ATTEMPTS = 250
CAMERA_POSITIONS_PER_GATE = 5
CAMERA_HOLD_SECONDS = 0.05
CAMERA_FRAME_MARGIN = 0.95
