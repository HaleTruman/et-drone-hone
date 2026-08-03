"""Camera pose generation for dataset capture."""

import math
import random

import unreal

from src.dataset_generation.common import distance
from src.dataset_generation.config import (
    X_RANGE_CM, Y_RANGE_CM, Z_RANGE_CM, MIN_CAMERA_GATE_DISTANCE_CM, MAX_CAMERA_GATE_DISTANCE_CM,
    MAX_CAMERA_POSE_ATTEMPTS, PITCH_RANGE_DEG, ROLL_DEG, CAMERA_FRAME_MARGIN,
    OBSTACLE_CLEARANCE_CM, FRAME_WIDTH, FRAME_HEIGHT, GATE_CORNER_NAMES, BBOX_CORNER_NAMES,
    ENABLE_COVERAGE_CAMERA_SAMPLING, CAMERA_CANDIDATES_PER_POSE,
    CAMERA_ACCEPT_SCORE_THRESHOLD, CAMERA_GATE_DISTANCE_BINS_CM,
    CAMERA_GATE_AZIMUTH_BINS_DEG, CAMERA_GATE_ELEVATION_BINS_DEG,
    CAMERA_IMAGE_TARGET_BINS, CAMERA_BBOX_AREA_BINS,
    CAMERA_TRUNCATION_TARGET_FRACTION, CAMERA_HARD_NEGATIVE_TARGET_FRACTION,
    ENABLE_SPLINE_FLIGHT_CAMERA_SAMPLING,
    ENABLE_CONTINUOUS_SPLINE_FLIGHT_CAPTURE,
    FLIGHT_SIM_DT_RANGE_SECONDS,
    FLIGHT_SPEED_RANGE_MPS,
    FLIGHT_MAX_ACCEL_MPS2,
    FLIGHT_LOOKAHEAD_TIME_SECONDS,
    FLIGHT_LOOKAHEAD_DISTANCE_RANGE_CM,
    FLIGHT_SPLINE_OFFSET_DISTANCE_RANGE_CM,
    FLIGHT_SPLINE_LATERAL_OFFSET_RANGE_CM,
    FLIGHT_SPLINE_VERTICAL_OFFSET_RANGE_CM,
    FLIGHT_OFFSET_CHANGE_PER_SECOND_CM,
    FLIGHT_ATTITUDE_YAW_FRAME_FRACTION,
    FLIGHT_ATTITUDE_PITCH_FRAME_FRACTION,
    FLIGHT_ATTITUDE_CHANGE_PER_SECOND_DEG,
    FLIGHT_ROLL_RANGE_DEG,
    FLIGHT_CAMERA_APPROACH_DISTANCE_RANGE_CM,
    FLIGHT_CAMERA_SPLINE_LATERAL_OFFSET_RANGE_CM,
    FLIGHT_CAMERA_SPLINE_VERTICAL_OFFSET_RANGE_CM,
    FLIGHT_CAMERA_YAW_FRAME_FRACTION,
    FLIGHT_CAMERA_PITCH_FRAME_FRACTION,
    FLIGHT_CAMERA_ROLL_RANGE_DEG,
    FLIGHT_CAMERA_MIN_GATE_POINTS_IN_FRAME,
)
from src.dataset_generation.gate_randomization.track import random_location
from src.dataset_generation.gate_randomization.track import clamped_vector
from src.dataset_generation.gate_randomization.track import (
    cumulative_polyline_distances,
    sample_point_and_tangent_at_distance,
)
from src.dataset_generation.obstacles import point_clears_obstacles
from src.dataset_generation.common import component_name, dot

def random_camera_location(gate_location):
    for _ in range(100):
        candidate = random_location()
        gate_distance = distance(candidate, gate_location)
        if (
            MIN_CAMERA_GATE_DISTANCE_CM <= gate_distance <= MAX_CAMERA_GATE_DISTANCE_CM
            and point_clears_obstacles(candidate, OBSTACLE_CLEARANCE_CM)
        ):
            return candidate

    for _ in range(100):
        yaw = random.uniform(0.0, 360.0)
        pitch = random.uniform(-30.0, 30.0)
        radius = random.uniform(MIN_CAMERA_GATE_DISTANCE_CM, MAX_CAMERA_GATE_DISTANCE_CM)
        horizontal_radius = radius * math.cos(math.radians(pitch))
        candidate = clamped_vector(
            gate_location.x + math.cos(math.radians(yaw)) * horizontal_radius,
            gate_location.y + math.sin(math.radians(yaw)) * horizontal_radius,
            gate_location.z + math.sin(math.radians(pitch)) * radius,
        )
        if point_clears_obstacles(candidate, OBSTACLE_CLEARANCE_CM):
            return candidate

    raise RuntimeError(
        f"Could not find a camera position at least {OBSTACLE_CLEARANCE_CM} cm "
        "from obstacle volumes."
    )

def random_background_camera_location():
    for _ in range(MAX_CAMERA_POSE_ATTEMPTS):
        candidate = random_location()
        if point_clears_obstacles(candidate, OBSTACLE_CLEARANCE_CM):
            return candidate

    raise RuntimeError(
        f"Could not find a background camera position at least {OBSTACLE_CLEARANCE_CM} cm "
        "from obstacle volumes."
    )

def random_range(bounds):
    return random.uniform(float(bounds[0]), float(bounds[1]))

def vector_length(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)

def normalized_vector(vector):
    length = vector_length(vector)
    if length <= 0.000001:
        return unreal.Vector(0.0, 0.0, 0.0)
    return unreal.Vector(vector.x / length, vector.y / length, vector.z / length)

def scaled_vector(vector, scale):
    return unreal.Vector(vector.x * scale, vector.y * scale, vector.z * scale)

def added_vectors(*vectors):
    result = unreal.Vector(0.0, 0.0, 0.0)
    for vector in vectors:
        result += vector
    return result

def angle_360(angle_deg):
    return float(angle_deg) % 360.0

def angle_180(angle_deg):
    angle = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if angle == -180.0 else angle

def bin_name_for_range(value, named_ranges, default_name="other"):
    for name, bounds in named_ranges.items():
        lower, upper = bounds
        if lower <= value <= upper:
            return name
    return default_name

def distance_bin_name(distance_cm):
    for index, bounds in enumerate(CAMERA_GATE_DISTANCE_BINS_CM):
        if bounds[0] <= distance_cm <= bounds[1]:
            return f"distance_{index}"
    return "distance_other"

def visibility_bin_name(inside_count, projected_count):
    if inside_count <= 0:
        return "none"
    if inside_count >= projected_count:
        return "full"
    if inside_count <= 3:
        return "sparse"
    return "partial"

def random_named_range(named_ranges):
    name = random.choice(tuple(named_ranges.keys()))
    return name, named_ranges[name]

def camera_marker_components(actor, marker_names):
    components = []
    seen = set()
    for class_name in ("ArrowComponent", "SceneComponent", "ActorComponent"):
        component_class = getattr(unreal, class_name, None)
        if not component_class:
            continue
        try:
            for component in actor.get_components_by_class(component_class):
                key = str(component)
                if key not in seen:
                    seen.add(key)
                    components.append(component)
        except Exception:
            pass

    found = {}
    for component in components:
        name = component_name(component)
        for marker_name in marker_names:
            if name == marker_name or name.endswith(marker_name) or marker_name in name:
                found[marker_name] = component
    return found

def component_world_location(component):
    for getter_name in (
        "get_world_location",
        "get_component_location",
        "k2_get_component_location",
    ):
        get_location = getattr(component, getter_name, None)
        if get_location:
            try:
                return get_location()
            except Exception:
                pass
    return None

def gate_projection_points(gate):
    points = []
    for marker_names in (GATE_CORNER_NAMES, BBOX_CORNER_NAMES):
        for component in camera_marker_components(gate, marker_names).values():
            location = component_world_location(component)
            if location is not None:
                points.append(location)

    if points:
        return points

    try:
        origin, extent = gate.get_actor_bounds(False)
    except Exception:
        return [gate.get_actor_location()]

    return [
        unreal.Vector(origin.x + sx * extent.x, origin.y + sy * extent.y, origin.z + sz * extent.z)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]

def project_point_for_pose(camera, camera_location, camera_rotation, world_location):
    math_library = unreal.MathLibrary
    forward = math_library.get_forward_vector(camera_rotation)
    right = math_library.get_right_vector(camera_rotation)
    up = math_library.get_up_vector(camera_rotation)
    delta = world_location - camera_location
    camera_x = dot(delta, forward)
    camera_y = dot(delta, right)
    camera_z = dot(delta, up)
    if camera_x <= 0.001:
        return None

    horizontal_fov, aspect_ratio = camera_horizontal_fov_and_aspect(camera)
    vertical_fov = math.degrees(
        2.0 * math.atan(math.tan(math.radians(horizontal_fov) * 0.5) / aspect_ratio)
    )
    focal_x = (FRAME_WIDTH * 0.5) / math.tan(math.radians(horizontal_fov) * 0.5)
    focal_y = (FRAME_HEIGHT * 0.5) / math.tan(math.radians(vertical_fov) * 0.5)
    pixel_x = (FRAME_WIDTH * 0.5) + (camera_y / camera_x) * focal_x
    pixel_y = (FRAME_HEIGHT * 0.5) - (camera_z / camera_x) * focal_y
    return {
        "x": float(pixel_x),
        "y": float(pixel_y),
        "inside": bool(0.0 <= pixel_x < FRAME_WIDTH and 0.0 <= pixel_y < FRAME_HEIGHT),
    }

def pose_projection_summary(camera, gate, camera_location, camera_rotation):
    pixels = [
        project_point_for_pose(camera, camera_location, camera_rotation, point)
        for point in gate_projection_points(gate)
    ]
    pixels = [pixel for pixel in pixels if pixel is not None]
    if not pixels:
        return {
            "projected_count": 0,
            "inside_count": 0,
            "intersects_frame": False,
            "area_fraction": 0.0,
            "center_norm_x": 0.0,
            "center_norm_y": 0.0,
            "truncated": False,
        }

    min_x = min(pixel["x"] for pixel in pixels)
    max_x = max(pixel["x"] for pixel in pixels)
    min_y = min(pixel["y"] for pixel in pixels)
    max_y = max(pixel["y"] for pixel in pixels)
    clipped_min_x = max(0.0, min(float(FRAME_WIDTH - 1), min_x))
    clipped_max_x = max(0.0, min(float(FRAME_WIDTH - 1), max_x))
    clipped_min_y = max(0.0, min(float(FRAME_HEIGHT - 1), min_y))
    clipped_max_y = max(0.0, min(float(FRAME_HEIGHT - 1), max_y))
    intersects_frame = not (
        max_x < 0.0 or max_y < 0.0 or min_x >= FRAME_WIDTH or min_y >= FRAME_HEIGHT
    )
    inside_count = sum(1 for pixel in pixels if pixel["inside"])
    area = max(0.0, clipped_max_x - clipped_min_x) * max(0.0, clipped_max_y - clipped_min_y)
    area_fraction = area / float(FRAME_WIDTH * FRAME_HEIGHT)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    return {
        "projected_count": len(pixels),
        "inside_count": inside_count,
        "intersects_frame": bool(intersects_frame),
        "area_fraction": float(area_fraction),
        "center_norm_x": float((center_x - FRAME_WIDTH * 0.5) / (FRAME_WIDTH * 0.5)),
        "center_norm_y": float((center_y - FRAME_HEIGHT * 0.5) / (FRAME_HEIGHT * 0.5)),
        "truncated": bool(intersects_frame and inside_count < len(pixels)),
    }

class CameraCoverageState:
    def __init__(self):
        self.counts = {}
        self.pose_count = 0
        self.hard_negative_count = 0
        self.truncated_count = 0

    def count(self, key):
        return self.counts.get(key, 0)

    def register(self, descriptor):
        self.pose_count += 1
        if descriptor.get("hard_negative"):
            self.hard_negative_count += 1
        if descriptor.get("truncated"):
            self.truncated_count += 1
        for key in descriptor_keys(descriptor):
            self.counts[key] = self.counts.get(key, 0) + 1

    def wants_hard_negative(self):
        if self.pose_count <= 0:
            return False
        return (self.hard_negative_count / float(self.pose_count)) < CAMERA_HARD_NEGATIVE_TARGET_FRACTION

    def wants_truncation(self):
        if self.pose_count <= 0:
            return False
        return (self.truncated_count / float(self.pose_count)) < CAMERA_TRUNCATION_TARGET_FRACTION

    def compact_summary(self):
        return {
            "poses": int(self.pose_count),
            "truncated": int(self.truncated_count),
            "hard_negative": int(self.hard_negative_count),
            "counts": {
                f"{key[0]}:{key[1]}": int(value)
                for key, value in sorted(self.counts.items())
            },
        }

def descriptor_keys(descriptor):
    return (
        ("distance", descriptor["distance_bin"]),
        ("azimuth", descriptor["azimuth_bin"]),
        ("elevation", descriptor["elevation_bin"]),
        ("image", descriptor["image_bin"]),
        ("area", descriptor["area_bin"]),
        ("visibility", descriptor["visibility_bin"]),
        ("truncation", "truncated" if descriptor["truncated"] else "untruncated"),
    )

def coverage_score(state, descriptor):
    if not state:
        return 1.0

    score = 0.0
    for key in descriptor_keys(descriptor):
        score += 1.0 / float(1 + state.count(key))

    if descriptor.get("truncated") and state.wants_truncation():
        score += 2.0
    if descriptor.get("hard_negative") and state.wants_hard_negative():
        score += 1.5
    if descriptor["visibility_bin"] == "none" and not state.wants_hard_negative():
        score -= 2.0
    return score / float(len(descriptor_keys(descriptor)))

def pose_descriptor(camera, gate, camera_location, camera_rotation, azimuth_name, elevation_name):
    gate_center = actor_bounds_center(gate)
    gate_distance = distance(camera_location, gate_center)
    summary = pose_projection_summary(camera, gate, camera_location, camera_rotation)
    image_bin = image_bin_name(summary["center_norm_x"], summary["center_norm_y"])
    area_bin = bin_name_for_range(
        summary["area_fraction"],
        CAMERA_BBOX_AREA_BINS,
        default_name="area_other",
    )
    visibility_bin = visibility_bin_name(
        summary["inside_count"],
        max(1, summary["projected_count"]),
    )
    return {
        "distance_bin": distance_bin_name(gate_distance),
        "azimuth_bin": azimuth_name,
        "elevation_bin": elevation_name,
        "image_bin": image_bin,
        "area_bin": area_bin,
        "visibility_bin": visibility_bin,
        "truncated": bool(summary["truncated"]),
        "hard_negative": bool(summary["intersects_frame"] is False or summary["inside_count"] <= 0),
        "intersects_frame": bool(summary["intersects_frame"]),
        "area_fraction": summary["area_fraction"],
        "inside_count": summary["inside_count"],
    }

def image_bin_name(center_norm_x, center_norm_y):
    x = float(center_norm_x)
    y = float(center_norm_y)
    if abs(x) <= 0.3 and abs(y) <= 0.3:
        return "center"
    if abs(x) >= 0.72 and abs(y) >= 0.58:
        return "corner"
    if x < -0.45:
        return "left_edge"
    if x > 0.45:
        return "right_edge"
    if y < -0.45:
        return "top_edge"
    if y > 0.45:
        return "bottom_edge"
    return "off_center"

def candidate_camera_pose(camera, target_gate, coverage_state):
    gate_center = actor_bounds_center(target_gate)
    gate_rotation = target_gate.get_actor_rotation()
    math_library = unreal.MathLibrary
    gate_forward = math_library.get_forward_vector(gate_rotation)
    gate_right = math_library.get_right_vector(gate_rotation)
    gate_up = math_library.get_up_vector(gate_rotation)

    distance_name, distance_bounds = random_named_range(
        {f"distance_{index}": bounds for index, bounds in enumerate(CAMERA_GATE_DISTANCE_BINS_CM)}
    )
    azimuth_name, azimuth_bounds = random_named_range(CAMERA_GATE_AZIMUTH_BINS_DEG)
    elevation_name, elevation_bounds = random_named_range(CAMERA_GATE_ELEVATION_BINS_DEG)
    image_name, image_bounds = random_named_range(CAMERA_IMAGE_TARGET_BINS)

    radius = random_range(distance_bounds)
    azimuth = math.radians(angle_360(random_range(azimuth_bounds)))
    elevation = math.radians(random_range(elevation_bounds))
    horizontal = math.cos(elevation)
    radial = added_vectors(
        scaled_vector(gate_forward, -math.cos(azimuth) * horizontal),
        scaled_vector(gate_right, math.sin(azimuth) * horizontal),
        scaled_vector(gate_up, math.sin(elevation)),
    )
    radial = normalized_vector(radial)
    camera_location = clamped_vector(
        gate_center.x + radial.x * radius,
        gate_center.y + radial.y * radius,
        gate_center.z + radial.z * radius,
    )
    if not point_clears_obstacles(camera_location, OBSTACLE_CLEARANCE_CM):
        return None

    look_at = unreal.MathLibrary.find_look_at_rotation(camera_location, gate_center)
    max_yaw_offset, max_pitch_offset = camera_frame_half_angles(camera)
    if image_name == "corner":
        target_x = random.choice((-1.0, 1.0)) * random.uniform(0.55, 0.9)
        target_y = random.choice((-1.0, 1.0)) * random.uniform(0.55, 0.9)
    else:
        target_x = random.uniform(float(image_bounds[0]), float(image_bounds[1]))
        target_y = random.uniform(float(image_bounds[2]), float(image_bounds[3]))

    if coverage_state and coverage_state.wants_hard_negative() and random.random() < 0.35:
        target_x = random.choice((-1.15, 1.15)) + random.uniform(-0.18, 0.18)
        target_y = random.uniform(-0.9, 0.9)
    elif coverage_state and coverage_state.wants_truncation() and random.random() < 0.45:
        if random.random() < 0.5:
            target_x = random.choice((-0.92, 0.92)) + random.uniform(-0.12, 0.12)
        else:
            target_y = random.choice((-0.92, 0.92)) + random.uniform(-0.12, 0.12)

    yaw_offset = -target_x * max_yaw_offset
    pitch_offset = target_y * max_pitch_offset
    camera_rotation = unreal.Rotator(
        pitch=look_at.pitch + pitch_offset,
        yaw=look_at.yaw + yaw_offset,
        roll=ROLL_DEG,
    )
    if not PITCH_RANGE_DEG[0] <= camera_rotation.pitch <= PITCH_RANGE_DEG[1]:
        return None

    descriptor = pose_descriptor(
        camera,
        target_gate,
        camera_location,
        camera_rotation,
        azimuth_name,
        elevation_name,
    )
    descriptor["distance_bin"] = distance_name
    return camera_location, camera_rotation, gate_center, yaw_offset, pitch_offset, descriptor

def coverage_camera_pose_for_gate(camera, target_gate, coverage_state):
    best = None
    best_score = None
    for _ in range(CAMERA_CANDIDATES_PER_POSE):
        candidate = candidate_camera_pose(camera, target_gate, coverage_state)
        if not candidate:
            continue
        descriptor = candidate[-1]
        score = coverage_score(coverage_state, descriptor)
        valid_positive = descriptor["intersects_frame"] and descriptor["inside_count"] > 0
        valid_hard_negative = descriptor["hard_negative"] and coverage_state and coverage_state.wants_hard_negative()
        if valid_positive or valid_hard_negative:
            if best is None or score > best_score:
                best = candidate
                best_score = score
            if score >= CAMERA_ACCEPT_SCORE_THRESHOLD and (
                descriptor["truncated"] == bool(coverage_state and coverage_state.wants_truncation())
                or descriptor["hard_negative"] == bool(coverage_state and coverage_state.wants_hard_negative())
            ):
                break

    if not best:
        return None

    if coverage_state:
        coverage_state.register(best[-1])
    return best[:-1]

def actor_bounds_center(actor):
    try:
        origin, extent = actor.get_actor_bounds(False)
    except Exception:
        return actor.get_actor_location()

    return origin

def random_point_in_actor_bounds(actor):
    try:
        origin, extent = actor.get_actor_bounds(False)
    except Exception:
        return actor.get_actor_location()

    return unreal.Vector(
        random.uniform(origin.x - extent.x * 0.35, origin.x + extent.x * 0.35),
        random.uniform(origin.y - extent.y * 0.35, origin.y + extent.y * 0.35),
        random.uniform(origin.z - extent.z * 0.35, origin.z + extent.z * 0.35),
    )

def camera_component(camera):
    get_component = getattr(camera, "get_cine_camera_component", None)
    if get_component:
        component = get_component()
        if component:
            return component

    get_component = getattr(camera, "get_camera_component", None)
    if get_component:
        component = get_component()
        if component:
            return component

    return None

def camera_frame_half_angles(camera):
    horizontal_fov, aspect_ratio = camera_horizontal_fov_and_aspect(camera)

    vertical_fov = math.degrees(
        2.0 * math.atan(math.tan(math.radians(horizontal_fov) * 0.5) / aspect_ratio)
    )
    return horizontal_fov * 0.5 * CAMERA_FRAME_MARGIN, vertical_fov * 0.5 * CAMERA_FRAME_MARGIN

def camera_horizontal_fov_and_aspect(camera):
    component = camera_component(camera)
    horizontal_fov = 90.0
    aspect_ratio = 16.0 / 9.0
    if component:
        for property_name in ("field_of_view", "FieldOfView"):
            try:
                horizontal_fov = float(component.get_editor_property(property_name))
                break
            except Exception:
                pass

        for property_name in ("aspect_ratio", "AspectRatio"):
            try:
                aspect_ratio = float(component.get_editor_property(property_name))
                break
            except Exception:
                pass

    return horizontal_fov, aspect_ratio

def random_camera_rotation(camera, camera_location, gate_center):
    look_at = unreal.MathLibrary.find_look_at_rotation(camera_location, gate_center)
    max_yaw_offset, max_pitch_offset = camera_frame_half_angles(camera)
    yaw_offset = random.uniform(-max_yaw_offset, max_yaw_offset)
    pitch_offset = random.uniform(-max_pitch_offset, max_pitch_offset)

    return unreal.Rotator(
        pitch=look_at.pitch + pitch_offset,
        yaw=look_at.yaw + yaw_offset,
        roll=ROLL_DEG,
    ), yaw_offset, pitch_offset

def legacy_camera_pose_for_gate(camera, target_gate):
    gate_location = target_gate.get_actor_location()
    gate_center = actor_bounds_center(target_gate)
    last_pose = None

    for _ in range(MAX_CAMERA_POSE_ATTEMPTS):
        camera_location = random_camera_location(gate_location)
        camera_rotation, yaw_offset, pitch_offset = random_camera_rotation(
            camera,
            camera_location,
            gate_center,
        )
        last_pose = (camera_location, camera_rotation, gate_center, yaw_offset, pitch_offset)
        if PITCH_RANGE_DEG[0] <= camera_rotation.pitch <= PITCH_RANGE_DEG[1]:
            return last_pose

    unreal.log_warning(
        "Could not find a camera pose within PITCH_RANGE_DEG after "
        f"{MAX_CAMERA_POSE_ATTEMPTS} attempts; using last valid look-at pose. "
        f"Last pitch={last_pose[1].pitch if last_pose else 'n/a'} range={PITCH_RANGE_DEG}"
    )
    return last_pose

def camera_pose_for_gate(camera, target_gate):
    if ENABLE_COVERAGE_CAMERA_SAMPLING:
        pose = coverage_camera_pose_for_gate(camera, target_gate, None)
        if pose:
            return pose
    return legacy_camera_pose_for_gate(camera, target_gate)

def coverage_camera_pose_for_gate_or_fallback(camera, target_gate, coverage_state):
    if ENABLE_COVERAGE_CAMERA_SAMPLING:
        pose = coverage_camera_pose_for_gate(camera, target_gate, coverage_state)
        if pose:
            return pose
        unreal.log_warning(
            "Coverage camera sampler could not find a valid pose; falling back to "
            "legacy random camera pose generation."
        )
    return legacy_camera_pose_for_gate(camera, target_gate)


def flight_camera_pose_for_gate(
    camera,
    target_gate,
    track_layout,
    gate_index,
    pose_index,
    poses_per_gate,
):
    if not ENABLE_SPLINE_FLIGHT_CAMERA_SAMPLING or not track_layout:
        return None

    spline_points = track_layout.get("spline_points") or []
    gate_distances = track_layout.get("gate_distances") or []
    if not spline_points or gate_index >= len(gate_distances):
        return None

    cumulative = cumulative_polyline_distances(spline_points)
    if not cumulative or cumulative[-1] <= 0.0:
        return None

    target_distance = float(gate_distances[gate_index])
    gate_center = actor_bounds_center(target_gate)
    max_yaw_offset, max_pitch_offset = camera_frame_half_angles(camera)
    pose_alpha = (float(pose_index) + random.uniform(0.15, 0.85)) / max(1.0, float(poses_per_gate))
    far_approach = float(FLIGHT_CAMERA_APPROACH_DISTANCE_RANGE_CM[1])
    near_approach = float(FLIGHT_CAMERA_APPROACH_DISTANCE_RANGE_CM[0])
    approach_distance = far_approach + (near_approach - far_approach) * pose_alpha

    best_pose = None
    best_score = None
    for _ in range(MAX_CAMERA_POSE_ATTEMPTS):
        jittered_distance = approach_distance + random.uniform(-120.0, 120.0)
        camera_path_distance = target_distance - max(0.0, jittered_distance)
        if camera_path_distance <= 0.0 and target_distance < near_approach:
            camera_path_distance = random.uniform(0.0, min(cumulative[-1], near_approach))

        sample = sample_point_and_tangent_at_distance(
            spline_points,
            cumulative,
            camera_path_distance,
        )
        path_rotation = unreal.MathLibrary.find_look_at_rotation(
            unreal.Vector(0.0, 0.0, 0.0),
            sample["tangent"],
        )
        right = unreal.MathLibrary.get_right_vector(path_rotation)
        up = unreal.Vector(0.0, 0.0, 1.0)
        lateral_offset = random.uniform(*FLIGHT_CAMERA_SPLINE_LATERAL_OFFSET_RANGE_CM)
        vertical_offset = random.uniform(*FLIGHT_CAMERA_SPLINE_VERTICAL_OFFSET_RANGE_CM)
        camera_location = clamped_vector(
            sample["location"].x + right.x * lateral_offset,
            sample["location"].y + right.y * lateral_offset,
            sample["location"].z + up.z * vertical_offset,
        )
        if not point_clears_obstacles(camera_location, OBSTACLE_CLEARANCE_CM):
            continue

        look_at = unreal.MathLibrary.find_look_at_rotation(camera_location, gate_center)
        yaw_offset = random.uniform(
            -max_yaw_offset * FLIGHT_CAMERA_YAW_FRAME_FRACTION,
            max_yaw_offset * FLIGHT_CAMERA_YAW_FRAME_FRACTION,
        )
        pitch_offset = random.uniform(
            -max_pitch_offset * FLIGHT_CAMERA_PITCH_FRAME_FRACTION,
            max_pitch_offset * FLIGHT_CAMERA_PITCH_FRAME_FRACTION,
        )
        camera_rotation = unreal.Rotator(
            pitch=look_at.pitch + pitch_offset,
            yaw=look_at.yaw + yaw_offset,
            roll=random.uniform(*FLIGHT_CAMERA_ROLL_RANGE_DEG),
        )
        if not PITCH_RANGE_DEG[0] <= camera_rotation.pitch <= PITCH_RANGE_DEG[1]:
            continue

        center_projection = project_point_for_pose(
            camera,
            camera_location,
            camera_rotation,
            gate_center,
        )
        summary = pose_projection_summary(camera, target_gate, camera_location, camera_rotation)
        if (
            not center_projection
            or not center_projection["inside"]
            or summary["inside_count"] < FLIGHT_CAMERA_MIN_GATE_POINTS_IN_FRAME
        ):
            continue

        center_score = abs(center_projection["x"] - FRAME_WIDTH * 0.5) / FRAME_WIDTH
        center_score += abs(center_projection["y"] - FRAME_HEIGHT * 0.5) / FRAME_HEIGHT
        attitude_score = abs(yaw_offset) / max(1.0, max_yaw_offset)
        attitude_score += abs(pitch_offset) / max(1.0, max_pitch_offset)
        score = center_score - attitude_score * 0.15
        candidate = (camera_location, camera_rotation, gate_center, yaw_offset, pitch_offset)
        if best_pose is None or score < best_score:
            best_pose = candidate
            best_score = score
            if center_score < 0.12:
                break

    return best_pose


def flight_camera_pose_for_gate_or_fallback(
    camera,
    target_gate,
    track_layout,
    gate_index,
    pose_index,
    poses_per_gate,
    coverage_state,
):
    pose = flight_camera_pose_for_gate(
        camera,
        target_gate,
        track_layout,
        gate_index,
        pose_index,
        poses_per_gate,
    )
    if pose:
        return pose

    if ENABLE_SPLINE_FLIGHT_CAMERA_SAMPLING and track_layout:
        unreal.log_warning(
            "Spline-flight camera sampler could not keep the target gate in frame; "
            "falling back to coverage camera pose generation."
        )
    return coverage_camera_pose_for_gate_or_fallback(camera, target_gate, coverage_state)

def random_background_camera_pose(camera):
    camera_location = random_background_camera_location()
    camera_rotation = unreal.Rotator(
        pitch=random.uniform(*PITCH_RANGE_DEG),
        yaw=random.uniform(0.0, 360.0),
        roll=ROLL_DEG,
    )
    look_target = None
    yaw_offset = 0.0
    pitch_offset = 0.0
    return camera_location, camera_rotation, look_target, yaw_offset, pitch_offset


def clamp_value(value, bounds):
    return max(float(bounds[0]), min(float(bounds[1]), float(value)))


def approach_value(current, target, max_delta):
    delta = float(target) - float(current)
    max_delta = abs(float(max_delta))
    if delta > max_delta:
        return float(current) + max_delta
    if delta < -max_delta:
        return float(current) - max_delta
    return float(target)


def nearest_upcoming_gate(gates, gate_distances, spline_distance_cm):
    if not gates or not gate_distances:
        return None, None, None

    best_index = None
    best_delta = None
    for gate_index, gate_distance in enumerate(gate_distances[: len(gates)]):
        delta = float(gate_distance) - float(spline_distance_cm)
        if delta >= 0.0 and (best_delta is None or delta < best_delta):
            best_index = gate_index
            best_delta = delta

    if best_index is None:
        best_index = len(gates) - 1
        best_delta = float(gate_distances[best_index]) - float(spline_distance_cm)

    return best_index, gates[best_index], best_delta


def flight_sample_metadata(
    frame_index,
    time_seconds,
    dt_seconds,
    spline_distance_cm,
    speed_mps,
    camera_location,
    previous_location,
    tangent,
    target_gate,
    target_gate_distance_delta_cm,
    yaw_offset,
    pitch_offset,
    lateral_offset,
    vertical_offset,
):
    velocity_cmps = normalized_vector(tangent)
    velocity_cmps = scaled_vector(velocity_cmps, float(speed_mps) * 100.0)
    if previous_location is not None and dt_seconds > 0.0:
        velocity_cmps = scaled_vector(camera_location - previous_location, 1.0 / dt_seconds)

    return {
        "mode": "continuous_spline_flight",
        "frame_index": int(frame_index),
        "time_seconds": float(time_seconds),
        "dt_seconds": float(dt_seconds),
        "spline_distance_m": float(spline_distance_cm) / 100.0,
        "speed_mps": float(speed_mps),
        "velocity_cm_per_second": velocity_cmps,
        "target_gate_label": target_gate.get_actor_label() if target_gate else None,
        "target_gate_distance_delta_m": (
            float(target_gate_distance_delta_cm) / 100.0
            if target_gate_distance_delta_cm is not None
            else None
        ),
        "lateral_offset_m": float(lateral_offset) / 100.0,
        "vertical_offset_m": float(vertical_offset) / 100.0,
        "spline_offset_distance_m": math.sqrt(
            float(lateral_offset) * float(lateral_offset)
            + float(vertical_offset) * float(vertical_offset)
        )
        / 100.0,
        "yaw_offset_deg": float(yaw_offset),
        "pitch_offset_deg": float(pitch_offset),
    }


def random_flight_spline_offset():
    min_distance = max(0.0, float(FLIGHT_SPLINE_OFFSET_DISTANCE_RANGE_CM[0]))
    max_distance = max(min_distance, float(FLIGHT_SPLINE_OFFSET_DISTANCE_RANGE_CM[1]))
    distance_cm = random.uniform(min_distance, max_distance)
    angle = random.uniform(0.0, math.pi)
    lateral_offset = math.cos(angle) * distance_cm
    vertical_offset = math.sin(angle) * distance_cm
    return clamp_flight_spline_offset(lateral_offset, vertical_offset)


def clamp_flight_spline_offset(lateral_offset, vertical_offset):
    lateral_offset = clamp_value(lateral_offset, FLIGHT_SPLINE_LATERAL_OFFSET_RANGE_CM)
    vertical_offset = clamp_value(vertical_offset, FLIGHT_SPLINE_VERTICAL_OFFSET_RANGE_CM)
    max_distance = max(0.0, float(FLIGHT_SPLINE_OFFSET_DISTANCE_RANGE_CM[1]))
    magnitude = math.sqrt(lateral_offset * lateral_offset + vertical_offset * vertical_offset)
    if magnitude > max_distance and magnitude > 0.000001:
        scale = max_distance / magnitude
        lateral_offset *= scale
        vertical_offset *= scale
    return lateral_offset, vertical_offset


def continuous_spline_flight_poses(camera, gates, track_layout):
    if (
        not ENABLE_CONTINUOUS_SPLINE_FLIGHT_CAPTURE
        or not ENABLE_SPLINE_FLIGHT_CAMERA_SAMPLING
        or not track_layout
    ):
        return None

    spline_points = track_layout.get("spline_points") or []
    if len(spline_points) < 2:
        return None

    cumulative = cumulative_polyline_distances(spline_points)
    if not cumulative or cumulative[-1] <= 0.0:
        return None

    total_length_cm = float(cumulative[-1])
    gate_distances = track_layout.get("gate_distances") or []
    max_yaw_offset, max_pitch_offset = camera_frame_half_angles(camera)

    poses = []
    time_seconds = 0.0
    spline_distance_cm = 0.0
    speed_mps = random.uniform(*FLIGHT_SPEED_RANGE_MPS)
    target_speed_mps = random.uniform(*FLIGHT_SPEED_RANGE_MPS)
    lateral_offset, vertical_offset = random_flight_spline_offset()
    yaw_offset = random.uniform(
        -max_yaw_offset * FLIGHT_ATTITUDE_YAW_FRAME_FRACTION,
        max_yaw_offset * FLIGHT_ATTITUDE_YAW_FRAME_FRACTION,
    )
    pitch_offset = random.uniform(
        -max_pitch_offset * FLIGHT_ATTITUDE_PITCH_FRAME_FRACTION,
        max_pitch_offset * FLIGHT_ATTITUDE_PITCH_FRAME_FRACTION,
    )
    roll = random.uniform(*FLIGHT_ROLL_RANGE_DEG) * 0.25
    previous_location = None
    previous_yaw = None
    frame_index = 0

    while spline_distance_cm <= total_length_cm:
        sample = sample_point_and_tangent_at_distance(
            spline_points,
            cumulative,
            spline_distance_cm,
        )
        path_rotation = unreal.MathLibrary.find_look_at_rotation(
            unreal.Vector(0.0, 0.0, 0.0),
            sample["tangent"],
        )
        right = unreal.MathLibrary.get_right_vector(path_rotation)
        camera_location = clamped_vector(
            sample["location"].x + right.x * lateral_offset,
            sample["location"].y + right.y * lateral_offset,
            sample["location"].z + vertical_offset,
        )
        if not point_clears_obstacles(camera_location, OBSTACLE_CLEARANCE_CM):
            fallback_vertical = min(
                50.0,
                float(FLIGHT_SPLINE_OFFSET_DISTANCE_RANGE_CM[1]),
            )
            camera_location = clamped_vector(
                sample["location"].x,
                sample["location"].y,
                sample["location"].z + fallback_vertical,
            )
            if not point_clears_obstacles(camera_location, OBSTACLE_CLEARANCE_CM):
                spline_distance_cm = min(
                    total_length_cm,
                    spline_distance_cm + max(100.0, speed_mps * 100.0 * 0.1),
                )
                continue

        gate_index, target_gate, target_gate_delta = nearest_upcoming_gate(
            gates,
            gate_distances,
            spline_distance_cm,
        )
        speed_lookahead_cm = speed_mps * 100.0 * FLIGHT_LOOKAHEAD_TIME_SECONDS
        lookahead_distance = clamp_value(
            speed_lookahead_cm,
            FLIGHT_LOOKAHEAD_DISTANCE_RANGE_CM,
        )
        lookahead_sample = sample_point_and_tangent_at_distance(
            spline_points,
            cumulative,
            spline_distance_cm + lookahead_distance,
        )
        aim_target = lookahead_sample["location"]
        if target_gate and target_gate_delta is not None and target_gate_delta <= lookahead_distance * 1.6:
            aim_target = actor_bounds_center(target_gate)

        look_at = unreal.MathLibrary.find_look_at_rotation(camera_location, aim_target)
        turn_rate = 0.0
        if previous_yaw is not None:
            turn_rate = angle_180(look_at.yaw - previous_yaw)
        previous_yaw = look_at.yaw
        target_roll = clamp_value(-turn_rate * 0.65, FLIGHT_ROLL_RANGE_DEG)
        roll = approach_value(roll, target_roll, FLIGHT_ATTITUDE_CHANGE_PER_SECOND_DEG * 0.1)

        camera_rotation = unreal.Rotator(
            pitch=look_at.pitch + pitch_offset,
            yaw=look_at.yaw + yaw_offset,
            roll=roll,
        )

        poses.append(
            {
                "gate": target_gate,
                "location": camera_location,
                "rotation": camera_rotation,
                "target": aim_target,
                "yaw_offset": yaw_offset,
                "pitch_offset": pitch_offset,
                "show_gates": True,
                "flight": flight_sample_metadata(
                    frame_index,
                    time_seconds,
                    0.0 if frame_index == 0 else dt_seconds,
                    spline_distance_cm,
                    speed_mps,
                    camera_location,
                    previous_location,
                    sample["tangent"],
                    target_gate,
                    target_gate_delta,
                    yaw_offset,
                    pitch_offset,
                    lateral_offset,
                    vertical_offset,
                ),
            }
        )
        previous_location = camera_location
        frame_index += 1

        if spline_distance_cm >= total_length_cm:
            break

        dt_seconds = random.uniform(*FLIGHT_SIM_DT_RANGE_SECONDS)
        if random.random() < 0.18:
            target_speed_mps = random.uniform(*FLIGHT_SPEED_RANGE_MPS)
        speed_mps = approach_value(
            speed_mps,
            target_speed_mps,
            FLIGHT_MAX_ACCEL_MPS2 * dt_seconds,
        )
        speed_mps = clamp_value(speed_mps, FLIGHT_SPEED_RANGE_MPS)
        spline_distance_cm = min(total_length_cm, spline_distance_cm + speed_mps * 100.0 * dt_seconds)
        time_seconds += dt_seconds

        target_lateral_offset, target_vertical_offset = random_flight_spline_offset()
        lateral_offset = approach_value(
            lateral_offset,
            target_lateral_offset,
            FLIGHT_OFFSET_CHANGE_PER_SECOND_CM * dt_seconds,
        )
        vertical_offset = approach_value(
            vertical_offset,
            target_vertical_offset,
            FLIGHT_OFFSET_CHANGE_PER_SECOND_CM * dt_seconds,
        )
        lateral_offset, vertical_offset = clamp_flight_spline_offset(
            lateral_offset,
            vertical_offset,
        )
        yaw_offset = approach_value(
            yaw_offset,
            random.uniform(
                -max_yaw_offset * FLIGHT_ATTITUDE_YAW_FRAME_FRACTION,
                max_yaw_offset * FLIGHT_ATTITUDE_YAW_FRAME_FRACTION,
            ),
            FLIGHT_ATTITUDE_CHANGE_PER_SECOND_DEG * dt_seconds,
        )
        pitch_offset = approach_value(
            pitch_offset,
            random.uniform(
                -max_pitch_offset * FLIGHT_ATTITUDE_PITCH_FRAME_FRACTION,
                max_pitch_offset * FLIGHT_ATTITUDE_PITCH_FRAME_FRACTION,
            ),
            FLIGHT_ATTITUDE_CHANGE_PER_SECOND_DEG * dt_seconds,
        )

    unreal.log(
        f"Generated continuous spline flight with {len(poses)} frame(s), "
        f"path_length={total_length_cm / 100.0:.2f}m, "
        f"duration={time_seconds:.2f}s."
    )
    return poses

def set_camera_pose(
    camera,
    target_gate,
    camera_location,
    camera_rotation,
    look_target,
    yaw_offset,
    pitch_offset,
):
    camera.modify()
    camera.set_actor_location(camera_location, False, False)
    camera.set_actor_rotation(camera_rotation, False)
    target_gate_label = target_gate.get_actor_label() if target_gate else None
    unreal.log(
        "Moved camera "
        f"{camera.get_actor_label()} to location={camera.get_actor_location()} "
        f"rotation={camera.get_actor_rotation()} target_gate={target_gate_label} "
        f"gate_center={look_target} yaw_offset={yaw_offset:.2f} pitch_offset={pitch_offset:.2f}"
    )
