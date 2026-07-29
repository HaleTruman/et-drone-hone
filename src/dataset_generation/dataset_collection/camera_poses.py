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
)
from src.dataset_generation.gate_randomization.track import random_location
from src.dataset_generation.gate_randomization.track import clamped_vector
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
