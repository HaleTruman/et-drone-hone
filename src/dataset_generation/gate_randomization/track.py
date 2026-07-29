"""Race-track-style random gate layout generation."""

import math
import random

import unreal

from src.dataset_generation.common import clamp, distance, lerp
from src.dataset_generation.config import (
    X_RANGE_CM, Y_RANGE_CM, Z_RANGE_CM, PITCH_RANGE_DEG, ROLL_DEG, YAW_RANGE_DEG,
    TRACK_EDGE_MARGIN_CM, TRACK_WAYPOINTS_PER_GATE, TRACK_MIN_WAYPOINTS, TRACK_MAX_WAYPOINTS,
    TRACK_SPLINE_SAMPLES_PER_SEGMENT, TRACK_STEP_RANGE_CM, TRACK_TURN_RANGE_DEG,
    TRACK_Z_STEP_RANGE_CM, TRACK_VOLUME_ANCHOR_CHANCE,
    TRACK_START_X_CM, TRACK_START_Y_CM, TRACK_START_Z_CM,
    TRACK_END_X_CM, TRACK_END_Y_CM, TRACK_END_Z_CM,
    GATE_FORWARD_YAW_OFFSET_DEG, GATE_CENTER_SPLINE_Z_OFFSET_CM,
    OBSTACLE_CLEARANCE_CM, MAX_TRACK_LAYOUT_ATTEMPTS,
)
from src.dataset_generation.obstacles import all_points_clear_obstacles

def random_location():
    return unreal.Vector(
        random.uniform(*X_RANGE_CM),
        random.uniform(*Y_RANGE_CM),
        random.uniform(*Z_RANGE_CM),
    )

def random_rotation():
    return unreal.Rotator(
        pitch=random.uniform(*PITCH_RANGE_DEG),
        yaw=random.uniform(*YAW_RANGE_DEG),
        roll=ROLL_DEG,
    )

def clamped_vector(x, y, z):
    return unreal.Vector(
        clamp(x, X_RANGE_CM),
        clamp(y, Y_RANGE_CM),
        clamp(z, Z_RANGE_CM),
    )


def configured_track_endpoint(x, y, z, label):
    if not (
        X_RANGE_CM[0] <= x <= X_RANGE_CM[1]
        and Y_RANGE_CM[0] <= y <= Y_RANGE_CM[1]
        and Z_RANGE_CM[0] <= z <= Z_RANGE_CM[1]
    ):
        raise ValueError(
            f"Configured track {label} point ({x}, {y}, {z}) is outside "
            f"X_RANGE_CM={X_RANGE_CM}, Y_RANGE_CM={Y_RANGE_CM}, Z_RANGE_CM={Z_RANGE_CM}."
        )
    return unreal.Vector(float(x), float(y), float(z))


def track_start_point():
    return configured_track_endpoint(
        TRACK_START_X_CM,
        TRACK_START_Y_CM,
        TRACK_START_Z_CM,
        "start",
    )


def track_end_point():
    return configured_track_endpoint(
        TRACK_END_X_CM,
        TRACK_END_Y_CM,
        TRACK_END_Z_CM,
        "end",
    )


def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return clamped_vector(
        0.5
        * (
            (2.0 * p1.x)
            + (-p0.x + p2.x) * t
            + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
            + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
        ),
        0.5
        * (
            (2.0 * p1.y)
            + (-p0.y + p2.y) * t
            + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
            + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
        ),
        0.5
        * (
            (2.0 * p1.z)
            + (-p0.z + p2.z) * t
            + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2
            + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3
        ),
    )

def random_track_waypoints(gate_count):
    start = track_start_point()
    end = track_end_point()

    x_min = X_RANGE_CM[0] + TRACK_EDGE_MARGIN_CM
    x_max = X_RANGE_CM[1] - TRACK_EDGE_MARGIN_CM
    y_min = Y_RANGE_CM[0] + TRACK_EDGE_MARGIN_CM
    y_max = Y_RANGE_CM[1] - TRACK_EDGE_MARGIN_CM

    waypoint_count = int(math.ceil(gate_count * TRACK_WAYPOINTS_PER_GATE))
    waypoint_count = max(TRACK_MIN_WAYPOINTS, min(TRACK_MAX_WAYPOINTS, waypoint_count))
    waypoint_count = max(2, waypoint_count)

    anchors = shuffled_volume_anchors(x_min, x_max, y_min, y_max)
    current = start
    heading = unreal.MathLibrary.find_look_at_rotation(start, end).yaw
    waypoints = [current]

    for _ in range(waypoint_count - 2):
        if anchors and random.random() < TRACK_VOLUME_ANCHOR_CHANCE:
            current = anchors.pop()
            heading = unreal.MathLibrary.find_look_at_rotation(waypoints[-1], current).yaw
        else:
            heading += random.uniform(*TRACK_TURN_RANGE_DEG)
            step = random.uniform(*TRACK_STEP_RANGE_CM)
            z_step = random.uniform(*TRACK_Z_STEP_RANGE_CM)

            next_x = current.x + math.cos(math.radians(heading)) * step
            next_y = current.y + math.sin(math.radians(heading)) * step
            next_z = current.z + z_step

            if next_x < x_min or next_x > x_max:
                heading = 180.0 - heading + random.uniform(-70.0, 70.0)
                next_x = clamp(next_x, (x_min, x_max))

            if next_y < y_min or next_y > y_max:
                heading = -heading + random.uniform(-70.0, 70.0)
                next_y = clamp(next_y, (y_min, y_max))

            current = clamped_vector(next_x, next_y, next_z)
        waypoints.append(current)

    waypoints.append(end)

    return waypoints

def shuffled_volume_anchors(x_min, x_max, y_min, y_max):
    z_min = Z_RANGE_CM[0]
    z_max = Z_RANGE_CM[1]
    z_mid = (z_min + z_max) * 0.5
    x_mid = (x_min + x_max) * 0.5
    y_mid = (y_min + y_max) * 0.5
    inset_x = (x_max - x_min) * 0.18
    inset_y = (y_max - y_min) * 0.18

    anchors = [
        unreal.Vector(x_min, y_min, random.uniform(z_min, z_max)),
        unreal.Vector(x_max, y_min, random.uniform(z_min, z_max)),
        unreal.Vector(x_min, y_max, random.uniform(z_min, z_max)),
        unreal.Vector(x_max, y_max, random.uniform(z_min, z_max)),
        unreal.Vector(x_mid, y_min, random.uniform(z_min, z_max)),
        unreal.Vector(x_mid, y_max, random.uniform(z_min, z_max)),
        unreal.Vector(x_min, y_mid, random.uniform(z_min, z_max)),
        unreal.Vector(x_max, y_mid, random.uniform(z_min, z_max)),
        unreal.Vector(
            random.uniform(x_min + inset_x, x_max - inset_x),
            random.uniform(y_min + inset_y, y_max - inset_y),
            z_mid,
        ),
    ]
    random.shuffle(anchors)
    return anchors

def spline_points(control_points, samples_per_segment=None):
    if len(control_points) <= 1:
        return control_points

    if samples_per_segment is None:
        samples_per_segment = TRACK_SPLINE_SAMPLES_PER_SEGMENT
    samples_per_segment = max(1, int(samples_per_segment))

    points = []
    for index in range(len(control_points) - 1):
        p0 = control_points[max(0, index - 1)]
        p1 = control_points[index]
        p2 = control_points[index + 1]
        p3 = control_points[min(len(control_points) - 1, index + 2)]

        for sample_index in range(samples_per_segment):
            points.append(catmull_rom(p0, p1, p2, p3, sample_index / float(samples_per_segment)))

    points.append(control_points[-1])
    return points

def sample_points_by_distance(points, count):
    return [
        gate_center_from_spline_location(sample["location"])
        for sample in sample_points_and_tangents_by_distance(points, count)
    ]


def polyline_clearance_sample_points(points, max_spacing_cm):
    if not points:
        return []

    max_spacing_cm = max(1.0, float(max_spacing_cm))
    sampled = [points[0]]
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        segment_length = distance(previous, current)
        step_count = max(1, int(math.ceil(segment_length / max_spacing_cm)))
        for step_index in range(1, step_count + 1):
            alpha = step_index / float(step_count)
            sampled.append(
                clamped_vector(
                    lerp(previous.x, current.x, alpha),
                    lerp(previous.y, current.y, alpha),
                    lerp(previous.z, current.z, alpha),
                )
            )
    return sampled


def track_path_clears_obstacles(path_points, clearance_cm):
    sample_spacing = max(50.0, float(clearance_cm) * 0.5)
    return all_points_clear_obstacles(
        polyline_clearance_sample_points(path_points, sample_spacing),
        clearance_cm,
    )


def sample_points_and_tangents_by_distance(points, count):
    if count <= 1:
        return [
            {
                "location": points[0],
                "tangent": unreal.Vector(1.0, 0.0, 0.0),
            }
        ]

    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + distance(points[index - 1], points[index]))

    total_length = cumulative[-1]
    if total_length <= 0.0:
        return [
            {
                "location": points[0],
                "tangent": unreal.Vector(1.0, 0.0, 0.0),
            }
            for _ in range(count)
        ]

    sampled = []
    point_index = 1
    for sample_index in range(count):
        target_distance = total_length * sample_index / float(count - 1)
        while point_index < len(cumulative) - 1 and cumulative[point_index] < target_distance:
            point_index += 1

        previous_distance = cumulative[point_index - 1]
        next_distance = cumulative[point_index]
        segment_length = max(0.001, next_distance - previous_distance)
        alpha = (target_distance - previous_distance) / segment_length
        previous_point = points[point_index - 1]
        next_point = points[point_index]
        location = clamped_vector(
            lerp(previous_point.x, next_point.x, alpha),
            lerp(previous_point.y, next_point.y, alpha),
            lerp(previous_point.z, next_point.z, alpha),
        )
        tangent = unreal.Vector(
            next_point.x - previous_point.x,
            next_point.y - previous_point.y,
            next_point.z - previous_point.z,
        )
        sampled.append(
            {
                "location": location,
                "tangent": tangent,
            }
        )

    return sampled

def track_gate_locations(count):
    return sample_points_by_distance(
        spline_points(random_track_waypoints(count)),
        count,
    )

def generate_track_layout(count):
    last_samples = None
    last_path_points = None
    for _ in range(MAX_TRACK_LAYOUT_ATTEMPTS):
        path_points = spline_points(random_track_waypoints(count))
        samples = sample_points_and_tangents_by_distance(
            path_points,
            count,
        )
        locations = [
            gate_center_from_spline_location(sample["location"])
            for sample in samples
        ]
        gates_clear = all_points_clear_obstacles(locations, OBSTACLE_CLEARANCE_CM)
        path_clear = track_path_clears_obstacles(path_points, OBSTACLE_CLEARANCE_CM)
        if gates_clear and path_clear:
            break
        last_samples = samples
        last_path_points = path_points
    else:
        samples = last_samples
        path_points = last_path_points
        unreal.log_warning(
            f"Could not generate a track with all gates and spline path at least "
            f"{OBSTACLE_CLEARANCE_CM} cm from obstacle volumes after "
            f"{MAX_TRACK_LAYOUT_ATTEMPTS} attempts; using last layout."
        )

    locations = [
        gate_center_from_spline_location(sample["location"])
        for sample in samples
    ]
    rotations = [
        track_gate_rotation_from_tangent(sample["tangent"])
        for sample in samples
    ]
    return {
        "locations": locations,
        "rotations": rotations,
        "spline_points": path_points,
    }


def track_gate_layout(count):
    layout = generate_track_layout(count)
    return layout["locations"], layout["rotations"]
    return locations, rotations


def gate_center_from_spline_location(location):
    return unreal.Vector(
        location.x,
        location.y,
        location.z + GATE_CENTER_SPLINE_Z_OFFSET_CM,
    )


def track_gate_rotation_from_tangent(tangent):
    if distance(tangent, unreal.Vector(0.0, 0.0, 0.0)) <= 0.001:
        return random_rotation()

    rotation = unreal.MathLibrary.find_look_at_rotation(
        unreal.Vector(0.0, 0.0, 0.0),
        tangent,
    )
    return unreal.Rotator(
        pitch=clamp(rotation.pitch, PITCH_RANGE_DEG),
        yaw=rotation.yaw + GATE_FORWARD_YAW_OFFSET_DEG,
        roll=ROLL_DEG,
    )


def track_gate_rotation(locations, index):
    if len(locations) <= 1:
        return random_rotation()

    if index < len(locations) - 1:
        tangent = unreal.Vector(
            locations[index + 1].x - locations[index].x,
            locations[index + 1].y - locations[index].y,
            locations[index + 1].z - locations[index].z,
        )
    else:
        tangent = unreal.Vector(
            locations[index].x - locations[index - 1].x,
            locations[index].y - locations[index - 1].y,
            locations[index].z - locations[index - 1].z,
        )

    return track_gate_rotation_from_tangent(tangent)

def closest_gate_spacing(locations):
    if len(locations) <= 1:
        return None

    closest = None
    for left_index in range(len(locations)):
        for right_index in range(left_index + 1, len(locations)):
            spacing = distance(locations[left_index], locations[right_index])
            if closest is None or spacing < closest:
                closest = spacing
    return closest
