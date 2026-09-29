"""Race-track-style random gate layout generation."""

import math
import random

import unreal

from SyntheticData.src.dataset_generation.common import clamp, distance, lerp
from SyntheticData.src.dataset_generation.config import (
    X_RANGE_CM, Y_RANGE_CM, Z_RANGE_CM, PITCH_RANGE_DEG, ROLL_DEG, YAW_RANGE_DEG,
    TRACK_EDGE_MARGIN_CM, TRACK_WAYPOINTS_PER_GATE, TRACK_MIN_WAYPOINTS, TRACK_MAX_WAYPOINTS,
    TRACK_SPLINE_SAMPLES_PER_SEGMENT, TRACK_STEP_RANGE_CM, TRACK_TURN_RANGE_DEG,
    TRACK_Z_STEP_RANGE_CM, TRACK_VOLUME_ANCHOR_CHANCE,
    TRACK_START_X_CM, TRACK_START_Y_CM, TRACK_START_Z_CM,
    TRACK_END_X_CM, TRACK_END_Y_CM, TRACK_END_Z_CM,
    GATE_FORWARD_YAW_OFFSET_DEG, GATE_CENTER_SPLINE_Z_OFFSET_CM,
    TRACK_GATE_SPACING_WEIGHT_RANGE, OBSTACLE_CLEARANCE_CM, MAX_TRACK_LAYOUT_ATTEMPTS,
    TRACK_TOTAL_LENGTH_CM, TRACK_GATE_RANDOM_DISTANCE_MARGIN_CM,
    TRACK_BEND_COUNT_RANGE, TRACK_BEND_TURN_RANGE_DEG, TRACK_GATE_DISTANCE_RANGE_CM,
)
from SyntheticData.src.dataset_generation.obstacles import all_points_clear_obstacles

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
    x_min = X_RANGE_CM[0] + TRACK_EDGE_MARGIN_CM
    x_max = X_RANGE_CM[1] - TRACK_EDGE_MARGIN_CM
    y_min = Y_RANGE_CM[0] + TRACK_EDGE_MARGIN_CM
    y_max = Y_RANGE_CM[1] - TRACK_EDGE_MARGIN_CM

    bend_count = random.randint(
        int(TRACK_BEND_COUNT_RANGE[0]),
        int(TRACK_BEND_COUNT_RANGE[1]),
    )
    segment_count = bend_count + 1
    start_z, end_z, start_half, end_half = random_opposite_track_z_bounds()
    current = random_track_start_point(x_min, x_max, y_min, y_max, start_z)
    heading = random.uniform(0.0, 360.0)
    waypoints = [current]

    target_length = max(100.0, float(TRACK_TOTAL_LENGTH_CM))
    segment_lengths = track_segment_lengths(target_length, segment_count)

    for segment_index, segment_length in enumerate(segment_lengths):
        segment_alpha = (segment_index + 1) / float(segment_count)
        target_z = lerp(start_z, end_z, segment_alpha)
        if segment_index < segment_count - 1:
            target_z += random.uniform(*TRACK_Z_STEP_RANGE_CM) * 0.35

        if segment_index > 0:
            turn_direction = random.choice((-1.0, 1.0))
            heading += turn_direction * random.uniform(*TRACK_BEND_TURN_RANGE_DEG)
            heading += random.uniform(*TRACK_TURN_RANGE_DEG)

        current, heading = next_track_waypoint(
            current,
            heading,
            segment_length,
            x_min,
            x_max,
            y_min,
            y_max,
            target_z,
        )
        waypoints.append(current)

    unreal.log(
        f"Generated track spline z profile start_half={start_half} "
        f"start_z={start_z:.1f}cm end_half={end_half} end_z={end_z:.1f}cm."
    )
    return trim_polyline_to_length(waypoints, target_length)


def random_opposite_track_z_bounds():
    z_min = float(Z_RANGE_CM[0])
    z_max = float(Z_RANGE_CM[1])
    z_mid = (z_min + z_max) * 0.5
    start_half = random.choice(("lower", "upper"))
    end_half = "upper" if start_half == "lower" else "lower"

    if start_half == "lower":
        start_z = random.uniform(z_min, z_mid)
        end_z = random.uniform(z_mid, z_max)
    else:
        start_z = random.uniform(z_mid, z_max)
        end_z = random.uniform(z_min, z_mid)

    return start_z, end_z, start_half, end_half


def random_track_start_point(x_min, x_max, y_min, y_max, start_z):
    inset_x = min((x_max - x_min) * 0.25, TRACK_TOTAL_LENGTH_CM * 0.3)
    inset_y = min((y_max - y_min) * 0.25, TRACK_TOTAL_LENGTH_CM * 0.3)
    return unreal.Vector(
        random.uniform(x_min + inset_x, x_max - inset_x),
        random.uniform(y_min + inset_y, y_max - inset_y),
        clamp(start_z, Z_RANGE_CM),
    )


def track_segment_lengths(total_length, segment_count):
    weights = [random.uniform(0.75, 1.25) for _ in range(segment_count)]
    weight_sum = sum(weights)
    return [total_length * weight / weight_sum for weight in weights]


def next_track_waypoint(current, heading, segment_length, x_min, x_max, y_min, y_max, target_z):
    next_x = current.x + math.cos(math.radians(heading)) * segment_length
    next_y = current.y + math.sin(math.radians(heading)) * segment_length
    next_z = target_z

    if next_x < x_min or next_x > x_max:
        heading = 180.0 - heading + random.uniform(-20.0, 20.0)
        next_x = current.x + math.cos(math.radians(heading)) * segment_length

    if next_y < y_min or next_y > y_max:
        heading = -heading + random.uniform(-20.0, 20.0)
        next_y = current.y + math.sin(math.radians(heading)) * segment_length

    return (
        clamped_vector(
            clamp(next_x, (x_min, x_max)),
            clamp(next_y, (y_min, y_max)),
            next_z,
        ),
        heading,
    )


def polyline_length(points):
    total = 0.0
    for index in range(1, len(points)):
        total += distance(points[index - 1], points[index])
    return total


def trim_polyline_to_length(points, target_length):
    if len(points) <= 1:
        return points

    target_length = max(0.0, float(target_length))
    trimmed = [points[0]]
    travelled = 0.0
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        segment_length = distance(previous, current)
        if travelled + segment_length < target_length:
            trimmed.append(current)
            travelled += segment_length
            continue

        remaining = max(0.0, target_length - travelled)
        alpha = remaining / max(0.001, segment_length)
        trimmed.append(
            clamped_vector(
                lerp(previous.x, current.x, alpha),
                lerp(previous.y, current.y, alpha),
                lerp(previous.z, current.z, alpha),
            )
        )
        return trimmed

    return trimmed

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
    cumulative = cumulative_polyline_distances(points)
    if count <= 1:
        return [sample_point_and_tangent_at_distance(points, cumulative, 0.0)]

    total_length = cumulative[-1] if cumulative else 0.0
    target_distances = [
        total_length * sample_index / float(count - 1)
        for sample_index in range(count)
    ]
    return sample_points_and_tangents_at_distances(points, cumulative, target_distances)


def variable_gate_distances(total_length, count):
    if count <= 1:
        return [0.0]
    if total_length <= 0.0:
        return [0.0 for _ in range(count)]

    min_weight = max(0.05, float(TRACK_GATE_SPACING_WEIGHT_RANGE[0]))
    max_weight = max(min_weight, float(TRACK_GATE_SPACING_WEIGHT_RANGE[1]))
    weights = [random.uniform(min_weight, max_weight) for _ in range(count - 1)]
    weight_sum = sum(weights)
    distances = [0.0]
    running = 0.0
    for weight in weights[:-1]:
        running += total_length * weight / weight_sum
        distances.append(running)
    distances.append(total_length)
    return distances


def sample_points_and_tangents_by_variable_distance(points, count):
    cumulative = cumulative_polyline_distances(points)
    total_length = cumulative[-1] if cumulative else 0.0
    target_distances = variable_gate_distances(total_length, count)
    return sample_points_and_tangents_at_distances(points, cumulative, target_distances)


def random_gate_distances(total_length, count):
    if count <= 0:
        return []
    if total_length <= 0.0:
        return [0.0 for _ in range(count)]

    margin = min(
        max(0.0, float(TRACK_GATE_RANDOM_DISTANCE_MARGIN_CM)),
        total_length * 0.45,
    )
    if count == 1:
        return [random.uniform(margin, max(margin, total_length - margin))]

    min_spacing = max(0.0, float(TRACK_GATE_DISTANCE_RANGE_CM[0]))
    max_spacing = max(min_spacing, float(TRACK_GATE_DISTANCE_RANGE_CM[1]))
    usable_length = max(0.0, total_length - margin * 2.0)
    min_span = min_spacing * float(count - 1)
    max_span = max_spacing * float(count - 1)

    if min_span > usable_length:
        unreal.log_warning(
            f"Cannot place {count} gate(s) on a {total_length / 100.0:.1f}m spline "
            f"with minimum gate spacing {min_spacing / 100.0:.1f}m and "
            f"margin {margin / 100.0:.1f}m. Using the widest feasible spacing."
        )
        spacing = usable_length / float(max(1, count - 1))
        return [margin + spacing * index for index in range(count)]

    for _ in range(200):
        span = random.uniform(min_span, min(usable_length, max_span))
        gaps = constrained_random_gaps(span, count - 1, min_spacing, max_spacing)
        if not gaps:
            continue
        start = random.uniform(margin, total_length - margin - span)
        distances = [start]
        for gap in gaps:
            distances.append(distances[-1] + gap)
        return distances

    spacing = min(max_spacing, max(min_spacing, usable_length / float(max(1, count - 1))))
    start = random.uniform(margin, max(margin, total_length - margin - spacing * (count - 1)))
    return [start + spacing * index for index in range(count)]


def constrained_random_gaps(total_span, gap_count, min_spacing, max_spacing):
    if gap_count <= 0:
        return []

    remaining = float(total_span)
    gaps = []
    for gap_index in range(gap_count):
        remaining_gaps = gap_count - gap_index - 1
        low = max(min_spacing, remaining - max_spacing * remaining_gaps)
        high = min(max_spacing, remaining - min_spacing * remaining_gaps)
        if low > high:
            return None
        gap = random.uniform(low, high)
        gaps.append(gap)
        remaining -= gap
    return gaps


def sample_points_and_tangents_by_random_distance(points, count):
    cumulative = cumulative_polyline_distances(points)
    total_length = cumulative[-1] if cumulative else 0.0
    target_distances = random_gate_distances(total_length, count)
    return sample_points_and_tangents_at_distances(points, cumulative, target_distances)


def cumulative_polyline_distances(points):
    if not points:
        return []
    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + distance(points[index - 1], points[index]))
    return cumulative


def sample_points_and_tangents_at_distances(points, cumulative, target_distances):
    if not points:
        return []
    if len(points) <= 1 or not cumulative or cumulative[-1] <= 0.0:
        return [
            {
                "location": points[0],
                "tangent": unreal.Vector(1.0, 0.0, 0.0),
                "distance": 0.0,
            }
            for _ in target_distances
        ]

    sampled = []
    point_index = 1
    for target_distance in target_distances:
        target_distance = clamp(float(target_distance), (0.0, cumulative[-1]))
        while point_index < len(cumulative) - 1 and cumulative[point_index] < target_distance:
            point_index += 1

        sampled.append(sample_point_and_tangent_at_index(points, cumulative, point_index, target_distance))

    return sampled


def sample_point_and_tangent_at_distance(points, cumulative, target_distance):
    if not points:
        return {
            "location": unreal.Vector(0.0, 0.0, 0.0),
            "tangent": unreal.Vector(1.0, 0.0, 0.0),
            "distance": 0.0,
        }
    if len(points) <= 1 or not cumulative or cumulative[-1] <= 0.0:
        return {
            "location": points[0],
            "tangent": unreal.Vector(1.0, 0.0, 0.0),
            "distance": 0.0,
        }

    target_distance = clamp(float(target_distance), (0.0, cumulative[-1]))
    point_index = 1
    while point_index < len(cumulative) - 1 and cumulative[point_index] < target_distance:
        point_index += 1
    return sample_point_and_tangent_at_index(points, cumulative, point_index, target_distance)


def sample_point_and_tangent_at_index(points, cumulative, point_index, target_distance):
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
    return {
        "location": location,
        "tangent": tangent,
        "distance": target_distance,
    }

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
        samples = sample_points_and_tangents_by_random_distance(
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
    cumulative = cumulative_polyline_distances(path_points)
    total_length = cumulative[-1] if cumulative else 0.0
    return {
        "locations": locations,
        "rotations": rotations,
        "spline_points": path_points,
        "spline_total_length_cm": total_length,
        "gate_distances": [sample.get("distance", 0.0) for sample in samples],
        "gate_spacing_distances_cm": gate_spacing_distances(samples),
    }


def gate_spacing_distances(samples):
    distances = [float(sample.get("distance", 0.0)) for sample in samples]
    return [
        distances[index] - distances[index - 1]
        for index in range(1, len(distances))
    ]


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
