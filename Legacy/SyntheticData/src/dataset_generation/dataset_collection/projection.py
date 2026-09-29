"""Camera-frame coordinate conversion and 2D projection."""

import math

import unreal

from SyntheticData.src.dataset_generation.common import dot, vector_to_dict, vector_to_meters_dict, lerp
from SyntheticData.src.dataset_generation.config import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FRAME_MARGIN
from SyntheticData.src.dataset_generation.dataset_collection.camera_poses import camera_frame_half_angles

def camera_axes(camera):
    rotation = camera.get_actor_rotation()
    math_library = unreal.MathLibrary
    return (
        math_library.get_forward_vector(rotation),
        math_library.get_right_vector(rotation),
        math_library.get_up_vector(rotation),
    )

def world_vector_to_camera_frame(camera, world_vector):
    forward, right, up = camera_axes(camera)
    return unreal.Vector(
        dot(world_vector, forward),
        dot(world_vector, right),
        dot(world_vector, up),
    )

def world_location_to_camera_frame(camera, world_location):
    delta = world_location - camera.get_actor_location()
    return world_vector_to_camera_frame(camera, delta)

def world_vector_to_camera_frame_ruf(camera, world_vector):
    forward, right, up = camera_axes(camera)
    return {
        "right": float(dot(world_vector, right)),
        "up": float(dot(world_vector, up)),
        "forward": float(dot(world_vector, forward)),
    }

def world_location_to_camera_frame_ruf(camera, world_location):
    return world_vector_to_camera_frame_ruf(camera, world_location - camera.get_actor_location())

def ruf_tuple(camera_frame_vector):
    return (
        float(camera_frame_vector["right"]),
        float(camera_frame_vector["up"]),
        float(camera_frame_vector["forward"]),
    )

def tuple_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def tuple_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

def tuple_length(value):
    return math.sqrt(tuple_dot(value, value))

def tuple_normalized(value):
    length = tuple_length(value)
    if length <= 0.000001:
        return (0.0, 0.0, 0.0)
    return (value[0] / length, value[1] / length, value[2] / length)

def tuple_subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def tuple_scale(value, scale):
    return (value[0] * scale, value[1] * scale, value[2] * scale)

def gate_orientation_in_camera_frame(camera, gate):
    rotation = gate.get_actor_rotation()
    math_library = unreal.MathLibrary
    return {
        "forward": vector_to_dict(
            world_vector_to_camera_frame(camera, math_library.get_forward_vector(rotation))
        ),
        "right": vector_to_dict(
            world_vector_to_camera_frame(camera, math_library.get_right_vector(rotation))
        ),
        "up": vector_to_dict(
            world_vector_to_camera_frame(camera, math_library.get_up_vector(rotation))
        ),
    }

def gate_orientation_axes_camera_frame_ruf(camera, gate):
    rotation = gate.get_actor_rotation()
    math_library = unreal.MathLibrary
    return {
        "forward": world_vector_to_camera_frame_ruf(
            camera,
            math_library.get_forward_vector(rotation),
        ),
        "right": world_vector_to_camera_frame_ruf(
            camera,
            math_library.get_right_vector(rotation),
        ),
        "up": world_vector_to_camera_frame_ruf(
            camera,
            math_library.get_up_vector(rotation),
        ),
    }

def gate_relative_euler_camera_frame(camera, gate):
    axes = gate_orientation_axes_camera_frame_ruf(camera, gate)
    gate_forward = tuple_normalized(ruf_tuple(axes["forward"]))
    gate_up = tuple_normalized(ruf_tuple(axes["up"]))

    right_amount = gate_forward[0]
    up_amount = gate_forward[1]
    forward_amount = gate_forward[2]

    yaw_deg = math.degrees(math.atan2(right_amount, forward_amount))
    pitch_deg = math.degrees(
        math.atan2(up_amount, math.sqrt(right_amount * right_amount + forward_amount * forward_amount))
    )

    camera_up = (0.0, 1.0, 0.0)
    reference_up = tuple_subtract(
        camera_up,
        tuple_scale(gate_forward, tuple_dot(camera_up, gate_forward)),
    )
    reference_up = tuple_normalized(reference_up)

    if tuple_length(reference_up) <= 0.000001:
        reference_up = (1.0, 0.0, 0.0)

    roll_deg = math.degrees(
        math.atan2(
            tuple_dot(tuple_cross(reference_up, gate_up), gate_forward),
            tuple_dot(reference_up, gate_up),
        )
    )

    return {
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
    }

def quaternion_from_relative_euler_deg(euler_deg):
    yaw = math.radians(float(euler_deg["yaw_deg"]))
    pitch = math.radians(float(euler_deg["pitch_deg"]))
    roll = math.radians(float(euler_deg["roll_deg"]))

    half_yaw = yaw * 0.5
    half_pitch = pitch * 0.5
    half_roll = roll * 0.5

    yaw_w, yaw_y = math.cos(half_yaw), math.sin(half_yaw)
    pitch_w, pitch_x = math.cos(half_pitch), math.sin(half_pitch)
    roll_w, roll_z = math.cos(half_roll), math.sin(half_roll)

    w, x, y, z = quat_multiply(
        quat_multiply((yaw_w, 0.0, yaw_y, 0.0), (pitch_w, pitch_x, 0.0, 0.0)),
        (roll_w, 0.0, 0.0, roll_z),
    )
    return {
        "w": float(w),
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }

def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )

def camera_projection_params(camera):
    max_yaw_offset, max_pitch_offset = camera_frame_half_angles(camera)
    horizontal_fov = (max_yaw_offset / CAMERA_FRAME_MARGIN) * 2.0
    vertical_fov = (max_pitch_offset / CAMERA_FRAME_MARGIN) * 2.0
    focal_x = (FRAME_WIDTH * 0.5) / math.tan(math.radians(horizontal_fov) * 0.5)
    focal_y = (FRAME_HEIGHT * 0.5) / math.tan(math.radians(vertical_fov) * 0.5)
    return focal_x, focal_y

def camera_intrinsics(camera):
    focal_x, focal_y = camera_projection_params(camera)
    max_yaw_offset, max_pitch_offset = camera_frame_half_angles(camera)
    horizontal_fov = (max_yaw_offset / CAMERA_FRAME_MARGIN) * 2.0
    vertical_fov = (max_pitch_offset / CAMERA_FRAME_MARGIN) * 2.0
    return {
        "frame_width_px": FRAME_WIDTH,
        "frame_height_px": FRAME_HEIGHT,
        "focal_x_px": float(focal_x),
        "focal_y_px": float(focal_y),
        "principal_x_px": FRAME_WIDTH * 0.5,
        "principal_y_px": FRAME_HEIGHT * 0.5,
        "horizontal_fov_deg": float(horizontal_fov),
        "vertical_fov_deg": float(vertical_fov),
    }

def project_world_to_render_frame(camera, world_location):
    camera_space = world_location_to_camera_frame(camera, world_location)
    focal_x, focal_y = camera_projection_params(camera)
    in_front = camera_space.x > 0.001

    if in_front:
        pixel_x = (FRAME_WIDTH * 0.5) + (camera_space.y / camera_space.x) * focal_x
        pixel_y = (FRAME_HEIGHT * 0.5) - (camera_space.z / camera_space.x) * focal_y
    else:
        pixel_x = None
        pixel_y = None

    return {
        "world_m": vector_to_meters_dict(world_location),
        "camera_frame_m": vector_to_meters_dict(camera_space),
        "pixel_px": None
        if pixel_x is None
        else {
            "x": float(pixel_x),
            "y": float(pixel_y),
        },
        "in_front_of_camera": bool(in_front),
        "inside_frame": bool(
            in_front and 0.0 <= pixel_x < FRAME_WIDTH and 0.0 <= pixel_y < FRAME_HEIGHT
        ),
    }

def clamp_pixel(value, upper_bound):
    return max(0.0, min(float(upper_bound - 1), float(value)))

def bbox_from_projected_points(projected_points):
    pixels = []
    visible_pixels = []

    for projected in projected_points:
        pixel = projected.get("pixel_px") if projected else None
        if not pixel:
            continue

        pixels.append(pixel)
        if projected.get("inside_frame"):
            visible_pixels.append(pixel)

    if not pixels:
        return None

    min_x = min(pixel["x"] for pixel in pixels)
    min_y = min(pixel["y"] for pixel in pixels)
    max_x = max(pixel["x"] for pixel in pixels)
    max_y = max(pixel["y"] for pixel in pixels)
    left_x = clamp_pixel(min_x, FRAME_WIDTH)
    right_x = clamp_pixel(max_x, FRAME_WIDTH)
    top_y = clamp_pixel(min_y, FRAME_HEIGHT)
    bottom_y = clamp_pixel(max_y, FRAME_HEIGHT)

    intersects_frame = not (
        max_x < 0.0 or max_y < 0.0 or min_x >= FRAME_WIDTH or min_y >= FRAME_HEIGHT
    )

    return {
        "bottom_left": {
            "x_px": left_x,
            "y_px": top_y,
        },
        "top_right": {
            "x_px": right_x,
            "y_px": bottom_y,
        },
        "x_min_px": left_x,
        "y_min_px": top_y,
        "x_max_px": right_x,
        "y_max_px": bottom_y,
        "width_px": max(0.0, right_x - left_x),
        "height_px": max(0.0, bottom_y - top_y),
        "intersects_frame": bool(intersects_frame),
        "point_count": len(pixels),
        "inside_point_count": len(visible_pixels),
    }

def vector_lerp(a, b, alpha):
    return unreal.Vector(
        lerp(a.x, b.x, alpha),
        lerp(a.y, b.y, alpha),
        lerp(a.z, b.z, alpha),
    )
