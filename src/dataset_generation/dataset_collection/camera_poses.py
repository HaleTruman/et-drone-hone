"""Camera pose generation for dataset capture."""

import math
import random

import unreal

from src.dataset_generation.common import distance
from src.dataset_generation.config import (
    X_RANGE_CM, Y_RANGE_CM, Z_RANGE_CM, MIN_CAMERA_GATE_DISTANCE_CM, MAX_CAMERA_GATE_DISTANCE_CM,
    MAX_CAMERA_POSE_ATTEMPTS, PITCH_RANGE_DEG, ROLL_DEG, CAMERA_FRAME_MARGIN,
    OBSTACLE_CLEARANCE_CM,
)
from src.dataset_generation.gate_randomization.track import random_location
from src.dataset_generation.gate_randomization.track import clamped_vector
from src.dataset_generation.obstacles import point_clears_obstacles

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

def camera_pose_for_gate(camera, target_gate):
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
