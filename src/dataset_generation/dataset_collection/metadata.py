"""Dataset metadata construction and writing."""

import json

import unreal

from src.dataset_generation.common import (
    rotator_to_dict,
    vector_to_meters_dict,
    component_name,
)
from src.dataset_generation.config import (
    BBOX_CORNER_NAMES,
    GATE_CORNER_NAMES,
    GATE_VISIBILITY_SAMPLE_STEPS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SAVE_FRAME_METADATA,
)
from src.dataset_generation.dataset_collection.outputs import (
    dataset_camera_intrinsics_path,
    frame_file_path,
    frame_metadata_path,
    frame_metadata_jsonl_path,
    current_run_dir,
)
from src.dataset_generation.dataset_collection.projection import (
    world_location_to_camera_frame, world_location_to_camera_frame_ruf,
    gate_orientation_axes_camera_frame_ruf, gate_relative_euler_camera_frame, project_world_to_render_frame,
    bbox_from_projected_points, camera_intrinsics, quaternion_from_relative_euler_deg,
)
from src.dataset_generation.dataset_collection.visibility import trace_visibility_to_point, image_visibility


GATE_FRAME_SURFACES = (
    (
        "Corner_outer_TL",
        "Corner_outer_TR",
        "Corner_inner_TR",
        "Corner_inner_TL",
    ),
    (
        "Corner_inner_BL",
        "Corner_inner_BR",
        "Corner_outer_BR",
        "Corner_outer_BL",
    ),
    (
        "Corner_outer_TL",
        "Corner_inner_TL",
        "Corner_inner_BL",
        "Corner_outer_BL",
    ),
    (
        "Corner_inner_TR",
        "Corner_outer_TR",
        "Corner_outer_BR",
        "Corner_inner_BR",
    ),
)


def vector_lerp(a, b, alpha):
    return a + (b - a) * alpha


def sample_quad_points(corners, steps):
    top_left, top_right, bottom_right, bottom_left = corners
    for row in range(steps + 1):
        v = row / float(max(1, steps))
        left = vector_lerp(top_left, bottom_left, v)
        right = vector_lerp(top_right, bottom_right, v)
        for column in range(steps + 1):
            u = column / float(max(1, steps))
            yield vector_lerp(left, right, u)


def gate_frame_sample_points(corner_locations):
    for surface in GATE_FRAME_SURFACES:
        surface_corners = [corner_locations.get(corner_name) for corner_name in surface]
        if any(corner is None for corner in surface_corners):
            continue
        yield from sample_quad_points(surface_corners, GATE_VISIBILITY_SAMPLE_STEPS)


def sample_point_visible(world_context, camera, gate, world_location):
    projected = project_world_to_render_frame(camera, world_location)
    line_of_sight = trace_visibility_to_point(world_context, camera, gate, world_location)
    return image_visibility(projected, line_of_sight)["visible"]


def any_gate_frame_sample_visible(world_context, camera, gate, corner_locations):
    for world_location in gate_frame_sample_points(corner_locations):
        if sample_point_visible(world_context, camera, gate, world_location):
            return True
    return False

def gate_marker_components(gate, marker_names):
    components = []
    seen = set()
    for class_name in ("ArrowComponent", "SceneComponent", "ActorComponent"):
        component_class = getattr(unreal, class_name, None)
        if not component_class:
            continue
        try:
            for component in gate.get_components_by_class(component_class):
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


def gate_corner_components(gate):
    return gate_marker_components(gate, GATE_CORNER_NAMES)


def gate_bbox_corner_components(gate):
    return gate_marker_components(gate, BBOX_CORNER_NAMES)

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

    get_location = getattr(component, "get_socket_location", None)
    if get_location:
        try:
            return get_location("")
        except Exception:
            pass

    get_transform = getattr(component, "get_world_transform", None)
    if get_transform:
        transform = get_transform()
        get_translation = getattr(transform, "get_translation", None)
        if get_translation:
            return get_translation()

    return None


def missing_marker_projection():
    return {
        "world_m": None,
        "camera_frame_m": None,
        "pixel_px": None,
        "in_front_of_camera": False,
        "inside_frame": False,
        "missing": True,
    }


def nested_corner_metadata(corners):
    nested = {"outer": {}, "inner": {}}
    for corner_name, projection in corners.items():
        parts = corner_name.split("_")
        if len(parts) < 3:
            continue
        ring = parts[1].lower()
        location = parts[2].lower()
        if ring in nested:
            nested[ring][location] = projection
    return nested


def write_dataset_camera_intrinsics(camera):
    intrinsics = camera_intrinsics(camera)
    payload = {
        "camera": {
            "label": camera.get_actor_label(),
            "intrinsics": intrinsics,
            "coordinate_frame": {
                "relative_position_axes": ["right", "up", "forward"],
                "relative_orientation_euler_order": "yaw_deg, pitch_deg, roll_deg",
                "relative_orientation_quat_identity": {
                    "w": 1.0,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                },
            },
        }
    }
    with open(dataset_camera_intrinsics_path(), "w", encoding="utf-8") as intrinsics_file:
        json.dump(payload, intrinsics_file, indent=2)


def project_bbox_markers(camera, gate):
    bbox_components = gate_bbox_corner_components(gate)
    bbox_corners = {}

    for marker_name in BBOX_CORNER_NAMES:
        component = bbox_components.get(marker_name)
        if component:
            world_location = component_world_location(component)
            if world_location is not None:
                bbox_corners[marker_name] = project_world_to_render_frame(camera, world_location)
                continue

        bbox_corners[marker_name] = missing_marker_projection()

    return bbox_corners

def gate_metadata_for_frame(world_context, camera, gate):
    gate_location = gate.get_actor_location()
    relative_orientation_euler_deg = gate_relative_euler_camera_frame(camera, gate)
    corner_components = gate_corner_components(gate)
    corners = {}
    corner_locations = {}

    for corner_name in GATE_CORNER_NAMES:
        component = corner_components.get(corner_name)
        if component:
            world_location = component_world_location(component)
            if world_location is not None:
                corner_locations[corner_name] = world_location
                corner_projection = project_world_to_render_frame(camera, world_location)
                line_of_sight = trace_visibility_to_point(
                    world_context,
                    camera,
                    gate,
                    world_location,
                )
                corner_projection["visibility"] = image_visibility(corner_projection, line_of_sight)
                corners[corner_name] = corner_projection
                continue

        corners[corner_name] = {
            "world_m": None,
            "camera_frame_m": None,
            "pixel_px": None,
            "in_front_of_camera": False,
            "inside_frame": False,
            "visibility": {
                "visible": False,
                "inside_frame": False,
                "line_of_sight": {
                    "occlusion_tested": False,
                    "line_of_sight": None,
                    "occluded": None,
                    "occluder": None,
                },
            },
            "missing": True,
        }

    bbox_corners = project_bbox_markers(camera, gate)
    bbox_2d = bbox_from_projected_points(
        [corner for corner in corners.values() if not corner.get("missing")]
        + [corner for corner in bbox_corners.values() if not corner.get("missing")]
    )
    visible = any(
        corner.get("visibility", {}).get("visible")
        for corner in corners.values()
        if not corner.get("missing")
    )
    if not visible and bbox_2d and bbox_2d.get("intersects_frame"):
        visible = any_gate_frame_sample_visible(world_context, camera, gate, corner_locations)

    return {
        "label": gate.get_actor_label(),
        "visible": bool(visible),
        "world_location_m": vector_to_meters_dict(gate_location),
        "world_rotation_deg": rotator_to_dict(gate.get_actor_rotation()),
        "camera_frame_location_m": vector_to_meters_dict(
            world_location_to_camera_frame(camera, gate_location)
        ),
        "relative_position_camera_frame_m": {
            axis: value / 100.0
            for axis, value in world_location_to_camera_frame_ruf(
                camera,
                gate_location,
            ).items()
        },
        "relative_orientation_axes_camera_frame": gate_orientation_axes_camera_frame_ruf(
            camera,
            gate,
        ),
        "relative_orientation_euler_deg": relative_orientation_euler_deg,
        "relative_orientation_quat": quaternion_from_relative_euler_deg(
            relative_orientation_euler_deg
        ),
        "corners": nested_corner_metadata(corners),
        "bbox_corners": bbox_corners,
        "bbox_2d_px": bbox_2d,
        "visible_in_frame": bool(visible),
    }

def write_frame_metadata(world_context, camera, gates, frame_number, pose, frame_saved):
    if not SAVE_FRAME_METADATA:
        return

    frame_path = frame_file_path(frame_number)
    visible_gates = []
    for gate in gates:
        gate_metadata = gate_metadata_for_frame(world_context, camera, gate)
        if gate_metadata["visible"]:
            visible_gates.append(gate_metadata)

    target_gate = pose["gate"].get_actor_label() if pose.get("gate") else None

    record = {
        "frame_number": int(frame_number),
        "run_dir": current_run_dir(),
        "frame_path": frame_path,
        "frame_saved": bool(frame_saved),
        "frame_width_px": FRAME_WIDTH,
        "frame_height_px": FRAME_HEIGHT,
        "camera": {
            "label": camera.get_actor_label(),
            "world_location_m": vector_to_meters_dict(camera.get_actor_location()),
            "world_rotation_deg": rotator_to_dict(camera.get_actor_rotation()),
        },
        "target_gate": target_gate,
        "gates": visible_gates,
    }

    metadata_path = frame_metadata_path()
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            records = json.load(metadata_file)
    except Exception:
        records = []

    records.append(record)

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(records, metadata_file, indent=2)

    with open(frame_metadata_jsonl_path(), "a", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps(record, separators=(",", ":")) + "\n")
