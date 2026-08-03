"""Dataset metadata construction and writing."""

import json
import math

import unreal

from src.dataset_generation.common import (
    rotator_to_dict,
    vector_to_meters_dict,
    component_name,
)
from src.dataset_generation.config import (
    BBOX_CORNER_NAMES,
    DATASET_SCHEMA_VERSION,
    GATE_CORNER_NAMES,
    GATE_CUSTOM_DEPTH_STENCIL_BANDS,
    GATE_VISIBILITY_SAMPLE_STEPS,
    GATE_CUSTOM_DEPTH_STENCIL_BASE_VALUE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MAX_EXPECTED_MASK_STENCIL_VALUES,
    MASK_ID_TOLERANCE,
    MASK_MIN_COMPONENT_PIXELS,
    SAVE_RAW_MASK_DEBUG,
    SAVE_FRAME_METADATA,
)
from src.dataset_generation.dataset_collection.outputs import (
    dataset_camera_intrinsics_path,
    frame_file_path,
    frame_relative_path,
    mask_file_path,
    mask_relative_path,
    frame_metadata_path,
    frame_metadata_jsonl_path,
    current_run_dir,
    current_dataset_id,
    current_run_id,
    relative_to_dataset,
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

BLANK_MASK_WARNING_PATHS = set()


def orientation_convention_metadata():
    return {
        "relative_orientation_euler_deg": (
            "Gate model axes expressed in the camera frame. Yaw/pitch describe the "
            "gate forward/plane-normal direction in camera right/up/forward axes; "
            "roll is rotation of the gate up axis around that forward direction."
        ),
        "relative_orientation_axes_camera_frame": (
            "Gate local forward, right, and up unit axes represented in camera "
            "right/up/forward coordinates."
        ),
        "relative_orientation_quat": (
            "Quaternion equivalent of relative_orientation_euler_deg, identity when "
            "the gate forward axis points along camera forward and gate up aligns "
            "with camera up."
        ),
        "relative_position_axes": ["right_m", "up_m", "forward_m"],
    }


def mask_layer_metadata():
    return {
        "gate_mask": {
            "available": True,
            "encoding": "8-bit grayscale cleaned PNG; 0=background, gate pixels equal each gate custom_depth_stencil_value.",
        },
        "obstacle_mask": {
            "available": False,
            "reason": "No separate obstacle segmentation capture pass is configured yet.",
        },
    }


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


def gate_render_components(gate):
    components = []
    seen = set()
    for class_name in (
        "PrimitiveComponent",
        "MeshComponent",
        "StaticMeshComponent",
        "SkeletalMeshComponent",
        "InstancedStaticMeshComponent",
    ):
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
    return components


def gate_custom_depth_stencil_value(gate):
    for component in gate_render_components(gate):
        try:
            value = int(component.get_editor_property("custom_depth_stencil_value"))
            if value > 0:
                return value
        except Exception:
            pass
    return None


def configured_gate_stencil_value(gate_index):
    gate_index = int(gate_index)
    if gate_index < len(GATE_CUSTOM_DEPTH_STENCIL_BANDS):
        return min(255, max(1, int(GATE_CUSTOM_DEPTH_STENCIL_BANDS[gate_index])))
    return min(255, max(1, int(GATE_CUSTOM_DEPTH_STENCIL_BASE_VALUE) + gate_index))

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
        "schema_version": int(DATASET_SCHEMA_VERSION),
        "dataset_id": current_dataset_id(),
        "orientation_convention": orientation_convention_metadata(),
        "mask_layers": mask_layer_metadata(),
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


def write_mask_stencil_legend(gates):
    legend = {
        "schema_version": int(DATASET_SCHEMA_VERSION),
        "dataset_id": current_dataset_id(),
        "run_id": current_run_id(),
        "description": "Pixel values in cleaned mask PNGs mapped to gate actor labels.",
        "stencil_values": [],
    }
    for gate_index, gate in enumerate(gates):
        stencil_value = gate_custom_depth_stencil_value(gate) or configured_gate_stencil_value(
            gate_index,
        )
        legend["stencil_values"].append(
            {
                "label": gate.get_actor_label(),
                "stencil_value": int(stencil_value),
            }
        )

    legend_path = f"{current_run_dir()}\\mask_stencil_legend.json"
    with open(legend_path, "w", encoding="utf-8") as legend_file:
        json.dump(legend, legend_file, indent=2)


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


def vector_distance_m(a, b):
    if a is None or b is None:
        return None
    return math.sqrt(
        (a.x - b.x) * (a.x - b.x)
        + (a.y - b.y) * (a.y - b.y)
        + (a.z - b.z) * (a.z - b.z)
    ) / 100.0


def average_present(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def gate_model_metadata(corner_locations):
    return {
        "outer_width_m": average_present(
            [
                vector_distance_m(
                    corner_locations.get("Corner_outer_TL"),
                    corner_locations.get("Corner_outer_TR"),
                ),
                vector_distance_m(
                    corner_locations.get("Corner_outer_BL"),
                    corner_locations.get("Corner_outer_BR"),
                ),
            ]
        ),
        "outer_height_m": average_present(
            [
                vector_distance_m(
                    corner_locations.get("Corner_outer_TL"),
                    corner_locations.get("Corner_outer_BL"),
                ),
                vector_distance_m(
                    corner_locations.get("Corner_outer_TR"),
                    corner_locations.get("Corner_outer_BR"),
                ),
            ]
        ),
        "inner_width_m": average_present(
            [
                vector_distance_m(
                    corner_locations.get("Corner_inner_TL"),
                    corner_locations.get("Corner_inner_TR"),
                ),
                vector_distance_m(
                    corner_locations.get("Corner_inner_BL"),
                    corner_locations.get("Corner_inner_BR"),
                ),
            ]
        ),
        "inner_height_m": average_present(
            [
                vector_distance_m(
                    corner_locations.get("Corner_inner_TL"),
                    corner_locations.get("Corner_inner_BL"),
                ),
                vector_distance_m(
                    corner_locations.get("Corner_inner_TR"),
                    corner_locations.get("Corner_inner_BR"),
                ),
            ]
        ),
    }


def corner_visibility_entries(corners):
    return [
        corner
        for corner in corners.values()
        if corner and not corner.get("missing")
    ]


def geometric_visibility_summary(corners, bbox_2d, fallback_surface_sample_visible):
    entries = corner_visibility_entries(corners)
    inside_count = sum(1 for corner in entries if corner.get("inside_frame"))
    in_front_count = sum(1 for corner in entries if corner.get("in_front_of_camera"))
    line_of_sight_count = sum(
        1
        for corner in entries
        if corner.get("visibility", {})
        .get("line_of_sight", {})
        .get("line_of_sight")
        is True
    )
    return {
        "projected_intersects_frame": bool(bbox_2d and bbox_2d.get("intersects_frame")),
        "any_corner_inside_frame": bool(inside_count > 0),
        "corner_inside_frame_count": int(inside_count),
        "any_corner_in_front_of_camera": bool(in_front_count > 0),
        "corner_in_front_of_camera_count": int(in_front_count),
        "line_of_sight_visible": bool(line_of_sight_count > 0),
        "line_of_sight_visible_corner_count": int(line_of_sight_count),
        "fallback_surface_sample_visible": bool(fallback_surface_sample_visible),
        "fallback_surface_sample_note": (
            "Only evaluated when no corner is geometrically visible but the projected "
            "gate bbox intersects the frame; this is a sparse raycast fallback and "
            "is not the authoritative rendered-mask visibility signal."
        ),
    }


def occluder_from_corners(corners):
    for corner in corner_visibility_entries(corners):
        line_of_sight = corner.get("visibility", {}).get("line_of_sight", {})
        occluder = line_of_sight.get("occluder")
        if occluder:
            return occluder
        blocking_hit = line_of_sight.get("blocking_hit")
        if blocking_hit and blocking_hit.get("actor_label"):
            return blocking_hit.get("actor_label")
    return None


def occlusion_type_for_gate(visible_in_frame, geometric_visibility, corners, occluder_label):
    if not geometric_visibility.get("any_corner_in_front_of_camera"):
        return "behind_camera"
    if not geometric_visibility.get("projected_intersects_frame"):
        return "edge_clipped"
    if visible_in_frame and not occluder_label:
        return "none"
    if occluder_label:
        return "gate" if "gate" in occluder_label.lower() else "obstacle"
    if not geometric_visibility.get("line_of_sight_visible"):
        return "obstacle"
    entries = corner_visibility_entries(corners)
    if entries and any(not corner.get("inside_frame") for corner in entries):
        return "edge_clipped"
    return "none" if visible_in_frame else "obstacle"


def annotate_corner_occlusion(corner):
    visibility = corner.get("visibility", {})
    line_of_sight = visibility.get("line_of_sight", {})
    occluder = line_of_sight.get("occluder")
    if not corner.get("in_front_of_camera"):
        occlusion_type = "behind_camera"
    elif not corner.get("inside_frame"):
        occlusion_type = "edge_clipped"
    elif occluder:
        occlusion_type = "gate" if "gate" in occluder.lower() else "obstacle"
    elif line_of_sight.get("line_of_sight") is False:
        occlusion_type = "obstacle"
    else:
        occlusion_type = "none"

    visibility["occlusion_type"] = occlusion_type
    visibility["occluder_label"] = occluder
    corner["visibility"] = visibility


def visibility_fraction(mask_pixels, bbox_2d):
    pixel_count = int(mask_pixels.get("pixel_count", 0)) if mask_pixels else 0
    if not bbox_2d:
        return 0.0
    area = float(bbox_2d.get("width_px", 0.0)) * float(bbox_2d.get("height_px", 0.0))
    if area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, pixel_count / area))

def gate_metadata_for_frame(world_context, camera, gate, mask_index=None, stencil_value=None):
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
                annotate_corner_occlusion(corner_projection)
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
        annotate_corner_occlusion(corners[corner_name])

    bbox_corners = project_bbox_markers(camera, gate)
    bbox_2d = bbox_from_projected_points(
        [corner for corner in corners.values() if not corner.get("missing")]
        + [corner for corner in bbox_corners.values() if not corner.get("missing")]
    )
    geometric_visible = any(
        corner.get("visibility", {}).get("visible")
        for corner in corners.values()
        if not corner.get("missing")
    )
    fallback_surface_sample_visible = False
    if not geometric_visible and bbox_2d and bbox_2d.get("intersects_frame"):
        fallback_surface_sample_visible = any_gate_frame_sample_visible(
            world_context,
            camera,
            gate,
            corner_locations,
        )
        geometric_visible = fallback_surface_sample_visible

    mask_pixels = mask_pixels_for_stencil(mask_index, stencil_value)
    visible_in_frame = bool(mask_pixels.get("pixel_count", 0) > 0)
    fraction = visibility_fraction(mask_pixels, bbox_2d)
    geometric_visibility = geometric_visibility_summary(
        corners,
        bbox_2d,
        fallback_surface_sample_visible,
    )
    occluder_label = occluder_from_corners(corners)
    occlusion_type = occlusion_type_for_gate(
        visible_in_frame,
        geometric_visibility,
        corners,
        occluder_label,
    )

    return {
        "label": gate.get_actor_label(),
        "visible": bool(visible_in_frame),
        "geometric_visible": bool(geometric_visible),
        "custom_depth_stencil_value": stencil_value,
        "gate_model": gate_model_metadata(corner_locations),
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
        "visible_mask_pixels": mask_pixels,
        "visible_in_frame": bool(visible_in_frame),
        "geometric_visibility": geometric_visibility,
        "visibility_fraction": fraction,
        "occlusion_fraction": max(0.0, min(1.0, 1.0 - fraction)),
        "visibility_fraction_basis": "visible_mask_pixels.pixel_count / clipped projected bbox area in pixels",
        "occlusion_type": occlusion_type,
        "occluder_label": occluder_label,
    }

def write_frame_metadata(
    world_context,
    camera,
    gates,
    frame_number,
    pose,
    frame_saved,
    mask_saved=False,
):
    if not SAVE_FRAME_METADATA:
        return

    frame_path = frame_file_path(frame_number)
    mask_path = mask_file_path(frame_number) if mask_saved else None
    frame_run_relative_path = frame_relative_path(frame_number)
    mask_run_relative_path = mask_relative_path(frame_number) if mask_saved else None
    mask_index = load_mask_index(mask_path) if mask_path else None
    frame_gates = []
    for gate_index, gate in enumerate(gates):
        stencil_value = gate_custom_depth_stencil_value(gate) or configured_gate_stencil_value(
            gate_index,
        )
        gate_metadata = gate_metadata_for_frame(
            world_context,
            camera,
            gate,
            mask_index,
            stencil_value,
        )
        frame_gates.append(gate_metadata)

    if (
        mask_path
        and mask_index
        and frame_gates
        and not mask_index.get("has_nonzero_pixels")
        and mask_path not in BLANK_MASK_WARNING_PATHS
    ):
        BLANK_MASK_WARNING_PATHS.add(mask_path)
        unreal.log_warning(
            f"Mask image '{mask_path}' is all black even though visible gates were "
            "detected geometrically. Check the GateMaskCapture post-process material: "
            "it is not writing CustomStencil values into final color."
        )

    target_gate = pose["gate"].get_actor_label() if pose.get("gate") else None
    flight_metadata = frame_flight_metadata(pose)

    record = {
        "schema_version": int(DATASET_SCHEMA_VERSION),
        "dataset_id": current_dataset_id(),
        "run_id": current_run_id(),
        "frame_number": int(frame_number),
        "run_dir": current_run_dir(),
        "run_dir_relative_to_dataset": relative_to_dataset(current_run_dir()),
        "frame_path": frame_path,
        "frame_path_relative": frame_run_relative_path,
        "frame_path_relative_to_run": frame_run_relative_path,
        "frame_path_relative_to_dataset": relative_to_dataset(frame_path),
        "frame_saved": bool(frame_saved),
        "mask_path": mask_path,
        "mask_path_relative": mask_run_relative_path,
        "mask_path_relative_to_run": mask_run_relative_path,
        "mask_path_relative_to_dataset": relative_to_dataset(mask_path) if mask_path else None,
        "mask_saved": bool(mask_saved),
        "frame_width_px": FRAME_WIDTH,
        "frame_height_px": FRAME_HEIGHT,
        "orientation_convention": orientation_convention_metadata(),
        "mask_layers": mask_layer_metadata(),
        "camera": {
            "label": camera.get_actor_label(),
            "world_location_m": vector_to_meters_dict(camera.get_actor_location()),
            "world_rotation_deg": rotator_to_dict(camera.get_actor_rotation()),
        },
        "target_gate": target_gate,
        "flight": flight_metadata,
        "gates": frame_gates,
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


def frame_flight_metadata(pose):
    flight = pose.get("flight")
    if not flight:
        return None

    payload = dict(flight)
    velocity = payload.pop("velocity_cm_per_second", None)
    if velocity is not None:
        payload["velocity_m_per_second"] = vector_to_meters_dict(velocity)
    return payload


def load_mask_index(mask_path):
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:
        unreal.log_warning(
            f"Could not import Pillow/numpy for mask metadata extraction: {exc}"
        )
        return None

    try:
        with Image.open(mask_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        unreal.log_warning(f"Could not read mask image '{mask_path}': {exc}")
        return None

    if pixels.size <= 0:
        return None

    # The mask material should output CustomStencil / 255 as grayscale.
    # Average channels in case export introduces tiny channel differences.
    raw_values = np.rint(pixels.astype(np.float32).mean(axis=2)).astype(np.uint8)
    unique_values = np.unique(raw_values)
    nonzero_unique_values = unique_values[unique_values > 0]
    if len(nonzero_unique_values) > MAX_EXPECTED_MASK_STENCIL_VALUES:
        unreal.log_warning(
            f"Mask image '{mask_path}' contains {len(nonzero_unique_values)} nonzero "
            "pixel values, which looks like a depth/scene buffer rather than "
            "CustomStencil IDs. Skipping per-gate mask pixel extraction for this frame."
        )
        return {
            "values": raw_values,
            "srgb_decoded_values": raw_values,
            "width": int(raw_values.shape[1]),
            "height": int(raw_values.shape[0]),
            "has_nonzero_pixels": bool(raw_values.max() > 0),
            "valid_stencil_id_mask": False,
            "unique_nonzero_values": [
                int(value) for value in nonzero_unique_values[:MAX_EXPECTED_MASK_STENCIL_VALUES]
            ],
        }

    srgb_decoded_values = np.rint(
        np.power(raw_values.astype(np.float32) / 255.0, 2.2) * 255.0
    ).astype(np.uint8)
    return {
        "values": raw_values,
        "srgb_decoded_values": srgb_decoded_values,
        "width": int(raw_values.shape[1]),
        "height": int(raw_values.shape[0]),
        "has_nonzero_pixels": bool(raw_values.max() > 0),
        "valid_stencil_id_mask": True,
        "unique_nonzero_values": [int(value) for value in nonzero_unique_values],
    }


def sanitize_gate_mask_file(mask_path, gates):
    try:
        from PIL import Image
        import numpy as np
        import shutil
    except Exception as exc:
        unreal.log_warning(f"Could not import Pillow/numpy for mask cleanup: {exc}")
        return False

    gate_stencil_values = set()
    for gate_index, gate in enumerate(gates):
        stencil_value = gate_custom_depth_stencil_value(gate) or configured_gate_stencil_value(
            gate_index,
        )
        if stencil_value:
            gate_stencil_values.add(int(stencil_value))

    if not gate_stencil_values:
        return False

    try:
        with Image.open(mask_path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        unreal.log_warning(f"Could not read mask image '{mask_path}' for cleanup: {exc}")
        return False

    if SAVE_RAW_MASK_DEBUG:
        raw_path = raw_mask_debug_path(mask_path)
        try:
            shutil.copyfile(mask_path, raw_path)
        except Exception as exc:
            unreal.log_warning(f"Could not save raw mask debug copy '{raw_path}': {exc}")

    nonzero_unique_values = [int(value) for value in np.unique(values) if value > 0]
    cleaned = clean_mask_to_gate_stencils(values, gate_stencil_values, np)
    try:
        Image.fromarray(cleaned, mode="L").save(mask_path)
    except Exception as exc:
        unreal.log_warning(f"Could not write cleaned gate mask image '{mask_path}': {exc}")
        return False

    kept_values = sorted(int(value) for value in np.unique(cleaned) if value > 0)
    if not kept_values and len(nonzero_unique_values) > MAX_EXPECTED_MASK_STENCIL_VALUES:
        unreal.log_warning(
            f"Rejected mask '{mask_path}' because it contains "
            f"{len(nonzero_unique_values)} nonzero values before cleanup and no "
            "large connected gate-ID components. Raw copy saved for inspection."
        )
        return False

    unreal.log(
        f"Cleaned mask '{mask_path}' to gate stencil values {kept_values} "
        f"from {len(nonzero_unique_values)} raw nonzero value(s)."
    )
    return True


def raw_mask_debug_path(mask_path):
    if "." not in mask_path:
        return f"{mask_path}_raw"
    stem, extension = mask_path.rsplit(".", 1)
    return f"{stem}_raw.{extension}"


def clean_mask_to_gate_stencils(values, gate_stencil_values, np):
    cleaned = np.zeros_like(values, dtype=np.uint8)
    min_component_pixels = max(1, int(MASK_MIN_COMPONENT_PIXELS))
    sorted_stencil_values = sorted(int(value) for value in gate_stencil_values)

    if not sorted_stencil_values:
        return cleaned

    # SceneCapture exports can slightly shade/antialias the mask material. Bucket
    # each non-background pixel to the nearest configured gate ID instead of only
    # keeping exact byte matches.
    background_threshold = max(1, sorted_stencil_values[0] // 2)

    value_array = values.astype(np.int16)
    for stencil_index, stencil_value in enumerate(sorted_stencil_values):
        lower_bound = background_threshold
        if stencil_index > 0:
            previous_value = sorted_stencil_values[stencil_index - 1]
            lower_bound = (previous_value + stencil_value) // 2

        upper_bound = 255
        if stencil_index < len(sorted_stencil_values) - 1:
            next_value = sorted_stencil_values[stencil_index + 1]
            upper_bound = (stencil_value + next_value) // 2

        if stencil_index == 0:
            candidate = (value_array >= lower_bound) & (value_array <= upper_bound)
        else:
            candidate = (value_array > lower_bound) & (value_array <= upper_bound)

        if not candidate.any():
            continue
        filtered = connected_component_filter(candidate, min_component_pixels, np)
        cleaned[filtered] = stencil_value

    return cleaned


def connected_component_filter(mask, min_pixels, np):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    keep = np.zeros(mask.shape, dtype=bool)

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue

            component = []
            stack = [(x, y)]
            visited[y, x] = True
            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and not visited[next_y, next_x]
                        and mask[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_x, next_y))

            if len(component) >= min_pixels:
                for current_x, current_y in component:
                    keep[current_y, current_x] = True

    return keep


def mask_pixels_for_stencil(mask_index, stencil_value):
    empty = {
        "stencil_value": stencil_value,
        "decode_mode": None,
        "pixel_count": 0,
        "bbox_2d_px": None,
        "row_runs_px": [],
    }
    if not mask_index or stencil_value is None or not mask_index.get("valid_stencil_id_mask", True):
        return empty

    try:
        import numpy as np
    except Exception:
        return empty

    values = mask_index["values"]
    matches = values == int(stencil_value)
    decode_mode = "raw"
    if not matches.any() and "srgb_decoded_values" in mask_index:
        matches = mask_index["srgb_decoded_values"] == int(stencil_value)
        decode_mode = "srgb_decoded"
    y_indices, x_indices = np.nonzero(matches)
    pixel_count = int(x_indices.size)
    if pixel_count <= 0:
        return empty

    row_runs = []
    for y in np.unique(y_indices):
        row_x = x_indices[y_indices == y]
        row_x.sort()
        run_start = int(row_x[0])
        previous = int(row_x[0])
        for x_value in row_x[1:]:
            x_value = int(x_value)
            if x_value == previous + 1:
                previous = x_value
                continue
            row_runs.append(
                {
                    "y_px": int(y),
                    "x_start_px": run_start,
                    "x_end_px": previous,
                }
            )
            run_start = x_value
            previous = x_value
        row_runs.append(
            {
                "y_px": int(y),
                "x_start_px": run_start,
                "x_end_px": previous,
            }
        )

    x_min = int(x_indices.min())
    x_max = int(x_indices.max())
    y_min = int(y_indices.min())
    y_max = int(y_indices.max())
    return {
        "stencil_value": int(stencil_value),
        "decode_mode": decode_mode,
        "pixel_count": pixel_count,
        "bbox_2d_px": {
            "bottom_left": {
                "x_px": x_min,
                "y_px": y_min,
            },
            "top_right": {
                "x_px": x_max,
                "y_px": y_max,
            },
            "x_min_px": x_min,
            "y_min_px": y_min,
            "x_max_px": x_max,
            "y_max_px": y_max,
            "width_px": int(x_max - x_min + 1),
            "height_px": int(y_max - y_min + 1),
        },
        "row_runs_px": row_runs,
    }
