"""Duplicate placed track indicator segment actors along the generated spline."""

import math

import unreal

from src.dataset_generation.common import distance, lerp
from src.dataset_generation.config import (
    HIDE_TRACK_INDICATOR_TEMPLATE,
    TRACK_INDICATOR_SEGMENT_PREFIX,
    TRACK_INDICATOR_SEGMENT_ROTATION_OFFSET_DEG,
    TRACK_INDICATOR_SEGMENT_SPACING_CM,
    TRACK_INDICATOR_SEGMENTS_FOLDER,
    TRACK_INDICATOR_TEMPLATE_LABEL,
)
from src.dataset_generation.unreal_editor import all_level_actors, invalidate_viewports


def rebuild_track_indicator_segments(points):
    if len(points) < 2:
        remove_generated_segments()
        return []

    template = find_actor_by_label(TRACK_INDICATOR_TEMPLATE_LABEL)
    if not template:
        unreal.log_warning(
            f"Track indicator template actor '{TRACK_INDICATOR_TEMPLATE_LABEL}' was not found; "
            "skipping generated track indicator segments."
        )
        remove_generated_segments()
        return []

    remove_generated_segments()
    set_actor_visible(template, True)
    samples = sample_points_and_tangents(points, segment_spacing_cm(template))
    generated = []
    for index, sample in enumerate(samples):
        actor = duplicate_actor(template)
        if not actor:
            unreal.log_warning("Could not duplicate track indicator template actor.")
            continue

        actor.modify()
        set_actor_label(actor, f"{TRACK_INDICATOR_SEGMENT_PREFIX}{index:03d}")
        set_actor_folder(actor, TRACK_INDICATOR_SEGMENTS_FOLDER)
        actor.set_actor_location(sample["location"], False, False)
        actor.set_actor_rotation(rotation_from_tangent(sample["tangent"]), False)
        set_actor_visible(actor, True)
        set_actor_collision(actor, False)
        generated.append(actor)

    if HIDE_TRACK_INDICATOR_TEMPLATE:
        template.modify()
        set_actor_visible(template, False)

    invalidate_viewports()
    unreal.log(
        f"Rebuilt track indicator from '{TRACK_INDICATOR_TEMPLATE_LABEL}' with "
        f"{len(generated)} duplicated segment actor(s)."
    )
    return generated


def remove_generated_segments():
    for actor in list(all_level_actors()):
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        if not label.startswith(TRACK_INDICATOR_SEGMENT_PREFIX):
            continue

        actor.modify()
        destroy_actor(actor)


def find_actor_by_label(label):
    for actor in all_level_actors():
        try:
            if actor.get_actor_label() == label:
                return actor
        except Exception:
            pass
    return None


def duplicate_actor(template):
    actor_subsystem = get_editor_actor_subsystem()
    if actor_subsystem:
        duplicate = getattr(actor_subsystem, "duplicate_actor", None)
        if duplicate:
            try:
                return duplicate(template, template.get_actor_location())
            except TypeError:
                try:
                    return duplicate(template)
                except Exception:
                    pass
            except Exception as exc:
                unreal.log_warning(f"EditorActorSubsystem.duplicate_actor failed: {exc}")

    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if editor_level_library:
        duplicate = getattr(editor_level_library, "duplicate_actor", None)
        if duplicate:
            try:
                return duplicate(template, template.get_actor_location())
            except TypeError:
                try:
                    return duplicate(template)
                except Exception:
                    pass
            except Exception as exc:
                unreal.log_warning(f"EditorLevelLibrary.duplicate_actor failed: {exc}")

    return None


def destroy_actor(actor):
    actor_subsystem = get_editor_actor_subsystem()
    if actor_subsystem:
        destroy = getattr(actor_subsystem, "destroy_actor", None)
        if destroy:
            try:
                destroy(actor)
                return
            except Exception:
                pass

    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if editor_level_library:
        destroy = getattr(editor_level_library, "destroy_actor", None)
        if destroy:
            destroy(actor)


def get_editor_actor_subsystem():
    get_subsystem = getattr(unreal, "get_editor_subsystem", None)
    subsystem_class = getattr(unreal, "EditorActorSubsystem", None)
    if not get_subsystem or not subsystem_class:
        return None
    try:
        return get_subsystem(subsystem_class)
    except Exception:
        return None


def set_actor_label(actor, label):
    try:
        actor.set_actor_label(label)
    except Exception:
        pass


def set_actor_folder(actor, folder):
    for setter_name in ("set_folder_path", "set_folder_path_"):
        setter = getattr(actor, setter_name, None)
        if setter:
            try:
                setter(folder)
                return
            except Exception:
                pass
    try:
        actor.set_editor_property("folder_path", folder)
    except Exception:
        pass


def set_actor_visible(actor, visible):
    try:
        actor.set_actor_hidden_in_game(not visible)
    except Exception:
        pass
    set_hidden_editor = getattr(actor, "set_is_temporarily_hidden_in_editor", None)
    if set_hidden_editor:
        try:
            set_hidden_editor(not visible)
        except Exception:
            pass


def set_actor_collision(actor, enabled):
    set_collision = getattr(actor, "set_actor_enable_collision", None)
    if set_collision:
        try:
            set_collision(enabled)
        except Exception:
            pass

    collision_enabled = getattr(unreal, "CollisionEnabled", None)
    no_collision = getattr(collision_enabled, "NO_COLLISION", None) if collision_enabled else None
    query_and_physics = (
        getattr(collision_enabled, "QUERY_AND_PHYSICS", None)
        if collision_enabled
        else None
    )
    target_collision = query_and_physics if enabled else no_collision

    for component in actor_components(actor):
        if target_collision is not None:
            setter = getattr(component, "set_collision_enabled", None)
            if setter:
                try:
                    setter(target_collision)
                except Exception:
                    pass
        if not enabled:
            setter = getattr(component, "set_collision_profile_name", None)
            if setter:
                try:
                    setter("NoCollision")
                except Exception:
                    pass


def actor_components(actor):
    components = []
    seen = set()
    for class_name in ("PrimitiveComponent", "SceneComponent", "ActorComponent"):
        component_class = getattr(unreal, class_name, None)
        if not component_class:
            continue
        try:
            found = actor.get_components_by_class(component_class)
        except Exception:
            continue
        for component in found:
            key = str(component)
            if key in seen:
                continue
            seen.add(key)
            components.append(component)
    return components


def segment_spacing_cm(template):
    configured = float(TRACK_INDICATOR_SEGMENT_SPACING_CM)
    if configured > 0.0:
        return configured

    try:
        _, extent = template.get_actor_bounds(False)
        return max(25.0, max(abs(extent.x), abs(extent.y), abs(extent.z)) * 2.0)
    except Exception:
        return 100.0


def sample_points_and_tangents(points, spacing_cm):
    cumulative = cumulative_distances(points)
    total_length = cumulative[-1]
    if total_length <= 0.0:
        return []

    count = max(1, int(math.ceil(total_length / max(1.0, spacing_cm))) + 1)
    return [sample_at_distance(points, cumulative, min(total_length, index * spacing_cm)) for index in range(count)]


def cumulative_distances(points):
    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + distance(points[index - 1], points[index]))
    return cumulative


def sample_at_distance(points, cumulative, target_distance):
    point_index = 1
    while point_index < len(cumulative) - 1 and cumulative[point_index] < target_distance:
        point_index += 1

    previous_distance = cumulative[point_index - 1]
    next_distance = cumulative[point_index]
    segment_length = max(0.001, next_distance - previous_distance)
    alpha = (target_distance - previous_distance) / segment_length
    previous_point = points[point_index - 1]
    next_point = points[point_index]
    location = unreal.Vector(
        lerp(previous_point.x, next_point.x, alpha),
        lerp(previous_point.y, next_point.y, alpha),
        lerp(previous_point.z, next_point.z, alpha),
    )
    tangent = unreal.Vector(
        next_point.x - previous_point.x,
        next_point.y - previous_point.y,
        next_point.z - previous_point.z,
    )
    return {"location": location, "tangent": tangent}


def rotation_from_tangent(tangent):
    if distance(tangent, unreal.Vector(0.0, 0.0, 0.0)) <= 0.001:
        rotation = unreal.Rotator(0.0, 0.0, 0.0)
    else:
        rotation = make_track_frame_rotation(tangent)

    pitch_offset, yaw_offset, roll_offset = TRACK_INDICATOR_SEGMENT_ROTATION_OFFSET_DEG
    return unreal.Rotator(
        rotation.pitch + pitch_offset,
        rotation.yaw + yaw_offset,
        rotation.roll + roll_offset,
    )


def make_track_frame_rotation(tangent):
    ground_up = unreal.Vector(0.0, 0.0, 1.0)
    normalized_tangent = normalized_vector(tangent)
    leveled_up = vector_reject(ground_up, normalized_tangent)
    if vector_length(leveled_up) <= 0.001:
        leveled_up = vector_reject(unreal.Vector(0.0, 1.0, 0.0), normalized_tangent)
    leveled_up = normalized_vector(leveled_up)

    math_library = getattr(unreal, "MathLibrary", None)
    if math_library:
        make_rot_from_xz = getattr(math_library, "make_rot_from_xz", None)
        if make_rot_from_xz:
            return make_rot_from_xz(normalized_tangent, leveled_up)

        make_rot_from_xy = getattr(math_library, "make_rot_from_xy", None)
        if make_rot_from_xy:
            right = normalized_vector(cross(leveled_up, normalized_tangent))
            return make_rot_from_xy(normalized_tangent, right)

    return unreal.MathLibrary.find_look_at_rotation(unreal.Vector(0.0, 0.0, 0.0), normalized_tangent)


def vector_reject(vector, normal):
    projection_scale = dot(vector, normal)
    return unreal.Vector(
        vector.x - normal.x * projection_scale,
        vector.y - normal.y * projection_scale,
        vector.z - normal.z * projection_scale,
    )


def normalized_vector(vector):
    length = vector_length(vector)
    if length <= 0.001:
        return unreal.Vector(1.0, 0.0, 0.0)
    return unreal.Vector(vector.x / length, vector.y / length, vector.z / length)


def vector_length(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def dot(left, right):
    return left.x * right.x + left.y * right.y + left.z * right.z


def cross(left, right):
    return unreal.Vector(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )
