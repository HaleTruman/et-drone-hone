"""Visibility and occlusion trace helpers."""

import unreal

from src.dataset_generation.common import (
    actor_label,
    object_path_name,
    vector_to_meters_dict,
)
from src.dataset_generation.config import (
    VISIBILITY_TRACE_IGNORE_ACTOR_PREFIXES,
    VISIBILITY_TRACE_IGNORE_COMPONENT_NAME_FRAGMENTS,
    VISIBILITY_TRACE_IGNORE_COMPONENT_TAGS,
)

def hit_actor_label(hit_result):
    actor = hit_actor(hit_result)

    if not actor:
        return None

    try:
        return actor.get_actor_label()
    except Exception:
        return str(actor)


def hit_actor(hit_result):
    actor = hit_property(hit_result, "actor")
    if actor:
        return actor
    actor = hit_property(hit_result, "hit_actor")
    if actor:
        return actor

    try:
        return hit_result.get_actor()
    except Exception:
        return None


def hit_component(hit_result):
    for property_name in ("component", "hit_component"):
        component = hit_property(hit_result, property_name)
        if component:
            return component

    for getter_name in ("get_component", "get_hit_component"):
        getter = getattr(hit_result, getter_name, None)
        if getter:
            try:
                component = getter()
                if component:
                    return component
            except Exception:
                pass

    for property_name in ("component", "hit_component"):
        try:
            component = getattr(hit_result, property_name)
            if component:
                return component
        except Exception:
            pass

    return None


def object_name(obj):
    for getter_name in ("get_name", "get_fname"):
        getter = getattr(obj, getter_name, None)
        if getter:
            try:
                return str(getter())
            except Exception:
                pass
    return str(obj) if obj else ""


def object_class_name(obj):
    get_class = getattr(obj, "get_class", None)
    if get_class:
        try:
            return object_name(get_class())
        except Exception:
            pass
    return ""


def component_tags(component):
    try:
        tags = component.get_editor_property("component_tags")
    except Exception:
        return ()
    return tuple(str(tag) for tag in tags)


def hit_dict(hit_result):
    to_dict = getattr(hit_result, "to_dict", None)
    if not to_dict:
        return {}
    try:
        value = to_dict()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def dict_value_case_insensitive(values, property_name):
    normalized = property_name.lower()
    for key, value in values.items():
        if str(key).lower() == normalized:
            return value
    return None


def hit_property(hit_result, property_name):
    values = hit_dict(hit_result)
    if values:
        value = dict_value_case_insensitive(values, property_name)
        if value is not None:
            return value

    try:
        return getattr(hit_result, property_name)
    except Exception:
        pass
    try:
        return hit_result.get_editor_property(property_name)
    except Exception:
        return None


def hit_float_property(hit_result, property_name):
    value = hit_property(hit_result, property_name)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def hit_vector_property(hit_result, property_name):
    value = hit_property(hit_result, property_name)
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            return {
                "x": float(dict_value_case_insensitive(value, "x")) / 100.0,
                "y": float(dict_value_case_insensitive(value, "y")) / 100.0,
                "z": float(dict_value_case_insensitive(value, "z")) / 100.0,
            }
        except Exception:
            return None
    if not all(hasattr(value, axis) for axis in ("x", "y", "z")):
        return None
    try:
        return vector_to_meters_dict(value)
    except Exception:
        return None


def object_type_name(obj):
    try:
        obj_type = type(obj)
        return f"{obj_type.__module__}.{obj_type.__name__}"
    except Exception:
        return ""


def object_string(obj):
    try:
        return str(obj)
    except Exception:
        return ""


def public_attribute_names(obj, limit=80):
    try:
        names = dir(obj)
    except Exception:
        return []
    public_names = [name for name in names if not name.startswith("_")]
    return public_names[:limit]


def json_safe_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return object_string(value)


def compact_hit_details(hit_result):
    actor = hit_actor(hit_result)
    component = hit_component(hit_result)
    details = {
        "hit_type": object_type_name(hit_result),
        "hit_string": object_string(hit_result),
        "hit_public_attributes": public_attribute_names(hit_result),
        "actor_label": actor_label(actor) if actor else None,
        "actor_name": object_name(actor),
        "actor_class": object_class_name(actor),
        "actor_path": object_path_name(actor) if actor else "",
        "component_name": object_name(component),
        "component_class": object_class_name(component),
        "component_path": object_path_name(component) if component else "",
        "component_tags": list(component_tags(component)) if component else [],
        "distance_m": (
            None
            if hit_float_property(hit_result, "distance") is None
            else hit_float_property(hit_result, "distance") / 100.0
        ),
        "time": hit_float_property(hit_result, "time"),
        "location_m": hit_vector_property(hit_result, "location"),
        "impact_point_m": hit_vector_property(hit_result, "impact_point"),
        "trace_start_m": hit_vector_property(hit_result, "trace_start"),
        "trace_end_m": hit_vector_property(hit_result, "trace_end"),
    }
    return {key: value for key, value in details.items() if value not in (None, "", [])}


def hit_sort_distance(hit_result):
    for property_name in ("distance", "time"):
        value = hit_float_property(hit_result, property_name)
        if value is not None:
            return value
    return 0.0


def hit_should_be_ignored(hit_result):
    label = hit_actor_label(hit_result)
    prefixes = tuple(VISIBILITY_TRACE_IGNORE_ACTOR_PREFIXES)
    if label and any(label.startswith(prefix) for prefix in prefixes):
        return True

    component = hit_component(hit_result)
    if not component:
        return False

    ignored_tags = tuple(VISIBILITY_TRACE_IGNORE_COMPONENT_TAGS)
    if ignored_tags and any(tag in ignored_tags for tag in component_tags(component)):
        return True

    component_name = object_name(component)
    fragments = tuple(VISIBILITY_TRACE_IGNORE_COMPONENT_NAME_FRAGMENTS)
    return bool(component_name and any(fragment in component_name for fragment in fragments))


def first_blocking_hit(hit_results):
    ignored = []
    for hit_result in sorted(hit_results, key=hit_sort_distance):
        if hit_should_be_ignored(hit_result):
            ignored.append(compact_hit_details(hit_result))
            continue
        return hit_result, ignored
    return None, ignored


def single_trace_hit_result(hit):
    did_hit = False
    hit_result = None
    if isinstance(hit, tuple):
        did_hit = bool(hit[0])
        hit_result = hit[1] if len(hit) > 1 else None
    else:
        did_hit = bool(hit)
    return hit_result if did_hit else None


def multi_trace_hit_results(hit):
    if isinstance(hit, tuple):
        if len(hit) > 1 and hit[1]:
            return list(hit[1])
        return []
    return list(hit) if hit else []


def run_trace(line_trace, world_context, start, end, trace_channel, ignored_actors, draw_debug):
    try:
        return line_trace(
            world_context,
            start,
            end,
            trace_channel,
            True,
            ignored_actors,
            draw_debug,
            True,
        )
    except TypeError:
        return line_trace(
            world_context,
            start,
            end,
            trace_channel,
            True,
            ignored_actors,
            draw_debug,
            True,
            unreal.LinearColor(1.0, 0.0, 0.0, 1.0),
            unreal.LinearColor(0.0, 1.0, 0.0, 1.0),
            0.0,
        )


def trace_visibility_to_point(world_context, camera, gate, world_location):
    system_library = getattr(unreal, "SystemLibrary", None)
    line_trace_multi = getattr(system_library, "line_trace_multi", None) if system_library else None
    line_trace_single = getattr(system_library, "line_trace_single", None) if system_library else None
    trace_type_query = getattr(unreal, "TraceTypeQuery", None)
    draw_debug_trace = getattr(unreal, "DrawDebugTrace", None)

    if not (line_trace_multi or line_trace_single) or not trace_type_query:
        return {
            "occlusion_tested": False,
            "line_of_sight": None,
            "occluded": None,
            "occluder": None,
        }

    trace_channel = getattr(trace_type_query, "TRACE_TYPE_QUERY1", None)
    if trace_channel is None:
        return {
            "occlusion_tested": False,
            "line_of_sight": None,
            "occluded": None,
            "occluder": None,
        }

    draw_debug = getattr(draw_debug_trace, "NONE", None) if draw_debug_trace else None
    if draw_debug is None and draw_debug_trace:
        draw_debug = getattr(draw_debug_trace, "FOR_DURATION", None)

    ignored_actors = [camera] + visibility_trace_ignored_actors()
    start = camera.get_actor_location()

    try:
        ignored_occluders = []
        hit_result = None
        if line_trace_multi:
            hit_results = multi_trace_hit_results(
                run_trace(
                    line_trace_multi,
                    world_context,
                    start,
                    world_location,
                    trace_channel,
                    ignored_actors,
                    draw_debug,
                )
            )
            hit_result, ignored_occluders = first_blocking_hit(hit_results)
        elif line_trace_single:
            hit_result = single_trace_hit_result(
                run_trace(
                    line_trace_single,
                    world_context,
                    start,
                    world_location,
                    trace_channel,
                    ignored_actors,
                    draw_debug,
                )
            )
            if hit_result and hit_should_be_ignored(hit_result):
                ignored_occluders.append(compact_hit_details(hit_result))
                hit_result = None
    except Exception as exc:
        if not line_trace_single:
            return {
                "occlusion_tested": False,
                "line_of_sight": None,
                "occluded": None,
                "occluder": None,
                "error": str(exc),
            }
        try:
            hit_result = single_trace_hit_result(
                run_trace(
                    line_trace_single,
                    world_context,
                    start,
                    world_location,
                    trace_channel,
                    ignored_actors,
                    draw_debug,
                )
            )
            ignored_occluders = []
            if hit_result and hit_should_be_ignored(hit_result):
                ignored_occluders.append(compact_hit_details(hit_result))
                hit_result = None
        except Exception as fallback_exc:
            return {
                "occlusion_tested": False,
                "line_of_sight": None,
                "occluded": None,
                "occluder": None,
                "error": str(fallback_exc),
            }

    occluder = hit_actor_label(hit_result) if hit_result else None
    gate_label = gate.get_actor_label()
    visible = (hit_result is None) or occluder == gate_label
    result = {
        "occlusion_tested": True,
        "line_of_sight": bool(visible),
        "occluded": bool(not visible),
        "occluder": None if visible else occluder,
    }
    if ignored_occluders:
        result["ignored_occluders"] = [
            hit.get("actor_label") or hit.get("component_name") or hit.get("component_path")
            for hit in ignored_occluders
        ]
        result["ignored_hit_details"] = ignored_occluders
    if not visible and hit_result:
        result["blocking_hit"] = compact_hit_details(hit_result)
    return result


def visibility_trace_ignored_actors():
    ignored = []
    prefixes = tuple(VISIBILITY_TRACE_IGNORE_ACTOR_PREFIXES)
    if not prefixes:
        return ignored

    actor_subsystem = get_editor_actor_subsystem()
    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if actor_subsystem:
        actors = actor_subsystem.get_all_level_actors()
    elif editor_level_library:
        actors = editor_level_library.get_all_level_actors()
    else:
        return ignored

    for actor in actors:
        label = actor_label(actor)
        if any(label.startswith(prefix) for prefix in prefixes):
            ignored.append(actor)
    return ignored


def get_editor_actor_subsystem():
    get_subsystem = getattr(unreal, "get_editor_subsystem", None)
    subsystem_class = getattr(unreal, "EditorActorSubsystem", None)
    if not get_subsystem or not subsystem_class:
        return None
    try:
        return get_subsystem(subsystem_class)
    except Exception:
        return None

def image_visibility(projected, line_of_sight):
    visible = bool(projected.get("inside_frame") and line_of_sight.get("line_of_sight") is not False)
    return {
        "visible": visible,
        "inside_frame": bool(projected.get("inside_frame")),
        "line_of_sight": line_of_sight,
    }
