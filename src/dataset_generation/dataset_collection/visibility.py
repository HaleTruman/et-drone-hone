"""Visibility and occlusion trace helpers."""

import unreal

from src.dataset_generation.common import actor_label
from src.dataset_generation.config import VISIBILITY_TRACE_IGNORE_ACTOR_PREFIXES

def hit_actor_label(hit_result):
    try:
        actor = hit_result.get_actor()
    except Exception:
        actor = None

    if not actor:
        return None

    try:
        return actor.get_actor_label()
    except Exception:
        return str(actor)

def trace_visibility_to_point(world_context, camera, gate, world_location):
    system_library = getattr(unreal, "SystemLibrary", None)
    line_trace = getattr(system_library, "line_trace_single", None) if system_library else None
    trace_type_query = getattr(unreal, "TraceTypeQuery", None)
    draw_debug_trace = getattr(unreal, "DrawDebugTrace", None)

    if not line_trace or not trace_type_query:
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

    try:
        hit = line_trace(
            world_context,
            camera.get_actor_location(),
            world_location,
            trace_channel,
            True,
            ignored_actors,
            draw_debug,
            True,
        )
    except TypeError:
        try:
            hit = line_trace(
                world_context,
                camera.get_actor_location(),
                world_location,
                trace_channel,
                True,
                ignored_actors,
                draw_debug,
                True,
                unreal.LinearColor(1.0, 0.0, 0.0, 1.0),
                unreal.LinearColor(0.0, 1.0, 0.0, 1.0),
                0.0,
            )
        except Exception as exc:
            return {
                "occlusion_tested": False,
                "line_of_sight": None,
                "occluded": None,
                "occluder": None,
                "error": str(exc),
            }
    except Exception as exc:
        return {
            "occlusion_tested": False,
            "line_of_sight": None,
            "occluded": None,
            "occluder": None,
            "error": str(exc),
        }

    did_hit = False
    hit_result = None
    if isinstance(hit, tuple):
        did_hit = bool(hit[0])
        hit_result = hit[1] if len(hit) > 1 else None
    else:
        did_hit = bool(hit)

    occluder = hit_actor_label(hit_result) if hit_result else None
    gate_label = gate.get_actor_label()
    visible = (not did_hit) or occluder == gate_label
    return {
        "occlusion_tested": True,
        "line_of_sight": bool(visible),
        "occluded": bool(not visible),
        "occluder": None if visible else occluder,
    }


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
