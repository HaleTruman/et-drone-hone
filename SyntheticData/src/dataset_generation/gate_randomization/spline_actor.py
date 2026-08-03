"""Persistent editor spline actor visualization for generated gate tracks."""

import unreal

from SyntheticData.src.dataset_generation.config import (
    COMPILE_TRACK_BLUEPRINT_AFTER_SPLINE_UPDATE,
    CREATE_TRACK_INDICATOR_SEGMENTS,
    TRACK_RENDER_READY_BOOL_PROPERTY,
    TRACK_SPLINE_CREATE_IF_MISSING,
    TRACK_SPLINE_ACTOR_FOLDER,
    TRACK_SPLINE_ACTOR_LABEL,
)
from SyntheticData.src.dataset_generation.gate_randomization.track_indicator_segments import rebuild_track_indicator_segments
from SyntheticData.src.dataset_generation.unreal_editor import all_level_actors


def find_actor_by_label(label):
    for actor in all_level_actors():
        try:
            if actor.get_actor_label() == label:
                return actor
        except Exception:
            pass
    return None


def spawn_empty_actor(label):
    actor_class = getattr(unreal, "Actor", None)
    if not actor_class:
        raise RuntimeError("unreal.Actor is unavailable; cannot create track spline actor.")

    actor = None
    actor_subsystem = get_editor_actor_subsystem()
    if actor_subsystem and hasattr(actor_subsystem, "spawn_actor_from_class"):
        actor = actor_subsystem.spawn_actor_from_class(actor_class, unreal.Vector(0.0, 0.0, 0.0))

    if not actor:
        editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
        if editor_level_library and hasattr(editor_level_library, "spawn_actor_from_class"):
            actor = editor_level_library.spawn_actor_from_class(
                actor_class,
                unreal.Vector(0.0, 0.0, 0.0),
                unreal.Rotator(0.0, 0.0, 0.0),
            )

    if not actor:
        raise RuntimeError("Could not spawn an Actor for generated track spline visualization.")

    try:
        actor.set_actor_label(label)
    except Exception:
        pass
    set_actor_folder(actor, TRACK_SPLINE_ACTOR_FOLDER)
    return actor


def get_editor_actor_subsystem():
    get_subsystem = getattr(unreal, "get_editor_subsystem", None)
    subsystem_class = getattr(unreal, "EditorActorSubsystem", None)
    if not get_subsystem or not subsystem_class:
        return None
    try:
        return get_subsystem(subsystem_class)
    except Exception:
        return None


def actor_spline_component(actor):
    spline_class = getattr(unreal, "SplineComponent", None)
    if not spline_class:
        raise RuntimeError("unreal.SplineComponent is unavailable.")

    try:
        components = actor.get_components_by_class(spline_class)
        if components:
            return components[0]
    except Exception:
        pass

    add_component = getattr(actor, "add_component_by_class", None)
    if add_component:
        try:
            component = add_component(
                spline_class,
                False,
                unreal.Transform(),
                False,
            )
            component.set_editor_property("component_tags", ["GeneratedTrackSpline"])
            return component
        except Exception:
            pass

    add_component = getattr(actor, "add_component", None)
    if add_component:
        try:
            component = add_component(
                "GeneratedTrackSpline",
                False,
                unreal.Transform(),
                None,
            )
            return component
        except Exception:
            pass

    raise RuntimeError(
        "Could not add a SplineComponent to the track spline actor. "
        f"Create an actor with a SplineComponent named {TRACK_SPLINE_ACTOR_LABEL}, then rerun."
    )


def coordinate_space_world():
    coordinate_space = getattr(unreal, "SplineCoordinateSpace", None)
    if coordinate_space and hasattr(coordinate_space, "WORLD"):
        return coordinate_space.WORLD

    coordinate_space = getattr(unreal, "CoordinateSpace", None)
    if coordinate_space and hasattr(coordinate_space, "WORLD"):
        return coordinate_space.WORLD

    return 1


def curve_point_type():
    point_type = getattr(unreal, "SplinePointType", None)
    if point_type and hasattr(point_type, "CURVE"):
        return point_type.CURVE
    return None


def update_spline_component_points(spline_component, points):
    enable_construction_script_spline_input(spline_component)
    original_count = spline_point_count(spline_component)
    clear_spline_component_points(spline_component)
    cleared_count = spline_point_count(spline_component)

    world_space = coordinate_space_world()
    if cleared_count > 0:
        replace_spline_component_points(spline_component, points, world_space)
        final_count = spline_point_count(spline_component)
        unreal.log(
            f"Replaced spline points on '{TRACK_SPLINE_ACTOR_LABEL}' "
            f"without append: before={original_count}, after_clear={cleared_count}, "
            f"final={final_count}."
        )
        return

    add_point = getattr(spline_component, "add_spline_point", None)
    if not add_point:
        raise RuntimeError("SplineComponent.add_spline_point is unavailable.")

    for index, point in enumerate(points):
        try:
            add_point(point, world_space, False)
        except TypeError:
            add_point(point, world_space)

        point_type = curve_point_type()
        set_point_type = getattr(spline_component, "set_spline_point_type", None)
        if point_type is not None and set_point_type:
            try:
                set_point_type(index, point_type, False)
            except TypeError:
                set_point_type(index, point_type)

    update_spline = getattr(spline_component, "update_spline", None)
    if update_spline:
        update_spline()

    unreal.log(
        f"Updated spline points on '{TRACK_SPLINE_ACTOR_LABEL}': "
        f"before={original_count}, after_clear={cleared_count}, final={spline_point_count(spline_component)}."
    )


def replace_spline_component_points(spline_component, points, coordinate_space):
    add_point = getattr(spline_component, "add_spline_point", None)
    set_location = getattr(spline_component, "set_location_at_spline_point", None)
    remove_point = getattr(spline_component, "remove_spline_point", None)
    if not set_location:
        raise RuntimeError(
            "SplineComponent.set_location_at_spline_point is unavailable; "
            "cannot replace existing BP spline points without appending."
        )

    while spline_point_count(spline_component) > len(points):
        index = spline_point_count(spline_component) - 1
        if not remove_point:
            raise RuntimeError(
                "SplineComponent.remove_spline_point is unavailable; "
                "cannot trim existing BP spline points."
            )
        try:
            remove_point(index, False)
        except TypeError:
            remove_point(index)

    while spline_point_count(spline_component) < len(points):
        if not add_point:
            raise RuntimeError("SplineComponent.add_spline_point is unavailable.")
        try:
            add_point(points[spline_point_count(spline_component)], coordinate_space, False)
        except TypeError:
            add_point(points[spline_point_count(spline_component)], coordinate_space)

    for index, point in enumerate(points):
        try:
            set_location(index, point, coordinate_space, False)
        except TypeError:
            set_location(index, point, coordinate_space)

        point_type = curve_point_type()
        set_point_type = getattr(spline_component, "set_spline_point_type", None)
        if point_type is not None and set_point_type:
            try:
                set_point_type(index, point_type, False)
            except TypeError:
                set_point_type(index, point_type)

    update_spline = getattr(spline_component, "update_spline", None)
    if update_spline:
        update_spline()


def clear_spline_component_points(spline_component):
    clear_points = getattr(spline_component, "clear_spline_points", None)
    if clear_points:
        try:
            clear_points(False)
        except TypeError:
            clear_points()

    if spline_point_count(spline_component) > 0:
        remove_point = getattr(spline_component, "remove_spline_point", None)
        if remove_point:
            for index in range(spline_point_count(spline_component) - 1, -1, -1):
                try:
                    remove_point(index, False)
                except TypeError:
                    remove_point(index)

    update_spline = getattr(spline_component, "update_spline", None)
    if update_spline:
        update_spline()

    remaining = spline_point_count(spline_component)
    if remaining > 0:
        unreal.log_warning(
            f"Spline clear left {remaining} point(s) on '{TRACK_SPLINE_ACTOR_LABEL}'. "
            "New points may append if this component blocks runtime clearing."
        )


def spline_point_count(spline_component):
    getter = getattr(spline_component, "get_number_of_spline_points", None)
    if getter:
        try:
            return int(getter())
        except Exception:
            pass
    return 0


def update_track_spline_actor(points):
    if not points:
        return None

    actor = target_track_spline_actor()
    actor.modify()
    set_actor_folder(actor, TRACK_SPLINE_ACTOR_FOLDER)
    reset_track_ready_flag(actor)
    spline_component = actor_spline_component(actor)
    spline_component.modify()
    update_spline_component_points(spline_component, points)
    rebuild_track_meshes(actor, spline_component)
    if CREATE_TRACK_INDICATOR_SEGMENTS:
        rebuild_track_indicator_segments(points)
    unreal.log(
        f"Updated track spline actor '{TRACK_SPLINE_ACTOR_LABEL}' with {len(points)} spline points."
    )
    return actor


def reset_track_ready_flag(actor):
    if not TRACK_RENDER_READY_BOOL_PROPERTY:
        return
    try:
        actor.set_editor_property(TRACK_RENDER_READY_BOOL_PROPERTY, False)
    except Exception:
        pass


def target_track_spline_actor():
    actor = find_actor_by_label(TRACK_SPLINE_ACTOR_LABEL)
    if actor:
        return actor
    if TRACK_SPLINE_CREATE_IF_MISSING:
        return spawn_empty_actor(TRACK_SPLINE_ACTOR_LABEL)
    raise RuntimeError(
        f"Could not find track spline actor '{TRACK_SPLINE_ACTOR_LABEL}'. "
        "Create it in the level or set TRACK_SPLINE_CREATE_IF_MISSING=True."
    )


def rebuild_track_meshes(actor, spline_component):
    enable_construction_script_spline_input(spline_component)
    post_edit_change(spline_component)
    post_edit_change(actor)

    call_rebuild_track_meshes(actor)
    compile_actor_blueprint(actor)
    refresh_actor_render_components(actor)
    post_edit_change(actor)


def call_rebuild_track_meshes(actor):
    call_method = getattr(actor, "call_method", None)
    if not call_method:
        raise RuntimeError(
            f"Actor '{TRACK_SPLINE_ACTOR_LABEL}' does not expose call_method()."
        )

    call_method("RebuildTrackMeshes")
    unreal.log(f"Called RebuildTrackMeshes on '{TRACK_SPLINE_ACTOR_LABEL}'.")


def enable_construction_script_spline_input(spline_component):
    for property_name in (
        "input_spline_points_to_construction_script",
        "b_input_spline_points_to_construction_script",
        "bInputSplinePointsToConstructionScript",
    ):
        try:
            spline_component.set_editor_property(property_name, True)
            return
        except Exception:
            pass


def post_edit_change(obj):
    post_edit_change_method = getattr(obj, "post_edit_change", None)
    if post_edit_change_method:
        try:
            post_edit_change_method()
        except Exception:
            pass


def trigger_actor_editor_refresh(actor):
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    actor.set_actor_location(location, False, False)
    actor.set_actor_rotation(rotation, False)
    post_edit_move = getattr(actor, "post_edit_move", None)
    if post_edit_move:
        try:
            post_edit_move(True)
        except TypeError:
            post_edit_move()


def refresh_actor_render_components(actor):
    for component in actor_components(actor):
        for method_name in ("mark_render_state_dirty", "reregister_component", "post_edit_change"):
            method = getattr(component, method_name, None)
            if not method:
                continue
            try:
                method()
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


def compile_actor_blueprint(actor):
    if not COMPILE_TRACK_BLUEPRINT_AFTER_SPLINE_UPDATE:
        return

    blueprint = actor_blueprint_asset(actor)
    if not blueprint:
        unreal.log_warning(
            f"Could not find Blueprint asset for '{TRACK_SPLINE_ACTOR_LABEL}'; "
            "skipping automatic compile."
        )
        return

    blueprint_editor_library = getattr(unreal, "BlueprintEditorLibrary", None)
    compile_blueprint = (
        getattr(blueprint_editor_library, "compile_blueprint", None)
        if blueprint_editor_library
        else None
    )
    if not compile_blueprint:
        kismet_editor_utilities = getattr(unreal, "KismetEditorUtilities", None)
        compile_blueprint = (
            getattr(kismet_editor_utilities, "compile_blueprint", None)
            if kismet_editor_utilities
            else None
        )

    if not compile_blueprint:
        unreal.log_warning("No Blueprint compile API is available in Unreal Python.")
        return

    compile_blueprint(blueprint)
    unreal.log(f"Compiled Blueprint for '{TRACK_SPLINE_ACTOR_LABEL}' after spline update.")


def actor_blueprint_asset(actor):
    actor_class = actor.get_class()

    blueprint_editor_library = getattr(unreal, "BlueprintEditorLibrary", None)
    get_blueprint = (
        getattr(blueprint_editor_library, "get_blueprint_from_class", None)
        if blueprint_editor_library
        else None
    )
    if get_blueprint:
        try:
            blueprint = get_blueprint(actor_class)
            if blueprint:
                return blueprint
        except Exception:
            pass

    class_path = str(actor_class.get_path_name())
    if class_path.endswith("_C"):
        asset_path = class_path[:-2]
        asset = unreal.load_asset(asset_path)
        if asset:
            return asset

    return None


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
