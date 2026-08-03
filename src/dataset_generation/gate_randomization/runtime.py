"""Gate randomization runtime for dataset generation."""

from importlib import reload

import unreal

from src.dataset_generation.config import (
    ALLOW_ZERO_GATES, CAMERA_LABEL, CAMERA_POSITIONS_PER_GATE, CREATE_TRACK_SPLINE_ACTOR,
    ENABLE_GATE_CUSTOM_DEPTH_STENCIL, FORCE_ZERO_GATE_RUN, GATE_FOLDER,
    GATE_CUSTOM_DEPTH_STENCIL_BANDS, GATE_CUSTOM_DEPTH_STENCIL_BASE_VALUE,
    GATE_CUSTOM_DEPTH_STENCIL_VALUE,
    MIN_GATE_DISTANCE_CM, TRACK_RENDER_MAX_WAIT_SECONDS, TRACK_RENDER_READY_BOOL_PROPERTY,
    TRACK_RENDER_SETTLE_SECONDS, TRACK_RENDER_STABLE_TICKS,
    USE_PER_GATE_CUSTOM_DEPTH_STENCIL_VALUES, ZERO_GATE_CAMERA_POSITIONS,
)
from src.dataset_generation.gate_randomization import spline_actor
from src.dataset_generation.gate_randomization.track import closest_gate_spacing, generate_track_layout
from src.dataset_generation.unreal_editor import (
    actor_sort_key, all_level_actors, find_camera_actor, find_gate_actors, invalidate_viewports,
    load_level_if_needed, require_unreal_editor_python, set_actor_visible,
)


reload(spline_actor)
PENDING_RANDOMIZATION_STATE = None


def start(batch_number=1, on_complete=None):
    require_unreal_editor_python()
    load_level_if_needed()

    actors = find_gate_actors()
    camera = find_camera_actor()

    if not camera:
        raise RuntimeError(f"No camera actor found with label '{CAMERA_LABEL}'.")

    if FORCE_ZERO_GATE_RUN:
        with unreal.ScopedEditorTransaction("Prepare Zero-Gate Dataset Run"):
            for actor in actors:
                set_actor_visible(actor, False)
        unreal.log_warning(
            f"FORCE_ZERO_GATE_RUN=True. Hidden {len(actors)} gate actor(s) and "
            f"prepared {ZERO_GATE_CAMERA_POSITIONS} zero-gate background frame(s)."
        )
        invalidate_viewports()
        complete_randomization(camera, [], batch_number, on_complete, None)
        return

    if not actors and not ALLOW_ZERO_GATES:
        raise RuntimeError(
            f"No actors found in an outliner folder named '{GATE_FOLDER}'. "
            "If you just enabled EditorScriptingUtilities, restart Unreal and run the script again."
        )
    if not actors:
        unreal.log_warning(
            f"No actors found in folder '{GATE_FOLDER}'. "
            f"Prepared {ZERO_GATE_CAMERA_POSITIONS} zero-gate background frame(s)."
        )
        invalidate_viewports()
        complete_randomization(camera, [], batch_number, on_complete, None)
        return

    active_gates = sorted(actors, key=actor_sort_key)
    track_layout = generate_track_layout(len(active_gates))
    active_locations = track_layout["locations"]
    active_rotations = track_layout["rotations"]
    closest_spacing = closest_gate_spacing(active_locations)

    randomized_gates = []
    track_actor = None
    with unreal.ScopedEditorTransaction("Randomize Gates And Track"):
        disable_non_gate_custom_depth_stencil(actors)

        if CREATE_TRACK_SPLINE_ACTOR:
            track_actor = spline_actor.update_track_spline_actor(track_layout["spline_points"])

        for actor in actors:
            set_actor_visible(actor, True)

        for gate_index, (actor, location, rotation) in enumerate(
            zip(active_gates, active_locations, active_rotations)
        ):
            actor.modify()
            actor.set_actor_location(location, False, False)
            actor.set_actor_rotation(rotation, False)
            enable_gate_custom_depth_stencil(
                actor,
                gate_stencil_value(gate_index),
            )
            randomized_gates.append(actor)
            unreal.log(
                "Placed track gate actor "
                f"{actor.get_actor_label()} to location={actor.get_actor_location()} "
                f"rotation={actor.get_actor_rotation()}"
            )

    invalidate_viewports()
    if closest_spacing is not None and closest_spacing < MIN_GATE_DISTANCE_CM:
        unreal.log_warning(
            f"Closest generated gate spacing is {closest_spacing:.1f} cm, below "
            f"MIN_GATE_DISTANCE_CM={MIN_GATE_DISTANCE_CM}. All gates were used as requested; "
            "increase the placement bounds or reduce gate count if spacing is mandatory."
        )

    unreal.log(
        f"Placed {len(randomized_gates)} gate actor(s) from folder '{GATE_FOLDER}' "
        "along a generated spline-style race track "
        f"for {CAMERA_POSITIONS_PER_GATE} camera position(s) per gate."
    )
    wait_for_track_settle(
        camera,
        randomized_gates,
        batch_number,
        track_actor,
        on_complete,
        track_layout,
    )


def wait_for_track_settle(
    camera,
    gates,
    batch_number,
    track_actor=None,
    on_complete=None,
    track_layout=None,
):
    global PENDING_RANDOMIZATION_STATE

    settle_seconds = max(0.0, float(TRACK_RENDER_SETTLE_SECONDS))
    max_wait_seconds = max(settle_seconds, float(TRACK_RENDER_MAX_WAIT_SECONDS))
    if settle_seconds <= 0.0 and not track_actor:
        complete_randomization(camera, gates, batch_number, on_complete, track_layout)
        return

    register_tick = getattr(unreal, "register_slate_post_tick_callback", None)
    if not register_tick:
        unreal.log_warning(
            "unreal.register_slate_post_tick_callback is unavailable; "
            "continuing without a track render settle delay."
        )
        complete_randomization(camera, gates, batch_number, on_complete, track_layout)
        return

    unregister_pending_randomization()
    PENDING_RANDOMIZATION_STATE = {
        "camera": camera,
        "gates": gates,
        "batch_number": batch_number,
        "elapsed": 0.0,
        "settle_seconds": settle_seconds,
        "max_wait_seconds": max_wait_seconds,
        "track_actor": track_actor,
        "track_layout": track_layout,
        "last_component_count": component_count(track_actor),
        "stable_ticks": 0,
        "on_complete": on_complete,
        "callback": None,
    }
    PENDING_RANDOMIZATION_STATE["callback"] = register_tick(track_render_settle_tick)
    unreal.log("Waiting for track spline construction/rendering before capture.")


def track_render_settle_tick(delta_seconds):
    state = PENDING_RANDOMIZATION_STATE
    if not state:
        return

    state["elapsed"] += delta_seconds
    if not track_render_wait_complete(state):
        return

    camera = state["camera"]
    gates = state["gates"]
    batch_number = state["batch_number"]
    on_complete = state["on_complete"]
    track_layout = state["track_layout"]
    unregister_pending_randomization()
    unreal.log("Track render settle complete.")
    complete_randomization(camera, gates, batch_number, on_complete, track_layout)


def track_render_wait_complete(state):
    if state["elapsed"] < state["settle_seconds"]:
        return False

    track_actor = state.get("track_actor")
    if track_actor and track_ready_flag_is_true(track_actor):
        unreal.log(
            f"Track ready flag '{TRACK_RENDER_READY_BOOL_PROPERTY}' is true."
        )
        return True

    if track_actor and TRACK_RENDER_READY_BOOL_PROPERTY:
        ready_value = get_optional_editor_property(track_actor, TRACK_RENDER_READY_BOOL_PROPERTY)
        if ready_value is False:
            return timed_out_or_waiting(state)

    if track_actor and component_count_is_stable(state):
        return True

    return timed_out_or_waiting(state)


def timed_out_or_waiting(state):
    if state["elapsed"] < state["max_wait_seconds"]:
        return False
    unreal.log_warning(
        f"Track render wait timed out after {state['elapsed']:.2f}s; continuing anyway."
    )
    return True


def track_ready_flag_is_true(actor):
    if not TRACK_RENDER_READY_BOOL_PROPERTY:
        return False
    value = get_optional_editor_property(actor, TRACK_RENDER_READY_BOOL_PROPERTY)
    return value is True


def get_optional_editor_property(obj, property_name):
    try:
        return obj.get_editor_property(property_name)
    except Exception:
        return None


def component_count_is_stable(state):
    track_actor = state.get("track_actor")
    current_count = component_count(track_actor)
    if current_count != state.get("last_component_count"):
        state["last_component_count"] = current_count
        state["stable_ticks"] = 0
        return False

    state["stable_ticks"] += 1
    return state["stable_ticks"] >= max(1, int(TRACK_RENDER_STABLE_TICKS))


def component_count(actor):
    if not actor:
        return 0

    component_class = getattr(unreal, "ActorComponent", None)
    if not component_class:
        return 0
    try:
        return len(actor.get_components_by_class(component_class))
    except Exception:
        return 0


def gate_stencil_value(gate_index):
    if USE_PER_GATE_CUSTOM_DEPTH_STENCIL_VALUES:
        gate_index = int(gate_index)
        if gate_index < len(GATE_CUSTOM_DEPTH_STENCIL_BANDS):
            return min(255, max(1, int(GATE_CUSTOM_DEPTH_STENCIL_BANDS[gate_index])))
        return min(255, max(1, int(GATE_CUSTOM_DEPTH_STENCIL_BASE_VALUE) + gate_index))
    return min(255, max(0, int(GATE_CUSTOM_DEPTH_STENCIL_VALUE)))


def enable_gate_custom_depth_stencil(actor, stencil_value):
    if not ENABLE_GATE_CUSTOM_DEPTH_STENCIL:
        return

    components = gate_render_components(actor)
    configured_count = 0
    for component in components:
        component.modify()
        render_enabled = set_optional_editor_property(component, "render_custom_depth", True)
        stencil_set = set_optional_editor_property(
            component,
            "custom_depth_stencil_value",
            int(stencil_value),
        )
        if not render_enabled:
            render_enabled = call_optional_method(component, "set_render_custom_depth", True)
        if not stencil_set:
            stencil_set = call_optional_method(
                component,
                "set_custom_depth_stencil_value",
                int(stencil_value),
            )
        if render_enabled or stencil_set:
            configured_count += 1

    unreal.log(
        f"Enabled CustomDepth stencil={int(stencil_value)} on "
        f"{configured_count}/{len(components)} render component(s) "
        f"for gate {actor.get_actor_label()}."
    )


def gate_render_components(actor):
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
            for component in actor.get_components_by_class(component_class):
                key = str(component)
                if key not in seen:
                    seen.add(key)
                    components.append(component)
        except Exception:
            pass
    return components


def disable_non_gate_custom_depth_stencil(gate_actors):
    if not ENABLE_GATE_CUSTOM_DEPTH_STENCIL:
        return

    gate_actor_keys = {object_identity_key(actor) for actor in gate_actors}
    disabled_count = 0
    component_count = 0
    for actor in all_level_actors():
        if object_identity_key(actor) in gate_actor_keys:
            continue
        for component in gate_render_components(actor):
            component_count += 1
            component.modify()
            disabled = set_optional_editor_property(component, "render_custom_depth", False)
            disabled = call_optional_method(component, "set_render_custom_depth", False) or disabled
            if disabled:
                disabled_count += 1

    unreal.log(
        f"Disabled CustomDepth on {disabled_count}/{component_count} non-gate render component(s) "
        "before assigning gate stencil IDs."
    )


def object_identity_key(obj):
    get_path_name = getattr(obj, "get_path_name", None)
    if get_path_name:
        try:
            return str(get_path_name())
        except Exception:
            pass
    return str(obj)


def set_optional_editor_property(obj, property_name, value):
    try:
        obj.set_editor_property(property_name, value)
        return True
    except Exception:
        return False


def call_optional_method(obj, method_name, *args):
    method = getattr(obj, method_name, None)
    if not method:
        return False
    try:
        method(*args)
        return True
    except Exception:
        return False


def unregister_pending_randomization():
    global PENDING_RANDOMIZATION_STATE

    if not PENDING_RANDOMIZATION_STATE:
        return

    callback = PENDING_RANDOMIZATION_STATE.get("callback")
    unregister_tick = getattr(unreal, "unregister_slate_post_tick_callback", None)
    if callback is not None and unregister_tick:
        try:
            unregister_tick(callback)
        except Exception:
            pass

    PENDING_RANDOMIZATION_STATE = None


def complete_randomization(camera, gates, batch_number, on_complete, track_layout=None):
    if on_complete:
        on_complete(camera, gates, batch_number, track_layout)
