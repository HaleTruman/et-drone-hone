"""SceneCapture2D synchronization and render target export."""

import unreal

from src.dataset_generation.common import object_path_name, actor_label, component_owner, component_name
from src.dataset_generation.config import RENDER_TARGET_PATH, FRAME_FILE_EXTENSION, SCENE_CAPTURE_WARMUP_CAPTURES
from src.dataset_generation.dataset_collection.camera_poses import camera_horizontal_fov_and_aspect
from src.dataset_generation.dataset_collection.outputs import frame_output_dir, frame_file_name
from src.dataset_generation.unreal_editor import all_level_actors

def load_render_target():
    render_target = unreal.load_asset(RENDER_TARGET_PATH)
    if not render_target:
        raise RuntimeError(f"Could not load render target '{RENDER_TARGET_PATH}'.")
    prepare_render_target_for_png(render_target)
    return render_target

def scene_capture_component_target(component):
    for property_name in ("texture_target", "TextureTarget"):
        try:
            return component.get_editor_property(property_name)
        except Exception:
            pass
    return None

def actor_scene_capture_components(actor):
    components = []
    scene_capture_class = getattr(unreal, "SceneCaptureComponent2D", None)
    if scene_capture_class:
        try:
            components.extend(actor.get_components_by_class(scene_capture_class))
        except Exception:
            pass

    for property_name in ("capture_component2d", "CaptureComponent2D"):
        try:
            component = actor.get_editor_property(property_name)
            if component:
                components.append(component)
        except Exception:
            pass

    unique = []
    seen = set()
    for component in components:
        key = object_path_name(component)
        if key not in seen:
            seen.add(key)
            unique.append(component)
    return unique

def find_scene_capture_for_render_target(render_target):
    render_target_path = object_path_name(render_target)

    for actor in all_level_actors():
        for component in actor_scene_capture_components(actor):
            target = scene_capture_component_target(component)
            if object_path_name(target) == render_target_path:
                unreal.log(
                    "Using SceneCaptureComponent2D "
                    f"{component_name(component)} on actor {actor_label(actor)} "
                    f"for render target {RENDER_TARGET_PATH}"
                )
                configure_scene_capture_for_manual_capture(component)
                return component

    raise RuntimeError(
        f"SAVE_FRAMES=True, but no SceneCaptureComponent2D in the level uses "
        f"render target '{RENDER_TARGET_PATH}'. Assign this render target to a "
        "SceneCaptureComponent2D, then rerun the script."
    )

def set_editor_property_if_present(obj, property_name, value):
    try:
        obj.get_editor_property(property_name)
    except Exception:
        return False

    try:
        obj.set_editor_property(property_name, value)
        return True
    except Exception:
        return False

def configure_scene_capture_for_manual_capture(scene_capture_component):
    set_editor_property_if_present(scene_capture_component, "capture_every_frame", False)
    set_editor_property_if_present(scene_capture_component, "capture_on_movement", False)

    persist_set = False
    for property_name in (
        "always_persist_rendering_state",
        "b_always_persist_rendering_state",
    ):
        persist_set = (
            set_editor_property_if_present(scene_capture_component, property_name, True)
            or persist_set
        )

    if persist_set:
        unreal.log(
            "Configured SceneCaptureComponent2D for manual capture with persistent rendering state."
        )
    else:
        unreal.log_warning(
            "Could not set Always Persist Rendering State on SceneCaptureComponent2D. "
            "If frames are dark with Capture Every Frame disabled, enable it manually on the component."
        )

def set_component_world_transform(component, location, rotation):
    setter = getattr(component, "set_world_location_and_rotation", None)
    if setter:
        try:
            setter(location, rotation, False, None, False)
            return
        except TypeError:
            try:
                setter(location, rotation, False)
                return
            except Exception:
                pass
        except Exception:
            pass

    k2_setter = getattr(component, "k2_set_world_location_and_rotation", None)
    if k2_setter:
        try:
            k2_setter(location, rotation, False, None, False)
            return
        except TypeError:
            try:
                k2_setter(location, rotation, False, None)
                return
            except Exception:
                pass
        except Exception:
            pass

    owner = component_owner(component)
    if owner:
        owner.set_actor_location(location, False, False)
        owner.set_actor_rotation(rotation, False)
        return

    raise RuntimeError("Could not move SceneCaptureComponent2D to the camera transform.")

def sync_scene_capture_to_camera(scene_capture_component, camera):
    if not scene_capture_component:
        return

    set_component_world_transform(
        scene_capture_component,
        camera.get_actor_location(),
        camera.get_actor_rotation(),
    )

    horizontal_fov, _aspect_ratio = camera_horizontal_fov_and_aspect(camera)
    try:
        scene_capture_component.set_editor_property("fov_angle", horizontal_fov)
    except Exception:
        pass

    capture_scene = getattr(scene_capture_component, "capture_scene", None)
    if not capture_scene:
        raise RuntimeError("SceneCaptureComponent2D does not expose capture_scene().")

    for _ in range(max(0, SCENE_CAPTURE_WARMUP_CAPTURES)):
        capture_scene()
    capture_scene()

def prepare_render_target_for_png(render_target):
    if FRAME_FILE_EXTENSION.lower() != "png":
        return

    texture_render_target_format = getattr(unreal, "TextureRenderTargetFormat", None)
    if not texture_render_target_format:
        unreal.log_warning(
            "Could not inspect unreal.TextureRenderTargetFormat. "
            "If exported PNG files do not open, set the render target format to RTF_RGBA8."
        )
        return

    png_format = None
    for enum_name in ("RTF_RGBA8", "RTF_RGBA8_SRGB"):
        candidate = getattr(texture_render_target_format, enum_name, None)
        if candidate is not None:
            png_format = candidate
            break

    if png_format is None:
        unreal.log_warning(
            "Could not find an 8-bit RGBA render target format. "
            "If exported PNG files do not open, set the render target format to RTF_RGBA8."
        )
        return

    try:
        current_format = render_target.get_editor_property("render_target_format")
    except Exception:
        current_format = None

    if current_format == png_format:
        return

    render_target.modify()
    render_target.set_editor_property("render_target_format", png_format)

    update_resource = getattr(render_target, "update_resource_immediate", None)
    if update_resource:
        try:
            update_resource(True)
        except TypeError:
            update_resource()

    unreal.log(
        f"Set render target '{RENDER_TARGET_PATH}' format to {png_format} "
        "so frame exports are real PNG files."
    )

def export_render_target_frame(world_context, render_target, frame_number, pose):
    rendering_library = getattr(unreal, "RenderingLibrary", None)
    if not rendering_library or not hasattr(rendering_library, "export_render_target"):
        raise RuntimeError(
            "unreal.RenderingLibrary.export_render_target is unavailable. "
            "Enable the Python/Editor scripting support that exposes Kismet Rendering Library."
        )

    output_dir = frame_output_dir()
    file_name = frame_file_name(frame_number)
    rendering_library.export_render_target(
        world_context,
        render_target,
        output_dir,
        file_name,
    )
    gate_label = pose["gate"].get_actor_label() if pose.get("gate") else None
    unreal.log(
        f"Saved render target frame {file_name} to {output_dir} "
        f"for gate={gate_label}"
    )
