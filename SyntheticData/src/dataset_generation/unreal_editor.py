"""Unreal Editor environment and actor lookup helpers."""

import unreal

from SyntheticData.src.dataset_generation.config import GATE_FOLDER, LEVEL_PATH, CAMERA_LABEL

def require_unreal_editor_python():
    module_file = str(getattr(unreal, "__file__", ""))
    if "site-packages" in module_file.replace("\\", "/"):
        raise RuntimeError(
            "Run this script from inside Unreal Editor, not from a project venv. "
            f"It imported unreal from: {module_file}. "
        )

def actor_folder(actor):
    for attr in ("get_folder_path", "get_folder_path_"):
        getter = getattr(actor, attr, None)
        if getter:
            try:
                return str(getter())
            except Exception:
                pass

    try:
        return str(actor.get_editor_property("folder_path"))
    except Exception:
        return ""

def is_in_gate_folder(actor):
    folder = actor_folder(actor).replace("\\", "/").strip("/")
    return (
        folder == GATE_FOLDER
        or folder.startswith(f"{GATE_FOLDER}/")
        or folder.endswith(f"/{GATE_FOLDER}")
        or f"/{GATE_FOLDER}/" in folder
    )

def actor_sort_key(actor):
    try:
        return actor.get_actor_label()
    except Exception:
        return str(actor.get_path_name())

def find_gate_actors():
    actors = all_level_actors()
    return [actor for actor in actors if is_in_gate_folder(actor)]

def find_camera_actor():
    for actor in all_level_actors():
        try:
            if actor.get_actor_label() == CAMERA_LABEL:
                return actor
        except Exception:
            pass
    return None

def all_level_actors():
    actor_subsystem = get_editor_subsystem("EditorActorSubsystem")
    if actor_subsystem:
        return actor_subsystem.get_all_level_actors()

    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if editor_level_library:
        return editor_level_library.get_all_level_actors()

    if not editor_level_library:
        raise RuntimeError(
            "Neither EditorActorSubsystem nor EditorLevelLibrary is available. "
            "Run this from Unreal Editor's Python environment with editor scripting enabled."
        )

def load_level_if_needed():
    current_level = ""

    unreal_editor_subsystem = get_editor_subsystem("UnrealEditorSubsystem")
    if unreal_editor_subsystem:
        try:
            current_world = unreal_editor_subsystem.get_editor_world()
            current_level = current_world.get_outermost().get_name() if current_world else ""
        except Exception:
            current_level = ""

    level_subsystem = get_editor_subsystem("LevelEditorSubsystem")
    if current_level == LEVEL_PATH:
        return

    if level_subsystem:
        level_subsystem.load_level(LEVEL_PATH)
        return

    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if not editor_level_library:
        return
    try:
        current_world = editor_level_library.get_editor_world()
        current_level = current_world.get_outermost().get_name() if current_world else ""
    except Exception:
        current_level = ""

    if current_level != LEVEL_PATH:
        editor_level_library.load_level(LEVEL_PATH)

def editor_world():
    unreal_editor_subsystem = get_editor_subsystem("UnrealEditorSubsystem")
    if unreal_editor_subsystem:
        try:
            return unreal_editor_subsystem.get_editor_world()
        except Exception:
            pass

    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if editor_level_library:
        try:
            return editor_level_library.get_editor_world()
        except Exception:
            pass

    return None

def invalidate_viewports():
    level_subsystem = get_editor_subsystem("LevelEditorSubsystem")
    if level_subsystem and hasattr(level_subsystem, "editor_invalidate_viewports"):
        level_subsystem.editor_invalidate_viewports()
        return

    editor_level_library = getattr(unreal, "EditorLevelLibrary", None)
    if not editor_level_library:
        return
    invalidate = getattr(editor_level_library, "editor_invalidate_viewports", None)
    if invalidate:
        invalidate()

def get_editor_subsystem(class_name):
    get_subsystem = getattr(unreal, "get_editor_subsystem", None)
    subsystem_class = getattr(unreal, class_name, None)
    if not get_subsystem or not subsystem_class:
        return None
    try:
        return get_subsystem(subsystem_class)
    except Exception:
        return None

def set_actor_visible(actor, visible):
    actor.modify()
    actor.set_actor_hidden_in_game(not visible)

    set_hidden_editor = getattr(actor, "set_is_temporarily_hidden_in_editor", None)
    if set_hidden_editor:
        set_hidden_editor(not visible)

    set_collision = getattr(actor, "set_actor_enable_collision", None)
    if set_collision:
        set_collision(visible)
