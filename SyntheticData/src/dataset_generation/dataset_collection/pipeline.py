"""Dataset collection runtime for prepared gate layouts."""

import random

import unreal

from SyntheticData.src.dataset_generation.config import (
    CAMERA_POSITIONS_PER_GATE, CAMERA_HOLD_SECONDS, SAVE_FRAMES, SAVE_FRAME_METADATA, SAVE_MASKS,
    CAMERA_LABEL, RUN_COUNT,
    ZERO_GATE_FRAME_PERCENT, ZERO_GATE_CAMERA_POSITIONS,
    RENDER_TARGET_PATH, MASK_RENDER_TARGET_PATH,
)
from SyntheticData.src.dataset_generation.dataset_collection.camera_poses import (
    CameraCoverageState,
    continuous_spline_flight_poses,
    flight_camera_pose_for_gate_or_fallback,
    random_background_camera_pose,
    set_camera_pose,
)
from SyntheticData.src.dataset_generation.dataset_collection.capture import (
    load_render_target,
    load_mask_render_target,
    find_scene_capture_for_render_target,
    sync_scene_capture_to_camera,
    export_render_target_frame,
    export_render_target_mask,
)
from SyntheticData.src.dataset_generation.dataset_collection.metadata import (
    sanitize_gate_mask_file,
    write_dataset_camera_intrinsics,
    write_frame_metadata,
    write_mask_stencil_legend,
)
import SyntheticData.src.dataset_generation.dataset_collection.outputs as outputs
from SyntheticData.src.dataset_generation.dataset_collection.outputs import (
    create_run_dir,
    mask_file_path,
    reset_frame_metadata,
)
from SyntheticData.src.dataset_generation.unreal_editor import (
    require_unreal_editor_python, load_level_if_needed, invalidate_viewports, editor_world,
    find_camera_actor, set_actor_visible,
)

CAMERA_SEQUENCE_STATE = None
COMPLETION_CALLBACK = None

def start(camera=None, gates=None, track_layout=None, batch_number=1, on_complete=None):
    require_unreal_editor_python()
    load_level_if_needed()

    if gates is None:
        gates = []
    if camera is None:
        camera = find_camera_actor()

    if not camera:
        raise RuntimeError(f"No camera actor found with label '{CAMERA_LABEL}'.")

    start_camera_sequence(camera, gates, track_layout, batch_number, on_complete)

def zero_gate_pose_count_for_normal_run(normal_pose_count):
    if ZERO_GATE_FRAME_PERCENT <= 0.0:
        return 0
    if ZERO_GATE_FRAME_PERCENT >= 100.0:
        return max(1, normal_pose_count)

    zero_pose_count = normal_pose_count * (
        ZERO_GATE_FRAME_PERCENT / (100.0 - ZERO_GATE_FRAME_PERCENT)
    )
    return max(1, int(round(zero_pose_count)))

def make_pose(
    gate,
    camera_location,
    camera_rotation,
    look_target,
    yaw_offset,
    pitch_offset,
    show_gates,
):
    return {
        "gate": gate,
        "location": camera_location,
        "rotation": camera_rotation,
        "target": look_target,
        "yaw_offset": yaw_offset,
        "pitch_offset": pitch_offset,
        "show_gates": show_gates,
    }

def make_background_pose(camera):
    (
        camera_location,
        camera_rotation,
        look_target,
        yaw_offset,
        pitch_offset,
    ) = random_background_camera_pose(camera)
    return make_pose(
        None,
        camera_location,
        camera_rotation,
        look_target,
        yaw_offset,
        pitch_offset,
        False,
    )


def start_camera_sequence(camera, gates, track_layout, batch_number, on_complete=None):
    global CAMERA_SEQUENCE_STATE, COMPLETION_CALLBACK

    COMPLETION_CALLBACK = on_complete
    outputs.CURRENT_RUN_DIR = create_run_dir()
    render_target = load_render_target() if SAVE_FRAMES else None
    mask_render_target = load_mask_render_target() if SAVE_MASKS else None
    scene_capture_component = (
        find_scene_capture_for_render_target(render_target, RENDER_TARGET_PATH) if SAVE_FRAMES else None
    )
    mask_scene_capture_component = (
        find_scene_capture_for_render_target(mask_render_target, MASK_RENDER_TARGET_PATH)
        if SAVE_MASKS
        else None
    )
    world_context = (
        (editor_world() or camera)
        if (SAVE_FRAMES or SAVE_MASKS or SAVE_FRAME_METADATA)
        else None
    )
    if SAVE_FRAME_METADATA:
        write_dataset_camera_intrinsics(camera)
        reset_frame_metadata()
        write_mask_stencil_legend(gates)

    poses = []
    coverage_state = CameraCoverageState()
    if gates:
        flight_poses = continuous_spline_flight_poses(camera, gates, track_layout)
        if flight_poses:
            poses.extend(flight_poses)
            unreal.log(
                f"Using continuous spline flight capture with {len(poses)} frame(s)."
            )
        else:
            normal_pose_count = len(gates) * CAMERA_POSITIONS_PER_GATE
            if ZERO_GATE_FRAME_PERCENT < 100.0:
                for gate_index, gate in enumerate(gates):
                    for pose_index in range(CAMERA_POSITIONS_PER_GATE):
                        (
                            camera_location,
                            camera_rotation,
                            look_target,
                            yaw_offset,
                            pitch_offset,
                        ) = flight_camera_pose_for_gate_or_fallback(
                            camera,
                            gate,
                            track_layout,
                            gate_index,
                            pose_index,
                            CAMERA_POSITIONS_PER_GATE,
                            coverage_state,
                        )
                        poses.append(
                            make_pose(
                                gate,
                                camera_location,
                                camera_rotation,
                                look_target,
                                yaw_offset,
                                pitch_offset,
                                True,
                            )
                        )

            zero_gate_pose_count = zero_gate_pose_count_for_normal_run(normal_pose_count)
            for _ in range(zero_gate_pose_count):
                poses.append(make_background_pose(camera))

            if zero_gate_pose_count and not track_layout:
                random.shuffle(poses)
                unreal.log(
                    f"Added {zero_gate_pose_count} zero-gate frame(s) to this run batch "
                    f"for ZERO_GATE_FRAME_PERCENT={ZERO_GATE_FRAME_PERCENT:.1f}."
                )
            elif zero_gate_pose_count:
                unreal.log(
                    f"Appended {zero_gate_pose_count} zero-gate frame(s) after spline-flight "
                    f"poses for ZERO_GATE_FRAME_PERCENT={ZERO_GATE_FRAME_PERCENT:.1f}."
                )
            unreal.log(f"Camera coverage sampling summary: {coverage_state.compact_summary()}")
    else:
        for _ in range(ZERO_GATE_CAMERA_POSITIONS):
            poses.append(make_background_pose(camera))

    if not poses:
        finish_collection(batch_number)
        return

    unregister_camera_sequence()
    CAMERA_SEQUENCE_STATE = {
        "camera": camera,
        "gates": gates,
        "run_dir": outputs.CURRENT_RUN_DIR,
        "poses": poses,
        "index": 0,
        "elapsed": 0.0,
        "callback": None,
        "render_target": render_target,
        "mask_render_target": mask_render_target,
        "scene_capture_component": scene_capture_component,
        "mask_scene_capture_component": mask_scene_capture_component,
        "world_context": world_context,
        "all_gate_actors": gates,
        "frame_number": 1,
        "captured_frames": [],
        "save_pending": False,
        "batch_number": batch_number,
    }

    apply_camera_sequence_pose()

    register_tick = getattr(unreal, "register_slate_post_tick_callback", None)
    if not register_tick:
        unreal.log_warning(
            "unreal.register_slate_post_tick_callback is unavailable; "
            "only the first camera position was applied."
        )
        save_pending_frame()
        finalize_run_outputs(CAMERA_SEQUENCE_STATE)
        unregister_camera_sequence()
        finish_collection(batch_number)
        return

    CAMERA_SEQUENCE_STATE["callback"] = register_tick(camera_sequence_tick)

def camera_sequence_tick(delta_seconds):
    global CAMERA_SEQUENCE_STATE
    if not CAMERA_SEQUENCE_STATE:
        return

    CAMERA_SEQUENCE_STATE["elapsed"] += delta_seconds
    save_pending_frame()

    if CAMERA_SEQUENCE_STATE["elapsed"] < CAMERA_HOLD_SECONDS:
        return

    CAMERA_SEQUENCE_STATE["elapsed"] = 0.0
    CAMERA_SEQUENCE_STATE["index"] += 1

    if CAMERA_SEQUENCE_STATE["index"] >= len(CAMERA_SEQUENCE_STATE["poses"]):
        completed_batch = CAMERA_SEQUENCE_STATE.get("batch_number", 1)
        unreal.log(f"Camera sequence complete for run batch {completed_batch}/{RUN_COUNT}")
        finalize_run_outputs(CAMERA_SEQUENCE_STATE)
        for gate in CAMERA_SEQUENCE_STATE.get("all_gate_actors", []):
            set_actor_visible(gate, True)
        unregister_camera_sequence()
        finish_collection(completed_batch)
        return

    apply_camera_sequence_pose()

def apply_camera_sequence_pose():
    state = CAMERA_SEQUENCE_STATE
    pose = state["poses"][state["index"]]
    for gate in state["all_gate_actors"]:
        set_actor_visible(gate, bool(pose.get("show_gates", True)))
    set_camera_pose(
        state["camera"],
        pose["gate"],
        pose["location"],
        pose["rotation"],
        pose["target"],
        pose["yaw_offset"],
        pose["pitch_offset"],
    )
    invalidate_viewports()
    state["save_pending"] = SAVE_FRAMES or SAVE_MASKS or SAVE_FRAME_METADATA
    unreal.log(
        f"Camera sequence position {state['index'] + 1}/{len(state['poses'])}; "
        f"holding for {CAMERA_HOLD_SECONDS:.1f}s; "
        f"zero_gate_frame={pose.get('gate') is None}"
    )

def save_pending_frame():
    state = CAMERA_SEQUENCE_STATE
    if not state or not state.get("save_pending"):
        return

    pose = state["poses"][state["index"]]
    frame_saved = False
    mask_saved = False
    if SAVE_FRAMES:
        sync_scene_capture_to_camera(
            state["scene_capture_component"],
            state["camera"],
        )
        export_render_target_frame(
            state["world_context"],
            state["render_target"],
            state["frame_number"],
            pose,
        )
        frame_saved = True
    if SAVE_MASKS:
        sync_scene_capture_to_camera(
            state["mask_scene_capture_component"],
            state["camera"],
        )
        export_render_target_mask(
            state["world_context"],
            state["mask_render_target"],
            state["frame_number"],
            pose,
        )
        mask_saved = True

    state["captured_frames"].append(
        {
            "frame_number": state["frame_number"],
            "pose": pose,
            "frame_saved": frame_saved,
            "mask_saved": mask_saved,
        }
    )
    state["frame_number"] += 1
    state["save_pending"] = False


def finalize_run_outputs(state):
    if not state:
        return

    captured_frames = state.get("captured_frames", [])
    if not captured_frames:
        return

    if SAVE_MASKS:
        unreal.log(f"Cleaning {len(captured_frames)} captured mask PNG(s).")
        for frame in captured_frames:
            if not frame.get("mask_saved"):
                continue
            pose = frame["pose"]
            metadata_gates = state["gates"] if pose.get("show_gates", True) else []
            sanitize_gate_mask_file(
                mask_file_path(frame["frame_number"]),
                metadata_gates,
            )

    if SAVE_FRAME_METADATA:
        unreal.log(f"Writing metadata for {len(captured_frames)} captured frame(s).")
        for frame in captured_frames:
            pose = frame["pose"]
            apply_metadata_pose(state, pose)
            metadata_gates = state["gates"] if pose.get("show_gates", True) else []
            write_frame_metadata(
                state["world_context"],
                state["camera"],
                metadata_gates,
                frame["frame_number"],
                pose,
                frame.get("frame_saved", False),
                frame.get("mask_saved", False),
            )


def apply_metadata_pose(state, pose):
    for gate in state["all_gate_actors"]:
        set_actor_visible(gate, bool(pose.get("show_gates", True)))
    camera = state["camera"]
    camera.set_actor_location(pose["location"], False, False)
    camera.set_actor_rotation(pose["rotation"], False)

def unregister_camera_sequence():
    global CAMERA_SEQUENCE_STATE
    if not CAMERA_SEQUENCE_STATE:
        return

    callback = CAMERA_SEQUENCE_STATE.get("callback")
    unregister_tick = getattr(unreal, "unregister_slate_post_tick_callback", None)
    if callback is not None and unregister_tick:
        try:
            unregister_tick(callback)
        except Exception:
            pass

    CAMERA_SEQUENCE_STATE = None

def finish_collection(batch_number):
    global COMPLETION_CALLBACK

    callback = COMPLETION_CALLBACK
    COMPLETION_CALLBACK = None
    if callback:
        callback(batch_number)
