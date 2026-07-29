"""Public runtime wrapper for dataset collection."""

from src.dataset_generation.dataset_collection import pipeline
from src.dataset_generation.dataset_collection.capture import (
    restore_scene_capture_every_frame_for_render_target,
)


def start(camera=None, gates=None, batch_number=1, on_complete=None):
    pipeline.start(
        camera=camera,
        gates=gates,
        batch_number=batch_number,
        on_complete=on_complete,
    )


def restore_capture_every_frame():
    return restore_scene_capture_every_frame_for_render_target()
