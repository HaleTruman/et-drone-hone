"""Public runtime wrapper for dataset collection."""

from src.dataset_generation.dataset_collection import pipeline


def start(camera=None, gates=None, batch_number=1, on_complete=None):
    pipeline.start(
        camera=camera,
        gates=gates,
        batch_number=batch_number,
        on_complete=on_complete,
    )
