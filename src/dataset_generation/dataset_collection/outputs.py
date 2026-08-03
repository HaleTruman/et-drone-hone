"""Run folder and metadata output paths."""

import json
import os
from pathlib import Path

from src.dataset_generation.config import (
    DATASETS_DIRECTORY_NAME, DATASET_DIRECTORY_PREFIX,
    RUNS_DIRECTORY_NAME, RUN_DIRECTORY_PREFIX, FRAME_DIRECTORY_NAME, MASK_DIRECTORY_NAME,
    FRAME_FILE_EXTENSION, FRAME_METADATA_FILE_NAME, FRAME_METADATA_JSONL_FILE_NAME,
)

CURRENT_DATASET_DIR = None
CURRENT_RUN_DIR = None

def project_root():
    return str(Path(__file__).resolve().parents[3])

def datasets_root_dir():
    output_dir = os.path.join(project_root(), DATASETS_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_dataset_dir():
    global CURRENT_DATASET_DIR

    datasets_dir = datasets_root_dir()
    highest_index = 0
    for name in os.listdir(datasets_dir):
        if not name.startswith(DATASET_DIRECTORY_PREFIX):
            continue
        suffix = name[len(DATASET_DIRECTORY_PREFIX) :]
        if suffix.isdigit():
            highest_index = max(highest_index, int(suffix))

    dataset_dir = os.path.join(
        datasets_dir,
        f"{DATASET_DIRECTORY_PREFIX}{highest_index + 1:03d}",
    )
    os.makedirs(dataset_dir, exist_ok=False)
    os.makedirs(os.path.join(dataset_dir, RUNS_DIRECTORY_NAME), exist_ok=True)
    CURRENT_DATASET_DIR = dataset_dir
    return dataset_dir

def current_dataset_dir():
    if not CURRENT_DATASET_DIR:
        raise RuntimeError("Dataset directory has not been initialized.")
    return CURRENT_DATASET_DIR

def current_dataset_id():
    return os.path.basename(current_dataset_dir())

def runs_root_dir():
    output_dir = os.path.join(current_dataset_dir(), RUNS_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def dataset_camera_intrinsics_path():
    return os.path.join(current_dataset_dir(), "camera_intrinsics.json")

def create_run_dir():
    runs_dir = runs_root_dir()
    highest_index = 0
    for name in os.listdir(runs_dir):
        if not name.startswith(RUN_DIRECTORY_PREFIX):
            continue
        suffix = name[len(RUN_DIRECTORY_PREFIX) :]
        if suffix.isdigit():
            highest_index = max(highest_index, int(suffix))

    run_dir = os.path.join(runs_dir, f"{RUN_DIRECTORY_PREFIX}{highest_index + 1:03d}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, FRAME_DIRECTORY_NAME), exist_ok=True)
    os.makedirs(os.path.join(run_dir, MASK_DIRECTORY_NAME), exist_ok=True)
    return run_dir

def current_run_dir():
    if not CURRENT_RUN_DIR:
        raise RuntimeError("Run directory has not been initialized.")
    return CURRENT_RUN_DIR

def current_run_id():
    return os.path.basename(current_run_dir())

def relative_to_dataset(path):
    return os.path.relpath(path, current_dataset_dir()).replace(os.sep, "/")

def relative_to_run(path):
    return os.path.relpath(path, current_run_dir()).replace(os.sep, "/")

def frame_output_dir():
    output_dir = os.path.join(current_run_dir(), FRAME_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def frame_file_name(frame_number):
    return f"frame_{frame_number:06d}.{FRAME_FILE_EXTENSION}"

def frame_file_path(frame_number):
    return os.path.join(frame_output_dir(), frame_file_name(frame_number))

def frame_relative_path(frame_number):
    return relative_to_run(frame_file_path(frame_number))

def mask_output_dir():
    output_dir = os.path.join(current_run_dir(), MASK_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def mask_file_name(frame_number):
    return f"frame_{frame_number:06d}.{FRAME_FILE_EXTENSION}"

def mask_file_path(frame_number):
    return os.path.join(mask_output_dir(), mask_file_name(frame_number))

def mask_relative_path(frame_number):
    return relative_to_run(mask_file_path(frame_number))

def frame_metadata_path():
    return os.path.join(current_run_dir(), FRAME_METADATA_FILE_NAME)

def frame_metadata_jsonl_path():
    return os.path.join(current_run_dir(), FRAME_METADATA_JSONL_FILE_NAME)

def reset_frame_metadata():
    with open(frame_metadata_path(), "w", encoding="utf-8") as metadata_file:
        json.dump([], metadata_file, indent=2)
    with open(frame_metadata_jsonl_path(), "w", encoding="utf-8") as metadata_file:
        metadata_file.write("")
