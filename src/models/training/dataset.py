"""PyTorch dataset and deterministic frame-level splitting."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional

from src.models.training.targets import GateFrame


class GateDetectionDataset(Dataset):
    def __init__(self, samples: Sequence[GateFrame], augment: bool = False) -> None:
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        horizontal_flip = self.augment and random.random() < 0.5
        if horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augment:
            image = transform_functional.adjust_brightness(
                image, random.uniform(0.8, 1.2)
            )
            image = transform_functional.adjust_contrast(
                image, random.uniform(0.8, 1.2)
            )
            image = transform_functional.adjust_saturation(
                image, random.uniform(0.85, 1.15)
            )
        image_tensor = transform_functional.pil_to_tensor(image).float() / 255.0

        if sample.gates:
            boxes = torch.from_numpy(
                np.stack([gate.bbox_xyxy for gate in sample.gates])
            ).float()
            keypoints = torch.from_numpy(
                np.stack(
                    [
                        np.column_stack(
                            (gate.outer_corners, gate.keypoint_visibility)
                        )
                        for gate in sample.gates
                    ]
                )
            ).float()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 4, 3), dtype=torch.float32)
        labels = torch.ones((len(sample.gates),), dtype=torch.int64)
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(sample.gates)
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(sample.gates),), dtype=torch.int64),
        }
        if horizontal_flip and len(sample.gates):
            flipped_left = sample.width - target["boxes"][:, 2].clone()
            flipped_right = sample.width - target["boxes"][:, 0].clone()
            target["boxes"][:, 0] = flipped_left
            target["boxes"][:, 2] = flipped_right
            target["keypoints"][:, :, 0] = (
                sample.width - 1 - target["keypoints"][:, :, 0]
            )
            target["keypoints"] = target["keypoints"][:, [1, 0, 3, 2], :]
        return image_tensor, target, index


# Import-compatible alias.
GatePoseDataset = GateDetectionDataset


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], int]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]], list[int]]:
    images, targets, indices = zip(*batch)
    return list(images), list(targets), list(indices)


def split_train_validation(
    samples: Sequence[GateFrame],
    validation_fraction: float,
    seed: int,
) -> tuple[list[GateFrame], list[GateFrame]]:
    if len(samples) < 2:
        raise ValueError("At least two frames are required for a train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_count])
    train = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    validation = [
        sample for index, sample in enumerate(samples) if index in validation_indices
    ]
    return train, validation


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
