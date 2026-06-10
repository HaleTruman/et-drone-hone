"""JPEG decode and RGB normalization for the lightmask model."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def jpeg_bytes_to_rgb(jpeg_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    return image


def jpeg_bytes_to_tensor(jpeg_bytes: bytes) -> torch.Tensor:
    image = jpeg_bytes_to_rgb(jpeg_bytes)
    data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).unsqueeze(0)
    data = data.to(dtype=torch.float32).div(255.0)
    return (data - IMAGENET_MEAN) / IMAGENET_STD
