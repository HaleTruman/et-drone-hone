from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from schema import DEFAULT_LUT_PATH, MaskFrame, mask_count


def load_lut(path: str | Path = DEFAULT_LUT_PATH) -> np.ndarray:
    lut = np.load(path)["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut


def _apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    bgr = image.astype(np.uint32)
    rgb_key = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return lut[rgb_key] != 0


def image_to_mask(image: np.ndarray, frame_label: str, lut_path: str | Path = DEFAULT_LUT_PATH) -> MaskFrame:
    base_mask = _apply_lut(image, load_lut(lut_path))
    height, width = base_mask.shape
    return MaskFrame(frame_path=frame_label, width=int(width), height=int(height), base_mask=base_mask)


def jpeg_bytes_to_mask(jpeg_bytes: bytes, frame_label: str, lut_path: str | Path = DEFAULT_LUT_PATH) -> MaskFrame:
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode jpeg bytes for frame: {frame_label}")
    return image_to_mask(image, frame_label, lut_path)


def frame_to_mask(frame_path: str | Path, lut_path: str | Path = DEFAULT_LUT_PATH) -> MaskFrame:
    frame_path = str(frame_path)
    image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read frame image: {frame_path}")
    return image_to_mask(image, frame_path, lut_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one frame to a LUT-derived mask.")
    parser.add_argument("frame")
    parser.add_argument("--lut", default=DEFAULT_LUT_PATH)
    args = parser.parse_args()
    result = frame_to_mask(args.frame, args.lut)
    print(json.dumps({
        "frame_path": result.frame_path,
        "width": result.width,
        "height": result.height,
        "base_pixels": mask_count(result.base_mask),
    }, indent=2))


if __name__ == "__main__":
    main()
