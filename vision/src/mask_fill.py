from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from mask import frame_to_mask
from schema import DEFAULT_FILL_RADIUS_PX, FillFrame, MaskFrame, mask_count

CLOSE_KERNEL_PX = 3
OPEN_KERNEL_PX = 3


def clean_close_open(mask: np.ndarray, close_kernel_px: int = CLOSE_KERNEL_PX,
                     open_kernel_px: int = OPEN_KERNEL_PX) -> np.ndarray:
    close_size = max(1, int(close_kernel_px))
    open_size = max(1, int(open_kernel_px))
    close_kernel = np.ones((close_size, close_size), dtype=np.uint8)
    open_kernel = np.ones((open_size, open_size), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return opened.astype(bool)


def fill_close_pixels(mask_frame: MaskFrame, fill_radius_px: float = DEFAULT_FILL_RADIUS_PX) -> FillFrame:
    radius = max(0.0, float(fill_radius_px))
    source = mask_frame.base_mask.astype(np.uint8)
    if radius == 0:
        filled = source.astype(bool)
    else:
        size = max(1, int(round(radius * 2 + 1)))
        kernel = np.ones((size, size), dtype=np.uint8)
        filled = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel).astype(bool)
    cleaned = clean_close_open(filled, CLOSE_KERNEL_PX, OPEN_KERNEL_PX)
    yellow = cleaned & ~mask_frame.base_mask
    return FillFrame(frame=mask_frame, yellow_mask=yellow, filled_mask=cleaned, fill_radius_px=radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview close-gap yellow fill for one frame.")
    parser.add_argument("frame")
    parser.add_argument("--lut", default="assets/color_lut_v1.npz")
    parser.add_argument("--radius", type=float, default=DEFAULT_FILL_RADIUS_PX)
    args = parser.parse_args()
    result = fill_close_pixels(frame_to_mask(args.frame, args.lut), args.radius)
    print(json.dumps({
        "frame_path": result.frame.frame_path,
        "width": result.frame.width,
        "height": result.frame.height,
        "base_pixels": mask_count(result.frame.base_mask),
        "yellow_pixels": mask_count(result.yellow_mask),
        "filled_pixels": mask_count(result.filled_mask),
    }, indent=2))


if __name__ == "__main__":
    main()
