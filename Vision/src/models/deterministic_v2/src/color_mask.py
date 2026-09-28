"""RGB LUT color masking for projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # pragma: no cover
    from .config import ProjectionConfig
    from .schema import FrameMeta, sha256_file, utc_now
except ImportError:  # pragma: no cover
    from config import ProjectionConfig
    from schema import FrameMeta, sha256_file, utc_now


DECODER_CONTRACT = "PIL.Image.open(path).convert('RGB')"


def decode_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def rgb_keys(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB uint8 image")
    return (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )


def _metadata_from_npz(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    raw = value.item() if getattr(value, "shape", None) == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return {}
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


class ColorMasker:
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.lut_path = config.lut_path
        with np.load(self.lut_path, allow_pickle=False) as payload:
            if "lut" not in payload:
                raise ValueError(f"{self.lut_path} is missing array 'lut'")
            self.lut = payload["lut"].astype(np.uint8, copy=False)
            self.metadata = _metadata_from_npz(payload["metadata_json"] if "metadata_json" in payload else None)
        if self.lut.dtype != np.uint8:
            raise ValueError("color LUT must have dtype uint8")
        if self.lut.shape != (1 << 24,):
            raise ValueError(f"color LUT must have shape {(1 << 24,)}, got {self.lut.shape}")
        self.lut_sha256 = sha256_file(self.lut_path)

    def process(self, meta: FrameMeta) -> tuple[np.ndarray, dict[str, Any]]:
        rgb = decode_rgb(Path(meta.source_path))
        mask_bits = self.lut[rgb_keys(rgb)].astype(np.uint8, copy=False)
        height, width = mask_bits.shape
        layer_counts: dict[str, int] = {}
        for layer in self.config.mask_layers:
            bit = int(layer["bit"])
            layer_counts[str(layer["prefix"])] = int(np.count_nonzero(mask_bits & np.uint8(1 << bit)))
        values, counts = np.unique(mask_bits, return_counts=True)
        return mask_bits, {
            "schema": "projection-color-mask.v1",
            "run_id": meta.run_id,
            "frame_ordinal": meta.frame_ordinal,
            "frame_id": meta.frame_id,
            "source_path": meta.source_path,
            "image_width": width,
            "image_height": height,
            "created_at": utc_now(),
            "lut_path": str(self.lut_path),
            "lut_sha256": self.lut_sha256,
            "decoder": DECODER_CONTRACT,
            "nonzero_pixel_count": int(np.count_nonzero(mask_bits)),
            "layer_pixel_counts": layer_counts,
            "bitfield_counts": {str(int(value)): int(count) for value, count in zip(values, counts) if int(value)},
        }


def write_mask_binary(path: Path, mask_bits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    np.ascontiguousarray(mask_bits, dtype=np.uint8).tofile(temp)
    temp.replace(path)
