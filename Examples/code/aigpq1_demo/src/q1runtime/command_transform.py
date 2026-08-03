from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CommandFrameTransform:
    """Transforms desired local-NED velocity into the simulator command frame."""

    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    name: str = "identity"

    @classmethod
    def identity(cls) -> "CommandFrameTransform":
        return cls(matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    @classmethod
    def from_options(
        cls,
        *,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
        swap_xy: bool = False,
        name: str = "manual",
    ) -> "CommandFrameTransform":
        matrix = np.eye(3, dtype=float)
        if swap_xy:
            matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float) @ matrix
        signs = np.diag(
            [
                -1.0 if invert_x else 1.0,
                -1.0 if invert_y else 1.0,
                -1.0 if invert_z else 1.0,
            ]
        )
        matrix = signs @ matrix
        return cls.from_matrix(matrix, name=name)

    @classmethod
    def from_matrix(cls, matrix: Any, *, name: str = "matrix") -> "CommandFrameTransform":
        array = np.asarray(matrix, dtype=float)
        if array.shape != (3, 3):
            raise ValueError("command transform matrix must be 3x3")
        return cls(
            matrix=tuple(tuple(float(value) for value in row) for row in array),
            name=name,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CommandFrameTransform":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "recommended_transform" in payload:
            payload = payload["recommended_transform"]
        matrix = payload.get("matrix")
        if matrix is None:
            raise ValueError("transform JSON must contain matrix or recommended_transform.matrix")
        return cls.from_matrix(matrix, name=str(payload.get("name", Path(path).stem)))

    def apply(self, velocity_local_ned_mps: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float) @ np.asarray(velocity_local_ned_mps, dtype=float)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matrix": [[float(value) for value in row] for row in self.matrix],
        }
