"""Ingress helpers for Regressor -> Surveyer JSON frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_regressor_frame(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Regressor frame JSON is not an object: {resolved}")
    if not isinstance(payload.get("run"), dict):
        raise ValueError(f"Regressor frame JSON is missing run object: {resolved}")
    if not isinstance(payload.get("gates"), list):
        raise ValueError(f"Regressor frame JSON is missing gates list: {resolved}")
    return payload


def read_regressor_jsonl(path: Path, *, max_frames: int = 0) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Regressor JSONL not found: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Regressor JSONL row is not an object: {resolved}")
            rows.append(payload)
            if int(max_frames) > 0 and len(rows) >= int(max_frames):
                break
    return rows


def iter_regressor_frame_files(input_dir: Path) -> list[Path]:
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Regressor JSON directory not found: {directory}")
    return sorted(path for path in directory.glob("frame_*.json") if path.is_file())


def read_regressor_frames(
    *,
    input_jsonl: Path | None = None,
    input_dir: Path | None = None,
    max_frames: int = 0,
) -> list[dict[str, Any]]:
    if input_jsonl is None and input_dir is None:
        raise ValueError("Either input_jsonl or input_dir is required.")
    if input_jsonl is not None:
        return read_regressor_jsonl(input_jsonl, max_frames=max_frames)

    files = iter_regressor_frame_files(Path(input_dir))  # type: ignore[arg-type]
    if int(max_frames) > 0:
        files = files[: int(max_frames)]
    return [read_regressor_frame(path) for path in files]


def iter_regressor_frames(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(payloads)


__all__ = [
    "iter_regressor_frame_files",
    "iter_regressor_frames",
    "read_regressor_frame",
    "read_regressor_frames",
    "read_regressor_jsonl",
]
