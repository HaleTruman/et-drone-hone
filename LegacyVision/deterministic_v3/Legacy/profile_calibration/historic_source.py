"""Read-only, local iterator over recorded live-ingress frame manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class HistoricFrameRecord:
    """One validated JPEG record from ``frames.jsonl``."""

    run_id: str
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    relative_path: str
    source_path: Path

    def pipeline_input(self) -> dict:
        jpeg_bytes = self.source_path.read_bytes()
        if len(jpeg_bytes) != self.jpeg_size:
            raise ValueError(
                f"JPEG size changed for {self.relative_path}: "
                f"{len(jpeg_bytes)} != {self.jpeg_size}")
        return {
            "frame_id": self.frame_id,
            "sim_time_ns": self.sim_time_ns,
            "jpeg_bytes": jpeg_bytes,
        }


class HistoricRunSource:
    """Yield ordered, path-confined records without importing another UI."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.manifest_path = self.run_dir / "frames.jsonl"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"historic frame manifest not found: {self.manifest_path}")

    def __iter__(self) -> Iterator[HistoricFrameRecord]:
        seen_ids: set[int] = set()
        previous_time = -1
        with self.manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    frame_id = int(record["frame_id"])
                    sim_time_ns = int(record["sim_time_ns"])
                    jpeg_size = int(record["jpeg_size"])
                    relative = Path(str(record["path"]))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid frame manifest record at line {line_number}"
                    ) from error
                if frame_id in seen_ids:
                    raise ValueError(f"duplicate frame_id in manifest: {frame_id}")
                if sim_time_ns < previous_time:
                    raise ValueError("frame manifest timing is not monotonic")
                if (relative.is_absolute() or not relative.parts or
                        relative.parts[0] != "vision_frames"):
                    raise ValueError(
                        f"frame path must be relative to vision_frames: {relative}")
                source_path = (self.run_dir / relative).resolve()
                if not source_path.is_relative_to(self.run_dir):
                    raise ValueError(f"frame path escapes run: {relative}")
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                seen_ids.add(frame_id)
                previous_time = sim_time_ns
                yield HistoricFrameRecord(
                    self.run_dir.name,
                    frame_id,
                    sim_time_ns,
                    jpeg_size,
                    relative.as_posix(),
                    source_path,
                )
