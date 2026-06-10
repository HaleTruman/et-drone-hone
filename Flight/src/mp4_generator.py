"""Build an MP4 playback video from saved vision frame logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

NS_PER_SECOND = 1_000_000_000
DEFAULT_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs" / "runs"
FRAME_NAME_RE = re.compile(r"frame-(?P<frame_id>\d+)-(?P<sim_time_ns>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# Change this to the run folder name you want to render.
RUN = "run-20260610T164651Z"
DEFAULT_FPS = 30.0
MAX_GAP_S = 1.0


@dataclass(frozen=True)
class LoggedFrame:
    path: Path
    frame_id: int | None
    sim_time_ns: int | None


def main() -> None:
    log_path = _resolve_log_path(RUN)
    frames = _load_frames(log_path)
    if not frames:
        raise SystemExit(f"No frames found under {log_path}")

    output_path = _resolve_output_path(log_path)
    fps = _infer_fps(frames, DEFAULT_FPS)
    written_count = write_video(frames, output_path, fps=fps, max_gap_s=MAX_GAP_S)

    first_time = frames[0].sim_time_ns
    last_time = frames[-1].sim_time_ns
    duration_s = (
        (last_time - first_time) / NS_PER_SECOND
        if first_time is not None and last_time is not None and last_time >= first_time
        else written_count / fps
    )
    print(f"Wrote {output_path}")
    print(f"Input frames: {len(frames)}")
    print(f"Video frames: {written_count}")
    print(f"FPS: {fps:.3f}")
    print(f"Approx duration: {duration_s:.3f}s")


def write_video(frames: list[LoggedFrame], output_path: Path, fps: float, max_gap_s: float | None = 1.0) -> int:
    cv2 = _load_cv2()
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    first_image = cv2.imread(str(frames[0].path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Could not read first frame: {frames[0].path}")
    height, width = first_image.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(output_path, fps, width, height)
    written_count = 0
    try:
        for index, frame_record in enumerate(frames):
            image = first_image if index == 0 else cv2.imread(str(frame_record.path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable frame: {frame_record.path}")
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            repeat_count = _repeat_count(frames, index, fps, max_gap_s)
            for _ in range(repeat_count):
                writer.write(image)
                written_count += 1
    finally:
        writer.release()

    if written_count == 0:
        raise ValueError("No video frames were written")
    return written_count


def _resolve_log_path(run: str) -> Path:
    path = (DEFAULT_LOGS_DIR / run).resolve()
    if not path.exists():
        raise SystemExit(f"Run does not exist: {path}")
    return path


def _load_frames(log_path: Path) -> list[LoggedFrame]:
    if log_path.is_file() and log_path.name == "frames.jsonl":
        return _load_frames_jsonl(log_path)
    if log_path.is_file() and log_path.name == "run.json":
        return _load_frames_from_run_dir(log_path.parent)
    if log_path.is_dir():
        frames = _load_frames_from_run_dir(log_path)
        if frames:
            return frames
        return _scan_frame_dir(log_path)
    raise SystemExit(f"Unsupported log path: {log_path}")


def _load_frames_from_run_dir(run_dir: Path) -> list[LoggedFrame]:
    manifest_path = run_dir / "frames.jsonl"
    if manifest_path.is_file():
        return _load_frames_jsonl(manifest_path)

    frame_dir = run_dir / "vision_frames"
    if frame_dir.is_dir():
        return _scan_frame_dir(frame_dir)

    frame_dir = run_dir / "frames"
    if frame_dir.is_dir():
        return _scan_frame_dir(frame_dir)

    return []


def _load_frames_jsonl(manifest_path: Path) -> list[LoggedFrame]:
    frames: list[LoggedFrame] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}: {exc}") from exc
            frame = _frame_from_record(record, manifest_path.parent)
            if frame is not None:
                frames.append(frame)
    return _sort_frames(frames)


def _frame_from_record(record: Any, base_dir: Path) -> LoggedFrame | None:
    if not isinstance(record, dict):
        return None
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    frame_id = _optional_int(record.get("frame_id"))
    sim_time_ns = _optional_int(record.get("sim_time_ns"))
    return LoggedFrame(path=path.resolve(), frame_id=frame_id, sim_time_ns=sim_time_ns)


def _scan_frame_dir(frame_dir: Path) -> list[LoggedFrame]:
    extensions = ("*.jpg", "*.jpeg", "*.png")
    frames: list[LoggedFrame] = []
    for pattern in extensions:
        for path in frame_dir.glob(pattern):
            match = FRAME_NAME_RE.fullmatch(path.name)
            frame_id = int(match.group("frame_id")) if match else None
            sim_time_ns = int(match.group("sim_time_ns")) if match else None
            frames.append(LoggedFrame(path=path.resolve(), frame_id=frame_id, sim_time_ns=sim_time_ns))
    return _sort_frames(frames)


def _sort_frames(frames: list[LoggedFrame]) -> list[LoggedFrame]:
    return sorted(
        frames,
        key=lambda frame: (
            frame.sim_time_ns is None,
            frame.sim_time_ns if frame.sim_time_ns is not None else 0,
            frame.frame_id is None,
            frame.frame_id if frame.frame_id is not None else 0,
            frame.path.name,
        ),
    )


def _infer_fps(frames: list[LoggedFrame], default_fps: float) -> float:
    deltas_s = [
        (current.sim_time_ns - previous.sim_time_ns) / NS_PER_SECOND
        for previous, current in zip(frames, frames[1:])
        if previous.sim_time_ns is not None
        and current.sim_time_ns is not None
        and current.sim_time_ns > previous.sim_time_ns
    ]
    if not deltas_s:
        return default_fps
    return 1.0 / median(deltas_s)


def _repeat_count(frames: list[LoggedFrame], index: int, fps: float, max_gap_s: float | None) -> int:
    if index >= len(frames) - 1:
        return 1
    current_time = frames[index].sim_time_ns
    next_time = frames[index + 1].sim_time_ns
    if current_time is None or next_time is None or next_time <= current_time:
        return 1
    duration_s = (next_time - current_time) / NS_PER_SECOND
    if max_gap_s is not None:
        duration_s = min(duration_s, max_gap_s)
    return max(1, round(duration_s * fps))


def _open_video_writer(output_path: Path, fps: float, width: int, height: int) -> Any:
    cv2 = _load_cv2()
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise ValueError(f"Could not open MP4 writer for {output_path}")


def _resolve_output_path(log_path: Path) -> Path:
    run_dir = log_path.parent if log_path.is_file() else log_path
    if run_dir.name in {"vision_frames", "frames"}:
        run_dir = run_dir.parent
    return (run_dir / f"{run_dir.name}.mp4").resolve()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenCV is required to write video. Install project requirements or run with "
            "Flight\\.venv\\Scripts\\python.exe."
        ) from exc
    return cv2


if __name__ == "__main__":
    main()
