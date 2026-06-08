import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class LiveFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    path: str


@dataclass(frozen=True)
class LiveRun:
    path: str
    name: str
    timestamp: datetime | None
    schema_version: Any
    metadata: dict[str, Any]
    events: list[dict[str, Any]]
    cycles: list[dict[str, Any]]
    frames: list[LiveFrame]
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        if self.timestamp is None:
            return self.name
        pattern = "%Y-%m-%d %H:%M:%S.%f UTC" if self.timestamp.microsecond else "%Y-%m-%d %H:%M:%S UTC"
        return self.timestamp.strftime(pattern)


@dataclass(frozen=True)
class FrameSync:
    cycle: dict[str, Any] | None
    cycle_index: int | None
    target_sim_time_ns: int | None
    error_ms: float | None


def discover_live_run_dirs(root_dir: str) -> list[str]:
    root = Path(root_dir)
    paths = list((root / "logs" / "runs").glob("run-*"))
    paths.extend((root / "data" / "live_runs").glob("run-*"))
    return [
        str(path.resolve())
        for path in sorted(paths, key=_run_sort_key, reverse=True)
        if path.is_dir() and (path / "run.json").is_file()
    ]


def load_live_run(path: str) -> LiveRun:
    run_dir = Path(path).resolve()
    raw = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Live run log root must be a JSON object.")

    metadata = raw.get("metadata", {})
    events = raw.get("events", [])
    cycles = raw.get("cycles", [])
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object.")
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ValueError("events must be a list of JSON objects.")
    if not isinstance(cycles, list) or not all(isinstance(cycle, dict) for cycle in cycles):
        raise ValueError("cycles must be a list of JSON objects.")

    return LiveRun(
        path=str(run_dir),
        name=run_dir.name,
        timestamp=_timestamp_from_name(run_dir.name),
        schema_version=raw.get("schema_version", "unknown"),
        metadata=metadata,
        events=events,
        cycles=cycles,
        frames=_load_frames(_frames_manifest_path(run_dir)),
        raw=raw,
    )


def frame_data_uri(run: LiveRun, frame: LiveFrame) -> str:
    frames_dir = _frames_dir(Path(run.path)).resolve()
    image_path = (frames_dir / frame.path).resolve()
    if image_path.parent != frames_dir:
        raise ValueError("Frame path must stay within the run frames directory.")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def nearest_cycle_for_frame(run: LiveRun, frame: LiveFrame) -> FrameSync:
    offset_ns = _timesync_offset_ns(run.events)
    if offset_ns is None:
        return FrameSync(cycle=None, cycle_index=None, target_sim_time_ns=None, error_ms=None)

    target_sim_time_ns = frame.sim_time_ns - offset_ns
    candidates = [
        (index, cycle, _telemetry_sim_time_ns(cycle))
        for index, cycle in enumerate(run.cycles)
    ]
    candidates = [(index, cycle, sim_time_ns) for index, cycle, sim_time_ns in candidates if sim_time_ns is not None]
    if not candidates:
        return FrameSync(cycle=None, cycle_index=None, target_sim_time_ns=target_sim_time_ns, error_ms=None)

    cycle_index, cycle, cycle_sim_time_ns = min(candidates, key=lambda item: abs(item[2] - target_sim_time_ns))
    return FrameSync(
        cycle=cycle,
        cycle_index=cycle_index,
        target_sim_time_ns=target_sim_time_ns,
        error_ms=abs(cycle_sim_time_ns - target_sim_time_ns) / 1_000_000,
    )


def telemetry_times_s(cycles: list[dict[str, Any]]) -> list[float]:
    values = [_telemetry_sim_time_ns(cycle) for cycle in cycles]
    available = [value for value in values if value is not None]
    if not available:
        return [float(index) for index in range(len(cycles))]
    start = available[0]
    return [(value - start) / 1_000_000_000 if value is not None else float(index) for index, value in enumerate(values)]


def _load_frames(manifest_path: Path) -> list[LiveFrame]:
    if not manifest_path.is_file():
        return []
    frames: list[LiveFrame] = []
    seen_paths: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        path = str(record["path"])
        if path in seen_paths:
            continue
        seen_paths.add(path)
        frames.append(
            LiveFrame(
                frame_id=int(record["frame_id"]),
                sim_time_ns=int(record["sim_time_ns"]),
                jpeg_size=int(record["jpeg_size"]),
                path=path,
            )
        )
    return frames


def _frames_manifest_path(run_dir: Path) -> Path:
    frames_manifest = run_dir / "frames" / "frames.jsonl"
    if frames_manifest.is_file():
        return frames_manifest
    return run_dir / "vision_frames" / "frames.jsonl"


def _frames_dir(run_dir: Path) -> Path:
    frames_dir = run_dir / "frames"
    if frames_dir.is_dir():
        return frames_dir
    return run_dir / "vision_frames"


def _timesync_offset_ns(events: list[dict[str, Any]]) -> int | None:
    offsets: list[int] = []
    for event in events:
        bridge = event.get("bridge")
        if not isinstance(bridge, dict):
            continue
        timesync = bridge.get("latest_timesync")
        telemetry = bridge.get("latest_telemetry")
        if not isinstance(timesync, dict) or not isinstance(telemetry, dict):
            continue
        response_time_ns = timesync.get("response_time_ns")
        sim_time_ns = telemetry.get("sim_time_ns")
        if isinstance(response_time_ns, int) and isinstance(sim_time_ns, int):
            offsets.append(response_time_ns - sim_time_ns)
    return round(median(offsets)) if offsets else None


def _telemetry_sim_time_ns(cycle: dict[str, Any]) -> int | None:
    telemetry = cycle.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    value = telemetry.get("sim_time_ns")
    return value if isinstance(value, int) else None


def _run_sort_key(path: Path) -> tuple[datetime, str]:
    return _timestamp_from_name(path.name) or datetime.min, path.name


def _timestamp_from_name(name: str) -> datetime | None:
    for pattern in ("run-%Y%m%dT%H%M%SZ", "run-%Y%m%dT%H%M%S.%fZ"):
        try:
            return datetime.strptime(name, pattern)
        except ValueError:
            pass
    return None
