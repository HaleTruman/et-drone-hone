import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlencode


@dataclass(frozen=True)
class LiveFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    path: str
    cycle: int | None = None


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
    gate_map_cycles: list[dict[str, Any]]
    vision_observations: list[dict[str, Any]]
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


def live_run_option(path: str) -> dict[str, str]:
    run_dir = Path(path).resolve()
    timestamp = _timestamp_from_name(run_dir.name)
    if timestamp is None:
        label = run_dir.name
    else:
        pattern = "%Y-%m-%d %H:%M:%S.%f UTC" if timestamp.microsecond else "%Y-%m-%d %H:%M:%S UTC"
        label = timestamp.strftime(pattern)
    return {"label": f"{label}  |  {run_dir.name}", "value": str(run_dir)}


def load_live_run_cached(path: str) -> LiveRun:
    run_dir = Path(path).resolve()
    return _load_live_run_cached(str(run_dir), *live_run_signature(str(run_dir)))


def live_run_signature(path: str) -> tuple[int, int, int, int, int, int, int]:
    return _run_signature(Path(path).resolve())


def load_live_run(path: str) -> LiveRun:
    run_dir = Path(path).resolve()
    raw = json.loads((run_dir / "run.json").read_text(encoding="utf-8-sig"))
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
    from app.data import _load_telemetry_sidecar, _normalized_cycles

    cycles = _normalized_cycles(cycles, _load_telemetry_sidecar(run_dir / "run.json"))
    gate_map_cycles = _load_gate_map_sidecar(run_dir / "run.json")
    vision_observations = _load_vision_observations(run_dir, raw)
    raw = dict(raw)
    raw["planned_paths"] = _load_path_records(run_dir, raw, "planned_paths")
    raw["test_paths"] = _load_path_records(run_dir, raw, "test_paths")

    return LiveRun(
        path=str(run_dir),
        name=run_dir.name,
        timestamp=_timestamp_from_name(run_dir.name),
        schema_version=raw.get("schema_version", "unknown"),
        metadata=metadata,
        events=events,
        cycles=cycles,
        frames=_load_frames(_frames_manifest_path(run_dir)),
        gate_map_cycles=gate_map_cycles,
        vision_observations=vision_observations,
        raw=raw,
    )


def frame_image_url(run: LiveRun, frame_index: int) -> str:
    frame = run.frames[int(frame_index)]
    image_path = frame_image_path(run, frame)
    cache_key = f"{frame.frame_id}-{frame.jpeg_size}-{_mtime_ns(image_path)}"
    return "/live-frame?" + urlencode({"run": run.path, "index": int(frame_index), "v": cache_key})


def frame_image_path(run: LiveRun, frame: LiveFrame) -> Path:
    run_dir = Path(run.path).resolve()
    frame_path = Path(frame.path)
    if frame_path.parent == Path("."):
        image_path = (_frames_dir(run_dir) / frame_path).resolve()
    else:
        image_path = (run_dir / frame_path).resolve()
    if not _is_relative_to(image_path, run_dir):
        raise ValueError("Frame path must stay within the run directory.")
    return image_path


def nearest_cycle_for_frame(run: LiveRun, frame: LiveFrame) -> FrameSync:
    candidates: list[FrameSync] = []
    offset_ns = _timesync_offset_ns(run.events)
    target_sim_time_ns = frame.sim_time_ns - offset_ns if offset_ns is not None else None

    for index, cycle in enumerate(run.cycles):
        if cycle.get("vision_frame_id") != frame.frame_id:
            continue
        cycle_sim_time_ns = _telemetry_sim_time_ns(cycle)
        candidates.append(
            FrameSync(
                cycle=cycle,
                cycle_index=index,
                target_sim_time_ns=target_sim_time_ns,
                error_ms=abs(cycle_sim_time_ns - target_sim_time_ns) / 1_000_000
                if cycle_sim_time_ns is not None and target_sim_time_ns is not None
                else None,
            )
        )

    if frame.cycle is not None:
        for index, cycle in enumerate(run.cycles):
            if cycle.get("cycle") == frame.cycle:
                cycle_sim_time_ns = _telemetry_sim_time_ns(cycle)
                candidates.append(
                    FrameSync(
                        cycle=cycle,
                        cycle_index=index,
                        target_sim_time_ns=target_sim_time_ns,
                        error_ms=abs(cycle_sim_time_ns - target_sim_time_ns) / 1_000_000
                        if cycle_sim_time_ns is not None and target_sim_time_ns is not None
                        else None,
                    )
                )

    if target_sim_time_ns is not None:
        timestamp_candidates = [(index, cycle, _telemetry_sim_time_ns(cycle)) for index, cycle in enumerate(run.cycles)]
        timestamp_candidates = [(index, cycle, sim_time_ns) for index, cycle, sim_time_ns in timestamp_candidates if sim_time_ns is not None]
        if timestamp_candidates:
            cycle_index, cycle, cycle_sim_time_ns = min(timestamp_candidates, key=lambda item: abs(item[2] - target_sim_time_ns))
            candidates.append(
                FrameSync(
                    cycle=cycle,
                    cycle_index=cycle_index,
                    target_sim_time_ns=target_sim_time_ns,
                    error_ms=abs(cycle_sim_time_ns - target_sim_time_ns) / 1_000_000,
                )
            )

    if not candidates:
        return FrameSync(cycle=None, cycle_index=None, target_sim_time_ns=target_sim_time_ns, error_ms=None)
    return min(
        candidates,
        key=lambda item: (
            float("inf") if item.error_ms is None else float(item.error_ms),
            item.cycle_index if item.cycle_index is not None else 10**12,
        ),
    )


def vision_observation_for_frame(run: LiveRun, frame: LiveFrame) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    for record in run.vision_observations:
        frame_id = record.get("frame_id") if isinstance(record, dict) else None
        if frame_id is None:
            frame_id = _observation_frame_id(record.get("observation") if isinstance(record, dict) else None)
        try:
            if int(frame_id) == int(frame.frame_id):
                selected = record
        except (TypeError, ValueError):
            continue
    return selected


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
    seen_frame_ids: set[int] = set()
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        frame_id = int(record["frame_id"])
        if frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(frame_id)
        path = str(record["path"])
        frames.append(
            LiveFrame(
                frame_id=frame_id,
                sim_time_ns=int(record["sim_time_ns"]),
                jpeg_size=int(record["jpeg_size"]),
                path=path,
                cycle=int(record["cycle"]) if record.get("cycle") is not None else None,
            )
        )
    return frames


@lru_cache(maxsize=16)
def _load_live_run_cached(
    resolved_path: str,
    run_mtime_ns: int,
    telemetry_mtime_ns: int,
    frames_mtime_ns: int,
    gate_map_mtime_ns: int,
    vision_observations_mtime_ns: int,
    planned_paths_mtime_ns: int,
    test_paths_mtime_ns: int,
) -> LiveRun:
    return load_live_run(resolved_path)


def _run_signature(run_dir: Path) -> tuple[int, int, int, int, int, int, int]:
    run_path = run_dir / "run.json"
    telemetry_path = run_dir / "telemetry.json"
    frames_path = _frames_manifest_path(run_dir)
    gate_map_path = run_dir / "gate_map.json"
    observations_path = _vision_observations_path(run_dir)
    planned_paths_path = run_dir / "lists" / "planned_paths.jsonl"
    test_paths_path = run_dir / "lists" / "test_paths.jsonl"
    return (
        _mtime_ns(run_path),
        _mtime_ns(telemetry_path),
        _mtime_ns(frames_path),
        _mtime_ns(gate_map_path),
        _mtime_ns(observations_path),
        _mtime_ns(planned_paths_path),
        _mtime_ns(test_paths_path),
    )


def _load_gate_map_sidecar(run_path: Path) -> list[dict[str, Any]]:
    candidates = [
        run_path.parent / "gate_map.json",
        run_path.with_name(f"{run_path.stem}-gate_map.json"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        cycles = raw.get("cycles", []) if isinstance(raw, dict) else []
        if isinstance(cycles, list):
            return [cycle for cycle in cycles if isinstance(cycle, dict)]
    return []


def _load_vision_observations(run_dir: Path, raw: dict[str, Any]) -> list[dict[str, Any]]:
    observations_path = _vision_observations_path(run_dir)
    if observations_path.is_file():
        return _load_jsonl_records(observations_path)
    records = raw.get("vision_observations")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _load_path_records(run_dir: Path, raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    jsonl_path = run_dir / "lists" / f"{key}.jsonl"
    if jsonl_path.is_file():
        return _load_jsonl_records(jsonl_path)
    records = raw.get(key)
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def _vision_observations_path(run_dir: Path) -> Path:
    return run_dir / "lists" / "vision_observations.jsonl"


def _observation_frame_id(observation: Any) -> int | None:
    run = observation.get("run") if isinstance(observation, dict) else None
    if not isinstance(run, dict):
        return None
    frame_id = run.get("cycle")
    if frame_id is not None:
        try:
            return int(frame_id)
        except (TypeError, ValueError):
            return None
    frame_label = run.get("frame_id")
    if isinstance(frame_label, str) and frame_label.startswith("frame_"):
        try:
            return int(frame_label.removeprefix("frame_"))
        except ValueError:
            return None
    return None


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _frames_manifest_path(run_dir: Path) -> Path:
    frames_manifest = run_dir / "frames.jsonl"
    if frames_manifest.is_file():
        return frames_manifest
    frames_manifest = run_dir / "frames" / "frames.jsonl"
    if frames_manifest.is_file():
        return frames_manifest
    return run_dir / "vision_frames" / "frames.jsonl"


def _frames_dir(run_dir: Path) -> Path:
    frames_dir = run_dir / "frames"
    if frames_dir.is_dir():
        return frames_dir
    return run_dir / "vision_frames"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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
        response_time_ns = timesync.get("response_time_ns", timesync.get("tc1"))
        sim_time_ns = telemetry.get("sim_time_ns")
        if sim_time_ns is None and isinstance(telemetry.get("time_usec"), int):
            sim_time_ns = telemetry["time_usec"] * 1_000
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
