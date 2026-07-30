from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    path: str
    cycle: int | None = None


@dataclass(frozen=True)
class RunBundle:
    path: str
    name: str
    timestamp: datetime | None
    metadata: dict[str, Any]
    events: list[dict[str, Any]]
    cycles: list[dict[str, Any]]
    frames: list[FrameRecord]
    observations: list[dict[str, Any]]
    gate_map_cycles: list[dict[str, Any]]
    planned_paths: list[dict[str, Any]]
    test_paths: list[dict[str, Any]]
    raw_counts: dict[str, int]

    @property
    def label(self) -> str:
        if self.timestamp is None:
            return self.name
        pattern = "%Y-%m-%d %H:%M:%S.%f UTC" if self.timestamp.microsecond else "%Y-%m-%d %H:%M:%S UTC"
        return self.timestamp.strftime(pattern)


def discover_runs(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    candidates = list((root / "logs" / "runs").glob("run-*"))
    candidates.extend((root / "data" / "live_runs").glob("run-*"))
    return [
        path.resolve()
        for path in sorted(candidates, key=_run_sort_key, reverse=True)
        if path.is_dir() and (path / "run.json").is_file()
    ]


def run_option(path: Path) -> dict[str, Any]:
    timestamp = _timestamp_from_name(path.name)
    label = path.name if timestamp is None else timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    status = _load_json(path / "status.json") if (path / "status.json").is_file() else {}
    summary = _load_json(path / "summary.json") if (path / "summary.json").is_file() else {}
    return {
        "path": str(path),
        "name": path.name,
        "label": label,
        "finalized": bool(status.get("finalized") or summary.get("finalized")),
        "updated_wall_time_utc": status.get("updated_wall_time_utc"),
        "counts": {
            "frames": _count_jsonl(_frames_manifest_path(path)),
            "cycles": int(summary.get("cycles", 0) or _count_jsonl(path / "lists" / "cycles.jsonl")),
            "observations": int(summary.get("vision_observations", 0) or _observation_count_hint(path)),
        },
    }


def load_run_cached(path: str) -> RunBundle:
    run_dir = Path(path).resolve()
    return _load_run_cached(str(run_dir), *_signature(run_dir))


def frame_image_path(run: RunBundle, frame_index: int) -> Path:
    run_dir = Path(run.path).resolve()
    frame = run.frames[frame_index]
    stored = Path(frame.path)
    if stored.is_absolute():
        candidate = stored.resolve()
    else:
        candidate = (run_dir / stored).resolve()
    if not _is_relative_to(candidate, run_dir):
        raise ValueError("Frame path escapes run directory.")
    return candidate


def run_summary(run: RunBundle) -> dict[str, Any]:
    telemetry_times = [_telemetry_sim_time_ns(cycle) for cycle in run.cycles]
    telemetry_times = [value for value in telemetry_times if value is not None]
    observed_ids = {_frame_id_from_observation(record) for record in run.observations}
    span_s = None
    if len(telemetry_times) >= 2:
        span_s = (telemetry_times[-1] - telemetry_times[0]) / 1_000_000_000
    modes = sorted({str(cycle.get("system_mode", "unknown")) for cycle in run.cycles})
    return {
        "path": run.path,
        "name": run.name,
        "label": run.label,
        "metadata": run.metadata,
        "counts": {
            "frames": len(run.frames),
            "cycles": len(run.cycles),
            "events": len(run.events),
            "observations": len(run.observations),
            "gate_map_cycles": len(run.gate_map_cycles),
            "planned_paths": len(run.planned_paths),
            "test_paths": len(run.test_paths),
        },
        "telemetry_span_s": span_s,
        "system_modes": modes,
        "frame_id_range": [run.frames[0].frame_id, run.frames[-1].frame_id] if run.frames else None,
        "observed_frame_indices": [
            index
            for index, frame in enumerate(run.frames)
            if frame.frame_id in observed_ids
        ],
    }


def frame_payload(run: RunBundle, frame_index: int) -> dict[str, Any]:
    if not run.frames:
        raise IndexError("Run has no frames.")
    index = max(0, min(int(frame_index), len(run.frames) - 1))
    frame = run.frames[index]
    sync = nearest_cycle_for_frame(run, frame)
    observation = observation_for_frame(run, frame)
    gates = _observation_gates(observation)
    return {
        "index": index,
        "count": len(run.frames),
        "frame": {
            "frame_id": frame.frame_id,
            "cycle": frame.cycle,
            "sim_time_ns": frame.sim_time_ns,
            "jpeg_size": frame.jpeg_size,
            "path": frame.path,
        },
        "sync": sync,
        "telemetry": sync.get("cycle"),
        "observation": observation,
        "observation_gates": gates,
        "nearby": nearby_frame_rows(run, index),
        "telemetry_series": telemetry_series(run, sync.get("cycle_index")),
        "scene": scene_payload(run, frame, sync.get("cycle")),
        "timeline": timeline_payload(run, index),
    }


def scene_payload(run: RunBundle, frame: FrameRecord, cycle: dict[str, Any] | None) -> dict[str, Any]:
    cycle_number = _cycle_number(frame, cycle)
    telemetry = cycle.get("telemetry") if isinstance(cycle, dict) and isinstance(cycle.get("telemetry"), dict) else {}
    drone = {
        "position_local_ned_m": _point3_or_none(telemetry.get("position_local_ned_m")),
        "attitude_quaternion": _quat4_or_none(telemetry.get("attitude_quaternion") or telemetry.get("attitude")),
    }
    planned_path = _planned_path_for_cycle(run, cycle_number)
    test_path = _test_path_for_cycle(run, cycle_number)
    planned_path_is_test_path = False
    if test_path is None and _looks_like_legacy_test_path(planned_path):
        test_path = planned_path
        planned_path = None
    planned_path_payload_source = planned_path
    if planned_path_payload_source is None and test_path is not None:
        planned_path_payload_source = test_path
        planned_path_is_test_path = True
    gate_map = _gate_map_for_frame(run, frame, cycle_number)
    return {
        "coordinate_system": {
            "world": "local_ned",
            "body": "frd",
            "positive": {"north": "+x", "east": "+y", "down": "+z", "roll": "right", "pitch": "up", "yaw": "right"},
        },
        "cycle": cycle_number,
        "drone": drone,
        "observation_gates": _observation_gates(observation_for_frame(run, frame)),
        "gate_map": gate_map,
        "test_path": _planned_path_payload(test_path),
        "planned_path": _planned_path_payload(planned_path_payload_source),
        "planned_path_is_test_path": planned_path_is_test_path,
    }


def nearby_frame_rows(run: RunBundle, frame_index: int, radius: int = 42) -> list[dict[str, Any]]:
    start = max(0, frame_index - radius)
    end = min(len(run.frames), frame_index + radius + 1)
    rows = []
    observed_ids = {_frame_id_from_observation(record) for record in run.observations}
    for index in range(start, end):
        frame = run.frames[index]
        rows.append(
            {
                "index": index,
                "frame_id": frame.frame_id,
                "cycle": frame.cycle,
                "sim_time_ns": frame.sim_time_ns,
                "has_observation": frame.frame_id in observed_ids,
            }
        )
    return rows


def telemetry_series(run: RunBundle, selected_cycle_index: int | None) -> dict[str, Any]:
    if not run.cycles:
        return {"times_s": [], "selected": None, "position": [], "velocity": [], "rates": [], "groups": []}
    stride = max(1, len(run.cycles) // 1600)
    sampled = run.cycles[::stride]
    base_ns = next((_telemetry_sim_time_ns(cycle) for cycle in run.cycles if _telemetry_sim_time_ns(cycle) is not None), None)
    times: list[float] = []
    position: list[list[float | None]] = []
    velocity: list[list[float | None]] = []
    acceleration: list[list[float | None]] = []
    attitude: list[list[float | None]] = []
    rates: list[list[float | None]] = []
    imu_acceleration: list[list[float | None]] = []
    imu_gyro: list[list[float | None]] = []
    sim_truth_position: list[list[float | None]] = []
    sim_truth_velocity: list[list[float | None]] = []
    sim_truth_attitude: list[list[float | None]] = []
    sim_truth_rates: list[list[float | None]] = []
    for cycle in sampled:
        sim_ns = _telemetry_sim_time_ns(cycle)
        times.append((sim_ns - base_ns) / 1_000_000_000 if sim_ns is not None and base_ns is not None else float(len(times)))
        telemetry = cycle.get("telemetry") if isinstance(cycle.get("telemetry"), dict) else {}
        imu = telemetry.get("imu") if isinstance(telemetry.get("imu"), dict) else {}
        sim_truth = telemetry.get("sim_truth") if isinstance(telemetry.get("sim_truth"), dict) else {}
        truth_odometry = sim_truth.get("odometry") if isinstance(sim_truth.get("odometry"), dict) else {}
        truth_local_position = sim_truth.get("local_position_ned") if isinstance(sim_truth.get("local_position_ned"), dict) else {}
        truth_attitude = sim_truth.get("attitude") if isinstance(sim_truth.get("attitude"), dict) else {}
        position.append(_vec3(telemetry.get("position_local_ned_m")))
        velocity.append(_vec3(telemetry.get("velocity_local_ned_mps")))
        acceleration.append(_vec3(telemetry.get("acceleration_local_ned_mps2")))
        attitude.append(_vec4(telemetry.get("attitude_quaternion") or telemetry.get("attitude")))
        rates.append(_vec3(telemetry.get("body_rates_frd_rps") or telemetry.get("body_rates_rps")))
        imu_acceleration.append(_vec3(imu.get("acceleration_body_frd_mps2")))
        imu_gyro.append(_vec3(imu.get("gyro_body_frd_rps")))
        sim_truth_position.append(_vec3(truth_odometry.get("position_local_ned_m") or truth_local_position.get("position_local_ned_m")))
        sim_truth_velocity.append(_vec3(truth_odometry.get("velocity_local_ned_mps") or truth_local_position.get("velocity_local_ned_mps")))
        sim_truth_attitude.append(_vec4(truth_odometry.get("attitude_quaternion")))
        sim_truth_rates.append(_vec3(truth_odometry.get("body_rates_frd_rps") or truth_attitude.get("body_rates_frd_rps")))
    selected = None
    if selected_cycle_index is not None and 0 <= selected_cycle_index < len(run.cycles):
        sim_ns = _telemetry_sim_time_ns(run.cycles[selected_cycle_index])
        if sim_ns is not None and base_ns is not None:
            selected = (sim_ns - base_ns) / 1_000_000_000
    groups = [
        _telemetry_group(
            "position",
            "Position Local NED",
            "m",
            ["x", "y", "z"],
            [("estimate", position), ("sim truth", sim_truth_position)],
        ),
        _telemetry_group(
            "velocity",
            "Velocity Local NED",
            "m/s",
            ["vx", "vy", "vz"],
            [("estimate", velocity), ("sim truth", sim_truth_velocity)],
        ),
        _telemetry_group(
            "acceleration",
            "Acceleration Local NED",
            "m/s2",
            ["ax", "ay", "az"],
            [("estimate", acceleration)],
        ),
        _telemetry_group(
            "body_rates",
            "Body Rates FRD",
            "rad/s",
            ["p", "q", "r"],
            [("estimate", rates), ("sim truth", sim_truth_rates)],
        ),
        _telemetry_group(
            "attitude",
            "Attitude Quaternion",
            "",
            ["w", "x", "y", "z"],
            [("estimate", attitude), ("sim truth", sim_truth_attitude)],
        ),
        _telemetry_group(
            "imu_acceleration",
            "IMU Acceleration Body FRD",
            "m/s2",
            ["xacc", "yacc", "zacc"],
            [("imu", imu_acceleration)],
        ),
        _telemetry_group(
            "imu_gyro",
            "IMU Gyro Body FRD",
            "rad/s",
            ["xgyro", "ygyro", "zgyro"],
            [("imu", imu_gyro)],
        ),
    ]
    return {
        "times_s": times,
        "selected": selected,
        "position": position,
        "velocity": velocity,
        "rates": rates,
        "groups": [group for group in groups if group is not None],
    }


def timeline_payload(run: RunBundle, selected_frame_index: int) -> dict[str, Any]:
    if not run.frames:
        return {"frames": [], "selected": 0}
    stride = max(1, len(run.frames) // 900)
    observed_ids = {_frame_id_from_observation(record) for record in run.observations}
    return {
        "selected": selected_frame_index,
        "frames": [
            {
                "index": index,
                "frame_id": run.frames[index].frame_id,
                "observed": run.frames[index].frame_id in observed_ids,
            }
            for index in range(0, len(run.frames), stride)
        ],
    }


def nearest_cycle_for_frame(run: RunBundle, frame: FrameRecord) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for index, cycle in enumerate(run.cycles):
        if cycle.get("vision_frame_id") == frame.frame_id:
            payload = _sync_payload(run, frame, cycle, index)
            payload["match_source"] = "vision_frame_id"
            candidates.append(payload)
    if frame.cycle is not None:
        for index, cycle in enumerate(run.cycles):
            if cycle.get("cycle") == frame.cycle or cycle.get("inner_cycle") == frame.cycle:
                payload = _sync_payload(run, frame, cycle, index)
                payload["match_source"] = "frame_cycle"
                candidates.append(payload)
    target_ns = _frame_target_sim_time_ns(run, frame)
    if target_ns is None:
        if candidates:
            return candidates[0]
        return {"cycle": None, "cycle_index": None, "target_sim_time_ns": None, "alignment_error_ms": None, "match_source": None}
    timestamp_candidates = [(index, cycle, _telemetry_sim_time_ns(cycle)) for index, cycle in enumerate(run.cycles)]
    timestamp_candidates = [(index, cycle, sim_ns) for index, cycle, sim_ns in timestamp_candidates if sim_ns is not None]
    if timestamp_candidates:
        index, cycle, sim_ns = min(timestamp_candidates, key=lambda item: abs(item[2] - target_ns))
        candidates.append(
            {
                "cycle": cycle,
                "cycle_index": index,
                "target_sim_time_ns": target_ns,
                "alignment_error_ms": abs(sim_ns - target_ns) / 1_000_000,
                "match_source": "timestamp",
            }
        )
    if not candidates:
        return {"cycle": None, "cycle_index": None, "target_sim_time_ns": target_ns, "alignment_error_ms": None, "match_source": None}
    return min(
        candidates,
        key=lambda item: (
            float("inf") if item.get("alignment_error_ms") is None else float(item["alignment_error_ms"]),
            item.get("cycle_index") if isinstance(item.get("cycle_index"), int) else 10**12,
        ),
    )


def observation_for_frame(run: RunBundle, frame: FrameRecord) -> dict[str, Any] | None:
    selected = None
    for record in run.observations:
        if _frame_id_from_observation(record) == frame.frame_id:
            selected = record
    return selected


@lru_cache(maxsize=12)
def _load_run_cached(
    resolved_path: str,
    run_mtime_ns: int,
    telemetry_mtime_ns: int,
    frames_mtime_ns: int,
    gate_map_mtime_ns: int,
    obs_mtime_ns: int,
    vf_mtime_ns: int,
    test_paths_mtime_ns: int,
) -> RunBundle:
    return load_run(Path(resolved_path))


def load_run(run_dir: Path) -> RunBundle:
    raw = _load_json(run_dir / "run.json")
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
    events = _load_records(raw, "events", run_dir / "lists" / "events.jsonl")
    cycles = [_normalize_cycle(cycle) for cycle in _load_cycles(raw, run_dir)]
    frames = _load_frames(_frames_manifest_path(run_dir))
    gate_map_cycles = _load_sidecar_cycles(run_dir / "gate_map.json", run_dir / "lists" / "gate_map.jsonl")
    observations = _load_observations(run_dir, raw)
    planned_paths = _load_records(raw, "planned_paths", run_dir / "lists" / "planned_paths.jsonl")
    test_paths = _load_records(raw, "test_paths", run_dir / "lists" / "test_paths.jsonl")
    return RunBundle(
        path=str(run_dir.resolve()),
        name=run_dir.name,
        timestamp=_timestamp_from_name(run_dir.name),
        metadata=metadata,
        events=events,
        cycles=cycles,
        frames=frames,
        observations=observations,
        gate_map_cycles=gate_map_cycles,
        planned_paths=planned_paths,
        test_paths=test_paths,
        raw_counts={
            "events": len(raw.get("events", [])) if isinstance(raw.get("events"), list) else 0,
            "cycles": len(raw.get("cycles", [])) if isinstance(raw.get("cycles"), list) else 0,
            "vision_frames": len(raw.get("vision_frames", [])) if isinstance(raw.get("vision_frames"), list) else 0,
            "vision_observations": len(raw.get("vision_observations", [])) if isinstance(raw.get("vision_observations"), list) else 0,
            "test_paths": len(raw.get("test_paths", [])) if isinstance(raw.get("test_paths"), list) else 0,
        },
    )


def _load_cycles(raw: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    cycles = _load_records(raw, "cycles", run_dir / "lists" / "cycles.jsonl")
    telemetry_sidecar = _load_sidecar_samples(run_dir / "telemetry.json", run_dir / "lists" / "telemetry.jsonl")
    return cycles or telemetry_sidecar


def _load_records(raw: dict[str, Any], key: str, jsonl_path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(jsonl_path)
    if records:
        return records
    value = raw.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _load_observations(run_dir: Path, raw: dict[str, Any]) -> list[dict[str, Any]]:
    records = _load_records(raw, "vision_observations", run_dir / "lists" / "vision_observations.jsonl")
    by_frame: dict[int, dict[str, Any]] = {}
    for record in records:
        frame_id = _frame_id_from_observation(record)
        if frame_id is not None:
            by_frame[frame_id] = record
    vision_frames = _load_records(raw, "vision_frames", run_dir / "lists" / "vision_frames.jsonl")
    for record in vision_frames:
        frame = record.get("frame") if isinstance(record.get("frame"), dict) else None
        observation = frame.get("observation") if isinstance(frame, dict) else None
        if isinstance(observation, dict):
            frame_id = _frame_id_from_observation({"observation": observation, **{k: v for k, v in record.items() if k != "frame"}})
            if frame_id is not None:
                by_frame.setdefault(frame_id, {"source": "vision_frames", "observation": observation, "frame_record": record})
    return [by_frame[key] for key in sorted(by_frame)]


def _load_frames(path: Path) -> list[FrameRecord]:
    frames = []
    seen: set[int] = set()
    for record in _load_jsonl(path):
        try:
            frame_id = int(record["frame_id"])
            if frame_id in seen:
                continue
            seen.add(frame_id)
            frames.append(
                FrameRecord(
                    frame_id=frame_id,
                    sim_time_ns=int(record["sim_time_ns"]),
                    jpeg_size=int(record.get("jpeg_size", 0)),
                    path=str(record["path"]),
                    cycle=int(record["cycle"]) if record.get("cycle") is not None else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return frames


def _load_sidecar_samples(json_path: Path, jsonl_path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(jsonl_path)
    if records:
        return records
    raw = _load_json(json_path) if json_path.is_file() else {}
    samples = raw.get("samples")
    return [item for item in samples if isinstance(item, dict)] if isinstance(samples, list) else []


def _load_sidecar_cycles(json_path: Path, jsonl_path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(jsonl_path)
    if records:
        return records
    raw = _load_json(json_path) if json_path.is_file() else {}
    cycles = raw.get("cycles")
    return [item for item in cycles if isinstance(item, dict)] if isinstance(cycles, list) else []


def _normalize_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(cycle)
    telemetry = normalized.get("telemetry")
    if isinstance(telemetry, dict):
        telemetry = dict(telemetry)
        vehicle_state = telemetry.get("vehicle_state") if isinstance(telemetry.get("vehicle_state"), dict) else telemetry.get("odometry")
        if isinstance(vehicle_state, dict):
            vehicle_state = dict(vehicle_state)
            if "body_rates_frd_rps" not in vehicle_state and "body_rates_rps" in vehicle_state:
                vehicle_state["body_rates_frd_rps"] = vehicle_state["body_rates_rps"]
            telemetry.setdefault("position_local_ned_m", vehicle_state.get("position_local_ned_m"))
            telemetry.setdefault("velocity_local_ned_mps", vehicle_state.get("velocity_local_ned_mps"))
            telemetry.setdefault("acceleration_local_ned_mps2", vehicle_state.get("acceleration_local_ned_mps2"))
            telemetry.setdefault("attitude_quaternion", vehicle_state.get("attitude_quaternion") or vehicle_state.get("attitude"))
            telemetry.setdefault("body_rates_frd_rps", vehicle_state.get("body_rates_frd_rps"))
            normalized.setdefault("vehicle_state", vehicle_state)
        normalized["telemetry"] = telemetry
    return normalized


def _sync_payload(run: RunBundle, frame: FrameRecord, cycle: dict[str, Any], index: int) -> dict[str, Any]:
    target_ns = _frame_target_sim_time_ns(run, frame)
    sim_ns = _telemetry_sim_time_ns(cycle)
    error_ms = abs(sim_ns - target_ns) / 1_000_000 if sim_ns is not None and target_ns is not None else None
    return {"cycle": cycle, "cycle_index": index, "target_sim_time_ns": target_ns, "alignment_error_ms": error_ms}


def _frame_target_sim_time_ns(run: RunBundle, frame: FrameRecord) -> int | None:
    offset = _timesync_offset_ns(run.events, run.cycles)
    return frame.sim_time_ns - offset if offset is not None else frame.sim_time_ns


def _timesync_offset_ns(events: list[dict[str, Any]], cycles: list[dict[str, Any]]) -> int | None:
    offsets = []
    for record in [*events, *cycles]:
        bridge = record.get("bridge") if isinstance(record, dict) else None
        if not isinstance(bridge, dict):
            continue
        timesync = bridge.get("latest_timesync")
        telemetry = bridge.get("latest_telemetry")
        if not isinstance(telemetry, dict):
            telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else None
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
    telemetry = cycle.get("telemetry") if isinstance(cycle, dict) else None
    value = telemetry.get("sim_time_ns") if isinstance(telemetry, dict) else cycle.get("sim_time_ns")
    return int(value) if isinstance(value, int) else None


def _observation_gates(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    observation = record.get("observation") if isinstance(record, dict) else None
    gates = observation.get("gates") if isinstance(observation, dict) else None
    return [_observation_gate_payload(gate) for gate in gates if isinstance(gate, dict)] if isinstance(gates, list) else []


def _observation_gate_payload(gate: dict[str, Any]) -> dict[str, Any]:
    position_xyz = _point3_or_none(gate.get("position_xyz"))
    position_local_ned = _point3_or_none(gate.get("position_local_ned") or gate.get("position_local_ned_m"))
    position_relative_ned = _point3_or_none(gate.get("position_relative_ned_m"))
    orientation_xyz = _point3_or_none(gate.get("orientation_xyz"))
    orientation_quaternion = _quat4_or_none(
        gate.get("orientation_local_ned_quat")
        or gate.get("orientation_quat")
        or gate.get("quaternion")
        or gate.get("quat")
    )
    return {
        **gate,
        "id": gate.get("id") or gate.get("gate_id"),
        "position_xyz": position_xyz,
        "position_local_ned": position_local_ned,
        "position_local_ned_m": position_local_ned,
        "position_relative_ned_m": position_relative_ned,
        "position_confidence": gate.get("position_confidence")
        if gate.get("position_confidence") is not None
        else gate.get("confidence"),
        "orientation_xyz": orientation_xyz,
        "orientation_local_ned_quat": orientation_quaternion,
        "orientation_quat": orientation_quaternion,
        "orientation_confidence": gate.get("orientation_confidence"),
        "has_orientation": orientation_xyz is not None or orientation_quaternion is not None,
    }


def _cycle_number(frame: FrameRecord, cycle: dict[str, Any] | None) -> int | None:
    if isinstance(cycle, dict):
        for key in ("cycle", "inner_cycle"):
            value = cycle.get(key)
            if isinstance(value, int):
                return value
    return frame.cycle


def _gate_map_for_frame(run: RunBundle, frame: FrameRecord, cycle_number: int | None) -> list[dict[str, Any]]:
    for record in run.gate_map_cycles:
        try:
            frame_id = int(record.get("frame_id"))
        except (TypeError, ValueError):
            continue
        if frame_id == frame.frame_id:
            gates = record.get("gate_map")
            return [_gate_payload(gate) for gate in gates if isinstance(gate, dict)] if isinstance(gates, list) else []
    return _gate_map_for_cycle(run, cycle_number)


def _gate_map_for_cycle(run: RunBundle, cycle_number: int | None) -> list[dict[str, Any]]:
    selected: dict[str, Any] | None = None
    if cycle_number is not None:
        for record in run.gate_map_cycles:
            value = record.get("cycle")
            if isinstance(value, int) and value <= cycle_number:
                selected = record
    if selected is None and run.gate_map_cycles:
        selected = run.gate_map_cycles[-1]
    gates = selected.get("gate_map") if isinstance(selected, dict) else None
    return [_gate_payload(gate) for gate in gates if isinstance(gate, dict)] if isinstance(gates, list) else []


def _planned_path_for_cycle(run: RunBundle, cycle_number: int | None) -> dict[str, Any] | None:
    return _path_for_cycle(run.planned_paths, "planned_path", cycle_number)


def _test_path_for_cycle(run: RunBundle, cycle_number: int | None) -> dict[str, Any] | None:
    return _path_for_cycle(run.test_paths, "test_path", cycle_number)


def _path_for_cycle(records: list[dict[str, Any]], key: str, cycle_number: int | None) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    for record in records:
        if not isinstance(record, dict):
            continue
        record_cycle = record.get("cycle")
        if cycle_number is not None and isinstance(record_cycle, int) and record_cycle > cycle_number:
            continue
        planned_path = record.get(key)
        if isinstance(planned_path, dict):
            selected = planned_path
    return selected


def _planned_path_payload(planned_path: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(planned_path, dict):
        return None
    origin = _point3_or_none(planned_path.get("origin_local_ned_m")) or [0.0, 0.0, 0.0]
    relative_points = planned_path.get("points_relative_ned_m")
    local_points = planned_path.get("points_local_ned_m")
    points: list[list[float]] = []
    if isinstance(local_points, list):
        points = [point for point in (_point3_or_none(point) for point in local_points) if point is not None]
    elif isinstance(relative_points, list):
        for point in relative_points:
            relative = _point3_or_none(point)
            if relative is not None:
                points.append([origin[i] + relative[i] for i in range(3)])
    return {
        "points_local_ned_m": points,
        "origin_local_ned_m": origin,
        "gate_ids": planned_path.get("gate_ids") if isinstance(planned_path.get("gate_ids"), list) else [],
        "source": planned_path.get("source"),
    }


def _looks_like_legacy_test_path(path: dict[str, Any] | None) -> bool:
    if not isinstance(path, dict):
        return False
    source = str(path.get("source") or "")
    return source in {"straight_line", "test_path"}


def _gate_payload(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": gate.get("id") or gate.get("gate_id") or gate.get("sequence"),
        "position_local_ned_m": _point3_or_none(gate.get("position_local_ned_m") or gate.get("pos")),
        "position_relative_ned_m": _point3_or_none(gate.get("position_relative_ned_m")),
        "quaternion": _quat4_or_none(gate.get("quaternion") or gate.get("quat")),
        "confidence": gate.get("confidence") or gate.get("position_confidence"),
        "sequence": gate.get("sequence"),
        "crossed": bool(gate.get("crossed")),
        "source": gate.get("source"),
        "outer_width_m": gate.get("outer_width_m"),
        "outer_height_m": gate.get("outer_height_m"),
        "inner_width_m": gate.get("inner_width_m"),
        "inner_height_m": gate.get("inner_height_m"),
        "depth_m": gate.get("depth_m"),
    }


def _point3_or_none(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _quat4_or_none(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _frame_id_from_observation(record: dict[str, Any] | None) -> int | None:
    if not isinstance(record, dict):
        return None
    for key in ("frame_id", "cycle"):
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    observation = record.get("observation")
    run = observation.get("run") if isinstance(observation, dict) else None
    if isinstance(run, dict):
        value = run.get("cycle")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        label = run.get("frame_id")
        if isinstance(label, str) and label.startswith("frame_"):
            try:
                return int(label.removeprefix("frame_"))
            except ValueError:
                return None
    frame_record = record.get("frame_record")
    frame = frame_record.get("frame") if isinstance(frame_record, dict) else None
    if isinstance(frame, dict):
        try:
            return int(frame.get("frame_id"))
        except (TypeError, ValueError):
            return None
    return None


def _vec3(value: Any) -> list[float | None]:
    if isinstance(value, list) and len(value) >= 3:
        return [_number_or_none(value[0]), _number_or_none(value[1]), _number_or_none(value[2])]
    return [None, None, None]


def _vec4(value: Any) -> list[float | None]:
    if isinstance(value, list) and len(value) >= 4:
        return [_number_or_none(value[0]), _number_or_none(value[1]), _number_or_none(value[2]), _number_or_none(value[3])]
    return [None, None, None, None]


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _telemetry_group(
    group_id: str,
    title: str,
    unit: str,
    axes: list[str],
    series: list[tuple[str, list[list[float | None]]]],
) -> dict[str, Any] | None:
    populated = [
        {"label": label, "values": values}
        for label, values in series
        if any(any(value is not None for value in row) for row in values)
    ]
    if not populated:
        return None
    return {
        "id": group_id,
        "title": title,
        "unit": unit,
        "axes": axes,
        "series": populated,
    }


def _signature(run_dir: Path) -> tuple[int, int, int, int, int, int, int]:
    return (
        _mtime_ns(run_dir / "run.json"),
        _mtime_ns(run_dir / "telemetry.json"),
        _mtime_ns(_frames_manifest_path(run_dir)),
        _mtime_ns(run_dir / "gate_map.json"),
        _mtime_ns(run_dir / "lists" / "vision_observations.jsonl"),
        _mtime_ns(run_dir / "lists" / "vision_frames.jsonl"),
        _mtime_ns(run_dir / "lists" / "test_paths.jsonl"),
    )


def _frames_manifest_path(run_dir: Path) -> Path:
    for path in (run_dir / "frames.jsonl", run_dir / "frames" / "frames.jsonl", run_dir / "vision_frames" / "frames.jsonl"):
        if path.is_file():
            return path
    return run_dir / "frames.jsonl"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _observation_count_hint(run_dir: Path) -> int:
    count = _count_jsonl(run_dir / "lists" / "vision_observations.jsonl")
    return count if count else _count_jsonl(run_dir / "lists" / "vision_frames.jsonl")


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _run_sort_key(path: Path) -> tuple[datetime, str]:
    return _timestamp_from_name(path.name) or datetime.min, path.name


def _timestamp_from_name(name: str) -> datetime | None:
    for pattern in ("run-%Y%m%dT%H%M%SZ", "run-%Y%m%dT%H%M%S.%fZ"):
        try:
            return datetime.strptime(name, pattern)
        except ValueError:
            pass
    return None
