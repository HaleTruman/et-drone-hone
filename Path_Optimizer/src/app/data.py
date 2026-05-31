from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLog:
    path: str
    name: str
    timestamp: datetime | None
    schema_version: Any
    metadata: dict[str, Any]
    events: list[dict[str, Any]]
    cycles: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        if self.timestamp is None:
            return self.name
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f UTC")


def discover_run_files(root_dir: str) -> list[str]:
    logs_dir = Path(root_dir) / "logs"
    if not logs_dir.is_dir():
        return []
    return [
        str(path.resolve())
        for path in sorted(logs_dir.glob("run-*.json"), key=_run_sort_key, reverse=True)
        if path.is_file()
    ]


def load_run(path: str) -> RunLog:
    resolved_path = str(Path(path).resolve())
    with open(resolved_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Run log root must be a JSON object.")

    metadata = raw.get("metadata", {})
    events = raw.get("events", [])
    cycles = raw.get("cycles", [])
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object.")
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ValueError("events must be a list of JSON objects.")
    if not isinstance(cycles, list) or not all(isinstance(cycle, dict) for cycle in cycles):
        raise ValueError("cycles must be a list of JSON objects.")

    name = os.path.basename(resolved_path)
    return RunLog(
        path=resolved_path,
        name=name,
        timestamp=_timestamp_from_name(name),
        schema_version=raw.get("schema_version", "unknown"),
        metadata=metadata,
        events=events,
        cycles=cycles,
        raw=raw,
    )


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def _flatten(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if not value:
                flattened[prefix] = "{}"
            for key, nested_value in value.items():
                _flatten(nested_value, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(value, list):
            if not value:
                flattened[prefix] = "[]"
            for index, nested_value in enumerate(value):
                _flatten(nested_value, f"{prefix}[{index}]")
            return
        flattened[prefix] = value

    _flatten(record, "")
    return flattened


def value_at(record: dict[str, Any], *keys: str | int) -> Any:
    value: Any = record
    for key in keys:
        if isinstance(key, int):
            if not isinstance(value, list) or key >= len(value):
                return None
            value = value[key]
        else:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        if value is None:
            return None
    return value


def cycle_times_s(cycles: list[dict[str, Any]]) -> list[float]:
    return [
        float(cycle.get("sim_time_ns", index)) / 1_000_000_000
        if "sim_time_ns" in cycle
        else float(index)
        for index, cycle in enumerate(cycles)
    ]


def _run_sort_key(path: Path) -> tuple[datetime, str]:
    return _timestamp_from_name(path.name) or datetime.min, path.name


def _timestamp_from_name(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, "run-%Y%m%dT%H%M%S.%fZ.json")
    except ValueError:
        return None
