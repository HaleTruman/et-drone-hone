import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class Logger:
    def __init__(self, metadata: dict[str, Any] | None = None):
        self.records: dict[str, Any] = {
            "schema_version": 1,
            "metadata": metadata or {},
            "events": [],
            "cycles": [],
            "vision_frames": [],
            "planned_paths": [],
        }
        self.telemetry_records: dict[str, Any] = {
            "schema_version": 1,
            "metadata": self.records["metadata"],
            "samples": [],
        }

    def log_event(self, event: str, **data: Any) -> None:
        self.records["events"].append({"event": event, **data})

    def log_cycle(self, **data: Any) -> None:
        self.records["cycles"].append(data)
        self._log_cycle_telemetry(data)

    def log_telemetry(self, telemetry: Any, **context: Any) -> None:
        self._append_telemetry_sample(telemetry, context)
        self.log_event("telemetry", telemetry=telemetry)

    def log_gate_map(self, gate_map: Any) -> None:
        self.log_event("gate_map", gate_map=gate_map)

    def log_mpcc_solution(self, solution: Any) -> None:
        self.log_event("mpcc_solution", solution=solution)

    def log_vision_frame(self, frame: Any, **data: Any) -> None:
        self.records["vision_frames"].append({"frame": frame, **data})

    def log_planned_path(self, planned_path: Any, **data: Any) -> None:
        self.records["planned_paths"].append({"planned_path": planned_path, **data})

    def save_run(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.records, indent=2, default=self._json_default), encoding="utf-8")
        self.telemetry_path_for_run(output_path).write_text(
            json.dumps(self.telemetry_records, indent=2, default=self._json_default),
            encoding="utf-8",
        )

    @staticmethod
    def timestamped_path(logs_dir: str | Path) -> Path:
        logs_dir = Path(logs_dir)
        timestamp = Logger.rounded_timestamp()
        path = logs_dir / f"run-{timestamp}.json"
        while path.exists():
            timestamp = Logger._add_second(timestamp)
            path = logs_dir / f"run-{timestamp}.json"
        return path

    @staticmethod
    def timestamped_dir(logs_dir: str | Path) -> Path:
        logs_dir = Path(logs_dir)
        timestamp = Logger.rounded_timestamp()
        path = logs_dir / f"run-{timestamp}"
        while path.exists():
            timestamp = Logger._add_second(timestamp)
            path = logs_dir / f"run-{timestamp}"
        return path

    @staticmethod
    def telemetry_path_for_run(run_path: str | Path) -> Path:
        path = Path(run_path)
        if path.name == "run.json":
            return path.parent / "telemetry.json"
        return path.with_name(f"{path.stem}-telemetry.json")

    @staticmethod
    def rounded_timestamp() -> str:
        now = datetime.now(timezone.utc) + timedelta(microseconds=500_000)
        return now.replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _add_second(timestamp: str) -> str:
        value = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ") + timedelta(seconds=1)
        return value.strftime("%Y%m%dT%H%M%SZ")

    def _log_cycle_telemetry(self, cycle: dict[str, Any]) -> None:
        telemetry = cycle.get("telemetry")
        if telemetry is None:
            return
        context = {
            key: cycle[key]
            for key in ("cycle", "sim_time_ns", "wall_elapsed_ms", "deadline_lateness_ms")
            if key in cycle
        }
        self._append_telemetry_sample(telemetry, context)

    def _append_telemetry_sample(self, telemetry: Any, context: dict[str, Any]) -> None:
        sample = dict(context)
        sample["telemetry"] = telemetry
        self.telemetry_records["samples"].append(sample)

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if hasattr(value, "__dict__"):
            return value.__dict__
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
