import json
import logging as python_logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class Logger:
    _pending_run_dir: Path | None = None

    def __init__(self, metadata: dict[str, Any] | None = None, run_dir: str | Path | None = None):
        self.run_dir = Path(run_dir) if run_dir is not None else self._consume_pending_run_dir()
        self._sequence = 0
        self._telemetry_keys: set[tuple[Any, Any]] = set()
        self._last_status_write_s = 0.0
        self._status_write_period_s = 1.0
        self._finalized = False
        self.records: dict[str, Any] = {
            "schema_version": 2,
            "metadata": metadata or {},
            "events": [],
            "cycles": [],
            "vision_frames": [],
            "vision_observations": [],
            "planned_paths": [],
            "test_paths": [],
        }
        self.telemetry_records: dict[str, Any] = {
            "schema_version": 2,
            "metadata": self.records["metadata"],
            "samples": [],
        }
        self.gate_map_records: dict[str, Any] = {
            "schema_version": 2,
            "metadata": self.records["metadata"],
            "cycles": [],
        }
        self._last_cycle: dict[str, Any] | None = None
        self._last_event: dict[str, Any] | None = None
        self._last_command: Any | None = None
        self._max_deadline_lateness_ms = 0.0
        self._max_loop_elapsed_ms = 0.0
        self._console_handler: python_logging.Handler | None = None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._lists_dir().mkdir(parents=True, exist_ok=True)
            self._write_json(self.run_dir / "metadata.json", self.records["metadata"])
            self._console_handler = self._configure_python_logging(self.run_dir)
            self._write_status(reason="logger_started", force=True)

    def log_event(self, event: str, **data: Any) -> None:
        record = self._with_record_fields({"event": event, **data}, record_type="event")
        self.records["events"].append(record)
        self._last_event = record
        self._append_jsonl("events.jsonl", record)
        self._write_status(reason=event)

    def log_cycle(self, **data: Any) -> None:
        record = self._with_record_fields(data, record_type="cycle")
        self.records["cycles"].append(record)
        self._last_cycle = record
        self._last_command = record.get("command")
        self._max_deadline_lateness_ms = max(
            self._max_deadline_lateness_ms,
            self._number(record.get("deadline_lateness_ms")),
        )
        self._max_loop_elapsed_ms = max(
            self._max_loop_elapsed_ms,
            self._number(record.get("loop_elapsed_ms")),
        )
        self._append_jsonl("cycles.jsonl", record)
        self._log_cycle_telemetry(record)
        self._write_status(reason="cycle")

    def log_telemetry(self, telemetry: Any, **context: Any) -> None:
        self._append_telemetry_sample(telemetry, context)

    def log_gate_map(self, gate_map: Any, **context: Any) -> None:
        record = self._with_record_fields({**context, "gate_map": gate_map}, record_type="gate_map")
        self.gate_map_records["cycles"].append(record)
        self._append_jsonl("gate_map.jsonl", record)

    def log_mpcc_solution(self, solution: Any) -> None:
        self.log_event("mpcc_solution", solution=solution)

    def log_vision_frame(self, frame: Any, **data: Any) -> None:
        record = self._with_record_fields({"frame": frame, **data}, record_type="vision_frame")
        self.records["vision_frames"].append(record)
        self._append_jsonl("vision_frames.jsonl", record)

    def log_vision_observation(self, observation: Any, **data: Any) -> None:
        payload = (
            observation.to_controller_payload(output_dir="memory")
            if hasattr(observation, "to_controller_payload")
            else observation
        )
        record = self._with_record_fields({"observation": payload, **data}, record_type="vision_observation")
        self.records["vision_observations"].append(record)
        self._append_jsonl("vision_observations.jsonl", record)

    def log_planned_path(self, planned_path: Any, **data: Any) -> None:
        record = self._with_record_fields({"planned_path": planned_path, **data}, record_type="planned_path")
        self.records["planned_paths"].append(record)
        self._append_jsonl("planned_paths.jsonl", record)

    def log_test_path(self, test_path: Any, **data: Any) -> None:
        record = self._with_record_fields({"test_path": test_path, **data}, record_type="test_path")
        self.records["test_paths"].append(record)
        self._append_jsonl("test_paths.jsonl", record)

    def log_exception(self, event: str, error: BaseException, **context: Any) -> None:
        self.log_event(
            event,
            level="error",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
            **context,
        )

    def save_run(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_dir = output_path.parent
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lists_dir().mkdir(parents=True, exist_ok=True)
        self._finalized = True
        self._write_json(output_path, self.records)
        self._write_json(self.telemetry_path_for_run(output_path), self.telemetry_records)
        self._write_json(self.gate_map_path_for_run(output_path), self.gate_map_records)
        self._write_json(self.run_dir / "summary.json", self.summary())
        self._write_jsonl_sidecars()
        self._write_status(reason="save_run", force=True)
        self.close()

    def close(self) -> None:
        if self._console_handler is None:
            return
        root_logger = python_logging.getLogger()
        root_logger.removeHandler(self._console_handler)
        self._console_handler.close()
        self._console_handler = None

    def summary(self) -> dict[str, Any]:
        command_count = sum(
            1
            for cycle in self.records["cycles"]
            if isinstance(cycle.get("command"), dict) and cycle["command"].get("emitted")
        )
        return {
            "schema_version": 1,
            "metadata": self.records["metadata"],
            "events": len(self.records["events"]),
            "cycles": len(self.records["cycles"]),
            "telemetry_samples": len(self.telemetry_records["samples"]),
            "gate_map_cycles": len(self.gate_map_records["cycles"]),
            "vision_frames": len(self.records["vision_frames"]),
            "vision_observations": len(self.records["vision_observations"]),
            "planned_paths": len(self.records["planned_paths"]),
            "test_paths": len(self.records["test_paths"]),
            "commands_emitted": command_count,
            "max_deadline_lateness_ms": self._max_deadline_lateness_ms,
            "max_loop_elapsed_ms": self._max_loop_elapsed_ms,
            "last_event": self._compact_record(self._last_event),
            "last_cycle": self._compact_record(self._last_cycle),
            "finalized": self._finalized,
        }

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
        Logger._pending_run_dir = path
        return path

    @staticmethod
    def telemetry_path_for_run(run_path: str | Path) -> Path:
        path = Path(run_path)
        if path.name == "run.json":
            return path.parent / "telemetry.json"
        return path.with_name(f"{path.stem}-telemetry.json")

    @staticmethod
    def gate_map_path_for_run(run_path: str | Path) -> Path:
        path = Path(run_path)
        if path.name == "run.json":
            return path.parent / "gate_map.json"
        return path.with_name(f"{path.stem}-gate_map.json")

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
            for key in (
                "cycle",
                "inner_cycle",
                "outer_cycle",
                "sim_time_ns",
                "wall_elapsed_ms",
                "deadline_lateness_ms",
                "loop_elapsed_ms",
            )
            if key in cycle
        }
        self._append_telemetry_sample(telemetry, context)

    def _append_telemetry_sample(self, telemetry: Any, context: dict[str, Any]) -> None:
        sample = self._with_record_fields(dict(context), record_type="telemetry")
        sample["telemetry"] = telemetry
        telemetry_key = self._telemetry_key(sample)
        if telemetry_key in self._telemetry_keys:
            return
        self._telemetry_keys.add(telemetry_key)
        self.telemetry_records["samples"].append(sample)
        self._append_jsonl("telemetry.jsonl", sample)

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if hasattr(value, "__dict__"):
            return value.__dict__
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    @classmethod
    def _consume_pending_run_dir(cls) -> Path | None:
        path = cls._pending_run_dir
        cls._pending_run_dir = None
        return path

    def _with_record_fields(self, record: dict[str, Any], *, record_type: str) -> dict[str, Any]:
        self._sequence += 1
        enriched = dict(record)
        enriched.setdefault("schema_version", 1)
        enriched.setdefault("record_type", record_type)
        enriched.setdefault("level", "info")
        enriched.setdefault("component", "flight")
        enriched.setdefault("sequence", self._sequence)
        enriched.setdefault("wall_time_utc", datetime.now(timezone.utc).isoformat())
        enriched.setdefault("monotonic_s", time.monotonic())
        return enriched

    def _append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        path = self._lists_dir() / filename
        encoded = json.dumps(record, separators=(",", ":"), default=self._json_default)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            if os.environ.get("FLIGHT_LOG_FSYNC") == "1":
                os.fsync(handle.fileno())

    def _write_jsonl_sidecars(self) -> None:
        if self.run_dir is None:
            return
        streams = {
            "events.jsonl": self.records["events"],
            "cycles.jsonl": self.records["cycles"],
            "telemetry.jsonl": self.telemetry_records["samples"],
            "gate_map.jsonl": self.gate_map_records["cycles"],
            "vision_frames.jsonl": self.records["vision_frames"],
            "vision_observations.jsonl": self.records["vision_observations"],
            "planned_paths.jsonl": self.records["planned_paths"],
            "test_paths.jsonl": self.records["test_paths"],
        }
        for filename, records in streams.items():
            path = self._lists_dir() / filename
            lines = [
                json.dumps(record, separators=(",", ":"), default=self._json_default)
                for record in records
            ]
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _lists_dir(self) -> Path:
        assert self.run_dir is not None
        return self.run_dir / "lists"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=self._json_default), encoding="utf-8")
        tmp_path.replace(path)

    def _write_status(self, *, reason: str, force: bool = False) -> None:
        if self.run_dir is None:
            return
        now_s = time.monotonic()
        if not force and now_s - self._last_status_write_s < self._status_write_period_s:
            return
        self._last_status_write_s = now_s
        self._write_json(
            self.run_dir / "status.json",
            {
                "schema_version": 1,
                "reason": reason,
                "updated_wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "events": len(self.records["events"]),
                "cycles": len(self.records["cycles"]),
                "telemetry_samples": len(self.telemetry_records["samples"]),
                "gate_map_cycles": len(self.gate_map_records["cycles"]),
                "last_event": self._compact_record(self._last_event),
                "last_cycle": self._compact_record(self._last_cycle),
                "last_command": self._last_command,
                "max_deadline_lateness_ms": self._max_deadline_lateness_ms,
                "max_loop_elapsed_ms": self._max_loop_elapsed_ms,
                "finalized": self._finalized,
            },
        )

    def _compact_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        keys = (
            "event",
            "record_type",
            "level",
            "sequence",
            "wall_time_utc",
            "inner_cycle",
            "outer_cycle",
            "cycle",
            "sim_time_ns",
            "system_mode",
            "modes",
            "reason",
            "error_type",
            "error",
        )
        return {key: record[key] for key in keys if key in record}

    def _telemetry_key(self, sample: dict[str, Any]) -> tuple[Any, Any]:
        telemetry = sample.get("telemetry")
        sim_time_ns = sample.get("sim_time_ns")
        if sim_time_ns is None:
            sim_time_ns = getattr(telemetry, "sim_time_ns", None)
        cycle = sample.get("inner_cycle", sample.get("cycle"))
        return sim_time_ns, cycle

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _configure_python_logging(run_dir: Path) -> python_logging.Handler | None:
        log_path = run_dir / "console.log"
        root_logger = python_logging.getLogger()
        for handler in root_logger.handlers:
            if getattr(handler, "_flight_run_log_path", None) == str(log_path):
                return None
        handler = python_logging.FileHandler(log_path, encoding="utf-8", delay=True)
        handler._flight_run_log_path = str(log_path)  # type: ignore[attr-defined]
        handler.setFormatter(
            python_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(handler)
        root_logger.setLevel(min(root_logger.level or python_logging.INFO, python_logging.INFO))
        return handler
