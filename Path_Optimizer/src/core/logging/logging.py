import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Logger:
    def __init__(self, metadata: dict[str, Any] | None = None):
        self.records: dict[str, Any] = {
            "schema_version": 1,
            "metadata": metadata or {},
            "events": [],
            "cycles": [],
        }

    def log_event(self, event: str, **data: Any) -> None:
        self.records["events"].append({"event": event, **data})

    def log_cycle(self, **data: Any) -> None:
        self.records["cycles"].append(data)

    def log_telemetry(self, telemetry: Any) -> None:
        self.log_event("telemetry", telemetry=telemetry)

    def log_gate_map(self, gate_map: Any) -> None:
        self.log_event("gate_map", gate_map=gate_map)

    def log_mpcc_solution(self, solution: Any) -> None:
        self.log_event("mpcc_solution", solution=solution)

    def save_run(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.records, indent=2, default=self._json_default), encoding="utf-8")

    @staticmethod
    def timestamped_path(logs_dir: str | Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return Path(logs_dir) / f"run-{timestamp}.json"

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if hasattr(value, "__dict__"):
            return value.__dict__
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
