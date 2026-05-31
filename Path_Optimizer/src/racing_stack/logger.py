import json
from pathlib import Path
from typing import Any


class Logger:
    def __init__(self):
        self.records: dict[str, list[Any]] = {
            "telemetry": [],
            "gate_map": [],
            "mpcc_solution": [],
        }

    def log_telemetry(self, telemetry: Any) -> None:
        self.records["telemetry"].append(telemetry)

    def log_gate_map(self, gate_map: Any) -> None:
        self.records["gate_map"].append(gate_map)

    def log_mpcc_solution(self, solution: Any) -> None:
        self.records["mpcc_solution"].append(solution)

    def save_run(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.records, indent=2, default=self._json_default), encoding="utf-8")

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "__dict__"):
            return value.__dict__
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
