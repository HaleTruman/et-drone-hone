import json
import re

import numpy as np

from core.modes.system_mode import SystemMode
from core.logging import Logger


def test_logger_writes_labeled_json(tmp_path) -> None:
    output_path = tmp_path / "logs" / "run.json"
    logger = Logger(metadata={"scenario": "test"})

    logger.log_event("initialized", system_mode=SystemMode.IDLE)
    logger.log_cycle(cycle=0, state_vector=np.zeros(13))
    logger.save_run(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["scenario"] == "test"
    assert payload["events"][0]["system_mode"] == "IDLE"
    assert payload["cycles"][0]["state_vector"] == [0.0] * 13


def test_logger_writes_telemetry_sidecar_json(tmp_path) -> None:
    output_path = tmp_path / "logs" / "run-20260607T120000Z" / "run.json"
    logger = Logger(metadata={"scenario": "telemetry-test"})

    logger.log_cycle(cycle=3, sim_time_ns=123, telemetry={"sim_time_ns": 123, "position_local_ned_m": [1.0, 2.0, -3.0]})
    logger.save_run(output_path)

    payload = json.loads((output_path.parent / "telemetry.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["scenario"] == "telemetry-test"
    assert payload["samples"] == [
        {
            "cycle": 3,
            "sim_time_ns": 123,
            "telemetry": {"sim_time_ns": 123, "position_local_ned_m": [1.0, 2.0, -3.0]},
        }
    ]


def test_logger_writes_gate_map_sidecar_json(tmp_path) -> None:
    output_path = tmp_path / "logs" / "run-20260607T120000Z" / "run.json"
    logger = Logger(metadata={"scenario": "gate-map-test"})

    logger.log_gate_map(
        [{"id": "gate-1", "position_local_ned_m": [1.0, 2.0, -3.0], "sequence": 0}],
        cycle=5,
        sim_time_ns=456,
    )
    logger.save_run(output_path)

    payload = json.loads((output_path.parent / "gate_map.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["scenario"] == "gate-map-test"
    assert payload["cycles"] == [
        {
            "cycle": 5,
            "sim_time_ns": 456,
            "gate_map": [{"id": "gate-1", "position_local_ned_m": [1.0, 2.0, -3.0], "sequence": 0}],
        }
    ]


def test_timestamped_paths_use_whole_seconds_and_do_not_collide(tmp_path) -> None:
    first = Logger.timestamped_dir(tmp_path)
    first.mkdir()
    second = Logger.timestamped_dir(tmp_path)

    assert re.fullmatch(r"run-\d{8}T\d{6}Z", first.name)
    assert second != first
