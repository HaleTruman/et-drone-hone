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


def test_timestamped_paths_use_whole_seconds_and_do_not_collide(tmp_path) -> None:
    first = Logger.timestamped_dir(tmp_path)
    first.mkdir()
    second = Logger.timestamped_dir(tmp_path)

    assert re.fullmatch(r"run-\d{8}T\d{6}Z", first.name)
    assert second != first


def test_offline_run_logs_consistent_event_snapshots(monkeypatch, tmp_path) -> None:
    from drone import run

    monkeypatch.setattr(Logger, "timestamped_dir", lambda _logs_dir: tmp_path / "run-20260607T120000Z")

    output_path = run(duration_s=1.0, loop_hz=10.0, idle_s=1.0)
    events = json.loads(output_path.read_text(encoding="utf-8"))["events"]
    telemetry = json.loads((output_path.parent / "telemetry.json").read_text(encoding="utf-8"))["samples"]

    assert output_path == tmp_path / "run-20260607T120000Z" / "run.json"
    assert len(telemetry) == 11
    assert telemetry[0]["cycle"] == 0
    assert telemetry[0]["telemetry"]["sim_time_ns"] == 0
    assert [event["event"] for event in events] == ["initialized", "system_mode_changed", "shutdown"]
    assert [event["sim_time_ns"] for event in events] == [0, 1_000_000_000, 1_000_000_000]
    assert [event["system_mode"] for event in events] == ["IDLE", "ARMED", "ARMED"]
    for event in events:
        assert set(event["bridge"]) == {
            "connected",
            "heartbeat_started",
            "telemetry_subscribed",
        }
