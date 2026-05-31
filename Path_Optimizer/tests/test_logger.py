import json

import numpy as np

from racing_stack.flight_state import FlightMode
from racing_stack.logger import Logger


def test_logger_writes_labeled_json(tmp_path) -> None:
    output_path = tmp_path / "logs" / "run.json"
    logger = Logger(metadata={"scenario": "test"})

    logger.log_event("initialized", flight_mode=FlightMode.IDLE)
    logger.log_cycle(cycle=0, state_vector=np.zeros(13))
    logger.save_run(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["scenario"] == "test"
    assert payload["events"][0]["flight_mode"] == "IDLE"
    assert payload["cycles"][0]["state_vector"] == [0.0] * 13


def test_offline_run_logs_consistent_event_snapshots(monkeypatch, tmp_path) -> None:
    from drone import run

    monkeypatch.setattr(Logger, "timestamped_path", lambda _logs_dir: tmp_path / "run.json")

    output_path = run(duration_s=1.0, loop_hz=10.0, idle_s=1.0)
    events = json.loads(output_path.read_text(encoding="utf-8"))["events"]

    assert [event["event"] for event in events] == ["initialized", "flight_mode_changed", "shutdown"]
    assert [event["sim_time_ns"] for event in events] == [0, 1_000_000_000, 1_000_000_000]
    assert [event["flight_mode"] for event in events] == ["IDLE", "ARMED", "ARMED"]
    for event in events:
        assert set(event["bridge"]) == {
            "connected",
            "heartbeat_started",
            "telemetry_subscribed",
        }
