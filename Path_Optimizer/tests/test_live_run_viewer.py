import json

from core.app.live_data import discover_live_run_dirs, load_live_run, nearest_cycle_for_frame


def _write_run(tmp_path):
    run_dir = tmp_path / "data" / "live_runs" / "run-20260601T120000.000000Z"
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {"scenario": "test"},
                "events": [
                    {
                        "event": "connected",
                        "bridge": {
                            "latest_timesync": {"response_time_ns": 1_000_000_000_000},
                            "latest_telemetry": {"sim_time_ns": 10_000_000_000},
                        },
                    }
                ],
                "cycles": [
                    {"cycle": 0, "telemetry": {"sim_time_ns": 10_000_000_000}},
                    {"cycle": 1, "telemetry": {"sim_time_ns": 10_040_000_000}},
                ],
            }
        ),
        encoding="utf-8",
    )
    frame = {"frame_id": 7, "sim_time_ns": 1_000_039_000_000, "jpeg_size": 3, "path": "frame.jpg"}
    (frames_dir / "frames.jsonl").write_text(f"{json.dumps(frame)}\n{json.dumps(frame)}\n", encoding="utf-8")
    (frames_dir / "frame.jpg").write_bytes(b"jpg")
    return run_dir


def test_live_run_discovery_and_manifest_deduplication(tmp_path) -> None:
    run_dir = _write_run(tmp_path)

    assert discover_live_run_dirs(str(tmp_path)) == [str(run_dir.resolve())]
    run = load_live_run(str(run_dir))

    assert len(run.frames) == 1
    assert run.frames[0].frame_id == 7


def test_live_frame_sync_uses_timesync_offset(tmp_path) -> None:
    run = load_live_run(str(_write_run(tmp_path)))

    sync = nearest_cycle_for_frame(run, run.frames[0])

    assert sync.cycle_index == 1
    assert sync.cycle == run.cycles[1]
    assert sync.error_ms == 1.0
