import json

from app.data import discover_run_files, discover_simulation_run_files, load_run


def test_simulation_runs_are_discovered_separately(tmp_path) -> None:
    flat_log = tmp_path / "logs" / "run-20260602T120000Z.json"
    sim_log = tmp_path / "logs" / "sim" / "run-20260602T120001Z" / "run.json"
    flat_log.parent.mkdir(parents=True)
    sim_log.parent.mkdir(parents=True)
    payload = {"schema_version": 1, "metadata": {}, "events": [], "cycles": []}
    flat_log.write_text(json.dumps(payload), encoding="utf-8")
    sim_log.write_text(json.dumps(payload), encoding="utf-8")

    assert discover_run_files(str(tmp_path)) == [str(flat_log.resolve())]
    assert discover_simulation_run_files(str(tmp_path)) == [str(sim_log.resolve())]
    assert load_run(str(sim_log)).name == "run-20260602T120001Z"
